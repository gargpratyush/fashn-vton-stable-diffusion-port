#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
// bcrypt.h requires the Windows type declarations first.
// clang-format off
#include <windows.h>
#include <bcrypt.h>
// clang-format on
#else
#include <openssl/evp.h>
#endif

namespace fashn_prepare {

    Model::Model(const std::filesystem::path& path, const CPUOptions& cpu)
        : env_(ORT_LOGGING_LEVEL_WARNING, "fashn-prepare") {
        if (cpu.intra_threads < 1 || cpu.intra_threads > 64)
            throw std::runtime_error("ORT intra-op threads must be in [1,64]");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(cpu.intra_threads);
        options.SetInterOpNumThreads(1);
        if (!cpu.arena)
            options.DisableCpuMemArena();
        if (!cpu.spinning)
            options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        session_ = Ort::Session(env_, path.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        if (session_.GetInputCount() != 1)
            throw std::runtime_error("Expected one ONNX input");
        input_ = session_.GetInputNameAllocated(0, allocator).get();
        for (size_t i = 0; i < session_.GetOutputCount(); ++i)
            outputs_.emplace_back(session_.GetOutputNameAllocated(i, allocator).get());
    }

    std::vector<Ort::Value> Model::run(std::vector<float>& data, int width, int height) {
        std::array<int64_t, 4> shape{1, 3, height, width};
        if (data.size() != size_t(width) * height * 3)
            throw std::runtime_error("Invalid ONNX input size");
        auto memory      = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input       = Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), shape.data(), shape.size());
        const char* name = input_.c_str();
        std::vector<const char*> names;
        for (const auto& output : outputs_)
            names.push_back(output.c_str());
        auto result = session_.Run(Ort::RunOptions{}, &name, &input, 1, names.data(), names.size());
        for (const auto& tensor : result) {
            if (tensor.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                throw std::runtime_error("Expected F32 ONNX outputs");
            const auto* values = tensor.GetTensorData<float>();
            for (size_t i = 0; i < tensor.GetTensorTypeAndShapeInfo().GetElementCount(); ++i)
                if (!std::isfinite(values[i]))
                    throw std::runtime_error("Non-finite ONNX output");
        }
        return result;
    }

    struct Filter {
        int start;
        std::vector<int> weights;
    };

    // Pillow-compatible Lanczos3: antialias downsampling and quantize coefficients
    // to 22 bits; clip each separable pass to uint8, not just the final image.
    static std::vector<Filter> lanczos_filter(int source, int target) {
        std::vector<Filter> filters;
        const double scale = double(source) / target, filter_scale = std::max(1.0, scale);
        const double support = 3 * filter_scale, pi = std::acos(-1.0);
        for (int out = 0; out < target; ++out) {
            double center = (out + .5) * scale;
            int start     = std::max(0, int(center - support + .5));
            int end       = std::min(source, int(center + support + .5));
            std::vector<double> weights;
            double sum = 0;
            for (int in = start; in < end; ++in) {
                double x      = (in - center + .5) / filter_scale;
                double weight = x == 0 ? 1 : (std::abs(x) < 3 ? std::sin(pi * x) * std::sin(pi * x / 3) / (pi * pi * x * x / 3) : 0);
                weights.push_back(weight);
                sum += weight;
            }
            Filter filter{start, {}};
            for (double value : weights) {
                double scaled = value / sum * (1 << 22);
                filter.weights.push_back(int(scaled + (scaled < 0 ? -.5 : .5)));
            }
            filters.push_back(std::move(filter));
        }
        return filters;
    }

    cv::Mat pre_resize(const cv::Mat& image) {
        const double scale = std::min(1.0, 864.0 / std::max(image.cols, image.rows));
        if (scale == 1)
            return image.clone();
        int width = int(image.cols * scale), height = int(image.rows * scale);
        if (width < 1 || height < 1)
            throw std::runtime_error("Image aspect ratio is too extreme");
        const auto horizontal = lanczos_filter(image.cols, width), vertical = lanczos_filter(image.rows, height);
        cv::Mat temp(image.rows, width, image.type()), result(height, width, image.type());
        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 3; ++c) {
                    int sum            = 1 << 21;
                    const auto& filter = horizontal[x];
                    for (size_t i = 0; i < filter.weights.size(); ++i)
                        sum += image.at<cv::Vec3b>(y, filter.start + int(i))[c] * filter.weights[i];
                    temp.at<cv::Vec3b>(y, x)[c] = uint8_t(std::clamp(sum >> 22, 0, 255));
                }
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 3; ++c) {
                    int sum            = 1 << 21;
                    const auto& filter = vertical[y];
                    for (size_t i = 0; i < filter.weights.size(); ++i)
                        sum += temp.at<cv::Vec3b>(filter.start + int(i), x)[c] * filter.weights[i];
                    result.at<cv::Vec3b>(y, x)[c] = uint8_t(std::clamp(sum >> 22, 0, 255));
                }
        return result;
    }

    cv::Mat resize_pad(const cv::Mat& image, cv::Rect* crop, bool pose) {
        double scale = std::min(576.0 / image.cols, 864.0 / image.rows);
        int width = int(image.cols * scale), height = int(image.rows * scale);
        if (width < 1 || height < 1)
            throw std::runtime_error("Image aspect ratio is too extreme");
        cv::Mat resized, padded;
        cv::resize(image, resized, {width, height}, 0, 0,
                   pose ? cv::INTER_NEAREST_EXACT : (scale > 1 ? cv::INTER_LANCZOS4 : cv::INTER_AREA));
        int left = (576 - width) / 2, top = (864 - height) / 2;
        cv::copyMakeBorder(resized, padded, top, 864 - height - top, left, 576 - width - left, cv::BORDER_CONSTANT, 0);
        if (crop)
            *crop = {left, top, width, height};
        return padded;
    }

    static std::vector<float> chw(const cv::Mat& image, const std::array<double, 3>& mean, const std::array<double, 3>& stddev, bool rgb = false) {
        std::vector<float> result(image.total() * 3);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < image.rows; ++y)
                for (int x = 0; x < image.cols; ++x)
                    result[c * image.total() + y * image.cols + x] = float((image.at<cv::Vec3b>(y, x)[rgb ? 2 - c : c] - mean[c]) / stddev[c]);
        return result;
    }

    struct Box {
        float x0, y0, x1, y1, score;
    };

    static std::vector<Box> detect_boxes(Model& detector, const cv::Mat& image) {
        const double ratio = std::min(640.0 / image.cols, 640.0 / image.rows);
        cv::Mat resized, padded(640, 640, CV_8UC3, cv::Scalar::all(114));
        cv::resize(image, resized, {int(image.cols * ratio), int(image.rows * ratio)}, 0, 0, cv::INTER_LINEAR);
        resized.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));
        auto input   = chw(padded, {0, 0, 0}, {1, 1, 1});
        auto outputs = detector.run(input, 640, 640);
        if (outputs.size() != 1 || outputs[0].GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>({1, 8400, 85}))
            throw std::runtime_error("Unexpected YOLOX output shape");
        const float* data = outputs[0].GetTensorData<float>();
        std::vector<Box> boxes;
        int row = 0;
        for (int stride : {8, 16, 32})
            for (int y = 0; y < 640 / stride; ++y)
                for (int x = 0; x < 640 / stride; ++x, ++row) {
                    const float* p    = data + row * 85;
                    const float score = p[4] * p[5];
                    if (score <= .1f)
                        continue;
                    float cx = float((double(p[0]) + x) * stride), cy = float((double(p[1]) + y) * stride);
                    float width = std::exp(p[2]) * stride, height = std::exp(p[3]) * stride;
                    boxes.push_back({(cx - width / 2) / float(ratio), (cy - height / 2) / float(ratio),
                                     (cx + width / 2) / float(ratio), (cy + height / 2) / float(ratio), score});
                }
        std::stable_sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
        std::vector<Box> kept;
        for (const auto& box : boxes) {
            bool suppressed = false;
            for (const auto& previous : kept) {
                float intersection = std::max(0.f, std::min(box.x1, previous.x1) - std::max(box.x0, previous.x0) + 1) *
                                     std::max(0.f, std::min(box.y1, previous.y1) - std::max(box.y0, previous.y0) + 1);
                float area = (box.x1 - box.x0 + 1) * (box.y1 - box.y0 + 1) + (previous.x1 - previous.x0 + 1) * (previous.y1 - previous.y0 + 1);
                if (intersection / (area - intersection) > .45f) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed)
                kept.push_back(box);
        }
        kept.erase(std::remove_if(kept.begin(), kept.end(), [](const Box& box) { return box.score <= .3f; }), kept.end());
        if (kept.empty())
            kept.push_back({0, 0, float(image.cols), float(image.rows), 0});
        return kept;
    }

    Pose detect_pose(Model& detector, Model& pose_model, const cv::Mat& image) {
        std::vector<Pose> candidates;
        for (const auto& box : detect_boxes(detector, image)) {
            double cx = (double(box.x0) + box.x1) / 2, cy = (double(box.y0) + box.y1) / 2;
            double sw = (double(box.x1) - box.x0) * 1.25, sh = (double(box.y1) - box.y0) * 1.25;
            if (sw > sh * (288.0 / 384))
                sh = sw / (288.0 / 384);
            else
                sw = sh * (288.0 / 384);
            cv::Point2f source[] = {{float(cx), float(cy)}, {float(cx), float(cy - sw / 2)}, {0, 0}};
            source[2]            = source[1] + cv::Point2f(-(source[0].y - source[1].y), source[0].x - source[1].x);
            cv::Point2f target[] = {{144, 192}, {144, 48}, {0, 48}};
            cv::Mat warped;
            cv::warpAffine(image, warped, cv::getAffineTransform(source, target), {288, 384}, cv::INTER_LINEAR);
            auto input   = chw(warped, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
            auto outputs = pose_model.run(input, 288, 384);
            if (outputs.size() != 2 || outputs[0].GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>({1, 133, 576}) ||
                outputs[1].GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>({1, 133, 768}))
                throw std::runtime_error("Unexpected DWPose output shapes");
            Pose points{};
            for (int i = 0; i < 133; ++i) {
                const float* px = outputs[0].GetTensorData<float>() + i * 576;
                const float* py = outputs[1].GetTensorData<float>() + i * 768;
                int ix = int(std::max_element(px, px + 576) - px), iy = int(std::max_element(py, py + 768) - py);
                double score = std::min(px[ix], py[iy]);
                if (score <= 0)
                    ix = iy = -1;
                points[i < 17 ? i : i + 1] = {ix / 2.0 / 288 * sw + cx - sw / 2, iy / 2.0 / 384 * sh + cy - sh / 2, score};
            }
            points[17]       = {(points[5].x + points[6].x) / 2, (points[5].y + points[6].y) / 2,
                          points[5].score > .3 && points[6].score > .3 ? 1.0 : 0.0};
            const int from[] = {17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3};
            const int to[]   = {1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17};
            auto original    = points;
            for (int i = 0; i < 15; ++i)
                points[to[i]] = original[from[i]];
            candidates.push_back(points);
        }
        std::vector<double> scores, areas;
        for (const auto& points : candidates) {
            double score = 0, minx = INFINITY, miny = INFINITY, maxx = -INFINITY, maxy = -INFINITY;
            for (int i = 1; i < 14; ++i) {
                if (points[i].score > .3)
                    score += points[i].score;
                // Preserve the pinned reference's first-candidate validity mask.
                if (candidates[0][i].score > .3) {
                    if (points[i].x > 0) {
                        minx = std::min(minx, points[i].x);
                        maxx = std::max(maxx, points[i].x);
                    }
                    if (points[i].y > 0) {
                        miny = std::min(miny, points[i].y);
                        maxy = std::max(maxy, points[i].y);
                    }
                }
            }
            scores.push_back(score);
            double area = (maxx - minx) * (maxy - miny);
            areas.push_back(std::isfinite(area) ? area * score : 0);
        }
        const auto& selection = std::all_of(areas.begin(), areas.end(), [](double v) { return v == 0; }) ? scores : areas;
        Pose result           = candidates[std::max_element(selection.begin(), selection.end()) - selection.begin()];
        if (std::count_if(result.begin(), result.begin() + 18, [](const Keypoint& p) { return p.score > .3; }) < 3)
            throw std::runtime_error("DWPose found fewer than three visible body keypoints");
        for (auto& point : result) {
            point.x /= image.cols;
            point.y /= image.rows;
        }
        return result;
    }

    cv::Mat draw_pose(const Pose& pose, cv::Size size) {
        cv::Mat canvas(size, CV_8UC1, cv::Scalar(0)), limbs = canvas.clone();
        const int edges[][2] = {{1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}};
        auto point           = [&](int i) { return cv::Point(int(pose[i].x * size.width), int(pose[i].y * size.height)); };
        for (int i = 0; i < 17; ++i) {
            int a = edges[i][0], b = edges[i][1];
            if (pose[a].score <= .3 || pose[b].score <= .3)
                continue;
            auto p = point(a), q = point(b);
            cv::Point center(int(std::floor((p.x + q.x) / 2.0)), int(std::floor((p.y + q.y) / 2.0)));
            int length = int(std::hypot(p.x - q.x, p.y - q.y)), angle = int(std::atan2(p.y - q.y, p.x - q.x) * 180 / std::acos(-1.0));
            std::vector<cv::Point> polygon;
            cv::ellipse2Poly(center, {length / 2, 4}, angle, 0, 360, 1, polygon);
            cv::fillConvexPoly(limbs, polygon, int(20 + i * (220.0 / 17)));
        }
        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width; ++x)
                canvas.at<uint8_t>(y, x) = uint8_t(limbs.at<uint8_t>(y, x) * .6);
        for (int i = 0; i < 18; ++i)
            if (pose[i].score > .3) {
                for (int e = 0; e < 17; ++e)
                    if (edges[e][0] == i || edges[e][1] == i) {
                        cv::circle(canvas, point(i), 4, std::min(255, int(int(20 + e * (220.0 / 17)) * 1.3)), -1);
                        break;
                    }
            }
        for (int start : {92, 113}) {
            int edge = 0;
            for (int finger = 0; finger < 5; ++finger)
                for (int segment = 0; segment < 4; ++segment, ++edge) {
                    int a = start + (segment == 0 ? 0 : 1 + finger * 4 + segment - 1), b = start + 1 + finger * 4 + segment;
                    auto p = point(a), q = point(b);
                    if (pose[a].score >= .3 && pose[b].score >= .3 && p.x > 0 && p.y > 0 && q.x > 0 && q.y > 0)
                        cv::line(canvas, p, q, int(160 + edge * (60.0 / 19)), 2);
                }
            for (int i = start; i < start + 21; ++i) {
                auto p = point(i);
                if (pose[i].score >= .3 && p.x > 0 && p.y > 0)
                    cv::circle(canvas, p, 4, 240, -1);
            }
        }
        for (int i = 24; i < 92; ++i) {
            auto p = point(i);
            if (pose[i].score >= .3 && p.x > 0 && p.y > 0)
                cv::circle(canvas, p, 3, 200, -1);
        }
        return canvas;
    }

    cv::Mat parse_person(Model& parser, const cv::Mat& image) {
        cv::Mat resized;
        cv::resize(image, resized, {384, 576}, 0, 0, cv::INTER_AREA);
        std::vector<float> input(resized.total() * 3);
        const float mean[] = {.485f, .456f, .406f}, stddev[] = {.229f, .224f, .225f};
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < 576; ++y)
                for (int x = 0; x < 384; ++x) {
                    float value                              = resized.at<cv::Vec3b>(y, x)[2 - c] / 255.f;
                    input[c * resized.total() + y * 384 + x] = (value - mean[c]) / stddev[c];
                }
        auto outputs = parser.run(input, 384, 576);
        if (outputs.size() != 1 || outputs[0].GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>({1, 18, 144, 96}))
            throw std::runtime_error("Unexpected human parser output shape");
        cv::Mat labels(image.size(), CV_8UC1, cv::Scalar(0)), best(image.size(), CV_32F, cv::Scalar(-INFINITY));
        for (int c = 0; c < 18; ++c) {
            const float* logits = outputs[0].GetTensorData<float>() + c * 144 * 96;
            // PyTorch align_corners=False uses F32 source coordinates. OpenCV's
            // higher-precision coordinates can flip close argmax boundaries.
            for (int y = 0; y < image.rows; ++y)
                for (int x = 0; x < image.cols; ++x) {
                    float sy = std::max(0.f, (144.f / image.rows) * (y + .5f) - .5f);
                    float sx = std::max(0.f, (96.f / image.cols) * (x + .5f) - .5f);
                    int y0 = int(sy), x0 = int(sx), y1 = std::min(143, y0 + 1), x1 = std::min(95, x0 + 1);
                    float wy = sy - y0, wx = sx - x0;
                    float value = (1 - wy) * ((1 - wx) * logits[y0 * 96 + x0] + wx * logits[y0 * 96 + x1]) +
                                  wy * ((1 - wx) * logits[y1 * 96 + x0] + wx * logits[y1 * 96 + x1]);
                    if (value > best.at<float>(y, x)) {
                        best.at<float>(y, x)     = value;
                        labels.at<uint8_t>(y, x) = uint8_t(c);
                    }
                }
        }
        return labels;
    }

    cv::Mat mask_image(const cv::Mat& image, const cv::Mat& labels, const std::string& category, bool garment) {
        if (labels.type() != CV_8UC1 || labels.size() != image.size())
            throw std::runtime_error("Invalid parser label map");
        const bool upper = category == "tops" || category == "one-pieces", lower = category == "bottoms" || category == "one-pieces";
        if (!upper && !lower)
            throw std::runtime_error("Invalid category");
        std::array<bool, 18> selected{}, identity{};
        if (upper)
            for (int id : {3, 4, 10})
                selected[id] = true;
        if (lower)
            for (int id : {5, 6, 7})
                selected[id] = true;
        if (!garment) {
            if (upper)
                selected[12] = selected[16] = true;
            if (lower)
                selected[14] = true;
        }
        cv::Mat mask(image.size(), CV_8UC1), excluded = mask.clone(), result = image.clone();
        for (int id : {1, 2, 17, 8, 11, 9})
            identity[id] = true;
        if (category == "tops")
            identity[14] = true;
        if (category == "bottoms")
            identity[12] = true;
        if (upper)
            identity[13] = true;
        if (lower)
            identity[15] = true;
        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x) {
                int id = labels.at<uint8_t>(y, x);
                if (id >= 18)
                    throw std::runtime_error("Invalid parser class ID");
                mask.at<uint8_t>(y, x)     = selected[id] ? 255 : 0;
                excluded.at<uint8_t>(y, x) = identity[id] ? 255 : 0;
            }
        if (garment) {
            if (cv::countNonZero(mask) == 0)
                throw std::runtime_error("No garment pixels for selected category");
            result.setTo(cv::Scalar::all(127), ~mask);
            return result;
        }
        double scale = image.rows / 864.0;
        cv::Mat buffer, bounded(image.size(), CV_8UC1, cv::Scalar(0)), dilated, outside, inside, signed_distance, blurred;
        int k = std::max(1, int(4 * scale)), radius = std::max(1, int(18 * scale));
        cv::dilate(mask, buffer, cv::Mat::ones(k, k, CV_8UC1));
        bounded(cv::boundingRect(mask)).setTo(255);
        cv::dilate(mask, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * radius + 1, 2 * radius + 1}));
        cv::distanceTransform(~dilated, outside, cv::DIST_L2, 5);
        cv::distanceTransform(dilated, inside, cv::DIST_L2, 5);
        signed_distance = outside - inside;
        cv::GaussianBlur(signed_distance, blurred, {0, 0}, radius / 2.5, 0, cv::BORDER_REPLICATE);
        cv::Mat contour = blurred <= 0, flood = ~contour;
        cv::floodFill(flood, {0, 0}, 0);
        contour |= flood;
        contour |= mask;
        cv::distanceTransform(~contour, outside, cv::DIST_L2, 5);
        bounded.setTo(0, (bounded & ~contour) & (outside > 100 * scale));
        cv::dilate(bounded, dilated, cv::Mat::ones(2 * int(16 * scale) + 1, 2 * int(33 * scale) + 1, CV_8UC1),
                   {int(33 * scale), int(16 * scale)});
        result.setTo(cv::Scalar::all(127), buffer | (dilated & ~excluded));
        return result;
    }

    std::string sha256(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot read for SHA256: " + path.string());
        std::array<unsigned char, 32> digest{};
        std::array<char, 65536> buffer{};
#ifdef _WIN32
        struct Handles {
            BCRYPT_ALG_HANDLE algorithm = nullptr;
            BCRYPT_HASH_HANDLE hash     = nullptr;
            ~Handles() {
                if (hash)
                    BCryptDestroyHash(hash);
                if (algorithm)
                    BCryptCloseAlgorithmProvider(algorithm, 0);
            }
        } handles;
        if (BCryptOpenAlgorithmProvider(&handles.algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
            BCryptCreateHash(handles.algorithm, &handles.hash, nullptr, 0, nullptr, 0, 0) < 0)
            throw std::runtime_error("SHA256 initialization failed");
        while (file.read(buffer.data(), buffer.size()) || file.gcount()) {
            if (BCryptHashData(handles.hash, reinterpret_cast<PUCHAR>(buffer.data()), ULONG(file.gcount()), 0) < 0)
                throw std::runtime_error("SHA256 update failed");
        }
        if (BCryptFinishHash(handles.hash, digest.data(), ULONG(digest.size()), 0) < 0)
            throw std::runtime_error("SHA256 finish failed");
#else
        std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
        if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
            throw std::runtime_error("SHA256 initialization failed");
        while (file.read(buffer.data(), buffer.size()) || file.gcount())
            if (EVP_DigestUpdate(context.get(), buffer.data(), size_t(file.gcount())) != 1)
                throw std::runtime_error("SHA256 update failed");
        if (EVP_DigestFinal_ex(context.get(), digest.data(), nullptr) != 1)
            throw std::runtime_error("SHA256 finish failed");
#endif
        if (file.bad())
            throw std::runtime_error("SHA256 file read failed");
        std::ostringstream result;
        for (auto byte : digest)
            result << std::hex << std::setw(2) << std::setfill('0') << int(byte);
        return result.str();
    }
}  // namespace fashn_prepare
