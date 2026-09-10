#ifndef __SD_FASHN_PREPROCESS_H__
#define __SD_FASHN_PREPROCESS_H__

#include <onnxruntime_cxx_api.h>
#include <array>
#include <filesystem>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace fashn_prepare {

    struct Keypoint {
        double x, y, score;
    };
    using Pose = std::array<Keypoint, 134>;

    struct CPUOptions {
        int intra_threads = 8;
        bool arena = true;
        bool spinning = true;
    };

    class Model {
        Ort::Env env_;
        Ort::Session session_{nullptr};
        std::string input_;
        std::vector<std::string> outputs_;

    public:
        explicit Model(const std::filesystem::path& path, const CPUOptions& cpu = {});
        std::vector<Ort::Value> run(std::vector<float>& data, int width, int height);
    };

    cv::Mat pre_resize(const cv::Mat& image);
    cv::Mat resize_pad(const cv::Mat& image, cv::Rect* crop = nullptr, bool pose = false);
    Pose detect_pose(Model& detector, Model& pose_model, const cv::Mat& bgr);
    cv::Mat draw_pose(const Pose& pose, cv::Size size);
    cv::Mat parse_person(Model& parser, const cv::Mat& bgr);
    cv::Mat mask_image(const cv::Mat& image, const cv::Mat& labels, const std::string& category, bool garment);
    std::string sha256(const std::filesystem::path& path);

}  // namespace fashn_prepare
#endif
