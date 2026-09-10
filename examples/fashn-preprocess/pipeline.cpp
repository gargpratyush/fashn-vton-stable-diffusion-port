#include "pipeline.h"
#include <fstream>
#include "json.hpp"

namespace fashn_prepare {

    const std::map<std::string, std::string>& pose_weight_hashes() {
        static const std::map<std::string, std::string> hashes = {
            {"yolox_l.onnx", "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411"},
            {"dw-ll_ucoco_384.onnx", "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843"}};
        return hashes;
    }

    static std::filesystem::path validate_pose_directory(const std::filesystem::path& directory,
                                                         const std::filesystem::path& parser_directory,
                                                         bool consent) {
        if (!parser_directory.empty() && !consent)
            throw std::runtime_error("Parser requires explicit non-commercial research/evaluation consent");
        for (const auto& [name, hash] : pose_weight_hashes())
            if (sha256(directory / name) != hash)
                throw std::runtime_error("Unexpected DWPose checkpoint SHA256: " + name);
        return directory;
    }

    Preparer::Preparer(const std::filesystem::path& pose_dir, const std::filesystem::path& parser_dir, bool consent, bool lazy_parser, const CPUOptions& cpu)
        : cpu_(cpu),
          detector_(validate_pose_directory(pose_dir, parser_dir, consent) / "yolox_l.onnx", cpu),
          pose_(pose_dir / "dw-ll_ucoco_384.onnx", cpu) {
        if (!parser_dir.empty()) {
            std::ifstream input(parser_dir / "manifest.json");
            if (!input)
                throw std::runtime_error("Cannot open evaluation parser manifest");
            auto manifest = nlohmann::ordered_json::parse(input);
            if (manifest.value("schema", "") != "fashn-parser-onnx-eval-v1" ||
                manifest.value("checkpoint_sha256", "") != "e43c8c8a9b04f28798f0a4630cf18caa2cdb27a0d454fae43a5716e6f7078244" ||
                manifest.value("onnx_sha256", "") != sha256(parser_dir / "parser.onnx") ||
                manifest.value("license_sha256", "") != sha256(parser_dir / "LICENSE"))
                throw std::runtime_error("Invalid evaluation parser manifest or artifact hash");
            parser_directory_ = std::filesystem::absolute(parser_dir);
            parser_manifest_ = manifest.dump();
            if (!lazy_parser)
                parser_ = std::make_unique<Model>(parser_directory_ / "parser.onnx", cpu_);
        }
    }

    Model& Preparer::parser() {
        if (!has_parser())
            throw std::runtime_error("Research/evaluation parser is not configured");
        if (!parser_) {
            const auto manifest = nlohmann::ordered_json::parse(parser_manifest_);
            // Startup validation alone is insufficient after a potentially long lazy-load delay.
            if (sha256(parser_directory_ / "parser.onnx") != manifest.at("onnx_sha256").get<std::string>() ||
                sha256(parser_directory_ / "LICENSE") != manifest.at("license_sha256").get<std::string>())
                throw std::runtime_error("Evaluation parser artifacts changed after startup validation");
            parser_ = std::make_unique<Model>(parser_directory_ / "parser.onnx", cpu_);
        }
        return *parser_;
    }

    PreparedImages Preparer::prepare(const cv::Mat& person_bgr, const cv::Mat& garment_bgr, const std::string& category, const std::string& photo_type, bool segmentation_free, const std::function<bool()>& cancelled) {
        auto check_cancelled = [&] {
            if (cancelled && cancelled())
                throw PreparationCancelled();
        };
        check_cancelled();
        if (category != "tops" && category != "bottoms" && category != "one-pieces")
            throw std::runtime_error("Invalid category");
        if (photo_type != "flat-lay" && photo_type != "model")
            throw std::runtime_error("Invalid garment photo type");
        if ((!segmentation_free || photo_type == "model") && !has_parser())
            throw std::runtime_error("Requested mode requires a configured research/evaluation parser");
        for (const auto* image : {&person_bgr, &garment_bgr})
            if (image->empty() || image->type() != CV_8UC3 || image->total() > 20000000)
                throw std::runtime_error("Expected an 8-bit three-channel image of at most 20 megapixels");
        PreparedImages output;
        output.person_preresize  = pre_resize(person_bgr);
        output.garment_preresize = pre_resize(garment_bgr);
        cv::Mat person = output.person_preresize, garment = output.garment_preresize;
        check_cancelled();
        output.person_points = detect_pose(detector_, pose_, person);
        check_cancelled();
        auto person_pose = draw_pose(output.person_points, person.size());
        cv::Mat garment_pose(garment.size(), CV_8UC1, cv::Scalar(0));
        if (photo_type == "model") {
            output.garment_points = detect_pose(detector_, pose_, garment);
            check_cancelled();
            garment_pose = draw_pose(output.garment_points, garment.size());
        }
        if (!segmentation_free) {
            auto& model = parser();
            check_cancelled();
            output.person_segmentation = parse_person(model, person);
            check_cancelled();
            person = mask_image(person, output.person_segmentation, category, false);
        }
        if (photo_type == "model") {
            auto& model = parser();
            check_cancelled();
            output.garment_segmentation = parse_person(model, garment);
            check_cancelled();
            garment = mask_image(garment, output.garment_segmentation, category, true);
        }
        output.images = {resize_pad(person, &output.crop), resize_pad(garment),
                         resize_pad(person_pose, nullptr, true), resize_pad(garment_pose, nullptr, true)};
        check_cancelled();
        return output;
    }

}  // namespace fashn_prepare
