#include "pipeline.h"

#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <set>
#include <stdexcept>
#include "json.hpp"

using json   = nlohmann::ordered_json;
namespace fs = std::filesystem;

static void write_json(const fs::path& path, const json& value) {
    std::ofstream output(path);
    output << value.dump(2) << "\n";
    output.close();
    if (!output)
        throw std::runtime_error("Cannot write JSON: " + path.string());
}

static void write_image(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Cannot write image: " + path.string());
}

int main(int argc, char** argv) {
    try {
        const std::set<std::string> valued = {"--dwpose-dir", "--person-image", "--garment-image", "--category",
                                              "--garment-photo-type", "--output", "--parser-dir"};
        const std::set<std::string> flags  = {"--no-segmentation-free", "--accept-parser-research-license", "--ort-no-arena"};
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout << "sd-fashn-prepare --dwpose-dir DIR --person-image FILE --garment-image FILE "
                             "--category tops|bottoms|one-pieces --garment-photo-type flat-lay|model --output NEW_DIR\n"
                             "Optional: --no-segmentation-free --parser-dir DIR --accept-parser-research-license --ort-no-arena\n"
                             "Parser DIR contains an explicitly exported evaluation-only parser.onnx, manifest.json and LICENSE.\n"
                             "No weights or dependencies are downloaded. RGB decoding ignores EXIF orientation.\n";
                return 0;
            }
            if (args.count(key))
                throw std::runtime_error("Duplicate option: " + key);
            if (flags.count(key))
                args[key] = "true";
            else if (valued.count(key) && i + 1 < argc)
                args[key] = argv[++i];
            else
                throw std::runtime_error("Unknown option or missing value: " + key);
        }
        for (const auto& key : valued)
            if (key != "--parser-dir" && (!args.count(key) || args[key].empty()))
                throw std::runtime_error("Required option: " + key);
        const std::string category = args["--category"], photo = args["--garment-photo-type"];
        if (category != "tops" && category != "bottoms" && category != "one-pieces")
            throw std::runtime_error("Invalid category");
        if (photo != "flat-lay" && photo != "model")
            throw std::runtime_error("Invalid garment photo type");
        const bool mask_person = args.count("--no-segmentation-free") != 0, need_parser = mask_person || photo == "model";
        if (need_parser && (!args.count("--accept-parser-research-license") || !args.count("--parser-dir")))
            throw std::runtime_error("Parser-dependent modes require --parser-dir and --accept-parser-research-license (non-commercial research/evaluation only)");
        const fs::path output = args["--output"], pose_dir = args["--dwpose-dir"];
        if (fs::exists(output))
            throw std::runtime_error("Choose a new output directory");
        fashn_prepare::CPUOptions cpu;
        cpu.arena = args.count("--ort-no-arena") == 0;
        fashn_prepare::Preparer preparer(pose_dir, need_parser ? args["--parser-dir"] : "",
                                         args.count("--accept-parser-research-license") != 0, true, cpu);
        if (preparer.has_parser())
            std::cout << "RESEARCH/EVALUATION ONLY: restricted human parser enabled.\n";
        auto load = [](const std::string& path) {
            auto image = cv::imread(path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
            if (image.empty() || image.total() > 20000000)
                throw std::runtime_error("Cannot decode RGB image or image exceeds 20 megapixels: " + path);
            return image;
        };
        const auto prepared = preparer.prepare(load(args["--person-image"]), load(args["--garment-image"]),
                                               category, photo, !mask_person);
        json poses;
        auto pose_json = [](const fashn_prepare::Pose& pose) {
            json points = json::array();
            for (const auto& p : pose)
                points.push_back({p.x, p.y, p.score});
            return points;
        };
        poses["person"] = pose_json(prepared.person_points);
        if (photo == "model") {
            poses["garment"] = pose_json(prepared.garment_points);
        }
        fs::create_directories(output);
        write_image(output / "person-preresize.png", prepared.person_preresize);
        write_image(output / "garment-preresize.png", prepared.garment_preresize);
        write_json(output / "poses.json", poses);
        if (mask_person) {
            write_image(output / "person_segmentation.png", prepared.person_segmentation);
        }
        if (photo == "model") {
            write_image(output / "garment_segmentation.png", prepared.garment_segmentation);
        }
        const auto crop     = prepared.crop;
        const char* names[] = {"ca_image", "garment_image", "person_pose", "garment_pose"};
        for (size_t i = 0; i < prepared.images.size(); ++i)
            write_image(output / (std::string(names[i]) + ".png"), prepared.images[i]);
        json manifest = {{"schema", "fashn-vton-prepared-v1"}, {"category", category}, {"crop", {{"x", crop.x}, {"y", crop.y}, {"width", crop.width}, {"height", crop.height}}}};
        for (const char* name : {"ca_image", "garment_image", "person_pose", "garment_pose"})
            manifest[name] = std::string(name) + ".png";
        json provenance = {{"source_revision", "7c0f10af3f91ad4048fe9729c470a13ef905d25a"},
                           {"preparation", "native-opencv-ort"},
                           {"opencv", CV_VERSION},
                           {"onnxruntime", OrtGetApiBase()->GetVersionString()},
                           {"ort_cpu_arena", cpu.arena},
                           {"segmentation_free", !mask_person},
                           {"garment_photo_type", photo},
                           {"parser", json::parse(preparer.parser_manifest())},
                           {"dwpose_sha256", fashn_prepare::pose_weight_hashes()},
                           {"inputs_sha256", {{"person", fashn_prepare::sha256(args["--person-image"])}, {"garment", fashn_prepare::sha256(args["--garment-image"])}}}};
        json artifacts;
        for (const auto& entry : fs::directory_iterator(output))
            artifacts[entry.path().filename().string()] = fashn_prepare::sha256(entry.path());
        provenance["artifacts"] = artifacts;
        write_json(output / "provenance.json", provenance);
        write_json(output / "manifest.json", manifest);
        std::cout << "Prepared inputs: " << (output / "manifest.json").string() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FASHN preparation failed: " << error.what() << "\n";
        return 1;
    }
}
