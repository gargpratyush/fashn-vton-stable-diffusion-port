#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "preprocess.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: test-fashn-vton-preprocess input-dir output-dir\n";
        return 2;
    }
    try {
        std::filesystem::path input = argv[1], output = argv[2];
        auto image = cv::imread((input / "image.png").string()), labels = cv::imread((input / "labels.png").string(), cv::IMREAD_GRAYSCALE);
        if (image.empty() || labels.empty())
            throw std::runtime_error("Missing primitive fixtures");
        std::filesystem::create_directories(output);
        auto write = [&](const std::string& name, const cv::Mat& value) {
            if (!cv::imwrite((output / name).string(), value))
                throw std::runtime_error("Cannot write primitive result");
        };
        auto resized = fashn_prepare::pre_resize(image);
        write("preresize.png", resized);
        write("padded.png", fashn_prepare::resize_pad(resized));
        write("pose-padded.png", fashn_prepare::resize_pad(labels, nullptr, true));
        for (const std::string category : {"tops", "bottoms", "one-pieces"}) {
            write(category + "-person.png", fashn_prepare::mask_image(image, labels, category, false));
            write(category + "-garment.png", fashn_prepare::mask_image(image, labels, category, true));
        }
        std::cout << fashn_prepare::sha256(input / "image.png") << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
