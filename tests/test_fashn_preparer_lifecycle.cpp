#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "json.hpp"
#include "pipeline.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

static nlohmann::ordered_json process_memory() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)))
        throw std::runtime_error("Cannot read process memory");
    return {{"working_set_bytes", counters.WorkingSetSize}, {"peak_working_set_bytes", counters.PeakWorkingSetSize},
            {"private_commit_bytes", counters.PrivateUsage}};
#else
    return nullptr;
#endif
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: test-fashn-preparer-lifecycle pose-dir parser-dir person garment new-output lazy|eager|tamper-license [--threads N] [--no-arena] [--no-spinning]\n";
        return 2;
    }
    try {
        const std::string mode = argv[6];
        if (mode != "lazy" && mode != "eager" && mode != "tamper-license")
            throw std::runtime_error("Expected lazy, eager or tamper-license");
        const bool lazy = mode != "eager";
        fashn_prepare::CPUOptions cpu;
        for (int i = 7; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--no-arena")
                cpu.arena = false;
            else if (option == "--no-spinning")
                cpu.spinning = false;
            else if (option == "--threads" && i + 1 < argc) {
                const std::string text = argv[++i];
                size_t consumed = 0;
                cpu.intra_threads = std::stoi(text, &consumed);
                if (consumed != text.size() || cpu.intra_threads < 1 || cpu.intra_threads > 64)
                    throw std::runtime_error("Expected thread count in [1,64]");
            } else
                throw std::runtime_error("Unknown or incomplete runtime option");
        }
        const std::filesystem::path output(argv[5]);
        if (std::filesystem::exists(output))
            throw std::runtime_error("Use a new output directory");
        std::filesystem::create_directories(output);
        std::filesystem::path parser_directory(argv[2]);
        if (mode == "tamper-license") {
            const auto fixture = output / "parser-fixture";
            std::filesystem::create_directory(fixture);
            for (const auto* name : {"manifest.json", "LICENSE"})
                std::filesystem::copy_file(parser_directory / name, fixture / name);
            std::filesystem::create_hard_link(parser_directory / "parser.onnx", fixture / "parser.onnx");
            parser_directory = fixture;
        }
        nlohmann::ordered_json report{{"passed", false}, {"lazy", lazy}};
        report["cpu_options"] = {{"intra_threads", cpu.intra_threads}, {"arena", cpu.arena}, {"spinning", cpu.spinning}};
        auto save = [&]() {
            std::ofstream stream(output / "report.json");
            stream << report.dump(2) << "\n";
            stream.close();
            if (!stream)
                throw std::runtime_error("Cannot write lifecycle report");
        };
        save();
        auto start = std::chrono::steady_clock::now();
        fashn_prepare::Preparer preparer(argv[1], parser_directory, true, lazy, cpu);
        report["startup_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (!preparer.has_parser() || preparer.parser_loaded() == lazy)
            throw std::runtime_error("Wrong configured/loaded startup state");
        report["parser_loaded_at_startup"] = preparer.parser_loaded();
        report["startup_process"] = process_memory();
        save();
        auto person = cv::imread(argv[3], cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        auto garment = cv::imread(argv[4], cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        if (mode == "tamper-license") {
            std::ofstream changed(parser_directory / "LICENSE", std::ios::app);
            changed << "\nTest-only modified license fixture\n";
            changed.close();
            if (!changed)
                throw std::runtime_error("Cannot modify isolated license fixture");
            bool rejected = false;
            try {
                preparer.prepare(person, garment, "tops", "model", true);
            } catch (const std::runtime_error& error) {
                rejected = std::string(error.what()) == "Evaluation parser artifacts changed after startup validation";
                if (!rejected)
                    throw;
            }
            if (!rejected || preparer.parser_loaded())
                throw std::runtime_error("Changed license was accepted or parser loaded before integrity rejection");
            for (const auto* name : {"parser.onnx", "manifest.json", "LICENSE"})
                std::filesystem::remove(parser_directory / name);
            std::filesystem::remove(parser_directory);
            report["passed"] = true;
            report["changed_license_rejected"] = true;
            save();
            return 0;
        }
        bool cancelled = false;
        try {
            preparer.prepare(person, garment, "tops", "model", true, [] { return true; });
        } catch (const fashn_prepare::PreparationCancelled&) {
            cancelled = true;
        }
        if (!cancelled || preparer.parser_loaded() == lazy)
            throw std::runtime_error("Cancellation unexpectedly loaded the parser");
        for (int index = 0; index < 4; ++index) {
            const std::string photo = index == 1 || index == 3 ? "model" : "flat-lay";
            const bool free = index != 2;
            start = std::chrono::steady_clock::now();
            auto prepared = preparer.prepare(person, garment, "tops", photo, free);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (preparer.parser_loaded() != (!lazy || index > 0))
                throw std::runtime_error("Wrong first-use parser loading state");
            for (size_t image = 0; image < prepared.images.size(); ++image)
                if (!cv::imwrite((output / (std::to_string(index) + "-" + std::to_string(image) + ".png")).string(),
                                 prepared.images[image]))
                    throw std::runtime_error("Cannot write prepared fixture");
            report["requests"].push_back({{"index", index}, {"photo", photo}, {"segmentation_free", free},
                                          {"seconds", seconds}, {"parser_loaded", preparer.parser_loaded()},
                                          {"process", process_memory()},
                                          {"crop", {prepared.crop.x, prepared.crop.y, prepared.crop.width, prepared.crop.height}}});
            save();
        }
        report["passed"] = true;
        save();
        std::cout << report.dump(2) << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Preparer lifecycle failed: " << error.what() << "\n";
        return 1;
    }
}
