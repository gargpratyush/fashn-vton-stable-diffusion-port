#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include "core/rng_mt19937.hpp"
#include "fashn_test_utils.h"
#include "ggml-cpu.h"
#include "json.hpp"
#include "runtime/fashn_vton_sampling.h"

static int failed = 0;
static void check(bool value, const char* message) {
    if (!value) {
        std::cerr << "FAIL " << message << "\n";
        ++failed;
    }
}

struct FakeFashnRunner : DiffusionModelRunner {
    int calls         = 0;
    bool saw_bad_null = false;
    int fail_at       = -1;
    std::vector<float> times;
    explicit FakeFashnRunner(ggml_backend_t backend)
        : DiffusionModelRunner(backend, "") {}
    std::string get_desc() override { return "fake_fashn"; }
    void get_param_tensors(std::map<std::string, ggml_tensor*>&, const std::string&) override {}
    sd::Tensor<float> compute(int, const DiffusionParams& params) override {
        const auto& extra = std::get<FashnVTONDiffusionExtra>(params.extra);
        ++calls;
        times.push_back(params.timesteps->values()[0]);
        if (calls == fail_at) {
            LOG_ERROR("Injected test forward failure");
            return {};
        }
        bool null = extra.categories->values()[0] == 0;
        if (null) {
            for (const auto* value : {extra.ca_images, extra.garment_images, extra.person_poses, extra.garment_poses}) {
                for (float v : value->values()) {
                    saw_bad_null |= v != 0.f;
                }
            }
        }
        sd::Tensor<float> velocity(params.x->shape());
        velocity.fill_(null ? -1.f : 3.f);
        return velocity;
    }
};

static void test_oracle(const std::filesystem::path& directory) {
    std::ifstream manifest_file(directory / "manifest.json");
    auto manifest = nlohmann::json::parse(manifest_file, nullptr, false);
    if (!manifest.is_object()) {
        check(false, "read oracle manifest");
        return;
    }
    ModelLoader inputs, conditional, unconditional, euler;
    if (!inputs.init_from_file((directory / "inputs.safetensors").string()) ||
        !conditional.init_from_file((directory / "conditional.safetensors").string()) ||
        !unconditional.init_from_file((directory / "unconditional.safetensors").string()) ||
        !euler.init_from_file((directory / "euler.safetensors").string())) {
        check(false, "read oracle tensors");
        return;
    }
    FashnVTONSamplingParams params{manifest.at("steps"), manifest.at("cfg"), manifest.at("shift"),
                                   manifest.at("skip_cfg_last_n_steps")};
    int index              = manifest.at("step_index");
    auto schedule          = fashn_vton_schedule(params);
    auto expected_schedule = load_fixture(inputs, "schedule");
    check(static_cast<int64_t>(schedule.size()) == expected_schedule.numel(), "oracle schedule length");
    float schedule_error = 0;
    for (size_t i = 0; i < schedule.size() && static_cast<int64_t>(i) < expected_schedule.numel(); ++i) {
        schedule_error = std::max(schedule_error, std::abs(schedule[i] - expected_schedule.values()[i]));
    }
    check(schedule_error <= 1e-7f, "full schedule matches original F32 operations");
    auto image = load_fixture(inputs, "noise");
    MT19937RNG rng;
    rng.manual_seed(manifest.at("seed").get<uint64_t>());
    auto noise        = rng.randn(static_cast<uint32_t>(image.numel()));
    float noise_error = 0;
    for (size_t i = 0; i < noise.size(); ++i) {
        noise_error = std::max(noise_error, std::abs(noise[i] - image.values()[i]));
    }
    check(noise_error <= 1e-6f, "CPU RNG matches full-canvas PyTorch noise");
    auto vc       = load_fixture(conditional, "velocity");
    auto vu       = load_fixture(unconditional, "velocity");
    auto expected = load_fixture(euler, "updated_image");
    bool skip     = index >= params.steps - params.skip_cfg_last_n_steps;
    if (index < 0 || index + 1 >= static_cast<int>(schedule.size()) ||
        !fashn_vton_euler_update(image, vc, skip ? nullptr : &vu, params.cfg, schedule[index + 1] - schedule[index]) ||
        image.shape() != expected.shape()) {
        check(false, "oracle Euler update shape/parameters");
        return;
    }
    float euler_error = 0;
    for (int64_t i = 0; i < image.numel(); ++i) {
        euler_error = std::max(euler_error, std::abs(image.values()[i] - expected.values()[i]));
    }
    check(euler_error <= 1e-6f, "full-canvas Euler matches original reference");
    std::cout << "Oracle max errors: schedule=" << schedule_error << " noise=" << noise_error
              << " Euler=" << euler_error << "\n";
}

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "Usage: test-fashn-vton-sampling [native-fixture-directory]\n";
        return 2;
    }
    if (argc == 2) {
        test_oracle(argv[1]);
    }
    FashnVTONSamplingParams params;
    params.steps           = 4;
    auto schedule          = fashn_vton_schedule(params);
    const float expected[] = {0.f, 0.069227785f, 0.182425524f, 0.400978973f, 1.f};
    check(schedule.size() == 5, "four-step schedule length");
    for (size_t i = 0; i < schedule.size(); ++i) {
        check(std::abs(schedule[i] - expected[i]) <= 2e-7f, "original four-step schedule");
    }
    params.steps = 1;
    schedule     = fashn_vton_schedule(params);
    check(schedule == std::vector<float>({0, 1}), "one-step endpoints");
    params.steps = 0;
    check(fashn_vton_schedule(params).empty(), "reject zero steps");
    params.steps = 4;
    params.cfg   = std::numeric_limits<float>::quiet_NaN();
    check(fashn_vton_schedule(params).empty(), "reject nonfinite cfg");
    params.cfg = 1.5f;

    auto backend = ggml_backend_cpu_init();
    if (!backend) {
        return 1;
    }
    {
        sd::Tensor<float> noise({576, 864, 3, 1});
        noise.fill_(1.f);
        sd::Tensor<float> rgb({576, 864, 3, 1});
        rgb.fill_(0.5f);
        sd::Tensor<float> pose({576, 864, 1, 1});
        pose.fill_(-1.f);
        sd::Tensor<int32_t> category({1}, {1});
        FashnVTONDiffusionExtra condition{&rgb, &rgb, &pose, &pose, &category};
        for (float cfg : {0.f, 1.f, 1.5f}) {
            for (int skip : {0, 1, 4}) {
                params.cfg                   = cfg;
                params.skip_cfg_last_n_steps = skip;
                FakeFashnRunner runner(backend);
                int progress_count = 0;
                auto output        = sample_fashn_vton(runner, 1, noise, condition, params, {},
                                                       [&](int step, int total) {
                                                    check(step == ++progress_count && total == 4, "progress sequence");
                                                });
                int expected_calls = 4 + (cfg == 1.f ? 0 : 4 - skip);
                check(runner.calls == expected_calls, "CFG optimization branch count");
                check(!runner.saw_bad_null, "null tensors are normalized zero, not black");
                check(!output.empty(), "sampling returns RGB");
                auto times            = fashn_vton_schedule(params);
                float guided_duration = times[4 - skip];
                float wanted          = 1.f + guided_duration * (-1.f + cfg * 4.f) + (1.f - guided_duration) * 3.f;
                if (!output.empty()) {
                    for (float pixel : output.values()) {
                        if (std::abs(pixel - wanted) > 2e-6f) {
                            check(false, "Euler flow direction, guidance and last-step policy");
                            break;
                        }
                    }
                }
                check(progress_count == 4, "all progress callbacks");
                check(!runner.runner_started(), "runner released after sample");
            }
        }
        params.cfg                   = 1.5f;
        params.skip_cfg_last_n_steps = 1;
        FakeFashnRunner observed(backend);
        int observed_steps   = 0;
        auto observed_result = sample_fashn_vton(observed, 1, noise, condition, params, {}, {},
                                                 [&](int step, const sd::Tensor<float>& image, const sd::Tensor<float>& vc, const sd::Tensor<float>* vu) {
                                                     check(step == observed_steps++, "observer order");
                                                     check(image.shape() == noise.shape() && vc.values()[0] == 3.f, "observer receives updated image and conditional velocity");
                                                     check((vu != nullptr) == (step < 3), "observer CFG branch semantics");
                                                     return true;
                                                 });
        check(!observed_result.empty() && observed_steps == 4, "observer covers complete trajectory");
        FakeFashnRunner observer_failed(backend);
        check(sample_fashn_vton(observer_failed, 1, noise, condition, params, {}, {},
                                [](int, const sd::Tensor<float>&, const sd::Tensor<float>&, const sd::Tensor<float>*) { return false; })
                  .empty(),
              "observer I/O failure propagates");
        check(observer_failed.calls == 2 && !observer_failed.runner_started(), "observer failure stops and releases runner");
        FakeFashnRunner cancelled(backend);
        check(sample_fashn_vton(cancelled, 1, noise, condition, params, [&]() { return cancelled.calls == 1; }).empty(),
              "cancel between branches");
        check(cancelled.calls == 1, "cancellation skips unconditional forward");
        FakeFashnRunner final_cancelled(backend);
        bool cancel_at_end = false;
        check(sample_fashn_vton(final_cancelled, 1, noise, condition, params, [&]() { return cancel_at_end; }, [&](int step, int total) { cancel_at_end = step == total; }).empty(), "cancellation from final progress callback discards output");
        FakeFashnRunner failing(backend);
        failing.fail_at = 2;
        check(sample_fashn_vton(failing, 1, noise, condition, params).empty(), "forward failure propagates");
        category.values()[0] = 0;
        check(sample_fashn_vton(failing, 1, noise, condition, params).empty(), "public null category rejected");
    }
    ggml_backend_free(backend);
    std::cout << "FASHN sampler: " << failed << " failures\n";
    return failed ? 1 : 0;
}
