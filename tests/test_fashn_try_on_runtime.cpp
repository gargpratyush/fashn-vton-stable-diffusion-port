#include <array>
#include <stdexcept>

#include "core/api_boundary.h"
#include "core/rng.hpp"
#include "ggml-cpu.h"
#include "model/diffusion/model.hpp"
#include "runtime/fashn_try_on.h"

struct ZeroRNG : RNG {
    bool fail     = false;
    uint64_t seed = 0;
    void manual_seed(uint64_t value) override { seed = value; }
    std::vector<float> randn(uint32_t count) override {
        if (fail) {
            throw std::bad_alloc();
        }
        return std::vector<float>(count);
    }
};

struct RuntimeRunner : DiffusionModelRunner {
    int calls = 0;
    bool fail = false;
    explicit RuntimeRunner(ggml_backend_t backend)
        : DiffusionModelRunner(backend, "") {}
    std::string get_desc() override { return "runtime-test"; }
    void get_param_tensors(std::map<std::string, ggml_tensor*>&, const std::string&) override {}
    sd::Tensor<float> compute(int, const DiffusionParams& params) override {
        ++calls;
        runner_started_ = true;
        if (fail) {
            throw std::runtime_error("injected forward failure");
        }
        const auto& extra = std::get<FashnVTONDiffusionExtra>(params.extra);
        if (extra.ca_images->values()[0] != 128 / 127.5f - 1.f) {
            throw std::runtime_error("prepared normalization changed");
        }
        return sd::Tensor<float>(params.x->shape());
    }
};

static void fail_progress(int, int, float, void*) {
    throw std::runtime_error("injected progress callback");
}

static bool fail_final_cancel(void* data) {
    int& calls = *static_cast<int*>(data);
    if (++calls == 7) {
        throw std::runtime_error("injected post-allocation callback");
    }
    return false;
}

int main() {
    auto backend = ggml_backend_cpu_init();
    int result   = 0;
    {
        RuntimeRunner runner(backend);
        ZeroRNG rng;
        FashnTryOnRuntime runtime{runner, rng, 1, false, 0, [] {},
                                  [] { return SD_CANCEL_RESET; }, [] { return true; }};
        sd_try_on_params_t params;
        sd_try_on_params_init(&params);
        params.steps      = 1;
        params.cfg        = 1;
        params.crop_x     = 575;
        params.crop_y     = 863;
        params.crop_width = params.crop_height = 1;
        std::vector<uint8_t> rgb(576 * 864 * 3, 128), pose(576 * 864, 0);
        params.ca_image = params.garment_image = {576, 864, 3, rgb.data()};
        params.person_pose = params.garment_pose = {576, 864, 1, pose.data()};
        for (int phase = 0; phase < 6; ++phase) {
            rng.fail          = phase == 1;
            runner.fail       = phase == 2;
            int cancellations = 0;
            sd_try_on_callbacks_t callbacks{sizeof(sd_try_on_callbacks_t),
                                            phase == 3 ? fail_progress : nullptr,
                                            phase == 4 ? fail_final_cancel : nullptr, &cancellations};
            sd_image_t* images = nullptr;
            int count          = 99;
            const bool ok      = sd_api_boundary("injected runtime failure", false, [&] {
                return generate_fashn_try_on(runtime, &params, &callbacks, &images, &count);
            });
            if (phase == 0 || phase == 5) {
                if (!ok || count != 1 || images == nullptr ||
                    images[0].width != 1 || images[0].height != 1 ||
                    images[0].data[0] != 127 || images[0].data[1] != 127 || images[0].data[2] != 127) {
                    result = 1;
                }
            } else if (ok || images != nullptr || count != 0) {
                result = 1;
            }
            if (runner.runner_started()) {
                result = 1;
            }
            free_sd_images(images, count);
        }
        params.crop_width  = 2;
        sd_image_t* images = nullptr;
        int count          = 9;
        if (generate_fashn_try_on(runtime, &params, nullptr, &images, &count) || count != 0) {
            result = 1;
        }
    }
    ggml_backend_free(backend);
    return result;
}
