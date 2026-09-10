#ifndef __SD_RUNTIME_FASHN_VTON_SAMPLING_H__
#define __SD_RUNTIME_FASHN_VTON_SAMPLING_H__

#include <cmath>
#include <functional>
#include <vector>

#include "model/diffusion/model.hpp"

struct FashnVTONSamplingParams {
    int steps                 = 30;
    float cfg                 = 1.5f;
    float shift               = 1.5f;
    int skip_cfg_last_n_steps = 1;

    bool validate() const {
        if (steps < 1 || steps > 1000 || !std::isfinite(cfg) || cfg < 0.f ||
            !std::isfinite(shift) || std::abs(shift) > 20.f ||
            skip_cfg_last_n_steps < 0 || skip_cfg_last_n_steps > steps) {
            LOG_ERROR("Invalid FASHN sampling parameters: steps 1..1000, finite nonnegative CFG, shift -20..20, skip 0..steps required");
            return false;
        }
        return true;
    }
};

inline std::vector<float> fashn_vton_schedule(const FashnVTONSamplingParams& params) {
    if (!params.validate()) {
        return {};
    }
    std::vector<float> result(params.steps + 1);
    const float step = -1.f / params.steps;
    const float a    = static_cast<float>(std::exp(-static_cast<double>(params.shift)));
    result.front()   = 0.f;
    result.back()    = 1.f;
    for (int i = 1; i < params.steps; ++i) {
        int j = params.steps - i;
        // Match torch.linspace(1,0,N+1), then its reciprocal-based F32 shift.
        float t           = j < (params.steps + 1) / 2 ? 1.f + step * j : -step * i;
        float denominator = a + (1.f / t - 1.f);
        result[i]         = (1.f / denominator) * a;
    }
    for (int i = 1; i <= params.steps; ++i) {
        if (!std::isfinite(result[i]) || result[i] <= result[i - 1]) {
            LOG_ERROR("FASHN shift/step combination produces a collapsed F32 schedule");
            return {};
        }
    }
    return result;
}

inline std::vector<std::pair<float, int>> fashn_vton_modulation_pairs(const FashnVTONSamplingParams& sampling, int category) {
    auto schedule = fashn_vton_schedule(sampling);
    if (schedule.empty())
        return {};
    if (category < 1 || category > 3) {
        LOG_ERROR("FASHN modulation requests require user category 1..3");
        return {};
    }
    std::vector<std::pair<float, int>> pairs;
    for (int i = 0; i < sampling.steps; ++i) {
        pairs.emplace_back(schedule[i], category);
        if (sampling.cfg != 1.f && i < sampling.steps - sampling.skip_cfg_last_n_steps)
            pairs.emplace_back(schedule[i], 0);
    }
    return pairs;
}

inline bool fashn_vton_euler_update(sd::Tensor<float>& image,
                                    const sd::Tensor<float>& conditional,
                                    const sd::Tensor<float>* unconditional,
                                    float cfg,
                                    float delta) {
    if (image.empty() || image.shape() != conditional.shape() ||
        (unconditional != nullptr && image.shape() != unconditional->shape()) ||
        !std::isfinite(cfg) || cfg < 0.f || !std::isfinite(delta) || delta <= 0.f) {
        LOG_ERROR("Invalid FASHN Euler tensor shapes or parameters");
        return false;
    }
    for (int64_t i = 0; i < image.numel(); ++i) {
        const float vc = conditional.values()[i];
        float velocity = vc;
        if (unconditional != nullptr) {
            float vu = unconditional->values()[i];
            velocity = vu + cfg * (vc - vu);
        }
        float updated = image.values()[i] + delta * velocity;
        if (!std::isfinite(vc) || !std::isfinite(velocity) || !std::isfinite(updated)) {
            LOG_ERROR("Non-finite value in FASHN Euler update");
            return false;
        }
        image.values()[i] = updated;
    }
    return true;
}

inline sd::Tensor<float> sample_fashn_vton(
    DiffusionModelRunner& runner,
    int threads,
    const sd::Tensor<float>& initial_noise,
    const FashnVTONDiffusionExtra& condition,
    const FashnVTONSamplingParams& sampling,
    const std::function<bool()>& cancelled        = {},
    const std::function<void(int, int)>& progress = {},
    const std::function<bool(int, const sd::Tensor<float>&, const sd::Tensor<float>&, const sd::Tensor<float>*)>& observe_step = {}) {
    auto schedule = fashn_vton_schedule(sampling);
    if (schedule.empty()) {
        return {};
    }
    if (threads < 1 || initial_noise.shape() != std::vector<int64_t>({576, 864, 3, 1}) ||
        condition.categories == nullptr || condition.categories->shape() != std::vector<int64_t>({1}) ||
        condition.categories->values()[0] < 1 || condition.categories->values()[0] > 3) {
        LOG_ERROR("FASHN sampling requires one 576x864 RGB noise tensor and user category 1..3");
        return {};
    }
    struct EndRunner {
        DiffusionModelRunner& runner;
        ~EndRunner() { runner.runner_end(); }
    } end_runner{runner};
    sd::Tensor<float> image = initial_noise;
    sd::Tensor<float> zero_rgb({576, 864, 3, 1});
    sd::Tensor<float> zero_pose({576, 864, 1, 1});
    sd::Tensor<int32_t> zero_category({1}, {0});
    FashnVTONDiffusionExtra null_condition{&zero_rgb, &zero_rgb, &zero_pose, &zero_pose, &zero_category};
    sd::Tensor<float> time({1});
    DiffusionParams params;
    params.x          = &image;
    params.timesteps  = &time;
    auto is_cancelled = [&]() {
        if (cancelled && cancelled()) {
            LOG_INFO("FASHN sampling cancelled");
            return true;
        }
        return false;
    };
    for (int i = 0; i < sampling.steps; ++i) {
        if (is_cancelled()) {
            return {};
        }
        time.values()[0] = schedule[i];
        params.extra     = condition;
        auto conditional = runner.compute(threads, params);
        if (conditional.empty() || is_cancelled()) {
            return {};
        }
        bool use_cfg = sampling.cfg != 1.f && i < sampling.steps - sampling.skip_cfg_last_n_steps;
        sd::Tensor<float> unconditional;
        if (use_cfg) {
            params.extra  = null_condition;
            unconditional = runner.compute(threads, params);
            if (unconditional.empty() || is_cancelled()) {
                return {};
            }
        }
        if (!fashn_vton_euler_update(image, conditional, use_cfg ? &unconditional : nullptr,
                                     sampling.cfg, schedule[i + 1] - schedule[i])) {
            return {};
        }
        if (progress) {
            progress(i + 1, sampling.steps);
        }
        if (observe_step && !observe_step(i, image, conditional, use_cfg ? &unconditional : nullptr)) {
            LOG_ERROR("FASHN trajectory observer failed");
            return {};
        }
        if (is_cancelled()) {
            return {};
        }
    }
    return image;
}

#endif  // __SD_RUNTIME_FASHN_VTON_SAMPLING_H__
