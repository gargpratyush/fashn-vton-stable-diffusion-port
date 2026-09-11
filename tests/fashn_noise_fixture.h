#ifndef __SD_TESTS_FASHN_NOISE_FIXTURE_H__
#define __SD_TESTS_FASHN_NOISE_FIXTURE_H__

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "fashn_test_utils.h"

inline sd::Tensor<float> load_fashn_noise_fixture(const std::string& path) {
    ModelLoader loader;
    if (!loader.init_from_file(path))
        throw std::invalid_argument("Cannot open initial noise fixture");
    const auto& tensors = loader.get_tensor_storage_map();
    const auto found = tensors.find("noise");
    if (tensors.size() != 1 || found == tensors.end())
        throw std::invalid_argument("Initial noise fixture must contain only 'noise'");
    const auto& metadata = found->second;
    if (metadata.type != GGML_TYPE_F32 || metadata.n_dims < 3 || metadata.n_dims > 4 ||
        metadata.ne[0] != 576 || metadata.ne[1] != 864 || metadata.ne[2] != 3 || metadata.ne[3] != 1)
        throw std::invalid_argument("Initial noise must be F32 with shape [1,3,864,576] or [3,864,576]");
    auto noise = load_fixture(loader, "noise");
    if (noise.empty() || !std::all_of(noise.values().begin(), noise.values().end(),
                                    [](float value) { return std::isfinite(value); }))
        throw std::invalid_argument("Initial noise must contain finite F32 values");
    return noise.reshape({576, 864, 3, 1});
}

#endif
