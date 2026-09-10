#ifndef __SD_TESTS_FASHN_TEST_UTILS_H__
#define __SD_TESTS_FASHN_TEST_UTILS_H__

#include "core/tensor.hpp"
#include "core/util.h"
#include "model_loader.h"

inline sd::Tensor<float> load_fixture(ModelLoader& loader, const std::string& name) {
    const auto& tensors = loader.get_tensor_storage_map();
    auto found          = tensors.find(name);
    std::vector<float> data;
    if (found == tensors.end() || !loader.load_float_tensor(name, data, 8)) {
        LOG_ERROR("Cannot read FASHN fixture '%s'", name.c_str());
        return {};
    }
    const auto& metadata = found->second;
    return sd::Tensor<float>(std::vector<int64_t>(metadata.ne, metadata.ne + metadata.n_dims), std::move(data));
}

#endif  // __SD_TESTS_FASHN_TEST_UTILS_H__
