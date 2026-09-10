#ifndef __SD_TESTS_FASHN_TEST_RUNNER_H__
#define __SD_TESTS_FASHN_TEST_RUNNER_H__

#include <fstream>
#include <iostream>
#include "fashn_test_utils.h"
#include "json.hpp"
#include "model/diffusion/fashn_vton_model.h"
#include "model_io/safetensors_io.h"

inline bool read_fashn_f32_matrices(const std::string& path, std::set<std::string>& names) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Cannot open F32 matrix policy: " << path << "\n";
        return false;
    }
    try {
        nlohmann::json policy;
        file >> policy;
        if (!policy.is_object() || policy.size() != 1 || !policy.contains("f32_matrices") ||
            !policy["f32_matrices"].is_array())
            throw std::invalid_argument("Expected only an f32_matrices array");
        for (const auto& value : policy["f32_matrices"]) {
            if (!value.is_string())
                throw std::invalid_argument("Matrix names must be strings");
            auto name = FashnVTONConfig::tensor_suffix(value.get<std::string>());
            if (name.empty() || !names.insert(name).second)
                throw std::invalid_argument("Empty or duplicate matrix name");
        }
    } catch (const nlohmann::json::exception& error) {
        std::cerr << "Invalid matrix policy JSON: " << error.what() << "\n";
        return false;
    } catch (const std::invalid_argument& error) {
        std::cerr << "Invalid matrix policy: " << error.what() << "\n";
        return false;
    }
    return true;
}

inline bool assign_fashn_test_matrix_types(String2TensorStorage& tensors, ggml_type matrix_type, const std::set<std::string>& f32_matrices, std::map<std::string, std::string>& matrix_types) {
    matrix_types.clear();
    for (auto& [name, tensor] : tensors) {
        const auto suffix   = FashnVTONConfig::tensor_suffix(name);
        const bool eligible = FashnVTONConfig::is_quantizable_matrix(name, tensor);
        if (f32_matrices.count(suffix) && ggml_is_quantized(tensor.type)) {
            std::cerr << "Restoring precision requires floating source weights: " << name << "\n";
            return false;
        }
        tensor.expected_type = eligible && !f32_matrices.count(suffix) ? matrix_type : GGML_TYPE_F32;
        if (eligible)
            matrix_types[suffix] = ggml_type_name(tensor.expected_type);
    }
    if (matrix_types.size() != 104) {
        std::cerr << "Expected exactly 104 eligible matrices\n";
        return false;
    }
    for (const auto& name : f32_matrices) {
        if (!matrix_types.count(name)) {
            std::cerr << "Not an eligible matrix: " << name << "\n";
            return false;
        }
    }
    return true;
}

inline ggml_backend_t initialize_backend(const std::string& name) {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto device = ggml_backend_dev_get(i);
        if (name == ggml_backend_dev_name(device)) {
            auto backend = ggml_backend_dev_init(device, nullptr);
            if (!backend) {
                std::cerr << "Cannot initialize backend: " << name << "\n";
                return nullptr;
            }
            std::cout << "Backend: " << ggml_backend_name(backend) << "; device: " << name
                      << "; " << ggml_backend_dev_description(device) << "\n";
            return backend;
        }
    }
    std::cerr << "Requested backend is unavailable: " << name << "; use --list-backends\n";
    return nullptr;
}

struct FashnGraphTestContext {
    ggml_backend_t backend;
    std::shared_ptr<ModelManager> manager = std::make_shared<ModelManager>();
    std::unique_ptr<FashnVTONRunner> runner;
    std::map<std::string, std::string> matrix_types;
    bool uses_mmap = true;

    explicit FashnGraphTestContext(const std::string& name)
        : backend(initialize_backend(name)) {}
    ~FashnGraphTestContext() { close(); }
    void close() {
        if (runner) {
            runner->runner_end();
            manager->unregister_param_tensors("fashn");
        }
        runner.reset();
        manager.reset();
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
    }
    bool init(const std::string& checkpoint, bool flash_attention = false, ggml_type matrix_type = GGML_TYPE_F32, bool upcast_matrices = false, int threads = 8, const std::set<std::string>& f32_matrices = {}, int load_threads = 0, bool mmap = true) {
        if (!backend || threads < 1)
            return false;
        manager->set_n_threads(threads);
        manager->set_enable_mmap(mmap);
        uses_mmap = mmap;
        manager->set_segmented_compute_disabled(true);
        auto& loader = manager->loader();
        if (load_threads > 0)
            loader.set_n_threads(load_threads);
        if (!loader.init_from_file_and_convert_name(checkpoint) ||
            loader.get_sd_version() != VERSION_FASHN_VTON_1_5)
            return false;
        if (!assign_fashn_test_matrix_types(loader.get_tensor_storage_map(), matrix_type, f32_matrices, matrix_types))
            return false;
        runner = std::make_unique<FashnVTONRunner>(backend, loader.get_tensor_storage_map(), "model.diffusion_model", manager);
        runner->set_flash_attention_enabled(flash_attention);
        runner->full_precision_matrix_compute = upcast_matrices || matrix_type == GGML_TYPE_F32;
        if (std::string(ggml_backend_name(backend)) == "BLAS" && !runner->enable_blas_cpu_fallback(threads))
            return false;
        std::map<std::string, ggml_tensor*> params;
        runner->get_param_tensors(params);
        if (params.size() != 365) {
            std::cerr << "Wrong registered tensor count: " << params.size() << "\n";
            return false;
        }
        return manager->register_runner_params("fashn", *runner, ModelManager::ResidencyMode::ParamBackend, backend, backend) &&
               manager->validate_registered_tensors();
    }

    bool prepare_modulations(int threads, const std::vector<std::pair<float, int>>& pairs,
                             size_t max_bytes = 128 * 1024 * 1024, std::function<bool()> cancelled = {}) {
        if (uses_mmap) {
            std::cerr << "Experimental modulation precomputation requires --mmap off to avoid retaining source views\n";
            return false;
        }
        return runner->prepare_modulation_request(threads, pairs, [this]() {
            runner->runner_end();
            return manager->unregister_param_tensors("fashn") &&
                   manager->register_runner_params("fashn", *runner, ModelManager::ResidencyMode::ParamBackend, backend, backend) &&
                   manager->validate_registered_tensors();
        }, max_bytes, std::move(cancelled));
    }
};

inline bool save_capture(const std::string& path, const std::map<std::string, sd::Tensor<float>>& captures) {
    ggml_init_params init = {ggml_tensor_overhead() * captures.size(), nullptr, true};
    ggml_context* ctx     = ggml_init(init);
    if (!ctx)
        return false;
    std::vector<TensorWriteInfo> tensors;
    for (const auto& [name, value] : captures) {
        TensorWriteInfo info;
        info.n_dims = static_cast<int>(value.dim());
        for (int i = 0; i < info.n_dims; ++i)
            info.ne[i] = value.shape()[i];
        info.tensor       = ggml_new_tensor(ctx, GGML_TYPE_F32, info.n_dims, info.ne);
        info.tensor->data = const_cast<float*>(value.data());
        ggml_set_name(info.tensor, name.c_str());
        tensors.push_back(info);
    }
    std::string error;
    bool success = write_safetensors_file(path, tensors, &error);
    ggml_free(ctx);
    if (!success)
        LOG_ERROR("%s", error.c_str());
    return success;
}
#endif
