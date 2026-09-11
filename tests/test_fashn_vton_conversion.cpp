#include <algorithm>
#include <cmath>
#include <iostream>
#include "fashn_test_runner.h"
#include "model/diffusion/fashn_vton.h"
#include "stable-diffusion.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: test-fashn-vton-conversion source converted f32|f16|bf16|q8_0|q4_0|q5_0|q4_K|q5_K"
                     " [--runtime-policy] [--f32-matrices policy.json]\n";
        return 2;
    }
    bool runtime_policy = false;
    std::set<std::string> f32_matrices;
    for (int i = 4; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--runtime-policy") {
            runtime_policy = true;
        } else if (option == "--f32-matrices" && i + 1 < argc) {
            if (!read_fashn_f32_matrices(argv[++i], f32_matrices))
                return 2;
        } else {
            std::cerr << "Unknown or incomplete conversion verification option\n";
            return 2;
        }
    }
    if (!runtime_policy && !f32_matrices.empty()) {
        std::cerr << "F32 restoration requires --runtime-policy\n";
        return 2;
    }
    const std::string mode = argv[3];
    ggml_type target       = fashn_test_matrix_type(mode);
    if (target == GGML_TYPE_COUNT)
        return 2;
    ModelLoader source, converted;
    if (!source.init_from_file_and_convert_name(argv[1]) || !converted.init_from_file_and_convert_name(argv[2]) ||
        source.get_sd_version() != VERSION_FASHN_VTON_1_5 || converted.get_sd_version() != VERSION_FASHN_VTON_1_5)
        return 1;
    if (source.get_tensor_storage_map().size() != converted.get_tensor_storage_map().size())
        return 1;
    std::map<std::string, std::string> matrix_types;
    if (runtime_policy && !assign_fashn_test_matrix_types(source.get_tensor_storage_map(), target, f32_matrices, matrix_types))
        return 1;
    int checked = 0, quantized = 0;
    double worst_relative_l2 = 0;
    for (const auto& [name, tensor] : source.get_tensor_storage_map()) {
        auto match = converted.get_tensor_storage_map().find(name);
        if (match == converted.get_tensor_storage_map().end())
            return 1;
        bool eligible = FashnVTONConfig::is_quantizable_matrix(name, tensor);
        auto wanted = runtime_policy ? tensor.expected_type :
                      (ggml_is_quantized(target) && !eligible ? GGML_TYPE_F32 : target);
        bool q8 = wanted == GGML_TYPE_Q8_0;
        bool quantized_tensor = ggml_is_quantized(wanted);
        if (match->second.type != wanted) {
            std::cerr << "Wrong conversion policy for " << name << "\n";
            return 1;
        }
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (match->second.ne[i] != tensor.ne[i]) {
                std::cerr << "Wrong converted shape for " << name << "\n";
                return 1;
            }
        }
        auto a = load_fixture(source, name), b = load_fixture(converted, name);
        if (a.empty() || a.numel() != b.numel())
            return 1;
        if (quantized_tensor) {
            if (tensor.ne[0] % ggml_blck_size(wanted) != 0)
                return 1;
            std::vector<uint8_t> expected(match->second.nbytes()), actual(expected.size());
            // The converter supplies uniform importance, not a null importance vector.
            std::vector<float> importance(static_cast<size_t>(tensor.ne[0]), 1.f);
            ggml_quantize_chunk(wanted, a.data(), expected.data(), 0,
                                a.numel() / tensor.ne[0], tensor.ne[0], importance.data());
            auto ctx = ggml_init({ggml_tensor_overhead(), nullptr, true});
            if (!ctx)
                return 1;
            std::unique_ptr<ggml_context, decltype(&ggml_free)> guard(ctx, ggml_free);
            auto raw = ggml_new_tensor(ctx, wanted, match->second.n_dims, match->second.ne);
            raw->data = actual.data();
            if (!converted.load_tensor(match->second, raw) || actual != expected) {
                std::cerr << "Quantized blocks differ from original-weight quantization: " << name << "\n";
                return 1;
            }
        }
        double error2 = 0, norm2 = 0;
        for (int64_t base = 0; base < a.numel(); base += 32) {
            float maximum = 0;
            int64_t end   = std::min<int64_t>(a.numel(), base + 32);
            for (int64_t i = base; i < end; ++i)
                maximum = std::max(maximum, std::abs(a.values()[i]));
            // Q8_0 nearest rounding plus FP16 scale storage error, not an
            // inference-equivalence or image-quality acceptance threshold.
            float scale        = maximum / 127.f;
            float stored_scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
            float bound        = scale * .5f + 127.f * std::abs(scale - stored_scale) + maximum * 1e-6f;
            for (int64_t i = base; i < end; ++i) {
                float expected = a.values()[i];
                if (wanted == GGML_TYPE_F16)
                    expected = ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected));
                else if (wanted == GGML_TYPE_BF16)
                    expected = ggml_bf16_to_fp32(ggml_fp32_to_bf16(expected));
                if (!std::isfinite(b.values()[i]) ||
                    (q8 ? std::abs(expected - b.values()[i]) > bound :
                          (!quantized_tensor && expected != b.values()[i]))) {
                    std::cerr << "Tensor roundtrip mismatch: " << name << " index " << i
                              << " expected " << expected << " actual " << b.values()[i] << " bound " << bound << "\n";
                    return 1;
                }
                double error = double(a.values()[i]) - b.values()[i];
                error2 += error * error;
                norm2 += double(a.values()[i]) * a.values()[i];
            }
        }
        worst_relative_l2 = std::max(worst_relative_l2, std::sqrt(error2 / std::max(norm2, 1e-24)));
        ++checked;
        quantized += quantized_tensor;
    }
    if (ggml_is_quantized(target)) {
        sd_ctx_params_t params;
        sd_ctx_params_init(&params);
        params.diffusion_model_path = argv[2];
        auto ctx                    = new_sd_ctx(&params);
        if (ctx != nullptr) {
            free_sd_ctx(ctx);
            std::cerr << "Unvalidated quantized inference must remain gated\n";
            return 1;
        }
    }
    std::cout << "Passed " << checked << " tensors; exact quantized blocks checked for " << quantized
              << " matrices; runtime policy " << runtime_policy << "; worst weight relative L2 "
              << worst_relative_l2 << "\n";
    return 0;
}
