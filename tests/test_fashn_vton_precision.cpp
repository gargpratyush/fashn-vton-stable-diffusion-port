#include "fashn_test_runner.h"

int main() {
    String2TensorStorage tensors;
    for (const auto& [name, shape] : FashnVTONConfig::expected_shapes()) {
        TensorStorage tensor;
        tensor.n_dims = static_cast<int>(shape.size());
        tensor.type   = GGML_TYPE_BF16;
        for (size_t i = 0; i < shape.size(); ++i)
            tensor.ne[i] = shape[i];
        tensors[FashnVTONConfig::canonical_name(name)] = tensor;
    }
    std::map<std::string, std::string> types;
    if (!assign_fashn_test_matrix_types(tensors, GGML_TYPE_Q8_0, {}, types) || types.size() != 104)
        return 1;
    for (const auto& [name, tensor] : tensors) {
        const auto expected = FashnVTONConfig::is_quantizable_matrix(name, tensor) ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
        if (tensor.expected_type != expected)
            return 1;
    }
    const std::string restored = "x_patch_mixer.0.linear1.weight";
    if (!assign_fashn_test_matrix_types(tensors, GGML_TYPE_Q8_0, {restored}, types) ||
        types[restored] != "f32" || types["x_patch_mixer.0.linear2.weight"] != "q8_0")
        return 1;
    if (assign_fashn_test_matrix_types(tensors, GGML_TYPE_Q8_0, {"nonexistent.weight"}, types) ||
        assign_fashn_test_matrix_types(tensors, GGML_TYPE_Q8_0, {"y_embedder.weight"}, types))
        return 1;
    for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K}) {
        if (fashn_test_matrix_type(ggml_type_name(type)) != type ||
            !assign_fashn_test_matrix_types(tensors, type, {}, types) || types.size() != 104)
            return 1;
        for (const auto& [name, tensor] : tensors) {
            const auto expected = FashnVTONConfig::is_quantizable_matrix(name, tensor) ? type : GGML_TYPE_F32;
            if (tensor.expected_type != expected)
                return 1;
        }
        if (!assign_fashn_test_matrix_types(tensors, type, {restored}, types) ||
            types[restored] != "f32" || types["x_patch_mixer.0.linear2.weight"] != ggml_type_name(type)) {
            return 1;
        }
    }
    if (fashn_test_matrix_type("q2_K") != GGML_TYPE_COUNT ||
        assign_fashn_test_matrix_types(tensors, GGML_TYPE_Q2_K, {}, types))
        return 1;
    auto& row = tensors[FashnVTONConfig::canonical_name(restored)].ne[0];
    const auto original_row = row;
    row = 1248;
    if (assign_fashn_test_matrix_types(tensors, GGML_TYPE_Q4_K, {}, types))
        return 1;
    row = original_row;
    tensors[FashnVTONConfig::canonical_name(restored)].type = GGML_TYPE_Q8_0;
    if (assign_fashn_test_matrix_types(tensors, GGML_TYPE_Q8_0, {restored}, types))
        return 1;
    std::cout << "Diagnostic matrix selection and protected parameters: passed\n";
    return 0;
}
