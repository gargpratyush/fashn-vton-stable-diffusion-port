#ifndef __SD_MODEL_DIFFUSION_FASHN_VTON_H__
#define __SD_MODEL_DIFFUSION_FASHN_VTON_H__

#include <cinttypes>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "core/util.h"
#include "model.h"

struct FashnVTONConfig {
    int64_t hidden_size       = 1280;
    int64_t mlp_hidden_size   = 5120;
    int64_t num_heads         = 10;
    int64_t head_dim          = 128;
    int64_t patch_size        = 12;
    int64_t target_channels   = 7;
    int64_t garment_channels  = 4;
    int64_t patch_mixer_depth = 4;
    int64_t double_depth      = 8;
    int64_t single_depth      = 16;
    int64_t category_rows     = 4;
    int64_t time_embed_dim    = 256;
    int input_height          = 864;
    int input_width           = 576;
    int theta                 = 10000;
    std::vector<int> axes_dim = {16, 56, 56};

    static std::string tensor_suffix(const std::string& name,
                                     const std::string& prefix = "model.diffusion_model") {
        const std::string root = prefix.empty() ? "" : prefix + ".";
        return !root.empty() && starts_with(name, root) ? name.substr(root.size()) : name;
    }

    static bool is_model_root(const std::string& name) {
        return name == "patch_mixer_token" ||
               starts_with(name, "x_embedder.") ||
               starts_with(name, "garment_embedder.") ||
               starts_with(name, "t_embedder.") ||
               starts_with(name, "y_embedder.") ||
               starts_with(name, "x_patch_mixer.") ||
               starts_with(name, "double_blocks.") ||
               starts_with(name, "single_blocks.") ||
               starts_with(name, "final_layer.");
    }

    static bool is_quantizable_matrix(const std::string& name, const TensorStorage& tensor) {
        const auto suffix = tensor_suffix(name);
        return (starts_with(suffix, "x_patch_mixer.") || starts_with(suffix, "double_blocks.") ||
                starts_with(suffix, "single_blocks.")) &&
               ends_with(suffix, ".weight") && !contains(suffix, ".modulation.") &&
               !contains(suffix, "_mod.") && tensor.n_dims == 2 && tensor.ne[0] % 32 == 0;
    }

    static std::string canonical_name(const std::string& name) {
        if (is_model_root(name)) {
            return "model.diffusion_model." + name;
        }
        return name;
    }

    static bool is_candidate_tensor(const std::string& name, const TensorStorage& tensor) {
        const std::string suffix = tensor_suffix(name);
        return starts_with(suffix, "garment_embedder.") ||
               starts_with(suffix, "x_patch_mixer.") ||
               suffix == "patch_mixer_token" ||
               (suffix == "x_embedder.proj.weight" &&
                tensor.ne[0] == 12 && tensor.ne[1] == 12 &&
                tensor.ne[2] == 7 && tensor.ne[3] == 1280);
    }

    static bool is_candidate(const String2TensorStorage& tensors) {
        for (const auto& [name, tensor] : tensors) {
            if (is_candidate_tensor(name, tensor)) {
                return true;
            }
        }
        return false;
    }

    static bool validate_file_names(const std::vector<TensorStorage>& tensors,
                                    const std::string& load_prefix,
                                    std::string* error) {
        bool candidate = false;
        for (const auto& tensor : tensors) {
            candidate |= is_candidate_tensor(tensor.name, tensor);
        }
        if (error != nullptr) {
            error->clear();
        }
        if (!candidate) {
            return true;
        }
        std::set<std::string> names;
        for (const auto& tensor : tensors) {
            std::string name = starts_with(tensor.name, load_prefix) ? tensor.name : load_prefix + tensor.name;
            name             = canonical_name(name);
            if (!names.insert(name).second) {
                if (error != nullptr) {
                    *error = "FASHN tensor prefix collision before loading: " + name;
                }
                return false;
            }
        }
        return true;
    }

    static FashnVTONConfig detect_from_weights(const String2TensorStorage& tensors,
                                               const std::string& prefix = "model.diffusion_model") {
        FashnVTONConfig config;
        std::set<std::string> patch_blocks, double_blocks, single_blocks;
        bool inferred = false;
        for (const auto& [name, tensor] : tensors) {
            const std::string suffix = tensor_suffix(name, prefix);
            if (suffix == "x_embedder.proj.weight") {
                config.hidden_size     = tensor.ne[3];
                config.patch_size      = tensor.ne[0];
                config.target_channels = tensor.ne[2];
                inferred               = true;
            } else if (suffix == "garment_embedder.proj.weight") {
                config.garment_channels = tensor.ne[2];
            } else if (suffix == "t_embedder.mlp.in_layer.weight") {
                config.time_embed_dim = tensor.ne[0];
            } else if (suffix == "y_embedder.weight") {
                config.category_rows = tensor.ne[1];
            } else if (suffix == "double_blocks.0.img_attn.norm.query_norm.scale") {
                config.head_dim = tensor.ne[0];
            } else if (suffix == "double_blocks.0.img_mlp.0.weight") {
                config.mlp_hidden_size = tensor.ne[1];
            }
            auto collect_index = [&](const std::string& family, std::set<std::string>& indices) {
                if (starts_with(suffix, family)) {
                    size_t end = suffix.find('.', family.size());
                    if (end != std::string::npos) {
                        indices.insert(suffix.substr(family.size(), end - family.size()));
                    }
                }
            };
            collect_index("x_patch_mixer.", patch_blocks);
            collect_index("double_blocks.", double_blocks);
            collect_index("single_blocks.", single_blocks);
        }
        config.patch_mixer_depth = static_cast<int64_t>(patch_blocks.size());
        config.double_depth      = static_cast<int64_t>(double_blocks.size());
        config.single_depth      = static_cast<int64_t>(single_blocks.size());
        config.num_heads         = config.head_dim > 0 && config.hidden_size % config.head_dim == 0
                                       ? config.hidden_size / config.head_dim
                                       : 0;
        if (inferred) {
            LOG_VERBOSE("fashn vton: hidden_size = %" PRId64 ", patch_size = %" PRId64
                        ", patch_mixer_depth = %" PRId64 ", double_depth = %" PRId64 ", single_depth = %" PRId64,
                        config.hidden_size, config.patch_size, config.patch_mixer_depth,
                        config.double_depth, config.single_depth);
        }
        return config;
    }

    static std::map<std::string, std::vector<int64_t>> expected_shapes() {
        std::map<std::string, std::vector<int64_t>> shapes;
        auto linear = [&](const std::string& name, int64_t output, int64_t input) {
            shapes[name + ".weight"] = {input, output};
            shapes[name + ".bias"]   = {output};
        };
        auto norm = [&](const std::string& name) {
            shapes[name + ".query_norm.scale"] = {128};
            shapes[name + ".key_norm.scale"]   = {128};
        };
        shapes["x_embedder.proj.weight"]       = {12, 12, 7, 1280};
        shapes["x_embedder.proj.bias"]         = {1280};
        shapes["garment_embedder.proj.weight"] = {12, 12, 4, 1280};
        shapes["garment_embedder.proj.bias"]   = {1280};
        linear("t_embedder.mlp.in_layer", 1280, 256);
        linear("t_embedder.mlp.out_layer", 1280, 1280);
        shapes["y_embedder.weight"] = {1280, 4};
        for (const auto& [family, depth] : std::map<std::string, int>{{"x_patch_mixer", 4}, {"single_blocks", 16}}) {
            for (int index = 0; index < depth; ++index) {
                const std::string root = family + "." + std::to_string(index) + ".";
                linear(root + "linear1", 8960, 1280);
                linear(root + "linear2", 1280, 6400);
                linear(root + "modulation.lin", 3840, 1280);
                norm(root + "norm");
            }
        }
        for (int index = 0; index < 8; ++index) {
            for (const std::string stream : {"img", "txt"}) {
                const std::string root = "double_blocks." + std::to_string(index) + "." + stream + "_";
                linear(root + "attn.qkv", 3840, 1280);
                linear(root + "attn.proj", 1280, 1280);
                norm(root + "attn.norm");
                linear(root + "mod.lin", 7680, 1280);
                linear(root + "mlp.0", 5120, 1280);
                linear(root + "mlp.2", 1280, 5120);
            }
        }
        linear("final_layer.linear", 432, 1280);
        linear("final_layer.adaLN_modulation.1", 2560, 1280);
        shapes["patch_mixer_token"] = {432, 1, 1};
        return shapes;
    }

    bool validate_v15(const String2TensorStorage& tensors,
                      const std::string& prefix,
                      std::string* error) const {
        auto fail = [&](const std::string& message) {
            if (error != nullptr) {
                *error = message;
            }
            return false;
        };
        if (error != nullptr) {
            error->clear();
        }
        const auto expected = expected_shapes();
        std::set<std::string> found;
        for (const auto& [name, tensor] : tensors) {
            const std::string suffix = tensor_suffix(name, prefix);
            if (!found.insert(suffix).second) {
                return fail("FASHN tensor prefix collision: " + suffix);
            }
            auto shape = expected.find(suffix);
            if (shape == expected.end()) {
                return fail("Unexpected FASHN tensor: " + name);
            }
            if (tensor.n_dims < 1 || tensor.n_dims > SD_MAX_DIMS) {
                return fail("Invalid FASHN tensor rank: " + name);
            }
            // GGUF may omit trailing singleton dimensions, including the unused buffer's.
            for (int dim = 0; dim < SD_MAX_DIMS; ++dim) {
                int64_t wanted = dim < static_cast<int>(shape->second.size()) ? shape->second[dim] : 1;
                if (tensor.ne[dim] != wanted || (dim >= tensor.n_dims && wanted != 1)) {
                    return fail("FASHN tensor shape mismatch: " + name);
                }
            }
            bool q8_matrix = tensor.type == GGML_TYPE_Q8_0 && is_quantizable_matrix(name, tensor);
            if ((tensor.type != GGML_TYPE_F32 && tensor.type != GGML_TYPE_F16 && tensor.type != GGML_TYPE_BF16 && !q8_matrix) ||
                tensor.is_f8_e4m3 || tensor.is_f8_e5m2 || tensor.is_f64 ||
                tensor.is_i64 || tensor.is_int8_tensorwise) {
                return fail("Unsupported FASHN tensor dtype (floating or eligible Q8_0 matrix required): " + name);
            }
        }
        for (const auto& [name, shape] : expected) {
            if (name != "patch_mixer_token" && found.count(name) == 0) {
                return fail("Missing FASHN tensor: " + name);
            }
        }
        if (hidden_size != 1280 || mlp_hidden_size != 5120 || num_heads != 10 || head_dim != 128 ||
            patch_size != 12 || target_channels != 7 || garment_channels != 4 ||
            patch_mixer_depth != 4 || double_depth != 8 || single_depth != 16 ||
            category_rows != 4 || time_embed_dim != 256 ||
            input_height != 864 || input_width != 576 || theta != 10000 ||
            axes_dim != std::vector<int>({16, 56, 56})) {
            return fail("Unsupported FASHN configuration; only the released VTON 1.5 preset is recognized");
        }
        return true;
    }
};

#endif  // __SD_MODEL_DIFFUSION_FASHN_VTON_H__
