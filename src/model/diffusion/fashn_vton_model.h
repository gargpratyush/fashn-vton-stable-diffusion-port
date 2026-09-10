#ifndef __SD_MODEL_DIFFUSION_FASHN_VTON_MODEL_H__
#define __SD_MODEL_DIFFUSION_FASHN_VTON_MODEL_H__

#include <array>
#include <cmath>
#include <stdexcept>

#include "model/diffusion/fashn_vton.h"
#include "model/diffusion/fashn_modulation.h"
#include "model/diffusion/flux.hpp"

struct FashnCachedConditioning {
    ggml_tensor* time;
    ggml_tensor* category;
    ggml_tensor* vec;
    std::vector<ggml_tensor*> layers;
};

class FashnPatchEmbed : public UnaryBlock {
    int64_t channels_;
    int64_t hidden_;
    int patch_;

protected:
    void init_params(ggml_context* ctx, const String2TensorStorage& tensors, const std::string prefix) override {
        auto type        = get_type(prefix + "weight", tensors, GGML_TYPE_F32);
        params["weight"] = ggml_new_tensor_4d(ctx, type, patch_, patch_, channels_, hidden_);
        params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_);
    }

public:
    FashnPatchEmbed(int64_t channels, int64_t hidden, int patch)
        : channels_(channels), hidden_(hidden), patch_(patch) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto patches = DiT::patchify(ctx->ggml_ctx, x, patch_, patch_, true);
        auto weight  = ggml_reshape_2d(ctx->ggml_ctx, params.at("weight"), patch_ * patch_ * channels_, hidden_);
        return ggml_ext_linear(ctx->ggml_ctx, patches, weight, params.at("bias"), true);
    }
};

class FashnVTONModel : public GGMLBlock {
    FashnVTONConfig config_;

    static void capture_patch(GGMLRunnerContext* ctx, const std::string& name, ggml_tensor* x, int64_t w, int64_t h) {
        if (ctx->debug_tensors == nullptr) {
            return;
        }
        auto spatial = ggml_cont(ctx->ggml_ctx, ggml_transpose(ctx->ggml_ctx, x));
        spatial      = ggml_reshape_4d(ctx->ggml_ctx, spatial, w, h, x->ne[0], 1);
        ctx->capture_tensor(name, spatial);
    }

public:
    explicit FashnVTONModel(const FashnVTONConfig& config)
        : config_(config) {
        blocks["x_embedder.proj"]       = std::make_shared<FashnPatchEmbed>(7, config.hidden_size, 12);
        blocks["garment_embedder.proj"] = std::make_shared<FashnPatchEmbed>(4, config.hidden_size, 12);
        blocks["t_embedder.mlp"]        = std::make_shared<Flux::MLPEmbedder>(256, config.hidden_size);
        blocks["y_embedder"]            = std::make_shared<Embedding>(4, config.hidden_size);
        for (int64_t i = 0; i < config.patch_mixer_depth; ++i) {
            blocks["x_patch_mixer." + std::to_string(i)] =
                std::make_shared<Flux::SingleStreamBlock>(config.hidden_size, config.num_heads);
        }
        for (int64_t i = 0; i < config.double_depth; ++i) {
            blocks["double_blocks." + std::to_string(i)] =
                std::make_shared<Flux::DoubleStreamBlock>(config.hidden_size, config.num_heads, 4.f, static_cast<int>(i), true);
        }
        for (int64_t i = 0; i < config.single_depth; ++i) {
            blocks["single_blocks." + std::to_string(i)] =
                std::make_shared<Flux::SingleStreamBlock>(config.hidden_size, config.num_heads);
        }
        blocks["final_layer"] = std::make_shared<Flux::LastLayer>(config.hidden_size, 12, 3);
    }

    std::array<ggml_tensor*, 3> conditioning(GGMLRunnerContext* ctx, ggml_tensor* times, ggml_tensor* category) {
        auto c = ctx->ggml_ctx;
        auto features = ggml_ext_timestep_embedding(c, times, 256, 10000, 1000.f);
        auto t = std::static_pointer_cast<Flux::MLPEmbedder>(blocks.at("t_embedder.mlp"))->forward(ctx, features);
        auto y = std::static_pointer_cast<Embedding>(blocks.at("y_embedder"))->forward(ctx, category);
        y = ggml_reshape_2d(c, y, config_.hidden_size, 1);
        return {t, y, ggml_add(c, t, y)};
    }

    ggml_tensor* project_modulation(GGMLRunnerContext* ctx, ggml_tensor* vec, size_t index) {
        const auto& layer = fashn_modulation_layers().at(index);
        if (layer.stream == 2)
            return std::static_pointer_cast<Flux::LastLayer>(blocks.at(layer.block))->project_modulation(ctx, vec);
        if (layer.stream >= 0)
            return std::static_pointer_cast<Flux::DoubleStreamBlock>(blocks.at(layer.block))->project_modulation(ctx, vec, layer.stream == 0);
        return std::static_pointer_cast<Flux::SingleStreamBlock>(blocks.at(layer.block))->project_modulation(ctx, vec);
    }

    static FashnCachedConditioning cached_conditioning(ggml_context* c, ggml_tensor* data) {
        auto view = [&](size_t offset, int width) {
            return ggml_reshape_2d(c, ggml_view_1d(c, data, width, offset * sizeof(float)), width, 1);
        };
        FashnCachedConditioning result{view(0, 1280), view(1280, 1280), view(2560, 1280), {}};
        size_t offset = 3840;
        for (const auto& layer : fashn_modulation_layers()) {
            result.layers.push_back(view(offset, layer.width));
            offset += layer.width;
        }
        return result;
    }

    static std::vector<Flux::ModulationOut> supplied_mods(GGMLRunnerContext* ctx, ggml_tensor* raw) {
        auto shaped = ggml_reshape_3d(ctx->ggml_ctx, raw, 1280, 1, raw->ne[0] / 1280);
        std::vector<Flux::ModulationOut> result{Flux::ModulationOut(ctx, shaped, 0)};
        if (raw->ne[0] == 7680)
            result.emplace_back(ctx, shaped, 3);
        return result;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* times, ggml_tensor* ca, ggml_tensor* garment, ggml_tensor* pose, ggml_tensor* garment_pose, ggml_tensor* category, ggml_tensor* pe, const FashnCachedConditioning* cached = nullptr) {
        auto c      = ctx->ggml_ctx;
        auto target = ggml_concat(c, ggml_concat(c, x, ca, 2), pose, 2);
        auto source = ggml_concat(c, garment, garment_pose, 2);
        auto img    = std::static_pointer_cast<FashnPatchEmbed>(blocks.at("x_embedder.proj"))->forward(ctx, target);
        auto txt    = std::static_pointer_cast<FashnPatchEmbed>(blocks.at("garment_embedder.proj"))->forward(ctx, source);
        capture_patch(ctx, "x_embedder.call0.output", img, 48, 72);
        capture_patch(ctx, "garment_embedder.call0.output", txt, 48, 72);
        auto conditioning_values = cached ? std::array<ggml_tensor*, 3>{cached->time, cached->category, cached->vec} :
                                            conditioning(ctx, times, category);
        ctx->capture_tensor("t_embedder.call0.output", conditioning_values[0]);
        ctx->capture_tensor("y_embedder.call0.output", conditioning_values[1]);
        auto vec = conditioning_values[2];
        ctx->capture_tensor("double_blocks.0.call0.kwargs.vec", vec);
        ctx->capture_tensor("pe_embedder.call0.output", pe);
        ctx->capture_tensor("pe_embedder.call1.output", pe);
        size_t mod_index = 0;
        auto mods = [&]() {
            return cached ? supplied_mods(ctx, cached->layers.at(mod_index++)) : std::vector<Flux::ModulationOut>{};
        };
        for (int64_t i = 0; i < config_.patch_mixer_depth; ++i) {
            std::string name = "x_patch_mixer." + std::to_string(i);
            img              = std::static_pointer_cast<Flux::SingleStreamBlock>(blocks.at(name))->forward(ctx, img, vec, pe, nullptr, mods());
            if (i == 0 || i == config_.patch_mixer_depth - 1) {
                ctx->capture_tensor(name + ".call0.output", img);
            }
        }
        auto joint_pe = ggml_concat(c, pe, pe, 3);
        for (int64_t i = 0; i < config_.double_depth; ++i) {
            std::string name = "double_blocks." + std::to_string(i);
            auto img_mods = mods();
            auto txt_mods = mods();
            auto result      = std::static_pointer_cast<Flux::DoubleStreamBlock>(blocks.at(name))->forward(ctx, img, txt, vec, joint_pe, nullptr, img_mods, txt_mods);
            img              = result.first;
            txt              = result.second;
            if (i == 0 || i == config_.double_depth - 1) {
                ctx->capture_tensor(name + ".call0.output.0", img);
                ctx->capture_tensor(name + ".call0.output.1", txt);
            }
        }
        img = ggml_concat(c, txt, img, 1);
        for (int64_t i = 0; i < config_.single_depth; ++i) {
            std::string name = "single_blocks." + std::to_string(i);
            img              = std::static_pointer_cast<Flux::SingleStreamBlock>(blocks.at(name))->forward(ctx, img, vec, joint_pe, nullptr, mods());
            if (i == 0 || i == config_.single_depth - 1) {
                ctx->capture_tensor(name + ".call0.output", img);
            }
        }
        img          = ggml_ext_slice(c, img, 1, txt->ne[1], img->ne[1]);
        auto patches = std::static_pointer_cast<Flux::LastLayer>(blocks.at("final_layer"))->forward(ctx, img, vec, cached ? cached->layers.at(mod_index) : nullptr);
        ctx->capture_tensor("final_layer.call0.output", patches);
        auto velocity = DiT::unpatchify(c, patches, 72, 48, 12, 12, true);
        ctx->capture_tensor("velocity", velocity);
        return velocity;
    }
};

struct FashnVTONRunner : public DiffusionModelRunner {
    FashnVTONConfig config;
    FashnVTONModel model;
    sd::Tensor<float> positions;
    int last_flash_attention_nodes     = 0;
    bool full_precision_matrix_compute = true;
    bool fused_gelu = false;
    int last_supported_nodes           = 0;
    int last_f32_matmuls               = 0;
    std::string last_unsupported_operation;
    std::map<std::pair<uint32_t, int>, sd::Tensor<float>> modulation_cache;
    bool modulation_cache_required = false;
    bool modulation_cache_complete = false;
    size_t modulation_inactive_bytes = 0;
    int modulation_cache_threads = 0;
    bool modulation_cache_upcast = false;
    std::vector<std::pair<float, int>> modulation_request;
    std::function<bool()> modulation_reset_weights;
    std::function<bool()> modulation_cancelled;
    size_t modulation_budget = 128 * 1024 * 1024;
    size_t modulation_refills = 0;
    ggml_backend_dev_t blas_cpu_fallback = nullptr;
    int blas_threads = 0;
    std::map<std::string, int> last_matrix_backends;

    // Diagnostic-only CPU software BLAS; public GPU and precision guards remain unchanged.
    bool enable_blas_cpu_fallback(int threads) {
        auto device = ggml_backend_get_device(runtime_backend);
        auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        auto setter = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_set_n_threads"));
        if (std::string(ggml_backend_name(runtime_backend)) != "BLAS" || !cpu || !setter ||
            threads < 1 || !full_precision_matrix_compute) {
            LOG_ERROR("Diagnostic BLAS requires a CPU fallback, explicit F32 matrix computation and a thread setter");
            return false;
        }
        setter(runtime_backend, threads);
        blas_cpu_fallback = cpu;
        blas_threads = threads;
        return true;
    }

    bool supports_operation(ggml_tensor* node) const {
        return ggml_backend_supports_op(runtime_backend, node) ||
               (blas_cpu_fallback && ggml_backend_dev_supports_op(blas_cpu_fallback, node));
    }

    bool valid_blas_configuration(int threads) const {
        if (blas_cpu_fallback && (!full_precision_matrix_compute || threads != blas_threads)) {
            LOG_ERROR("Diagnostic BLAS arithmetic or thread count changed; reconfigure explicitly");
            return false;
        }
        return true;
    }

    size_t modulation_cache_bytes() const {
        size_t bytes = 0;
        for (const auto& item : modulation_cache)
            bytes += item.second.numel() * sizeof(float);
        return bytes;
    }

    size_t inactive_modulation_parameter_bytes() const {
        return modulation_cache_complete ? modulation_inactive_bytes : 0;
    }

    size_t calculate_modulation_parameter_bytes() {
        std::map<std::string, ggml_tensor*> tensors;
        get_param_tensors(tensors);
        size_t bytes = 0;
        for (const auto& item : tensors) {
            const auto& name = item.first;
            if (name.find(".modulation.") != std::string::npos || name.find(".img_mod.") != std::string::npos ||
                name.find(".txt_mod.") != std::string::npos || name.find(".adaLN_modulation.") != std::string::npos ||
                name.find(".t_embedder.") != std::string::npos || name.find(".y_embedder.") != std::string::npos)
                bytes += ggml_nbytes(item.second);
        }
        return bytes;
    }

    // The scope must finish runtime work, release storage, and re-register the same unmapped parameters.
    bool prepare_modulations(int threads, const std::vector<std::pair<float, int>>& pairs,
                             const std::function<bool()>& reset_weights,
                             size_t max_bytes = 128 * 1024 * 1024,
                             const std::function<bool()>& cancelled = {}) {
        if (!reset_weights || threads < 1 || pairs.empty() || weight_adapter || !valid_blas_configuration(threads) ||
            (!blas_cpu_fallback && ggml_backend_dev_type(ggml_backend_get_device(runtime_backend)) != GGML_BACKEND_DEVICE_TYPE_CPU)) {
            LOG_ERROR("Modulation precomputation requires CPU, no weight adapter, valid pairs and a weight-release scope");
            return false;
        }
        std::map<std::pair<uint32_t, int>, float> unique;
        for (const auto& pair : pairs) {
            if (!std::isfinite(pair.first) || pair.first < 0.f || pair.first > 1.f || pair.second < 0 || pair.second > 3) {
                LOG_ERROR("Invalid modulation time/category pair");
                return false;
            }
            unique.emplace(fashn_modulation_key(pair.first, pair.second), pair.first);
        }
        if (unique.size() > max_bytes / (fashn_modulation_values() * sizeof(float))) {
            LOG_ERROR("Modulation schedule exceeds its bounded cache budget; use the ordinary uncached path");
            return false;
        }
        if (cancelled && cancelled()) {
            LOG_INFO("Modulation precomputation cancelled");
            return false;
        }
        bool hit = modulation_cache_complete && modulation_cache_threads == threads &&
                   modulation_cache_upcast == full_precision_matrix_compute && modulation_cache_bytes() <= max_bytes;
        for (const auto& pair : unique)
            hit = hit && modulation_cache.count(pair.first);
        if (hit)
            return true;

        auto observer = std::move(diagnostic_observer);
        struct RestoreObserver {
            FashnVTONRunner& runner;
            decltype(observer)& saved;
            ~RestoreObserver() { runner.diagnostic_observer = std::move(saved); }
        } restore{*this, observer};
        auto notify = [&](const char* phase) {
            if (observer)
                observer(phase, nullptr, threads);
        };
        modulation_cache_required = true;
        modulation_cache_complete = false;
        modulation_cache.clear();
        notify("modulation_precompute_begin");
        if (!reset_weights()) {
            LOG_ERROR("Could not reset modulation parameter storage");
            return false;
        }
        for (const auto& pair : unique)
            modulation_cache.emplace(pair.first, sd::Tensor<float>({static_cast<int64_t>(fashn_modulation_values())}));
        notify("modulation_cache_allocated");
        bool success = true;
        size_t offset = 0;
        for (int layer = -1; layer < static_cast<int>(fashn_modulation_layers().size()) && success; ++layer) {
            const size_t width = layer < 0 ? 3840 : fashn_modulation_layers()[layer].width;
            for (const auto& pair : unique) {
                if (cancelled && cancelled()) {
                    LOG_INFO("Modulation precomputation cancelled");
                    success = false;
                    break;
                }
                auto& entry = modulation_cache.at(pair.first);
                sd::Tensor<float> time({1}, {pair.second});
                sd::Tensor<int32_t> category({1}, {pair.first.second});
                sd::Tensor<float> vec;
                if (layer >= 0)
                    vec = sd::Tensor<float>({1280, 1}, std::vector<float>(entry.data() + 2560, entry.data() + 3840));
                auto graph = [&]() {
                    auto gf = new_graph_custom(1024);
                    auto ctx = get_context();
                    ctx.debug_tensors = nullptr;
                    ctx.full_precision_linear_weights = full_precision_matrix_compute;
                    ggml_tensor* result;
                    if (layer < 0) {
                        auto values = model.conditioning(&ctx, make_input(time), make_input(category));
                        result = ggml_concat(ctx.ggml_ctx, ggml_concat(ctx.ggml_ctx, values[0], values[1], 0), values[2], 0);
                    } else {
                        result = model.project_modulation(&ctx, make_input(vec), layer);
                    }
                    ggml_build_forward_expand(gf, result);
                    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                        auto node = ggml_graph_node(gf, i);
                        if (node->op == GGML_OP_MUL_MAT && full_precision_matrix_compute)
                            ggml_mul_mat_set_prec(node, GGML_PREC_F32);
                        if (!supports_operation(node)) {
                            LOG_ERROR("Unsupported modulation operation: %s", ggml_op_name(node->op));
                            return static_cast<ggml_cgraph*>(nullptr);
                        }
                    }
                    return gf;
                };
                auto result = take_or_empty(GGMLRunner::compute(graph, threads, false, false));
                if (result.numel() != static_cast<int64_t>(width) ||
                    !std::all_of(result.values().begin(), result.values().end(), [](float v) { return std::isfinite(v); })) {
                    LOG_ERROR("Incomplete or non-finite modulation precomputation");
                    success = false;
                    break;
                }
                std::copy(result.values().begin(), result.values().end(), entry.values().begin() + offset);
            }
            success = reset_weights() && success;
            notify("modulation_layer_released");
            offset += width;
        }
        if (!success) {
            modulation_cache.clear();
            return false;
        }
        modulation_cache_threads = threads;
        modulation_cache_upcast = full_precision_matrix_compute;
        modulation_inactive_bytes = calculate_modulation_parameter_bytes();
        modulation_cache_complete = true;
        ++modulation_refills;
        notify("modulation_precompute_end");
        return true;
    }

    bool refill_modulations(int threads, size_t index) {
        const size_t capacity = modulation_budget / (fashn_modulation_values() * sizeof(float));
        if (!capacity || index >= modulation_request.size() || !modulation_reset_weights) {
            LOG_ERROR("Invalid modulation request/window");
            return false;
        }
        const size_t begin = index / capacity * capacity;
        const size_t end = std::min(modulation_request.size(), begin + capacity);
        std::vector<std::pair<float, int>> window(modulation_request.begin() + begin, modulation_request.begin() + end);
        return prepare_modulations(threads, window,
                                   modulation_reset_weights, modulation_budget, modulation_cancelled);
    }

    bool prepare_modulation_request(int threads, const std::vector<std::pair<float, int>>& pairs,
                                     std::function<bool()> reset_weights, size_t max_bytes = 128 * 1024 * 1024,
                                     std::function<bool()> cancelled = {}) {
        if (threads < 1 || pairs.empty() || !reset_weights ||
            max_bytes < fashn_modulation_values() * sizeof(float)) {
            LOG_ERROR("Invalid modulation request or insufficient cache budget");
            return false;
        }
        std::vector<std::pair<float, int>> unique;
        std::set<std::pair<uint32_t, int>> keys;
        for (const auto& pair : pairs) {
            if (!std::isfinite(pair.first) || pair.first < 0.f || pair.first > 1.f || pair.second < 0 || pair.second > 3) {
                LOG_ERROR("Invalid modulation request time/category");
                return false;
            }
            if (keys.insert(fashn_modulation_key(pair.first, pair.second)).second)
                unique.push_back(pair);
        }
        modulation_request = std::move(unique);
        modulation_reset_weights = std::move(reset_weights);
        modulation_cancelled = std::move(cancelled);
        modulation_budget = max_bytes;
        return refill_modulations(threads, 0);
    }

    void finish_modulation_request() {
        modulation_request.clear();
        modulation_reset_weights = {};
        modulation_cancelled = {};
    }

    static FashnVTONConfig validated_config(const String2TensorStorage& tensors, const std::string& prefix) {
        auto config = FashnVTONConfig::detect_from_weights(tensors, prefix);
        std::string error;
        if (!config.validate_v15(tensors, prefix, &error)) {
            throw std::invalid_argument(error);
        }
        return config;
    }

    FashnVTONRunner(ggml_backend_t backend, const String2TensorStorage& tensors, const std::string& prefix, std::shared_ptr<RunnerWeightManager> manager)
        : DiffusionModelRunner(backend, prefix, manager),
          config(validated_config(tensors, prefix)),
          model(config),
          positions({2, 2, 64, 3456}) {
        model.init(params_ctx, tensors, prefix);
        size_t offset = 0;
        for (int row = 0; row < 72; ++row) {
            for (int col = 0; col < 48; ++col) {
                const int coordinates[] = {0, row, col};
                for (int axis = 0; axis < 3; ++axis) {
                    for (int j = 0; j < config.axes_dim[axis]; j += 2) {
                        double angle                 = coordinates[axis] / std::pow(10000.0, static_cast<double>(j) / config.axes_dim[axis]);
                        float cosine                 = static_cast<float>(std::cos(angle));
                        float sine                   = static_cast<float>(std::sin(angle));
                        positions.values()[offset++] = cosine;
                        positions.values()[offset++] = -sine;
                        positions.values()[offset++] = sine;
                        positions.values()[offset++] = cosine;
                    }
                }
            }
        }
    }

    using DiffusionModelRunner::get_param_tensors;

    std::string get_desc() override { return "fashn_vton"; }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
        model.get_param_tensors(tensors, prefix);
    }

    static bool valid_image(const sd::Tensor<float>* tensor, int channels, bool normalized) {
        if (tensor == nullptr || tensor->shape() != std::vector<int64_t>({576, 864, channels, 1})) {
            return false;
        }
        for (float value : tensor->values()) {
            if (!std::isfinite(value) || (normalized && (value < -1.f || value > 1.f))) {
                return false;
            }
        }
        return true;
    }

    sd::Tensor<float> compute(int threads, const DiffusionParams& params) override {
        return compute_with_capture(threads, params, nullptr);
    }

    sd::Tensor<float> compute_with_capture(int threads, const DiffusionParams& params, std::map<std::string, sd::Tensor<float>>* captures) {
        if (!valid_blas_configuration(threads))
            return {};
        const auto* extra = std::get_if<FashnVTONDiffusionExtra>(&params.extra);
        if (extra == nullptr || !valid_image(params.x, 3, false) ||
            !valid_image(extra->ca_images, 3, true) || !valid_image(extra->garment_images, 3, true) ||
            !valid_image(extra->person_poses, 1, true) || !valid_image(extra->garment_poses, 1, true) ||
            params.timesteps == nullptr || params.timesteps->shape() != std::vector<int64_t>({1}) ||
            !std::isfinite(params.timesteps->values()[0]) || params.timesteps->values()[0] < 0.f ||
            params.timesteps->values()[0] > 1.f ||
            extra->categories == nullptr || extra->categories->shape() != std::vector<int64_t>({1}) ||
            extra->categories->values()[0] < 0 || extra->categories->values()[0] > 3 || threads < 1) {
            LOG_ERROR("FASHN requires finite B=1, 576x864 RGB/pose tensors, time in [0,1], and category in [0,3]");
            return {};
        }
        if (captures != nullptr) {
            captures->clear();
        }
        const sd::Tensor<float>* cached_values = nullptr;
        if (modulation_cache_required) {
            const auto key = fashn_modulation_key(params.timesteps->values()[0], extra->categories->values()[0]);
            if (weight_adapter || threads != modulation_cache_threads || full_precision_matrix_compute != modulation_cache_upcast) {
                LOG_ERROR("Changed modulation arithmetic, adapter or thread configuration");
                return {};
            }
            if (!modulation_cache.count(key) && !modulation_request.empty()) {
                auto found = std::find_if(modulation_request.begin(), modulation_request.end(),
                                          [&](const auto& pair) { return fashn_modulation_key(pair.first, pair.second) == key; });
                if (found == modulation_request.end() || !refill_modulations(threads, found - modulation_request.begin())) {
                    LOG_ERROR("Could not prepare requested modulation window");
                    return {};
                }
            }
            auto found = modulation_cache.find(key);
            if (!modulation_cache_complete || found == modulation_cache.end()) {
                LOG_ERROR("Missing modulation cache key or changed arithmetic/threads; precompute the requested schedule first");
                return {};
            }
            cached_values = &found->second;
        }
        std::vector<ggml_tensor*> matrix_nodes;
        auto graph = [&]() -> ggml_cgraph* {
            auto gf                           = new_graph_custom(32768);
            auto ctx                          = get_context();
            ctx.full_precision_gelu           = true;
            ctx.fused_full_precision_gelu      = fused_gelu;
            ctx.full_precision_linear_weights = full_precision_matrix_compute;
            ctx.flash_attn_f32_kv             = true;
            ctx.attention_fallback_device     = blas_cpu_fallback;
            if (captures == nullptr) {
                ctx.debug_tensors = nullptr;
            }
            FashnCachedConditioning cached{};
            if (cached_values)
                cached = model.cached_conditioning(ctx.ggml_ctx, make_input(*cached_values));
            auto out = model.forward(&ctx, make_input(*params.x), make_input(*params.timesteps),
                                     make_input(*extra->ca_images), make_input(*extra->garment_images),
                                     make_input(*extra->person_poses), make_input(*extra->garment_poses),
                                     make_input(*extra->categories), make_input(positions), cached_values ? &cached : nullptr);
            ggml_build_forward_expand(gf, out);
            last_flash_attention_nodes = 0;
            last_supported_nodes       = 0;
            last_f32_matmuls           = 0;
            last_unsupported_operation.clear();
            matrix_nodes.clear();
            last_matrix_backends.clear();
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                auto node = ggml_graph_node(gf, i);
                if (node->op == GGML_OP_MUL_MAT)
                    matrix_nodes.push_back(node);
                last_flash_attention_nodes += node->op == GGML_OP_FLASH_ATTN_EXT;
                if (full_precision_matrix_compute && node->op == GGML_OP_MUL_MAT &&
                    (node->src[0]->type != GGML_TYPE_F32 || node->src[1]->type != GGML_TYPE_F32)) {
                    LOG_ERROR("FASHN full-precision matrix execution requires F32 operands");
                    return nullptr;
                }
                if (full_precision_matrix_compute && node->op == GGML_OP_MUL_MAT) {
                    ggml_mul_mat_set_prec(node, GGML_PREC_F32);
                    ++last_f32_matmuls;
                }
                if (!supports_operation(node)) {
                    last_unsupported_operation = std::string(ggml_op_name(node->op)) + ": " + node->name;
                    LOG_ERROR("FASHN backend %s cannot execute %s",
                              ggml_backend_name(ctx.backend), last_unsupported_operation.c_str());
                    return nullptr;
                }
                ++last_supported_nodes;
            }
            if (ctx.flash_attn_enabled && last_flash_attention_nodes != 28) {
                last_unsupported_operation = "Expected 28 F32 flash operations; found " + std::to_string(last_flash_attention_nodes);
                LOG_ERROR("FASHN requires all 28 F32 K/V flash attention operations; backend lacks support");
                return nullptr;
            }
            return gf;
        };
        auto read = [&]() {
            for (auto node : matrix_nodes) {
                auto backend = workspace_.scheduler() ?
                    ggml_backend_sched_get_tensor_backend(workspace_.scheduler(), node) : runtime_backend;
                if (!backend) {
                    LOG_ERROR("Missing matrix backend assignment");
                    return false;
                }
                ++last_matrix_backends[ggml_backend_name(backend)];
            }
            if (captures != nullptr) {
                for (const auto& [tensor, name] : debug_tensors) {
                    auto value = read_graph_tensor(tensor, name.c_str());
                    if (!value.has_value()) {
                        return false;
                    }
                    captures->emplace(name, std::move(*value));
                }
            }
            return true;
        };
        return restore_trailing_singleton_dims(GGMLRunner::compute(graph, threads, false, false, read), 4);
    }
};

#endif  // __SD_MODEL_DIFFUSION_FASHN_VTON_MODEL_H__
