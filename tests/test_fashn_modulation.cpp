#include <limits>
#include "fashn_test_runner.h"
#include "runtime/fashn_vton_sampling.h"
#include "stable-diffusion.h"

struct ModulationReferenceRunner : FashnVTONRunner {
    using FashnVTONRunner::FashnVTONRunner;

    sd::Tensor<float> direct(float t, int category_value, int layer) {
        sd::Tensor<float> time({1}, {t});
        sd::Tensor<int32_t> category({1}, {category_value});
        auto graph = [&]() {
            auto gf = new_graph_custom(1024);
            auto ctx = get_context();
            ctx.debug_tensors = nullptr;
            ctx.full_precision_linear_weights = true;
            auto values = model.conditioning(&ctx, make_input(time), make_input(category));
            auto out = layer < 0 ? ggml_concat(ctx.ggml_ctx, ggml_concat(ctx.ggml_ctx, values[0], values[1], 0), values[2], 0) :
                                   model.project_modulation(&ctx, values[2], layer);
            ggml_build_forward_expand(gf, out);
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                auto node = ggml_graph_node(gf, i);
                if (node->op == GGML_OP_MUL_MAT)
                    ggml_mul_mat_set_prec(node, GGML_PREC_F32);
            }
            return gf;
        };
        return take_or_empty(GGMLRunner::compute(graph, 2, false));
    }
};

int main(int argc, char** argv) {
    if (argc > 2)
        return 2;
    if (fashn_modulation_layers().size() != 37 || fashn_modulation_values() != 206080 ||
        fashn_modulation_key(0.f, 0) == fashn_modulation_key(-0.f, 0) ||
        fashn_modulation_key(.5f, 0) == fashn_modulation_key(.5f, 2))
        return 1;
    auto c = ggml_init({4 * 1024 * 1024, nullptr, false});
    if (!c)
        return 1;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(c, ggml_free);
    auto data = ggml_new_tensor_1d(c, GGML_TYPE_F32, fashn_modulation_values());
    for (size_t i = 0; i < fashn_modulation_values(); ++i)
        ggml_get_data_f32(data)[i] = static_cast<float>(i);
    auto cached = FashnVTONModel::cached_conditioning(c, data);
    if (ggml_get_data_f32(cached.time)[0] != 0 || ggml_get_data_f32(cached.category)[0] != 1280 ||
        ggml_get_data_f32(cached.vec)[0] != 2560 || cached.layers.size() != 37)
        return 1;
    GGMLRunnerContext graph_context{};
    graph_context.ggml_ctx = c;
    size_t offset = 3840;
    for (size_t i = 0; i < cached.layers.size(); ++i) {
        const auto width = fashn_modulation_layers()[i].width;
        if (cached.layers[i]->ne[0] != width || ggml_get_data_f32(cached.layers[i])[0] != offset)
            return 1;
        if (width != 2560) {
            auto mods = FashnVTONModel::supplied_mods(&graph_context, cached.layers[i]);
            if (mods.size() != (width == 7680 ? 2 : 1))
                return 1;
            for (size_t part = 0; part < mods.size(); ++part) {
                if (ggml_get_data_f32(mods[part].shift)[17] != offset + part * 3840 + 17 ||
                    ggml_get_data_f32(mods[part].scale)[17] != offset + part * 3840 + 1297 ||
                    ggml_get_data_f32(mods[part].gate)[17] != offset + part * 3840 + 2577)
                    return 1;
            }
        }
        offset += width;
    }
    if (offset != fashn_modulation_values())
        return 1;

    String2TensorStorage tensors;
    for (const auto& [name, shape] : FashnVTONConfig::expected_shapes()) {
        TensorStorage tensor;
        tensor.n_dims = static_cast<int>(shape.size());
        tensor.type = GGML_TYPE_BF16;
        for (size_t i = 0; i < shape.size(); ++i)
            tensor.ne[i] = shape[i];
        tensors[FashnVTONConfig::canonical_name(name)] = tensor;
    }
    std::map<std::string, std::string> types;
    if (!assign_fashn_test_matrix_types(tensors, GGML_TYPE_BF16, {}, types))
        return 1;
    auto backend = initialize_backend("CPU");
    if (!backend)
        return 1;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend_guard(backend, ggml_backend_free);
    FashnVTONRunner runner(backend, tensors, "model.diffusion_model", nullptr);
    bool reset_called = false;
    auto reset = [&]() { reset_called = true; return false; };
    std::vector<std::pair<float, int>> long_schedule;
    for (int i = 0; i < 1000; ++i) {
        long_schedule.emplace_back(i / 1000.f, 2);
        long_schedule.emplace_back(i / 1000.f, 0);
    }
    if (runner.prepare_modulations(2, long_schedule, reset) ||
        runner.prepare_modulations(2, {{0.f, 2}}, reset, 1) ||
        runner.prepare_modulations(2, {{std::numeric_limits<float>::quiet_NaN(), 2}}, reset) ||
        reset_called || runner.modulation_cache_bytes() != 0)
        return 1;
    if (runner.prepare_modulation_request(2, long_schedule, reset, 1024 * 1024) ||
        !reset_called || runner.modulation_request.size() != 2000 || runner.modulation_cache_bytes() != 0)
        return 1;
    runner.finish_modulation_request();
    reset_called = false;
    if (runner.prepare_modulation_request(2, {{0.f, 2}}, reset, 1024 * 1024, [] { return true; }) || reset_called)
        return 1;
    for (int steps : {20, 30, 50, 1000}) {
        auto pairs = fashn_vton_modulation_pairs({steps, 1.5f, 1.5f, 1}, 2);
        if (pairs.size() != static_cast<size_t>(2 * steps - 1) ||
            pairs.front() != std::make_pair(0.f, 2) || pairs[1] != std::make_pair(0.f, 0) || pairs.back().second != 2)
            return 1;
    }
    if (fashn_vton_modulation_pairs({20, 1.f, 1.5f, 0}, 2).size() != 20)
        return 1;
    if (argc == 2) {
        FashnGraphTestContext real("CPU");
        if (!real.init(argv[1], true, GGML_TYPE_BF16, true, 2, {}, 2, false))
            return 1;
        int polls = 0;
        if (real.prepare_modulations(2, {{0.f, 2}, {0.f, 0}}, 2 * 1024 * 1024, [&] { return ++polls >= 4; }) ||
            polls < 4 || real.runner->modulation_cache_bytes() != 0 ||
            real.manager->params_memory_snapshot().assigned_bytes != 0)
            return 1;
        if (!real.prepare_modulations(2, {{0.f, 2}}, 1024 * 1024) ||
            real.manager->params_memory_snapshot().assigned_bytes != 0 ||
            real.runner->inactive_modulation_parameter_bytes() < 1000000000)
            return 1;
        auto refills = real.runner->modulation_refills;
        if (!real.prepare_modulations(2, {{0.f, 2}}, 1024 * 1024) || real.runner->modulation_refills != refills)
            return 1;
        real.runner->finish_modulation_request();
        if (real.runner->modulation_cancelled || real.runner->modulation_reset_weights)
            return 1;
        real.runner->runner_end();
        if (!real.manager->unregister_param_tensors("fashn"))
            return 1;
        real.runner.reset();
        auto probe = std::make_unique<ModulationReferenceRunner>(real.backend, real.manager->loader().get_tensor_storage_map(),
                                                                "model.diffusion_model", real.manager);
        auto* reference = probe.get();
        real.runner = std::move(probe);
        auto register_params = [&]() {
            return real.manager->register_runner_params("fashn", *real.runner, ModelManager::ResidencyMode::ParamBackend,
                                                         real.backend, real.backend) &&
                   real.manager->validate_registered_tensors();
        };
        if (!register_params())
            return 1;
        std::vector<std::pair<float, int>> pairs;
        for (float t : {0.f, .4f, .8f})
            for (int category = 0; category < 4; ++category)
                pairs.emplace_back(t, category);
        if (!real.prepare_modulations(2, pairs))
            return 1;
        size_t position = 0;
        for (int layer = -1; layer < 37; ++layer) {
            const size_t width = layer < 0 ? 3840 : fashn_modulation_layers()[layer].width;
            for (auto pair : pairs) {
                auto original = reference->direct(pair.first, pair.second, layer);
                const auto& cached = real.runner->modulation_cache.at(fashn_modulation_key(pair.first, pair.second));
                if (original.numel() != static_cast<int64_t>(width) ||
                    std::memcmp(original.data(), cached.data() + position, width * sizeof(float)) != 0)
                    return 1;
            }
            real.runner->runner_end();
            if (!real.manager->unregister_param_tensors("fashn") || !register_params())
                return 1;
            position += width;
        }
        std::cout << "All 37 projections and conditioning captures match exactly at three times and four categories\n";
        for (const char* arguments : {"{}", "{\"fashn_modulation_cache\":\"true\"}",
             "{\"fashn_modulation_cache\":true,\"fashn_modulation_cache_mib\":0}",
             "{\"fashn_modulation_cache\":true,\"fashn_modulation_cache_mib\":129}",
             "{\"fashn_modulation_cache\":true,\"fashn_fused_gelu\":\"true\"}",
             "{\"fashn_modulation_cache\":true,\"unknown\":1}"}) {
            sd_ctx_params_t options;
            sd_ctx_params_init(&options);
            options.diffusion_model_path = argv[1];
            options.model_args = arguments;
            auto invalid = new_sd_ctx(&options);
            if (invalid) {
                free_sd_ctx(invalid);
                return 1;
            }
        }
        for (int conflict = 0; conflict < 3; ++conflict) {
            sd_ctx_params_t options;
            sd_ctx_params_init(&options);
            options.diffusion_model_path = argv[1];
            options.model_args = "{\"fashn_modulation_cache\":true,\"fashn_modulation_cache_mib\":1,\"fashn_fused_gelu\":true}";
            options.enable_mmap = conflict == 1;
            options.eager_load = conflict == 2;
            options.n_threads = 2;
            options.wtype = SD_TYPE_BF16;
            auto result = new_sd_ctx(&options);
            bool correct = (result != nullptr) == (conflict == 0);
            if (result)
                free_sd_ctx(result);
            if (!correct)
                return 1;
        }
    }
    std::cout << "Modulation layout, keys, bounded requests, cancellation and reuse: passed\n";
    return 0;
}
