#include <chrono>
#include "fashn_test_runner.h"

struct StrictGELURunner : GGMLRunner {
    bool fused;
    int layout;
    explicit StrictGELURunner(ggml_backend_t backend, bool fused, int layout)
        : GGMLRunner(backend), fused(fused), layout(layout) {}
    std::string get_desc() override { return "strict_gelu"; }
    sd::Tensor<float> run(const sd::Tensor<float>& input, int threads) {
        auto graph = [&]() {
            auto gf = new_graph_custom(64);
            auto ctx = get_context();
            ctx.full_precision_gelu = true;
            ctx.fused_full_precision_gelu = fused;
            auto x = make_input(input);
            if (layout == 1)
                x = ggml_view_2d(ctx.ggml_ctx, x, 5120, x->ne[1], x->nb[1], 3840 * sizeof(float));
            else if (layout == 2)
                x = ggml_transpose(ctx.ggml_ctx, x);
            else if (layout == 3)
                x = ggml_permute(ctx.ggml_ctx, x, 1, 2, 0, 3);
            if (!fused && layout >= 2)
                x = ggml_cont(ctx.ggml_ctx, x);
            auto out = Flux::gelu(&ctx, x);
            GGML_ASSERT((out->op == GGML_OP_MAP_CUSTOM1) == fused);
            ggml_build_forward_expand(gf, out);
            return gf;
        };
        return take_or_empty(GGMLRunner::compute(graph, threads, false));
    }
};

static sd::Tensor<float> gelu_input(const std::vector<int64_t>& shape) {
    sd::Tensor<float> data(shape);
    for (int64_t i = 0; i < data.numel(); ++i)
        data.values()[i] = static_cast<float>(static_cast<int>((i * 7919) % 24001) - 12000) / 1000.f;
    return data;
}

int main(int argc, char** argv) {
    const bool benchmark = argc == 2 && std::string(argv[1]) == "--benchmark";
    if (argc > 1 && !benchmark)
        return 2;
    auto backend = initialize_backend("CPU");
    if (!backend)
        return 1;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> guard(backend, ggml_backend_free);
    const std::vector<std::vector<int64_t>> shapes{{24001}, {8960, 7}, {17, 13}, {7, 5, 3, 2}};
    for (int layout = 0; layout < 4; ++layout) {
        auto input = gelu_input(shapes[layout]);
        for (int threads : {1, 16}) {
            StrictGELURunner original(backend, false, layout), fused(backend, true, layout);
            auto expected = original.run(input, threads), actual = fused.run(input, threads);
            if (actual.shape() != expected.shape() || actual.empty())
                return 1;
            for (int64_t i = 0; i < actual.numel(); ++i) {
                if (!std::isfinite(actual.values()[i]) || std::abs(actual.values()[i] - expected.values()[i]) > 2e-6f)
                    return 1;
                if (layout == 0) {
                    const double x = input.values()[i];
                    const double analytic = .5 * x * (1 + std::tanh(.7978845608028654 * (x + .044715 * x * x * x)));
                    if (std::abs(actual.values()[i] - analytic) > 2e-6)
                        return 1;
                }
            }
        }
    }
    std::cout << "Strict GELU analytic parity, row gaps, transposes, 4D strides and thread counts: passed\n";
    if (benchmark) {
        nlohmann::ordered_json report;
        for (int layout : {0, 1}) {
            for (float radius : {1.f, 12.f}) {
                auto input = gelu_input(layout == 0 ? std::vector<int64_t>{5120, 3456} : std::vector<int64_t>{8960, 6912});
                for (auto& value : input.values())
                    value *= radius / 12.f;
                for (bool fused : {false, true}) {
                    StrictGELURunner runner(backend, fused, layout);
                    std::vector<double> durations;
                    for (int repeat = 0; repeat < 4; ++repeat) {
                        const auto start = std::chrono::steady_clock::now();
                        auto result = runner.run(input, 16);
                        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                        if (result.empty())
                            return 1;
                        durations.push_back(seconds);
                    }
                    report.push_back({{"layout", layout == 0 ? "double_mlp_dense" : "single_mlp_qkv_view"},
                                      {"uniform_input_radius", radius},
                                      {"fused", fused}, {"threads", 16}, {"seconds", durations},
                                      {"arena_bytes", runner.diagnostic_runtime_bytes()},
                                      {"note", "Runner wall includes graph construction and output copy; first run is cold."}});
                }
            }
        }
        std::cout << report.dump(2) << "\n";
    }
    return 0;
}
