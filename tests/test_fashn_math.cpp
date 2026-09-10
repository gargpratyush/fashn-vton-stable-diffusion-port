#include <chrono>
#include <filesystem>
#include "fashn_test_runner.h"

struct FashnMatmulProbe : GGMLRunner {
    bool primary_supported = false;
    explicit FashnMatmulProbe(ggml_backend_t backend) : GGMLRunner(backend) {}
    std::string get_desc() override { return "fashn_matmul_probe"; }
    sd::Tensor<float> run(const sd::Tensor<float>& weight, const sd::Tensor<float>& input, int threads) {
        auto device = ggml_backend_get_device(runtime_backend);
        auto setter = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_set_n_threads"));
        if (!setter) {
            std::cerr << "Backend has no supported thread-count setter\n";
            return {};
        }
        setter(runtime_backend, threads);
        auto graph = [&]() {
            auto gf = new_graph_custom(32);
            auto out = ggml_mul_mat(compute_ctx, make_input(weight), make_input(input));
            ggml_mul_mat_set_prec(out, GGML_PREC_F32);
            primary_supported = ggml_backend_supports_op(runtime_backend, out);
            ggml_build_forward_expand(gf, out);
            return gf;
        };
        return take_or_empty(GGMLRunner::compute(graph, threads, false));
    }
};

static sd::Tensor<float> math_values(int64_t k, int64_t columns, uint32_t seed, float scale) {
    sd::Tensor<float> result({k, columns});
    for (int64_t i = 0; i < result.numel(); ++i) {
        uint32_t value = static_cast<uint32_t>(i) * 2654435761U + seed;
        value ^= value >> 16;
        result.values()[i] = (static_cast<float>(value & 65535) - 32768.f) * (scale / 32768.f);
    }
    return result;
}

struct FashnAttentionProbe : GGMLRunner {
    std::map<std::string, std::vector<size_t>> flash_strides;
    explicit FashnAttentionProbe(ggml_backend_t backend) : GGMLRunner(backend) {}
    std::string get_desc() override { return "fashn_attention_probe"; }
    sd::Tensor<float> run(const sd::Tensor<float>& q, const sd::Tensor<float>& k, const sd::Tensor<float>& v, int threads) {
        auto graph = [&]() -> ggml_cgraph* {
            auto gf = new_graph_custom(128);
            auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            auto out = ggml_ext_attention_ext(compute_ctx, runtime_backend, make_input(q), make_input(k), make_input(v),
                                              10, nullptr, false, true, 1.f, true, cpu);
            ggml_build_forward_expand(gf, out);
            int flashes = 0;
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                auto node = ggml_graph_node(gf, i);
                if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                    ++flashes;
                    for (int source = 0; source < 3; ++source) {
                        if (node->src[source]->type != GGML_TYPE_F32) {
                            std::cerr << "Expected strict F32 attention operands\n";
                            return nullptr;
                        }
                        flash_strides[std::string(1, "qkv"[source])] =
                            std::vector<size_t>(node->src[source]->nb, node->src[source]->nb + 4);
                    }
                }
            }
            if (flashes != 1) {
                std::cerr << "Expected exactly one strict F32 flash attention operation\n";
                return nullptr;
            }
            return gf;
        };
        return take_or_empty(GGMLRunner::compute(graph, threads, false));
    }
};

static int benchmark_cfg_matrices(ggml_backend_t backend) {
    const size_t budget = size_t(1024) * 1024 * 1024;
    nlohmann::ordered_json report = nlohmann::ordered_json::array();
    for (auto shape : {std::array<int, 3>{1280, 5120, 3456}, {1280, 8960, 6912}, {6400, 1280, 6912}}) {
        const auto [k, m, n] = shape;
        auto weight = math_values(k, m, 101, .02f);
        auto first = math_values(k, n, 203, 1.f), second = math_values(k, n, 307, 1.f);
        sd::Tensor<float> pair({k, n, 2});
        std::copy(first.values().begin(), first.values().end(), pair.values().begin());
        std::copy(second.values().begin(), second.values().end(), pair.values().begin() + first.numel());
        auto flat = pair.reshape({k, n * 2});
        sd::Tensor<float> expected_first, expected_second;
        for (int mode : {0, 1, 2, 0}) {
            FashnMatmulProbe probe(backend);
            std::vector<double> seconds;
            sd::Tensor<float> a, b;
            for (int repeat = 0; repeat < 4; ++repeat) {
                auto start = std::chrono::steady_clock::now();
                if (mode == 0) {
                    a = probe.run(weight, first, 16);
                    b = probe.run(weight, second, 16);
                } else {
                    a = probe.run(weight, mode == 1 ? pair : flat, 16);
                }
                seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
                if (a.numel() != int64_t(m) * n * (mode == 0 ? 1 : 2) ||
                    (mode == 0 && b.numel() != a.numel()) || probe.diagnostic_runtime_bytes() > budget) {
                    std::cerr << "CFG matrix probe failed its output or 1 GiB workspace budget\n";
                    return 1;
                }
            }
            if (expected_first.empty()) {
                expected_first = a;
                expected_second = b;
            }
            double error = 0, norm = 0;
            for (int branch = 0; branch < 2; ++branch) {
                const auto& expected = branch == 0 ? expected_first : expected_second;
                const float* actual = mode == 0 ? (branch == 0 ? a.data() : b.data()) :
                                                 a.data() + branch * expected.numel();
                for (int64_t i = 0; i < expected.numel(); ++i) {
                    const double delta = double(actual[i]) - expected.values()[i];
                    error += delta * delta;
                    norm += double(expected.values()[i]) * expected.values()[i];
                }
            }
            const double relative = std::sqrt(error / std::max(norm, 1e-30));
            if (!std::isfinite(relative) || relative > 1e-3)
                return 1;
            report.push_back({{"k", k}, {"m", m}, {"n_per_branch", n}, {"backend", ggml_backend_name(backend)},
                              {"mode", mode == 0 ? "sequential" : mode == 1 ? "batch_dimension" : "flattened_tokens"},
                              {"seconds", seconds}, {"relative_l2", relative}, {"threads", 16},
                              {"workspace_bytes", probe.diagnostic_runtime_bytes()}, {"workspace_budget_bytes", budget},
                              {"note", "Synthetic independent branches; includes graph/input/output costs, not full CFG or process-memory acceptance."}});
        }
    }
    std::cout << report.dump(2) << "\n";
    return 0;
}

static int benchmark_cfg_attention(ggml_backend_t backend) {
    nlohmann::ordered_json report = nlohmann::ordered_json::array();
    for (int tokens : {3456, 6912}) {
        std::array<sd::Tensor<float>, 3> first, second, paired;
        for (int i = 0; i < 3; ++i) {
            first[i] = math_values(1280, tokens, 101 + i * 102, 1.f);
            second[i] = math_values(1280, tokens, 503 + i * 102, 1.f);
            paired[i] = sd::Tensor<float>({1280, tokens, 2});
            std::copy(first[i].values().begin(), first[i].values().end(), paired[i].values().begin());
            std::copy(second[i].values().begin(), second[i].values().end(), paired[i].values().begin() + first[i].numel());
        }
        sd::Tensor<float> expected_first, expected_second;
        for (bool batch : {false, true, false}) {
            FashnAttentionProbe probe(backend);
            sd::Tensor<float> a, b;
            std::vector<double> seconds;
            for (int repeat = 0; repeat < 4; ++repeat) {
                const auto start = std::chrono::steady_clock::now();
                a = batch ? probe.run(paired[0], paired[1], paired[2], 16) :
                            probe.run(first[0], first[1], first[2], 16);
                if (!batch)
                    b = probe.run(second[0], second[1], second[2], 16);
                seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
                if (a.numel() != int64_t(1280) * tokens * (batch ? 2 : 1) || (!batch && b.numel() != a.numel()))
                    return 1;
            }
            if (expected_first.empty()) {
                expected_first = a;
                expected_second = b;
            }
            double max_abs = 0;
            for (int branch = 0; branch < 2; ++branch) {
                const auto& expected = branch ? expected_second : expected_first;
                const float* actual = batch ? a.data() + branch * expected.numel() : (branch ? b.data() : a.data());
                for (int64_t i = 0; i < expected.numel(); ++i) {
                    if (!std::isfinite(actual[i]))
                        return 1;
                    max_abs = std::max(max_abs, std::abs(double(actual[i]) - expected.values()[i]));
                }
            }
            if (max_abs > 2e-5)
                return 1;
            report.push_back({{"tokens", tokens}, {"batch", batch}, {"seconds", seconds}, {"max_abs", max_abs},
                              {"workspace_bytes", probe.diagnostic_runtime_bytes()},
                              {"note", "Independent synthetic branches verify attention batch isolation, not full FASHN B=2 support."}});
        }
    }
    std::cout << report.dump(2) << "\n";
    return 0;
}

int main(int argc, char** argv) {
    const bool benchmark = argc == 2 && std::string(argv[1]) == "--benchmark";
    const bool check_blas = argc == 2 && std::string(argv[1]) == "--check-blas";
    const bool attention_benchmark = argc == 3 && std::string(argv[1]) == "--attention-benchmark";
    const bool cfg_benchmark = argc == 2 && std::string(argv[1]) == "--cfg-benchmark";
    const bool cfg_attention = argc == 2 && std::string(argv[1]) == "--cfg-attention-benchmark";
    if (argc > 1 && !benchmark && !check_blas && !attention_benchmark && !cfg_benchmark && !cfg_attention)
        return 2;
    auto cpu = initialize_backend("CPU");
    if (!cpu)
        return 1;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu_guard(cpu, ggml_backend_free);
    if (cfg_attention)
        return benchmark_cfg_attention(cpu);
    auto weights = math_values(32, 64, 101, .02f), input = math_values(32, 48, 203, 1.f);
    {
        FashnMatmulProbe probe(cpu);
        auto result = probe.run(weights, input, 2);
        if (result.numel() != 64 * 48)
            return 1;
        for (int n = 0; n < 48; ++n) {
            for (int m = 0; m < 64; ++m) {
                double expected = 0;
                for (int k = 0; k < 32; ++k)
                    expected += double(weights.values()[m * 32 + k]) * input.values()[n * 32 + k];
                if (!std::isfinite(result.values()[n * 64 + m]) ||
                    std::abs(result.values()[n * 64 + m] - expected) > 1e-6)
                    return 1;
            }
        }
    }
    if (attention_benchmark) {
        const std::filesystem::path directory(argv[2]);
        if (std::filesystem::exists(directory))
            return 2;
        std::filesystem::create_directories(directory);
        nlohmann::ordered_json report = nlohmann::ordered_json::array();
        for (int tokens : {3456, 6912}) {
            auto q = math_values(1280, tokens, 101, 1.f), k = math_values(1280, tokens, 203, 1.f);
            auto v = math_values(1280, tokens, 307, 1.f);
            FashnAttentionProbe probe(cpu);
            std::vector<double> seconds;
            sd::Tensor<float> result;
            for (int iteration = 0; iteration < 3; ++iteration) {
                auto start = std::chrono::steady_clock::now();
                result = probe.run(q, k, v, 16);
                seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
                if (result.empty())
                    return 1;
            }
            if (!save_capture((directory / ("attention-" + std::to_string(tokens) + ".safetensors")).string(), {{"attention", result}}))
                return 1;
            report.push_back({{"tokens", tokens}, {"heads", 10}, {"head_dim", 128}, {"threads", 16},
                              {"seconds", seconds}, {"flash_strides_bytes", probe.flash_strides},
                              {"workspace_bytes", probe.diagnostic_workspace_bytes()},
                              {"note", "Includes input copy/layout preparation, graph execution and output copy."}});
        }
        std::ofstream file(directory / "native.json");
        file << report.dump(2) << "\n";
        file.close();
        return file ? 0 : 1;
    }
    if (!benchmark && !check_blas && !cfg_benchmark) {
        std::cout << "F32 matmul layout and double-accumulated reference: passed\n";
        return 0;
    }
    auto blas = initialize_backend("BLAS");
    if (!blas)
        return 1;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> blas_guard(blas, ggml_backend_free);
    if (cfg_benchmark)
        return benchmark_cfg_matrices(blas);
    if (check_blas) {
        auto q = math_values(1280, 33, 101, 1.f), k = math_values(1280, 65, 203, 1.f);
        auto v = math_values(1280, 65, 307, 1.f);
        FashnAttentionProbe control(cpu), candidate(blas);
        auto expected = control.run(q, k, v, 2), actual = candidate.run(q, k, v, 2);
        if (actual.empty() || actual.shape() != expected.shape() || actual.values() != expected.values())
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
        FashnVTONRunner guarded(blas, tensors, "model.diffusion_model", nullptr);
        guarded.full_precision_matrix_compute = false;
        if (guarded.enable_blas_cpu_fallback(2))
            return 1;
        guarded.full_precision_matrix_compute = true;
        if (!guarded.enable_blas_cpu_fallback(2) || guarded.valid_blas_configuration(3))
            return 1;
        guarded.full_precision_matrix_compute = false;
        if (guarded.valid_blas_configuration(2))
            return 1;
        std::cout << "BLAS fallback F32 flash parity, direct-low-precision and changed-thread guards: passed\n";
        return 0;
    }
    struct Shape { const char* name; int k, m, n; };
    const Shape shapes[] = {{"modulation_gemv", 1280, 7680, 1}, {"target_patch", 1008, 1280, 3456},
                            {"double_mlp", 1280, 5120, 3456}, {"single_qkv_mlp", 1280, 8960, 6912},
                            {"single_projection", 6400, 1280, 6912}};
    nlohmann::ordered_json report = nlohmann::ordered_json::array();
    for (auto shape : shapes) {
        auto w = math_values(shape.k, shape.m, 101, .02f), x = math_values(shape.k, shape.n, 203, 1.f);
        sd::Tensor<float> reference;
        for (auto backend : {cpu, blas}) {
            FashnMatmulProbe probe(backend);
            std::vector<double> times;
            sd::Tensor<float> result;
            for (int iteration = 0; iteration < 3; ++iteration) {
                auto start = std::chrono::steady_clock::now();
                result = probe.run(w, x, 16);
                times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
                if (result.empty())
                    return 1;
            }
            double error = 0, norm = 0;
            if (backend == cpu)
                reference = result;
            if (result.shape() != reference.shape())
                return 1;
            for (int64_t i = 0; i < result.numel(); ++i) {
                double delta = double(result.values()[i]) - reference.values()[i];
                error += delta * delta;
                norm += double(reference.values()[i]) * reference.values()[i];
            }
            double relative = std::sqrt(error / std::max(norm, 1e-30));
            if (!std::isfinite(relative) || relative > 1e-3)
                return 1;
            report.push_back({{"shape", shape.name}, {"k", shape.k}, {"m", shape.m}, {"n", shape.n},
                              {"backend", ggml_backend_name(backend)}, {"threads", 16},
                              {"primary_supported", probe.primary_supported}, {"seconds", times},
                              {"relative_l2", relative}, {"primary_workspace_bytes", probe.diagnostic_runtime_bytes()},
                              {"note", "Wall includes graph construction, allocation/input copies and output copy; first iteration is cold."}});
        }
    }
    std::cout << report.dump(2) << "\n";
    return 0;
}
