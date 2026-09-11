#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "fashn_op_profile.h"
#include "fashn_test_runner.h"
#include "fashn_test_utils.h"
#include "ggml-cpu.h"
#include "json.hpp"
#include "model/diffusion/fashn_vton_model.h"
#include "model_io/safetensors_io.h"

static bool supports_graph(ggml_backend_t backend, ggml_cgraph* graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32)
            ggml_mul_mat_set_prec(node, GGML_PREC_F32);
        if (!ggml_backend_supports_op(backend, node)) {
            std::cerr << "Backend " << ggml_backend_name(backend) << " lacks operation " << ggml_op_name(node->op) << "\n";
            return false;
        }
    }
    return true;
}

struct FashnGELUTestRunner : GGMLRunner {
    explicit FashnGELUTestRunner(ggml_backend_t backend)
        : GGMLRunner(backend) {}
    std::string get_desc() override { return "fashn_gelu_test"; }
    sd::Tensor<float> run(const sd::Tensor<float>& input) {
        auto graph = [&]() {
            auto gf                 = new_graph_custom(64);
            auto ctx                = get_context();
            ctx.full_precision_gelu = true;
            ggml_build_forward_expand(gf, Flux::gelu(&ctx, make_input(input)));
            return supports_graph(ctx.backend, gf) ? gf : nullptr;
        };
        return take_or_empty(GGMLRunner::compute(graph, 2));
    }
};

struct FashnAttentionTestRunner : GGMLRunner {
    bool graph_valid = false;
    explicit FashnAttentionTestRunner(ggml_backend_t backend)
        : GGMLRunner(backend) {}
    std::string get_desc() override { return "fashn_attention_test"; }
    sd::Tensor<float> run(const sd::Tensor<float>& q, const sd::Tensor<float>& k, const sd::Tensor<float>& v, int mode) {
        auto graph = [&]() {
            auto gf     = new_graph_custom(128);
            auto ctx    = get_context();
            auto output = ggml_ext_attention_ext(ctx.ggml_ctx, ctx.backend, make_input(q),
                                                 make_input(k), make_input(v), 2, nullptr,
                                                 false, mode != 0, 1.f, mode == 1);
            ggml_build_forward_expand(gf, output);
            int flash_nodes = 0;
            graph_valid     = true;
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                auto node = ggml_graph_node(gf, i);
                if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                    ++flash_nodes;
                    auto type   = mode == 1 ? GGML_TYPE_F32 : GGML_TYPE_F16;
                    graph_valid = graph_valid && node->src[1]->type == type && node->src[2]->type == type;
                }
            }
            graph_valid = graph_valid && flash_nodes == (mode == 0 ? 0 : 1) && supports_graph(ctx.backend, gf);
            return graph_valid ? gf : nullptr;
        };
        return take_or_empty(GGMLRunner::compute(graph, 2));
    }
};

static int test_attention(ggml_backend_t backend) {
    bool success = true;
    {
        FashnAttentionTestRunner runner(backend);
        sd::Tensor<float> q({256, 33, 1}), k({256, 65, 1}), v({256, 65, 1});
        for (int64_t i = 0; i < q.numel(); ++i) {
            q.values()[i] = std::sin(i * .17f);
        }
        for (int64_t i = 0; i < k.numel(); ++i) {
            k.values()[i] = std::cos(i * .13f);
            v.values()[i] = std::sin(i * .07f) * 3.f;
        }
        auto manual = runner.run(q, k, v, 0);
        success     = runner.graph_valid && !manual.empty();
        for (int mode : {1, 2}) {
            auto flash = runner.run(q, k, v, mode);
            success    = success && runner.graph_valid && flash.numel() == manual.numel();
            for (int64_t i = 0; success && i < flash.numel(); ++i) {
                const float tolerance = mode == 1 ? 2e-5f : 2e-3f;
                success               = std::isfinite(flash.values()[i]) &&
                          std::abs(flash.values()[i] - manual.values()[i]) <= tolerance;
            }
        }
    }
    std::cout << "F32 flash parity and legacy F16 K/V primitive: " << (success ? "passed" : "failed") << "\n";
    return success ? 0 : 1;
}

static int test_gelu(ggml_backend_t backend) {
    bool success = true;
    {
        FashnGELUTestRunner runner(backend);
        sd::Tensor<float> input({24001});
        for (int i = 0; i <= 24000; ++i) {
            input.values()[i] = (i - 12000) / 1000.f;
        }
        auto result = runner.run(input);
        success     = result.numel() == input.numel();
        for (int64_t i = 0; success && i < result.numel(); ++i) {
            double x        = input.values()[i];
            double expected = 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
            success         = std::abs(result.values()[i] - expected) <= 2e-6;
        }
    }
    std::cout << "Full-F32 GELU primitive: " << (success ? "passed" : "failed") << "\n";
    return success ? 0 : 1;
}

static bool compare_capture(ModelLoader& reference, const std::string& name, const sd::Tensor<float>& actual, nlohmann::ordered_json& report) {
    auto expected = load_fixture(reference, name);
    if (expected.empty() || actual.numel() != expected.numel()) {
        LOG_ERROR("Fixture size mismatch for %s", name.c_str());
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        int64_t a = i < actual.dim() ? actual.shape()[i] : 1;
        int64_t b = i < expected.dim() ? expected.shape()[i] : 1;
        if (a != b) {
            LOG_ERROR("Fixture shape mismatch for %s", name.c_str());
            return false;
        }
    }
    double squared_error = 0, squared_reference = 0, max_abs = 0;
    int64_t outside_pointwise = 0;
    for (int64_t i = 0; i < actual.numel(); ++i) {
        double a = actual.values()[i], b = expected.values()[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            LOG_ERROR("Non-finite value in %s", name.c_str());
            return false;
        }
        double error = std::abs(a - b);
        squared_error += error * error;
        squared_reference += b * b;
        max_abs = std::max(max_abs, error);
        outside_pointwise += error > 1e-4 + 1e-4 * std::abs(b);
    }
    double relative_l2 = std::sqrt(squared_error) / std::max(std::sqrt(squared_reference), 1e-12);
    bool primitive     = starts_with(name, "x_embedder.") || starts_with(name, "garment_embedder.") ||
                     starts_with(name, "t_embedder.") || starts_with(name, "y_embedder.") ||
                     starts_with(name, "pe_embedder.");
    bool passed  = relative_l2 <= 1e-3 && (!primitive || outside_pointwise == 0);
    report[name] = {{"passed", passed}, {"max_abs", max_abs}, {"relative_l2", relative_l2}, {"rmse", std::sqrt(squared_error / actual.numel())}, {"outside_pointwise_1e4", outside_pointwise}};
    std::cout << (passed ? "PASS " : "FAIL ") << name << " max_abs=" << max_abs << " relative_l2=" << relative_l2 << std::endl;
    return passed;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--profile-primitives") {
        auto backend = initialize_backend("CPU");
        if (!backend)
            return 1;
        FashnOpProfile profile;
        sd_set_backend_eval_callback(FashnOpProfile::callback, &profile);
        int gelu      = test_gelu(backend);
        int attention = test_attention(backend);
        sd_set_backend_eval_callback(nullptr, nullptr);
        ggml_backend_free(backend);
        std::cout << profile.report().dump(2) << "\n";
        return gelu || attention || profile.rows.empty();
    }
    if (argc == 2 && std::string(argv[1]) == "--list-backends") {
        ggml_backend_load_all();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto device = ggml_backend_dev_get(i);
            std::cout << ggml_backend_dev_name(device) << "\t" << ggml_backend_dev_description(device) << "\n";
        }
        return 0;
    }
    if (argc == 1 || (argc == 3 && std::string(argv[1]) == "--backend")) {
        auto backend = initialize_backend(argc == 1 ? "CPU" : argv[2]);
        if (!backend)
            return 1;
        int gelu      = test_gelu(backend);
        int attention = test_attention(backend);
        ggml_backend_free(backend);
        return gelu || attention;
    }
    if (argc < 4) {
        std::cerr << "Usage: test-fashn-vton-graph [--list-backends | --backend NAME]\n"
                     "       test-fashn-vton-graph checkpoint fixtures output [--backend NAME] [--threads N] [--flash-attention] [--matrix-type f32|f16|bf16|q8_0|q4_0|q5_0|q4_K|q5_K] [--upcast-matrices] [--f32-matrices policy.json] [--no-capture-files] [--profile-ops] [--precompute-modulations] [--fused-gelu]\n";
        return 2;
    }
    bool flash_attention     = false;
    bool upcast_matrices     = false;
    bool profile_ops         = false;
    bool save_captures       = true;
    bool precompute_modulations = false;
    bool fused_gelu = false;
    ggml_type matrix_type    = GGML_TYPE_F32;
    std::string backend_name = "CPU";
    int threads              = 8;
    std::set<std::string> f32_matrices;
    for (int i = 4; i < argc; ++i) {
        std::string option = argv[i];
        if (option == "--flash-attention")
            flash_attention = true;
        else if (option == "--precompute-modulations")
            precompute_modulations = true;
        else if (option == "--fused-gelu")
            fused_gelu = true;
        else if (option == "--upcast-matrices")
            upcast_matrices = true;
        else if (option == "--profile-ops")
            profile_ops = true;
        else if (option == "--no-capture-files")
            save_captures = false;
        else if (option == "--f32-matrices" && i + 1 < argc) {
            if (!read_fashn_f32_matrices(argv[++i], f32_matrices))
                return 2;
        } else if (option == "--backend" && i + 1 < argc)
            backend_name = argv[++i];
        else if (option == "--threads" && i + 1 < argc) {
            std::string text = argv[++i];
            size_t consumed  = 0;
            try {
                threads = std::stoi(text, &consumed);
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid thread count\n";
                return 2;
            } catch (const std::out_of_range&) {
                std::cerr << "Thread count out of range\n";
                return 2;
            }
            if (consumed != text.size() || threads < 1) {
                std::cerr << "Thread count must be positive\n";
                return 2;
            }
        } else if (option == "--matrix-type" && i + 1 < argc) {
            std::string value = argv[++i];
            matrix_type = fashn_test_matrix_type(value);
            if (matrix_type == GGML_TYPE_COUNT) {
                std::cerr << "Invalid matrix type\n";
                return 2;
            }
        } else {
            std::cerr << "Unknown option\n";
            return 2;
        }
    }
    FashnGraphTestContext context(backend_name);
    if (!context.backend)
        return 1;
    const std::filesystem::path fixtures(argv[2]), output(argv[3]);
    std::filesystem::create_directories(output);
    ModelLoader inputs;
    if (!inputs.init_from_file((fixtures / "inputs.safetensors").string())) {
        return 1;
    }
    auto x              = load_fixture(inputs, "noise");
    auto times          = load_fixture(inputs, "times");
    auto ca             = load_fixture(inputs, "ca_images");
    auto garment        = load_fixture(inputs, "garment_images");
    auto pose           = load_fixture(inputs, "person_poses");
    auto garment_pose   = load_fixture(inputs, "garment_poses");
    auto category_float = load_fixture(inputs, "category");
    if (category_float.numel() != 1 || category_float.values()[0] < 1 || category_float.values()[0] > 3 ||
        !std::isfinite(category_float.values()[0]) || std::floor(category_float.values()[0]) != category_float.values()[0]) {
        std::cerr << "Invalid fixture category\n";
        return 1;
    }
    sd::Tensor<int32_t> category({1}, {static_cast<int32_t>(category_float.values()[0])});
    if (!context.init(argv[1], flash_attention, matrix_type, upcast_matrices, threads, f32_matrices,
                      precompute_modulations ? 2 : 0, !precompute_modulations)) {
        return 1;
    }
    if (precompute_modulations &&
        (times.numel() != 1 || !context.prepare_modulations(threads, {{times.values()[0], category.values()[0]}, {times.values()[0], 0}})))
        return 1;
    context.runner->fused_gelu = fused_gelu;
    DiffusionParams params;
    params.x         = &x;
    params.timesteps = &times;
    params.extra     = FashnVTONDiffusionExtra{&ca, &garment, &pose, &garment_pose, &category};
    bool success     = true;
    nlohmann::ordered_json report;
    auto device                      = ggml_backend_get_device(context.backend);
    report["backend"]                = {{"name", ggml_backend_name(context.backend)},
                                        {"device", ggml_backend_dev_name(device)},
                                        {"description", ggml_backend_dev_description(device)}};
    report["full_precision_compute"] = context.runner->full_precision_matrix_compute;
    report["threads"]                = threads;
    report["matrix_types"]           = context.matrix_types;
    report["precompute_modulations"] = precompute_modulations;
    report["fused_gelu"] = fused_gelu;
    report["modulation_cache_bytes"] = context.runner->modulation_cache_bytes();
    report["inactive_modulation_parameter_bytes"] = context.runner->inactive_modulation_parameter_bytes();
    for (const std::string branch : {"conditional", "unconditional"}) {
        if (branch == "unconditional") {
            ca.fill_(0.f);
            garment.fill_(0.f);
            pose.fill_(0.f);
            garment_pose.fill_(0.f);
            category.fill_(0);
        }
        ModelLoader reference;
        if (!reference.init_from_file((fixtures / (branch + ".safetensors")).string())) {
            return 1;
        }
        std::map<std::string, sd::Tensor<float>> captures;
        FashnOpProfile profile;
        if (profile_ops)
            sd_set_backend_eval_callback(FashnOpProfile::callback, &profile);
        auto start    = std::chrono::steady_clock::now();
        auto velocity = context.runner->compute_with_capture(threads, params, &captures);
        auto seconds  = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        sd_set_backend_eval_callback(nullptr, nullptr);
        if (profile_ops)
            report["profiling"][branch] = profile.report();
        std::cout << branch << " forward: " << seconds << " seconds" << std::endl;
        if (velocity.empty() || captures.size() != 17) {
            LOG_ERROR("Incomplete native forward/capture");
            report["passed"]                = false;
            report["error"]                 = "Incomplete native forward/capture";
            report["unsupported_operation"] = context.runner->last_unsupported_operation;
            std::ofstream failure(output / "comparison.json");
            failure << report.dump(2) << "\n";
            failure.close();
            if (!failure)
                LOG_ERROR("Could not persist failure report");
            return 1;
        }
        const int expected_flash_nodes = flash_attention ? 28 : 0;
        if (context.runner->last_flash_attention_nodes != expected_flash_nodes) {
            LOG_ERROR("Wrong flash attention graph node count: expected %d, found %d",
                      expected_flash_nodes, context.runner->last_flash_attention_nodes);
            return 1;
        }
        if (save_captures && !save_capture((output / (branch + ".safetensors")).string(), captures)) {
            return 1;
        }
        for (const auto& [name, value] : captures) {
            success = compare_capture(reference, name, value, report[branch]) && success;
        }
        report[branch + "_seconds"]     = seconds;
        report["flash_attention_nodes"] = context.runner->last_flash_attention_nodes;
        report["matrix_type"]           = ggml_type_name(matrix_type);
        report["upcast_matrices"]       = upcast_matrices;
        report["supported_graph_nodes"] = context.runner->last_supported_nodes;
        report["f32_precision_matmuls"] = context.runner->last_f32_matmuls;
        report["matrix_backends"] = context.runner->last_matrix_backends;
        report["workspace_bytes_by_backend"] = context.runner->diagnostic_workspace_bytes();
    }
    report["passed"] = success;
    std::ofstream result(output / "comparison.json");
    result << report.dump(2) << "\n";
    result.close();
    if (!result) {
        return 1;
    }
    return success ? 0 : 1;
}
