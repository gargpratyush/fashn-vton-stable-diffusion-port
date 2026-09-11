#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include "core/rng_mt19937.hpp"
#include "fashn_test_runner.h"
#include "fashn_memory_profile.h"
#include "fashn_noise_fixture.h"
#include "json.hpp"
#include "runtime/fashn_vton_sampling.h"

int main(int argc, char** argv) {
    if (argc < 11) {
        std::cerr << "Usage: test-fashn-vton-trajectory checkpoint conditions output steps cfg shift skip seed category threads"
                     " [--matrix-type f32|bf16|f16|q8_0|q4_0|q5_0|q4_K|q5_K] [--upcast-matrices] [--f32-matrices policy.json]"
                     " [--mmap on|off] [--load-threads N] [--load-only | --probe-forwards N]"
                     " [--no-record] [--memory-profile] [--classify-pages] [--precompute-modulations] [--modulation-cache-mib N] [--fused-gelu] [--backend CPU|BLAS]"
                     " [--initial-noise fixture.safetensors]\n";
        return 2;
    }
    try {
        ggml_type matrix_type = GGML_TYPE_BF16;
        bool explicit_upcast  = false;
        bool mmap = true, load_only = false, recording = true, memory_profile = false, classify_pages = false;
        bool precompute_modulations = false;
        bool fused_gelu = false;
        std::string backend_name = "CPU";
        std::string initial_noise_path;
        int modulation_cache_mib = 128;
        int load_threads = 0, probe_forwards = 0;
        std::set<std::string> f32_matrices;
        for (int i = 11; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--initial-noise" && i + 1 < argc) {
                if (!initial_noise_path.empty())
                    throw std::invalid_argument("Specify initial noise only once");
                initial_noise_path = argv[++i];
                if (initial_noise_path.empty())
                    throw std::invalid_argument("Initial noise path must not be empty");
            } else if (option == "--backend" && i + 1 < argc) {
                backend_name = argv[++i];
                if (backend_name != "CPU" && backend_name != "BLAS")
                    throw std::invalid_argument("Trajectory supports only CPU or diagnostic BLAS with CPU fallback");
            } else if (option == "--fused-gelu") {
                fused_gelu = true;
            } else if (option == "--precompute-modulations") {
                precompute_modulations = true;
            } else if (option == "--memory-profile") {
                memory_profile = true;
            } else if (option == "--classify-pages") {
                classify_pages = true;
            } else if (option == "--load-only") {
                load_only = true;
            } else if (option == "--no-record") {
                recording = false;
            } else if (option == "--mmap" && i + 1 < argc) {
                const std::string value = argv[++i];
                if (value != "on" && value != "off")
                    throw std::invalid_argument("mmap must be on or off");
                mmap = value == "on";
            } else if ((option == "--load-threads" || option == "--probe-forwards" || option == "--modulation-cache-mib") && i + 1 < argc) {
                const std::string value = argv[++i];
                size_t consumed = 0;
                int number = std::stoi(value, &consumed);
                if (consumed != value.size() || number < 1 || number > 1024)
                    throw std::invalid_argument("Diagnostic counts must be integers in [1,1024]");
                if (option == "--modulation-cache-mib")
                    modulation_cache_mib = number;
                else
                    (option == "--load-threads" ? load_threads : probe_forwards) = number;
            } else if (option == "--upcast-matrices") {
                explicit_upcast = true;
            } else if (option == "--f32-matrices" && i + 1 < argc) {
                if (!read_fashn_f32_matrices(argv[++i], f32_matrices))
                    return 2;
            } else if (option == "--matrix-type" && i + 1 < argc) {
                const std::string type = argv[++i];
                matrix_type = fashn_test_matrix_type(type);
                if (matrix_type == GGML_TYPE_COUNT)
                    throw std::invalid_argument("Unknown diagnostic matrix type");
            } else {
                throw std::invalid_argument("Unknown or incomplete diagnostic option");
            }
        }
        const bool upcast_matrices = explicit_upcast || !ggml_is_quantized(matrix_type);
        if (backend_name == "BLAS" && !upcast_matrices)
            throw std::invalid_argument("BLAS requires F32 matrix computation; direct Q8 arithmetic is not supported");
        FashnVTONSamplingParams sampling{std::stoi(argv[4]), std::stof(argv[5]), std::stof(argv[6]), std::stoi(argv[7])};
        const uint64_t seed      = std::stoull(argv[8]);
        const int category_value = std::stoi(argv[9]), threads = std::stoi(argv[10]);
        if (!sampling.validate() || category_value < 1 || category_value > 3 || threads < 1)
            throw std::invalid_argument("Invalid sampling, category or thread parameters");
        if ((classify_pages && !memory_profile) || (load_only && probe_forwards))
            throw std::invalid_argument("Page classification requires profiling; load-only and forward probes are exclusive");
        if (precompute_modulations && (mmap || load_only))
            throw std::invalid_argument("Modulation precomputation requires --mmap off and an inference mode");
        if (modulation_cache_mib > 128 || (!precompute_modulations && modulation_cache_mib != 128))
            throw std::invalid_argument("Modulation cache must be enabled and bounded to 1..128 MiB");
        if (load_only && !initial_noise_path.empty())
            throw std::invalid_argument("Initial noise is not used in load-only mode");
        sd::Tensor<float> imported_noise;
        if (!initial_noise_path.empty())
            imported_noise = load_fashn_noise_fixture(initial_noise_path);
        const std::filesystem::path output(argv[3]);
        if (std::filesystem::exists(output))
            throw std::invalid_argument("Use a new output directory");
        std::filesystem::create_directories(output);
        std::unique_ptr<FashnMemoryProfile> profile;
        std::shared_ptr<LoadDiagnostics> load_stats;
        if (memory_profile) {
            profile = std::make_unique<FashnMemoryProfile>((output / "memory-profile.jsonl").string(), classify_pages);
            load_stats = std::make_shared<LoadDiagnostics>();
            profile->record("process_setup");
        }
        FashnGraphTestContext context(backend_name);
        auto notify = [trace = profile.get(), owner = &context, stats = load_stats.get()](
                          const char* phase, ggml_cgraph* graph = nullptr, int n_threads = 0) noexcept {
            if (trace)
                trace->record(phase, owner, stats, graph, n_threads);
        };
        if (load_stats) {
            load_stats->event = [notify](const char* phase) { notify(phase); };
            context.manager->loader().set_diagnostics(load_stats);
        }
        notify("context_setup");
        if (!context.init(argv[1], true, matrix_type, upcast_matrices, threads, f32_matrices, load_threads, mmap))
            return 1;
        if (profile)
            context.runner->diagnostic_observer = notify;
        context.runner->fused_gelu = fused_gelu;
        notify("metadata_ready");
        double sampling_seconds = 0;
        std::vector<double> forward_seconds;
        std::vector<size_t> forward_workspace_bytes;
        std::vector<std::map<std::string, size_t>> forward_backend_workspace;
        if (load_only) {
            if (!context.manager->load_all_params_eagerly())
                return 1;
            notify("load_only_ready");
        } else {
            ModelLoader loader;
            if (!loader.init_from_file(argv[2]))
                return 1;
            auto ca = load_fixture(loader, "ca_images"), garment = load_fixture(loader, "garment_images");
            auto pose = load_fixture(loader, "person_poses"), garment_pose = load_fixture(loader, "garment_poses");
            // GGML metadata may omit the trailing batch singleton.
            for (auto* tensor : {&ca, &garment, &pose, &garment_pose}) {
                if (tensor->empty())
                    return 1;
                if (tensor->dim() == 3)
                    *tensor = tensor->reshape({tensor->shape()[0], tensor->shape()[1], tensor->shape()[2], 1});
            }
            sd::Tensor<int32_t> category({1}, {category_value});
            FashnVTONDiffusionExtra conditions{&ca, &garment, &pose, &garment_pose, &category};
            MT19937RNG rng;
            rng.manual_seed(seed);
            sd::Tensor<float> noise = imported_noise.empty()
                ? sd::Tensor<float>({576, 864, 3, 1}, rng.randn(576 * 864 * 3))
                : std::move(imported_noise);
            if (recording && !save_capture((output / "initial.safetensors").string(), {{"noise", noise}}))
                return 1;
            notify("conditions_ready");
            auto start = std::chrono::steady_clock::now();
            if (precompute_modulations) {
                auto pairs = fashn_vton_modulation_pairs(sampling, category_value);
                if (!context.prepare_modulations(threads, pairs, size_t(modulation_cache_mib) * 1024 * 1024))
                    return 1;
            }
            if (probe_forwards) {
                sd::Tensor<float> time({1}, {0.f});
                DiffusionParams params;
                params.x = &noise;
                params.timesteps = &time;
                params.extra = conditions;
                sd::Tensor<float> velocity;
                for (int index = 0; index < probe_forwards; ++index) {
                    auto forward_start = std::chrono::steady_clock::now();
                    velocity = context.runner->compute(threads, params);
                    if (velocity.empty())
                        return 1;
                    forward_seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - forward_start).count());
                    forward_workspace_bytes.push_back(context.runner->diagnostic_runtime_bytes());
                    forward_backend_workspace.push_back(context.runner->diagnostic_workspace_bytes());
                    std::cout << "Conditional t0 forward " << index + 1 << "/" << probe_forwards << std::endl;
                }
                if (!save_capture((output / "probe.safetensors").string(), {{"velocity", velocity}}))
                    return 1;
                context.runner->runner_end();
            } else {
                auto result = sample_fashn_vton(*context.runner, threads, noise, conditions, sampling, {},
                    [](int step, int total) { std::cout << "Native trajectory step " << step << "/" << total << std::endl; },
                    [&](int step, const sd::Tensor<float>& image, const sd::Tensor<float>& vc, const sd::Tensor<float>* vu) {
                        if (!recording)
                            return true;
                        notify("step_observer_begin");
                        sd::Tensor<float> guided = vc;
                        if (vu)
                            for (int64_t i = 0; i < guided.numel(); ++i)
                                guided.values()[i] = vu->values()[i] + sampling.cfg * (vc.values()[i] - vu->values()[i]);
                        bool saved = save_capture((output / ("step-" + std::to_string(step) + ".safetensors")).string(),
                                                  {{"image", image}, {"guided_velocity", guided}});
                        notify("step_observer_end");
                        return saved;
                    });
                if (result.empty() || (!recording && !save_capture((output / "final.safetensors").string(), {{"image", result}})))
                    return 1;
            }
            sampling_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            notify("sampling_complete");
        }
        notify("before_model_release");
        const auto modulation_refills = context.runner->modulation_refills;
        const auto matrix_backends = context.runner->last_matrix_backends;
        const std::string backend_description = ggml_backend_dev_description(ggml_backend_get_device(context.backend));
        context.close();
        notify("model_released");
        if (profile && !profile->good())
            return 1;
        nlohmann::ordered_json manifest = {
            {"steps", sampling.steps}, {"cfg", sampling.cfg}, {"shift", sampling.shift}, {"skip_cfg_last_n_steps", sampling.skip_cfg_last_n_steps}, {"seed", seed}, {"category", category_value}, {"threads", threads}, {"schedule", fashn_vton_schedule(sampling)}, {"backend", "CPU"}, {"matrix_type", ggml_type_name(matrix_type)}, {"upcast_matrices", upcast_matrices}, {"matrix_types", context.matrix_types}, {"diagnostic_only", true}, {"attention", "F32 flash"}, {"sampling_seconds", sampling_seconds},
            {"mmap", mmap}, {"load_threads", load_threads ? load_threads : threads},
            {"load_only", load_only}, {"probe_forwards", probe_forwards},
            {"recorded_trajectory", recording && !load_only && !probe_forwards},
            {"memory_profile", memory_profile}, {"page_classification", classify_pages}};
        manifest["precompute_modulations"] = precompute_modulations;
        manifest["noise_source"] = initial_noise_path.empty() ? "cpu_rng" : "imported_fixture";
        manifest["backend"] = backend_name;
        manifest["backend_description"] = backend_description;
        manifest["matrix_backends"] = matrix_backends;
        manifest["probe_backend_workspace_bytes"] = forward_backend_workspace;
        manifest["fused_gelu"] = fused_gelu;
        manifest["modulation_cache_mib"] = modulation_cache_mib;
        manifest["modulation_refills"] = modulation_refills;
        manifest["probe_forward_seconds"] = forward_seconds;
        manifest["probe_workspace_bytes"] = forward_workspace_bytes;
        std::ofstream file(output / "manifest.json");
        file << manifest.dump(2) << "\n";
        file.close();
        return file ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Trajectory failed: " << error.what() << "\n";
        return 1;
    }
}
