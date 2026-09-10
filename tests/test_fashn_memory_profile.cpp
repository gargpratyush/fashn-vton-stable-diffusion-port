#include <filesystem>
#include "fashn_memory_profile.h"

int main() {
    auto stats = std::make_shared<LoadDiagnostics>();
    {
        ScopedLoadScratch read(stats.get(), false);
        read.set(64);
        {
            ScopedLoadScratch f32(stats.get(), true);
            f32.set(32);
            read.set(128);
            if (stats->scratch_live != 160 || stats->scratch_peak != 160)
                return 1;
            read.set(16);
            if (stats->scratch_live != 48)
                return 1;
        }
        if (stats->scratch_live != 16)
            return 1;
    }
    if (stats->scratch_live || stats->read_live || stats->f32_live || stats->read_peak != 128 || stats->f32_peak != 32)
        return 1;
    auto root = std::filesystem::temp_directory_path() /
                ("fashn-memory-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(root))
        return 1;
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    ggml_context* ctx = ggml_init({1024 * 1024, nullptr, false});
    if (!ctx)
        return 1;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_guard(ctx, ggml_free);
    auto source = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 32, 64);
    std::vector<float> values(2048);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * .125f;
    ggml_fp32_to_bf16_row(values.data(), static_cast<ggml_bf16_t*>(source->data), values.size());
    ggml_set_name(source, "weight");
    TensorWriteInfo info;
    info.tensor = source;
    info.n_dims = 2;
    info.ne[0] = 32;
    info.ne[1] = 64;
    std::string error;
    if (!write_safetensors_file((root / "source.safetensors").string(), {info}, &error))
        return 1;
    ModelLoader loader;
    loader.set_n_threads(2);
    stats = std::make_shared<LoadDiagnostics>();
    loader.set_diagnostics(stats);
    std::vector<std::string> events;
    stats->event = [&](const char* phase) { events.emplace_back(phase); };
    if (!loader.init_from_file((root / "source.safetensors").string()))
        return 1;
    auto target = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 32, 64);
    auto callback = [&](const TensorStorage& storage, ggml_tensor** out) {
        if (storage.name != "weight")
            return false;
        *out = target;
        return true;
    };
    if (!loader.load_tensors(callback, true, nullptr, false))
        return 1;
    std::vector<uint8_t> expected(ggml_nbytes(target));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), expected.data(), 0, 64, 32, nullptr);
    if (std::memcmp(target->data, expected.data(), expected.size()) != 0 ||
        stats->converted_tensors != 1 || stats->f32_peak != 8192 ||
        stats->read_peak < 4096 || stats->scratch_live != 0 ||
        loader.mapped_files().size() != 1 || events.back() != "conversion_end")
        return 1;
    {
        FashnMemoryProfile profile((root / "profile.jsonl").string(), true);
        profile.record("unit", nullptr, stats.get());
        if (!profile.good())
            return 1;
    }
    std::ifstream file(root / "profile.jsonl");
    nlohmann::json row;
    file >> row;
    if (row["phase"] != "unit" || row["conversion"]["f32_peak_bytes"] != 8192)
        return 1;
#ifdef _WIN32
    if (row["process"]["working_set_bytes"].get<uint64_t>() == 0 ||
        row["resident_bytes_by_region_type"]["private"].get<uint64_t>() == 0)
        return 1;
#endif
    std::cout << "Loading scratch, mapped source, phase memory and unchanged Q8 conversion: passed\n";
    return 0;
}
