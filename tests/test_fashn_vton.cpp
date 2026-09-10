#include <iostream>
#include <limits>
#include <string>

#include "model/diffusion/fashn_vton.h"
#include "model_loader.h"
#include "name_conversion.h"
#include "stable-diffusion.h"

static int failures = 0;
static int checks   = 0;

static void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    }
}

static String2TensorStorage make_metadata(const std::string& prefix = "") {
    String2TensorStorage tensors;
    for (const auto& [name, shape] : FashnVTONConfig::expected_shapes()) {
        const std::string full_name = prefix + name;
        tensors[full_name]          = TensorStorage(full_name, GGML_TYPE_BF16, shape.data(),
                                                    static_cast<int>(shape.size()), 0);
    }
    return tensors;
}

static bool valid(const String2TensorStorage& tensors, std::string* error) {
    auto config = FashnVTONConfig::detect_from_weights(tensors);
    return config.validate_v15(tensors, "model.diffusion_model", error);
}

static void expect_invalid(String2TensorStorage& tensors, const std::string& reason) {
    std::string error;
    require(!valid(tensors, &error), "reject malformed metadata: " + reason);
    require(error.find(reason) != std::string::npos, "specific error for " + reason + ": " + error);
    ModelLoader loader;
    loader.get_tensor_storage_map() = std::move(tensors);
    require(loader.get_sd_version() == VERSION_COUNT, "malformed FASHN must not fall back to FLUX");
}

static void test_contract() {
    std::string error;
    auto tensors = make_metadata();
    require(tensors.size() == 366, "366 stored tensors");
    int64_t scalars = 0;
    for (const auto& [name, tensor] : tensors) {
        scalars += tensor.nelements();
    }
    require(scalars == 971814240, "released scalar count");
    require(valid(tensors, &error), "complete raw contract: " + error);
    tensors.erase("patch_mixer_token");
    require(valid(tensors, &error), "known unused buffer is optional");
    tensors                                = make_metadata();
    tensors.at("patch_mixer_token").n_dims = 1;
    require(valid(tensors, &error), "GGUF singleton rank normalization");
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        tensors.at("y_embedder.weight").type = type;
        require(valid(tensors, &error), "supported floating dtype");
    }
    tensors                                                = make_metadata();
    tensors.at("double_blocks.0.img_attn.qkv.weight").type = GGML_TYPE_Q8_0;
    require(valid(tensors, &error), "eligible Q8_0 matrix metadata for conversion diagnostics");
    for (const char* name : {"garment_embedder.proj.weight", "y_embedder.weight",
                             "single_blocks.0.modulation.lin.weight", "final_layer.linear.weight"}) {
        auto invalid          = make_metadata();
        invalid.at(name).type = GGML_TYPE_Q8_0;
        expect_invalid(invalid, "dtype");
    }
    tensors.at("double_blocks.0.img_attn.qkv.weight").type = GGML_TYPE_Q4_0;
    expect_invalid(tensors, "dtype");

    tensors = make_metadata();
    tensors.erase("double_blocks.3.img_attn.qkv.weight");
    expect_invalid(tensors, "Missing FASHN tensor");
    tensors                                          = make_metadata();
    tensors.at("garment_embedder.proj.weight").ne[2] = 3;
    expect_invalid(tensors, "shape mismatch");
    tensors                                    = make_metadata();
    tensors.at("x_embedder.proj.weight").ne[4] = 2;
    expect_invalid(tensors, "shape mismatch");
    tensors                                     = make_metadata();
    tensors.at("x_embedder.proj.weight").n_dims = 0;
    expect_invalid(tensors, "rank");
    tensors                              = make_metadata();
    tensors.at("y_embedder.weight").type = GGML_TYPE_I32;
    expect_invalid(tensors, "dtype");
    tensors                                = make_metadata();
    tensors.at("y_embedder.weight").is_f64 = true;
    expect_invalid(tensors, "dtype");
    tensors                                            = make_metadata();
    tensors["single_blocks.16.linear1.weight"]         = tensors.at("single_blocks.0.linear1.weight");
    tensors.at("single_blocks.16.linear1.weight").name = "single_blocks.16.linear1.weight";
    expect_invalid(tensors, "Unexpected FASHN tensor");
    tensors                                                    = make_metadata();
    tensors["model.diffusion_model.y_embedder.weight"]         = tensors.at("y_embedder.weight");
    tensors.at("model.diffusion_model.y_embedder.weight").name = "model.diffusion_model.y_embedder.weight";
    const size_t before                                        = tensors.size();
    ModelLoader collision_loader;
    collision_loader.get_tensor_storage_map() = std::move(tensors);
    require(!collision_loader.convert_tensors_name(), "prefix collision fails name conversion");
    require(collision_loader.get_tensor_storage_map().size() == before, "invalid conversion preserves colliding keys");
    require(collision_loader.get_sd_version() == VERSION_COUNT, "prefix collision is not accepted");

    tensors             = make_metadata();
    auto config         = FashnVTONConfig::detect_from_weights(tensors);
    config.input_height = 768;
    require(!config.validate_v15(tensors, "model.diffusion_model", &error), "reject non-production resolution");
    config          = FashnVTONConfig::detect_from_weights(tensors);
    config.axes_dim = {32, 48, 48};
    require(!config.validate_v15(tensors, "model.diffusion_model", &error), "reject different RoPE axes");
}

static void test_detection_and_names() {
    require(sd_version_is_dit(VERSION_FASHN_VTON_1_5), "FASHN is a DiT");
    require(!sd_version_is_flux(VERSION_FASHN_VTON_1_5), "FASHN is not the FLUX pipeline");
    require(!sd_version_uses_flux_vae(VERSION_FASHN_VTON_1_5), "FASHN does not require FLUX VAE");
    for (const std::string prefix : {"", "model.diffusion_model."}) {
        ModelLoader loader;
        loader.get_tensor_storage_map() = make_metadata(prefix);
        require(loader.get_sd_version() == VERSION_FASHN_VTON_1_5, "detect raw/prefixed FASHN before FLUX");
        require(loader.convert_tensors_name(), "normalize FASHN names");
        require(loader.convert_tensors_name(), "repeat name normalization");
        require(loader.get_sd_version() == VERSION_FASHN_VTON_1_5, "idempotent normalized detection");
        require(loader.get_tensor_storage_map().size() == 366, "normalization preserves every tensor");
        for (const auto& [suffix, shape] : FashnVTONConfig::expected_shapes()) {
            std::string name = "model.diffusion_model." + suffix;
            auto found       = loader.get_tensor_storage_map().find(name);
            require(found != loader.get_tensor_storage_map().end(), "canonical name: " + name);
            if (found != loader.get_tensor_storage_map().end()) {
                require(found->second.name == name, "map key and tensor name agree");
            }
        }
    }
    require(convert_tensor_name("garment_embedder.proj.weight", VERSION_FASHN_VTON_1_5) ==
                "model.diffusion_model.garment_embedder.proj.weight",
            "preserve FASHN module suffix");
    require(convert_tensor_name("unrelated.weight", VERSION_FASHN_VTON_1_5) == "unrelated.weight",
            "do not invent a prefix for unrelated tensors");

    ModelLoader flux;
    const int64_t qkv_shape[]                = {3072, 9216};
    const std::string flux_name              = "model.diffusion_model.double_blocks.0.img_attn.qkv.weight";
    flux.get_tensor_storage_map()[flux_name] = TensorStorage(flux_name, GGML_TYPE_F16, qkv_shape, 2, 0);
    require(flux.get_sd_version() == VERSION_FLUX, "existing FLUX detection unchanged");

    ModelLoader hunyuan;
    const std::string hunyuan_name =
        "model.diffusion_model.txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.weight";
    hunyuan.get_tensor_storage_map()[hunyuan_name] = TensorStorage(hunyuan_name, GGML_TYPE_F16, qkv_shape, 2, 0);
    require(hunyuan.get_sd_version() == VERSION_HUNYUAN_VIDEO, "existing Hunyuan detection unchanged");
}

static void test_file_name_collisions() {
    auto metadata = make_metadata();
    std::vector<TensorStorage> tensors;
    for (const auto& [name, tensor] : metadata) {
        tensors.push_back(tensor);
    }
    std::string error;
    require(FashnVTONConfig::validate_file_names(tensors, "", &error), "raw file has unique canonical names");
    require(FashnVTONConfig::validate_file_names(tensors, "model.diffusion_model.", &error),
            "adding a prefix to the original file is safe");
    TensorStorage duplicate = metadata.at("y_embedder.weight");
    duplicate.name          = "model.diffusion_model.y_embedder.weight";
    tensors.push_back(duplicate);
    for (const std::string prefix : {"", "model.diffusion_model."}) {
        require(!FashnVTONConfig::validate_file_names(tensors, prefix, &error),
                "reject collision before prefix insertion can overwrite a tensor");
        require(error.find("prefix collision") != std::string::npos, "specific file collision diagnostic");
    }
    tensors.clear();
    duplicate.name = "unrelated.weight";
    tensors.push_back(duplicate);
    tensors.push_back(duplicate);
    require(FashnVTONConfig::validate_file_names(tensors, "", &error),
            "do not change other model families' file ingestion");
}

static void test_official_checkpoint(const char* path) {
    for (const std::string prefix : {"", "model.diffusion_model."}) {
        ModelLoader loader;
        bool loaded = loader.init_from_file(path, prefix);
        require(loaded, "read official checkpoint metadata");
        if (!loaded) {
            continue;
        }
        require(loader.get_sd_version() == VERSION_FASHN_VTON_1_5, "official checkpoint is FASHN");
        require(loader.convert_tensors_name(), "normalize official checkpoint names");
        require(loader.get_sd_version() == VERSION_FASHN_VTON_1_5, "official normalized checkpoint is FASHN");
    }
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = path;
    params.n_threads  = 1;
    sd_ctx_t* ctx     = new_sd_ctx(&params);
    require(ctx != nullptr, "FASHN context initializes without text encoder or VAE");
    if (ctx != nullptr) {
        require(sd_ctx_supports_try_on(ctx), "dedicated try-on capability");
        require(!sd_ctx_supports_image_generation(ctx), "not generic image generation");
        require(!sd_ctx_supports_video_generation(ctx), "not video generation");
        sd_try_on_params_t request;
        sd_try_on_params_init(&request);
        require(request.struct_size == sizeof(request) && request.steps == 30 && request.cfg == 1.5f,
                "initialized try-on defaults");
        sd_image_t* images = nullptr;
        int count          = 99;
        require(!generate_try_on(ctx, &request, &images, &count) && images == nullptr && count == 0,
                "missing prepared inputs rejected with empty outputs");
        request.struct_size = 0;
        require(!generate_try_on(ctx, &request, &images, &count), "truncated parameter struct rejected");
        sd_try_on_params_init(&request);
        request.category = static_cast<sd_try_on_category_t>(0);
        require(!generate_try_on(ctx, &request, &images, &count), "public category zero rejected");
        sd_try_on_params_init(&request);
        request.cfg = std::numeric_limits<float>::quiet_NaN();
        require(!generate_try_on(ctx, &request, &images, &count), "nonfinite guidance rejected");
        std::vector<uint8_t> rgb(576 * 864 * 3), pose(576 * 864);
        sd_try_on_params_init(&request);
        request.ca_image = request.garment_image = {576, 864, 3, rgb.data()};
        request.person_pose = request.garment_pose = {576, 864, 1, pose.data()};
        request.crop_x                             = 575;
        request.crop_width                         = 2;
        request.crop_height                        = 864;
        require(!generate_try_on(ctx, &request, &images, &count), "crop outside canvas rejected");
        request.crop_x = request.crop_width = request.crop_height = 0;
        bool cancelled                                            = true;
        sd_try_on_callbacks_t callbacks{};
        callbacks.struct_size = sizeof(callbacks);
        callbacks.cancelled   = [](void* data) { return *static_cast<bool*>(data); };
        callbacks.data        = &cancelled;
        count                 = 99;
        require(!generate_try_on_with_callbacks(ctx, &request, &callbacks, &images, &count) &&
                    images == nullptr && count == 0,
                "request-local cancellation before generation survives context reset");
        callbacks.struct_size = 0;
        require(!generate_try_on_with_callbacks(ctx, &request, &callbacks, &images, &count),
                "truncated callback structure rejected");
        sd_img_gen_params_t image_request;
        sd_img_gen_params_init(&image_request);
        require(!generate_image(ctx, &image_request, &images, &count), "generic image API rejects FASHN safely");
        free_sd_ctx(ctx);
    }
}

int main(int argc, char** argv) {
    test_contract();
    test_detection_and_names();
    test_file_name_collisions();
    if (argc == 2) {
        test_official_checkpoint(argv[1]);
    } else if (argc != 1) {
        std::cerr << "Usage: test-fashn-vton [official-checkpoint.safetensors]\n";
        return 2;
    }
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
