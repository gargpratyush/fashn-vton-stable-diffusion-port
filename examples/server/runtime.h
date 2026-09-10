#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <json.hpp>
#include "common/common.h"
#include "common/resource_owners.hpp"
#include "common/try_on.h"
#include "stable-diffusion.h"

constexpr size_t SD_TRY_ON_MAX_REQUEST_BYTES = 4 * SD_PREPARED_IMAGE_MAX_ENCODED_SIZE + 4096;

using json = nlohmann::json;

struct ArgOptions;
struct SDContextParams;
struct AsyncJobManager;
namespace fashn_prepare { class Preparer; }

struct SDSvrParams {
    std::string listen_ip = "127.0.0.1";
    int listen_port       = 1234;
    std::string serve_html_path;
    std::string try_on_dwpose_dir;
    std::string try_on_parser_dir;
    bool accept_parser_research_license = false;
    bool try_on_ort_no_arena = false;
    bool normal_exit = false;
    sd_log_level_t log_level = SD_LOG_INFO;
    bool color       = false;

    ArgOptions get_options();
    bool validate();
    bool resolve_and_validate();
    std::string to_string() const;
};

struct LoraEntry {
    std::string name;
    std::string path;
    std::string fullpath;
};

struct UpscalerEntry {
    std::string name;
    std::string path;
    std::string fullpath;
    std::string model_name;
    int scale = 4;
};

struct ServerRuntime {
    sd_ctx_t* sd_ctx;
    std::mutex* sd_ctx_mutex;
    const SDSvrParams* svr_params;
    const SDContextParams* ctx_params;
    const SDGenerationParams* default_gen_params;
    std::vector<LoraEntry>* lora_cache;
    std::mutex* lora_mutex;
    std::vector<UpscalerEntry>* upscaler_cache;
    std::mutex* upscaler_mutex;
    AsyncJobManager* async_job_manager;
    std::shared_ptr<fashn_prepare::Preparer> try_on_preparer;
    bool try_on_parser_enabled = false;
};

struct ImgGenJobRequest {
    SDGenerationParams gen_params;
    std::string output_format = "png";
    int output_compression    = 100;

    sd_img_gen_params_t to_sd_img_gen_params_t() {
        return gen_params.to_sd_img_gen_params_t();
    }
};

struct VidGenJobRequest {
    SDGenerationParams gen_params;
    std::string output_format = "webm";
    int output_compression    = 100;

    sd_vid_gen_params_t to_sd_vid_gen_params_t() {
        return gen_params.to_sd_vid_gen_params_t();
    }
};

struct TryOnJobRequest {
    PreparedTryOnInputs inputs;
    bool raw = false;
    std::array<std::string, 2> raw_images;
    std::string category;
    std::string garment_photo_type;
    bool segmentation_free = true;
    void release_raw_images() {
        for (auto& image : raw_images)
            std::string().swap(image);
    }
};

bool initialize_try_on_preparer(ServerRuntime& runtime, std::string& error_message);
bool parse_try_on_request(const json& body, TryOnJobRequest& request, std::string& error_message,
                          bool allow_raw = false, bool allow_parser = false);

std::string base64_encode(const std::vector<uint8_t>& bytes);
std::string normalize_output_format(std::string output_format);
std::vector<std::string> supported_img_output_formats(bool allow_webp = true);
std::vector<std::string> supported_vid_output_formats();
bool assign_output_options(ImgGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           bool allow_webp,
                           std::string& error_message);
bool assign_output_options(VidGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           std::string& error_message);
std::string video_mime_type(const std::string& output_format);
bool runtime_supports_generation_mode(const ServerRuntime& runtime, SDMode mode);
std::string unsupported_generation_mode_error(SDMode mode);
void refresh_lora_cache(ServerRuntime& rt);
std::string get_lora_full_path(ServerRuntime& rt, const std::string& path);
void refresh_upscaler_cache(ServerRuntime& rt);
int64_t unix_timestamp_now();
