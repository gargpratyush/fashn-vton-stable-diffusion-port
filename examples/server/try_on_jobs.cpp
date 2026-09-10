#include "async_jobs.h"

#include <cmath>
#include <set>

#include "common/media_io.h"

bool parse_try_on_request(const json& body, TryOnJobRequest& request, std::string& error_message, bool allow_raw, bool allow_parser) {
    const std::set<std::string> allowed = {"inputs", "raw_inputs", "steps", "cfg", "shift", "skip_cfg_last_n_steps", "seed", "sample_count"};
    if (!body.is_object() || body.contains("inputs") == body.contains("raw_inputs")) {
        error_message = "try_on requires exactly one of inputs or raw_inputs";
        return false;
    }
    for (const auto& item : body.items()) {
        if (allowed.count(item.key()) == 0) {
            error_message = "unknown try_on field: " + item.key();
            return false;
        }
    }
    if (body.contains("raw_inputs")) {
        if (!allow_raw) {
            error_message = "raw_inputs requires operator-configured native preprocessing";
            return false;
        }
        const auto& raw                    = body["raw_inputs"];
        const std::set<std::string> fields = {"person_image", "garment_image", "category", "garment_photo_type", "segmentation_free"};
        if (!raw.is_object() || raw.size() != fields.size()) {
            error_message = "raw_inputs requires person_image, garment_image, category, garment_photo_type, segmentation_free";
            return false;
        }
        for (const auto& field : fields) {
            if (!raw.contains(field) || (field == "segmentation_free" ? !raw[field].is_boolean() : !raw[field].is_string())) {
                error_message = "invalid raw_inputs field: " + field;
                return false;
            }
        }
        request.category           = raw["category"].get<std::string>();
        request.garment_photo_type = raw["garment_photo_type"].get<std::string>();
        request.segmentation_free  = raw["segmentation_free"].get<bool>();
        sd_try_on_params_init(&request.inputs.params);
        if (request.category == "tops")
            request.inputs.params.category = SD_TRY_ON_TOPS;
        else if (request.category == "bottoms")
            request.inputs.params.category = SD_TRY_ON_BOTTOMS;
        else if (request.category == "one-pieces")
            request.inputs.params.category = SD_TRY_ON_ONE_PIECES;
        else {
            error_message = "category must be tops, bottoms, or one-pieces";
            return false;
        }
        if (request.garment_photo_type != "flat-lay" && request.garment_photo_type != "model") {
            error_message = "garment_photo_type must be flat-lay or model";
            return false;
        }
        if ((!request.segmentation_free || request.garment_photo_type == "model") && !allow_parser) {
            error_message = "Requested mode requires an operator-consented research/evaluation parser";
            return false;
        }
        const char* names[] = {"person_image", "garment_image"};
        for (size_t i = 0; i < request.raw_images.size(); ++i) {
            request.raw_images[i] = raw[names[i]].get<std::string>();
            SDImageOwner decoded;
            if (!decode_base64_image(request.raw_images[i], 3, 0, 0, decoded, true, SD_RAW_IMAGE_MAX_PIXELS)) {
                error_message = "Raw images must be bounded embedded 8-bit RGB PNGs: 3 MiB base64, 4096 per dimension, 4194304 pixels maximum";
                return false;
            }
        }
        // Queue encoded data, not potentially large decoded pixel buffers.
        request.raw = true;
    } else if (!request.inputs.load_embedded(body["inputs"])) {
        error_message = "invalid prepared inputs: check schema, category, crop, and exact 576x864 RGB/RGB/L/L 8-bit PNGs";
        return false;
    }
    auto& params = request.inputs.params;
    auto integer = [&](const char* name, int low, int high, int& value) {
        if (!body.contains(name)) {
            return true;
        }
        const auto& field = body[name];
        if (!field.is_number_integer() || field.get<double>() < low || field.get<double>() > high) {
            error_message = std::string(name) + " must be an integer in [" + std::to_string(low) + "," + std::to_string(high) + "]";
            return false;
        }
        value = field.get<int>();
        return true;
    };
    if (!integer("steps", 1, 1000, params.steps) ||
        !integer("sample_count", 1, 4, params.sample_count) ||
        !integer("skip_cfg_last_n_steps", 0, params.steps, params.skip_cfg_last_n_steps)) {
        return false;
    }
    for (const char* name : {"cfg", "shift"}) {
        if (body.contains(name)) {
            if (!body[name].is_number()) {
                error_message = std::string(name) + " must be a finite number";
                return false;
            }
            float value = body[name].get<float>();
            if (!std::isfinite(value) || (std::string(name) == "cfg" ? value < 0.f : std::abs(value) > 20.f)) {
                error_message = "cfg must be finite and nonnegative; shift must be finite in [-20,20]";
                return false;
            }
            (std::string(name) == "cfg" ? params.cfg : params.shift) = value;
        }
    }
    if (body.contains("seed")) {
        const auto& seed = body["seed"];
        if (!seed.is_number_integer() || (!seed.is_number_unsigned() && seed.get<int64_t>() < 0)) {
            error_message = "seed must be a nonnegative uint64 integer";
            return false;
        }
        params.seed = seed.get<uint64_t>();
    }
    return true;
}

bool execute_try_on_job(ServerRuntime& runtime, AsyncGenerationJob& job, std::vector<std::string>& output_images, std::string& error_message) {
    if (job.try_on.raw) {
        job.try_on_phase.store(TryOnPhase::Preparing);
        if (!prepare_raw_try_on_job(runtime, job, error_message))
            return false;
    }
    job.try_on_phase.store(TryOnPhase::Sampling);
    SDImageVec results;
    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        sd_image_t* raw = nullptr;
        int count       = 0;
        sd_try_on_callbacks_t callbacks{};
        callbacks.struct_size = sizeof(callbacks);
        callbacks.data        = &job;
        callbacks.cancelled   = [](void* data) {
            return static_cast<AsyncGenerationJob*>(data)->cancellation_requested.load();
        };
        callbacks.progress = [](int step, int, float, void* data) {
            static_cast<AsyncGenerationJob*>(data)->completed_steps.store(step);
        };
        bool ok = generate_try_on_with_callbacks(runtime.sd_ctx, &job.try_on.inputs.params, &callbacks, &raw, &count);
        results.adopt(raw, count);
        if (!ok || count == 0) {
            error_message = "generate_try_on failed or returned no images";
            return false;
        }
    }
    job.try_on_phase.store(TryOnPhase::Encoding);
    for (int i = 0; i < results.count(); ++i) {
        if (job.cancellation_requested.load()) {
            error_message = "try-on encoding cancelled";
            output_images.clear();
            return false;
        }
        auto bytes = encode_image_to_vector(EncodedImageFormat::PNG, results[i].data, results[i].width,
                                            results[i].height, results[i].channel,
                                            "FASHN VTON 1.5 prepared-input try_on", 100);
        if (bytes.empty()) {
            error_message = "failed to encode try-on PNG";
            output_images.clear();
            return false;
        }
        output_images.push_back(base64_encode(bytes));
        if (output_images.back().capacity() > SD_TRY_ON_RESULT_IMAGE_CAPACITY) {
            error_message = "encoded try-on image exceeds its reserved memory capacity";
            output_images.clear();
            return false;
        }
    }
    return true;
}
