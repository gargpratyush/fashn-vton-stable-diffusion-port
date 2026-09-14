#include "fashn_try_on.h"

#include <chrono>
#include <memory>

#include "core/rng.hpp"
#include "model/diffusion/fashn_vton_model.h"
#include "runtime/fashn_vton_sampling.h"

static sd::Tensor<float> fashn_prepared_image(const sd_image_t& image) {
    sd::Tensor<float> tensor({image.width, image.height, image.channel, 1});
    const size_t pixels = size_t(image.width) * image.height;
    for (uint32_t c = 0; c < image.channel; ++c) {
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
            tensor.values()[c * pixels + pixel] = image.data[pixel * image.channel + c] / 127.5f - 1.f;
        }
    }
    return tensor;
}

bool generate_fashn_try_on(FashnTryOnRuntime& runtime, const sd_try_on_params_t* params, const sd_try_on_callbacks_t* callbacks, sd_image_t** images_out, int* num_images_out) {
    if (images_out != nullptr) {
        *images_out = nullptr;
    }
    if (num_images_out != nullptr) {
        *num_images_out = 0;
    }
    if (params == nullptr || images_out == nullptr || num_images_out == nullptr ||
        params->struct_size < sizeof(sd_try_on_params_t)) {
        LOG_ERROR("generate_try_on requires initialized parameters and output pointers");
        return false;
    }
    if (callbacks != nullptr && callbacks->struct_size < sizeof(sd_try_on_callbacks_t)) {
        LOG_ERROR("Invalid FASHN callback structure size");
        return false;
    }
    FashnVTONSamplingParams sampling{params->steps, params->cfg, params->shift, params->skip_cfg_last_n_steps};
    if (!sampling.validate() || params->category < SD_TRY_ON_TOPS || params->category > SD_TRY_ON_ONE_PIECES ||
        params->sample_count < 1 || params->sample_count > 4) {
        LOG_ERROR("Invalid FASHN sampling/category/sample count (supported count: 1..4)");
        return false;
    }
    const FashnVTONConfig config;
    const int width            = config.input_width;
    const int height           = config.input_height;
    const sd_image_t* images[] = {&params->ca_image, &params->garment_image, &params->person_pose, &params->garment_pose};
    for (int i = 0; i < 4; ++i) {
        if (images[i]->data == nullptr || images[i]->width != static_cast<uint32_t>(width) ||
            images[i]->height != static_cast<uint32_t>(height) ||
            images[i]->channel != static_cast<uint32_t>(i < 2 ? 3 : 1)) {
            LOG_ERROR("FASHN prepared inputs must be 576x864, RGB/RGB/grayscale/grayscale");
            return false;
        }
    }
    int crop_w = params->crop_width;
    int crop_h = params->crop_height;
    if (crop_w == 0 && crop_h == 0 && params->crop_x == 0 && params->crop_y == 0) {
        crop_w = width;
        crop_h = height;
    }
    if (crop_w < 1 || crop_h < 1 || crop_w > width || crop_h > height ||
        params->crop_x < 0 || params->crop_y < 0 ||
        params->crop_x > width - crop_w || params->crop_y > height - crop_h) {
        LOG_ERROR("FASHN crop must be a positive rectangle inside the prepared canvas");
        return false;
    }
    runtime.reset_cancellation();
    auto is_cancelled = [&]() {
        return runtime.cancel_mode() == SD_CANCEL_ALL ||
               (callbacks != nullptr && callbacks->cancelled != nullptr && callbacks->cancelled(callbacks->data));
    };
    if (is_cancelled()) {
        LOG_INFO("FASHN request cancelled before preparation");
        return false;
    }
    auto ca           = fashn_prepared_image(params->ca_image);
    auto garment      = fashn_prepared_image(params->garment_image);
    auto pose         = fashn_prepared_image(params->person_pose);
    auto garment_pose = fashn_prepared_image(params->garment_pose);
    sd::Tensor<int32_t> category({1}, {static_cast<int32_t>(params->category)});
    FashnVTONDiffusionExtra condition{&ca, &garment, &pose, &garment_pose, &category};
    runtime.rng.manual_seed(params->seed);
    const size_t per_sample = size_t(width) * height * 3;
    auto noise              = runtime.rng.randn(static_cast<uint32_t>(per_sample * params->sample_count));
    auto* cached_runner     = runtime.modulation_cache ? dynamic_cast<FashnVTONRunner*>(&runtime.model) : nullptr;
    struct EndModulationRequest {
        FashnVTONRunner* runner;
        ~EndModulationRequest() {
            if (runner != nullptr) {
                runner->finish_modulation_request();
            }
        }
    } end_modulation_request{cached_runner};
    if (runtime.modulation_cache) {
        if (cached_runner == nullptr) {
            LOG_ERROR("FASHN modulation caching requires the FASHN runner");
            return false;
        }
        auto pairs = fashn_vton_modulation_pairs(sampling, static_cast<int>(params->category));
        if (pairs.empty() || !cached_runner->prepare_modulation_request(
                                 runtime.threads, pairs, runtime.reset_weights, runtime.modulation_budget, is_cancelled)) {
            return false;
        }
    }
    std::vector<sd::Tensor<float>> completed;
    for (int sample = 0; sample < params->sample_count; ++sample) {
        if (runtime.cancel_mode() == SD_CANCEL_NEW_LATENTS) {
            break;
        }
        sd::Tensor<float> initial({width, height, 3, 1},
                                  std::vector<float>(noise.begin() + sample * per_sample,
                                                     noise.begin() + (sample + 1) * per_sample));
        auto start  = std::chrono::steady_clock::now();
        auto output = sample_fashn_vton(runtime.model, runtime.threads, initial, condition, sampling,
                                        is_cancelled, [&](int step, int total) {
                                            auto cb = callbacks != nullptr ? callbacks->progress : sd_get_progress_callback();
                                            if (cb != nullptr) {
                                                float seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
                                                cb(sample * total + step, params->sample_count * total, seconds / step,
                                                   callbacks != nullptr ? callbacks->data : sd_get_progress_callback_data());
                                            }
                                        });
        if (output.empty() || is_cancelled()) {
            return false;
        }
        completed.push_back(std::move(output));
    }
    if (completed.empty() || is_cancelled()) {
        LOG_INFO("FASHN request cancelled before any samples completed");
        return false;
    }
    auto free_results = [&](sd_image_t* images) {
        free_sd_images(images, static_cast<int>(completed.size()));
    };
    std::unique_ptr<sd_image_t, decltype(free_results)> result_owner(
        static_cast<sd_image_t*>(calloc(completed.size(), sizeof(sd_image_t))), free_results);
    auto* results = result_owner.get();
    if (results == nullptr) {
        LOG_ERROR("Cannot allocate FASHN result array");
        return false;
    }
    for (size_t sample = 0; sample < completed.size(); ++sample) {
        results[sample]      = {static_cast<uint32_t>(crop_w), static_cast<uint32_t>(crop_h), 3, nullptr};
        results[sample].data = static_cast<uint8_t*>(malloc(static_cast<size_t>(crop_w) * crop_h * 3));
        if (results[sample].data == nullptr) {
            LOG_ERROR("Cannot allocate FASHN RGB output");
            return false;
        }
        for (int row = 0; row < crop_h; ++row) {
            for (int col = 0; col < crop_w; ++col) {
                for (int c = 0; c < 3; ++c) {
                    size_t source                                      = size_t(c) * width * height + (row + params->crop_y) * width + col + params->crop_x;
                    float pixel                                        = std::clamp((completed[sample].values()[source] + 1.f) * 0.5f, 0.f, 1.f);
                    results[sample].data[(row * crop_w + col) * 3 + c] = static_cast<uint8_t>(pixel * 255.f);
                }
            }
        }
    }
    if (is_cancelled()) {
        LOG_INFO("FASHN request cancelled during output conversion");
        return false;
    }
    *images_out     = result_owner.release();
    *num_images_out = static_cast<int>(completed.size());
    return true;
}
