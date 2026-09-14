#ifndef __SD_RUNTIME_FASHN_TRY_ON_H__
#define __SD_RUNTIME_FASHN_TRY_ON_H__

#include <functional>

#include "stable-diffusion.h"

class RNG;
struct DiffusionModelRunner;
struct FashnVTONRunner;

struct FashnTryOnRuntime {
    DiffusionModelRunner& model;
    RNG& rng;
    int threads;
    bool modulation_cache;
    size_t modulation_budget;
    std::function<void()> reset_cancellation;
    std::function<sd_cancel_mode_t()> cancel_mode;
    std::function<bool()> reset_weights;
};

bool generate_fashn_try_on(FashnTryOnRuntime& runtime, const sd_try_on_params_t* params, const sd_try_on_callbacks_t* callbacks, sd_image_t** images_out, int* num_images_out);

#endif
