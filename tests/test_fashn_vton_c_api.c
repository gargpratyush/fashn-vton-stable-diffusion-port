#include "stable-diffusion.h"

int main(void) {
    sd_try_on_params_t request;
    sd_try_on_params_init(&request);
    if (request.struct_size != sizeof(request) || request.category != SD_TRY_ON_TOPS ||
        request.steps != 30 || request.cfg != 1.5f || request.shift != 1.5f ||
        request.skip_cfg_last_n_steps != 1 || request.sample_count != 1 ||
        sd_ctx_supports_try_on(NULL)) {
        return 1;
    }
    sd_image_t* images = NULL;
    int count          = 9;
    if (generate_try_on(NULL, &request, &images, &count) || images != NULL || count != 0) {
        return 1;
    }
    free_sd_images(images, count);
    sd_try_on_callbacks_t callbacks = {sizeof(sd_try_on_callbacks_t), NULL, NULL, NULL};
    count                           = 9;
    if (generate_try_on_with_callbacks(NULL, &request, &callbacks, &images, &count) || images != NULL || count != 0) {
        return 1;
    }
    return 0;
}
