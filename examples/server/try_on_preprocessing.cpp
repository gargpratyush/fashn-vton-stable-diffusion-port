#include "async_jobs.h"
#include "common/log.h"

#ifdef SD_FASHN_PREPROCESS
#include <cstring>
#include <opencv2/imgproc.hpp>
#include "pipeline.h"
#endif

bool initialize_try_on_preparer(ServerRuntime& runtime, std::string& error_message) {
    const auto& options = *runtime.svr_params;
    if (options.try_on_dwpose_dir.empty())
        return true;
    if (!sd_ctx_supports_try_on(runtime.sd_ctx)) {
        error_message = "Native try-on preparation requires a FASHN model";
        return false;
    }
#ifdef SD_FASHN_PREPROCESS
    try {
        fashn_prepare::CPUOptions cpu;
        cpu.arena = !options.try_on_ort_no_arena;
        runtime.try_on_preparer = std::make_shared<fashn_prepare::Preparer>(
            options.try_on_dwpose_dir, options.try_on_parser_dir, options.accept_parser_research_license, true, cpu);
        runtime.try_on_parser_enabled = runtime.try_on_preparer->has_parser();
        if (runtime.try_on_parser_enabled)
            LOG_WARN("RESEARCH/EVALUATION ONLY: restricted native human parser enabled");
        return true;
    } catch (const std::exception& error) {
        error_message = std::string("Native try-on initialization failed: ") + error.what();
        return false;
    }
#else
    error_message = "Raw try-on requires a build with SD_FASHN_PREPROCESS=ON";
    return false;
#endif
}

bool prepare_raw_try_on_job(ServerRuntime& runtime, AsyncGenerationJob& job, std::string& error_message) {
#ifdef SD_FASHN_PREPROCESS
    try {
        if (!runtime.try_on_preparer)
            throw std::runtime_error("Native preparation is not configured");
        std::array<SDImageOwner, 2> decoded;
        std::array<cv::Mat, 2> bgr;
        for (size_t i = 0; i < decoded.size(); ++i) {
            if (!decode_base64_image(job.try_on.raw_images[i], 3, 0, 0, decoded[i], true, SD_RAW_IMAGE_MAX_PIXELS))
                throw std::runtime_error("Cannot decode bounded raw PNG");
            auto image = decoded[i].get();
            cv::Mat rgb(image.height, image.width, CV_8UC3, image.data);
            cv::cvtColor(rgb, bgr[i], cv::COLOR_RGB2BGR);
        }
        job.try_on.release_raw_images();
        auto prepared         = runtime.try_on_preparer->prepare(
            bgr[0], bgr[1], job.try_on.category, job.try_on.garment_photo_type,
            job.try_on.segmentation_free, [&] { return job.cancellation_requested.load(); });
        for (size_t i = 0; i < prepared.images.size(); ++i) {
            cv::Mat pixels = prepared.images[i];
            if (i < 2)
                cv::cvtColor(prepared.images[i], pixels, cv::COLOR_BGR2RGB);
            size_t row_bytes = pixels.cols * pixels.elemSize();
            auto data        = static_cast<uint8_t*>(malloc(row_bytes * pixels.rows));
            if (!data)
                throw std::bad_alloc();
            job.try_on.inputs.images[i].reset({uint32_t(pixels.cols), uint32_t(pixels.rows),
                                               uint32_t(pixels.channels()), data});
            for (int row = 0; row < pixels.rows; ++row)
                std::memcpy(data + row * row_bytes, pixels.ptr(row), row_bytes);
        }
        auto& params       = job.try_on.inputs.params;
        params.crop_x      = prepared.crop.x;
        params.crop_y      = prepared.crop.y;
        params.crop_width  = prepared.crop.width;
        params.crop_height = prepared.crop.height;
        job.try_on.inputs.bind_images();
        return true;
    } catch (const std::exception& error) {
        error_message = std::string("Native preparation failed: ") + error.what();
        LOG_ERROR("%s", error_message.c_str());
        return false;
    }
#else
    error_message = "Native preparation is unavailable in this build";
    return false;
#endif
}
