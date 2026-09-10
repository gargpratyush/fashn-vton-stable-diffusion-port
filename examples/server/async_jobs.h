#pragma once

#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>


#include "runtime.h"

enum class AsyncJobKind {
    ImgGen,
    VidGen,
    TryOn,
};

enum class AsyncJobStatus {
    Queued,
    Generating,
    Completed,
    Failed,
    Cancelled,
};

enum class TryOnPhase { Queued, Preparing, Sampling, Encoding, Done };
constexpr size_t SD_TRY_ON_RESULT_IMAGE_CAPACITY = size_t(4) * 1024 * 1024;

const char* async_job_kind_name(AsyncJobKind kind);
const char* async_job_status_name(AsyncJobStatus status);

struct AsyncGenerationJob {
    std::string id;
    AsyncJobKind kind     = AsyncJobKind::ImgGen;
    AsyncJobStatus status = AsyncJobStatus::Queued;
    int64_t created_at    = unix_timestamp_now();
    int64_t started_at    = 0;
    int64_t completed_at  = 0;
    ImgGenJobRequest img_gen;
    VidGenJobRequest vid_gen;
    TryOnJobRequest try_on;
    std::atomic<bool> cancellation_requested{false};
    std::atomic<int> completed_steps{0};
    std::atomic<TryOnPhase> try_on_phase{TryOnPhase::Queued};
    int total_steps = 0;
    size_t try_on_image_charge = 0;
    std::vector<std::string> result_images_b64;
    std::string result_media_b64;
    std::string result_media_mime_type;
    int result_frame_count = 0;
    int result_fps         = 0;
    std::string error_code;
    std::string error_message;
};

std::string make_async_instance_id();

struct AsyncJobManager {
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<std::string, std::shared_ptr<AsyncGenerationJob>> jobs;
    std::unordered_map<std::string, int64_t> expired_jobs;
    std::deque<std::string> queue;
    uint64_t next_id              = 0;
    const std::string instance_id = make_async_instance_id();
    bool stop                     = false;
    size_t max_pending_jobs       = 64;
    size_t max_try_on_image_bytes = size_t(512) * 1024 * 1024;
    int64_t completed_ttl_seconds = 600;
    int64_t failed_ttl_seconds    = 600;
};

void purge_expired_jobs(AsyncJobManager& manager);
size_t count_pending_jobs(const AsyncJobManager& manager);
// Manager lock required; active jobs retain their admission reservation until completion.
size_t reserved_try_on_image_bytes(const AsyncJobManager& manager);
bool reserve_try_on_images(const AsyncJobManager& manager, AsyncGenerationJob& job);
std::string make_async_job_id(AsyncJobManager& manager);
bool cancel_queued_job(AsyncJobManager& manager, AsyncGenerationJob& job);
json make_async_job_json(const AsyncJobManager& manager, const AsyncGenerationJob& job);
bool execute_img_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::vector<std::string>& output_images,
                         std::string& error_message);
bool execute_vid_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::string& output_media_b64,
                         std::string& output_media_mime_type,
                         int& output_frame_count,
                         int& output_fps,
                         std::string& error_message);
void async_job_worker(ServerRuntime& runtime);
void stop_try_on_jobs(AsyncJobManager& manager);
bool execute_try_on_job(ServerRuntime& runtime, AsyncGenerationJob& job, std::vector<std::string>& output_images, std::string& error_message);
bool prepare_raw_try_on_job(ServerRuntime& runtime, AsyncGenerationJob& job, std::string& error_message);
