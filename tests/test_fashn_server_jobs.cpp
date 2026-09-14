#include <atomic>
#include <stdexcept>

#include "examples/common/scoped_thread.h"
#include "examples/server/async_jobs.h"

int main() {
    {
        AsyncJobManager jobs;
        std::mutex mutex;
        ServerRuntime runtime{};
        runtime.sd_ctx_mutex      = &mutex;
        runtime.async_job_manager = &jobs;
        try {
            ServerGenerationLock lock(runtime);
            throw std::runtime_error("native context failure");
        } catch (const std::runtime_error&) {
        }
        if (!jobs.worker_failed.load()) {
            return 1;
        }
        try {
            ServerGenerationLock lock(runtime);
            return 1;
        } catch (const std::runtime_error&) {
        }
    }
    std::atomic<bool> stop{false};
    std::atomic<bool> finished{false};
    try {
        ScopedThread worker([&] {
            while (!stop.load()) {
                std::this_thread::yield();
            }
            finished.store(true); }, [&] { stop.store(true); });
        throw std::runtime_error("server setup failure");
    } catch (const std::runtime_error&) {
    }
    if (!finished.load()) {
        return 1;
    }
    AsyncJobManager manager;
    for (int i = 0; i < 3; ++i) {
        auto job                  = std::make_shared<AsyncGenerationJob>();
        job->id                   = std::to_string(i);
        job->kind                 = AsyncJobKind::TryOn;
        job->status               = i == 0 ? AsyncJobStatus::Generating : AsyncJobStatus::Queued;
        job->try_on.raw_images[0] = std::string(1024, 'x');
        job->try_on_image_charge  = 1024;
        job->cancellation_requested.store(i == 2);
        manager.jobs.emplace(job->id, job);
        manager.queue.push_back(job->id);
    }
    fail_async_worker(manager);
    if (!manager.stop || !manager.worker_failed || !manager.queue.empty() ||
        reserved_try_on_image_bytes(manager) != 0) {
        return 1;
    }
    for (const auto& entry : manager.jobs) {
        const auto& job = *entry.second;
        if (job.completed_at == 0 || job.try_on.raw_images[0].size() != 0 ||
            job.status != (job.id == "2" ? AsyncJobStatus::Cancelled : AsyncJobStatus::Failed)) {
            return 1;
        }
        if (job.id != "2" && make_async_job_json(manager, job)["error"]["code"] != "worker_failed") {
            return 1;
        }
    }
    for (const std::string phase : {"execute", "finalize", "none"}) {
        AsyncJobManager jobs;
        ServerRuntime runtime{};
        runtime.async_job_manager = &jobs;
        jobs.stop                 = true;
        jobs.fault_injection      = [&](const char* reached) {
            if (phase == reached) {
                throw std::bad_alloc();
            }
        };
        for (int i = 0; i < 2; ++i) {
            auto job  = std::make_shared<AsyncGenerationJob>();
            job->id   = std::to_string(i);
            job->kind = static_cast<AsyncJobKind>(-1);
            jobs.jobs.emplace(job->id, job);
            jobs.queue.push_back(job->id);
        }
        async_job_worker(runtime);
        if (jobs.worker_failed.load() != (phase != "none") || !jobs.queue.empty()) {
            return 1;
        }
        for (const auto& entry : jobs.jobs) {
            if (entry.second->status != AsyncJobStatus::Failed || entry.second->completed_at == 0) {
                return 1;
            }
        }
    }
    return 0;
}
