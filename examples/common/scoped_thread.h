#ifndef __SD_EXAMPLES_COMMON_SCOPED_THREAD_H__
#define __SD_EXAMPLES_COMMON_SCOPED_THREAD_H__

#include <functional>
#include <thread>
#include <utility>

class ScopedThread {
    std::function<void()> stop_;
    std::thread thread_;

public:
    template <typename Work, typename Stop>
    ScopedThread(Work&& work, Stop&& stop)
        : stop_(std::forward<Stop>(stop)), thread_(std::forward<Work>(work)) {}
    ScopedThread(const ScopedThread&)            = delete;
    ScopedThread& operator=(const ScopedThread&) = delete;
    ~ScopedThread() {
        finish();
    }
    void finish() {
        if (thread_.joinable()) {
            stop_();
            thread_.join();
        }
    }
};

#endif
