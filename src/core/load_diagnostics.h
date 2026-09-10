#ifndef __SD_CORE_LOAD_DIAGNOSTICS_H__
#define __SD_CORE_LOAD_DIAGNOSTICS_H__

#include <atomic>
#include <cstdint>
#include <functional>

struct LoadDiagnostics {
    std::atomic<uint64_t> read_live{0}, read_peak{0};
    std::atomic<uint64_t> f32_live{0}, f32_peak{0};
    std::atomic<uint64_t> scratch_live{0}, scratch_peak{0};
    std::atomic<uint64_t> converted_tensors{0};
    // Called on the loading thread, potentially while workers run; observers must not throw or mutate the loader.
    std::function<void(const char*)> event;

    static void update_peak(std::atomic<uint64_t>& peak, uint64_t value) {
        auto previous = peak.load(std::memory_order_relaxed);
        while (previous < value &&
               !peak.compare_exchange_weak(previous, value, std::memory_order_relaxed)) {}
    }
};

class ScopedLoadScratch {
    LoadDiagnostics* diagnostics_;
    bool f32_;
    uint64_t bytes_ = 0;

public:
    ScopedLoadScratch(LoadDiagnostics* diagnostics, bool f32)
        : diagnostics_(diagnostics), f32_(f32) {}
    ScopedLoadScratch(const ScopedLoadScratch&) = delete;
    ScopedLoadScratch& operator=(const ScopedLoadScratch&) = delete;
    ~ScopedLoadScratch() { set(0); }

    void set(uint64_t bytes) {
        if (diagnostics_ == nullptr)
            return;
        auto& live = f32_ ? diagnostics_->f32_live : diagnostics_->read_live;
        auto& peak = f32_ ? diagnostics_->f32_peak : diagnostics_->read_peak;
        if (bytes >= bytes_) {
            auto delta = bytes - bytes_;
            LoadDiagnostics::update_peak(peak, live.fetch_add(delta) + delta);
            LoadDiagnostics::update_peak(diagnostics_->scratch_peak,
                                        diagnostics_->scratch_live.fetch_add(delta) + delta);
        } else {
            live.fetch_sub(bytes_ - bytes);
            diagnostics_->scratch_live.fetch_sub(bytes_ - bytes);
        }
        bytes_ = bytes;
    }
};

#endif
