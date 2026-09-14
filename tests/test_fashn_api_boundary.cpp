#include <memory>
#include <stdexcept>

#include "core/api_boundary.h"
#include "stable-diffusion.h"

struct BoundaryLifetime {
    int& live;
    explicit BoundaryLifetime(int& count)
        : live(count) { ++live; }
    ~BoundaryLifetime() { --live; }
};

int main() {
    int live     = 0;
    int failures = 0;
    for (int phase = 0; phase < 3; ++phase) {
        bool result = sd_api_boundary("injected API failure", false, [&]() -> bool {
            auto owner = std::make_unique<BoundaryLifetime>(live);
            if (phase == 0) {
                throw std::bad_alloc();
            }
            if (phase == 1) {
                throw std::runtime_error("initialization/callback failure");
            }
            throw 42; }, [&]() noexcept { ++failures; });
        if (result || live != 0 || failures != phase + 1) {
            return 1;
        }
    }
    if (!sd_api_boundary("success", false, [] { return true; }) || new_sd_ctx(nullptr) != nullptr) {
        return 1;
    }
    free_sd_ctx(nullptr);
    return 0;
}
