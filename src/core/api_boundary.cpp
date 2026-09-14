#include "api_boundary.h"
#include "core/util.h"

void sd_boundary_error(const char* operation, const char* message) noexcept {
    try {
        LOG_ERROR("%s: %s", operation, message);
    } catch (...) {
        // A client logging callback must not defeat the C/thread exception boundary.
        std::fprintf(stderr, "%s: %s (logging callback threw)\n", operation, message);
    }
}
