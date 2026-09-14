#ifndef __SD_CORE_API_BOUNDARY_H__
#define __SD_CORE_API_BOUNDARY_H__

#include <cstdio>
#include <exception>
#include <new>
#include <type_traits>
#include <utility>

void sd_boundary_error(const char* operation, const char* message) noexcept;

template <typename Result, typename Function, typename OnFailure>
Result sd_api_boundary(const char* operation, Result failure, Function&& function, OnFailure&& on_failure) noexcept {
    static_assert(std::is_nothrow_invocable_v<OnFailure&>, "Boundary cleanup must not throw");
    try {
        return function();
    } catch (const std::bad_alloc&) {
        sd_boundary_error(operation, "allocation failed");
    } catch (const std::exception& error) {
        sd_boundary_error(operation, error.what());
    } catch (...) {
        sd_boundary_error(operation, "unknown C++ exception");
    }
    on_failure();
    return failure;
}

template <typename Result, typename Function>
Result sd_api_boundary(const char* operation, Result failure, Function&& function) noexcept {
    return sd_api_boundary(operation, failure, std::forward<Function>(function), []() noexcept {});
}

#endif
