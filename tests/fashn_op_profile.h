#ifndef __SD_TESTS_FASHN_OP_PROFILE_H__
#define __SD_TESTS_FASHN_OP_PROFILE_H__

#include <chrono>
#include <map>
#include "ggml.h"
#include "json.hpp"
#include "stable-diffusion.h"

struct FashnOpProfile {
    struct Row {
        int64_t calls        = 0;
        double seconds       = 0;
        int64_t output_bytes = 0;
    };
    std::map<std::string, Row> rows;
    std::chrono::steady_clock::time_point start;

    static bool callback(ggml_tensor* tensor, bool ask, void* data) {
        auto& profile = *static_cast<FashnOpProfile*>(data);
        if (ask) {
            profile.start = std::chrono::steady_clock::now();
        } else {
            auto& row = profile.rows[ggml_op_desc(tensor)];
            ++row.calls;
            row.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - profile.start).count();
            row.output_bytes += ggml_nbytes(tensor);
        }
        return true;
    }
    nlohmann::ordered_json report() const {
        nlohmann::ordered_json result;
        result["measurement"] = "Intrusive per-node synchronized graph execution; disables cross-node fusion and includes dispatch overhead. Not an end-to-end latency benchmark.";
        result["operations"]  = nlohmann::ordered_json::object();
        for (const auto& [name, row] : rows)
            result["operations"][name] = {{"calls", row.calls}, {"seconds", row.seconds}, {"output_bytes", row.output_bytes}};
        return result;
    }
};
#endif
