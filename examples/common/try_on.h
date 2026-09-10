#ifndef __SD_EXAMPLES_COMMON_TRY_ON_H__
#define __SD_EXAMPLES_COMMON_TRY_ON_H__

#include <array>
#include <string>

#include "json.hpp"
#include "resource_owners.hpp"

struct PreparedTryOnInputs {
    std::array<SDImageOwner, 4> images;
    sd_try_on_params_t params{};

    bool load_manifest(const std::string& path);
    bool load_embedded(const nlohmann::json& document);
    void bind_images();

private:
    bool load_document(const nlohmann::json& document, const std::string& directory, bool embedded);
};

#endif  // __SD_EXAMPLES_COMMON_TRY_ON_H__
