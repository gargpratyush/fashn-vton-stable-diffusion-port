#include "try_on.h"

#include <filesystem>
#include <fstream>
#include <set>

#include "common.h"
#include "json.hpp"
#include "log.h"
#include "media_io.h"

bool PreparedTryOnInputs::load_manifest(const std::string& path) {
    sd_try_on_params_init(&params);
    std::ifstream input(path);
    if (!input) {
        LOG_ERROR("Cannot open prepared try-on manifest '%s'", path.c_str());
        return false;
    }
    auto document = nlohmann::json::parse(input, nullptr, false);
    return load_document(document, std::filesystem::path(path).parent_path().string(), false);
}

bool PreparedTryOnInputs::load_embedded(const nlohmann::json& document) {
    return load_document(document, "", true);
}

bool PreparedTryOnInputs::load_document(const nlohmann::json& document, const std::string& directory, bool embedded) {
    sd_try_on_params_init(&params);
    if (!document.is_object() || !document.contains("schema") ||
        document["schema"] != "fashn-vton-prepared-v1") {
        LOG_ERROR("Expected a JSON object with schema 'fashn-vton-prepared-v1'");
        return false;
    }
    const std::set<std::string> allowed = {"schema", "ca_image", "garment_image", "person_pose", "garment_pose", "category", "crop"};
    for (const auto& item : document.items()) {
        if (allowed.count(item.key()) == 0) {
            LOG_ERROR("Unknown prepared try-on manifest key '%s'", item.key().c_str());
            return false;
        }
    }
    if (!document.contains("category") || !document["category"].is_string()) {
        LOG_ERROR("Prepared try-on manifest requires a garment category");
        return false;
    }
    const std::string category = document["category"].get<std::string>();
    if (category == "tops") {
        params.category = SD_TRY_ON_TOPS;
    } else if (category == "bottoms") {
        params.category = SD_TRY_ON_BOTTOMS;
    } else if (category == "one-pieces") {
        params.category = SD_TRY_ON_ONE_PIECES;
    } else {
        LOG_ERROR("Try-on category must be tops, bottoms, or one-pieces");
        return false;
    }
    if (document.contains("crop")) {
        const auto& crop    = document["crop"];
        const char* names[] = {"x", "y", "width", "height"};
        int* values[]       = {&params.crop_x, &params.crop_y, &params.crop_width, &params.crop_height};
        if (!crop.is_object() || crop.size() != 4) {
            LOG_ERROR("Try-on crop requires x, y, width, and height");
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            if (!crop.contains(names[i]) || !crop[names[i]].is_number_integer() ||
                crop[names[i]].get<double>() < 0 || crop[names[i]].get<double>() > 864) {
                LOG_ERROR("Invalid integer crop field '%s'", names[i]);
                return false;
            }
            *values[i] = crop[names[i]].get<int>();
        }
        if (params.crop_width < 1 || params.crop_height < 1 || params.crop_width > 576 ||
            params.crop_height > 864 || params.crop_x > 576 - params.crop_width ||
            params.crop_y > 864 - params.crop_height) {
            LOG_ERROR("Try-on crop is outside the 576x864 canvas");
            return false;
        }
    }
    const char* keys[] = {"ca_image", "garment_image", "person_pose", "garment_pose"};
    for (int i = 0; i < 4; ++i) {
        if (!document.contains(keys[i]) || !document[keys[i]].is_string() || document[keys[i]].get<std::string>().empty()) {
            LOG_ERROR("Missing prepared image path '%s'", keys[i]);
            return false;
        }
        if (embedded) {
            if (!decode_base64_image(document[keys[i]].get<std::string>(), i < 2 ? 3 : 1, 576, 864, images[i], true)) {
                LOG_ERROR("Invalid embedded prepared PNG '%s'", keys[i]);
                return false;
            }
            continue;
        }
        auto image_path = (std::filesystem::path(directory) / document[keys[i]].get<std::string>()).string();
        int w = 0, h = 0, channels = 0;
        int required_channels = i < 2 ? 3 : 1;
        if (!get_u8_image_info_from_file(image_path.c_str(), w, h, channels) ||
            w != 576 || h != 864 || channels != required_channels) {
            LOG_ERROR("Prepared '%s' must be 576x864 with exactly %d channels; no resizing or color-pose conversion is performed",
                      keys[i], required_channels);
            return false;
        }
        if (!load_sd_image_from_file(images[i].put(), image_path.c_str(), 0, 0, required_channels)) {
            LOG_ERROR("Failed to decode prepared image '%s'", keys[i]);
            return false;
        }
    }
    bind_images();
    return true;
}

void PreparedTryOnInputs::bind_images() {
    params.ca_image      = images[0].get();
    params.garment_image = images[1].get();
    params.person_pose   = images[2].get();
    params.garment_pose  = images[3].get();
}
