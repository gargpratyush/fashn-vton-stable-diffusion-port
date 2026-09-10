#ifndef __SD_MODEL_DIFFUSION_FASHN_MODULATION_H__
#define __SD_MODEL_DIFFUSION_FASHN_MODULATION_H__

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

struct FashnModulationLayer {
    std::string block;
    int stream;
    int width;
};

inline const std::vector<FashnModulationLayer>& fashn_modulation_layers() {
    static const auto layers = [] {
        std::vector<FashnModulationLayer> result;
        for (int i = 0; i < 4; ++i)
            result.push_back({"x_patch_mixer." + std::to_string(i), -1, 3840});
        for (int i = 0; i < 8; ++i) {
            result.push_back({"double_blocks." + std::to_string(i), 0, 7680});
            result.push_back({"double_blocks." + std::to_string(i), 1, 7680});
        }
        for (int i = 0; i < 16; ++i)
            result.push_back({"single_blocks." + std::to_string(i), -1, 3840});
        result.push_back({"final_layer", 2, 2560});
        return result;
    }();
    return layers;
}

inline size_t fashn_modulation_values() {
    size_t result = 3 * 1280;  // Preserve time, category and combined-vector captures too.
    for (const auto& layer : fashn_modulation_layers())
        result += layer.width;
    return result;
}

inline std::pair<uint32_t, int> fashn_modulation_key(float time, int category) {
    uint32_t bits;
    std::memcpy(&bits, &time, sizeof(bits));
    return {bits, category};
}

#endif
