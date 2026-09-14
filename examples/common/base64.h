#ifndef __SD_EXAMPLES_COMMON_BASE64_H__
#define __SD_EXAMPLES_COMMON_BASE64_H__

#include <cstdint>
#include <string>
#include <vector>

inline std::string sd_base64_encode(const std::vector<uint8_t>& bytes) {
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    uint32_t value = 0;
    int bits       = -6;
    for (uint8_t byte : bytes) {
        value = (value << 8) | byte;
        bits += 8;
        while (bits >= 0) {
            result += alphabet[(value >> bits) & 63];
            bits -= 6;
        }
    }
    if (bits > -6) {
        result += alphabet[((value << 8) >> (bits + 8)) & 63];
    }
    while (result.size() % 4) {
        result += '=';
    }
    return result;
}

#endif
