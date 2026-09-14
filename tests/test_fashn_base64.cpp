#include <iostream>

#include "examples/common/base64.h"

int main() {
    const std::vector<std::pair<std::string, std::string>> vectors = {
        {"", ""}, {"f", "Zg=="}, {"fo", "Zm8="}, {"foo", "Zm9v"}, {"foob", "Zm9vYg=="}, {"fooba", "Zm9vYmE="}, {"foobar", "Zm9vYmFy"}};
    for (const auto& entry : vectors) {
        if (sd_base64_encode({entry.first.begin(), entry.first.end()}) != entry.second) {
            return 1;
        }
    }
    if (sd_base64_encode({137, 80, 78, 71, 13, 10, 26, 10}) != "iVBORw0KGgo=" ||
        sd_base64_encode({255, 255, 255, 255, 255}) != "//////8=") {
        return 1;
    }
    std::vector<uint8_t> bytes;
    for (int i = 0; i < 256; ++i) {
        bytes.push_back(static_cast<uint8_t>(i));
    }
    std::cout << sd_base64_encode(bytes) << "\n";
    return 0;
}
