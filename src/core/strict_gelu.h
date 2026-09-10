#ifndef __SD_CORE_STRICT_GELU_H__
#define __SD_CORE_STRICT_GELU_H__

#include <cmath>
#include "ggml.h"

#if defined(_MSC_VER) && _MSC_VER >= 1930
#pragma float_control(precise, on, push)
#endif
inline float sd_strict_gelu_value(float x) {
    // VS2022's precise scope also disables contraction; other compilers get explicit F32 rounding barriers.
#if defined(_MSC_VER) && _MSC_VER >= 1930
    using Rounded = const float;
#else
    using Rounded = volatile float;
#endif
    Rounded square = x * x;
    Rounded cubic = square * x;
    Rounded scaled = cubic * 0.044715f;
    Rounded inner = x + scaled;
    Rounded argument = inner * 0.7978845608028654f;
    Rounded half_tanh = std::tanh(argument) * .5f;
    Rounded gate = half_tanh + .5f;
    return x * gate;
}
#if defined(_MSC_VER) && _MSC_VER >= 1930
#pragma float_control(pop)
#endif

inline void sd_strict_gelu_compute(ggml_tensor* dst, const ggml_tensor* src, int ith, int nth, void*) {
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    const int64_t rows = ggml_nrows(src);
    auto out = static_cast<float*>(dst->data);
    for (int64_t row = ith; row < rows; row += nth) {
        const auto i1 = row % src->ne[1];
        const auto i2 = row / src->ne[1] % src->ne[2];
        const auto i3 = row / (src->ne[1] * src->ne[2]);
        const auto data = static_cast<const char*>(src->data) + i1 * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3];
        for (int64_t column = 0; column < src->ne[0]; ++column)
            out[row * src->ne[0] + column] = sd_strict_gelu_value(*reinterpret_cast<const float*>(data + column * src->nb[0]));
    }
}

inline ggml_tensor* sd_strict_gelu(ggml_context* ctx, ggml_tensor* input) {
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    return ggml_map_custom1(ctx, input, sd_strict_gelu_compute, GGML_N_TASKS_MAX, nullptr);
}

#endif
