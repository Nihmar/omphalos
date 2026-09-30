// Dequantization kernels — ggml quantized blocks to f32/f16 (PLAN.md §9, §10).
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace omph::kernels {

// Dequantizes `n` elements of ggml type `type` from `src` into `dst`.
// `dst` holds float when out_f16 == false, __half otherwise.
// Returns false for unsupported types or bad element counts.
bool dequantize(uint32_t type, const void * src, void * dst, int64_t n, bool out_f16,
                hipStream_t stream);

// Number of threads per block used by the dequant kernels.
constexpr int kDequantBlock = 256;

} // namespace omph::kernels
