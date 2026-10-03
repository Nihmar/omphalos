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

// Rows ids[0 .. n_rows) (device) of a quantized table of `type` (row_bytes per
// row, ne values; host-mapped memory, e.g. the token embedding) into dst as
// f32, n_rows x ne: one gather into `staging` (n_rows * row_bytes), then one
// dequantize (#223). The same values as dequantizing each row.
bool dequantize_rows(uint32_t type, const void * table, int64_t row_bytes, const int32_t * ids, int64_t n_rows,
                     int64_t ne, void * staging, float * dst, hipStream_t stream);

// Number of threads per block used by the dequant kernels.
constexpr int kDequantBlock = 256;

} // namespace omph::kernels
