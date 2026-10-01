// f16 GEMM on the WMMA units (#132): the prefill's matmuls.
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace omph::kernels {

// y(T, n_out) = x(T, k) @ w(n_out, k)^T: w and x f16 row-major, y f32 with rows
// ldy floats apart (a slice of w's rows writes its columns of a wider y).
// f32 accumulation in a fixed order: deterministic. Needs k % 64 == 0,
// n_out % 16 == 0, ldy % 4 == 0 and 16-byte aligned pointers; false otherwise.
bool gemm_f16(const void * w, const void * x, float * y, int64_t n_out, int64_t k, int64_t tokens,
              int64_t ldy, hipStream_t stream);

} // namespace omph::kernels
