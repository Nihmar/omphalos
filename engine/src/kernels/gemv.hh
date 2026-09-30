// Fused dequant+dot GEMV for the repacked Q4_K layout (PLAN.md §10.1).
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace omph::kernels {

// y[r] = dot(w_row_r, x) for a Q4_K tensor in the layout of format/repack.hh.
//   packed: base of the repacked tensor
//   x_f16:  k activation values (k % 256 == 0), y: rows results
bool gemv_q4k(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
              hipStream_t stream);

} // namespace omph::kernels
