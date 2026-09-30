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

// Same contract for a tensor in the repacked IQ4_XS layout. The 16-entry
// IQ4_NL codebook lives in the kernel (same values as format::kIq4Codebook).
bool gemv_iq4_xs(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
                 hipStream_t stream);

// Same contract for a tensor in the repacked IQ3_XXS layout (4-byte grid
// lookups, a 4-bit scale and four 7-bit sign indices per 32-weight sub-block).
bool gemv_iq3_xxs(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
                  hipStream_t stream);

} // namespace omph::kernels
