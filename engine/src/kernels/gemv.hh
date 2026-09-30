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

// Same contract for a tensor in the repacked IQ3_S layout (9-bit grid indices,
/// four sign-mask bytes per 32-weight sub-block, packed 4-bit scales).
bool gemv_iq3_s(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
                hipStream_t stream);

// Max resident workgroups per CU for each kernel (occupancy probe).
struct GemvOccupancy {
    int q4k = 0;
    int iq4 = 0;
    int iq3 = 0;
    int iq3s = 0;
    int block = 0;
    size_t smem_q4k = 0;
    size_t smem_iq4 = 0;
    size_t smem_iq3 = 0;
    size_t smem_iq3s = 0;
};
GemvOccupancy gemv_occupancy();

// Same contract for a tensor in the repacked Q2_K layout (2-bit quants, a 4-bit
// scale and a 4-bit min per 16-weight sub-block).
bool gemv_q2k(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
              hipStream_t stream);

// Same contract for a tensor in the repacked Q6_K layout (6-bit quants split
// over ql/qh, signed 8-bit scales per 16 weights).
bool gemv_q6k(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
              hipStream_t stream);

} // namespace omph::kernels
