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

// Same contract for IQ2_XS and IQ2_XXS in their repacked layouts (8-byte grid
// entries, a 4-bit scale per 32-weight sub-block, sign indices resolved with the
// 128-byte table).
bool gemv_iq2_xs(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
                 hipStream_t stream);
bool gemv_iq2_xxs(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
                  hipStream_t stream);

// Same contract for a tensor in the repacked IQ2_S layout (10-bit grid indices
// into a 1024-entry grid, sign masks, 4-bit scales).
// IQ1_M on the GGUF blocks as they are (identity "repack").
bool gemv_iq1_m(const void * blocks, const void * x_f16, float * y, int64_t rows, int64_t k,
                hipStream_t stream);

bool gemv_iq2_s(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
                hipStream_t stream);

// Small-batch Q4_K: four tokens per weight read. `x_f16` holds four activation
// vectors back to back (stride k), `y` four result vectors (stride rows). This is
// what turns the prefill from "materialize f16" into "read the weights once per
// group of tokens", and it is the shape MTP verification needs.
bool gemv_q4k_b4(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
                 hipStream_t stream);

// Same contract as gemv_q4k_b4 for IQ3_S and IQ4_XS.
bool gemv_iq3s_b4(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
                  hipStream_t stream);
bool gemv_iq4_xs_b4(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
                    hipStream_t stream);
bool gemv_iq3_xxs_b4(const void * packed, const void * x_f16, float * y, int64_t rows, int64_t k,
                     hipStream_t stream);

// BF16 weights (type 30), single token: a plain dot product, no codebook.
bool gemv_bf16(const void * w, const void * x_f16, float * y, int64_t rows, int64_t k,
               hipStream_t stream);

} // namespace omph::kernels
