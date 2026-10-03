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

// y = x @ w^T with w a whole repacked tensor (n_out x k, GGUF type `type`):
// the W tile is decoded into LDS as the GEMM goes, no f16 copy of w (#141).
// Bit-identical to dequant_repacked + gemm_f16. Needs k % 256 == 0 and the
// constraints of gemm_f16; false for a type without a repacked layout.
bool gemm_q(uint32_t type, const void * packed, const void * x, float * y, int64_t n_out,
            int64_t k, int64_t tokens, int64_t ldy, hipStream_t stream);

// gemm_q for an FFN's up projection with SwiGLU in its epilogue (#221):
// out_f16[t][r] = f16(silu(gate[t][r]) * up[t][r]), up never written; both
// T x n_out row-major (stride n_out). Bit-identical to gemm_q into up, then
// swiglu_f16(gate, up).
bool gemm_q_swiglu(uint32_t type, const void * packed, const void * x, const float * gate, void * out_f16,
                   int64_t n_out, int64_t k, int64_t tokens, hipStream_t stream);

} // namespace omph::kernels
