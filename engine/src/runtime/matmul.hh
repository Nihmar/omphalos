// The f16 matmul of the dequant + GEMM path: f16 inputs, f32 accumulate.
#pragma once

#include <cstdint>

namespace omph::runtime {

// y(T, out) = x(T, in) @ W(out, in)^T, all row-major.
// W is f16 (dequantized weights), x is f16 (activations), y is f32 with rows
// ldy apart (0: out_features), so that W can come in slices of rows.
// The engine's own WMMA kernel (#132; it replaced hipBLASLt).
class Linear {
public:
    bool run(const void * w, const void * x, float * y, int64_t out_features,
             int64_t in_features, int64_t tokens, int64_t ldy = 0);
};

} // namespace omph::runtime
