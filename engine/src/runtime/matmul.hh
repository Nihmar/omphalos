// hipBLASLt wrapper for the naive path: f16 inputs, f32 accumulate.
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

namespace omph::runtime {

// y(T, out) = x(T, in) @ W(out, in)^T, all row-major.
// W is f16 (dequantized weights), x is f16 (activations), y is f32.
class Linear {
public:
    Linear();
    ~Linear();
    Linear(const Linear &) = delete;
    Linear & operator=(const Linear &) = delete;

    bool run(const void * w, const void * x, float * y, int64_t out_features,
             int64_t in_features, int64_t tokens);

private:
    hipblasLtHandle_t handle_ = nullptr;
    void * workspace_ = nullptr;
    std::size_t workspace_size_ = 0;
};

} // namespace omph::runtime
