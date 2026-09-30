// Elementwise / reduction kernels for the naive path (PLAN.md §10.6).
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace omph::kernels {

// dst[i] = (half)src[i]
bool cast_f32_to_f16(const float * src, void * dst, int64_t n, hipStream_t stream);

// y[row, :] = x[row, :] / sqrt(mean(x^2) + eps) * w[:] * scale
bool rms_norm(const float * x, const float * w, float * y, int64_t rows, int64_t n, float eps,
              float scale, hipStream_t stream);

// x[i] *= y[i] and x[i] += y[i], in place; out[i] = x[i] + y[i].
bool mul_inplace(float * x, const float * y, int64_t n, hipStream_t stream);
bool add_inplace(float * x, const float * y, int64_t n, hipStream_t stream);
bool add_out(const float * x, const float * y, float * out, int64_t n, hipStream_t stream);

// y (f16) = rms_norm(a + b) * w per row of n; b and sum_out may be null, else
// sum_out (f32) receives a + b, the residual stream. n % 4 == 0, n <= 8192.
bool add_rms_norm_f16(const float * a, const float * b, float * sum_out, const float * w,
                      void * y_f16, int64_t rows, int64_t n, float eps, hipStream_t stream);

// out[i] = silu(x[i]) * y[i], written as f16 — the SwiGLU epilogue feeding the
// down projection.
bool swiglu_f16(const float * x, const float * y, void * out_f16, int64_t n, hipStream_t stream);

} // namespace omph::kernels
