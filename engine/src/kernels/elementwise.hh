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

} // namespace omph::kernels
