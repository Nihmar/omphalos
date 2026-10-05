// Elementwise / reduction kernels for the naive path (PLAN.md §10.6).
#pragma once

#include <cstdint>

#include "kernels/argmax_key.hh"

#include <hip/hip_runtime.h>

namespace omph::kernels {

// dst[i] = (half)src[i]
bool cast_f32_to_f16(const float * src, void * dst, int64_t n, hipStream_t stream);

// y[row, :] = x[row, :] / sqrt(mean(x^2) + eps) * w[:] * scale
bool rms_norm(const float * x, const float * w, float * y, int64_t rows, int64_t n, float eps,
              float scale, hipStream_t stream);

// x[i] += y[i], in place; out[i] = x[i] + y[i].
bool add_inplace(float * x, const float * y, int64_t n, hipStream_t stream);
bool add_out(const float * x, const float * y, float * out, int64_t n, hipStream_t stream);

// y (f16) = rms_norm(a + b) * w per row of n; b and sum_out may be null, else
// sum_out (f32) receives a + b, the residual stream. n % 4 == 0, n <= 8192.
bool add_rms_norm_f16(const float * a, const float * b, float * sum_out, const float * w,
                      void * y_f16, int64_t rows, int64_t n, float eps, hipStream_t stream);

// Greedy argmax of x[0..n) on the device (#102): *key = max over i of
// argmax_key_pack(x[i], i) (kernels/argmax_key.hh), so the token is
// argmax_key_index(*key) — the lowest index among equal maxima, as the host
// argmax_finite picks it (-0.0 counts as +0.0; a non-finite value never wins,
// and a row with none gives index 0, #318/#335). `key` (8 bytes, device) is
// cleared and written on `stream`; n < 2^32.
bool argmax_f32(const float * x, int64_t n, unsigned long long * key, hipStream_t stream);

// The wave size the kernels were compiled for, or 0 if the probe fails. Every
// kernel assumes 32 (gfx1200's default): five-step shuffle reductions, `lane =
// tid & 31`, one wave per 32-element quantization block. A -mwavefrontsize64
// build would reduce over half a wave and be silently wrong, so the tools
// refuse to run unless this is 32 (#99). One translation unit stands for all:
// they share the target's flags.
int kernel_wave_size();

// out[i] = silu(x[i]) * y[i], written as f16 — the SwiGLU epilogue feeding the
// down projection.
bool swiglu_f16(const float * x, const float * y, void * out_f16, int64_t n, hipStream_t stream);

// rows x k f16 values quantized to int8 in blocks of `block` (32..1024, a
// multiple of 32) along k with one scale each (amax / 127, as llama.cpp's
// q8_1), and back to f16: the activations an int8 GEMM would see (#213).
bool q8_roundtrip_f16(const void * x, void * out, int64_t rows, int64_t k, int block, hipStream_t stream);

} // namespace omph::kernels
