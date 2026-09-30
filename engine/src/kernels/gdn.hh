// Gated delta net kernels (PLAN.md §10.2, autoregressive path).
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace omph::kernels {

// out = sigmoid(x), in place.
bool sigmoid_inplace(float * x, int64_t n, hipStream_t stream);

// alpha[i] = softplus(alpha[i] + dt_bias[i % heads]), in place.
bool softplus_bias_inplace(float * alpha, const float * dt_bias, int64_t tokens, int64_t heads,
                           hipStream_t stream);

// x[i] *= w[i % heads], in place (e.g. gate = a_softplus * ssm_a).
bool mul_row_inplace(float * x, const float * w, int64_t tokens, int64_t heads,
                     hipStream_t stream);

// Depthwise causal conv1d over the fused qkv channels with an explicit state.
//   state: (kernel-1, channels) f32, read only; qkv: (tokens, channels);
//   w: (channels, kernel); out: (tokens, channels).
//   out[t, c] = sum_j w[c, j] * input[t + j, c], input = [state rows; qkv rows].
bool conv1d_state(const float * qkv, const float * w, const float * state, float * out,
                  int64_t tokens, int64_t channels, int64_t kernel, hipStream_t stream);

// Shifts the conv state: new_state[r, c] = input[tokens + r, c] with
// input = [state rows (kernel-1); qkv rows (tokens)]. Writes to a separate
// buffer, so the caller can double-buffer and swap.
bool conv_state_update(const float * qkv, const float * state, float * new_state, int64_t tokens,
                       int64_t channels, int64_t kernel, hipStream_t stream);

// Extracts q/k/v from the fused conv output (channel order: q | k | v).
bool split_qkv(const float * fused, float * q, float * k, float * v, int64_t tokens,
               int64_t q_dims, int64_t kv_dims, int64_t v_dims, hipStream_t stream);

// out = x / (1 + exp(-x)), in place.
bool silu_inplace(float * x, int64_t n, hipStream_t stream);

// Delta rule steps for one token (state layout: [heads, s, s]).
//   state[h,i,j] *= exp(gate[h])
//   sk[h, j]      = sum_i state[h, i, j] * k[h % n_kh, i]
//   d[h, j]       = (v[h, j] - sk[h, j]) * beta[h]
//   state[h,i,j] += k[h % n_kh, i] * d[h, j]
//   o[h, j]       = scale * sum_i state[h, i, j] * q[h % n_kh, i]
bool delta_decay(float * state, const float * gate, int64_t heads, int64_t s, hipStream_t stream);
bool delta_sk(const float * state, const float * k, float * sk, int64_t heads, int64_t n_kh,
              int64_t s, hipStream_t stream);
bool delta_d(const float * v, const float * sk, const float * beta, float * d, int64_t heads,
             int64_t s, hipStream_t stream);
bool delta_update(float * state, const float * k, const float * d, int64_t heads, int64_t n_kh,
                  int64_t s, hipStream_t stream);
bool delta_o(const float * state, const float * q, float * o, int64_t heads, int64_t n_kh,
             int64_t s, float scale, hipStream_t stream);

// out = rms_norm(o, w, eps) * silu(z)   (rows x n, n % 256 == 0)
bool gated_norm(const float * o, const float * w, const float * z, float * out, int64_t rows,
                int64_t n, float eps, hipStream_t stream);

// One-token delta rule in a single launch (PLAN.md §10.2): decay, sk = S^T k,
// d = (v - sk) * beta, S += k (x) d and o = scale * S^T q for every head.
//   state: (heads, s, s); q/k: (n_kh, s); v/o: (heads, s); gate: (heads,)
//   beta: (heads,); one workgroup per head, 256 threads.
bool delta_step_fused(float * state, const float * q, const float * k, const float * v,
                      const float * beta, const float * gate, float * o, int64_t heads,
                      int64_t n_kh, int64_t s, float scale, hipStream_t stream);

// Delta-net front end in one launch: the depthwise causal conv over
// [state; qkv], silu, the q/k/v split and the state shift. The grid covers
// tokens*channels conv elements plus (kernel-1)*channels state elements.
bool conv_silu_split_fused(const float * qkv, const float * w, const float * state,
                           float * new_state, float * q, float * k, float * v, int64_t tokens,
                           int64_t channels, int64_t kernel, int64_t q_dims, int64_t kv_dims,
                           int64_t v_dims, hipStream_t stream);

} // namespace omph::kernels
