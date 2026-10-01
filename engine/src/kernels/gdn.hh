// Gated delta net kernels (PLAN.md §10.2, autoregressive path).
#pragma once

#include <cstdint>

#include <hip/hip_fp16.h>
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

// One token of a Gated DeltaNet layer after its projections, in one launch
// (PLAN.md §10.2; #78, #83): per value head, the conv1d + silu of its q / k
// (key head h % n_kh) and v from the qkv projection and the conv state, the L2
// norm of q and k, the gates (beta = sigmoid(beta_raw), decay =
// exp(softplus(alpha_raw + dt_bias) * ssm_a), the raw gates being this head's
// rows of the BF16 beta / alpha projections dotted with x16), the delta rule on the head's
// state (decay, sk = S^T k, d = (v - sk) * beta, S += k (x) d, o = l2_scale *
// S^T q) and the gated RMSNorm (o / sqrt(mean(o^2) + eps_norm) * norm_w *
// silu(z)) written as f16. State size s = 128; one 256-thread workgroup per
// value head. Launch t = 0 .. tokens - 1 in order; t = 0 also shifts the conv
// state from conv_cur into conv_new.
struct GdnStep {
    float * state = nullptr;           // (heads, 128, 128)
    const float * qkv = nullptr;       // (tokens, channels): q | k | v
    const float * conv_w = nullptr;    // (channels, conv_k)
    const float * conv_cur = nullptr;  // (conv_k - 1, channels)
    float * conv_new = nullptr;        // (conv_k - 1, channels)
    const __half * x16 = nullptr;      // (k_in) the normed layer input, token t
    const uint16_t * w_beta = nullptr;   // (heads, k_in) BF16 beta projection
    const uint16_t * w_alpha = nullptr;  // (heads, k_in) BF16 alpha projection
    long long k_in = 0;
    const float * dt_bias = nullptr;   // (heads)
    const float * ssm_a = nullptr;     // (heads)
    const float * z = nullptr;         // (heads, 128), token t
    const float * norm_w = nullptr;    // (128)
    __half * out16 = nullptr;          // (heads, 128), token t
    long long tokens = 0;
    long long t = 0;
    long long channels = 0;
    long long q_dims = 0;
    long long kv_dims = 0;
    long long conv_k = 0;
    long long n_kh = 0;
    float eps_l2 = 0.0f;               // the L2 norm's eps / s
    float l2_scale = 0.0f;             // 1 / sqrt(s), also the output scale
    float eps_norm = 0.0f;
};
bool gdn_step(const GdnStep & a, int64_t heads, hipStream_t stream);

} // namespace omph::kernels
