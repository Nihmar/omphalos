// Gated delta net kernels (PLAN.md §10.2, autoregressive path).
#pragma once

#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace omph::kernels {

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
    // Speculative verification (#122), all optional:
    float * state_out = nullptr;       // the updated state goes here instead of `state`
    float * replay = nullptr;          // per token t: gdn_replay_floats(heads, n_kh) floats at
                                       // t * that, the token's rank-1 factors (decay, k, d)
    float * conv_hist = nullptr;       // at t = 0: [conv_cur; qkv] rows, (conv_k - 1 + tokens)
                                       // x channels, to rebuild the tail of a shorter prefix
};
bool gdn_step(const GdnStep & a, int64_t heads, hipStream_t stream);

// Floats of one token's replay record: decay (heads), k (n_kh x 128), d (heads x 128).
inline int64_t gdn_replay_floats(const int64_t heads, const int64_t n_kh) {
    return heads + n_kh * 128 + heads * 128;
}

// Applies the first `tokens` records of `replay` to `state` in place:
// S <- decay S; S <- S + k d^T per token, the float operations of gdn_step in
// its order, so the result equals the state gdn_step reached after them,
// bit for bit (#104).
bool gdn_replay(float * state, const float * replay, int64_t tokens, int64_t heads,
                int64_t n_kh, hipStream_t stream);

// conv_dst rows r < conv_k - 1 = conv_hist rows accepted + r: the conv tail after
// the first `accepted` tokens of a recorded step.
bool gdn_conv_select(const float * conv_hist, float * conv_dst, int64_t accepted,
                     int64_t conv_k, int64_t channels, hipStream_t stream);

} // namespace omph::kernels
