// Attention block kernels: fused Q/gate split, partial NeoX RoPE, naive causal
// attention with a KV cache (PLAN.md §10.4).
#include "kernels/attn.hh"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace omph::kernels {
namespace {

constexpr int kBlock = 256;

__global__ void split_qg_kernel(const float * __restrict__ q_full, float * __restrict__ q,
                                float * __restrict__ gate, const long long tokens,
                                const long long heads, const long long hd) {
    const long long e = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= tokens * heads * hd) {
        return;
    }
    const long long d = e % hd;
    const long long h = (e / hd) % heads;
    const long long t = e / (hd * heads);
    const long long base = (t * heads + h) * 2 * hd;
    q[e] = q_full[base + d];
    gate[e] = q_full[base + hd + d];
}

__global__ void rope_neox_kernel(float * __restrict__ x, const long long tokens,
                                 const long long heads, const long long hd, const int n_rot,
                                 const float freq_base, const long long pos_offset) {
    const int half = n_rot / 2;
    const long long e = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= tokens * heads * (long long) half) {
        return;
    }
    const int i = (int) (e % half);
    const long long h = (e / half) % heads;
    const long long t = e / (half * heads);
    float * row = x + (t * heads + h) * hd;
    const float theta = (float) (t + pos_offset) / powf(freq_base, (float) (2 * i) / (float) n_rot);
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float x0 = row[i];
    const float x1 = row[i + half];
    row[i] = x0 * c - x1 * s;
    row[i + half] = x0 * s + x1 * c;
}

// One thread per (token, head, dim); the scores are recomputed per pass, which
// is deliberately naive — decode speed work comes in M3+.
__global__ void attention_kernel(const float * __restrict__ q, const float * __restrict__ k_cache,
                                 const float * __restrict__ v_cache,
                                 const float * __restrict__ gate, float * __restrict__ out,
                                 const long long tokens, const long long seq, const long long nh,
                                 const long long nkv, const long long hd, const float scale,
                                 const long long head_ratio) {
    const long long e = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= tokens * nh * hd) {
        return;
    }
    const long long d = e % hd;
    const long long h = (e / hd) % nh;
    const long long t = e / (hd * nh);
    const long long kvh = h / head_ratio;
    const long long q_row = seq - tokens + t;

    const float * qr = q + (t * nh + h) * hd;

    float max_score = -INFINITY;
    for (long long s = 0; s <= q_row; ++s) {
        const float * ks = k_cache + (s * nkv + kvh) * hd;
        float dot = 0.0f;
        for (long long i = 0; i < hd; ++i) {
            dot += qr[i] * ks[i];
        }
        max_score = fmaxf(max_score, dot * scale);
    }
    float sum = 0.0f;
    for (long long s = 0; s <= q_row; ++s) {
        const float * ks = k_cache + (s * nkv + kvh) * hd;
        float dot = 0.0f;
        for (long long i = 0; i < hd; ++i) {
            dot += qr[i] * ks[i];
        }
        sum += expf(dot * scale - max_score);
    }
    float acc = 0.0f;
    for (long long s = 0; s <= q_row; ++s) {
        const float * ks = k_cache + (s * nkv + kvh) * hd;
        float dot = 0.0f;
        for (long long i = 0; i < hd; ++i) {
            dot += qr[i] * ks[i];
        }
        const float w = expf(dot * scale - max_score) / sum;
        acc += w * v_cache[(s * nkv + kvh) * hd + d];
    }
    if (gate != nullptr) {
        acc *= 1.0f / (1.0f + expf(-gate[(t * nh + h) * hd + d]));
    }
    out[e] = acc;
}

} // namespace

bool split_qg(const float * q_full, float * q, float * gate, const int64_t tokens,
              const int64_t heads, const int64_t head_dim, hipStream_t stream) {
    if (q_full == nullptr || q == nullptr || gate == nullptr || tokens <= 0 || heads <= 0 ||
        head_dim <= 0) {
        return false;
    }
    const long long n = (long long) tokens * heads * head_dim;
    const unsigned grid = (unsigned) ((n + kBlock - 1) / kBlock);
    split_qg_kernel<<<grid, kBlock, 0, stream>>>(q_full, q, gate, (long long) tokens,
                                                 (long long) heads, (long long) head_dim);
    return hipGetLastError() == hipSuccess;
}

bool rope_neox(float * x, const int64_t tokens, const int64_t heads, const int64_t head_dim,
               const int64_t n_rot, const float freq_base, const int64_t pos_offset,
               hipStream_t stream) {
    if (x == nullptr || tokens <= 0 || heads <= 0 || n_rot <= 0 || n_rot % 2 != 0 ||
        n_rot > head_dim) {
        return false;
    }
    const long long n = (long long) tokens * heads * (n_rot / 2);
    const unsigned grid = (unsigned) ((n + kBlock - 1) / kBlock);
    rope_neox_kernel<<<grid, kBlock, 0, stream>>>(x, (long long) tokens, (long long) heads,
                                                  (long long) head_dim, (int) n_rot, freq_base,
                                                  (long long) pos_offset);
    return hipGetLastError() == hipSuccess;
}

bool attention(const float * q, const float * k_cache, const float * v_cache, const float * gate,
               float * out, const int64_t tokens, const int64_t seq, const int64_t nh,
               const int64_t nkv, const int64_t hd, const float scale, hipStream_t stream) {
    if (q == nullptr || k_cache == nullptr || v_cache == nullptr || out == nullptr || tokens <= 0 ||
        seq < tokens || nh <= 0 || nkv <= 0 || hd <= 0 || nh % nkv != 0) {
        return false;
    }
    const long long n = (long long) tokens * nh * hd;
    const unsigned grid = (unsigned) ((n + kBlock - 1) / kBlock);
    attention_kernel<<<grid, kBlock, 0, stream>>>(q, k_cache, v_cache, gate, out,
                                                  (long long) tokens, (long long) seq,
                                                  (long long) nh, (long long) nkv, (long long) hd,
                                                  scale, (long long) (nh / nkv));
    return hipGetLastError() == hipSuccess;
}

} // namespace omph::kernels
