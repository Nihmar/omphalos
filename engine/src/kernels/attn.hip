// Attention block kernels: fused Q/gate split, partial NeoX RoPE, naive causal
// attention with a KV cache (PLAN.md §10.4).
#include "kernels/attn.hh"

#include <hip/hip_fp16.h>
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

// Flash-style attention: one workgroup per (query token, head) with one thread
// per head dimension. Each cache row's score is computed once and staged in LDS,
// instead of being recomputed for every (row, dimension) pair three times over
// (which is what the first, naive kernel did). The cache is walked in blocks of
// hd rows with online softmax rescaling.
__global__ void attention_flash_kernel(const float * __restrict__ q,
                                       const float * __restrict__ k_cache,
                                       const float * __restrict__ v_cache,
                                       const float * __restrict__ gate, float * __restrict__ out,
                                       const long long tokens, const long long seq,
                                       const long long nh, const long long nkv, const long long hd,
                                       const float scale, const long long head_ratio) {
    extern __shared__ float sh[];
    float * sc = sh;  // hd scores, then the weights
    const long long h = blockIdx.y;
    const long long t = blockIdx.x;
    const int d = (int) threadIdx.x;
    const long long kvh = h / head_ratio;
    const long long q_row = seq - tokens + t;
    const float * qr = q + (t * nh + h) * hd;

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;
    for (long long s0 = 0; s0 <= q_row; s0 += hd) {
        const long long bs = q_row - s0 + 1 < hd ? q_row - s0 + 1 : hd;
        if (d < bs) {
            const float * ks = k_cache + ((s0 + d) * nkv + kvh) * hd;
            float dot = 0.0f;
            for (long long i = 0; i < hd; ++i) {
                dot += qr[i] * ks[i];
            }
            sc[d] = dot * scale;
        }
        __syncthreads();
        float bm = -INFINITY;
        for (long long j = 0; j < bs; ++j) {
            bm = fmaxf(bm, sc[j]);
        }
        const float m_new = fmaxf(m, bm);
        const float corr = expf(m - m_new);
        // Every thread must finish scanning the scores before any of them is
        // replaced by its weight, or threads disagree on the block max.
        __syncthreads();
        if (d < bs) {
            sc[d] = expf(sc[d] - m_new);
        }
        __syncthreads();
        float lsum = 0.0f;
        float a = 0.0f;
        for (long long j = 0; j < bs; ++j) {
            const float wj = sc[j];
            lsum += wj;
            a += wj * v_cache[((s0 + j) * nkv + kvh) * hd + d];
        }
        l = l * corr + lsum;
        acc = acc * corr + a;
        m = m_new;
        __syncthreads();
    }
    float val = acc / l;
    if (gate != nullptr) {
        val *= 1.0f / (1.0f + expf(-gate[(t * nh + h) * hd + d]));
    }
    out[(t * nh + h) * hd + d] = val;
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
    if (hd > 1024) {
        return false;  // the flash layout needs one thread per head dimension
    }
    const dim3 grid((unsigned) tokens, (unsigned) nh);
    const size_t smem = (size_t) hd * sizeof(float);
    attention_flash_kernel<<<grid, (unsigned) hd, smem, stream>>>(
        q, k_cache, v_cache, gate, out, (long long) tokens, (long long) seq, (long long) nh,
        (long long) nkv, (long long) hd, scale, (long long) (nh / nkv));
    return hipGetLastError() == hipSuccess;
}

// ------------------------------------------------------------ quantized KV

namespace {

constexpr long long kKvBlock = 32;


__global__ void hadamard_kernel(float * __restrict__ x, const long long rows, const int n) {
    extern __shared__ float sh[];
    const long long row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float * xr = x + row * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sh[i] = xr[i];
    }
    __syncthreads();
    for (int len = 1; len < n; len <<= 1) {
        for (int i = (int) threadIdx.x; i < n / 2; i += (int) blockDim.x) {
            const int a = (i / len) * 2 * len + (i % len);
            const int b = a + len;
            const float u = sh[a];
            const float v = sh[b];
            sh[a] = u + v;
            sh[b] = u - v;
        }
        __syncthreads();
    }
    const float norm = rsqrtf((float) n);  // orthonormal, so scores are exactly Q.K
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        xr[i] = sh[i] * norm;
    }
}

__global__ void kv_quant_kernel(const float * __restrict__ k_src, const float * __restrict__ v_src,
                                uint8_t * __restrict__ k_q8, __half * __restrict__ k_scales,
                                uint8_t * __restrict__ v_q4, __half * __restrict__ v_scales,
                                __half * __restrict__ k16, __half * __restrict__ v16,
                                const long long pos, const long long n_rows,
                                const long long nkv, const long long hd,
                                const long long window) {
    const long long row = blockIdx.x;  // token * nkv + kv head
    if (row >= n_rows) {
        return;
    }
    const long long token = pos + row / nkv;
    const long long head = row % nkv;
    const long long dst = (token * nkv + head) * hd;   // quantized element offset
    const long long sdst = (token * nkv + head) * (hd / kKvBlock);  // scale offset
    const long long nb = hd / kKvBlock;
    const int lane = (int) threadIdx.x;

    // FP16 window (PLAN §13.4): the most recent `window` tokens keep an exact K
    // and V in the rotated basis, in a ring, so the attention can read them
    // instead of the quantized form. Only the last `window` tokens of this launch
    // are written: in a prefill chunk longer than the window, earlier tokens
    // would share a slot with later ones (a write race), and the ring must end
    // up holding exactly the last `window` positions of the sequence.
    if (k16 != nullptr && token + window >= pos + n_rows / nkv) {
        const long long slot = (token % window) * nkv + head;
        for (long long i = lane; i < hd; i += 32) {
            k16[slot * hd + i] = __float2half(k_src[row * hd + i]);
            v16[slot * hd + i] = __float2half(v_src[row * hd + i]);
        }
    }
    for (long long b = 0; b < nb; ++b) {
        const long long off = row * hd + b * kKvBlock;
        float amax = 0.0f;
        if (lane < kKvBlock / 8) {  // 4 lanes x 8 values
            for (int i = 0; i < 8; ++i) {
                amax = fmaxf(amax, fabsf(k_src[off + lane * 8 + i]));
            }
        }
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) {  // all 32 lanes, so every writer sees it
            amax = fmaxf(amax, __shfl_xor(amax, s));
        }
        const float sc = amax / 127.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        if (lane == 0) {
            k_scales[sdst + b] = __float2half(sc);
        }
        if (lane < kKvBlock / 8) {
            for (int i = 0; i < 8; ++i) {
                const int q = (int) rintf(k_src[off + lane * 8 + i] * inv);
                k_q8[dst + b * kKvBlock + lane * 8 + i] =
                    (uint8_t) (int8_t) (q > 127 ? 127 : (q < -127 ? -127 : q));
            }
        }
        // V: the same block, packed two values per byte
        float vmax = 0.0f;
        if (lane < kKvBlock / 8) {
            for (int i = 0; i < 8; ++i) {
                vmax = fmaxf(vmax, fabsf(v_src[off + lane * 8 + i]));
            }
        }
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) {  // all 32 lanes: 16 of them write V bytes
            vmax = fmaxf(vmax, __shfl_xor(vmax, s));
        }
        const float vsc = vmax / 7.0f;
        const float vinv = vmax > 0.0f ? 7.0f / vmax : 0.0f;
        if (lane == 0) {
            v_scales[sdst + b] = __float2half(vsc);
        }
        if (lane < kKvBlock / 2) {  // 16 lanes, one output byte each: values i, i+16
            const int q0 = (int) rintf(v_src[off + lane] * vinv);
            const int q1 = (int) rintf(v_src[off + 16 + lane] * vinv);
            const int c0 = q0 > 7 ? 7 : (q0 < -8 ? -8 : q0);
            const int c1 = q1 > 7 ? 7 : (q1 < -8 ? -8 : q1);
            v_q4[dst / 2 + b * (kKvBlock / 2) + lane] =
                (uint8_t) (((c0 + 8) & 0xF) | (((c1 + 8) & 0xF) << 4));
        }
    }
}

__global__ void attention_flash_q8q4_kernel(
    const float * __restrict__ q, const uint8_t * __restrict__ k_q8,
    const __half * __restrict__ k_scales, const uint8_t * __restrict__ v_q4,
    const __half * __restrict__ v_scales, const __half * __restrict__ k16,
    const __half * __restrict__ v16, const float * __restrict__ gate,
    float * __restrict__ out, const long long tokens, const long long seq, const long long nh,
    const long long nkv, const long long hd, const float scale, const long long head_ratio,
    const long long window) {
    extern __shared__ float sh[];
    float * sc = sh;
    const long long h = blockIdx.y;
    const long long t = blockIdx.x;
    const int d = (int) threadIdx.x;
    const long long kvh = h / head_ratio;
    const long long q_row = seq - tokens + t;
    const float * qr = q + (t * nh + h) * hd;
    const long long nb = hd / kKvBlock;

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;
    for (long long s0 = 0; s0 <= q_row; s0 += hd) {
        const long long bs = q_row - s0 + 1 < hd ? q_row - s0 + 1 : hd;
        if (d < bs) {
            const long long p = s0 + d;
            float dot = 0.0f;
            // The ring holds positions [seq - window, seq): relative to the
            // sequence, not to this query, since later tokens of the same
            // chunk have already replaced the slots of earlier ones.
            if (k16 != nullptr && p + window >= seq) {
                const __half * kw = k16 + ((p % window) * nkv + kvh) * hd;
                for (long long i = 0; i < hd; ++i) {
                    dot = fmaf(qr[i], __half2float(kw[i]), dot);
                }
            } else {
                const long long kv = p * nkv + kvh;
                const uint8_t * kq = k_q8 + kv * hd;
                const __half * ksc = k_scales + kv * nb;
                for (long long b = 0; b < nb; ++b) {
                    const float ks = __half2float(ksc[b]);
                    for (int i = 0; i < kKvBlock; ++i) {
                        const float k = (float) (int8_t) kq[b * kKvBlock + i];
                        dot = fmaf(qr[b * kKvBlock + i], k * ks, dot);
                    }
                }
            }
            sc[d] = dot * scale;
        }
        __syncthreads();
        float bm = -INFINITY;
        for (long long j = 0; j < bs; ++j) {
            bm = fmaxf(bm, sc[j]);
        }
        const float m_new = fmaxf(m, bm);
        const float corr = expf(m - m_new);
        // Every thread must finish scanning the scores before any of them is
        // replaced by its weight, or threads disagree on the block max.
        __syncthreads();
        if (d < bs) {
            sc[d] = expf(sc[d] - m_new);
        }
        __syncthreads();
        float lsum = 0.0f;
        float a = 0.0f;
        for (long long j = 0; j < bs; ++j) {
            const float wj = sc[j];
            const long long p = s0 + j;
            float vval;
            if (v16 != nullptr && p + window >= seq) {
                vval = __half2float(v16[((p % window) * nkv + kvh) * hd + d]);
            } else {
                const long long kv = p * nkv + kvh;
                const uint8_t * vq = v_q4 + kv * hd / 2;
                const float vs = __half2float(v_scales[kv * nb + d / kKvBlock]);
                const uint8_t byte = vq[(d / kKvBlock) * (kKvBlock / 2) + (d & 15)];
                const int nib = (d & 16) ? (byte >> 4) : (byte & 0xF);
                vval = (float) (nib - 8) * vs;
            }
            lsum += wj;
            a = fmaf(wj, vval, a);
        }
        l = l * corr + lsum;
        acc = acc * corr + a;
        m = m_new;
        __syncthreads();
    }
    float val = acc / l;
    // V is stored rotated (see the writer): undo it here, before the gate, because
    // H is symmetric and the gate is elementwise. The workgroup is exactly one
    // head, so the butterfly is an LDS pass.
    sc[d] = val;
    __syncthreads();
    for (int len = 1; len < (int) hd; len <<= 1) {
        for (int i = d; i < (int) hd / 2; i += blockDim.x) {
            const int a = (i / len) * 2 * len + (i % len);
            const int b = a + len;
            const float u = sc[a];
            const float v = sc[b];
            sc[a] = u + v;
            sc[b] = u - v;
        }
        __syncthreads();
    }
    val = sc[d] * rsqrtf((float) hd);
    float outv = val;
    if (gate != nullptr) {
        outv *= 1.0f / (1.0f + expf(-gate[(t * nh + h) * hd + d]));
    }
    out[(t * nh + h) * hd + d] = outv;
}

} // namespace

bool hadamard_f32(float * x, const int64_t rows, const int64_t n, hipStream_t stream) {
    if (x == nullptr || rows <= 0 || n <= 0 || (n & (n - 1)) != 0) {
        return false;
    }
    hadamard_kernel<<<(unsigned) rows, 256, (size_t) n * sizeof(float), stream>>>(x, rows,
                                                                                  (int) n);
    return hipGetLastError() == hipSuccess;
}

bool kv_quant(const float * k_src, const float * v_src, uint8_t * k_q8, void * k_scales,
              uint8_t * v_q4, void * v_scales, void * k16, void * v16, const int64_t pos,
              const int64_t tokens, const int64_t nkv, const int64_t hd, const int64_t window,
              hipStream_t stream) {
    if (k_src == nullptr || v_src == nullptr || k_q8 == nullptr || k_scales == nullptr ||
        v_q4 == nullptr || v_scales == nullptr || tokens <= 0 || nkv <= 0 || hd <= 0 ||
        hd % kKvBlock != 0) {
        return false;
    }
    const long long n_rows = tokens * nkv;
    kv_quant_kernel<<<(unsigned) n_rows, 32, 0, stream>>>(
        k_src, v_src, k_q8, static_cast<__half *>(k_scales), v_q4,
        static_cast<__half *>(v_scales), static_cast<__half *>(k16), static_cast<__half *>(v16),
        pos, n_rows, nkv, hd, window > 0 ? window : 1);
    return hipGetLastError() == hipSuccess;
}

bool attention_flash_q8q4(const float * q, const uint8_t * k_q8, const void * k_scales,
                          const uint8_t * v_q4, const void * v_scales, const void * k16,
                          const void * v16, const float * gate, float * out, const int64_t tokens,
                          const int64_t seq, const int64_t nh, const int64_t nkv, const int64_t hd,
                          const float scale, const int64_t head_ratio, const int64_t window,
                          hipStream_t stream) {
    const dim3 grid((unsigned) tokens, (unsigned) nh);
    attention_flash_q8q4_kernel<<<grid, (unsigned) hd, (size_t) hd * sizeof(float), stream>>>(
        q, k_q8, static_cast<const __half *>(k_scales), v_q4,
        static_cast<const __half *>(v_scales), static_cast<const __half *>(k16),
        static_cast<const __half *>(v16), gate, out, tokens, seq, nh, nkv, hd, scale, head_ratio,
        window > 0 ? window : 1);
    return hipGetLastError() == hipSuccess;
}

} // namespace omph::kernels
