// Attention block kernels: the fused prep (Q/gate split, QK-norm, partial NeoX
// RoPE, Hadamard rotation, quantized KV writer) and the GQA-grouped flash
// attention over either cache (PLAN.md §10.4, §13).
#include "kernels/attn.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace omph::kernels {
namespace {

constexpr long long kKvBlock = 32;  // quantized KV block along head_dim (PLAN.md §13)

} // namespace

// ------------------------------------------------------- fused attention prep

namespace {

constexpr int kPrepHd = 256;  // head_dim: one thread per dimension, 8 waves = 8 quant blocks

// Block-wide sum over the 256 threads (8 wave32s).
__device__ inline float prep_block_sum(float v, float * red) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor(v, off);
    }
    if ((threadIdx.x & 31) == 0) {
        red[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < kPrepHd / 32; ++i) {
        t += red[i];
    }
    __syncthreads();
    return t;
}

// NeoX RoPE on the first n_rot dimensions of a head held one value per thread.
__device__ inline float prep_rope(const float x, float * buf, const long long pos,
                                  const int n_rot, const float freq_base) {
    const int d = (int) threadIdx.x;
    buf[d] = x;
    __syncthreads();
    float y = x;
    const int half = n_rot / 2;
    if (d < n_rot) {
        const int i = d < half ? d : d - half;
        const float theta = (float) pos / powf(freq_base, (float) (2 * i) / (float) n_rot);
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = buf[i];
        const float x1 = buf[i + half];
        y = d < half ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    __syncthreads();
    return y;
}

// Orthonormal fast Walsh-Hadamard over the 256 values of a head.
__device__ inline float prep_hadamard(const float x, float * buf) {
    const int d = (int) threadIdx.x;
    buf[d] = x;
    __syncthreads();
    for (int len = 1; len < kPrepHd; len <<= 1) {
        if (d < kPrepHd / 2) {
            const int a = (d / len) * 2 * len + (d % len);
            const int b = a + len;
            const float u = buf[a];
            const float v = buf[b];
            buf[a] = u + v;
            buf[b] = u - v;
        }
        __syncthreads();
    }
    const float y = buf[d] * rsqrtf((float) kPrepHd);
    __syncthreads();
    return y;
}

// grid: (tokens, nh + 2 * nkv); 256 threads. Workgroup y < nh: query head
// (split q / gate, RMSNorm, RoPE, rotation); < nh + nkv: key head (RMSNorm,
// RoPE, rotation, written back and, with a quantized cache, quantized into
// it); else value head (rotation, written back, quantized). Each wave is one
// 32-element quantization block, so the block max is a wave reduction. The
// arithmetic is that of the separate kernels it replaced (split_qg, rms_norm,
// rope_neox, a Hadamard kernel and the KV quantizer, #79).
__global__ __launch_bounds__(kPrepHd) void attn_prep_kernel(const AttnPrep a) {
    __shared__ float buf[kPrepHd];
    __shared__ float red[kPrepHd / 32];
    const long long t = blockIdx.x;
    const long long role = blockIdx.y;
    const int d = (int) threadIdx.x;
    const long long pos = a.pos0 + t;
    if (role < a.nh) {
        const long long h = role;
        const float * src = a.qg + (t * a.nh + h) * 2 * kPrepHd;
        a.gate[(t * a.nh + h) * kPrepHd + d] = src[kPrepHd + d];
        float x = src[d];
        const float inv = 1.0f / sqrtf(prep_block_sum(x * x, red) / (float) kPrepHd + a.eps);
        x = x * inv * a.q_norm[d] * 1.0f;
        x = prep_rope(x, buf, pos, a.n_rot, a.freq_base);
        if (a.rotate) {
            x = prep_hadamard(x, buf);
        }
        a.q[(t * a.nh + h) * kPrepHd + d] = x;
        return;
    }
    const bool is_k = role < a.nh + a.nkv;
    const long long h = is_k ? role - a.nh : role - a.nh - a.nkv;
    float * row = (is_k ? a.k : a.v) + (t * a.nkv + h) * kPrepHd;
    float x = row[d];
    if (is_k) {
        const float inv = 1.0f / sqrtf(prep_block_sum(x * x, red) / (float) kPrepHd + a.eps);
        x = x * inv * a.k_norm[d] * 1.0f;
        x = prep_rope(x, buf, pos, a.n_rot, a.freq_base);
    }
    if (a.rotate) {
        x = prep_hadamard(x, buf);
    }
    row[d] = x;
    if (a.k_q8 == nullptr) {
        return;  // f32 cache: the caller copies the rows
    }
    const long long dst = (pos * a.nkv + h) * kPrepHd;               // element offset
    const long long sdst = (pos * a.nkv + h) * (kPrepHd / kKvBlock);  // scale offset
    const int b = d / (int) kKvBlock;
    const int lane = d & 31;
    if (a.k16 != nullptr && pos + a.window >= a.pos0 + a.tokens) {  // FP16 ring (#48)
        const long long slot = (pos % a.window) * a.nkv + h;
        (is_k ? a.k16 : a.v16)[slot * kPrepHd + d] = __float2half(x);
    }
    float amax = fabsf(x);
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, s));
    }
    if (is_k && !a.k_q4) {
        const float sc = amax / 127.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        if (lane == 0) {
            a.k_scales[sdst + b] = __float2half(sc);
        }
        const int q = (int) rintf(x * inv);
        a.k_q8[dst + d] = (uint8_t) (int8_t) (q > 127 ? 127 : (q < -127 ? -127 : q));
    } else {
        const float vsc = amax / 7.0f;
        const float vinv = amax > 0.0f ? 7.0f / amax : 0.0f;
        if (lane == 0) {
            (is_k ? a.k_scales : a.v_scales)[sdst + b] = __float2half(vsc);
        }
        const float x_hi = __shfl(x, (lane + 16) & 31);  // value lane + 16 of the block
        if (lane < (int) kKvBlock / 2) {
            const int q0 = (int) rintf(x * vinv);
            const int q1 = (int) rintf(x_hi * vinv);
            const int c0 = q0 > 7 ? 7 : (q0 < -8 ? -8 : q0);
            const int c1 = q1 > 7 ? 7 : (q1 < -8 ? -8 : q1);
            (is_k ? a.k_q8 : a.v_q4)[dst / 2 + b * (kKvBlock / 2) + lane] =
                (uint8_t) (((c0 + 8) & 0xF) | (((c1 + 8) & 0xF) << 4));
        }
    }
}

} // namespace

bool attn_prep(const AttnPrep & a, hipStream_t stream) {
    if (a.qg == nullptr || a.q == nullptr || a.gate == nullptr || a.k == nullptr ||
        a.v == nullptr || a.q_norm == nullptr || a.k_norm == nullptr || a.tokens <= 0 ||
        a.hd != kPrepHd || a.nh <= 0 || a.nkv <= 0 || a.n_rot <= 0 || a.n_rot % 2 != 0 ||
        a.n_rot > a.hd || (a.k_q8 != nullptr && (a.k_scales == nullptr || a.v_q4 == nullptr ||
                                                 a.v_scales == nullptr)) ||
        (a.k16 == nullptr) != (a.v16 == nullptr) || (a.k16 != nullptr && a.window <= 0)) {
        return false;
    }
    const dim3 grid((unsigned) a.tokens, (unsigned) (a.nh + 2 * a.nkv));
    attn_prep_kernel<<<grid, kPrepHd, 0, stream>>>(a);
    return hipGetLastError() == hipSuccess;
}

// ------------------------------------------------- GQA-grouped flash attention

namespace {

constexpr int kGqaHd = 256;     // head_dim: one thread per dimension
constexpr int kGqaKb = 16;      // keys per block: 256 threads = 16 keys x 16 chunks
constexpr int kGqaGroup = 6;    // query heads per KV head (24 / 4): fixed for this model
constexpr int kGqaTargetWgs = 128;  // split the keys until a launch has this many
constexpr int kGqaMaxSplits = 64;

struct KvArgs {
    const float * k_f32;
    const float * v_f32;
    const uint8_t * k_q8;
    const __half * k_scales;
    const uint8_t * v_q4;
    const __half * v_scales;
    const __half * k16;
    const __half * v16;
    long long window;
    bool k_q4;
};

// Dimensions [16c, 16c + 16) of the K row of position p for KV head kvh, as one
// 16-byte load for Q8 (dequantized; inside the FP16 window: the exact copy).
template <bool Q8>
__device__ inline void gqa_load_k16(const KvArgs & kv, const long long p, const long long kvh,
                                    const long long nkv, const int c, const long long seq,
                                    float * k) {
    const long long row = p * nkv + kvh;
    if constexpr (Q8) {
        if (kv.k16 != nullptr && p + kv.window >= seq) {
            const uint4 * src = reinterpret_cast<const uint4 *>(
                kv.k16 + ((p % kv.window) * nkv + kvh) * kGqaHd + c * 16);
            uint4 h[2] = {src[0], src[1]};
            const __half * hv = reinterpret_cast<const __half *>(h);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                k[i] = __half2float(hv[i]);
            }
            return;
        }
        const float sc = __half2float(kv.k_scales[row * (kGqaHd / kKvBlock) + c / 2]);
        if (kv.k_q4) {
            // V's Q4 format: byte j of the block holds element j (low nibble) and
            // j + 16 (high); chunk c is the low or high half of block c / 2.
            const uint4 b4 = *reinterpret_cast<const uint4 *>(kv.k_q8 + row * (kGqaHd / 2) +
                                                              (c / 2) * (kKvBlock / 2));
            const uint8_t * bv4 = reinterpret_cast<const uint8_t *>(&b4);
            const int sh = (c & 1) ? 4 : 0;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                k[i] = (float) ((int) ((bv4[i] >> sh) & 0xF) - 8) * sc;
            }
            return;
        }
        const uint4 b = *reinterpret_cast<const uint4 *>(kv.k_q8 + row * kGqaHd + c * 16);
        const int8_t * bv = reinterpret_cast<const int8_t *>(&b);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            k[i] = (float) bv[i] * sc;
        }
    } else {
        const float4 * src = reinterpret_cast<const float4 *>(kv.k_f32 + row * kGqaHd + c * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 f = src[i];
            k[4 * i] = f.x;
            k[4 * i + 1] = f.y;
            k[4 * i + 2] = f.z;
            k[4 * i + 3] = f.w;
        }
    }
}

template <bool Q8>
__device__ inline float gqa_load_v(const KvArgs & kv, const long long p, const long long kvh,
                                   const long long nkv, const int d, const long long seq) {
    if constexpr (Q8) {
        if (kv.v16 != nullptr && p + kv.window >= seq) {
            return __half2float(kv.v16[((p % kv.window) * nkv + kvh) * kGqaHd + d]);
        }
        const long long row = p * nkv + kvh;
        const uint8_t byte = kv.v_q4[row * (kGqaHd / 2) + (d / kKvBlock) * (kKvBlock / 2) + (d & 15)];
        const int nib = (d & 16) ? (byte >> 4) : (byte & 0xF);
        return (float) (nib - 8) *
               __half2float(kv.v_scales[row * (kGqaHd / kKvBlock) + d / kKvBlock]);
    } else {
        return kv.v_f32[(p * nkv + kvh) * kGqaHd + d];
    }
}

// grid: (query tiles of TQ tokens, nkv, splits); 256 threads. Row r of the tile
// is query token t0 + r / G, query head kvh * G + r % G; the row count is a
// compile-time constant so the accumulators stay in registers (TQ = 1 for a
// decode step, 4 for a prefill chunk). Rows past the last token are computed on
// zero queries and not stored. Writes one partial (m, l, acc) per (token, head,
// split) to `work`.
template <bool Q8, int TQ>
__global__ __launch_bounds__(kGqaHd) void attention_gqa_kernel(
    const float * __restrict__ q, const KvArgs kv, float * __restrict__ work,
    const long long tokens, const long long seq, const int nh, const int nkv,
    const float scale, const int splits, const long long blocks_per_split) {
    constexpr int G = kGqaGroup;
    constexpr int kRows = TQ * G;
    // Q rows in 16-dimension chunks padded to 17: the 16 lanes of a key read
    // different chunks, and the padding puts them on different banks.
    __shared__ float qs[kRows][16 * 17];
    __shared__ float sc[kRows][kGqaKb];
    __shared__ float corr[kRows];

    const int d = (int) threadIdx.x;
    const long long t0 = (long long) blockIdx.x * TQ;
    const int kvh = (int) blockIdx.y;
    const int split = (int) blockIdx.z;
    const int nt = (int) (tokens - t0 < TQ ? tokens - t0 : TQ);
    const int R = nt * G;
    const long long qpos0 = seq - tokens + t0;  // cache position of the tile's first query

    for (int r = 0; r < kRows; ++r) {
        const long long h = (long long) kvh * G + r % G;
        qs[r][(d / 16) * 17 + d % 16] =
            r < R ? q[((t0 + r / G) * nh + h) * kGqaHd + d] * scale : 0.0f;
    }

    // Keys up to the tile's last query; this split's share of their blocks.
    const long long n_keys = qpos0 + nt;
    const long long kb_total = (n_keys + kGqaKb - 1) / kGqaKb;
    const long long kb0 = (long long) split * blocks_per_split;
    const long long kb1 = kb0 + blocks_per_split < kb_total ? kb0 + blocks_per_split : kb_total;

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        acc[r] = 0.0f;
    }
    float m_row = -INFINITY;  // thread r < kRows owns row r's running max and sum
    float l_row = 0.0f;
    __syncthreads();

    for (long long kb = kb0; kb < kb1; ++kb) {
        const long long s0 = kb * kGqaKb;
        const int bs = (int) (n_keys - s0 < kGqaKb ? n_keys - s0 : kGqaKb);
        // This block's V for dimension d, issued now with the K loads below so
        // that the 32 accesses overlap instead of each waiting on the last.
        float vv[kGqaKb];
#pragma unroll
        for (int j = 0; j < kGqaKb; ++j) {
            vv[j] = j < bs ? gqa_load_v<Q8>(kv, s0 + j, kvh, nkv, d, seq) : 0.0f;
        }
        // Scores: thread (key jj, chunk c) dots its 16 dimensions of K, read once
        // straight from memory, with the matching chunk of every row; the 16
        // lanes of a key then reduce with shuffles.
        {
            const int jj = d / 16;
            const int c = d % 16;
            float kc[16];
            if (jj < bs) {
                gqa_load_k16<Q8>(kv, s0 + jj, kvh, nkv, c, seq, kc);
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    kc[i] = 0.0f;
                }
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                float part = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    part = fmaf(qs[r][c * 17 + i], kc[i], part);
                }
#pragma unroll
                for (int m = 8; m > 0; m >>= 1) {
                    part += __shfl_xor(part, m, 16);
                }
                if (c == 0) {
                    sc[r][jj] = (jj < bs && s0 + jj <= qpos0 + r / G) ? part : -INFINITY;
                }
            }
        }
        __syncthreads();
        // Online softmax, one thread per row; scores become weights in place.
        if (d < kRows) {
            float bm = -INFINITY;
            for (int j = 0; j < kGqaKb; ++j) {
                bm = fmaxf(bm, sc[d][j]);
            }
            const float m_new = fmaxf(m_row, bm);
            float c = 1.0f;
            float lsum = 0.0f;
            if (m_new != -INFINITY) {
                c = expf(m_row - m_new);  // 0 when this is the row's first key
                for (int j = 0; j < kGqaKb; ++j) {
                    const float w = sc[d][j] == -INFINITY ? 0.0f : expf(sc[d][j] - m_new);
                    sc[d][j] = w;
                    lsum += w;
                }
            } else {
                for (int j = 0; j < kGqaKb; ++j) {
                    sc[d][j] = 0.0f;
                }
            }
            l_row = l_row * c + lsum;
            m_row = m_new;
            corr[d] = c;
        }
        __syncthreads();
        // Each V value is read once and applied to every row of the tile.
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            acc[r] *= corr[r];
        }
#pragma unroll
        for (int j = 0; j < kGqaKb; ++j) {  // weights past bs are 0, vv too
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                acc[r] = fmaf(sc[r][j], vv[j], acc[r]);
            }
        }
        __syncthreads();  // sc and corr are rewritten by the next block
    }

    // Partial layout: [((token * nh + head) * splits + split) * (hd + 2)] = m, l, acc[hd]
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        if (r < R) {
            const long long h = (long long) kvh * G + r % G;
            float * dst = work + (((t0 + r / G) * nh + h) * splits + split) * (kGqaHd + 2);
            dst[2 + d] = acc[r];
        }
    }
    if (d < R) {
        const long long h = (long long) kvh * G + d % G;
        float * dst = work + (((t0 + d / G) * nh + h) * splits + split) * (kGqaHd + 2);
        dst[0] = m_row;
        dst[1] = l_row;
    }
}

// grid: (tokens, nh); 256 threads. Merges the splits in order, then un-rotates V
// (Q8/Q4 cache: stored rotated, H symmetric, before the elementwise gate) and
// applies the gate.
__global__ __launch_bounds__(kGqaHd) void attention_gqa_merge_kernel(
    const float * __restrict__ work, const float * __restrict__ gate, float * __restrict__ out,
    __half * __restrict__ out16, const int nh, const int splits, const bool v_rotated) {
    __shared__ float row[kGqaHd];
    __shared__ float ms[kGqaMaxSplits];
    __shared__ float ls[kGqaMaxSplits];
    const long long t = blockIdx.x;
    const int h = (int) blockIdx.y;
    const int d = (int) threadIdx.x;
    const float * part = work + ((t * nh + h) * splits) * (kGqaHd + 2);
    // One thread per split loads its (m, l): all in flight at once.
    if (d < splits) {
        ms[d] = part[d * (kGqaHd + 2)];
        ls[d] = part[d * (kGqaHd + 2) + 1];
    }
    __syncthreads();
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) {
        m = fmaxf(m, ms[s]);
    }
    float l = 0.0f;
    float o = 0.0f;
    for (int s = 0; s < splits; ++s) {
        // A split that held no key for this query has m = -inf, l = acc = 0.
        const float f = ms[s] == -INFINITY ? 0.0f : expf(ms[s] - m);
        l = fmaf(ls[s], f, l);
        o = fmaf(part[s * (kGqaHd + 2) + 2 + d], f, o);
    }
    float val = o / l;
    if (v_rotated) {
        row[d] = val;
        __syncthreads();
        for (int len = 1; len < kGqaHd; len <<= 1) {
            if (d < kGqaHd / 2) {
                const int a = (d / len) * 2 * len + (d % len);
                const int b = a + len;
                const float u = row[a];
                const float v = row[b];
                row[a] = u + v;
                row[b] = u - v;
            }
            __syncthreads();
        }
        val = row[d] * rsqrtf((float) kGqaHd);
    }
    if (gate != nullptr) {
        val *= 1.0f / (1.0f + expf(-gate[(t * nh + h) * kGqaHd + d]));
    }
    if (out16 != nullptr) {  // straight into the output projection's f16 input (#79)
        out16[(t * nh + h) * kGqaHd + d] = __float2half(val);
    } else {
        out[(t * nh + h) * kGqaHd + d] = val;
    }
}

// Query tokens per workgroup: a whole prefill chunk shares each K/V read four
// ways; a decode step (or a few MTP tokens) keeps the six-row form.
int gqa_tq(const int64_t tokens) {
    return tokens >= 8 ? 4 : 1;
}

// Splits for a call: enough workgroups to fill the GPU, never more than the key
// blocks of the longest tile.
int gqa_splits(const int64_t tokens, const int64_t seq, const int64_t nkv,
               long long & blocks_per_split) {
    const long long tq = gqa_tq(tokens);
    const long long tiles = (tokens + tq - 1) / tq;
    const long long kb_max = (seq + kGqaKb - 1) / kGqaKb;
    long long splits = (kGqaTargetWgs + tiles * nkv - 1) / (tiles * nkv);
    splits = splits < 1 ? 1 : (splits > kGqaMaxSplits ? kGqaMaxSplits : splits);
    splits = splits > kb_max ? kb_max : splits;
    blocks_per_split = (kb_max + splits - 1) / splits;
    return (int) ((kb_max + blocks_per_split - 1) / blocks_per_split);
}

} // namespace

size_t attention_gqa_work_bytes(const int64_t max_tokens, const int64_t nh, const int64_t nkv,
                                const int64_t hd) {
    // Worst case over the token counts a call can have, at any sequence length
    // (the split count only grows with the sequence up to kGqaMaxSplits).
    size_t worst = 0;
    for (int64_t t = 1; t <= max_tokens; ++t) {
        long long bps = 0;
        const size_t rows = (size_t) t * nh * gqa_splits(t, (int64_t) 1 << 40, nkv, bps);
        worst = rows > worst ? rows : worst;
    }
    return worst * (size_t) (hd + 2) * sizeof(float);
}

bool attention_gqa(const float * q, const KvCache & kv, const float * gate, float * out,
                   const int64_t tokens, const int64_t seq, const int64_t nh, const int64_t nkv,
                   const int64_t hd, const float scale, const bool v_rotated, void * work,
                   const size_t work_bytes, hipStream_t stream, void * out_f16) {
    const bool q8 = kv.k_q8 != nullptr;
    if (q == nullptr || (out == nullptr && out_f16 == nullptr) || work == nullptr || tokens <= 0 || seq < tokens ||
        nh <= 0 || nkv <= 0 || nh / nkv != kGqaGroup || nh % nkv != 0 || hd != kGqaHd) {
        return false;
    }
    if (q8 ? (kv.k_scales == nullptr || kv.v_q4 == nullptr || kv.v_scales == nullptr ||
              (kv.k16 == nullptr) != (kv.v16 == nullptr) || (kv.k16 != nullptr && kv.window <= 0))
           : (kv.k_f32 == nullptr || kv.v_f32 == nullptr)) {
        return false;
    }
    long long bps = 0;
    const int splits = gqa_splits(tokens, seq, nkv, bps);
    if ((size_t) tokens * nh * splits * (hd + 2) * sizeof(float) > work_bytes) {
        return false;
    }
    const KvArgs args{kv.k_f32,
                      kv.v_f32,
                      kv.k_q8,
                      static_cast<const __half *>(kv.k_scales),
                      kv.v_q4,
                      static_cast<const __half *>(kv.v_scales),
                      static_cast<const __half *>(kv.k16),
                      static_cast<const __half *>(kv.v16),
                      kv.window > 0 ? kv.window : 1,
                      kv.k_q4};
    const int tq = gqa_tq(tokens);
    const dim3 grid((unsigned) ((tokens + tq - 1) / tq), (unsigned) nkv, (unsigned) splits);
    auto * w = static_cast<float *>(work);
#define OMPH_GQA_LAUNCH(Q8, TQ)                                                              \
    attention_gqa_kernel<Q8, TQ><<<grid, kGqaHd, 0, stream>>>(q, args, w, tokens, seq, (int) nh, \
                                                              (int) nkv, scale, splits, bps)
    if (q8) {
        if (tq == 4) {
            OMPH_GQA_LAUNCH(true, 4);
        } else {
            OMPH_GQA_LAUNCH(true, 1);
        }
    } else {
        if (tq == 4) {
            OMPH_GQA_LAUNCH(false, 4);
        } else {
            OMPH_GQA_LAUNCH(false, 1);
        }
    }
#undef OMPH_GQA_LAUNCH
    attention_gqa_merge_kernel<<<dim3((unsigned) tokens, (unsigned) nh), kGqaHd, 0, stream>>>(
        w, gate, out, static_cast<__half *>(out_f16), (int) nh, splits, v_rotated);
    return hipGetLastError() == hipSuccess;
}

} // namespace omph::kernels
