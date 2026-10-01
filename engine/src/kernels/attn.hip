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
constexpr int kGqaMaxSplits = 256;  // the merge kernel's LDS
constexpr int64_t kDecTargetSplits = 64;   // key chunks at the context capacity (#136)
constexpr int64_t kDecMinChunk = 16;       // keys per chunk at least (one block)

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

// Dimensions [16c, 16c + 16) of the V row of position p (dequantized; inside
// the FP16 window: the exact copy).
template <bool Q8>
__device__ inline void gqa_load_v16(const KvArgs & kv, const long long p, const long long kvh,
                                    const long long nkv, const int c, const long long seq,
                                    float * v) {
    const long long row = p * nkv + kvh;
    if constexpr (Q8) {
        if (kv.v16 != nullptr && p + kv.window >= seq) {
            const uint4 * src = reinterpret_cast<const uint4 *>(
                kv.v16 + ((p % kv.window) * nkv + kvh) * kGqaHd + c * 16);
            uint4 h[2] = {src[0], src[1]};
            const __half * hv = reinterpret_cast<const __half *>(h);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                v[i] = __half2float(hv[i]);
            }
            return;
        }
        // byte j of block b holds dims 32b + j (low nibble) and 32b + 16 + j (high)
        const float sc = __half2float(kv.v_scales[row * (kGqaHd / kKvBlock) + c / 2]);
        const uint4 b4 = *reinterpret_cast<const uint4 *>(kv.v_q4 + row * (kGqaHd / 2) +
                                                          (c / 2) * (kKvBlock / 2));
        const uint8_t * bv4 = reinterpret_cast<const uint8_t *>(&b4);
        const int sh = (c & 1) ? 4 : 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            v[i] = (float) ((int) ((bv4[i] >> sh) & 0xF) - 8) * sc;
        }
    } else {
        const float4 * src = reinterpret_cast<const float4 *>(kv.v_f32 + row * kGqaHd + c * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 f = src[i];
            v[4 * i] = f.x;
            v[4 * i + 1] = f.y;
            v[4 * i + 2] = f.z;
            v[4 * i + 3] = f.w;
        }
    }
}

// xor-shuffles within the 16 lanes of a row through DPP (row_xmask): the
// ds_bpermute of __shfl_xor serialized the softmax on LDS round trips (#97).
template <int Mask>
__device__ inline float row_xor(const float x) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0x160 | Mask, 0xF, 0xF,
                                                   false));
}
__device__ inline float row_max16(float x) {
    x = fmaxf(x, row_xor<1>(x));
    x = fmaxf(x, row_xor<2>(x));
    x = fmaxf(x, row_xor<4>(x));
    return fmaxf(x, row_xor<8>(x));
}
__device__ inline float row_sum16(float x) {
    x += row_xor<1>(x);
    x += row_xor<2>(x);
    x += row_xor<4>(x);
    return x + row_xor<8>(x);
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
    // Q rows in 16-dimension chunks padded to 20 floats: 16-byte aligned, so a
    // chunk is four ds_read_b128, and the 16 lanes of a key (different chunks)
    // land on disjoint banks (it was 17: sixteen 4-byte reads, #97 / #136).
    constexpr int kQc = 20;
    __shared__ float qs[kRows][16 * kQc];
    __shared__ float sc[kRows][kGqaKb];
    __shared__ float corr[kRows];
    __shared__ float vs[kGqaKb][kGqaHd];  // the block's V, staged by 16-byte loads (#136)

    const int d = (int) threadIdx.x;
    const long long t0 = (long long) blockIdx.x * TQ;
    const int kvh = (int) blockIdx.y;
    const int split = (int) blockIdx.z;
    const int nt = (int) (tokens - t0 < TQ ? tokens - t0 : TQ);
    const int R = nt * G;
    const long long qpos0 = seq - tokens + t0;  // cache position of the tile's first query

    for (int r = 0; r < kRows; ++r) {
        const long long h = (long long) kvh * G + r % G;
        // scores in log2 units: exp2 in the softmax and the merge (#136)
        qs[r][(d / 16) * kQc + d % 16] =
            r < R ? q[((t0 + r / G) * nh + h) * kGqaHd + d] * (scale * 1.44269504088896341f)
                  : 0.0f;
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
    // The 16 lanes of group d / 16 own rows d / 16 + 16 i: their running max
    // and sum, the same in the 16 lanes (#136; one thread per row, looping
    // over the keys, was the latency of every block).
    constexpr int kRowSets = (kRows + 15) / 16;
    float m_run[kRowSets];
    float l_run[kRowSets];
#pragma unroll
    for (int i = 0; i < kRowSets; ++i) {
        m_run[i] = -INFINITY;
        l_run[i] = 0.0f;
    }
    __syncthreads();

    // Thread (key d / 16, chunk d % 16) loads 16 dims of one key's K and V with
    // 16-byte reads (V was 32 scalar loads per thread, #136). Loading a block
    // ahead measured no faster.
    const int jj = d / 16;
    const int c = d % 16;
    float kc[16];
    float vc[16];
    const auto load_block = [&](const long long kb, float * kd, float * vd) {
        const long long s = kb * kGqaKb + jj;
        if (kb < kb1 && s < n_keys) {
            gqa_load_k16<Q8>(kv, s, kvh, nkv, c, seq, kd);
            gqa_load_v16<Q8>(kv, s, kvh, nkv, c, seq, vd);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                kd[i] = 0.0f;
                vd[i] = 0.0f;
            }
        }
    };
    for (long long kb = kb0; kb < kb1; ++kb) {
        load_block(kb, kc, vc);
        const long long s0 = kb * kGqaKb;
        const int bs = (int) (n_keys - s0 < kGqaKb ? n_keys - s0 : kGqaKb);
        // this block's V into LDS for the per-dimension accumulation below
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            *reinterpret_cast<float4 *>(&vs[jj][c * 16 + i]) =
                make_float4(vc[i], vc[i + 1], vc[i + 2], vc[i + 3]);
        }
        // Scores: each thread dots its 16 dimensions of K with the matching
        // chunk of every row; the 16 lanes of a key then reduce through DPP.
        {
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                float part = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    part = fmaf(qs[r][c * kQc + i], kc[i], part);
                }
                part = row_sum16(part);  // the 16 chunks of the key, through DPP
                if (c == 0) {
                    sc[r][jj] = (jj < bs && s0 + jj <= qpos0 + r / G) ? part : -INFINITY;
                }
            }
        }
        __syncthreads();
        // Online softmax, 16 lanes per row (DPP reductions); scores become
        // weights in place. A row with no key yet keeps m = -inf, weights 0.
#pragma unroll
        for (int i = 0; i < kRowSets; ++i) {
            const int r = d / 16 + 16 * i;
            if (r < kRows) {  // uniform in the 16 lanes
                const int j = d % 16;
                const float x = sc[r][j];
                const float m_new = fmaxf(m_run[i], row_max16(x));
                const bool none = m_new == -INFINITY;
                const float m_use = none ? 0.0f : m_new;
                const float c = none ? 1.0f : exp2f(m_run[i] - m_use);
                const float w = exp2f(x - m_use);  // exp2(-inf) = 0
                sc[r][j] = w;
                l_run[i] = l_run[i] * c + row_sum16(w);
                m_run[i] = m_new;
                if (j == 0) {
                    corr[r] = c;
                }
            }
        }
        __syncthreads();
        // Each V value is read once and applied to every row of the tile.
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            acc[r] *= corr[r];
        }
#pragma unroll
        for (int j = 0; j < kGqaKb; ++j) {  // weights past bs are 0, V too
            const float vj = vs[j][d];
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                acc[r] = fmaf(sc[r][j], vj, acc[r]);
            }
        }
        __syncthreads();  // sc, corr and vs are rewritten by the next block
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
#pragma unroll
    for (int i = 0; i < kRowSets; ++i) {
        const int r = d / 16 + 16 * i;
        if (r < R && d % 16 == 0) {
            const long long h = (long long) kvh * G + r % G;
            float * dst = work + (((t0 + r / G) * nh + h) * splits + split) * (kGqaHd + 2);
            dst[0] = m_run[i];
            dst[1] = l_run[i];
        }
    }
}

// --- WMMA prefill attention (#97) ------------------------------------------
//
// A workgroup takes 16 query tokens of one KV head: 6 waves, wave g the query
// head kvh * 6 + g. Per tile of 32 keys, K and V are dequantized to f16 in LDS
// once for the 6 heads (V transposed, so both WMMA operands are 16-byte
// loads), each wave computes its 16 x 32 scores with v_wmma (q in registers,
// f16, prescaled), updates the online softmax in registers (the score tiles'
// rows sit in the same lanes as the output accumulators' rows), passes P
// through LDS into the A layout and accumulates O (16 x 256) += P V. The
// output goes to `work` as one partial (m, l, acc) per (token, head), which
// the merge kernel (splits = 1) normalizes, un-rotates and gates.
typedef _Float16 wmma_half8 __attribute__((ext_vector_type(8)));
typedef float wmma_float8 __attribute__((ext_vector_type(8)));

constexpr int kFaTokens = 16;            // query tokens per workgroup (one WMMA row tile)
constexpr int kFaKeys = 32;              // keys per tile (16: slower, 64: over the LDS)
constexpr int kFaSt = kFaKeys / 16;      // 16-key WMMA tiles per key tile
constexpr int kFaThreads = 32 * kGqaGroup;
constexpr int kFaKLd = kGqaHd + 8;       // Ks row stride (halves): conflict-free 16-lane reads
constexpr int kFaVLd = kFaKeys + 8;      // Vt row stride
constexpr int kFaPLd = kFaKeys + 8;      // Ps row stride

// grid: (ceil(tokens / 16), nkv); 192 threads. Scores are in log2 units (q is
// prescaled by scale * log2(e)): with one split the merge only divides by l.
template <bool Q8>
__global__ __launch_bounds__(kFaThreads) void attention_wmma_kernel(
    const float * __restrict__ q, const KvArgs kv, float * __restrict__ work,
    const long long tokens, const long long seq, const int nh, const int nkv,
    const float scale) {
    __shared__ _Float16 ks[kFaKeys * kFaKLd];
    __shared__ _Float16 vt[kGqaHd * kFaVLd];
    __shared__ _Float16 ps[kGqaGroup][kFaTokens * kFaPLd];
    const int tid = (int) threadIdx.x;
    const int lane = tid & 31;
    const int g = tid >> 5;
    const int kvh = (int) blockIdx.y;
    const int h = kvh * kGqaGroup + g;
    const long long t0 = (long long) blockIdx.x * kFaTokens;
    const int nt = (int) (tokens - t0 < kFaTokens ? tokens - t0 : kFaTokens);
    const long long qpos0 = seq - tokens + t0;  // cache position of the tile's first query

    // q as 16 A operands (k16 steps over the 256 dims): lane -> token lane % 16,
    // dims (lane / 16) * 8 .. + 7 of each step; prescaled, rows past the end 0
    wmma_half8 qa[kGqaHd / 16];
    {
        const int r = lane & 15;
        const bool in = r < nt;
        const float * qr = q + ((t0 + (in ? r : 0)) * nh + h) * kGqaHd + (lane >> 4) * 8;
#pragma unroll
        for (int s = 0; s < kGqaHd / 16; ++s) {
            const float4 a = *reinterpret_cast<const float4 *>(qr + s * 16);
            const float4 b = *reinterpret_cast<const float4 *>(qr + s * 16 + 4);
            const float f = in ? scale * 1.44269504088896341f : 0.0f;
            qa[s] = wmma_half8{(_Float16) (a.x * f), (_Float16) (a.y * f), (_Float16) (a.z * f),
                               (_Float16) (a.w * f), (_Float16) (b.x * f), (_Float16) (b.y * f),
                               (_Float16) (b.z * f), (_Float16) (b.w * f)};
        }
    }
    // O: 16 dim tiles; lane -> dim lane % 16 of the tile, rows (lane / 16) * 8 + v
    wmma_float8 o[kGqaHd / 16];
#pragma unroll
    for (int j = 0; j < kGqaHd / 16; ++j) {
        o[j] = wmma_float8{};
    }
    // running max / sum of rows (lane / 16) * 8 + v, the same in the 16 lanes of a half
    float m_run[8];
    float l_run[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        m_run[v] = -INFINITY;
        l_run[v] = 0.0f;
    }
    const long long n_keys = qpos0 + nt;  // keys up to the tile's last query
    const long long n_tiles = (n_keys + kFaKeys - 1) / kFaKeys;
    _Float16 * pw = ps[g];
    for (long long kt = 0; kt < n_tiles; ++kt) {
        const long long s0 = kt * kFaKeys;
        // K and V of keys s0 .. s0 + 31 into LDS as f16 (positions past n_keys: 0)
        __syncthreads();  // the previous tile is consumed
        // (one 16-dim chunk of one key per job; K and V in separate passes keep
        // the staging registers down)
        for (int job = tid; job < kFaKeys * (kGqaHd / 16); job += kFaThreads) {
            const int key = job / (kGqaHd / 16);
            const int c = job % (kGqaHd / 16);
            float kc[16];
            if (s0 + key < n_keys) {
                gqa_load_k16<Q8>(kv, s0 + key, kvh, nkv, c, seq, kc);
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    kc[i] = 0.0f;
                }
            }
            wmma_half8 k0;
            wmma_half8 k1;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                k0[i] = (_Float16) kc[i];
                k1[i] = (_Float16) kc[8 + i];
            }
            *reinterpret_cast<wmma_half8 *>(&ks[key * kFaKLd + c * 16]) = k0;
            *reinterpret_cast<wmma_half8 *>(&ks[key * kFaKLd + c * 16 + 8]) = k1;
        }
        for (int job = tid; job < kFaKeys * (kGqaHd / 16); job += kFaThreads) {
            const int key = job / (kGqaHd / 16);
            const int c = job % (kGqaHd / 16);
            float vc[16];
            if (s0 + key < n_keys) {
                gqa_load_v16<Q8>(kv, s0 + key, kvh, nkv, c, seq, vc);
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    vc[i] = 0.0f;
                }
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                vt[(c * 16 + i) * kFaVLd + key] = (_Float16) vc[i];
            }
        }
        __syncthreads();
        // S (16 x 32) = q k^T, two key tiles; lane -> key lane % 16, rows (lane / 16) * 8 + v
        wmma_float8 sc[kFaSt];
#pragma unroll
        for (int jt = 0; jt < kFaSt; ++jt) {
            sc[jt] = wmma_float8{};
        }
#pragma unroll
        for (int s = 0; s < kGqaHd / 16; ++s) {
#pragma unroll
            for (int jt = 0; jt < kFaSt; ++jt) {
                const wmma_half8 kb = *reinterpret_cast<const wmma_half8 *>(
                    &ks[(jt * 16 + (lane & 15)) * kFaKLd + s * 16 + (lane >> 4) * 8]);
                sc[jt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[s], kb, sc[jt]);
            }
        }
        // causal mask and the online softmax of this lane's 8 rows
        float corr[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int r = (lane >> 4) * 8 + v;
            const long long qpos = qpos0 + r;
            float x[kFaSt];
#pragma unroll
            for (int jt = 0; jt < kFaSt; ++jt) {
                const long long key = s0 + jt * 16 + (lane & 15);
                x[jt] = (r < nt && key <= qpos) ? sc[jt][v] : -INFINITY;
            }
            float mx = x[0];
#pragma unroll
            for (int jt = 1; jt < kFaSt; ++jt) {
                mx = fmaxf(mx, x[jt]);
            }
            mx = row_max16(mx);
            const float m_new = fmaxf(m_run[v], mx);
            // branch-free: a row with no key yet (or a padding row) keeps m = -inf,
            // corr = 1 and weights 0
            const bool none = m_new == -INFINITY;
            const float m_use = none ? 0.0f : m_new;
            corr[v] = none ? 1.0f : exp2f(m_run[v] - m_use);
            float sum = 0.0f;
#pragma unroll
            for (int jt = 0; jt < kFaSt; ++jt) {
                const float w = exp2f(x[jt] - m_use);  // exp2(-inf) = 0
                sc[jt][v] = w;
                sum += w;
            }
            sum = row_sum16(sum);
            l_run[v] = l_run[v] * corr[v] + sum;
            m_run[v] = m_new;
        }
        // P to the A layout through this wave's LDS: row r, keys contiguous
#pragma unroll
        for (int jt = 0; jt < kFaSt; ++jt) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                pw[((lane >> 4) * 8 + v) * kFaPLd + jt * 16 + (lane & 15)] = (_Float16) sc[jt][v];
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
        for (int j = 0; j < kGqaHd / 16; ++j) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                o[j][v] *= corr[v];
            }
        }
#pragma unroll
        for (int s = 0; s < kFaKeys / 16; ++s) {
            const wmma_half8 pa = *reinterpret_cast<const wmma_half8 *>(
                &pw[(lane & 15) * kFaPLd + s * 16 + (lane >> 4) * 8]);
#pragma unroll
            for (int j = 0; j < kGqaHd / 16; ++j) {
                const wmma_half8 vb = *reinterpret_cast<const wmma_half8 *>(
                    &vt[(j * 16 + (lane & 15)) * kFaVLd + s * 16 + (lane >> 4) * 8]);
                o[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pa, vb, o[j]);
            }
        }
    }
    // the partial of each (token, head), splits = 1: m, l, acc[hd]
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int r = (lane >> 4) * 8 + v;
        if (r < nt) {
            float * dst = work + ((t0 + r) * nh + h) * (kGqaHd + 2);
#pragma unroll
            for (int j = 0; j < kGqaHd / 16; ++j) {
                dst[2 + j * 16 + (lane & 15)] = o[j][v];
            }
            if ((lane & 15) == 0) {
                dst[0] = m_run[v];
                dst[1] = l_run[v];
            }
        }
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
        // m in log2 units (both attention kernels prescale q by log2(e))
        const float f = ms[s] == -INFINITY ? 0.0f : exp2f(ms[s] - m);
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

// Query tokens per workgroup: a decode step keeps the six-row form, a
// verification of 2-4 tokens (or a longer run) shares each K/V read among
// its tokens (#136). A row's arithmetic does not depend on the tile size.
int gqa_tq(const int64_t tokens) {
    return tokens == 1 ? 1 : (tokens == 2 ? 2 : 4);
}

// Splits for a call: enough workgroups to fill the GPU, never more than the key
// blocks of the longest tile. With one-token tiles (decode steps and
// speculative verifications) the partition depends on the sequence only: a
// token verified in a batch then sums its keys in the same order as when it is
// decoded alone, so speculative greedy equals plain greedy bit for bit (#97;
// it had depended on the batch size, which a near-tie exposed).
int gqa_splits(const int64_t tokens, const int64_t seq, const int64_t nkv,
               long long & blocks_per_split) {
    const long long tq = gqa_tq(tokens);
    const long long tiles = tq == 1 ? 1 : (tokens + tq - 1) / tq;
    const long long kb_max = (seq + kGqaKb - 1) / kGqaKb;
    long long splits = (kGqaTargetWgs + tiles * nkv - 1) / (tiles * nkv);
    splits = splits < 1 ? 1 : (splits > kGqaMaxSplits ? kGqaMaxSplits : splits);
    splits = splits > kb_max ? kb_max : splits;
    blocks_per_split = (kb_max + splits - 1) / splits;
    return (int) ((kb_max + blocks_per_split - 1) / blocks_per_split);
}

} // namespace

namespace {

// grid: (count, layers, 2 = k / v); a block copies one ring row.
__global__ void kv_ring_copy_kernel(const uint4 * __restrict__ src_k, const uint4 * __restrict__ src_v,
                                    uint4 * __restrict__ dst_k, uint4 * __restrict__ dst_v,
                                    const long long window, const long long row_vec,
                                    const long long pos_first) {
    const long long slot = (pos_first + blockIdx.x) % window;
    const long long off = ((long long) blockIdx.y * window + slot) * row_vec;
    const uint4 * src = (blockIdx.z == 0 ? src_k : src_v) + off;
    uint4 * dst = (blockIdx.z == 0 ? dst_k : dst_v) + off;
    for (long long i = threadIdx.x; i < row_vec; i += blockDim.x) {
        dst[i] = src[i];
    }
}

} // namespace

bool kv_ring_copy(const void * src_k, const void * src_v, void * dst_k, void * dst_v,
                  const int64_t layers, const int64_t window, const int64_t row_bytes,
                  const int64_t pos_first, const int64_t count, hipStream_t stream) {
    if (src_k == nullptr || src_v == nullptr || dst_k == nullptr || dst_v == nullptr ||
        layers <= 0 || window <= 0 || row_bytes <= 0 || row_bytes % 16 != 0 || pos_first < 0 ||
        count < 0 || count > window) {
        return false;
    }
    if (count == 0) {
        return true;
    }
    const dim3 grid((unsigned) count, (unsigned) layers, 2);
    kv_ring_copy_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const uint4 *>(src_k), static_cast<const uint4 *>(src_v),
        static_cast<uint4 *>(dst_k), static_cast<uint4 *>(dst_v), (long long) window,
        (long long) (row_bytes / 16), (long long) pos_first);
    return hipGetLastError() == hipSuccess;
}

int64_t attention_key_chunk(const int64_t max_seq) {
    const int64_t per = (max_seq + kDecTargetSplits - 1) / kDecTargetSplits;
    const int64_t chunk = (per + kGqaKb - 1) / kGqaKb * kGqaKb;
    return chunk < kDecMinChunk ? kDecMinChunk : chunk;
}

size_t attention_gqa_work_bytes(const int64_t max_tokens, const int64_t nh, const int64_t nkv,
                                const int64_t hd, const int64_t max_seq, const int64_t key_chunk) {
    // Worst case over the token counts a call can have: up to the sequence
    // capacity in key chunks (one split from kFaTokens tokens on, WMMA), or
    // at any sequence length without chunks (the split count only grows with
    // the sequence up to kGqaMaxSplits).
    size_t worst = 0;
    for (int64_t t = 1; t <= max_tokens; ++t) {
        long long bps = 0;
        const int64_t splits =
            key_chunk > 0 ? (t >= kFaTokens ? 1 : (max_seq + key_chunk - 1) / key_chunk)
                          : gqa_splits(t, (int64_t) 1 << 40, nkv, bps);
        const size_t rows = (size_t) t * nh * splits;
        worst = rows > worst ? rows : worst;
    }
    return worst * (size_t) (hd + 2) * sizeof(float);
}

bool attention_gqa(const float * q, const KvCache & kv, const float * gate, float * out,
                   const int64_t tokens, const int64_t seq, const int64_t nh, const int64_t nkv,
                   const int64_t hd, const float scale, const bool v_rotated, void * work,
                   const size_t work_bytes, hipStream_t stream, void * out_f16,
                   const bool allow_wmma, const int64_t key_chunk) {
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
    const bool fa = allow_wmma && tokens >= kFaTokens;
    long long bps = 0;
    int splits = 1;
    if (!fa && key_chunk > 0 && tokens < kFaTokens) {
        // fixed chunks of keys: a query's partition depends on its position only
        if (key_chunk % kGqaKb != 0) {
            return false;
        }
        bps = key_chunk / kGqaKb;
        const long long kb_max = (seq + kGqaKb - 1) / kGqaKb;
        splits = (int) ((kb_max + bps - 1) / bps);
    } else if (!fa) {
        splits = gqa_splits(tokens, seq, nkv, bps);
    }
    if (splits > kGqaMaxSplits) {
        return false;
    }
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
    auto * w = static_cast<float *>(work);
    if (fa) {
        const dim3 fgrid((unsigned) ((tokens + kFaTokens - 1) / kFaTokens), (unsigned) nkv);
        if (q8) {
            attention_wmma_kernel<true><<<fgrid, kFaThreads, 0, stream>>>(q, args, w, tokens, seq,
                                                                         (int) nh, (int) nkv, scale);
        } else {
            attention_wmma_kernel<false><<<fgrid, kFaThreads, 0, stream>>>(
                q, args, w, tokens, seq, (int) nh, (int) nkv, scale);
        }
        attention_gqa_merge_kernel<<<dim3((unsigned) tokens, (unsigned) nh), kGqaHd, 0, stream>>>(
            w, gate, out, static_cast<__half *>(out_f16), (int) nh, 1, v_rotated);
        return hipGetLastError() == hipSuccess;
    }
    const int tq = gqa_tq(tokens);
    const dim3 grid((unsigned) ((tokens + tq - 1) / tq), (unsigned) nkv, (unsigned) splits);
#define OMPH_GQA_LAUNCH(Q8, TQ)                                                              \
    attention_gqa_kernel<Q8, TQ><<<grid, kGqaHd, 0, stream>>>(q, args, w, tokens, seq, (int) nh, \
                                                              (int) nkv, scale, splits, bps)
    if (q8) {
        if (tq == 4) {
            OMPH_GQA_LAUNCH(true, 4);
        } else if (tq == 2) {
            OMPH_GQA_LAUNCH(true, 2);
        } else {
            OMPH_GQA_LAUNCH(true, 1);
        }
    } else {
        if (tq == 4) {
            OMPH_GQA_LAUNCH(false, 4);
        } else if (tq == 2) {
            OMPH_GQA_LAUNCH(false, 2);
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
