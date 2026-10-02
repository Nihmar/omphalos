// Fused dequant + dot GEMVs for the decode (PLAN.md §10.1), one kernel per
// GGUF type on the repacked layouts of format/repack.hh (and the GGUF bytes for
// IQ1_M / BF16).
//
// A warp computes a few consecutive output rows (8 warps per workgroup). The
// activation vector is staged in LDS as f16; each lane walks the row's
// sub-blocks in strides of 32 with aligned 16-byte weight loads, decodes the
// codes in registers into half2 and accumulates with v_dot2_f32_f16, so the
// quantized bytes are read exactly once. The *_b4 kernels do four tokens per
// weight read (prefill head, MTP verification).
#include "kernels/gemv.hh"

#include "format/repack.hh"
#include "kernels/iq_tables.hh"
#include "kernels/quant_decode.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <utility>
#include <vector>

namespace omph::kernels {
namespace {

// Packed-f16 helpers for the codebook GEMVs: a u32 holds two halves; the sign
// of each half is its bit 15, so a (positive) codebook value is negated by
// OR-ing its sign in, and v_dot2_f32_f16 does two exact products (small
// integers times f16) accumulated in f32 per instruction.
using qd::half2_v;
using qd::grid_pair_to_half2;
using qd::iq3xxs_group_signs;
using qd::iq4_meta;
using qd::q4k_meta;

__device__ inline float dot2(const uint32_t w, const uint32_t x, const float acc) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_v, w), __builtin_bit_cast(half2_v, x),
                                  acc, false);
}

// Staged x for the compute-bound lane-per-sub-block kernels (IQ2_S, IQ2_XS,
// IQ3_XXS): 40 halves per 32-weight sub-block. With the natural 32 (16 words)
// the lanes' 16-byte loads hit the LDS banks 8 ways; 20 words leaves the
// unavoidable 2. Measured per type (#75): the memory-bound kernels gain nothing
// and IQ3_S loses occupancy to the bigger chunk, so they keep 32.
constexpr long long kXPitch = 40;

constexpr long long kQ4kChunkSub = 256;  // 8192 weights staged per LDS refill
constexpr long long kQ4kRows = 4;        // output rows computed per warp

// Staged position p of each 8-weight run holds weight kQ4kXOrder(p): the even
// weights, then the odd ones. A word of nibbles gives the even weights in its
// low nibbles and the odd ones in its high nibbles, so each half2 of codes
// meets its activations as one aligned half2 (#94).
__device__ inline long long q4k_x_order(const long long p) {
    const long long j = p & 7;
    return (p & ~7LL) + (j < 4 ? 2 * j : 2 * (j - 4) + 1);
}

__global__ void gemv_q4k_kernel(const uint8_t * __restrict__ qs,
                                const uint8_t * __restrict__ meta, const __half * __restrict__ x,
                                float * __restrict__ y, const long long rows,
                                const long long n_sub) {
    extern __shared__ __half xs[];  // activation chunk, in q4k_x_order

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    // kQ4kRows consecutive output rows per warp: one activation value feeds all
    // of them and the accumulators give independent chains.
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kQ4kRows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kQ4kRows] = {};
    // persistent warps (#63): rows of one chunk stage x once, then the warps
    // take row groups of kQ4kRows in turn (the launch sizes the grid to what fits)
    if (n_sub <= kQ4kChunkSub) {
        {
            const long long base = 0;
            const long long len = n_sub;
            for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
                xs[i] = x[base * 32 + q4k_x_order(i)];
            }
        }
        __syncthreads();
        const long long n_groups = (rows + kQ4kRows - 1) / kQ4kRows;
        const long long stride = (long long) gridDim.x * (blockDim.x / 32);
        for (long long grp = (long long) blockIdx.x * (blockDim.x / 32) + warp; grp < n_groups; grp += stride) {
            const long long row0 = grp * kQ4kRows;
            float acc[kQ4kRows] = {};
            {
                const long long base = 0;
                const long long len = n_sub;
                for (long long s = base + lane; s < base + len; s += 32) {
                    const long long blk = s / 8;
                    const int sub = (int) (s % 8);
                    const uint4 * xv = reinterpret_cast<const uint4 *>(xs + (s - base) * 32);
                    const uint4 xq[4] = {xv[0], xv[1], xv[2], xv[3]};
                    // (ds * q - ms) . x = ds * (q . x) - ms * sum(x): sum(x) is shared by
                    // the rows.
                    constexpr uint32_t kOnes = 0x3C003C00u;  // half2 (1, 1)
                    float xsum = 0.0f;
        #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        xsum = dot2(xq[i].x, kOnes, dot2(xq[i].y, kOnes, xsum));
                        xsum = dot2(xq[i].z, kOnes, dot2(xq[i].w, kOnes, xsum));
                    }
                    uint4 q[kQ4kRows];
                    float ds[kQ4kRows];
                    float ms[kQ4kRows];
        #pragma unroll
                    for (int r = 0; r < kQ4kRows; ++r) {
                        const long long row = row0 + r <= rlast ? row0 + r : rlast;
                        q[r] = *reinterpret_cast<const uint4 *>(qs + row * blocks * 128 + s * 16);
                        q4k_meta(*reinterpret_cast<const uint4 *>(meta + (row * blocks + blk) * 16), sub,
                                 ds[r], ms[r]);
                    }
        #pragma unroll
                    for (int r = 0; r < kQ4kRows; ++r) {
                        const uint32_t w[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
                        float pe = 0.0f;  // even weights
                        float po = 0.0f;  // odd weights
        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const uint32_t lo = w[i] & 0x0F0F0F0Fu;
                            const uint32_t hi = (w[i] >> 4) & 0x0F0F0F0Fu;
                            pe = dot2(grid_pair_to_half2<0x04010400u>(lo), xq[i].x, pe);
                            po = dot2(grid_pair_to_half2<0x04010400u>(hi), xq[i].z, po);
                            pe = dot2(grid_pair_to_half2<0x04030402u>(lo), xq[i].y, pe);
                            po = dot2(grid_pair_to_half2<0x04030402u>(hi), xq[i].w, po);
                        }
                        acc[r] += fmaf(ds[r], pe + po, -ms[r] * xsum);
                    }
                }
            }
    #pragma unroll
        for (int r = 0; r < kQ4kRows; ++r) {
            float a = acc[r];
    #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a += __shfl_down(a, off);
            }
            if (lane == 0 && row0 + r < rows) {
                y[row0 + r] = a;
            }
        }
        }
        return;
    }
    for (long long base = 0; base < n_sub; base += kQ4kChunkSub) {
        const long long len = n_sub - base < kQ4kChunkSub ? n_sub - base : kQ4kChunkSub;
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[i] = x[base * 32 + q4k_x_order(i)];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const long long blk = s / 8;
            const int sub = (int) (s % 8);
            const uint4 * xv = reinterpret_cast<const uint4 *>(xs + (s - base) * 32);
            const uint4 xq[4] = {xv[0], xv[1], xv[2], xv[3]};
            // (ds * q - ms) . x = ds * (q . x) - ms * sum(x): sum(x) is shared by
            // the rows.
            constexpr uint32_t kOnes = 0x3C003C00u;  // half2 (1, 1)
            float xsum = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                xsum = dot2(xq[i].x, kOnes, dot2(xq[i].y, kOnes, xsum));
                xsum = dot2(xq[i].z, kOnes, dot2(xq[i].w, kOnes, xsum));
            }
            uint4 q[kQ4kRows];
            float ds[kQ4kRows];
            float ms[kQ4kRows];
#pragma unroll
            for (int r = 0; r < kQ4kRows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                q[r] = *reinterpret_cast<const uint4 *>(qs + row * blocks * 128 + s * 16);
                q4k_meta(*reinterpret_cast<const uint4 *>(meta + (row * blocks + blk) * 16), sub,
                         ds[r], ms[r]);
            }
#pragma unroll
            for (int r = 0; r < kQ4kRows; ++r) {
                const uint32_t w[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
                float pe = 0.0f;  // even weights
                float po = 0.0f;  // odd weights
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t lo = w[i] & 0x0F0F0F0Fu;
                    const uint32_t hi = (w[i] >> 4) & 0x0F0F0F0Fu;
                    pe = dot2(grid_pair_to_half2<0x04010400u>(lo), xq[i].x, pe);
                    po = dot2(grid_pair_to_half2<0x04010400u>(hi), xq[i].z, po);
                    pe = dot2(grid_pair_to_half2<0x04030402u>(lo), xq[i].y, pe);
                    po = dot2(grid_pair_to_half2<0x04030402u>(hi), xq[i].w, po);
                }
                acc[r] += fmaf(ds[r], pe + po, -ms[r] * xsum);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kQ4kRows; ++r) {
        float a = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a += __shfl_down(a, off);
        }
        if (lane == 0 && row0 + r < rows) {
            y[row0 + r] = a;
        }
    }
}

// The IQ4_NL codebook (omph::quant::kKvaluesIq4nl) + 128 as bytes, in four
// words: entries 0-3, 4-7, 8-11, 12-15.
constexpr uint32_t kIq4B0 = 0x3F2D1801u;
constexpr uint32_t kIq4B1 = 0x766A5D4Fu;
constexpr uint32_t kIq4B2 = 0xA6998D81u;
constexpr uint32_t kIq4B3 = 0xF1D9C5B5u;
constexpr uint32_t iq4_biased_word(const int w) {
    uint32_t v = 0;
    for (int j = 0; j < 4; ++j) {
        v |= (uint32_t) (uint8_t) (omph::quant::kKvaluesIq4nl[4 * w + j] + 128) << (8 * j);
    }
    return v;
}
static_assert(kIq4B0 == iq4_biased_word(0) && kIq4B1 == iq4_biased_word(1) &&
                  kIq4B2 == iq4_biased_word(2) && kIq4B3 == iq4_biased_word(3),
              "kIq4B* must be the IQ4_NL codebook + 128");

// Four 4-bit codes (one per byte) to their codebook values + 128, one per byte:
// two byte permutes look the low three bits up in each half of the table and a
// third picks the half by bit 3.
__device__ inline uint32_t iq4_lookup(const uint32_t codes) {
    const uint32_t idx = codes & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(kIq4B1, kIq4B0, idx);
    const uint32_t hi = __builtin_amdgcn_perm(kIq4B3, kIq4B2, idx);
    return __builtin_amdgcn_perm(hi, lo, 0x03020100u | ((codes & 0x08080808u) >> 1));
}

// Two biased codebook bytes (the bytes Sel picks) as a half2 of the values:
// 1024 + (v + 128) from the byte permute, minus 1152 (exact, |v| <= 127).
template <uint32_t Sel>
__device__ inline uint32_t iq4_pair_to_half2(const uint32_t g) {
    const uint32_t biased = __builtin_amdgcn_perm(0x64646464u, g, Sel);
    const half2_v v =
        __builtin_bit_cast(half2_v, biased) - half2_v{(_Float16) 1152, (_Float16) 1152};
    return __builtin_bit_cast(uint32_t, v);
}

constexpr long long kIq4ChunkSub = 256;  // 8192 weights staged per LDS refill
constexpr long long kIq4Rows = 4;        // output rows computed per warp

__global__ void gemv_iq4_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ meta,
                                const __half * __restrict__ x, float * __restrict__ y,
                                const long long rows, const long long n_sub,
                                const long long chunk_sub) {
    extern __shared__ __half xs[];  // activation chunk

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    // Each warp owns kIq4Rows consecutive output rows: the activation value is
    // loaded once and reused by all of them, and the accumulators give
    // independent chains.
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kIq4Rows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kIq4Rows] = {};
    // persistent warps (#63): rows of one chunk stage x once, then the warps
    // take row groups of kIq4Rows in turn (the launch sizes the grid to what fits)
    if (n_sub <= chunk_sub) {
        {
            const long long base = 0;
            const long long len = n_sub;
            for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
                xs[i] = x[base * 32 + i];
            }
        }
        __syncthreads();
        const long long n_groups = (rows + kIq4Rows - 1) / kIq4Rows;
        const long long stride = (long long) gridDim.x * (blockDim.x / 32);
        for (long long grp = (long long) blockIdx.x * (blockDim.x / 32) + warp; grp < n_groups; grp += stride) {
            const long long row0 = grp * kIq4Rows;
            float acc[kIq4Rows] = {};
            {
                const long long base = 0;
                const long long len = n_sub;
                for (long long s = base + lane; s < base + len; s += 32) {
                    const long long blk = s / 8;
                    const int sub = (int) (s % 8);
                    const uint4 * xv = reinterpret_cast<const uint4 *>(xs + (s - base) * 32);
                    const uint4 xq[4] = {xv[0], xv[1], xv[2], xv[3]};
                    // the low nibbles of word i are weights 4i..4i+3, the high ones
                    // 16+4i..16+4i+3
                    const uint32_t xl[8] = {xq[0].x, xq[0].y, xq[0].z, xq[0].w,
                                            xq[1].x, xq[1].y, xq[1].z, xq[1].w};
                    const uint32_t xh[8] = {xq[2].x, xq[2].y, xq[2].z, xq[2].w,
                                            xq[3].x, xq[3].y, xq[3].z, xq[3].w};
                    uint4 q[kIq4Rows];
                    uint2 mt[kIq4Rows];
        #pragma unroll
                    for (int r = 0; r < kIq4Rows; ++r) {
                        const long long row = row0 + r <= rlast ? row0 + r : rlast;  // clamped, never
                                                                                     // used when masked
                        q[r] = *reinterpret_cast<const uint4 *>(qs + row * blocks * 128 + s * 16);
                        mt[r] = *reinterpret_cast<const uint2 *>(meta + (row * blocks + blk) * 8);
                    }
        #pragma unroll
                    for (int r = 0; r < kIq4Rows; ++r) {
                        const uint32_t w[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
                        float pl = 0.0f;
                        float ph = 0.0f;
        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const uint32_t vl = iq4_lookup(w[i] & 0x0F0F0F0Fu);
                            const uint32_t vh = iq4_lookup((w[i] >> 4) & 0x0F0F0F0Fu);
                            pl = dot2(iq4_pair_to_half2<0x04010400u>(vl), xl[2 * i], pl);
                            ph = dot2(iq4_pair_to_half2<0x04010400u>(vh), xh[2 * i], ph);
                            pl = dot2(iq4_pair_to_half2<0x04030402u>(vl), xl[2 * i + 1], pl);
                            ph = dot2(iq4_pair_to_half2<0x04030402u>(vh), xh[2 * i + 1], ph);
                        }
                        acc[r] = fmaf(iq4_meta(mt[r], sub), pl + ph, acc[r]);
                    }
                }
            }
    #pragma unroll
        for (int r = 0; r < kIq4Rows; ++r) {
            float a = acc[r];
    #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a += __shfl_down(a, off);
            }
            if (lane == 0 && row0 + r < rows) {
                y[row0 + r] = a;
            }
        }
        }
        return;
    }
    for (long long base = 0; base < n_sub; base += chunk_sub) {
        const long long len = n_sub - base < chunk_sub ? n_sub - base : chunk_sub;
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[i] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const long long blk = s / 8;
            const int sub = (int) (s % 8);
            const uint4 * xv = reinterpret_cast<const uint4 *>(xs + (s - base) * 32);
            const uint4 xq[4] = {xv[0], xv[1], xv[2], xv[3]};
            // the low nibbles of word i are weights 4i..4i+3, the high ones
            // 16+4i..16+4i+3
            const uint32_t xl[8] = {xq[0].x, xq[0].y, xq[0].z, xq[0].w,
                                    xq[1].x, xq[1].y, xq[1].z, xq[1].w};
            const uint32_t xh[8] = {xq[2].x, xq[2].y, xq[2].z, xq[2].w,
                                    xq[3].x, xq[3].y, xq[3].z, xq[3].w};
            uint4 q[kIq4Rows];
            uint2 mt[kIq4Rows];
#pragma unroll
            for (int r = 0; r < kIq4Rows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;  // clamped, never
                                                                             // used when masked
                q[r] = *reinterpret_cast<const uint4 *>(qs + row * blocks * 128 + s * 16);
                mt[r] = *reinterpret_cast<const uint2 *>(meta + (row * blocks + blk) * 8);
            }
#pragma unroll
            for (int r = 0; r < kIq4Rows; ++r) {
                const uint32_t w[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
                float pl = 0.0f;
                float ph = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t vl = iq4_lookup(w[i] & 0x0F0F0F0Fu);
                    const uint32_t vh = iq4_lookup((w[i] >> 4) & 0x0F0F0F0Fu);
                    pl = dot2(iq4_pair_to_half2<0x04010400u>(vl), xl[2 * i], pl);
                    ph = dot2(iq4_pair_to_half2<0x04010400u>(vh), xh[2 * i], ph);
                    pl = dot2(iq4_pair_to_half2<0x04030402u>(vl), xl[2 * i + 1], pl);
                    ph = dot2(iq4_pair_to_half2<0x04030402u>(vh), xh[2 * i + 1], ph);
                }
                acc[r] = fmaf(iq4_meta(mt[r], sub), pl + ph, acc[r]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kIq4Rows; ++r) {
        float a = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a += __shfl_down(a, off);
        }
        if (lane == 0 && row0 + r < rows) {
            y[row0 + r] = a;
        }
    }
}


// IQ3_XXS: each 32-weight sub-block has 8 grid indices (4-byte entries of
// iq3xxs_grid), a 4-bit scale and four 7-bit sign indices packed in one word.
// Same outer-product shape as the other kernels: kIq3Rows rows per warp, the
// activation read scalar, the grid (1 KiB) and the sign table (128 B) in LDS.
constexpr long long kIq3ChunkSub = 256;  // 8192 weights staged per LDS refill
constexpr long long kIq3Rows = 2;        // output rows computed per warp

__global__ void gemv_iq3_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ aux,
                                const uint8_t * __restrict__ d, const __half * __restrict__ x,
                                float * __restrict__ y, const long long rows,
                                const long long n_sub, const long long chunk_sub) {
    extern __shared__ uint8_t smem[];
    __half * xs = reinterpret_cast<__half *>(smem);  // activation chunk
    uint32_t * grid = reinterpret_cast<uint32_t *>(smem + chunk_sub * kXPitch * 2);
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        grid[i] = omph::quant::kIq3xxsGrid[i];
    }

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kIq3Rows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kIq3Rows] = {};
    // persistent warps (#63): rows of one chunk stage x once, then the warps
    // take row groups of kIq3Rows in turn (the launch sizes the grid to what fits)
    if (n_sub <= kIq3ChunkSub) {
        {
            const long long base = 0;
            const long long len = n_sub;
            for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
                xs[(i / 32) * kXPitch + (i % 32)] = x[base * 32 + i];
            }
        }
        __syncthreads();
        const long long n_groups = (rows + kIq3Rows - 1) / kIq3Rows;
        const long long stride = (long long) gridDim.x * (blockDim.x / 32);
        for (long long grp = (long long) blockIdx.x * (blockDim.x / 32) + warp; grp < n_groups; grp += stride) {
            const long long row0 = grp * kIq3Rows;
            float acc[kIq3Rows] = {};
            {
                const long long base = 0;
                const long long len = n_sub;
                // software pipeline: the next sub-block's weights are loaded while this
                // one is computed (#63)
                uint2 qn[kIq3Rows];
                uint32_t an[kIq3Rows];
                __half dn[kIq3Rows];
                const auto fetch = [&](const long long s) {
        #pragma unroll
                    for (int r = 0; r < kIq3Rows; ++r) {
                        const long long row = row0 + r <= rlast ? row0 + r : rlast;
                        qn[r] = *reinterpret_cast<const uint2 *>(qs + row * blocks * 64 + s * 8);
                        an[r] = *reinterpret_cast<const uint32_t *>(aux + row * blocks * 32 + s * 4);
                        dn[r] = *reinterpret_cast<const __half *>(d + row * blocks * 2 + (s / 8) * 2);
                    }
                };
                if (base + lane < base + len) {
                    fetch(base + lane);
                }
                for (long long s = base + lane; s < base + len; s += 32) {
                    const uint4 * xv = reinterpret_cast<const uint4 *>(xs + (s - base) * kXPitch);
                    uint2 q[kIq3Rows];
                    uint32_t a[kIq3Rows];
                    float db[kIq3Rows];
        #pragma unroll
                    for (int r = 0; r < kIq3Rows; ++r) {
                        q[r] = qn[r];
                        a[r] = an[r];
                        db[r] = __half2float(dn[r]) * (0.5f + (float) (a[r] >> 28)) * 0.5f;
                    }
                    if (s + 32 < base + len) {
                        fetch(s + 32);
                    }
                    float part[kIq3Rows] = {};
        #pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const uint4 xq = xv[l];  // the 8 activations of this group, packed
        #pragma unroll
                        for (int r = 0; r < kIq3Rows; ++r) {
                            // same scheme as IQ3_S: byte pairs -> half2, signs OR-ed in
                            const uint32_t word = (l < 2 ? q[r].x : q[r].y) >> (16 * (l & 1));
                            const uint32_t b1 = grid[word & 0xFFu];
                            const uint32_t b2 = grid[(word >> 8) & 0xFFu];
                            const uint32_t sgn = iq3xxs_group_signs(a[r], l);
                            constexpr uint32_t kSign2 = 0x80008000u;
                            float p = part[r];
                            p = dot2((sgn & kSign2) | grid_pair_to_half2<0x04010400u>(b1), xq.x, p);
                            p = dot2(((sgn << 1) & kSign2) | grid_pair_to_half2<0x04030402u>(b1), xq.y, p);
                            p = dot2(((sgn << 2) & kSign2) | grid_pair_to_half2<0x04010400u>(b2), xq.z, p);
                            p = dot2(((sgn << 3) & kSign2) | grid_pair_to_half2<0x04030402u>(b2), xq.w, p);
                            part[r] = p;
                        }
                    }
        #pragma unroll
                    for (int r = 0; r < kIq3Rows; ++r) {
                        acc[r] = fmaf(db[r], part[r], acc[r]);
                    }
                }
            }
    #pragma unroll
        for (int r = 0; r < kIq3Rows; ++r) {
            float v = acc[r];
    #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down(v, off);
            }
            if (lane == 0 && row0 + r < rows) {
                y[row0 + r] = v;
            }
        }
        }
        return;
    }
    for (long long base = 0; base < n_sub; base += kIq3ChunkSub) {
        const long long len = n_sub - base < kIq3ChunkSub ? n_sub - base : kIq3ChunkSub;
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[(i / 32) * kXPitch + (i % 32)] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        // software pipeline: the next sub-block's weights are loaded while this
        // one is computed (#63)
        uint2 qn[kIq3Rows];
        uint32_t an[kIq3Rows];
        __half dn[kIq3Rows];
        const auto fetch = [&](const long long s) {
#pragma unroll
            for (int r = 0; r < kIq3Rows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                qn[r] = *reinterpret_cast<const uint2 *>(qs + row * blocks * 64 + s * 8);
                an[r] = *reinterpret_cast<const uint32_t *>(aux + row * blocks * 32 + s * 4);
                dn[r] = *reinterpret_cast<const __half *>(d + row * blocks * 2 + (s / 8) * 2);
            }
        };
        if (base + lane < base + len) {
            fetch(base + lane);
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const uint4 * xv = reinterpret_cast<const uint4 *>(xs + (s - base) * kXPitch);
            uint2 q[kIq3Rows];
            uint32_t a[kIq3Rows];
            float db[kIq3Rows];
#pragma unroll
            for (int r = 0; r < kIq3Rows; ++r) {
                q[r] = qn[r];
                a[r] = an[r];
                db[r] = __half2float(dn[r]) * (0.5f + (float) (a[r] >> 28)) * 0.5f;
            }
            if (s + 32 < base + len) {
                fetch(s + 32);
            }
            float part[kIq3Rows] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint4 xq = xv[l];  // the 8 activations of this group, packed
#pragma unroll
                for (int r = 0; r < kIq3Rows; ++r) {
                    // same scheme as IQ3_S: byte pairs -> half2, signs OR-ed in
                    const uint32_t word = (l < 2 ? q[r].x : q[r].y) >> (16 * (l & 1));
                    const uint32_t b1 = grid[word & 0xFFu];
                    const uint32_t b2 = grid[(word >> 8) & 0xFFu];
                    const uint32_t sgn = iq3xxs_group_signs(a[r], l);
                    constexpr uint32_t kSign2 = 0x80008000u;
                    float p = part[r];
                    p = dot2((sgn & kSign2) | grid_pair_to_half2<0x04010400u>(b1), xq.x, p);
                    p = dot2(((sgn << 1) & kSign2) | grid_pair_to_half2<0x04030402u>(b1), xq.y, p);
                    p = dot2(((sgn << 2) & kSign2) | grid_pair_to_half2<0x04010400u>(b2), xq.z, p);
                    p = dot2(((sgn << 3) & kSign2) | grid_pair_to_half2<0x04030402u>(b2), xq.w, p);
                    part[r] = p;
                }
            }
#pragma unroll
            for (int r = 0; r < kIq3Rows; ++r) {
                acc[r] = fmaf(db[r], part[r], acc[r]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kIq3Rows; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down(v, off);
        }
        if (lane == 0 && row0 + r < rows) {
            y[row0 + r] = v;
        }
    }
}


// IQ3_XXS for NT = 2..4 tokens (#126): the IQ3_S NT scheme (see there).
template <int NT, int R>
__global__ void gemv_iq3_ntg_kernel(const uint8_t * __restrict__ qs,
                                    const uint8_t * __restrict__ aux,
                                    const uint8_t * __restrict__ d, const __half * __restrict__ x,
                                    float * __restrict__ y, const long long rows,
                                    const long long n_sub) {
    __shared__ uint32_t grid[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        grid[i] = omph::quant::kIq3xxsGrid[i];
    }
    __syncthreads();
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * R;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    const long long k = n_sub * 32;

    float acc[R][NT] = {};
    if (row0 < rows) {
        for (long long s = lane; s < n_sub; s += 32) {
            uint2 q[R];
            uint32_t a[R];
            float db[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                q[r] = *reinterpret_cast<const uint2 *>(qs + row * blocks * 64 + s * 8);
                a[r] = *reinterpret_cast<const uint32_t *>(aux + row * blocks * 32 + s * 4);
                db[r] = __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 +
                                                                       (s / 8) * 2)) *
                        (0.5f + (float) (a[r] >> 28)) * 0.5f;
            }
            float part[R][NT] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint4 xq[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    xq[t] = reinterpret_cast<const uint4 *>(x + t * k + s * 32)[l];
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t word = (l < 2 ? q[r].x : q[r].y) >> (16 * (l & 1));
                    const uint32_t b1 = grid[word & 0xFFu];
                    const uint32_t b2 = grid[(word >> 8) & 0xFFu];
                    const uint32_t sgn = iq3xxs_group_signs(a[r], l);
                    constexpr uint32_t kSign2 = 0x80008000u;
                    const uint32_t w0 = (sgn & kSign2) | grid_pair_to_half2<0x04010400u>(b1);
                    const uint32_t w1 = ((sgn << 1) & kSign2) | grid_pair_to_half2<0x04030402u>(b1);
                    const uint32_t w2 = ((sgn << 2) & kSign2) | grid_pair_to_half2<0x04010400u>(b2);
                    const uint32_t w3 = ((sgn << 3) & kSign2) | grid_pair_to_half2<0x04030402u>(b2);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        float p = part[r][t];
                        p = dot2(w0, xq[t].x, p);
                        p = dot2(w1, xq[t].y, p);
                        p = dot2(w2, xq[t].z, p);
                        p = dot2(w3, xq[t].w, p);
                        part[r][t] = p;
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[r][t] = fmaf(db[r], part[r][t], acc[r][t]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down(v, off);
            }
            if (lane == 0 && row0 + r < rows) {
                y[t * rows + row0 + r] = v;
            }
        }
    }
}

// Q4_K and IQ4_XS for NT = 2..4 tokens (#126): the IQ3_S NT scheme. The
// activations come in their natural order from global memory; for Q4_K the
// (even, odd) half2 pairs the single-token kernel stages in LDS are built
// with two byte permutes per pair, so the products and their order (and the
// results) are the single-token kernel's.
template <int NT, int R>
__global__ void gemv_q4k_ntg_kernel(const uint8_t * __restrict__ qs,
                                    const uint8_t * __restrict__ meta,
                                    const __half * __restrict__ x, float * __restrict__ y,
                                    const long long rows, const long long n_sub) {
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * R;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    const long long k = n_sub * 32;

    float acc[R][NT] = {};
    if (row0 < rows) {
        for (long long s = lane; s < n_sub; s += 32) {
            const long long blk = s / 8;
            const int sub = (int) (s % 8);
            // x in q4k_x_order: per 8-weight run (x0,x2) (x4,x6) (x1,x3) (x5,x7)
            uint4 xq[NT][4];
            float xsum[NT];
            constexpr uint32_t kOnes = 0x3C003C00u;  // half2 (1, 1)
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const uint4 * xv = reinterpret_cast<const uint4 *>(x + t * k + s * 32);
                xsum[t] = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 n = xv[i];
                    xq[t][i] = make_uint4(__builtin_amdgcn_perm(n.y, n.x, 0x05040100u),
                                          __builtin_amdgcn_perm(n.w, n.z, 0x05040100u),
                                          __builtin_amdgcn_perm(n.y, n.x, 0x07060302u),
                                          __builtin_amdgcn_perm(n.w, n.z, 0x07060302u));
                    xsum[t] = dot2(xq[t][i].x, kOnes, dot2(xq[t][i].y, kOnes, xsum[t]));
                    xsum[t] = dot2(xq[t][i].z, kOnes, dot2(xq[t][i].w, kOnes, xsum[t]));
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                const uint4 q = *reinterpret_cast<const uint4 *>(qs + row * blocks * 128 + s * 16);
                float ds;
                float ms;
                q4k_meta(*reinterpret_cast<const uint4 *>(meta + (row * blocks + blk) * 16), sub, ds,
                         ms);
                const uint32_t w[4] = {q.x, q.y, q.z, q.w};
                float pe[NT] = {};
                float po[NT] = {};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t lo = w[i] & 0x0F0F0F0Fu;
                    const uint32_t hi = (w[i] >> 4) & 0x0F0F0F0Fu;
                    const uint32_t l0 = grid_pair_to_half2<0x04010400u>(lo);
                    const uint32_t h0 = grid_pair_to_half2<0x04010400u>(hi);
                    const uint32_t l1 = grid_pair_to_half2<0x04030402u>(lo);
                    const uint32_t h1 = grid_pair_to_half2<0x04030402u>(hi);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        pe[t] = dot2(l0, xq[t][i].x, pe[t]);
                        po[t] = dot2(h0, xq[t][i].z, po[t]);
                        pe[t] = dot2(l1, xq[t][i].y, pe[t]);
                        po[t] = dot2(h1, xq[t][i].w, po[t]);
                    }
                }
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[r][t] += fmaf(ds, pe[t] + po[t], -ms * xsum[t]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down(v, off);
            }
            if (lane == 0 && row0 + r < rows) {
                y[t * rows + row0 + r] = v;
            }
        }
    }
}

template <int NT, int R>
__global__ void gemv_iq4_ntg_kernel(const uint8_t * __restrict__ qs,
                                    const uint8_t * __restrict__ meta,
                                    const __half * __restrict__ x, float * __restrict__ y,
                                    const long long rows, const long long n_sub) {
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * R;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    const long long k = n_sub * 32;

    float acc[R][NT] = {};
    if (row0 < rows) {
        for (long long s = lane; s < n_sub; s += 32) {
            const long long blk = s / 8;
            const int sub = (int) (s % 8);
            uint32_t xl[NT][8];  // weights 0..15
            uint32_t xh[NT][8];  // weights 16..31
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const uint4 * xv = reinterpret_cast<const uint4 *>(x + t * k + s * 32);
                const uint4 a = xv[0], b = xv[1], c = xv[2], e = xv[3];
                xl[t][0] = a.x; xl[t][1] = a.y; xl[t][2] = a.z; xl[t][3] = a.w;
                xl[t][4] = b.x; xl[t][5] = b.y; xl[t][6] = b.z; xl[t][7] = b.w;
                xh[t][0] = c.x; xh[t][1] = c.y; xh[t][2] = c.z; xh[t][3] = c.w;
                xh[t][4] = e.x; xh[t][5] = e.y; xh[t][6] = e.z; xh[t][7] = e.w;
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                const uint4 q = *reinterpret_cast<const uint4 *>(qs + row * blocks * 128 + s * 16);
                const uint2 mt = *reinterpret_cast<const uint2 *>(meta + (row * blocks + blk) * 8);
                const uint32_t w[4] = {q.x, q.y, q.z, q.w};
                float pl[NT] = {};
                float ph[NT] = {};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t vl = iq4_lookup(w[i] & 0x0F0F0F0Fu);
                    const uint32_t vh = iq4_lookup((w[i] >> 4) & 0x0F0F0F0Fu);
                    const uint32_t l0 = iq4_pair_to_half2<0x04010400u>(vl);
                    const uint32_t h0 = iq4_pair_to_half2<0x04010400u>(vh);
                    const uint32_t l1 = iq4_pair_to_half2<0x04030402u>(vl);
                    const uint32_t h1 = iq4_pair_to_half2<0x04030402u>(vh);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        pl[t] = dot2(l0, xl[t][2 * i], pl[t]);
                        ph[t] = dot2(h0, xh[t][2 * i], ph[t]);
                        pl[t] = dot2(l1, xl[t][2 * i + 1], pl[t]);
                        ph[t] = dot2(h1, xh[t][2 * i + 1], ph[t]);
                    }
                }
                const float dl = iq4_meta(mt, sub);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[r][t] = fmaf(dl, pl[t] + ph[t], acc[r][t]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down(v, off);
            }
            if (lane == 0 && row0 + r < rows) {
                y[t * rows + row0 + r] = v;
            }
        }
    }
}

// The IQ2 family's 8 weights against NT activation groups, with the two
// dot2 chains of the single-token kernels in their order: out[t] += pa + pb.
template <int NT>
__device__ inline void iq2_dot(const uint32_t w0, const uint32_t w1, const uint32_t w2,
                               const uint32_t w3, const uint4 (&xq)[NT], float (&out)[NT]) {
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float pa = dot2(w0, xq[t].x, 0.0f);
        float pb = dot2(w2, xq[t].z, 0.0f);
        pa = dot2(w1, xq[t].y, pa);
        pb = dot2(w3, xq[t].w, pb);
        out[t] += pa + pb;
    }
}

// The NT activation groups l (8 halves each) of sub-block s, from global memory.
template <int NT>
__device__ inline void load_xq(const __half * x, const long long k, const long long s, const int l,
                               uint4 (&xq)[NT]) {
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        xq[t] = reinterpret_cast<const uint4 *>(x + t * k + s * 32)[l];
    }
}

// Writes acc[R][NT] (lane 0 after the warp reduction): y[t * rows + row].
template <int NT, int R>
__device__ inline void store_acc(float (&acc)[R][NT], float * y, const long long row0,
                                 const long long rows, const int lane) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down(v, off);
            }
            if (lane == 0 && row0 + r < rows) {
                y[t * rows + row0 + r] = v;
            }
        }
    }
}

// IQ2_XS / IQ2_XXS / IQ2_S for NT = 2..4 tokens (#126): the IQ3_S NT scheme.
template <int NT, int R>
__global__ void gemv_iq2xs_ntg_kernel(const uint8_t * __restrict__ qs,
                                      const uint8_t * __restrict__ sc,
                                      const uint8_t * __restrict__ d,
                                      const __half * __restrict__ x, float * __restrict__ y,
                                      const long long rows, const long long n_sub) {
    __shared__ uint64_t grid[512];
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        grid[i] = omph::quant::kIq2xsGrid[i];
    }
    __syncthreads();
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * R;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    const long long k = n_sub * 32;
    float acc[R][NT] = {};
    if (row0 < rows) {
        for (long long s = lane; s < n_sub; s += 32) {
            const long long blk = s / 8;
            const int ib = (int) (s % 8);
            uint32_t ent[R][2];
            float dbl[R][2];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                const uint16_t * e =
                    reinterpret_cast<const uint16_t *>(qs + row * blocks * 64 + s * 8);
                ent[r][0] = (uint32_t) e[0] | ((uint32_t) e[1] << 16);
                ent[r][1] = (uint32_t) e[2] | ((uint32_t) e[3] << 16);
                const uint8_t scb = sc[row * blocks * 8 + blk * 8 + ib];
                const float dv =
                    __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                dbl[r][0] = dv * (0.5f + (float) (scb & 0xF)) * 0.25f;
                dbl[r][1] = dv * (0.5f + (float) (scb >> 4)) * 0.25f;
            }
            float sub[R][2][NT] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint4 xq[NT];
                load_xq<NT>(x, k, s, l, xq);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t e = (ent[r][l >> 1] >> (16 * (l & 1))) & 0xFFFFu;
                    const uint64_t g = grid[e & 511u];
                    const uint32_t g1 = (uint32_t) g;
                    const uint32_t g2 = (uint32_t) (g >> 32);
                    const uint32_t f = e >> 9;
                    const uint32_t par = (uint32_t) __builtin_popcount(f) & 1u;
                    const uint32_t sgn = ((f & 0xFu) << 12) | ((f >> 4) << 29) | (par << 28);
                    constexpr uint32_t kSign2 = 0x80008000u;
                    iq2_dot<NT>((sgn & kSign2) | grid_pair_to_half2<0x04010400u>(g1),
                                ((sgn << 1) & kSign2) | grid_pair_to_half2<0x04030402u>(g1),
                                ((sgn << 2) & kSign2) | grid_pair_to_half2<0x04010400u>(g2),
                                ((sgn << 3) & kSign2) | grid_pair_to_half2<0x04030402u>(g2), xq,
                                sub[r][l >> 1]);
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[r][t] = fmaf(dbl[r][0], sub[r][0][t], acc[r][t]);
                    acc[r][t] = fmaf(dbl[r][1], sub[r][1][t], acc[r][t]);
                }
            }
        }
    }
    store_acc<NT, R>(acc, y, row0, rows, lane);
}

template <int NT, int R>
__global__ void gemv_iq2xxs_ntg_kernel(const uint8_t * __restrict__ qs,
                                       const uint8_t * __restrict__ d,
                                       const __half * __restrict__ x, float * __restrict__ y,
                                       const long long rows, const long long n_sub) {
    __shared__ uint64_t grid[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        grid[i] = omph::quant::kIq2xxsGrid[i];
    }
    __syncthreads();
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * R;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    const long long k = n_sub * 32;
    float acc[R][NT] = {};
    if (row0 < rows) {
        for (long long s = lane; s < n_sub; s += 32) {
            const long long blk = s / 8;
            uint32_t aux0[R];
            uint32_t aux1[R];
            float db[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                const uint32_t * w =
                    reinterpret_cast<const uint32_t *>(qs + row * blocks * 64 + s * 8);
                aux0[r] = w[0];
                aux1[r] = w[1];
                const float dv =
                    __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                db[r] = dv * (0.5f + (float) (aux1[r] >> 28)) * 0.25f;
            }
            float part[R][NT] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint4 xq[NT];
                load_xq<NT>(x, k, s, l, xq);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint64_t g = grid[(aux0[r] >> (8 * l)) & 0xFFu];
                    const uint32_t g1 = (uint32_t) g;
                    const uint32_t g2 = (uint32_t) (g >> 32);
                    const uint32_t sgn = iq3xxs_group_signs(aux1[r], l);
                    constexpr uint32_t kSign2 = 0x80008000u;
                    iq2_dot<NT>((sgn & kSign2) | grid_pair_to_half2<0x04010400u>(g1),
                                ((sgn << 1) & kSign2) | grid_pair_to_half2<0x04030402u>(g1),
                                ((sgn << 2) & kSign2) | grid_pair_to_half2<0x04010400u>(g2),
                                ((sgn << 3) & kSign2) | grid_pair_to_half2<0x04030402u>(g2), xq,
                                part[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[r][t] = fmaf(db[r], part[r][t], acc[r][t]);
                }
            }
        }
    }
    store_acc<NT, R>(acc, y, row0, rows, lane);
}

template <int NT, int R>
__global__ void gemv_iq2s_ntg_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ sg,
                                     const uint8_t * __restrict__ qh, const uint8_t * __restrict__ sc,
                                     const uint8_t * __restrict__ d, const __half * __restrict__ x,
                                     float * __restrict__ y, const long long rows,
                                     const long long n_sub) {
    __shared__ uint64_t grid[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        grid[i] = omph::quant::kIq2sGrid[i];
    }
    __syncthreads();
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * R;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    const long long k = n_sub * 32;
    float acc[R][NT] = {};
    if (row0 < rows) {
        for (long long s = lane; s < n_sub; s += 32) {
            const long long blk = s / 8;
            const int ib = (int) (s % 8);
            uint32_t qsw[R];
            uint32_t sgw[R];
            uint32_t qhb[R];
            float dbl[R][2];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                qsw[r] = *reinterpret_cast<const uint32_t *>(qs + row * blocks * 32 + s * 4);
                sgw[r] = *reinterpret_cast<const uint32_t *>(sg + row * blocks * 32 + s * 4);
                qhb[r] = qh[row * blocks * 8 + s];
                const uint8_t scb = sc[row * blocks * 8 + blk * 8 + ib];
                const float dv =
                    __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                dbl[r][0] = dv * (0.5f + (float) (scb & 0xF)) * 0.25f;
                dbl[r][1] = dv * (0.5f + (float) (scb >> 4)) * 0.25f;
            }
            float sub[R][2][NT] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint4 xq[NT];
                load_xq<NT>(x, k, s, l, xq);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t lo = (qsw[r] >> (8 * l)) & 0xFFu;
                    const uint32_t hi = ((qhb[r] >> (2 * l)) & 3u) << 8;
                    const uint64_t g = grid[lo | hi];
                    const uint32_t g1 = (uint32_t) g;
                    const uint32_t g2 = (uint32_t) (g >> 32);
                    constexpr uint32_t kSign2 = 0x80008000u;
                    iq2_dot<NT>(((sgw[r] << (4 * l)) & kSign2) | grid_pair_to_half2<0x04010400u>(g1),
                                ((sgw[r] << (4 * l + 1)) & kSign2) |
                                    grid_pair_to_half2<0x04030402u>(g1),
                                ((sgw[r] << (4 * l + 2)) & kSign2) |
                                    grid_pair_to_half2<0x04010400u>(g2),
                                ((sgw[r] << (4 * l + 3)) & kSign2) |
                                    grid_pair_to_half2<0x04030402u>(g2),
                                xq, sub[r][l >> 1]);
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[r][t] = fmaf(dbl[r][0], sub[r][0][t], acc[r][t]);
                    acc[r][t] = fmaf(dbl[r][1], sub[r][1][t], acc[r][t]);
                }
            }
        }
    }
    store_acc<NT, R>(acc, y, row0, rows, lane);
}

// Q2_K for NT = 2..4 tokens (#126): the single-token kernel's 128-weight groups
// with the activations read from global memory (natural order) per token.
template <int NT, int R>
__global__ void gemv_q2k_ntg_kernel(const uint8_t * __restrict__ sc, const uint8_t * __restrict__ qs,
                                    const uint8_t * __restrict__ d, const uint8_t * __restrict__ dmin,
                                    const __half * __restrict__ x, float * __restrict__ y,
                                    const long long rows, const long long n_sub) {
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * R;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    const long long k = n_sub * 32;
    const long long n_groups = n_sub / 4;
    float acc[R][NT] = {};
    if (row0 < rows) {
        for (long long g = lane; g < n_groups; g += 32) {
            const long long blk = g >> 1;
            const int half = (int) (g & 1);
            constexpr uint32_t kOnes = 0x3C003C00u;  // half2 (1, 1)
            uint32_t wa[R][4];
            uint32_t wb[R][4];
            uint64_t scw[R];
            float dv[R];
            float mv[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                const uint8_t * qrow = qs + row * blocks * 64 + g * 32;
                scw[r] = *reinterpret_cast<const uint64_t *>(sc + row * blocks * 16 + blk * 16 +
                                                             half * 8);
                dv[r] = __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                mv[r] = __half2float(
                    *reinterpret_cast<const __half *>(dmin + row * blocks * 2 + blk * 2));
                const uint4 qa = *reinterpret_cast<const uint4 *>(qrow);
                const uint4 qb = *reinterpret_cast<const uint4 *>(qrow + 16);
                wa[r][0] = qa.x; wa[r][1] = qa.y; wa[r][2] = qa.z; wa[r][3] = qa.w;
                wb[r][0] = qb.x; wb[r][1] = qb.y; wb[r][2] = qb.z; wb[r][3] = qb.w;
            }
            float pr[R][NT] = {};
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int shift = (u >> 1) * 2;
                // this 16-weight sub-block's activations, per token: 4 uint2 each
                uint2 xu[NT][4];
                float xsum[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const uint2 * xh = reinterpret_cast<const uint2 *>(x + t * k + g * 128);
                    float v = 0.0f;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        xu[t][q] = xh[u * 4 + q];
                        v = dot2(xu[t][q].x, kOnes, dot2(xu[t][q].y, kOnes, v));
                    }
                    xsum[t] = v;
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t scb = (uint32_t) (scw[r] >> (8 * u)) & 0xFFu;
                    float ca[NT] = {};
                    float cb[NT] = {};
#pragma unroll
                    for (int q = 0; q < 4; q += 2) {
                        const uint32_t c0 = (((u & 1) ? wb[r] : wa[r])[q] >> shift) & 0x03030303u;
                        const uint32_t c1 = (((u & 1) ? wb[r] : wa[r])[q + 1] >> shift) & 0x03030303u;
                        const uint32_t a0 = grid_pair_to_half2<0x04010400u>(c0);
                        const uint32_t b0 = grid_pair_to_half2<0x04010400u>(c1);
                        const uint32_t a1 = grid_pair_to_half2<0x04030402u>(c0);
                        const uint32_t b1 = grid_pair_to_half2<0x04030402u>(c1);
#pragma unroll
                        for (int t = 0; t < NT; ++t) {
                            ca[t] = dot2(a0, xu[t][q].x, ca[t]);
                            cb[t] = dot2(b0, xu[t][q + 1].x, cb[t]);
                            ca[t] = dot2(a1, xu[t][q].y, ca[t]);
                            cb[t] = dot2(b1, xu[t][q + 1].y, cb[t]);
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        pr[r][t] = fmaf(dv[r] * (float) (scb & 0xFu), ca[t] + cb[t], pr[r][t]);
                        pr[r][t] = fmaf(-mv[r] * (float) (scb >> 4), xsum[t], pr[r][t]);
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[r][t] += pr[r][t];
                }
            }
        }
    }
    store_acc<NT, R>(acc, y, row0, rows, lane);
}

// IQ3_S: 110-byte blocks. Per 32-weight sub-block: eight 9-bit grid indices
// (8 bits from qs, the 9th from one qh bit per index), four sign-mask bytes and
// a 4-bit scale shared by two sub-blocks. Grid (2 KiB, bytes) in LDS; each
// byte pair becomes a half2 with one permute and one packed add, the signs
// (pair-ordered by the repack, see repack.hh) are OR-ed in with one shift and
// one and_or per pair, and v_dot2 accumulates: about half the instructions of
// a float-per-weight loop (#63).
constexpr long long kIq3sChunkSub = 256;  // 8192 weights staged per LDS refill
constexpr long long kIq3sRows = 4;        // output rows computed per warp
constexpr int kIq3sNtPersistMax = 2;      // N-token kernels up to this N on persistent warps (#63)

__global__ void gemv_iq3s_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ qh,
                                 const uint8_t * __restrict__ signs,
                                 const uint8_t * __restrict__ scales, const uint8_t * __restrict__ d,
                                 const __half * __restrict__ x, float * __restrict__ y,
                                 const long long rows, const long long n_sub,
                                 const long long chunk_sub) {
    extern __shared__ uint8_t smem[];
    __half * xs = reinterpret_cast<__half *>(smem);
    uint32_t * grid = reinterpret_cast<uint32_t *>(smem + chunk_sub * 64);  // 512 x 4 bytes
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        grid[i] = omph::quant::kIq3sGrid[i];
    }

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kIq3sRows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kIq3sRows] = {};
    for (long long base = 0; base < n_sub; base += kIq3sChunkSub) {
        const long long len = n_sub - base < kIq3sChunkSub ? n_sub - base : kIq3sChunkSub;
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[i] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const uint4 * xv = reinterpret_cast<const uint4 *>(xs + (s - base) * 32);
            uint2 q[kIq3sRows];
            uint32_t qhb[kIq3sRows];
            uint32_t sg[kIq3sRows];
            float dbr[kIq3sRows];
#pragma unroll
            for (int r = 0; r < kIq3sRows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                q[r] = *reinterpret_cast<const uint2 *>(qs + row * blocks * 64 + s * 8);
                qhb[r] = qh[row * blocks * 8 + s];
                sg[r] = *reinterpret_cast<const uint32_t *>(signs + row * blocks * 32 + s * 4);
                const uint8_t sc = scales[row * blocks * 4 + s / 2];
                const float dv = __half2float(
                    *reinterpret_cast<const __half *>(d + row * blocks * 2 + (s / 8) * 2));
                dbr[r] = dv * (1.0f + 2.0f * (float) ((s & 1) ? (sc >> 4) : (sc & 0xF)));
            }
            float part[kIq3sRows] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint4 xq = xv[l];  // the 8 activations of this group, packed
#pragma unroll
                for (int r = 0; r < kIq3sRows; ++r) {
                    const uint32_t lo = (q[r].x >> (16 * (l & 1))) & 0xFFFFu;
                    const uint32_t word = l < 2 ? lo : (q[r].y >> (16 * (l & 1))) & 0xFFFFu;
                    const uint32_t i1 = (word & 0xFFu) | (((qhb[r] >> (2 * l)) & 1u) << 8);
                    const uint32_t i2 = (word >> 8) | (((qhb[r] >> (2 * l + 1)) & 1u) << 8);
                    const uint32_t b1 = grid[i1];
                    const uint32_t b2 = grid[i2];
                    const uint2 g1 = make_uint2(grid_pair_to_half2<0x04010400u>(b1),
                                                grid_pair_to_half2<0x04030402u>(b1));
                    const uint2 g2 = make_uint2(grid_pair_to_half2<0x04010400u>(b2),
                                                grid_pair_to_half2<0x04030402u>(b2));
                    // pair p = 4l + i: its two sign bits land on the half2 sign
                    // bits with one shift (layout: repack.hh); the grid values
                    // are positive, so OR-ing the sign in negates them.
                    constexpr uint32_t kSign2 = 0x80008000u;
                    float p = part[r];
                    p = dot2(((sg[r] << (4 * l)) & kSign2) | g1.x, xq.x, p);
                    p = dot2(((sg[r] << (4 * l + 1)) & kSign2) | g1.y, xq.y, p);
                    p = dot2(((sg[r] << (4 * l + 2)) & kSign2) | g2.x, xq.z, p);
                    p = dot2(((sg[r] << (4 * l + 3)) & kSign2) | g2.y, xq.w, p);
                    part[r] = p;
                }
            }
#pragma unroll
            for (int r = 0; r < kIq3sRows; ++r) {
                acc[r] = fmaf(dbr[r], part[r], acc[r]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kIq3sRows; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down(v, off);
        }
        if (lane == 0 && row0 + r < rows) {
            y[row0 + r] = v;
        }
    }
}

// IQ3_S with persistent warps (#63), for rows of up to one x chunk: as many
// workgroups as fit on the GPU at once; each stages x in LDS once, then its
// warps take groups of kIq3sRows rows in turn (grid stride). A launch then
// ends within one row group of each other warp instead of a whole round of
// workgroups: the tensors are small (7-120 us per launch) and those tails cost
// ~12 % of the bandwidth.
__global__ void gemv_iq3s_pw_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ qh,
                                    const uint8_t * __restrict__ signs, const uint8_t * __restrict__ scales,
                                    const uint8_t * __restrict__ d,
                                    const __half * __restrict__ x, float * __restrict__ y,
                                    const long long rows, const long long n_sub) {
    extern __shared__ uint8_t smem[];
    __half * xs = reinterpret_cast<__half *>(smem);
    uint32_t * grid = reinterpret_cast<uint32_t *>(smem + n_sub * 64);  // 512 x 4 bytes
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        grid[i] = omph::quant::kIq3sGrid[i];
    }
    for (long long i = threadIdx.x; i < n_sub * 32; i += blockDim.x) {
        xs[i] = x[i];
    }
    __syncthreads();

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    const long long n_groups = (rows + kIq3sRows - 1) / kIq3sRows;
    const long long stride = (long long) gridDim.x * (blockDim.x / 32);
    for (long long grp = (long long) blockIdx.x * (blockDim.x / 32) + warp; grp < n_groups; grp += stride) {
        const long long row0 = grp * kIq3sRows;
        float acc[kIq3sRows] = {};
        for (long long s = lane; s < n_sub; s += 32) {
            const uint4 * xv = reinterpret_cast<const uint4 *>(xs + s * 32);
            uint2 q[kIq3sRows];
            uint32_t qhb[kIq3sRows];
            uint32_t sg[kIq3sRows];
            float dbr[kIq3sRows];
#pragma unroll
            for (int r = 0; r < kIq3sRows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                q[r] = *reinterpret_cast<const uint2 *>(qs + row * blocks * 64 + s * 8);
                qhb[r] = qh[row * blocks * 8 + s];
                sg[r] = *reinterpret_cast<const uint32_t *>(signs + row * blocks * 32 + s * 4);
                const uint8_t sc = scales[row * blocks * 4 + s / 2];
                const float dv = __half2float(
                    *reinterpret_cast<const __half *>(d + row * blocks * 2 + (s / 8) * 2));
                dbr[r] = dv * (1.0f + 2.0f * (float) ((s & 1) ? (sc >> 4) : (sc & 0xF)));
            }
            float part[kIq3sRows] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint4 xq = xv[l];
#pragma unroll
                for (int r = 0; r < kIq3sRows; ++r) {
                    const uint32_t lo = (q[r].x >> (16 * (l & 1))) & 0xFFFFu;
                    const uint32_t word = l < 2 ? lo : (q[r].y >> (16 * (l & 1))) & 0xFFFFu;
                    const uint32_t i1 = (word & 0xFFu) | (((qhb[r] >> (2 * l)) & 1u) << 8);
                    const uint32_t i2 = (word >> 8) | (((qhb[r] >> (2 * l + 1)) & 1u) << 8);
                    const uint32_t b1 = grid[i1];
                    const uint32_t b2 = grid[i2];
                    const uint2 g1 = make_uint2(grid_pair_to_half2<0x04010400u>(b1),
                                                grid_pair_to_half2<0x04030402u>(b1));
                    const uint2 g2 = make_uint2(grid_pair_to_half2<0x04010400u>(b2),
                                                grid_pair_to_half2<0x04030402u>(b2));
                    constexpr uint32_t kSign2 = 0x80008000u;
                    float p = part[r];
                    p = dot2(((sg[r] << (4 * l)) & kSign2) | g1.x, xq.x, p);
                    p = dot2(((sg[r] << (4 * l + 1)) & kSign2) | g1.y, xq.y, p);
                    p = dot2(((sg[r] << (4 * l + 2)) & kSign2) | g2.x, xq.z, p);
                    p = dot2(((sg[r] << (4 * l + 3)) & kSign2) | g2.y, xq.w, p);
                    part[r] = p;
                }
            }
#pragma unroll
            for (int r = 0; r < kIq3sRows; ++r) {
                acc[r] = fmaf(dbr[r], part[r], acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < kIq3sRows; ++r) {
            float v = acc[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down(v, off);
            }
            if (lane == 0 && row0 + r < rows) {
                y[row0 + r] = v;
            }
        }
    }
}



// IQ3_S for NT = 2..4 tokens at once (a speculative verification, #126): the
// single-token kernel's decode, done once per weight and dotted with the NT
// activation vectors (back to back, stride k). The activations are read
// straight from global memory (L1 / L2 resident) with no LDS staging and no
// barrier in the loop: staged NT-fold in LDS they made the kernel 2-2.7x a
// single token, read like this it is 1.0-1.25x. The products are the
// single-token kernel's, in its order: the results are bit-identical. The
// single-token kernel is left as it is (the decode step is sensitive to its
// code generation).
template <int NT, int R>
__global__ void gemv_iq3s_ntg_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ qh,
                                    const uint8_t * __restrict__ signs,
                                    const uint8_t * __restrict__ scales,
                                    const uint8_t * __restrict__ d, const __half * __restrict__ x,
                                    float * __restrict__ y, const long long rows,
                                    const long long n_sub, const long long chunk_sub) {
    extern __shared__ uint8_t smem[];
    // the activations straight from global memory (L1 / L2): no staging, no
    // barrier in the loop; only the grid lives in LDS
    (void) chunk_sub;
    uint32_t * grid = reinterpret_cast<uint32_t *>(smem);
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        grid[i] = omph::quant::kIq3sGrid[i];
    }
    __syncthreads();

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    const long long k = n_sub * 32;
    // persistent warps (#63): row groups taken in turn, grid sized to what fits
    const long long n_groups = (rows + R - 1) / R;
    const long long stride = (long long) gridDim.x * (blockDim.x / 32);
    for (long long grp = (long long) blockIdx.x * (blockDim.x / 32) + warp; grp < n_groups; grp += stride) {
    const long long row0 = grp * R;
    float acc[R][NT] = {};
    {
        for (long long s = lane; s < n_sub; s += 32) {
            uint2 q[R];
            uint32_t qhb[R];
            uint32_t sg[R];
            float dbr[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                // the row is warp-uniform: its base pointers in SGPRs and 32-bit
                // lane offsets, instead of 64-bit address math per load (-9 %)
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                const long long rb = __builtin_amdgcn_readfirstlane((int) row) * blocks;
                const uint32_t so = (uint32_t) s;
                q[r] = *reinterpret_cast<const uint2 *>(qs + rb * 64 + so * 8u);
                qhb[r] = (qh + rb * 8)[so];
                sg[r] = *reinterpret_cast<const uint32_t *>(signs + rb * 32 + so * 4u);
                const uint8_t sc = (scales + rb * 4)[so / 2u];
                const float dv = __half2float(*reinterpret_cast<const __half *>(d + rb * 2 + (so / 8u) * 2u));
                dbr[r] = dv * (1.0f + 2.0f * (float) ((s & 1) ? (sc >> 4) : (sc & 0xF)));
            }
            float part[R][NT] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint4 xq[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    xq[t] = reinterpret_cast<const uint4 *>(x + t * k + s * 32)[l];
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t lo = (q[r].x >> (16 * (l & 1))) & 0xFFFFu;
                    const uint32_t word = l < 2 ? lo : (q[r].y >> (16 * (l & 1))) & 0xFFFFu;
                    const uint32_t i1 = (word & 0xFFu) | (((qhb[r] >> (2 * l)) & 1u) << 8);
                    const uint32_t i2 = (word >> 8) | (((qhb[r] >> (2 * l + 1)) & 1u) << 8);
                    const uint32_t b1 = grid[i1];
                    const uint32_t b2 = grid[i2];
                    constexpr uint32_t kSign2 = 0x80008000u;
                    const uint32_t w0 = ((sg[r] << (4 * l)) & kSign2) |
                                        grid_pair_to_half2<0x04010400u>(b1);
                    const uint32_t w1 = ((sg[r] << (4 * l + 1)) & kSign2) |
                                        grid_pair_to_half2<0x04030402u>(b1);
                    const uint32_t w2 = ((sg[r] << (4 * l + 2)) & kSign2) |
                                        grid_pair_to_half2<0x04010400u>(b2);
                    const uint32_t w3 = ((sg[r] << (4 * l + 3)) & kSign2) |
                                        grid_pair_to_half2<0x04030402u>(b2);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        float p = part[r][t];
                        p = dot2(w0, xq[t].x, p);
                        p = dot2(w1, xq[t].y, p);
                        p = dot2(w2, xq[t].z, p);
                        p = dot2(w3, xq[t].w, p);
                        part[r][t] = p;
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[r][t] = fmaf(dbr[r], part[r][t], acc[r][t]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down(v, off);
            }
            if (lane == 0 && row0 + r < rows) {
                y[t * rows + row0 + r] = v;
            }
        }
    }
}
}

// Q2_K: 84-byte blocks, sixteen 16-weight sub-blocks each with a 4-bit scale and
// a 4-bit min. A lane iteration covers one 128-weight group: the 32 quant bytes
// are read once (two uint4, extracted through explicit words so nothing depends
// on how the compiler treats an array) and eight sub-block scales.
constexpr long long kQ2kChunkSub = 256;
constexpr long long kQ2kRows = 4;
constexpr long long kQ2kXPitch = 132;  // halves per staged 128-weight group

__global__ void gemv_q2k_kernel(const uint8_t * __restrict__ sc, const uint8_t * __restrict__ qs,
                                const uint8_t * __restrict__ d, const uint8_t * __restrict__ dmin,
                                const __half * __restrict__ x, float * __restrict__ y,
                                const long long rows, const long long n_sub,
                                const long long chunk_sub) {
    extern __shared__ __half xs[];  // activation chunk

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kQ2kRows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kQ2kRows] = {};
    // persistent warps (#63): rows of one chunk stage x once, then the warps
    // take row groups of kQ2kRows in turn (the launch sizes the grid to what fits)
    if (n_sub <= chunk_sub) {
        {
            const long long base = 0;
            const long long len = n_sub;
            // Each lane reads its own 128-weight group: a 132-half pitch puts the
            // lanes' rows of x on different LDS banks (a 128-half pitch put them
            // all on one, #75).
            for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
                xs[(i / 128) * kQ2kXPitch + (i % 128)] = x[base * 32 + i];
            }
        }
        __syncthreads();
        const long long n_groups = (rows + kQ2kRows - 1) / kQ2kRows;
        const long long stride = (long long) gridDim.x * (blockDim.x / 32);
        for (long long grp = (long long) blockIdx.x * (blockDim.x / 32) + warp; grp < n_groups; grp += stride) {
            const long long row0 = grp * kQ2kRows;
            float acc[kQ2kRows] = {};
            {
                const long long base = 0;
                const long long len = n_sub;
                const long long g_first = base / 4;
                const long long g_last = (base + len) / 4;
                for (long long g = g_first + lane; g < g_last; g += 32) {
                    const long long blk = g >> 1;
                    const int half = (int) (g & 1);
                    const long long xoff = (g - base / 4) * kQ2kXPitch;
                    // sum (dl * c - ml) * x = dl * sum c * x - ml * sum x per group of 16:
                    // the 8 sums of x are shared by the rows; four 2-bit codes come out of
                    // one word as bytes ((w >> shift) & 0x03030303), become two half2
                    // (v_perm + packed add) and go through v_dot2 (#75).
                    const uint2 * xh = reinterpret_cast<const uint2 *>(xs + xoff);  // 4 halves each
                    constexpr uint32_t kOnes = 0x3C003C00u;  // half2 (1, 1)
                    float xsum[8];
        #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        float t = 0.0f;
        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const uint2 xv2 = xh[u * 4 + q];
                            t = dot2(xv2.x, kOnes, dot2(xv2.y, kOnes, t));
                        }
                        xsum[u] = t;
                    }
                    float part[kQ2kRows] = {};
        #pragma unroll
                    for (int r = 0; r < kQ2kRows; ++r) {
                        const long long row = row0 + r <= rlast ? row0 + r : rlast;
                        const uint8_t * qrow = qs + row * blocks * 64 + g * 32;
                        const uint64_t scw =
                            *reinterpret_cast<const uint64_t *>(sc + row * blocks * 16 + blk * 16 + half * 8);
                        const float dv =
                            __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                        const float mv = __half2float(
                            *reinterpret_cast<const __half *>(dmin + row * blocks * 2 + blk * 2));
                        const uint4 qa = *reinterpret_cast<const uint4 *>(qrow);
                        const uint4 qb = *reinterpret_cast<const uint4 *>(qrow + 16);
                        const uint32_t wa[4] = {qa.x, qa.y, qa.z, qa.w};
                        const uint32_t wb[4] = {qb.x, qb.y, qb.z, qb.w};
                        float pr = 0.0f;
        #pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            const int shift = (u >> 1) * 2;
                            const uint32_t scb = (uint32_t) (scw >> (8 * u)) & 0xFFu;
                            float ca = 0.0f;
                            float cb = 0.0f;
        #pragma unroll
                            for (int q = 0; q < 4; q += 2) {
                                const uint32_t c0 = (((u & 1) ? wb : wa)[q] >> shift) & 0x03030303u;
                                const uint32_t c1 = (((u & 1) ? wb : wa)[q + 1] >> shift) & 0x03030303u;
                                const uint2 x0 = xh[u * 4 + q];
                                const uint2 x1 = xh[u * 4 + q + 1];
                                ca = dot2(grid_pair_to_half2<0x04010400u>(c0), x0.x, ca);
                                cb = dot2(grid_pair_to_half2<0x04010400u>(c1), x1.x, cb);
                                ca = dot2(grid_pair_to_half2<0x04030402u>(c0), x0.y, ca);
                                cb = dot2(grid_pair_to_half2<0x04030402u>(c1), x1.y, cb);
                            }
                            pr = fmaf(dv * (float) (scb & 0xFu), ca + cb, pr);
                            pr = fmaf(-mv * (float) (scb >> 4), xsum[u], pr);
                        }
                        part[r] = pr;
                    }
        #pragma unroll
                    for (int r = 0; r < kQ2kRows; ++r) {
                        acc[r] += part[r];
                    }
                }
            }
    #pragma unroll
        for (int r = 0; r < kQ2kRows; ++r) {
            float v = acc[r];
    #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down(v, off);
            }
            if (lane == 0 && row0 + r < rows) {
                y[row0 + r] = v;
            }
        }
        }
        return;
    }
    for (long long base = 0; base < n_sub; base += chunk_sub) {
        const long long len = n_sub - base < chunk_sub ? n_sub - base : chunk_sub;
        // Each lane reads its own 128-weight group: a 132-half pitch puts the
        // lanes' rows of x on different LDS banks (a 128-half pitch put them
        // all on one, #75).
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[(i / 128) * kQ2kXPitch + (i % 128)] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        const long long g_first = base / 4;
        const long long g_last = (base + len) / 4;
        for (long long g = g_first + lane; g < g_last; g += 32) {
            const long long blk = g >> 1;
            const int half = (int) (g & 1);
            const long long xoff = (g - base / 4) * kQ2kXPitch;
            // sum (dl * c - ml) * x = dl * sum c * x - ml * sum x per group of 16:
            // the 8 sums of x are shared by the rows; four 2-bit codes come out of
            // one word as bytes ((w >> shift) & 0x03030303), become two half2
            // (v_perm + packed add) and go through v_dot2 (#75).
            const uint2 * xh = reinterpret_cast<const uint2 *>(xs + xoff);  // 4 halves each
            constexpr uint32_t kOnes = 0x3C003C00u;  // half2 (1, 1)
            float xsum[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                float t = 0.0f;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint2 xv2 = xh[u * 4 + q];
                    t = dot2(xv2.x, kOnes, dot2(xv2.y, kOnes, t));
                }
                xsum[u] = t;
            }
            float part[kQ2kRows] = {};
#pragma unroll
            for (int r = 0; r < kQ2kRows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                const uint8_t * qrow = qs + row * blocks * 64 + g * 32;
                const uint64_t scw =
                    *reinterpret_cast<const uint64_t *>(sc + row * blocks * 16 + blk * 16 + half * 8);
                const float dv =
                    __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                const float mv = __half2float(
                    *reinterpret_cast<const __half *>(dmin + row * blocks * 2 + blk * 2));
                const uint4 qa = *reinterpret_cast<const uint4 *>(qrow);
                const uint4 qb = *reinterpret_cast<const uint4 *>(qrow + 16);
                const uint32_t wa[4] = {qa.x, qa.y, qa.z, qa.w};
                const uint32_t wb[4] = {qb.x, qb.y, qb.z, qb.w};
                float pr = 0.0f;
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const int shift = (u >> 1) * 2;
                    const uint32_t scb = (uint32_t) (scw >> (8 * u)) & 0xFFu;
                    float ca = 0.0f;
                    float cb = 0.0f;
#pragma unroll
                    for (int q = 0; q < 4; q += 2) {
                        const uint32_t c0 = (((u & 1) ? wb : wa)[q] >> shift) & 0x03030303u;
                        const uint32_t c1 = (((u & 1) ? wb : wa)[q + 1] >> shift) & 0x03030303u;
                        const uint2 x0 = xh[u * 4 + q];
                        const uint2 x1 = xh[u * 4 + q + 1];
                        ca = dot2(grid_pair_to_half2<0x04010400u>(c0), x0.x, ca);
                        cb = dot2(grid_pair_to_half2<0x04010400u>(c1), x1.x, cb);
                        ca = dot2(grid_pair_to_half2<0x04030402u>(c0), x0.y, ca);
                        cb = dot2(grid_pair_to_half2<0x04030402u>(c1), x1.y, cb);
                    }
                    pr = fmaf(dv * (float) (scb & 0xFu), ca + cb, pr);
                    pr = fmaf(-mv * (float) (scb >> 4), xsum[u], pr);
                }
                part[r] = pr;
            }
#pragma unroll
            for (int r = 0; r < kQ2kRows; ++r) {
                acc[r] += part[r];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kQ2kRows; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down(v, off);
        }
        if (lane == 0 && row0 + r < rows) {
            y[row0 + r] = v;
        }
    }
}


// Q6_K: 210-byte blocks. A lane iteration covers one 128-weight group: 64 bytes
// of low nibbles, 32 bytes with the upper 2 bits and eight signed scales. The
// four quarters of a group are contiguous weight runs (l, l+32, l+64, l+96): one
// ql word and one qh word give four 6-bit codes of a quarter as bytes, which
// become two half2 of (code - 32) (v_perm to 1024 + code, packed add of -1056,
// exact) and go through v_dot2; one accumulator per 16-weight sub-block takes
// its scale at the end (#95).
constexpr long long kQ6kChunkSub = 256;
constexpr long long kQ6kRows = 2;
constexpr long long kQ6kXPitch = 132;  // halves per staged 128-weight group (as Q2_K)

template <uint32_t Sel>
__device__ inline uint32_t q6_pair_to_half2(const uint32_t g) {
    const uint32_t biased = __builtin_amdgcn_perm(0x64646464u, g, Sel);
    const half2_v v =
        __builtin_bit_cast(half2_v, biased) - half2_v{(_Float16) 1056, (_Float16) 1056};
    return __builtin_bit_cast(uint32_t, v);
}

__global__ void gemv_q6k_kernel(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                                const uint8_t * __restrict__ sc, const uint8_t * __restrict__ d,
                                const __half * __restrict__ x, float * __restrict__ y,
                                const long long rows, const long long n_sub,
                                const long long chunk_sub) {
    extern __shared__ __half xs[];  // activation chunk

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kQ6kRows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kQ6kRows] = {};
    // persistent warps (#63): rows of one chunk stage x once, then the warps
    // take row groups of kQ6kRows in turn (the launch sizes the grid to what fits)
    if (n_sub <= chunk_sub) {
        {
            const long long base = 0;
            const long long len = n_sub;
            // Each lane reads its own 128-weight group: the 132-half pitch spreads
            // the lanes over the LDS banks (a 128-half pitch puts them on one, #75).
            for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
                xs[(i / 128) * kQ6kXPitch + (i % 128)] = x[base * 32 + i];
            }
        }
        __syncthreads();
        const long long n_groups = (rows + kQ6kRows - 1) / kQ6kRows;
        const long long stride = (long long) gridDim.x * (blockDim.x / 32);
        for (long long grp = (long long) blockIdx.x * (blockDim.x / 32) + warp; grp < n_groups; grp += stride) {
            const long long row0 = grp * kQ6kRows;
            float acc[kQ6kRows] = {};
            {
                const long long base = 0;
                const long long len = n_sub;
                const long long g_first = base / 4;
                const long long g_last = (base + len) / 4;
                for (long long g = g_first + lane; g < g_last; g += 32) {
                    const long long blk = g >> 1;
                    const int half = (int) (g & 1);
                    const uint2 * xh =
                        reinterpret_cast<const uint2 *>(xs + (g - base / 4) * kQ6kXPitch);  // 4 halves
                    float part[kQ6kRows] = {};
        #pragma unroll
                    for (int r = 0; r < kQ6kRows; ++r) {
                        const long long row = row0 + r <= rlast ? row0 + r : rlast;
                        const uint4 * pql =
                            reinterpret_cast<const uint4 *>(ql + row * blocks * 128 + blk * 128 + half * 64);
                        const uint4 * pqh =
                            reinterpret_cast<const uint4 *>(qh + row * blocks * 64 + blk * 64 + half * 32);
                        const uint2 scw =
                            *reinterpret_cast<const uint2 *>(sc + row * blocks * 16 + blk * 16 + half * 8);
                        const float dv = __half2float(
                            *reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                        const uint4 l0 = pql[0], l1 = pql[1], l2 = pql[2], l3 = pql[3];
                        const uint4 h0 = pqh[0], h1 = pqh[1];
                        const uint32_t qlw[16] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w,
                                                  l2.x, l2.y, l2.z, l2.w, l3.x, l3.y, l3.z, l3.w};
                        const uint32_t qhw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
                        float sub[8] = {};
        #pragma unroll
                        for (int wi = 0; wi < 8; ++wi) {
                            const int is = wi >> 2;  // 16-weight sub-block within the quarter
                            const uint32_t lo = qlw[wi];
                            const uint32_t hi = qlw[wi + 8];
                            const uint32_t hb = qhw[wi];
                            // codes of weights 4wi..4wi+3 of each quarter, one per byte
                            const uint32_t c[4] = {
                                (lo & 0x0F0F0F0Fu) | ((hb & 0x03030303u) << 4),
                                (hi & 0x0F0F0F0Fu) | ((hb & 0x0C0C0C0Cu) << 2),
                                ((lo >> 4) & 0x0F0F0F0Fu) | (hb & 0x30303030u),
                                ((hi >> 4) & 0x0F0F0F0Fu) | ((hb >> 2) & 0x30303030u)};
        #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const uint2 xv = xh[j * 8 + wi];
                                float t = dot2(q6_pair_to_half2<0x04010400u>(c[j]), xv.x, sub[2 * j + is]);
                                sub[2 * j + is] = dot2(q6_pair_to_half2<0x04030402u>(c[j]), xv.y, t);
                            }
                        }
                        const uint32_t sw[2] = {scw.x, scw.y};
                        float pr = 0.0f;
        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float s8 = (float) (int8_t) (sw[i >> 2] >> (8 * (i & 3)));
                            pr = fmaf(s8, sub[i], pr);
                        }
                        part[r] = dv * pr;
                    }
        #pragma unroll
                    for (int r = 0; r < kQ6kRows; ++r) {
                        acc[r] += part[r];
                    }
                }
            }
    #pragma unroll
        for (int r = 0; r < kQ6kRows; ++r) {
            float v = acc[r];
    #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down(v, off);
            }
            if (lane == 0 && row0 + r < rows) {
                y[row0 + r] = v;
            }
        }
        }
        return;
    }
    for (long long base = 0; base < n_sub; base += chunk_sub) {
        const long long len = n_sub - base < chunk_sub ? n_sub - base : chunk_sub;
        // Each lane reads its own 128-weight group: the 132-half pitch spreads
        // the lanes over the LDS banks (a 128-half pitch puts them on one, #75).
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[(i / 128) * kQ6kXPitch + (i % 128)] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        const long long g_first = base / 4;
        const long long g_last = (base + len) / 4;
        for (long long g = g_first + lane; g < g_last; g += 32) {
            const long long blk = g >> 1;
            const int half = (int) (g & 1);
            const uint2 * xh =
                reinterpret_cast<const uint2 *>(xs + (g - base / 4) * kQ6kXPitch);  // 4 halves
            float part[kQ6kRows] = {};
#pragma unroll
            for (int r = 0; r < kQ6kRows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                const uint4 * pql =
                    reinterpret_cast<const uint4 *>(ql + row * blocks * 128 + blk * 128 + half * 64);
                const uint4 * pqh =
                    reinterpret_cast<const uint4 *>(qh + row * blocks * 64 + blk * 64 + half * 32);
                const uint2 scw =
                    *reinterpret_cast<const uint2 *>(sc + row * blocks * 16 + blk * 16 + half * 8);
                const float dv = __half2float(
                    *reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                const uint4 l0 = pql[0], l1 = pql[1], l2 = pql[2], l3 = pql[3];
                const uint4 h0 = pqh[0], h1 = pqh[1];
                const uint32_t qlw[16] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w,
                                          l2.x, l2.y, l2.z, l2.w, l3.x, l3.y, l3.z, l3.w};
                const uint32_t qhw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
                float sub[8] = {};
#pragma unroll
                for (int wi = 0; wi < 8; ++wi) {
                    const int is = wi >> 2;  // 16-weight sub-block within the quarter
                    const uint32_t lo = qlw[wi];
                    const uint32_t hi = qlw[wi + 8];
                    const uint32_t hb = qhw[wi];
                    // codes of weights 4wi..4wi+3 of each quarter, one per byte
                    const uint32_t c[4] = {
                        (lo & 0x0F0F0F0Fu) | ((hb & 0x03030303u) << 4),
                        (hi & 0x0F0F0F0Fu) | ((hb & 0x0C0C0C0Cu) << 2),
                        ((lo >> 4) & 0x0F0F0F0Fu) | (hb & 0x30303030u),
                        ((hi >> 4) & 0x0F0F0F0Fu) | ((hb >> 2) & 0x30303030u)};
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint2 xv = xh[j * 8 + wi];
                        float t = dot2(q6_pair_to_half2<0x04010400u>(c[j]), xv.x, sub[2 * j + is]);
                        sub[2 * j + is] = dot2(q6_pair_to_half2<0x04030402u>(c[j]), xv.y, t);
                    }
                }
                const uint32_t sw[2] = {scw.x, scw.y};
                float pr = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float s8 = (float) (int8_t) (sw[i >> 2] >> (8 * (i & 3)));
                    pr = fmaf(s8, sub[i], pr);
                }
                part[r] = dv * pr;
            }
#pragma unroll
            for (int r = 0; r < kQ6kRows; ++r) {
                acc[r] += part[r];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kQ6kRows; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down(v, off);
        }
        if (lane == 0 && row0 + r < rows) {
            y[row0 + r] = v;
        }
    }
}

// Q6_K for NT = 2..4 tokens (#174): the single-token kernel's operations per
// token (bit-identical results), the weights decoded once for all NT tokens and
// the activations read from global memory (L1 / L2), as the other NT kernels.
// Before it, a multi-token Q6_K matmul ran one launch per token.
template <int NT, int R>
__global__ void gemv_q6k_ntg_kernel(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                                    const uint8_t * __restrict__ sc, const uint8_t * __restrict__ d,
                                    const __half * __restrict__ x, float * __restrict__ y,
                                    const long long rows, const long long n_sub) {
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * R;
    if (row0 >= rows) {
        return;
    }
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    const long long k = n_sub * 32;

    float acc[R][NT] = {};
    for (long long g = lane; g < n_sub / 4; g += 32) {
        const long long blk = g >> 1;
        const int half = (int) (g & 1);
        float part[R][NT];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const long long row = row0 + r <= rlast ? row0 + r : rlast;
            const uint4 * pql =
                reinterpret_cast<const uint4 *>(ql + row * blocks * 128 + blk * 128 + half * 64);
            const uint4 * pqh =
                reinterpret_cast<const uint4 *>(qh + row * blocks * 64 + blk * 64 + half * 32);
            const uint2 scw = *reinterpret_cast<const uint2 *>(sc + row * blocks * 16 + blk * 16 + half * 8);
            const float dv = __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
            const uint4 l0 = pql[0], l1 = pql[1], l2 = pql[2], l3 = pql[3];
            const uint4 h0 = pqh[0], h1 = pqh[1];
            const uint32_t qlw[16] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w,
                                      l2.x, l2.y, l2.z, l2.w, l3.x, l3.y, l3.z, l3.w};
            const uint32_t qhw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
            float sub[NT][8] = {};
#pragma unroll
            for (int wi = 0; wi < 8; ++wi) {
                const int is = wi >> 2;
                const uint32_t lo = qlw[wi];
                const uint32_t hi = qlw[wi + 8];
                const uint32_t hb = qhw[wi];
                const uint32_t c[4] = {
                    (lo & 0x0F0F0F0Fu) | ((hb & 0x03030303u) << 4),
                    (hi & 0x0F0F0F0Fu) | ((hb & 0x0C0C0C0Cu) << 2),
                    ((lo >> 4) & 0x0F0F0F0Fu) | (hb & 0x30303030u),
                    ((hi >> 4) & 0x0F0F0F0Fu) | ((hb >> 2) & 0x30303030u)};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t w0 = q6_pair_to_half2<0x04010400u>(c[j]);
                    const uint32_t w1 = q6_pair_to_half2<0x04030402u>(c[j]);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const uint2 xv = reinterpret_cast<const uint2 *>(x + t * k + g * 128)[j * 8 + wi];
                        const float u = dot2(w0, xv.x, sub[t][2 * j + is]);
                        sub[t][2 * j + is] = dot2(w1, xv.y, u);
                    }
                }
            }
            const uint32_t sw[2] = {scw.x, scw.y};
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                float pr = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float s8 = (float) (int8_t) (sw[i >> 2] >> (8 * (i & 3)));
                    pr = fmaf(s8, sub[t][i], pr);
                }
                part[r][t] = dv * pr;
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[r][t] += part[r][t];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down(v, off);
            }
            if (lane == 0 && row0 + r < rows) {
                y[t * rows + row0 + r] = v;
            }
        }
    }
}


// IQ2_XS: 74-byte blocks. Per 32-weight sub-block: four uint16 entries (9-bit
// grid index into a 512-entry grid + 7-bit sign index) and one byte with two
// 4-bit scales, the low nibble for the first two entries. Grid (4 KiB) and sign
// table in LDS; one lane iteration covers one sub-block for kIq2Rows rows.
constexpr long long kIq2ChunkSub = 256;
constexpr long long kIq2Rows = 4;

__global__ void gemv_iq2xs_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ sc,
                                  const uint8_t * __restrict__ d, const __half * __restrict__ x,
                                  float * __restrict__ y, const long long rows,
                                  const long long n_sub, const long long chunk_sub) {
    extern __shared__ uint8_t smem[];
    __half * xs = reinterpret_cast<__half *>(smem);
    uint64_t * grid = reinterpret_cast<uint64_t *>(smem + chunk_sub * kXPitch * 2);
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        grid[i] = omph::quant::kIq2xsGrid[i];
    }

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kIq2Rows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kIq2Rows] = {};
    for (long long base = 0; base < n_sub; base += chunk_sub) {
        const long long len = n_sub - base < chunk_sub ? n_sub - base : chunk_sub;
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[(i / 32) * kXPitch + (i % 32)] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const __half * xv = xs + (s - base) * kXPitch;
            const long long blk = s / 8;
            const int ib = (int) (s % 8);
            uint32_t ent[kIq2Rows][2];
            float dbl[kIq2Rows][2];
#pragma unroll
            for (int r = 0; r < kIq2Rows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                const uint16_t * e =
                    reinterpret_cast<const uint16_t *>(qs + row * blocks * 64 + s * 8);
                ent[r][0] = (uint32_t) e[0] | ((uint32_t) e[1] << 16);
                ent[r][1] = (uint32_t) e[2] | ((uint32_t) e[3] << 16);
                const uint8_t scb = sc[row * blocks * 8 + blk * 8 + ib];
                const float dv =
                    __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                dbl[r][0] = dv * (0.5f + (float) (scb & 0xF)) * 0.25f;
                dbl[r][1] = dv * (0.5f + (float) (scb >> 4)) * 0.25f;
            }
            // Same scheme as IQ2_S / IQ3_XXS (#75): codebook bytes -> half2,
            // the pair-ordered sign word from the permuted entry (repack.cc:
            // evens reversed in bits 9-12, odds in 13-15; weight 7's sign is
            // the parity), two independent dot2 chains.
            float sub[kIq2Rows][2] = {};
            const uint4 * xq4 = reinterpret_cast<const uint4 *>(xv);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint4 xq = xq4[l];
#pragma unroll
                for (int r = 0; r < kIq2Rows; ++r) {
                    const uint32_t e = (ent[r][l >> 1] >> (16 * (l & 1))) & 0xFFFFu;
                    const uint64_t g = grid[e & 511u];
                    const uint32_t g1 = (uint32_t) g;
                    const uint32_t g2 = (uint32_t) (g >> 32);
                    const uint32_t f = e >> 9;
                    const uint32_t par = (uint32_t) __builtin_popcount(f) & 1u;
                    const uint32_t sgn = ((f & 0xFu) << 12) | ((f >> 4) << 29) | (par << 28);
                    constexpr uint32_t kSign2 = 0x80008000u;
                    float pa = dot2((sgn & kSign2) | grid_pair_to_half2<0x04010400u>(g1), xq.x, 0.0f);
                    float pb = dot2(((sgn << 2) & kSign2) | grid_pair_to_half2<0x04010400u>(g2), xq.z, 0.0f);
                    pa = dot2(((sgn << 1) & kSign2) | grid_pair_to_half2<0x04030402u>(g1), xq.y, pa);
                    pb = dot2(((sgn << 3) & kSign2) | grid_pair_to_half2<0x04030402u>(g2), xq.w, pb);
                    sub[r][l >> 1] += pa + pb;
                }
            }
#pragma unroll
            for (int r = 0; r < kIq2Rows; ++r) {
                acc[r] = fmaf(dbl[r][0], sub[r][0], acc[r]);
                acc[r] = fmaf(dbl[r][1], sub[r][1], acc[r]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kIq2Rows; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down(v, off);
        }
        if (lane == 0 && row0 + r < rows) {
            y[row0 + r] = v;
        }
    }
}

// IQ2_XXS: 66-byte blocks. Per 32-weight sub-block: one word with four grid
// byte-indices into a 256-entry grid, and one word carrying four 7-bit sign
// indices plus the 4-bit scale in the top nibble.
__global__ void gemv_iq2xxs_kernel(const uint8_t * __restrict__ qs,
                                   const uint8_t * __restrict__ d, const __half * __restrict__ x,
                                   float * __restrict__ y, const long long rows,
                                   const long long n_sub, const long long chunk_sub) {
    extern __shared__ uint8_t smem[];
    __half * xs = reinterpret_cast<__half *>(smem);
    uint64_t * grid = reinterpret_cast<uint64_t *>(smem + chunk_sub * 64);
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        grid[i] = omph::quant::kIq2xxsGrid[i];
    }

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kIq2Rows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kIq2Rows] = {};
    for (long long base = 0; base < n_sub; base += chunk_sub) {
        const long long len = n_sub - base < chunk_sub ? n_sub - base : chunk_sub;
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[i] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();
            continue;
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const __half * xv = xs + (s - base) * 32;
            const long long blk = s / 8;
            uint32_t aux0[kIq2Rows];
            uint32_t aux1[kIq2Rows];
            float db[kIq2Rows];
#pragma unroll
            for (int r = 0; r < kIq2Rows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                const uint32_t * w =
                    reinterpret_cast<const uint32_t *>(qs + row * blocks * 64 + s * 8);
                aux0[r] = w[0];
                aux1[r] = w[1];
                const float dv =
                    __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                db[r] = dv * (0.5f + (float) (aux1[r] >> 28)) * 0.25f;
            }
            // Same scheme as IQ3_XXS (#75): codebook bytes -> half2, the
            // pair-ordered sign word from the permuted aux, two dot2 chains.
            float part[kIq2Rows] = {};
            const uint4 * xq4 = reinterpret_cast<const uint4 *>(xv);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint4 xq = xq4[l];
#pragma unroll
                for (int r = 0; r < kIq2Rows; ++r) {
                    const uint64_t g = grid[(aux0[r] >> (8 * l)) & 0xFFu];
                    const uint32_t g1 = (uint32_t) g;
                    const uint32_t g2 = (uint32_t) (g >> 32);
                    const uint32_t sgn = iq3xxs_group_signs(aux1[r], l);
                    constexpr uint32_t kSign2 = 0x80008000u;
                    float pa = dot2((sgn & kSign2) | grid_pair_to_half2<0x04010400u>(g1), xq.x, 0.0f);
                    float pb = dot2(((sgn << 2) & kSign2) | grid_pair_to_half2<0x04010400u>(g2), xq.z, 0.0f);
                    pa = dot2(((sgn << 1) & kSign2) | grid_pair_to_half2<0x04030402u>(g1), xq.y, pa);
                    pb = dot2(((sgn << 3) & kSign2) | grid_pair_to_half2<0x04030402u>(g2), xq.w, pb);
                    part[r] += pa + pb;
                }
            }
#pragma unroll
            for (int r = 0; r < kIq2Rows; ++r) {
                acc[r] = fmaf(db[r], part[r], acc[r]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kIq2Rows; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down(v, off);
        }
        if (lane == 0 && row0 + r < rows) {
            y[row0 + r] = v;
        }
    }
}


// IQ2_S: 82-byte blocks. Per 32-weight sub-block: four low index bytes, four
// sign-mask bytes, one byte carrying the 2-bit index extension per entry and one
// byte with two 4-bit scales. Grid is 1024 x 8 B -> 8 KiB of LDS.
constexpr long long kIq2sChunkSub = 256;
constexpr long long kIq2sRows = 4;

__global__ void gemv_iq2s_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ sg,
                                 const uint8_t * __restrict__ qh, const uint8_t * __restrict__ sc,
                                 const uint8_t * __restrict__ d, const __half * __restrict__ x,
                                 float * __restrict__ y, const long long rows,
                                 const long long n_sub, const long long chunk_sub) {
    extern __shared__ uint8_t smem[];
    __half * xs = reinterpret_cast<__half *>(smem);
    uint64_t * grid = reinterpret_cast<uint64_t *>(smem + chunk_sub * kXPitch * 2);
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        grid[i] = omph::quant::kIq2sGrid[i];
    }

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kIq2sRows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kIq2sRows] = {};
    for (long long base = 0; base < n_sub; base += chunk_sub) {
        const long long len = n_sub - base < chunk_sub ? n_sub - base : chunk_sub;
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[(i / 32) * kXPitch + (i % 32)] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const __half * xv = xs + (s - base) * kXPitch;
            const long long blk = s / 8;
            const int ib = (int) (s % 8);
            uint32_t qsw[kIq2sRows];
            uint32_t sgw[kIq2sRows];
            uint32_t qhb[kIq2sRows];
            float dbl[kIq2sRows][2];
#pragma unroll
            for (int r = 0; r < kIq2sRows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                qsw[r] = *reinterpret_cast<const uint32_t *>(qs + row * blocks * 32 + s * 4);
                sgw[r] = *reinterpret_cast<const uint32_t *>(sg + row * blocks * 32 + s * 4);
                qhb[r] = qh[row * blocks * 8 + s];
                const uint8_t scb = sc[row * blocks * 8 + blk * 8 + ib];
                const float dv =
                    __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                dbl[r][0] = dv * (0.5f + (float) (scb & 0xF)) * 0.25f;
                dbl[r][1] = dv * (0.5f + (float) (scb >> 4)) * 0.25f;
            }
            // Same scheme as IQ3_S: codebook byte pairs -> half2, pair-ordered
            // signs OR-ed in, v_dot2; one u64 codebook entry per 8 weights.
            float sub[kIq2sRows][2] = {};
            const uint4 * xq4 = reinterpret_cast<const uint4 *>(xv);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint4 xq = xq4[l];
#pragma unroll
                for (int r = 0; r < kIq2sRows; ++r) {
                    const uint32_t lo = (qsw[r] >> (8 * l)) & 0xFFu;
                    const uint32_t hi = ((qhb[r] >> (2 * l)) & 3u) << 8;
                    const uint64_t g = grid[lo | hi];
                    const uint32_t g1 = (uint32_t) g;
                    const uint32_t g2 = (uint32_t) (g >> 32);
                    constexpr uint32_t kSign2 = 0x80008000u;
                    // two independent dot2 chains, so they do not stall on each other
                    float pa = dot2(((sgw[r] << (4 * l)) & kSign2) | grid_pair_to_half2<0x04010400u>(g1), xq.x, 0.0f);
                    float pb = dot2(((sgw[r] << (4 * l + 2)) & kSign2) | grid_pair_to_half2<0x04010400u>(g2), xq.z, 0.0f);
                    pa = dot2(((sgw[r] << (4 * l + 1)) & kSign2) | grid_pair_to_half2<0x04030402u>(g1), xq.y, pa);
                    pb = dot2(((sgw[r] << (4 * l + 3)) & kSign2) | grid_pair_to_half2<0x04030402u>(g2), xq.w, pb);
                    sub[r][l >> 1] += pa + pb;
                }
            }
#pragma unroll
            for (int r = 0; r < kIq2sRows; ++r) {
                acc[r] = fmaf(dbl[r][0], sub[r][0], acc[r]);
                acc[r] = fmaf(dbl[r][1], sub[r][1], acc[r]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kIq2sRows; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down(v, off);
        }
        if (lane == 0 && row0 + r < rows) {
            y[row0 + r] = v;
        }
    }
}

// BF16 weights, single token: bf16 is the top half of an f32, so the conversion
// is a shift and the kernel is a plain bandwidth-bound dot product. This exists
// because the f16 path has to convert these small matrices and then runs a tiny
// M = 1 GEMM, which costs ~40 us per call regardless of the size.
__device__ inline float bf16_to_f32(const uint16_t b) {
    return __uint_as_float((uint32_t) b << 16);
}

constexpr long long kBf16Threads = 256;  // eight warps per row

// One workgroup per output row, the k dimension split across the warps with an
// LDS reduction: with 48 rows there is no other way to fill the GPU, and the row
// at 10 KB is far too small for a warp-per-row scheme.
__global__ void gemv_bf16_kernel(const uint16_t * __restrict__ w, const __half * __restrict__ x,
                                 float * __restrict__ y, const long long rows, const long long k) {
    __shared__ float red[kBf16Threads / 32];
    const long long row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const int nwarp = (int) (blockDim.x / 32);
    const uint16_t * wrow = w + row * k;

    const long long per_warp = (k + nwarp - 1) / nwarp;
    const long long begin = (long long) warp * per_warp;
    const long long end = begin + per_warp < k ? begin + per_warp : k;

    float acc = 0.0f;
    for (long long i = begin + lane * 8; i < end; i += 32 * 8) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (i + j < end) {
                acc = fmaf(bf16_to_f32(wrow[i + j]), __half2float(x[i + j]), acc);
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        red[warp] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < nwarp; ++i) {
            total += red[i];
        }
        y[row] = total;
    }
}

// IQ1_M straight from the GGUF blocks (56 bytes per 256 weights: qs[32],
// qh[16], four u16 scale words whose top nibbles hold the f16 block scale).
// One lane per 32-weight sub-block, four rows per warp; the 2048-entry int8
// codebook (16 KiB) in LDS. Per group of 8 weights the value is
// dl * (g_j + delta), so the dot is dl * (sum g_j x_j + delta * sum x_j).
constexpr long long kIq1mChunkSub = 160;  // 5120 weights staged per LDS refill
constexpr long long kIq1mRows = 4;

__global__ void gemv_iq1_m_kernel(const uint8_t * __restrict__ blocks_in,
                                  const __half * __restrict__ x, float * __restrict__ y,
                                  const long long rows, const long long n_sub,
                                  const long long chunk_sub) {
    extern __shared__ uint8_t smem[];
    uint64_t * grid = reinterpret_cast<uint64_t *>(smem);                 // 2048 x 8 int8
    __half * xs = reinterpret_cast<__half *>(smem + 2048 * sizeof(uint64_t));
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
        grid[i] = omph::quant::kIq1sGrid[i];
    }
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kIq1mRows;
    const long long nblk = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kIq1mRows] = {};
    for (long long base = 0; base < n_sub; base += chunk_sub) {
        const long long len = n_sub - base < chunk_sub ? n_sub - base : chunk_sub;
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[i] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 < rows) {
            for (long long s = base + lane; s < base + len; s += 32) {
                const long long blk = s / 8;
                const int ib = (int) (s % 8);
                const __half * xv = xs + (s - base) * 32;
                float xsum[4];
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    float t = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        t += __half2float(xv[8 * l + j]);
                    }
                    xsum[l] = t;
                }
#pragma unroll
                for (int r = 0; r < kIq1mRows; ++r) {
                    const long long row = row0 + r <= rlast ? row0 + r : rlast;
                    const uint8_t * b = blocks_in + (row * nblk + blk) * 56;
                    const uint32_t qs4 = *reinterpret_cast<const uint32_t *>(b + 4 * ib);
                    const uint32_t qh2 = *reinterpret_cast<const uint16_t *>(b + 32 + 2 * ib);
                    const uint64_t sc = *reinterpret_cast<const uint64_t *>(b + 48);
                    const uint16_t s0 = (uint16_t) sc;
                    const uint16_t s1 = (uint16_t) (sc >> 16);
                    const uint16_t s2 = (uint16_t) (sc >> 32);
                    const uint16_t s3 = (uint16_t) (sc >> 48);
                    const uint16_t dbits = (uint16_t) ((s0 >> 12) | ((s1 >> 8) & 0x00F0) |
                                                       ((s2 >> 4) & 0x0F00) | (s3 & 0xF000));
                    const float d = __half2float(__builtin_bit_cast(__half, dbits));
                    const uint16_t scw = (uint16_t) (sc >> (16 * (ib >> 1)));
                    const int sh = 6 * (ib & 1);
                    const float dl1 = d * (2.0f * (float) ((scw >> sh) & 7) + 1.0f);
                    const float dl2 = d * (2.0f * (float) ((scw >> (sh + 3)) & 7) + 1.0f);
                    float p12 = 0.0f;  // groups 0, 1 (scale dl1)
                    float p34 = 0.0f;  // groups 2, 3 (scale dl2)
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const uint32_t qh = (qh2 >> (8 * (l >> 1))) & 0xFFu;
                        const uint32_t idx =
                            ((qs4 >> (8 * l)) & 0xFFu) | ((qh << ((l & 1) ? 4 : 8)) & 0x700u);
                        const float delta = (qh & ((l & 1) ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
                        const uint64_t g = grid[idx];
                        float t = delta * xsum[l];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            t = fmaf((float) (int8_t) (g >> (8 * j)), __half2float(xv[8 * l + j]), t);
                        }
                        if (l < 2) {
                            p12 += t;
                        } else {
                            p34 += t;
                        }
                    }
                    acc[r] = fmaf(dl1, p12, fmaf(dl2, p34, acc[r]));
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kIq1mRows; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down(v, off);
        }
        if (lane == 0 && row0 + r < rows) {
            y[row0 + r] = v;
        }
    }
}

} // namespace

// Workgroups of `kernel` (256 threads, `smem` bytes of LDS) the GPU holds at
// once: the grid of a persistent-warp launch (#63). Cached per kernel and LDS size.
int persistent_wgs(const void * kernel, const size_t smem) {
    static std::vector<std::pair<std::pair<const void *, size_t>, int>> cache;
    for (const auto & e : cache) {
        if (e.first.first == kernel && e.first.second == smem) return e.second;
    }
    int per_cu = 0;
    int dev = 0;
    hipDeviceProp_t prop{};
    (void) hipGetDevice(&dev);
    (void) hipGetDeviceProperties(&prop, dev);
    (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, kernel, 256, smem);
    const int n = (per_cu > 0 ? per_cu : 1) * prop.multiProcessorCount;
    cache.push_back({{kernel, smem}, n});
    return n;
}

// The grid of a single-token GEMV: every row group's workgroup, or (rows of
// one x chunk: the persistent-warp path) no more than fit at once.
unsigned gemv_grid(const void * kernel, const size_t smem, const int64_t rows, const int64_t rows_per_wg,
                   const bool one_chunk) {
    const int64_t need = (rows + rows_per_wg - 1) / rows_per_wg;
    if (!one_chunk) return (unsigned) need;
    const int64_t fit = persistent_wgs(kernel, smem);
    return (unsigned) (need < fit ? need : fit);
}

bool gemv_iq1_m(const void * blocks, const void * x_f16, float * y, const int64_t rows,
                const int64_t k, hipStream_t stream) {
    if (blocks == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0) {
        return false;
    }
    const int64_t n_sub = k / 32;
    const int64_t chunk = n_sub < kIq1mChunkSub ? n_sub : kIq1mChunkSub;
    const size_t smem = 2048 * sizeof(uint64_t) + (size_t) chunk * 64;
    const dim3 grid((unsigned) ((rows + 8 * kIq1mRows - 1) / (8 * kIq1mRows)));
    gemv_iq1_m_kernel<<<grid, 256, smem, stream>>>(static_cast<const uint8_t *>(blocks),
                                                   static_cast<const __half *>(x_f16), y, rows,
                                                   n_sub, chunk);
    return hipGetLastError() == hipSuccess;
}

bool gemv_q4k(const void * packed, const void * x_f16, float * y, const int64_t rows,
              const int64_t k, hipStream_t stream) {
    if (packed == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0) {
        return false;
    }
    const int64_t n_blocks = rows * (k / 256);
    const omph::format::Q4kLayout l = omph::format::q4k_layout(n_blocks);
    const size_t smem = (size_t) kQ4kChunkSub * 64;
    const auto * base = static_cast<const uint8_t *>(packed);
    const int64_t n_sub = k / 32;
    const dim3 grid(gemv_grid(reinterpret_cast<const void *>(gemv_q4k_kernel), smem, rows, 8 * kQ4kRows, n_sub <= kQ4kChunkSub));
    gemv_q4k_kernel<<<grid, 256, smem, stream>>>(
        base + l.qs_off, base + l.meta_off,
        static_cast<const __half *>(x_f16), y, rows, n_sub);
    return hipGetLastError() == hipSuccess;
}

bool gemv_iq2_s(const void * packed, const void * x_f16, float * y, const int64_t rows,
                const int64_t k, hipStream_t stream) {
    if (packed == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0) {
        return false;
    }
    const int64_t n_blocks = rows * (k / 256);
    const omph::format::Iq2sLayout l = omph::format::iq2s_layout(n_blocks);
    const int64_t n_sub = k / 32;
    const int64_t chunk = n_sub < kIq2sChunkSub ? n_sub : kIq2sChunkSub;
    const size_t smem = (size_t) chunk * kXPitch * 2 + 1024 * 8;
    const auto * base = static_cast<const uint8_t *>(packed);
    const dim3 grid((unsigned) ((rows + 8 * kIq2sRows - 1) / (8 * kIq2sRows)));
    gemv_iq2s_kernel<<<grid, 256, smem, stream>>>(base + l.qs_off, base + l.signs_off,
                                                  base + l.qh_off, base + l.sc_off, base + l.d_off,
                                                  static_cast<const __half *>(x_f16), y, rows,
                                                  n_sub, chunk);
    return hipGetLastError() == hipSuccess;
}

bool gemv_iq2_xs(const void * packed, const void * x_f16, float * y, const int64_t rows,
                 const int64_t k, hipStream_t stream) {
    if (packed == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0) {
        return false;
    }
    const int64_t n_blocks = rows * (k / 256);
    const omph::format::Iq2Layout l = omph::format::iq2_xs_layout(n_blocks);
    const int64_t n_sub = k / 32;
    const int64_t chunk = n_sub < kIq2ChunkSub ? n_sub : kIq2ChunkSub;
    const size_t smem = (size_t) chunk * kXPitch * 2 + 512 * 8;
    const auto * base = static_cast<const uint8_t *>(packed);
    const dim3 grid((unsigned) ((rows + 8 * kIq2Rows - 1) / (8 * kIq2Rows)));
    gemv_iq2xs_kernel<<<grid, 256, smem, stream>>>(base + l.qs_off, base + l.sc_off,
                                                   base + l.d_off,
                                                   static_cast<const __half *>(x_f16), y, rows,
                                                   n_sub, chunk);
    return hipGetLastError() == hipSuccess;
}

bool gemv_iq2_xxs(const void * packed, const void * x_f16, float * y, const int64_t rows,
                  const int64_t k, hipStream_t stream) {
    if (packed == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0) {
        return false;
    }
    const int64_t n_blocks = rows * (k / 256);
    const omph::format::Iq2Layout l = omph::format::iq2_xxs_layout(n_blocks);
    const int64_t n_sub = k / 32;
    const int64_t chunk = n_sub < kIq2ChunkSub ? n_sub : kIq2ChunkSub;
    const size_t smem = (size_t) chunk * 64 + 256 * 8;
    const auto * base = static_cast<const uint8_t *>(packed);
    const dim3 grid((unsigned) ((rows + 8 * kIq2Rows - 1) / (8 * kIq2Rows)));
    gemv_iq2xxs_kernel<<<grid, 256, smem, stream>>>(base + l.qs_off, base + l.d_off,
                                                    static_cast<const __half *>(x_f16), y, rows,
                                                    n_sub, chunk);
    return hipGetLastError() == hipSuccess;
}

bool gemv_q6k(const void * packed, const void * x_f16, float * y, const int64_t rows,
              const int64_t k, hipStream_t stream) {
    if (packed == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0) {
        return false;
    }
    const int64_t n_blocks = rows * (k / 256);
    const omph::format::Q6kLayout l = omph::format::q6k_layout(n_blocks);
    const int64_t n_sub = k / 32;
    const int64_t chunk = n_sub < kQ6kChunkSub ? n_sub : kQ6kChunkSub;
    const size_t smem = (size_t) (chunk / 4) * kQ6kXPitch * 2;  // padded groups
    const auto * base = static_cast<const uint8_t *>(packed);
    const dim3 grid(gemv_grid(reinterpret_cast<const void *>(gemv_q6k_kernel), smem, rows, 8 * kQ6kRows, n_sub <= chunk));
    gemv_q6k_kernel<<<grid, 256, smem, stream>>>(base + l.ql_off, base + l.qh_off,
                                                 base + l.sc_off, base + l.d_off,
                                                 static_cast<const __half *>(x_f16), y, rows,
                                                 n_sub, chunk);
    return hipGetLastError() == hipSuccess;
}

bool gemv_q2k(const void * packed, const void * x_f16, float * y, const int64_t rows,
              const int64_t k, hipStream_t stream) {
    if (packed == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0) {
        return false;
    }
    const int64_t n_blocks = rows * (k / 256);
    const omph::format::Q2kLayout l = omph::format::q2k_layout(n_blocks);
    const int64_t n_sub = k / 32;
    const int64_t chunk = n_sub < kQ2kChunkSub ? n_sub : kQ2kChunkSub;
    const size_t smem = (size_t) (chunk / 4) * kQ2kXPitch * 2;  // padded groups
    const auto * base = static_cast<const uint8_t *>(packed);
    const dim3 grid(gemv_grid(reinterpret_cast<const void *>(gemv_q2k_kernel), smem, rows, 8 * kQ2kRows, n_sub <= chunk));
    gemv_q2k_kernel<<<grid, 256, smem, stream>>>(base + l.sc_off, base + l.qs_off,
                                                 base + l.d_off, base + l.dmin_off,
                                                 static_cast<const __half *>(x_f16), y, rows,
                                                 n_sub, chunk);
    return hipGetLastError() == hipSuccess;
}

bool gemv_iq3_s(const void * packed, const void * x_f16, float * y, const int64_t rows,
                const int64_t k, hipStream_t stream) {
    if (packed == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0) {
        return false;
    }
    const int64_t n_blocks = rows * (k / 256);
    const omph::format::Iq3sLayout l = omph::format::iq3s_layout(n_blocks);
    const int64_t n_sub = k / 32;
    const int64_t chunk = n_sub < kIq3sChunkSub ? n_sub : kIq3sChunkSub;
    const size_t smem = (size_t) chunk * 64 + 512 * 4;
    const auto * base = static_cast<const uint8_t *>(packed);
    if (n_sub <= kIq3sChunkSub) {
        // persistent warps: as many workgroups as fit at once (queried once)
        static int fit = 0;
        if (fit == 0) {
            int per_cu = 0;
            int dev = 0;
            hipDeviceProp_t prop{};
            (void) hipGetDevice(&dev);
            (void) hipGetDeviceProperties(&prop, dev);
            (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, gemv_iq3s_pw_kernel, 256,
                                                                (size_t) kIq3sChunkSub * 64 + 512 * 4);
            fit = (per_cu > 0 ? per_cu : 1) * prop.multiProcessorCount;
        }
        const long long groups = (rows + kIq3sRows - 1) / kIq3sRows;
        const long long need = (groups + 7) / 8;
        const unsigned wgs = (unsigned) (need < fit ? need : fit);
        gemv_iq3s_pw_kernel<<<wgs, 256, (size_t) n_sub * 64 + 512 * 4, stream>>>(
            base + l.qs_off, base + l.qh_off, base + l.signs_off, base + l.scales_off, base + l.d_off,
            static_cast<const __half *>(x_f16), y, rows, n_sub);
        return hipGetLastError() == hipSuccess;
    }
    const dim3 grid((unsigned) ((rows + 8 * kIq3sRows - 1) / (8 * kIq3sRows)));
    gemv_iq3s_kernel<<<grid, 256, smem, stream>>>(base + l.qs_off, base + l.qh_off,
                                                  base + l.signs_off, base + l.scales_off,
                                                  base + l.d_off,
                                                  static_cast<const __half *>(x_f16), y, rows,
                                                  n_sub, chunk);
    return hipGetLastError() == hipSuccess;
}

// Launches an NT-token kernel: activations in LDS for NT tokens, so the chunk
// shrinks by NT to stay inside the 64 KB the single-token kernels assume.
template <int NT, int R = 4>
bool launch_iq2xs_ntg(const void * packed, const void * x_f16, float * y, const int64_t rows,
                      const int64_t k, hipStream_t stream) {
    const omph::format::Iq2Layout l = omph::format::iq2_xs_layout(rows * (k / 256));
    const auto * base = static_cast<const uint8_t *>(packed);
    gemv_iq2xs_ntg_kernel<NT, R><<<(unsigned) ((rows + 8 * R - 1) / (8 * R)), 256, 0, stream>>>(
        base + l.qs_off, base + l.sc_off, base + l.d_off, static_cast<const __half *>(x_f16), y,
        rows, k / 32);
    return hipGetLastError() == hipSuccess;
}

template <int NT, int R = 4>
bool launch_iq2xxs_ntg(const void * packed, const void * x_f16, float * y, const int64_t rows,
                       const int64_t k, hipStream_t stream) {
    const omph::format::Iq2Layout l = omph::format::iq2_xxs_layout(rows * (k / 256));
    const auto * base = static_cast<const uint8_t *>(packed);
    gemv_iq2xxs_ntg_kernel<NT, R><<<(unsigned) ((rows + 8 * R - 1) / (8 * R)), 256, 0, stream>>>(
        base + l.qs_off, base + l.d_off, static_cast<const __half *>(x_f16), y, rows, k / 32);
    return hipGetLastError() == hipSuccess;
}

template <int NT, int R = 4>
bool launch_iq2s_ntg(const void * packed, const void * x_f16, float * y, const int64_t rows,
                     const int64_t k, hipStream_t stream) {
    const omph::format::Iq2sLayout l = omph::format::iq2s_layout(rows * (k / 256));
    const auto * base = static_cast<const uint8_t *>(packed);
    gemv_iq2s_ntg_kernel<NT, R><<<(unsigned) ((rows + 8 * R - 1) / (8 * R)), 256, 0, stream>>>(
        base + l.qs_off, base + l.signs_off, base + l.qh_off, base + l.sc_off, base + l.d_off,
        static_cast<const __half *>(x_f16), y, rows, k / 32);
    return hipGetLastError() == hipSuccess;
}

template <int NT, int R = 4>
bool launch_q2k_ntg(const void * packed, const void * x_f16, float * y, const int64_t rows,
                    const int64_t k, hipStream_t stream) {
    const omph::format::Q2kLayout l = omph::format::q2k_layout(rows * (k / 256));
    const auto * base = static_cast<const uint8_t *>(packed);
    gemv_q2k_ntg_kernel<NT, R><<<(unsigned) ((rows + 8 * R - 1) / (8 * R)), 256, 0, stream>>>(
        base + l.sc_off, base + l.qs_off, base + l.d_off, base + l.dmin_off,
        static_cast<const __half *>(x_f16), y, rows, k / 32);
    return hipGetLastError() == hipSuccess;
}

template <int NT, int R = 4>
bool launch_q4k_ntg(const void * packed, const void * x_f16, float * y, const int64_t rows,
                    const int64_t k, hipStream_t stream) {
    const omph::format::Q4kLayout l = omph::format::q4k_layout(rows * (k / 256));
    const auto * base = static_cast<const uint8_t *>(packed);
    gemv_q4k_ntg_kernel<NT, R><<<(unsigned) ((rows + 8 * R - 1) / (8 * R)), 256, 0, stream>>>(
        base + l.qs_off, base + l.meta_off, static_cast<const __half *>(x_f16), y, rows, k / 32);
    return hipGetLastError() == hipSuccess;
}

template <int NT, int R = 4>
bool launch_iq4_ntg(const void * packed, const void * x_f16, float * y, const int64_t rows,
                    const int64_t k, hipStream_t stream) {
    const omph::format::Iq4Layout l = omph::format::iq4_layout(rows * (k / 256));
    const auto * base = static_cast<const uint8_t *>(packed);
    gemv_iq4_ntg_kernel<NT, R><<<(unsigned) ((rows + 8 * R - 1) / (8 * R)), 256, 0, stream>>>(
        base + l.qs_off, base + l.meta_off, static_cast<const __half *>(x_f16), y, rows, k / 32);
    return hipGetLastError() == hipSuccess;
}

template <int NT, int R = 4>
bool launch_iq3_ntg(const void * packed, const void * x_f16, float * y, const int64_t rows,
                    const int64_t k, hipStream_t stream) {
    const omph::format::Iq3XxsLayout l = omph::format::iq3_xxs_layout(rows * (k / 256));
    const auto * base = static_cast<const uint8_t *>(packed);
    gemv_iq3_ntg_kernel<NT, R><<<(unsigned) ((rows + 8 * R - 1) / (8 * R)), 256, 0, stream>>>(
        base + l.qs_off, base + l.aux_off, base + l.d_off, static_cast<const __half *>(x_f16), y,
        rows, k / 32);
    return hipGetLastError() == hipSuccess;
}

template <int NT, int R = 4>
bool launch_iq3s_ntg(const void * packed, const void * x_f16, float * y, const int64_t rows,
                     const int64_t k, hipStream_t stream) {
    const int64_t n_blocks = rows * (k / 256);
    const omph::format::Iq3sLayout l = omph::format::iq3s_layout(n_blocks);
    const auto * base = static_cast<const uint8_t *>(packed);
    static int fit = 0;  // workgroups that fit at once (persistent warps)
    if (fit == 0) {
        int per_cu = 0;
        int dev = 0;
        hipDeviceProp_t prop{};
        (void) hipGetDevice(&dev);
        (void) hipGetDeviceProperties(&prop, dev);
        (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, gemv_iq3s_ntg_kernel<NT, R>, 256, 512 * 4);
        fit = (per_cu > 0 ? per_cu : 1) * prop.multiProcessorCount;
    }
    const long long need = (rows + 8 * R - 1) / (8 * R);
    // NT = 4 is compute-bound: the full grid (each warp one group) measured faster
    const long long cap = NT <= kIq3sNtPersistMax ? (long long) fit : need;
    const dim3 grid((unsigned) (need < cap ? need : cap));
    gemv_iq3s_ntg_kernel<NT, R><<<grid, 256, 512 * 4, stream>>>(
        base + l.qs_off, base + l.qh_off, base + l.signs_off, base + l.scales_off, base + l.d_off,
        static_cast<const __half *>(x_f16), y, rows, k / 32, 0);
    return hipGetLastError() == hipSuccess;
}

bool gemv_iq3_xxs(const void * packed, const void * x_f16, float * y, const int64_t rows,
                 const int64_t k, hipStream_t stream) {
    if (packed == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0) {
        return false;
    }
    const int64_t n_blocks = rows * (k / 256);
    const omph::format::Iq3XxsLayout l = omph::format::iq3_xxs_layout(n_blocks);
    const int64_t n_sub = k / 32;
    const int64_t chunk = n_sub < kIq3ChunkSub ? n_sub : kIq3ChunkSub;
    const size_t smem = (size_t) chunk * kXPitch * 2 + 256 * 4;
    const auto * base = static_cast<const uint8_t *>(packed);
    const dim3 grid(gemv_grid(reinterpret_cast<const void *>(gemv_iq3_kernel), smem, rows, 8 * kIq3Rows, n_sub <= chunk));
    gemv_iq3_kernel<<<grid, 256, smem, stream>>>(base + l.qs_off, base + l.aux_off,
                                                 base + l.d_off,
                                                 static_cast<const __half *>(x_f16), y, rows,
                                                 n_sub, chunk);
    return hipGetLastError() == hipSuccess;
}

bool gemv_iq4_xs(const void * packed, const void * x_f16, float * y, const int64_t rows,
                 const int64_t k, hipStream_t stream) {
    if (packed == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0) {
        return false;
    }
    const int64_t n_blocks = rows * (k / 256);
    const omph::format::Iq4Layout l = omph::format::iq4_layout(n_blocks);
    const int64_t n_sub = k / 32;
    const int64_t chunk = n_sub < kIq4ChunkSub ? n_sub : kIq4ChunkSub;
    const size_t smem = (size_t) chunk * 64;
    if (smem > 64 * 1024) {
        return false;
    }
    const auto * base = static_cast<const uint8_t *>(packed);
    const dim3 grid(gemv_grid(reinterpret_cast<const void *>(gemv_iq4_kernel), smem, rows, 8 * kIq4Rows,
                              n_sub <= chunk));
    gemv_iq4_kernel<<<grid, 256, smem, stream>>>(base + l.qs_off, base + l.meta_off,
                                                 static_cast<const __half *>(x_f16), y, rows,
                                                 n_sub, chunk);
    return hipGetLastError() == hipSuccess;
}

GemvOccupancy gemv_occupancy() {
    GemvOccupancy o;
    const void * kernels[4] = {(const void *) gemv_q4k_kernel, (const void *) gemv_iq4_kernel,
                               (const void *) gemv_iq3_kernel, (const void *) gemv_iq3s_kernel};
    o.smem_q4k = (size_t) kQ4kChunkSub * 64;
    o.smem_iq4 = (size_t) kIq4ChunkSub * 64;
    o.smem_iq3 = (size_t) kIq3ChunkSub * kXPitch * 2 + 256 * 4;
    o.smem_iq3s = (size_t) kIq3sChunkSub * 64 + 512 * 4;
    int n = 0;
    const size_t smems[4] = {o.smem_q4k, o.smem_iq4, o.smem_iq3, o.smem_iq3s};
    int results[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&results[i], kernels[i], 256,
                                                            smems[i]);
    }
    o.q4k = results[0];
    o.iq4 = results[1];
    o.iq3 = results[2];
    o.iq3s = results[3];
    o.block = 256;
    (void) n;
    return o;
}

bool gemv_bf16(const void * w, const void * x_f16, float * y, const int64_t rows, const int64_t k,
               hipStream_t stream) {
    if (w == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 || k % 256 != 0) {
        return false;
    }
    gemv_bf16_kernel<<<(unsigned) rows, kBf16Threads, 0, stream>>>(
        static_cast<const uint16_t *>(w), static_cast<const __half *>(x_f16), y, rows, k);
    return hipGetLastError() == hipSuccess;
}

template <int NT, int R = 2>
bool launch_q6k_ntg(const void * packed, const void * x_f16, float * y, const int64_t rows,
                    const int64_t k, hipStream_t stream) {
    const omph::format::Q6kLayout l = omph::format::q6k_layout(rows * (k / 256));
    const auto * base = static_cast<const uint8_t *>(packed);
    gemv_q6k_ntg_kernel<NT, R><<<(unsigned) ((rows + 8 * R - 1) / (8 * R)), 256, 0, stream>>>(
        base + l.ql_off, base + l.qh_off, base + l.sc_off, base + l.d_off, static_cast<const __half *>(x_f16), y,
        rows, k / 32);
    return hipGetLastError() == hipSuccess;
}

bool gemv_multi(const uint32_t type, const void * packed, const void * x_f16, float * y,
                const int64_t rows, const int64_t k, const int nt, hipStream_t stream) {
    if (packed == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0 || nt < 2 || nt > 4) {
        return false;
    }
    switch (type) {
        case 21:
            if (nt == 2) return launch_iq3s_ntg<2, 4>(packed, x_f16, y, rows, k, stream);
            if (nt == 3) return launch_iq3s_ntg<3, 4>(packed, x_f16, y, rows, k, stream);
            return launch_iq3s_ntg<4, 4>(packed, x_f16, y, rows, k, stream);
        case 18:
            if (nt == 2) return launch_iq3_ntg<2>(packed, x_f16, y, rows, k, stream);
            if (nt == 3) return launch_iq3_ntg<3>(packed, x_f16, y, rows, k, stream);
            return launch_iq3_ntg<4, 2>(packed, x_f16, y, rows, k, stream);
        case 12:
            if (nt == 2) return launch_q4k_ntg<2>(packed, x_f16, y, rows, k, stream);
            if (nt == 3) return launch_q4k_ntg<3>(packed, x_f16, y, rows, k, stream);
            return launch_q4k_ntg<4>(packed, x_f16, y, rows, k, stream);
        case 23:
            if (nt == 2) return launch_iq4_ntg<2>(packed, x_f16, y, rows, k, stream);
            if (nt == 3) return launch_iq4_ntg<3>(packed, x_f16, y, rows, k, stream);
            return launch_iq4_ntg<4>(packed, x_f16, y, rows, k, stream);
        case 17:
            if (nt == 2) return launch_iq2xs_ntg<2>(packed, x_f16, y, rows, k, stream);
            if (nt == 3) return launch_iq2xs_ntg<3>(packed, x_f16, y, rows, k, stream);
            return launch_iq2xs_ntg<4>(packed, x_f16, y, rows, k, stream);
        case 16:
            if (nt == 2) return launch_iq2xxs_ntg<2>(packed, x_f16, y, rows, k, stream);
            if (nt == 3) return launch_iq2xxs_ntg<3>(packed, x_f16, y, rows, k, stream);
            return launch_iq2xxs_ntg<4>(packed, x_f16, y, rows, k, stream);
        case 22:
            if (nt == 2) return launch_iq2s_ntg<2>(packed, x_f16, y, rows, k, stream);
            if (nt == 3) return launch_iq2s_ntg<3>(packed, x_f16, y, rows, k, stream);
            return launch_iq2s_ntg<4>(packed, x_f16, y, rows, k, stream);
        case 10:
            if (nt == 2) return launch_q2k_ntg<2>(packed, x_f16, y, rows, k, stream);
            if (nt == 3) return launch_q2k_ntg<3>(packed, x_f16, y, rows, k, stream);
            return launch_q2k_ntg<4>(packed, x_f16, y, rows, k, stream);
        case 14:
            if (nt == 2) return launch_q6k_ntg<2>(packed, x_f16, y, rows, k, stream);
            if (nt == 3) return launch_q6k_ntg<3>(packed, x_f16, y, rows, k, stream);
            return launch_q6k_ntg<4>(packed, x_f16, y, rows, k, stream);
        default: return false;
    }
}

// --- dequantization from the repacked layouts (M8) --------------------------
//
// f16 rows of a repacked tensor for the GEMM path, read from VRAM: a --gemv
// runner keeps no GGUF bytes, and staging them from the file costs PCIe time.
// One thread per 32-weight sub-block: the GEMV kernels' decode, then each
// weight as the raw dequant kernels compute it (dequant.hip: same float
// expressions, same order), written as four uint4. The grid values come out
// of the decode signed (exact in f16), and db * (+-g) == +-(db * g) exactly, so
// the f16 output is bit-identical to dequantizing the GGUF bytes.
namespace {

// Thread i of a launch over rows row0 .. row0 + nrows - 1: its tensor row and
// sub-block, and the output row (relative to row0).
__device__ inline bool rp_index(const long long row0, const long long nrows,
                                const long long n_sub, long long & row, long long & s,
                                long long & out_row) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows * n_sub) {
        return false;
    }
    out_row = i / n_sub;
    row = row0 + out_row;
    s = i % n_sub;
    return true;
}

// One sub-block per thread, decoded by the type's decoder (quant_decode.hh)
// and written as four uint4.
template <class Dec>
__global__ void rp_kernel(const Dec dec, __half * __restrict__ dst, const long long row0,
                          const long long rows, const long long n_sub) {
    long long row, s, out_row;
    if (!rp_index(row0, rows, n_sub, row, s, out_row)) return;
    float v[32];
    dec.decode(dec.load(row, s), s, v);
    uint4 o[4];
    qd::pack32(v, o);
    uint4 * out = reinterpret_cast<uint4 *>(dst + out_row * n_sub * 32 + s * 32);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i] = o[i];
    }
}

} // namespace

bool dequant_repacked(const uint32_t type, const void * packed, void * dst_f16,
                      const int64_t rows, const int64_t k, hipStream_t stream, int64_t row0,
                      int64_t nrows) {
    if (nrows < 0) {
        nrows = rows - row0;
    }
    if (packed == nullptr || dst_f16 == nullptr || rows <= 0 || k <= 0 || k % 256 != 0 ||
        row0 < 0 || nrows <= 0 || row0 + nrows > rows) {
        return false;
    }
    const auto * base = static_cast<const uint8_t *>(packed);
    auto * dst = static_cast<__half *>(dst_f16);
    const int64_t n_blocks = rows * (k / 256);
    const long long n_sub = k / 32;
    const unsigned grid = (unsigned) ((nrows * n_sub + 255) / 256);
    const long long blocks = k / 256;
    const auto launch = [&](const auto dec) {
        rp_kernel<<<grid, 256, 0, stream>>>(dec, dst, row0, nrows, n_sub);
    };
    switch (type) {
        case 21: launch(qd::DecIq3s::make(base, n_blocks, blocks)); break;
        case 18: launch(qd::DecIq3xxs::make(base, n_blocks, blocks)); break;
        case 16: launch(qd::DecIq2xxs::make(base, n_blocks, blocks)); break;
        case 17: launch(qd::DecIq2xs::make(base, n_blocks, blocks)); break;
        case 22: launch(qd::DecIq2s::make(base, n_blocks, blocks)); break;
        case 23: launch(qd::DecIq4xs::make(base, n_blocks, blocks)); break;
        case 12: launch(qd::DecQ4k::make(base, n_blocks, blocks)); break;
        case 10: launch(qd::DecQ2k::make(base, n_blocks, blocks)); break;
        case 14: launch(qd::DecQ6k::make(base, n_blocks, blocks)); break;
        default: return false;
    }
    return hipGetLastError() == hipSuccess;
}

} // namespace omph::kernels
