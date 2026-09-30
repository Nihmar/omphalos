// Fused dequant+dot GEMV for the repacked Q4_K layout.
//
// One warp per output row (8 warps per workgroup). The activation vector is
// staged in LDS as f16; each lane walks 32-weight sub-blocks in strides of 32
// and loads 16 contiguous, aligned bytes per sub-block. Weights are dequantized
// in registers, so quantized bytes are read exactly once and nothing else
// touches memory.
#include "kernels/gemv.hh"

#include "format/repack.hh"
#include "kernels/iq_tables.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace omph::kernels {
namespace {

constexpr long long kQ4kChunkSub = 256;  // 8192 weights staged per LDS refill
constexpr long long kQ4kRows = 4;        // output rows computed per warp

__global__ void gemv_q4k_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ sc,
                                const uint8_t * __restrict__ mn, const uint8_t * __restrict__ d,
                                const uint8_t * __restrict__ dmin, const __half * __restrict__ x,
                                float * __restrict__ y, const long long rows,
                                const long long n_sub) {
    extern __shared__ __half xs[];  // activation chunk

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    // kQ4kRows consecutive output rows per warp: one activation value feeds all
    // of them and the accumulators give independent FMA chains.
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kQ4kRows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kQ4kRows] = {};
    for (long long base = 0; base < n_sub; base += kQ4kChunkSub) {
        const long long len = n_sub - base < kQ4kChunkSub ? n_sub - base : kQ4kChunkSub;
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
            const __half * xv = xs + (s - base) * 32;
            uint4 q[kQ4kRows];
            float ds[kQ4kRows];
            float ms[kQ4kRows];
#pragma unroll
            for (int r = 0; r < kQ4kRows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                q[r] = *reinterpret_cast<const uint4 *>(qs + row * blocks * 128 + s * 16);
                ds[r] = __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 +
                                                                       2 * blk)) *
                        (float) sc[row * blocks * 8 + blk * 8 + sub];
                ms[r] = __half2float(
                            *reinterpret_cast<const __half *>(dmin + row * blocks * 2 + 2 * blk)) *
                        (float) mn[row * blocks * 8 + blk * 8 + sub];
            }
            float part[kQ4kRows] = {};
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                // byte i: value 2i in the low nibble, 2i+1 in the high one
                const float x0 = __half2float(xv[2 * i]);
                const float x1 = __half2float(xv[2 * i + 1]);
#pragma unroll
                for (int r = 0; r < kQ4kRows; ++r) {
                    const uint8_t b = reinterpret_cast<const uint8_t *>(&q[r])[i];
                    part[r] = fmaf(fmaf(ds[r], (float) (b & 0xF), -ms[r]), x0, part[r]);
                    part[r] = fmaf(fmaf(ds[r], (float) (b >> 4), -ms[r]), x1, part[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < kQ4kRows; ++r) {
                acc[r] += part[r];
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

__device__ const int8_t kIq4Table[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                         1,    13,   25,  38,  53,  69,  89,  113};

constexpr long long kIq4ChunkSub = 256;  // 8192 weights staged per LDS refill
constexpr long long kIq4Rows = 4;        // output rows computed per warp

__global__ void gemv_iq4_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ sc,
                                const uint8_t * __restrict__ d, const __half * __restrict__ x,
                                float * __restrict__ y, const long long rows,
                                const long long n_sub, const long long chunk_sub) {
    extern __shared__ uint8_t smem[];
    __half * xs = reinterpret_cast<__half *>(smem);  // activation chunk
    float * table = reinterpret_cast<float *>(smem + chunk_sub * 64);
    if (threadIdx.x < 16) {
        table[threadIdx.x] = (float) kIq4Table[threadIdx.x];
    }

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    // Each warp owns kIq4Rows consecutive output rows: the activation value is
    // loaded once and reused by all of them (outer product), and the four
    // accumulators give independent FMA chains.
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kIq4Rows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kIq4Rows] = {};
    for (long long base = 0; base < n_sub; base += kIq4ChunkSub) {
        const long long len = n_sub - base < kIq4ChunkSub ? n_sub - base : kIq4ChunkSub;
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
            const __half * xv = xs + (s - base) * 32;
            uint4 q[kIq4Rows];
            float dl[kIq4Rows];
#pragma unroll
            for (int r = 0; r < kIq4Rows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;  // clamped, never
                                                                             // used when masked
                q[r] = *reinterpret_cast<const uint4 *>(
                    qs + row * blocks * 128 + s * 16);
                dl[r] = __half2float(*reinterpret_cast<const __half *>(
                            d + row * blocks * 2 + 2 * blk)) *
                        (float) ((int) sc[row * blocks * 8 + blk * 8 + sub] - 32);
            }
            float part[kIq4Rows] = {};
#pragma unroll
            for (int v = 0; v < 16; ++v) {
                const float xval = __half2float(xv[v]);
#pragma unroll
                for (int r = 0; r < kIq4Rows; ++r) {
                    const uint8_t code = reinterpret_cast<const uint8_t *>(&q[r])[v] & 0xF;
                    part[r] = fmaf(table[code], xval, part[r]);
                }
            }
#pragma unroll
            for (int v = 0; v < 16; ++v) {
                const float xval = __half2float(xv[v + 16]);
#pragma unroll
                for (int r = 0; r < kIq4Rows; ++r) {
                    const uint8_t code = reinterpret_cast<const uint8_t *>(&q[r])[v] >> 4;
                    part[r] = fmaf(table[code], xval, part[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < kIq4Rows; ++r) {
                acc[r] = fmaf(dl[r], part[r], acc[r]);
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
constexpr long long kIq3Rows = 4;        // output rows computed per warp        // output rows per warp        // output rows per warp        // output rows computed per warp

__global__ void gemv_iq3_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ aux,
                                const uint8_t * __restrict__ d, const __half * __restrict__ x,
                                float * __restrict__ y, const long long rows,
                                const long long n_sub, const long long chunk_sub) {
    extern __shared__ uint8_t smem[];
    __half * xs = reinterpret_cast<__half *>(smem);  // activation chunk
    uint32_t * grid = reinterpret_cast<uint32_t *>(smem + chunk_sub * 64);
    uint8_t * stab = reinterpret_cast<uint8_t *>(grid + 256);
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        grid[i] = omph::quant::kIq3xxsGrid[i];
    }
    if (threadIdx.x < 128) {
        stab[threadIdx.x] = omph::quant::kKsignsIq2xs[threadIdx.x];
    }

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kIq3Rows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kIq3Rows] = {};
    for (long long base = 0; base < n_sub; base += kIq3ChunkSub) {
        const long long len = n_sub - base < kIq3ChunkSub ? n_sub - base : kIq3ChunkSub;
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[i] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const __half * xv = xs + (s - base) * 32;
            uint2 q[kIq3Rows];
            uint32_t a[kIq3Rows];
            float db[kIq3Rows];
#pragma unroll
            for (int r = 0; r < kIq3Rows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                q[r] = *reinterpret_cast<const uint2 *>(qs + row * blocks * 64 + s * 8);
                a[r] = *reinterpret_cast<const uint32_t *>(aux + row * blocks * 32 + s * 4);
                db[r] = __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 +
                                                                       (s / 8) * 2)) *
                        (0.5f + (float) (a[r] >> 28)) * 0.5f;
            }
            float part[kIq3Rows] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
#pragma unroll
                for (int r = 0; r < kIq3Rows; ++r) {
                    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(&q[r]);
                    const uint8_t sg = stab[(a[r] >> (7 * l)) & 127];
                    const uint32_t g1 = grid[bytes[2 * l]];
                    const uint32_t g2 = grid[bytes[2 * l + 1]];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float g1v = (float) ((g1 >> (8 * j)) & 0xFF);
                        float g2v = (float) ((g2 >> (8 * j)) & 0xFF);
                        if (sg & (1u << j)) {
                            g1v = -g1v;
                        }
                        if (sg & (1u << (4 + j))) {
                            g2v = -g2v;
                        }
                        part[r] = fmaf(g1v, __half2float(xv[8 * l + j]), part[r]);
                        part[r] = fmaf(g2v, __half2float(xv[8 * l + 4 + j]), part[r]);
                    }
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


// IQ3_S: 110-byte blocks. Per 32-weight sub-block: eight 9-bit grid indices
// (8 bits from qs, the 9th from one qh bit per index), four sign-mask bytes and
// a 4-bit scale shared by two sub-blocks. Grid (2 KiB) in LDS, signs are plain
// masks so no table is needed.
constexpr long long kIq3sChunkSub = 256;  // 8192 weights staged per LDS refill
constexpr long long kIq3sRows = 4;        // output rows computed per warp

__global__ void gemv_iq3s_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ qh,
                                 const uint8_t * __restrict__ signs,
                                 const uint8_t * __restrict__ scales, const uint8_t * __restrict__ d,
                                 const __half * __restrict__ x, float * __restrict__ y,
                                 const long long rows, const long long n_sub,
                                 const long long chunk_sub) {
    extern __shared__ uint8_t smem[];
    __half * xs = reinterpret_cast<__half *>(smem);
    uint32_t * grid = reinterpret_cast<uint32_t *>(smem + chunk_sub * 64);
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
            const __half * xv = xs + (s - base) * 32;
            uint2 q[kIq3sRows];
            uint8_t qhb[kIq3sRows];
            const uint8_t * sgp[kIq3sRows];
            float dbr[kIq3sRows];
#pragma unroll
            for (int r = 0; r < kIq3sRows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                q[r] = *reinterpret_cast<const uint2 *>(qs + row * blocks * 64 + s * 8);
                qhb[r] = qh[row * blocks * 8 + s];
                sgp[r] = signs + row * blocks * 32 + s * 4;
                const uint8_t sc = scales[row * blocks * 4 + s / 2];
                const float dv = __half2float(
                    *reinterpret_cast<const __half *>(d + row * blocks * 2 + (s / 8) * 2));
                dbr[r] = dv * (1.0f + 2.0f * (float) ((s & 1) ? (sc >> 4) : (sc & 0xF)));
            }
            float part[kIq3sRows] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
#pragma unroll
                for (int r = 0; r < kIq3sRows; ++r) {
                    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(&q[r]);
                    const uint32_t hb = qhb[r];
                    const uint32_t i1 =
                        (uint32_t) bytes[2 * l] | (((hb >> (2 * l)) & 1u) << 8);
                    const uint32_t i2 =
                        (uint32_t) bytes[2 * l + 1] | (((hb >> (2 * l + 1)) & 1u) << 8);
                    const uint32_t g1 = grid[i1];
                    const uint32_t g2 = grid[i2];
                    const uint8_t sm = sgp[r][l];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float v1 = (float) ((g1 >> (8 * j)) & 0xFF);
                        float v2 = (float) ((g2 >> (8 * j)) & 0xFF);
                        if (sm & (1u << j)) {
                            v1 = -v1;
                        }
                        if (sm & (1u << (4 + j))) {
                            v2 = -v2;
                        }
                        part[r] = fmaf(v1, __half2float(xv[8 * l + j]), part[r]);
                        part[r] = fmaf(v2, __half2float(xv[8 * l + 4 + j]), part[r]);
                    }
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



// Q2_K: 84-byte blocks, sixteen 16-weight sub-blocks each with a 4-bit scale and
// a 4-bit min. A lane iteration covers one 128-weight group: the 32 quant bytes
// are read once (two uint4, extracted through explicit words so nothing depends
// on how the compiler treats an array) and eight sub-block scales.
constexpr long long kQ2kChunkSub = 256;
constexpr long long kQ2kRows = 4;

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
        const long long g_first = base / 4;
        const long long g_last = (base + len) / 4;
        for (long long g = g_first + lane; g < g_last; g += 32) {
            const long long blk = g >> 1;
            const int half = (int) (g & 1);
            const long long xoff = (g * 4 - base) * 32;
            float part[kQ2kRows] = {};
#pragma unroll
            for (int r = 0; r < kQ2kRows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                const uint8_t * qrow = qs + row * blocks * 64 + g * 32;
                const uint8_t * srow = sc + row * blocks * 16 + blk * 16 + half * 8;
                const float dv =
                    __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                const float mv = __half2float(
                    *reinterpret_cast<const __half *>(dmin + row * blocks * 2 + blk * 2));
                const uint4 qa = *reinterpret_cast<const uint4 *>(qrow);
                const uint4 qb = *reinterpret_cast<const uint4 *>(qrow + 16);
                const uint32_t wa[4] = {qa.x, qa.y, qa.z, qa.w};
                const uint32_t wb[4] = {qb.x, qb.y, qb.z, qb.w};
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const int shift = (u >> 1) * 2;
                    const uint8_t scb = srow[u];
                    const float dl = dv * (float) (scb & 0xF);
                    const float ml = mv * (float) (scb >> 4);
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const uint32_t word = ((u & 1) ? wb : wa)[i >> 2];
                        const int code = (int) ((word >> (8 * (i & 3))) >> shift) & 3;
                        const float xa = __half2float(xs[xoff + u * 16 + i]);
                        part[r] = fmaf(fmaf(dl, (float) code, -ml), xa, part[r]);
                    }
                }
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
// four quarters of a group are contiguous weight runs (l, l+32, l+64, l+96), so
// each activation is loaded once and the scale of each quarter folds into an
// 8-entry per-group table (d * sc[i]).
constexpr long long kQ6kChunkSub = 256;
constexpr long long kQ6kRows = 4;

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
        const long long g_first = base / 4;
        const long long g_last = (base + len) / 4;
        for (long long g = g_first + lane; g < g_last; g += 32) {
            const long long blk = g >> 1;
            const int half = (int) (g & 1);
            const __half * xv = xs + (g * 4 - base) * 32;
            float part[kQ6kRows] = {};
#pragma unroll
            for (int r = 0; r < kQ6kRows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                const uint8_t * pql = ql + row * blocks * 128 + blk * 128 + half * 64;
                const uint8_t * pqh = qh + row * blocks * 64 + blk * 64 + half * 32;
                const int8_t * psc =
                    reinterpret_cast<const int8_t *>(sc + row * blocks * 16 + blk * 16 + half * 8);
                const float dv = __half2float(
                    *reinterpret_cast<const __half *>(d + row * blocks * 2 + blk * 2));
                float dsc[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dsc[i] = dv * (float) psc[i];
                }
                const uint32_t qlw[16] = {
                    reinterpret_cast<const uint32_t *>(pql)[0],
                    reinterpret_cast<const uint32_t *>(pql)[1],
                    reinterpret_cast<const uint32_t *>(pql)[2],
                    reinterpret_cast<const uint32_t *>(pql)[3],
                    reinterpret_cast<const uint32_t *>(pql)[4],
                    reinterpret_cast<const uint32_t *>(pql)[5],
                    reinterpret_cast<const uint32_t *>(pql)[6],
                    reinterpret_cast<const uint32_t *>(pql)[7],
                    reinterpret_cast<const uint32_t *>(pql)[8],
                    reinterpret_cast<const uint32_t *>(pql)[9],
                    reinterpret_cast<const uint32_t *>(pql)[10],
                    reinterpret_cast<const uint32_t *>(pql)[11],
                    reinterpret_cast<const uint32_t *>(pql)[12],
                    reinterpret_cast<const uint32_t *>(pql)[13],
                    reinterpret_cast<const uint32_t *>(pql)[14],
                    reinterpret_cast<const uint32_t *>(pql)[15]};
                const uint32_t qhw[8] = {
                    reinterpret_cast<const uint32_t *>(pqh)[0],
                    reinterpret_cast<const uint32_t *>(pqh)[1],
                    reinterpret_cast<const uint32_t *>(pqh)[2],
                    reinterpret_cast<const uint32_t *>(pqh)[3],
                    reinterpret_cast<const uint32_t *>(pqh)[4],
                    reinterpret_cast<const uint32_t *>(pqh)[5],
                    reinterpret_cast<const uint32_t *>(pqh)[6],
                    reinterpret_cast<const uint32_t *>(pqh)[7]};
                float sub[8] = {};
#pragma unroll
                for (int l = 0; l < 32; ++l) {
                    const int wi = l >> 2;
                    const int bi = 8 * (l & 3);
                    const uint32_t ql_lo = (qlw[wi] >> bi) & 0xFF;
                    const uint32_t ql_hi = (qlw[wi + 8] >> bi) & 0xFF;
                    const uint32_t qh_b = (qhw[wi] >> bi) & 0xFF;
                    const int is = l >> 4;
                    const float x0 = __half2float(xv[l]);
                    const float x1 = __half2float(xv[32 + l]);
                    const float x2 = __half2float(xv[64 + l]);
                    const float x3 = __half2float(xv[96 + l]);
                    sub[0 + is] = fmaf(
                        (float) (int) ((ql_lo & 0xF) | ((qh_b & 3) << 4)) - 32.0f, x0, sub[0 + is]);
                    sub[2 + is] = fmaf(
                        (float) (int) ((ql_hi & 0xF) | (((qh_b >> 2) & 3) << 4)) - 32.0f, x1,
                        sub[2 + is]);
                    sub[4 + is] = fmaf(
                        (float) (int) ((ql_lo >> 4) | (((qh_b >> 4) & 3) << 4)) - 32.0f, x2,
                        sub[4 + is]);
                    sub[6 + is] = fmaf(
                        (float) (int) ((ql_hi >> 4) | (((qh_b >> 6) & 3) << 4)) - 32.0f, x3,
                        sub[6 + is]);
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    part[r] = fmaf(dsc[i], sub[i], part[r]);
                }
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
    uint64_t * grid = reinterpret_cast<uint64_t *>(smem + chunk_sub * 64);
    uint8_t * stab = reinterpret_cast<uint8_t *>(grid + 512);
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        grid[i] = omph::quant::kIq2xsGrid[i];
    }
    if (threadIdx.x < 128) {
        stab[threadIdx.x] = omph::quant::kKsignsIq2xs[threadIdx.x];
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
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const __half * xv = xs + (s - base) * 32;
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
            float sub[kIq2Rows][2] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
#pragma unroll
                for (int r = 0; r < kIq2Rows; ++r) {
                    const uint32_t e = ent[r][l >> 1] >> (16 * (l & 1));
                    const uint8_t sg = stab[(e >> 9) & 127];
                    const uint64_t g = grid[e & 511];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float v = (float) ((g >> (8 * j)) & 0xFF);
                        if (sg & (1u << j)) {
                            v = -v;
                        }
                        sub[r][l >> 1] =
                            fmaf(v, __half2float(xv[8 * l + j]), sub[r][l >> 1]);
                    }
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
    uint8_t * stab = reinterpret_cast<uint8_t *>(grid + 256);
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        grid[i] = omph::quant::kIq2xxsGrid[i];
    }
    if (threadIdx.x < 128) {
        stab[threadIdx.x] = omph::quant::kKsignsIq2xs[threadIdx.x];
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
            float part[kIq2Rows] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
#pragma unroll
                for (int r = 0; r < kIq2Rows; ++r) {
                    const uint32_t idx_byte = (aux0[r] >> (8 * l)) & 0xFF;
                    const uint8_t sg = stab[(aux1[r] >> (7 * l)) & 127];
                    const uint64_t g = grid[idx_byte];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float v = (float) ((g >> (8 * j)) & 0xFF);
                        if (sg & (1u << j)) {
                            v = -v;
                        }
                        part[r] = fmaf(v, __half2float(xv[8 * l + j]), part[r]);
                    }
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
    uint64_t * grid = reinterpret_cast<uint64_t *>(smem + chunk_sub * 64);
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
            xs[i] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const __half * xv = xs + (s - base) * 32;
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
            float sub[kIq2sRows][2] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
#pragma unroll
                for (int r = 0; r < kIq2sRows; ++r) {
                    const uint32_t lo = (qsw[r] >> (8 * l)) & 0xFF;
                    const uint32_t hi = ((qhb[r] >> (2 * l)) & 3) << 8;
                    const uint32_t sg = (sgw[r] >> (8 * l)) & 0xFF;
                    const uint64_t g = grid[lo | hi];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float v = (float) ((g >> (8 * j)) & 0xFF);
                        if (sg & (1u << j)) {
                            v = -v;
                        }
                        sub[r][l >> 1] = fmaf(v, __half2float(xv[8 * l + j]), sub[r][l >> 1]);
                    }
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

} // namespace

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
    const dim3 grid((unsigned) ((rows + 8 * kQ4kRows - 1) / (8 * kQ4kRows)));
    gemv_q4k_kernel<<<grid, 256, smem, stream>>>(
        base + l.qs_off, base + l.sc_off, base + l.mn_off, base + l.d_off, base + l.dmin_off,
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
    const size_t smem = (size_t) chunk * 64 + 1024 * 8;
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
    const size_t smem = (size_t) chunk * 64 + 512 * 8 + 128;
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
    const size_t smem = (size_t) chunk * 64 + 256 * 8 + 128;
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
    const size_t smem = (size_t) chunk * 64;
    const auto * base = static_cast<const uint8_t *>(packed);
    const dim3 grid((unsigned) ((rows + 8 * kQ6kRows - 1) / (8 * kQ6kRows)));
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
    const size_t smem = (size_t) chunk * 64;
    const auto * base = static_cast<const uint8_t *>(packed);
    const dim3 grid((unsigned) ((rows + 8 * kQ2kRows - 1) / (8 * kQ2kRows)));
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
    const dim3 grid((unsigned) ((rows + 8 * kIq3sRows - 1) / (8 * kIq3sRows)));
    gemv_iq3s_kernel<<<grid, 256, smem, stream>>>(base + l.qs_off, base + l.qh_off,
                                                  base + l.signs_off, base + l.scales_off,
                                                  base + l.d_off,
                                                  static_cast<const __half *>(x_f16), y, rows,
                                                  n_sub, chunk);
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
    const size_t smem = (size_t) chunk * 64 + 256 * 4 + 128;
    const auto * base = static_cast<const uint8_t *>(packed);
    const dim3 grid((unsigned) ((rows + 8 * kIq3Rows - 1) / (8 * kIq3Rows)));
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
    const size_t smem = (size_t) chunk * 64 + 16 * sizeof(float);
    if (smem > 64 * 1024) {
        return false;
    }
    const auto * base = static_cast<const uint8_t *>(packed);
    const dim3 grid((unsigned) ((rows + 8 * kIq4Rows - 1) / (8 * kIq4Rows)));
    gemv_iq4_kernel<<<grid, 256, smem, stream>>>(base + l.qs_off, base + l.sc_off, base + l.d_off,
                                                 static_cast<const __half *>(x_f16), y, rows,
                                                 n_sub, chunk);
    return hipGetLastError() == hipSuccess;
}

GemvOccupancy gemv_occupancy() {
    GemvOccupancy o;
    const void * kernels[4] = {(const void *) gemv_q4k_kernel, (const void *) gemv_iq4_kernel,
                               (const void *) gemv_iq3_kernel, (const void *) gemv_iq3s_kernel};
    o.smem_q4k = 0;
    o.smem_iq4 = (size_t) kIq4ChunkSub * 64 + 16 * sizeof(float);
    o.smem_iq3 = (size_t) kIq3ChunkSub * 64 + 256 * 4 + 128;
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

} // namespace omph::kernels
