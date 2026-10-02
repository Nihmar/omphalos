// Where the IQ3_S tile GEMM's time goes (#178, prefill at ~53 of ~100 TFLOPS):
// the engine's kernel (gemm.hip, gemm_iq3s_tiles_kernel) with each part
// switchable, on random tiles shaped like ffn_up (17408 x 5120):
//   W   the weight loads from global memory (else words made from the indices)
//   D   the decode into A operands (else the raw words as A)
//   S   the x staging into LDS and its barriers (else the LDS buffer is filled once)
//   B   the B operands read from LDS (else held in registers)
// All four off is the WMMA chain alone: the ceiling this grid can reach.
//   hipcc -O3 --offload-arch=gfx1200 -std=c++20 -I engine/src -o /var/tmp/tile_ablate bench/gemm/tile_ablate.hip
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/gemm.hh"
#include "kernels/quant_decode.hh"

namespace qd = omph::kernels::qd;

namespace {

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr int kW = 1, kD = 2, kS = 4, kB = 8;
constexpr int kG = 16;
constexpr int kR = 32;
constexpr int kP = 64;  // the B reads pipelined with the WMMAs (sched_group_barrier)  // x through registers into one LDS buffer (half the LDS: 2 WGs per WGP)  // grid: row tiles fastest (the WGs in flight share x in L2)
constexpr int kXLd = 40;
constexpr long long kTileBytes = 1760;

void check(hipError_t e, const char * what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

template <int M, int kTok>
__global__ __launch_bounds__(256) void k_gemm(const uint8_t * __restrict__ wt, const _Float16 * __restrict__ x,
                                             float * __restrict__ y, const int n_out, const long long blocks,
                                             const int tokens) {
    __shared__ uint2 grid[512];
    __shared__ _Float16 xs[(M & kR) ? 1 : 2][kTok * kXLd];
    const int tid = (int) threadIdx.x;
    for (int i = tid; i < 512; i += blockDim.x) {
        const uint32_t g = omph::quant::kIq3sGrid[i];
        grid[i] = make_uint2(qd::grid_pair_to_half2<0x04010400u>(g), qd::grid_pair_to_half2<0x04030402u>(g));
    }
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int h = lane >> 4;
    const int r = lane & 15;
    const long long tile = (long long) ((M & kG) ? blockIdx.x : blockIdx.y) * 8 + wv;
    const long long n0 = (long long) ((M & kG) ? blockIdx.y : blockIdx.x) * kTok;
    const long long k = blocks * 256;
    const long long tx = n0 + tid < tokens ? n0 + tid : 0;
    const uint4 * xrow = reinterpret_cast<const uint4 *>(x + tx * k);
    const auto stage = [&](const long long s, _Float16 * buf) {
        if (kTok < 256 && tid >= kTok) {
            return;
        }
        const uint4 * src = xrow + s * 4;
        uint4 * dst = reinterpret_cast<uint4 *>(buf + tid * kXLd);
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
    };
    half8 breg;
    for (int i = 0; i < 8; ++i) {
        breg[i] = (_Float16) (0.001f * (float) (lane + i));
    }
    constexpr int TJ = kTok / 16;
    float8 acc[TJ];
#pragma unroll
    for (int j = 0; j < TJ; ++j) {
        acc[j] = float8{};
    }
    stage(0, xs[0]);
    if constexpr (!(M & kS) && !(M & kR)) {
        stage(0, xs[1]);
    }
    __syncthreads();
    int buf = 0;
    for (long long b = 0; b < blocks; ++b) {
        uint4 q0, q1, s4;
        uint32_t qhw, scw;
        float dv;
        if constexpr ((M & kW) != 0) {
            const uint8_t * o = wt + (tile * blocks + b) * kTileBytes;
            q0 = *reinterpret_cast<const uint4 *>(o + lane * 32);
            q1 = *reinterpret_cast<const uint4 *>(o + lane * 32 + 16);
            s4 = *reinterpret_cast<const uint4 *>(o + 1024 + lane * 16);
            qhw = *reinterpret_cast<const uint32_t *>(o + 1536 + lane * 4);
            scw = *reinterpret_cast<const uint32_t *>(o + 1664 + r * 4);
            dv = qd::load_half(o + 1728 + r * 2);
        } else {
            const uint32_t z = (uint32_t) (b * 2654435761u) ^ (uint32_t) (tile * 40503u + lane);
            q0 = make_uint4(z, z * 3u, z * 5u, z * 7u);
            q1 = make_uint4(z * 9u, z * 11u, z * 13u, z * 15u);
            s4 = make_uint4(z ^ 1u, z ^ 2u, z ^ 3u, z ^ 4u);
            qhw = z >> 3;
            scw = z >> 5;
            dv = 0.001f;
        }
        const uint32_t qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const uint32_t sw[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll 1
        for (int sb = 0; sb < 8; ++sb) {
            const long long s = b * 8 + sb;
            uint4 nx[4];
            if constexpr ((M & kS) != 0 && (M & kR) != 0) {
                if (s + 1 < blocks * 8 && (kTok == 256 || tid < kTok)) {
                    const uint4 * src = xrow + (s + 1) * 4;
                    nx[0] = src[0];
                    nx[1] = src[1];
                    nx[2] = src[2];
                    nx[3] = src[3];
                }
            } else if constexpr ((M & kS) != 0) {
                if (s + 1 < blocks * 8) {
                    stage(s + 1, xs[buf ^ 1]);
                }
            }
            half8 a[2];
            if constexpr ((M & kD) != 0) {
                const uint32_t q = qw[sb];
                const uint32_t qhn = (qhw >> (4 * sb)) & 0xFu;
                const uint32_t s16 = (sw[sb >> 1] >> (16 * (sb & 1))) & 0xFFFFu;
                const uint32_t sg = h == 0 ? (((s16 & 0xFFu) << 8) | ((s16 >> 8) << 24))
                                           : ((s16 & 0xFFu) | ((s16 >> 8) << 16));
                const uint32_t sc = (scw >> (8 * (sb >> 1))) & 0xFFu;
                const float dbr = dv * (1.0f + 2.0f * (float) ((sb & 1) ? (sc >> 4) : (sc & 0xF)));
                const qd::half2_v d2 = {(_Float16) dbr, (_Float16) dbr};
#pragma unroll
                for (int gl = 0; gl < 2; ++gl) {
                    const int g = 2 * h + gl;
                    const uint32_t word = (q >> (16 * gl)) & 0xFFFFu;
                    const uint32_t i1 = (word & 0xFFu) | (((qhn >> (2 * gl)) & 1u) << 8);
                    const uint32_t i2 = (word >> 8) | (((qhn >> (2 * gl + 1)) & 1u) << 8);
                    const uint2 g1 = grid[i1];
                    const uint2 g2 = grid[i2];
                    constexpr uint32_t kSign2 = 0x80008000u;
                    const uint32_t wv4[4] = {((sg << (4 * g)) & kSign2) | g1.x, ((sg << (4 * g + 1)) & kSign2) | g1.y,
                                             ((sg << (4 * g + 2)) & kSign2) | g2.x,
                                             ((sg << (4 * g + 3)) & kSign2) | g2.y};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const qd::half2_v p = __builtin_bit_cast(qd::half2_v, wv4[i]) * d2;
                        a[gl][2 * i] = p[0];
                        a[gl][2 * i + 1] = p[1];
                    }
                }
            } else {
                // the raw words as A, the exponent bits cleared (no inf / NaN)
                const uint32_t q = qw[sb] & 0x3BFF3BFFu;
                const uint32_t e = sw[sb >> 1] & 0x3BFF3BFFu;
                const uint4 u0 = make_uint4(q, e, q ^ 0x00010001u, e ^ 0x00010001u);
                const uint4 u1 = make_uint4(e, q, e ^ 0x00020002u, q ^ 0x00020002u);
                a[0] = __builtin_bit_cast(half8, u0);
                a[1] = __builtin_bit_cast(half8, u1);
            }
            const _Float16 * cur = xs[(M & kS) && !(M & kR) ? buf : 0];
#pragma unroll
            for (int j = 0; j < TJ; ++j) {
#pragma unroll
                for (int gl = 0; gl < 2; ++gl) {
                    half8 bv;
                    if constexpr ((M & kB) != 0) {
                        bv = *reinterpret_cast<const half8 *>(cur + (j * 16 + (lane & 15)) * kXLd + h * 16 + gl * 8);
                    } else {
                        bv = breg;
                    }
                    acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[gl], bv, acc[j]);
                }
            }
            if constexpr ((M & kP) != 0) {
                // 8 DS reads ahead (the 4 grid lookups + 4 B), then a WMMA per B read
                __builtin_amdgcn_sched_group_barrier(0x100, 8, 0);
#pragma unroll
                for (int i = 0; i < 2 * TJ - 4; ++i) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 4, 0);
            }
            if constexpr ((M & kS) != 0 && (M & kR) != 0) {
                __syncthreads();  // every wave is done with this sub-block's x
                if (s + 1 < blocks * 8 && (kTok == 256 || tid < kTok)) {
                    uint4 * dst = reinterpret_cast<uint4 *>(xs[0] + tid * kXLd);
                    dst[0] = nx[0];
                    dst[1] = nx[1];
                    dst[2] = nx[2];
                    dst[3] = nx[3];
                }
                __syncthreads();
            } else if constexpr ((M & kS) != 0) {
                __syncthreads();
                buf ^= 1;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TJ; ++j) {
        const long long t = n0 + j * 16 + (lane & 15);
        if (t < tokens) {
            float * dst = y + t * (long long) n_out + tile * 16 + h * 8;
            reinterpret_cast<float4 *>(dst)[0] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
            reinterpret_cast<float4 *>(dst)[1] = make_float4(acc[j][4], acc[j][5], acc[j][6], acc[j][7]);
        }
    }
}


// 4 x 4 register blocking: a wave computes 4 row tiles x 4 token tiles (wave w:
// row group w & 1, token group w >> 1). Wave w still decodes row tile w, but
// writes its A operands to LDS for the two waves... four waves of its row group;
// per 16 WMMAs a wave reads 8 operands from LDS (4 A + 4 B) instead of 16 B.
// Loop: barrier; WMMAs on buffer s & 1 (x(s + 1) loaded into registers first);
// then A(s + 1) decoded and x(s + 1) written into buffer (s + 1) & 1, whose last
// reader finished before this barrier.
__global__ __launch_bounds__(256) void k_gemm44(const uint8_t * __restrict__ wt, const _Float16 * __restrict__ x,
                                               float * __restrict__ y, const int n_out, const long long blocks,
                                               const int tokens) {
    constexpr int kTk = 256;
    __shared__ uint2 grid[512];
    __shared__ _Float16 xs[2][kTk * kXLd];
    __shared__ half8 as[2][8 * 2 * 32];
    const int tid = (int) threadIdx.x;
    for (int i = tid; i < 512; i += blockDim.x) {
        const uint32_t g = omph::quant::kIq3sGrid[i];
        grid[i] = make_uint2(qd::grid_pair_to_half2<0x04010400u>(g), qd::grid_pair_to_half2<0x04030402u>(g));
    }
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int h = lane >> 4;
    const int r = lane & 15;
    const int rg = wv & 1;
    const int tg = wv >> 1;
    const long long tile0 = (long long) blockIdx.y * 8;
    const long long n0 = (long long) blockIdx.x * kTk;
    const long long k = blocks * 256;
    const long long tx = n0 + tid < tokens ? n0 + tid : 0;
    const uint4 * xrow = reinterpret_cast<const uint4 *>(x + tx * k);
    const long long nsub = blocks * 8;
    uint4 q0, q1, s4;
    uint32_t qhw = 0, scw = 0;
    float dv = 0.0f;
    const auto load_w = [&](const long long b) {
        const uint8_t * o = wt + ((tile0 + wv) * blocks + b) * kTileBytes;
        q0 = *reinterpret_cast<const uint4 *>(o + lane * 32);
        q1 = *reinterpret_cast<const uint4 *>(o + lane * 32 + 16);
        s4 = *reinterpret_cast<const uint4 *>(o + 1024 + lane * 16);
        qhw = *reinterpret_cast<const uint32_t *>(o + 1536 + lane * 4);
        scw = *reinterpret_cast<const uint32_t *>(o + 1664 + r * 4);
        dv = qd::load_half(o + 1728 + r * 2);
    };
    const auto decode = [&](const int sb, half8 * dst) {
        const uint32_t qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const uint32_t sw[4] = {s4.x, s4.y, s4.z, s4.w};
        const uint32_t q = qw[sb];
        const uint32_t qhn = (qhw >> (4 * sb)) & 0xFu;
        const uint32_t s16 = (sw[sb >> 1] >> (16 * (sb & 1))) & 0xFFFFu;
        const uint32_t sg = h == 0 ? (((s16 & 0xFFu) << 8) | ((s16 >> 8) << 24)) : ((s16 & 0xFFu) | ((s16 >> 8) << 16));
        const uint32_t sc = (scw >> (8 * (sb >> 1))) & 0xFFu;
        const float dbr = dv * (1.0f + 2.0f * (float) ((sb & 1) ? (sc >> 4) : (sc & 0xF)));
        const qd::half2_v d2 = {(_Float16) dbr, (_Float16) dbr};
#pragma unroll
        for (int gl = 0; gl < 2; ++gl) {
            const int g = 2 * h + gl;
            const uint32_t word = (q >> (16 * gl)) & 0xFFFFu;
            const uint32_t i1 = (word & 0xFFu) | (((qhn >> (2 * gl)) & 1u) << 8);
            const uint32_t i2 = (word >> 8) | (((qhn >> (2 * gl + 1)) & 1u) << 8);
            const uint2 g1 = grid[i1];
            const uint2 g2 = grid[i2];
            constexpr uint32_t kSign2 = 0x80008000u;
            const uint32_t wv4[4] = {((sg << (4 * g)) & kSign2) | g1.x, ((sg << (4 * g + 1)) & kSign2) | g1.y,
                                     ((sg << (4 * g + 2)) & kSign2) | g2.x, ((sg << (4 * g + 3)) & kSign2) | g2.y};
            half8 a;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const qd::half2_v p = __builtin_bit_cast(qd::half2_v, wv4[i]) * d2;
                a[2 * i] = p[0];
                a[2 * i + 1] = p[1];
            }
            dst[(wv * 2 + gl) * 32 + lane] = a;
        }
    };
    float8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = float8{};
        }
    }
    {
        const uint4 * src = xrow;
        uint4 * dst = reinterpret_cast<uint4 *>(xs[0] + tid * kXLd);
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
    }
    load_w(0);
    __syncthreads();  // the grid table
    decode(0, as[0]);
#pragma unroll 1
    for (long long s = 0; s < nsub; ++s) {
        const int cb = (int) (s & 1);
        __syncthreads();
        uint4 nx[4];
        const bool more = s + 1 < nsub;
        if (more) {
            const uint4 * src = xrow + (s + 1) * 4;
            nx[0] = src[0];
            nx[1] = src[1];
            nx[2] = src[2];
            nx[3] = src[3];
        }
#pragma unroll
        for (int gl = 0; gl < 2; ++gl) {
            half8 a[4], bv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = as[cb][((rg * 4 + i) * 2 + gl) * 32 + lane];
                bv[i] = *reinterpret_cast<const half8 *>(xs[cb] + ((tg * 4 + i) * 16 + (lane & 15)) * kXLd + h * 16 + gl * 8);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], bv[j], acc[i][j]);
                }
            }
        }
        if (more) {
            const int sb = (int) ((s + 1) & 7);
            if (sb == 0) {
                load_w((s + 1) >> 3);
            }
            decode(sb, as[cb ^ 1]);
            uint4 * dst = reinterpret_cast<uint4 *>(xs[cb ^ 1] + tid * kXLd);
            dst[0] = nx[0];
            dst[1] = nx[1];
            dst[2] = nx[2];
            dst[3] = nx[3];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long long t = n0 + (tg * 4 + j) * 16 + (lane & 15);
            if (t < tokens) {
                float * dst = y + t * (long long) n_out + (tile0 + rg * 4 + i) * 16 + h * 8;
                reinterpret_cast<float4 *>(dst)[0] = make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
                reinterpret_cast<float4 *>(dst)[1] = make_float4(acc[i][j][4], acc[i][j][5], acc[i][j][6], acc[i][j][7]);
            }
        }
    }
}


// 4 x 4 as above, in 16-k steps (one WMMA k each): x staged 16 halves per token
// unpadded (a lane's 16-byte read is contiguous across the wave: no bank
// conflicts), A one operand per row tile. 28 KB of LDS: 2 workgroups per WGP,
// so one's decode overlaps the other's WMMAs. The next block's weight words are
// prefetched a step ahead.
template <int F>  // ablation: 1 no barrier, 2 no decode, 4 no x loads, 8 no LDS reads
__global__ __launch_bounds__(256) void k_gemm44s(const uint8_t * __restrict__ wt, const _Float16 * __restrict__ x,
                                                float * __restrict__ y, const int n_out, const long long blocks,
                                                const int tokens) {
    constexpr int kTk = 256;
    __shared__ uint2 grid[512];
    __shared__ half8 xs[2][kTk * 2];
    __shared__ half8 as[2][8 * 32];
    const int tid = (int) threadIdx.x;
    for (int i = tid; i < 512; i += blockDim.x) {
        const uint32_t g = omph::quant::kIq3sGrid[i];
        grid[i] = make_uint2(qd::grid_pair_to_half2<0x04010400u>(g), qd::grid_pair_to_half2<0x04030402u>(g));
    }
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int h = lane >> 4;
    const int r = lane & 15;
    const int rg = wv & 1;
    const int tg = wv >> 1;
    const long long tile0 = (long long) blockIdx.y * 8;
    const long long n0 = (long long) blockIdx.x * kTk;
    const long long k = blocks * 256;
    const long long tx = n0 + tid < tokens ? n0 + tid : 0;
    const uint4 * xrow = reinterpret_cast<const uint4 *>(x + tx * k);
    const long long nsteps = blocks * 16;
    struct W {
        uint4 q0, q1, s4;
        uint32_t qhw, scw;
        float dv;
    };
    const auto load_w = [&](const long long b) {
        const uint8_t * o = wt + ((tile0 + wv) * blocks + b) * kTileBytes;
        W w;
        w.q0 = *reinterpret_cast<const uint4 *>(o + lane * 32);
        w.q1 = *reinterpret_cast<const uint4 *>(o + lane * 32 + 16);
        w.s4 = *reinterpret_cast<const uint4 *>(o + 1024 + lane * 16);
        w.qhw = *reinterpret_cast<const uint32_t *>(o + 1536 + lane * 4);
        w.scw = *reinterpret_cast<const uint32_t *>(o + 1664 + r * 4);
        w.dv = qd::load_half(o + 1728 + r * 2);
        return w;
    };
    const auto decode = [&](const W & w, const int sb, const int gl) {
        const uint32_t qw[8] = {w.q0.x, w.q0.y, w.q0.z, w.q0.w, w.q1.x, w.q1.y, w.q1.z, w.q1.w};
        const uint32_t sw[4] = {w.s4.x, w.s4.y, w.s4.z, w.s4.w};
        const uint32_t q = qw[sb];
        const uint32_t qhn = (w.qhw >> (4 * sb)) & 0xFu;
        const uint32_t s16 = (sw[sb >> 1] >> (16 * (sb & 1))) & 0xFFFFu;
        const uint32_t sg = h == 0 ? (((s16 & 0xFFu) << 8) | ((s16 >> 8) << 24)) : ((s16 & 0xFFu) | ((s16 >> 8) << 16));
        const uint32_t sc = (w.scw >> (8 * (sb >> 1))) & 0xFFu;
        const float dbr = w.dv * (1.0f + 2.0f * (float) ((sb & 1) ? (sc >> 4) : (sc & 0xF)));
        const qd::half2_v d2 = {(_Float16) dbr, (_Float16) dbr};
        const int g = 2 * h + gl;
        const uint32_t word = (q >> (16 * gl)) & 0xFFFFu;
        const uint32_t i1 = (word & 0xFFu) | (((qhn >> (2 * gl)) & 1u) << 8);
        const uint32_t i2 = (word >> 8) | (((qhn >> (2 * gl + 1)) & 1u) << 8);
        const uint2 g1 = grid[i1];
        const uint2 g2 = grid[i2];
        constexpr uint32_t kSign2 = 0x80008000u;
        const uint32_t wv4[4] = {((sg << (4 * g)) & kSign2) | g1.x, ((sg << (4 * g + 1)) & kSign2) | g1.y,
                                 ((sg << (4 * g + 2)) & kSign2) | g2.x, ((sg << (4 * g + 3)) & kSign2) | g2.y};
        half8 a;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const qd::half2_v p = __builtin_bit_cast(qd::half2_v, wv4[i]) * d2;
            a[2 * i] = p[0];
            a[2 * i + 1] = p[1];
        }
        return a;
    };
    // step st: sub-block st >> 1, k half gl = st & 1: x halves 32 sb + 16 h' ... the
    // engine's permutation: lane half h takes weights 16 h + 8 gl .. + 7 of the sub-block
    const auto xoff = [&](const long long st) { return (st >> 1) * 4 + (st & 1); };  // in uint4: + 2 h
    float8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = float8{};
        }
    }
    // x of a step for token tid: halves 16 h + 8 gl .. + 7 for h = 0, 1, i.e. uint4s
    // xoff(st) and xoff(st) + 2 of the token's row
    xs[0][tid * 2] = __builtin_bit_cast(half8, xrow[xoff(0)]);
    xs[0][tid * 2 + 1] = __builtin_bit_cast(half8, xrow[xoff(0) + 2]);
    W cw = load_w(0);
    __syncthreads();  // the grid table
    as[0][wv * 32 + lane] = decode(cw, 0, 0);
#pragma unroll 1
    for (long long st = 0; st < nsteps; ++st) {
        const int cb = (int) (st & 1);
        if constexpr (!(F & 1)) {
            __syncthreads();
        }
        const bool more = st + 1 < nsteps;
        const long long ns = more ? st + 1 : st;  // the last step refills a buffer nobody reads
        uint4 nx0, nx1;
        W nw;
        {
            if constexpr (!(F & 4)) {
                nx0 = xrow[xoff(ns)];
                nx1 = xrow[xoff(ns) + 2];
            } else {
                nx0 = make_uint4((uint32_t) ns, 1u, 2u, 3u);
                nx1 = nx0;
            }
        }
        const bool next_block = more && (ns & 15) == 0;
        if (next_block) {
            nw = load_w(ns >> 4);
        }
        half8 a[4], bv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (!(F & 8)) {
                a[i] = as[cb][(rg * 4 + i) * 32 + lane];
                bv[i] = xs[cb][((tg * 4 + i) * 16 + (lane & 15)) * 2 + h];
            } else {
                a[i] = __builtin_bit_cast(half8, make_uint4((uint32_t) st & 0x3BFF3BFFu, (uint32_t) i, 7u, 9u));
                bv[i] = a[i];
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], bv[j], acc[i][j]);
            }
        }
        // keep the next step's work (and its waits on the x loads) behind the WMMAs
        __builtin_amdgcn_sched_barrier(0);
        if (next_block) {
            cw = nw;
        }
        {
            if constexpr (!(F & 2)) {
                as[cb ^ 1][wv * 32 + lane] = decode(cw, (int) ((ns >> 1) & 7), (int) (ns & 1));
            } else {
                as[cb ^ 1][wv * 32 + lane] = __builtin_bit_cast(half8, make_uint4(cw.q0.x & 0x3BFF3BFFu, cw.q0.y & 0x3BFF3BFFu, 1u, 2u));
            }
            xs[cb ^ 1][tid * 2] = __builtin_bit_cast(half8, nx0);
            xs[cb ^ 1][tid * 2 + 1] = __builtin_bit_cast(half8, nx1);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long long t = n0 + (tg * 4 + j) * 16 + (lane & 15);
            if (t < tokens) {
                float * dst = y + t * (long long) n_out + (tile0 + rg * 4 + i) * 16 + h * 8;
                reinterpret_cast<float4 *>(dst)[0] = make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
                reinterpret_cast<float4 *>(dst)[1] = make_float4(acc[i][j][4], acc[i][j][5], acc[i][j][6], acc[i][j][7]);
            }
        }
    }
}


// 4 x 4 in 16-k steps, software-pipelined: in step st a wave holds step st's
// 8 operands in registers (read during step st - 1), issues the reads of step
// st + 1's operands, runs its 16 WMMAs, then produces step st + 2 (A decoded,
// x stored) into the buffer step st's operands came from: those were already
// in registers. One barrier per step, 28 KB of LDS.
__global__ __launch_bounds__(256) void k_gemm44p(const uint8_t * __restrict__ wt, const _Float16 * __restrict__ x,
                                                float * __restrict__ y, const int n_out, const long long blocks,
                                                const int tokens) {
    constexpr int kTk = 256;
    __shared__ uint2 grid[512];
    __shared__ half8 xs[2][kTk * 2];
    __shared__ half8 as[2][8 * 32];
    const int tid = (int) threadIdx.x;
    for (int i = tid; i < 512; i += blockDim.x) {
        const uint32_t g = omph::quant::kIq3sGrid[i];
        grid[i] = make_uint2(qd::grid_pair_to_half2<0x04010400u>(g), qd::grid_pair_to_half2<0x04030402u>(g));
    }
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int h = lane >> 4;
    const int r = lane & 15;
    const int rg = wv & 1;
    const int tg = wv >> 1;
    const long long tile0 = (long long) blockIdx.y * 8;
    const long long n0 = (long long) blockIdx.x * kTk;
    const long long k = blocks * 256;
    const long long tx = n0 + tid < tokens ? n0 + tid : 0;
    const uint4 * xrow = reinterpret_cast<const uint4 *>(x + tx * k);
    const long long nsteps = blocks * 16;
    // a step's words of the tile (sub-block sb): this lane's qs word and sign
    // half-word, the qh word, the row's scale word and d; small loads from the
    // cache (the tile's 1760 bytes are read 16 times)
    struct W {
        uint32_t q, s16, qhw, scw;
        float dv;
    };
    const auto load_w = [&](const long long st) {
        const long long b = st >> 4;
        const int sb = (int) ((st >> 1) & 7);
        const uint8_t * o = wt + ((tile0 + wv) * blocks + b) * kTileBytes;
        W w;
        w.q = *reinterpret_cast<const uint32_t *>(o + lane * 32 + sb * 4);
        w.s16 = *reinterpret_cast<const uint16_t *>(o + 1024 + lane * 16 + sb * 2);
        w.qhw = *reinterpret_cast<const uint32_t *>(o + 1536 + lane * 4);
        w.scw = *reinterpret_cast<const uint32_t *>(o + 1664 + r * 4);
        w.dv = qd::load_half(o + 1728 + r * 2);
        return w;
    };
    const auto decode = [&](const W & w, const int sb, const int gl) {
        const uint32_t q = w.q;
        const uint32_t qhn = (w.qhw >> (4 * sb)) & 0xFu;
        const uint32_t s16 = w.s16;
        const uint32_t sg = h == 0 ? (((s16 & 0xFFu) << 8) | ((s16 >> 8) << 24)) : ((s16 & 0xFFu) | ((s16 >> 8) << 16));
        const uint32_t sc = (w.scw >> (8 * (sb >> 1))) & 0xFFu;
        const float dbr = w.dv * (1.0f + 2.0f * (float) ((sb & 1) ? (sc >> 4) : (sc & 0xF)));
        const qd::half2_v d2 = {(_Float16) dbr, (_Float16) dbr};
        const int g = 2 * h + gl;
        const uint32_t word = (q >> (16 * gl)) & 0xFFFFu;
        const uint32_t i1 = (word & 0xFFu) | (((qhn >> (2 * gl)) & 1u) << 8);
        const uint32_t i2 = (word >> 8) | (((qhn >> (2 * gl + 1)) & 1u) << 8);
        const uint2 g1 = grid[i1];
        const uint2 g2 = grid[i2];
        constexpr uint32_t kSign2 = 0x80008000u;
        const uint32_t wv4[4] = {((sg << (4 * g)) & kSign2) | g1.x, ((sg << (4 * g + 1)) & kSign2) | g1.y,
                                 ((sg << (4 * g + 2)) & kSign2) | g2.x, ((sg << (4 * g + 3)) & kSign2) | g2.y};
        half8 a;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const qd::half2_v p = __builtin_bit_cast(qd::half2_v, wv4[i]) * d2;
            a[2 * i] = p[0];
            a[2 * i + 1] = p[1];
        }
        return a;
    };
    const auto xoff = [&](const long long st) { return (st >> 1) * 4 + (st & 1); };
    float8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = float8{};
        }
    }
    struct F {
        half8 a[4], b[4];
    };
    const auto fetch = [&](const int buf, F & f) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            f.a[i] = as[buf][(rg * 4 + i) * 32 + lane];
            f.b[i] = xs[buf][((tg * 4 + i) * 16 + (lane & 15)) * 2 + h];
        }
    };
    // prologue: steps 0 and 1 produced, step 0's operands in registers
    const W w0 = load_w(0);
    xs[0][tid * 2] = __builtin_bit_cast(half8, xrow[xoff(0)]);
    xs[0][tid * 2 + 1] = __builtin_bit_cast(half8, xrow[xoff(0) + 2]);
    xs[1][tid * 2] = __builtin_bit_cast(half8, xrow[xoff(1)]);
    xs[1][tid * 2 + 1] = __builtin_bit_cast(half8, xrow[xoff(1) + 2]);
    __syncthreads();  // the grid table
    as[0][wv * 32 + lane] = decode(w0, 0, 0);
    as[1][wv * 32 + lane] = decode(w0, 0, 1);
    __syncthreads();
    F f0, f1;
    fetch(0, f0);
    const auto step = [&](const long long st, const F & cur, F & nxt) {
        const int cb = (int) (st & 1);
        const long long ps = st + 2 < nsteps ? st + 2 : st;  // past the end: refill a dead buffer
        const uint4 nx0 = xrow[xoff(ps)];
        const uint4 nx1 = xrow[xoff(ps) + 2];
        const W nw = load_w(ps);
        if (st + 1 < nsteps) {
            fetch(cb ^ 1, nxt);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(cur.a[i], cur.b[j], acc[i][j]);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        as[cb][wv * 32 + lane] = decode(nw, (int) ((ps >> 1) & 7), (int) (ps & 1));
        xs[cb][tid * 2] = __builtin_bit_cast(half8, nx0);
        xs[cb][tid * 2 + 1] = __builtin_bit_cast(half8, nx1);
        __syncthreads();
    };
#pragma unroll 1
    for (long long st = 0; st < nsteps; st += 2) {
        step(st, f0, f1);
        step(st + 1, f1, f0);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long long t = n0 + (tg * 4 + j) * 16 + (lane & 15);
            if (t < tokens) {
                float * dst = y + t * (long long) n_out + (tile0 + rg * 4 + i) * 16 + h * 8;
                reinterpret_cast<float4 *>(dst)[0] = make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
                reinterpret_cast<float4 *>(dst)[1] = make_float4(acc[i][j][4], acc[i][j][5], acc[i][j][6], acc[i][j][7]);
            }
        }
    }
}


// The engine's kernel with 2 row tiles x 8 token tiles per wave (wave w: row
// tiles 2 (w >> 1) + 0, 1, tokens 128 (w & 1) ..): each wave decodes two row
// tiles into registers (the decode is doubled: each tile is decoded by two
// waves) and each B operand read from LDS feeds two WMMAs.
template <int SPLIT>  // SPLIT: 16-k LDS steps, unpadded (28 KB: 2 WGs per WGP)
__global__ __launch_bounds__(256) void k_gemm_r2(const uint8_t * __restrict__ wt, const _Float16 * __restrict__ x,
                                                float * __restrict__ y, const int n_out, const long long blocks,
                                                const int tokens) {
    constexpr int kTk = 256;
    constexpr int kLd = SPLIT ? 16 : kXLd;
    __shared__ uint2 grid[512];
    __shared__ _Float16 xs[2][kTk * kLd];
    const int tid = (int) threadIdx.x;
    for (int i = tid; i < 512; i += blockDim.x) {
        const uint32_t g = omph::quant::kIq3sGrid[i];
        grid[i] = make_uint2(qd::grid_pair_to_half2<0x04010400u>(g), qd::grid_pair_to_half2<0x04030402u>(g));
    }
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int h = lane >> 4;
    const int r = lane & 15;
    const long long tile = (long long) blockIdx.y * 8 + (wv >> 1) * 2;
    const int t0 = (wv & 1) * 8;  // first token tile
    const long long n0 = (long long) blockIdx.x * kTk;
    const long long k = blocks * 256;
    const long long tx = n0 + tid < tokens ? n0 + tid : 0;
    const uint4 * xrow = reinterpret_cast<const uint4 *>(x + tx * k);
    // SPLIT: step st = sub-block st >> 1, half gl = st & 1: lane half h' takes x
    // halves 16 h' + 8 gl .. + 7, staged as [token][h'][8]
    const auto stage = [&](const long long st, _Float16 * buf) {
        uint4 * dst = reinterpret_cast<uint4 *>(buf + tid * kLd);
        if constexpr (SPLIT) {
            const uint4 * src = xrow + (st >> 1) * 4 + (st & 1);
            dst[0] = src[0];
            dst[1] = src[2];
        } else {
            const uint4 * src = xrow + st * 4;
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
        }
    };
    float8 acc[2][8];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[i][j] = float8{};
        }
    }
    const long long nst = SPLIT ? blocks * 16 : blocks * 8;
    stage(0, xs[0]);
    __syncthreads();
    int buf = 0;
    for (long long b = 0; b < blocks; ++b) {
        uint4 q0[2], q1[2], s4[2];
        uint32_t qhw[2], scw[2];
        float dv[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint8_t * o = wt + ((tile + i) * blocks + b) * kTileBytes;
            q0[i] = *reinterpret_cast<const uint4 *>(o + lane * 32);
            q1[i] = *reinterpret_cast<const uint4 *>(o + lane * 32 + 16);
            s4[i] = *reinterpret_cast<const uint4 *>(o + 1024 + lane * 16);
            qhw[i] = *reinterpret_cast<const uint32_t *>(o + 1536 + lane * 4);
            scw[i] = *reinterpret_cast<const uint32_t *>(o + 1664 + r * 4);
            dv[i] = qd::load_half(o + 1728 + r * 2);
        }
#pragma unroll 1
        for (int sb = 0; sb < 8; ++sb) {
            half8 a[2][2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint32_t qw[8] = {q0[i].x, q0[i].y, q0[i].z, q0[i].w, q1[i].x, q1[i].y, q1[i].z, q1[i].w};
                const uint32_t sw[4] = {s4[i].x, s4[i].y, s4[i].z, s4[i].w};
                const uint32_t q = qw[sb];
                const uint32_t qhn = (qhw[i] >> (4 * sb)) & 0xFu;
                const uint32_t s16 = (sw[sb >> 1] >> (16 * (sb & 1))) & 0xFFFFu;
                const uint32_t sg = h == 0 ? (((s16 & 0xFFu) << 8) | ((s16 >> 8) << 24))
                                           : ((s16 & 0xFFu) | ((s16 >> 8) << 16));
                const uint32_t sc = (scw[i] >> (8 * (sb >> 1))) & 0xFFu;
                const float dbr = dv[i] * (1.0f + 2.0f * (float) ((sb & 1) ? (sc >> 4) : (sc & 0xF)));
                const qd::half2_v d2 = {(_Float16) dbr, (_Float16) dbr};
#pragma unroll
                for (int gl = 0; gl < 2; ++gl) {
                    const int g = 2 * h + gl;
                    const uint32_t word = (q >> (16 * gl)) & 0xFFFFu;
                    const uint32_t i1 = (word & 0xFFu) | (((qhn >> (2 * gl)) & 1u) << 8);
                    const uint32_t i2 = (word >> 8) | (((qhn >> (2 * gl + 1)) & 1u) << 8);
                    const uint2 g1 = grid[i1];
                    const uint2 g2 = grid[i2];
                    constexpr uint32_t kSign2 = 0x80008000u;
                    const uint32_t wv4[4] = {((sg << (4 * g)) & kSign2) | g1.x, ((sg << (4 * g + 1)) & kSign2) | g1.y,
                                             ((sg << (4 * g + 2)) & kSign2) | g2.x,
                                             ((sg << (4 * g + 3)) & kSign2) | g2.y};
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const qd::half2_v p = __builtin_bit_cast(qd::half2_v, wv4[e]) * d2;
                        a[i][gl][2 * e] = p[0];
                        a[i][gl][2 * e + 1] = p[1];
                    }
                }
            }
#pragma unroll
            for (int gl = 0; gl < 2; ++gl) {
                const long long st = SPLIT ? (b * 8 + sb) * 2 + gl : b * 8 + sb;
                if (SPLIT || gl == 0) {
                    if (st + 1 < nst) {
                        stage(st + 1, xs[buf ^ 1]);
                    }
                }
                const _Float16 * cur = xs[buf];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const half8 bv = SPLIT ? *reinterpret_cast<const half8 *>(cur + ((t0 + j) * 16 + (lane & 15)) * kLd + h * 8)
                                           : *reinterpret_cast<const half8 *>(cur + ((t0 + j) * 16 + (lane & 15)) * kLd + h * 16 + gl * 8);
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i][gl], bv, acc[i][j]);
                    }
                }
                if (SPLIT || gl == 1) {
                    __syncthreads();
                    buf ^= 1;
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const long long t = n0 + (t0 + j) * 16 + (lane & 15);
            if (t < tokens) {
                float * dst = y + t * (long long) n_out + (tile + i) * 16 + h * 8;
                reinterpret_cast<float4 *>(dst)[0] = make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
                reinterpret_cast<float4 *>(dst)[1] = make_float4(acc[i][j][4], acc[i][j][5], acc[i][j][6], acc[i][j][7]);
            }
        }
    }
}

struct Run {
    const uint8_t * w;
    const _Float16 * x;
    float * y;
    int rows;
    long long blocks;
    int tokens;
};

template <int M, int kTok>
double tflops(const Run & r, const int reps, const int launches) {
    const unsigned nt = (unsigned) ((r.tokens + kTok - 1) / kTok), nr = (unsigned) (r.rows / 128);
    const dim3 grid = (M & kG) ? dim3(nr, nt) : dim3(nt, nr);
    for (int i = 0; i < 5; ++i) {
        k_gemm<M, kTok><<<grid, 256>>>(r.w, r.x, r.y, r.rows, r.blocks, r.tokens);
    }
    hipEvent_t a, b;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);
    std::vector<float> v;
    for (int rep = 0; rep < reps; ++rep) {
        (void) hipEventRecord(a, nullptr);
        for (int i = 0; i < launches; ++i) {
            k_gemm<M, kTok><<<grid, 256>>>(r.w, r.x, r.y, r.rows, r.blocks, r.tokens);
        }
        (void) hipEventRecord(b, nullptr);
        (void) hipEventSynchronize(b);
        float ms = 0;
        (void) hipEventElapsedTime(&ms, a, b);
        v.push_back(ms / launches);
    }
    check(hipGetLastError(), "launch");
    std::sort(v.begin(), v.end());
    const double ms = v[v.size() / 2];
    return 2.0 * r.rows * (double) (r.blocks * 256) * r.tokens / ms / 1e9;
}

template <int M, int kTok = 256>
void row(const char * name, const Run & r, const int reps, const int launches) {
    int per = 0;
    (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_gemm<M, kTok>, 256, 0);
    std::printf("  %-34s %6.1f TFLOPS  (%d WG per WGP)\n", name, tflops<M, kTok>(r, reps, launches), per);
}

} // namespace

int main(int argc, char ** argv) {
    const int tokens = argc > 1 ? std::atoi(argv[1]) : 512;
    const int launches = argc > 2 ? std::atoi(argv[2]) : 20;
    const int rows = argc > 3 ? std::atoi(argv[3]) : 17408;
    const long long blocks = (argc > 4 ? std::atoll(argv[4]) : 5120) / 256;
    const int reps = 7;
    std::mt19937 rng(178);
    std::vector<uint8_t> wh((size_t) (rows / 16) * blocks * kTileBytes);
    for (auto & c : wh) c = (uint8_t) rng();
    for (long long t = 0; t < (rows / 16) * blocks; ++t) {  // small positive f16 scales
        for (int r = 0; r < 16; ++r) {
            const _Float16 d = (_Float16) (0.001f * (float) (rng() % 64 + 1));
            std::memcpy(&wh[(size_t) t * kTileBytes + 1728 + r * 2], &d, 2);
        }
    }
    std::vector<_Float16> xh((size_t) tokens * blocks * 256);
    for (auto & v : xh) v = (_Float16) ((float) ((int) (rng() % 2001) - 1000) / 1000.0f);
    uint8_t * w;
    _Float16 * x;
    float * y;
    check(hipMalloc(&w, wh.size()), "malloc");
    check(hipMalloc(&x, xh.size() * 2), "malloc");
    check(hipMalloc(&y, (size_t) tokens * rows * 4), "malloc");
    check(hipMemcpy(w, wh.data(), wh.size(), hipMemcpyHostToDevice), "copy");
    check(hipMemcpy(x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice), "copy");
    const Run r{w, x, y, rows, blocks, tokens};
    std::printf("IQ3_S tile GEMM %d x %lld, %d tokens, median of %d x %d launches\n", rows, blocks * 256, tokens, reps, launches);
    (void) tflops<kW | kD | kS | kB, 256>(r, 1, 200);  // ~1 s warm-up: the clocks ramp first
    {  // the engine's own gemm_q on the same tiles (link engine/src/kernels/gemm.hip)
        hipEvent_t e0, e1;
        (void) hipEventCreate(&e0);
        (void) hipEventCreate(&e1);
        std::vector<float> v;
        for (int rep = 0; rep < reps; ++rep) {
            (void) hipEventRecord(e0, nullptr);
            for (int i = 0; i < launches; ++i) {
                (void) omph::kernels::gemm_q(21, w, x, y, rows, blocks * 256, tokens, rows, nullptr);
            }
            (void) hipEventRecord(e1, nullptr);
            (void) hipEventSynchronize(e1);
            float ms = 0;
            (void) hipEventElapsedTime(&ms, e0, e1);
            v.push_back(ms / launches);
        }
        std::sort(v.begin(), v.end());
        std::printf("  %-34s %6.1f TFLOPS\n", "engine gemm_q", 2.0 * rows * (double) (blocks * 256) * tokens / v[reps / 2] / 1e9);
    }
    row<kW | kD | kS | kB>("full (the engine's kernel)", r, reps, launches);
    row<kD | kS | kB>("no W loads", r, reps, launches);
    row<kW | kS | kB>("no decode", r, reps, launches);
    row<kW | kD | kB>("no staging / barriers", r, reps, launches);
    row<kW | kD | kS>("no LDS B reads", r, reps, launches);
    row<kW | kD>("W + decode only", r, reps, launches);
    row<kS | kB>("x path only (staging + LDS B)", r, reps, launches);
    row<kB>("LDS B reads only", r, reps, launches);
    row<0>("WMMA alone", r, reps, launches);
    const auto vs_engine = [&](const char * name, auto kern) {
        const dim3 grid((unsigned) ((tokens + 255) / 256), (unsigned) (rows / 128));
        float * y2;
        check(hipMalloc(&y2, (size_t) tokens * rows * 4), "malloc");
        k_gemm<kW | kD | kS | kB, 256><<<grid, 256>>>(w, x, y, rows, blocks, tokens);
        kern<<<grid, 256>>>(w, x, y2, rows, blocks, tokens);
        check(hipDeviceSynchronize(), name);
        std::vector<float> a1((size_t) tokens * rows), a2(a1.size());
        check(hipMemcpy(a1.data(), y, a1.size() * 4, hipMemcpyDeviceToHost), "copy");
        check(hipMemcpy(a2.data(), y2, a2.size() * 4, hipMemcpyDeviceToHost), "copy");
        size_t differ = 0;
        for (size_t i = 0; i < a1.size(); ++i) differ += std::memcmp(&a1[i], &a2[i], 4) != 0;
        hipEvent_t e0, e1;
        (void) hipEventCreate(&e0);
        (void) hipEventCreate(&e1);
        std::vector<float> v;
        for (int rep = 0; rep < reps; ++rep) {
            (void) hipEventRecord(e0, nullptr);
            for (int i = 0; i < launches; ++i) kern<<<grid, 256>>>(w, x, y2, rows, blocks, tokens);
            (void) hipEventRecord(e1, nullptr);
            (void) hipEventSynchronize(e1);
            float ms = 0;
            (void) hipEventElapsedTime(&ms, e0, e1);
            v.push_back(ms / launches);
        }
        std::sort(v.begin(), v.end());
        int per = 0;
        (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, kern, 256, 0);
        std::printf("  %-34s %6.1f TFLOPS  (%d WG per WGP; %zu of %zu outputs differ from the engine's)\n", name,
                    2.0 * rows * (double) (blocks * 256) * tokens / v[reps / 2] / 1e9, per, differ, a1.size());
        (void) hipFree(y2);
    };
    vs_engine("4 x 4, A through LDS", k_gemm44);
    vs_engine("4 x 4, 16-k steps", k_gemm44s<0>);
    vs_engine("  no barrier", k_gemm44s<1>);
    vs_engine("  no decode", k_gemm44s<2>);
    vs_engine("  no x loads", k_gemm44s<4>);
    vs_engine("  no LDS reads", k_gemm44s<8>);
    vs_engine("  none of them", k_gemm44s<15>);
    vs_engine("4 x 4, pipelined", k_gemm44p);
    vs_engine("2 x 8, own decode", k_gemm_r2<0>);
    vs_engine("2 x 8, own decode, 16-k steps", k_gemm_r2<1>);
    row<kW | kD | kS | kB | kG>("full, row tiles fastest", r, reps, launches);
    row<kW | kD | kS | kB, 128>("full, 128 tokens per WG", r, reps, launches);
    row<kW | kD | kS | kB | kG, 128>("full, 128 tokens, row tiles fastest", r, reps, launches);
    row<kB, 128>("LDS B reads only, 128 tokens", r, reps, launches);
    row<kW | kD | kS | kB | kG | kP>("full, rows fastest, pipelined B", r, reps, launches);
    row<kB | kP>("LDS B reads only, pipelined", r, reps, launches);
    row<kW | kD | kS | kB | kG | kR | kP>("+ x via registers (1 LDS buffer)", r, reps, launches);
    row<kW | kD | kS | kB | kG | kP, 128>("pipelined, 128 tokens", r, reps, launches);
    return 0;
}
