// An int8 IQ3_S tile GEMM for the prefill (#213 phase 2): the engine's tile
// GEMM (gemm.hip, gemm_tiles_kernel: 8 waves x 16 W rows x 256 tokens, x one
// 32-weight sub-block at a time through LDS) with the iu8 WMMA instead of the
// f16 one. The weights go in as their exact integers (grid magnitude x sign,
// |q| <= 15), x as int8 in blocks of 32 with one f32 scale each (llama.cpp's
// q8_1, quantize_q8 below); each sub-block's int32 products are rescaled into
// the f32 accumulators by d * (1 + 2 sc) of the row times the token's x scale.
// Checked against the engine's gemm_q fed the same int8 x back in f16
// (OMPH_TEST_Q8ACT's numerics), timed against gemm_q on the f16 x, on random
// tiles shaped like ffn_up (17408 x 5120), 512 tokens.
//   hipcc -O3 --offload-arch=gfx1200 -std=c++20 -I engine/src -o /var/tmp/int8_tiles bench/gemm/int8_tiles.hip \
//       engine/src/kernels/gemm.hip engine/src/kernels/elementwise.hip && /var/tmp/int8_tiles
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/elementwise.hh"
#include "kernels/gemm.hh"
#include "kernels/quant_decode.hh"

namespace {

typedef float float8 __attribute__((ext_vector_type(8)));
typedef int int8v __attribute__((ext_vector_type(8)));
typedef int int2v __attribute__((ext_vector_type(2)));

constexpr long long kTileBytes = 1760;
constexpr int kTok = 256;
constexpr int kXLd = 40;  // bytes per staged token row: 32 int8, the f32 scale, 4 pad (conflict-free b64 reads)

void check(hipError_t e, const char * what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

// x (tokens x k f16) -> int8 in blocks of 32 + one f32 scale per block
__global__ void quantize_q8(const _Float16 * __restrict__ x, int8_t * __restrict__ xq, float * __restrict__ xs,
                            const long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) {
        return;
    }
    float v[32];
    const uint4 * src = reinterpret_cast<const uint4 *>(x + b * 32);
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4 u = src[i];
        const _Float16 * h = reinterpret_cast<const _Float16 *>(&u);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[i * 8 + j] = (float) h[j];
            amax = fmaxf(amax, fabsf(v[i * 8 + j]));
        }
    }
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    uint32_t w[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t p = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = (int) fminf(fmaxf(rintf(v[i * 4 + j] * id), -127.0f), 127.0f);
            p |= (uint32_t) (q & 0xFF) << (8 * j);
        }
        w[i] = p;
    }
    uint4 * dst = reinterpret_cast<uint4 *>(xq + b * 32);
    dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
    dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
    xs[b] = d;
}

// The 8 int8 weights (two dwords) of row lane & 15, group 2 h + gl of sub-block sb: the f16
// TileIq3s::frag without the scale, the signs applied bytewise.
__device__ inline int2v frag_i8(const uint32_t * grid, const omph::kernels::qd::TileIq3s::Blk & k, const int sb,
                                const int gl, const int h) {
    const uint32_t qw[8] = {k.q0.x, k.q0.y, k.q0.z, k.q0.w, k.q1.x, k.q1.y, k.q1.z, k.q1.w};
    const uint32_t sw[4] = {k.s4.x, k.s4.y, k.s4.z, k.s4.w};
    const uint32_t q = qw[sb];
    const uint32_t qhn = (k.qhw >> (4 * sb)) & 0xFu;
    const uint32_t s16 = (sw[sb >> 1] >> (16 * (sb & 1))) & 0xFFFFu;
    // weight 2i's sign: bit 7 - 4 gl - i of the low byte; weight 2i + 1's: of the high byte
    const uint32_t lo = ((s16 & 0xFFu) >> (4 - 4 * gl)) & 0xFu;
    const uint32_t hi = ((s16 >> 8) >> (4 - 4 * gl)) & 0xFu;
    const uint32_t word = (q >> (16 * gl)) & 0xFFFFu;
    const uint32_t i1 = (word & 0xFFu) | (((qhn >> (2 * gl)) & 1u) << 8);
    const uint32_t i2 = (word >> 8) | (((qhn >> (2 * gl + 1)) & 1u) << 8);
    const uint32_t g[2] = {grid[i1], grid[i2]};
    int2v a;
#pragma unroll
    for (int d = 0; d < 2; ++d) {
        // weights 4d .. 4d + 3: i = 2d + (k >> 1), parity k & 1 -> bit 3 - i of lo / hi
        const uint32_t m = (((lo >> (3 - 2 * d)) & 1u) * 0x000000FFu) | (((hi >> (3 - 2 * d)) & 1u) * 0x0000FF00u) |
                           (((lo >> (2 - 2 * d)) & 1u) * 0x00FF0000u) | (((hi >> (2 - 2 * d)) & 1u) * 0xFF000000u);
        a[d] = (int) ((g[d] ^ m) + (m & 0x01010101u));  // bytewise negate (|q| >= 1: no carry)
    }
    return a;
}

// V: 0 the full kernel; 1 no rescale (the int32 products summed straight into acc: wrong results,
// the VALU cost's measure)
template <int V, int G = 1>
__global__ __launch_bounds__(256) void gemm_i8(const uint8_t * __restrict__ wt, const int8_t * __restrict__ xq,
                                              const float * __restrict__ xs, float * __restrict__ y,
                                              const long long blocks, const int tokens, const long long ldy) {
    __shared__ uint32_t grid[512];
    __shared__ __attribute__((aligned(16))) uint8_t xl[2][kTok * kXLd];
    const int tid = (int) threadIdx.x;
    for (int i = tid; i < 512; i += 256) {
        grid[i] = omph::quant::kIq3sGrid[i];
    }
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int h = lane >> 4;
    const long long tile = (long long) blockIdx.y * 8 + wv;
    const long long n0 = (long long) blockIdx.x * kTok;
    const long long k = blocks * 256;
    const long long tx = n0 + tid < tokens ? n0 + tid : 0;
    const uint4 * xrow = reinterpret_cast<const uint4 *>(xq + tx * k);
    const float * srow = xs + tx * (k / 32);
    const auto stage = [&](const long long s, uint8_t * buf) {
        uint8_t * dst = buf + tid * kXLd;
        reinterpret_cast<uint4 *>(dst)[0] = xrow[s * 2];
        reinterpret_cast<uint4 *>(dst)[1] = xrow[s * 2 + 1];
        *reinterpret_cast<float *>(dst + 32) = srow[s];
    };
    float8 acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        acc[j] = float8{};
    }
    stage(0, xl[0]);
    __syncthreads();
    int buf = 0;
    for (long long b = 0; b < blocks; ++b) {
        const uint8_t * o = wt + (tile * blocks + b) * kTileBytes;
        const auto kb = omph::kernels::qd::TileIq3s::load(o, lane);
        // the D rows' scales: rows h * 8 .. + 7 of the tile
        const uint4 sc0 = *reinterpret_cast<const uint4 *>(o + 1664 + h * 32);
        const uint4 sc1 = *reinterpret_cast<const uint4 *>(o + 1664 + h * 32 + 16);
        const uint4 dd = *reinterpret_cast<const uint4 *>(o + 1728 + h * 16);
        const uint32_t scw[8] = {sc0.x, sc0.y, sc0.z, sc0.w, sc1.x, sc1.y, sc1.z, sc1.w};
        const _Float16 * dh = reinterpret_cast<const _Float16 *>(&dd);
#pragma unroll 1
        for (int sb = 0; sb < 8; ++sb) {
            const long long s = b * 8 + sb;
            if (s + 1 < blocks * 8) {
                stage(s + 1, xl[buf ^ 1]);
            }
            const int2v a0 = frag_i8(grid, kb, sb, 0, h);
            const int2v a1 = frag_i8(grid, kb, sb, 1, h);
            float wsc[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t nib = (scw[e] >> (8 * (sb >> 1) + 4 * (sb & 1))) & 0xFu;
                wsc[e] = (float) dh[e] * (float) (1 + 2 * (int) nib);
            }
            const uint8_t * cur = xl[buf];
            // G token tiles at a time: their 2 G WMMAs are independent, so the units stay busy
            // while the previous group's int32 products are rescaled
#pragma unroll
            for (int j0 = 0; j0 < 16; j0 += G) {
                int8v t[G];
                float sx[G];
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    const uint8_t * row = cur + ((j0 + g) * 16 + (lane & 15)) * kXLd;
                    const int2v b0 = *reinterpret_cast<const int2v *>(row + h * 16);
                    t[g] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, true, b0, int8v{}, false);
                    sx[g] = *reinterpret_cast<const float *>(row + 32);
                }
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    const uint8_t * row = cur + ((j0 + g) * 16 + (lane & 15)) * kXLd;
                    const int2v b1 = *reinterpret_cast<const int2v *>(row + h * 16 + 8);
                    t[g] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, true, b1, t[g], false);
                }
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    if constexpr (V == 1) {
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            acc[j0 + g][e] += (float) t[g][e];
                        }
                    } else {
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            acc[j0 + g][e] = fmaf((float) t[g][e], wsc[e] * sx[g], acc[j0 + g][e]);
                        }
                    }
                }
            }
            __syncthreads();
            buf ^= 1;
        }
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const long long t = n0 + j * 16 + (lane & 15);
        if (t < tokens) {
            float4 * dst = reinterpret_cast<float4 *>(y + t * ldy + tile * 16 + h * 8);
            dst[0] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
            dst[1] = make_float4(acc[j][4], acc[j][5], acc[j][6], acc[j][7]);
        }
    }
}

template <class F>
double median_ms(F && launch, const int reps, const int launches) {
    hipEvent_t a, b;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);
    std::vector<float> v;
    for (int rep = 0; rep < reps; ++rep) {
        (void) hipEventRecord(a, nullptr);
        for (int i = 0; i < launches; ++i) {
            launch();
        }
        (void) hipEventRecord(b, nullptr);
        (void) hipEventSynchronize(b);
        float ms = 0;
        (void) hipEventElapsedTime(&ms, a, b);
        v.push_back(ms / launches);
    }
    check(hipGetLastError(), "launch");
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char ** argv) {
    const int tokens = argc > 1 ? std::atoi(argv[1]) : 512;
    const int launches = argc > 2 ? std::atoi(argv[2]) : 20;
    const int rows = argc > 3 ? std::atoi(argv[3]) : 17408;
    const long long blocks = (argc > 4 ? std::atoll(argv[4]) : 5120) / 256;
    const long long k = blocks * 256;
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
    std::vector<_Float16> xh((size_t) tokens * k);
    for (auto & v : xh) v = (_Float16) ((float) ((int) (rng() % 2001) - 1000) / 1000.0f);
    uint8_t * w;
    _Float16 *x, *xr;
    int8_t * xq;
    float *xs, *y, *yr;
    check(hipMalloc(&w, wh.size()), "malloc");
    check(hipMalloc(&x, xh.size() * 2), "malloc");
    check(hipMalloc(&xr, xh.size() * 2), "malloc");
    check(hipMalloc(&xq, xh.size()), "malloc");
    check(hipMalloc(&xs, xh.size() / 32 * 4), "malloc");
    check(hipMalloc(&y, (size_t) tokens * rows * 4), "malloc");
    check(hipMalloc(&yr, (size_t) tokens * rows * 4), "malloc");
    check(hipMemcpy(w, wh.data(), wh.size(), hipMemcpyHostToDevice), "copy");
    check(hipMemcpy(x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice), "copy");
    const long long nb = (long long) tokens * k / 32;
    const auto quant = [&] { quantize_q8<<<(unsigned) ((nb + 255) / 256), 256>>>(x, xq, xs, nb); };
    const dim3 grid((unsigned) ((tokens + kTok - 1) / kTok), (unsigned) (rows / 128));
    const double flop = 2.0 * rows * (double) k * tokens;
    std::printf("IQ3_S tile GEMM %d x %lld, %d tokens, median of %d x %d launches\n", rows, k, tokens, reps, launches);

    // correctness: gemm_q on x through the same int8 round trip
    quant();
    gemm_i8<0><<<grid, 256>>>(w, xq, xs, y, blocks, tokens, rows);
    (void) omph::kernels::q8_roundtrip_f16(x, xr, tokens, k, 32, nullptr);
    (void) omph::kernels::gemm_q(21, w, xr, yr, rows, k, tokens, rows, nullptr);
    check(hipDeviceSynchronize(), "check run");
    std::vector<float> a((size_t) tokens * rows), r(a.size());
    check(hipMemcpy(a.data(), y, a.size() * 4, hipMemcpyDeviceToHost), "copy");
    check(hipMemcpy(r.data(), yr, r.size() * 4, hipMemcpyDeviceToHost), "copy");
    double num = 0, den = 0, worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double) (a[i] - r[i]) * (a[i] - r[i]);
        den += (double) r[i] * r[i];
        worst = std::max(worst, (double) std::fabs(a[i] - r[i]));
    }
    std::printf("  vs gemm_q on the int8 round trip of x: rel. RMS error %.2e, max abs %.3e (rms of y %.3e)\n",
                std::sqrt(num / den), worst, std::sqrt(den / a.size()));

    // ~1 s warm-up: the clocks ramp first
    (void) median_ms([&] { (void) omph::kernels::gemm_q(21, w, x, yr, rows, k, tokens, rows, nullptr); }, 1, 200);
    const double f16 = median_ms([&] { (void) omph::kernels::gemm_q(21, w, x, yr, rows, k, tokens, rows, nullptr); },
                                 reps, launches);
    const double i8 = median_ms([&] { gemm_i8<0><<<grid, 256>>>(w, xq, xs, y, blocks, tokens, rows); }, reps, launches);
    const double i8n = median_ms([&] { gemm_i8<1><<<grid, 256>>>(w, xq, xs, y, blocks, tokens, rows); }, reps, launches);
    const double g2 = median_ms([&] { gemm_i8<0, 2><<<grid, 256>>>(w, xq, xs, y, blocks, tokens, rows); }, reps, launches);
    const double g4 = median_ms([&] { gemm_i8<0, 4><<<grid, 256>>>(w, xq, xs, y, blocks, tokens, rows); }, reps, launches);
    const double g4n = median_ms([&] { gemm_i8<1, 4><<<grid, 256>>>(w, xq, xs, y, blocks, tokens, rows); }, reps, launches);
    const double q = median_ms(quant, reps, launches);
    int per = 0;
    (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, gemm_i8<0>, 256, 0);
    std::printf("  %-36s %7.3f ms  %6.1f TFLOPS\n", "engine gemm_q (f16 x)", f16, flop / f16 / 1e9);
    std::printf("  %-36s %7.3f ms  %6.1f TOPS  (%d WG per WGP)\n", "int8 tile GEMM", i8, flop / i8 / 1e9, per);
    std::printf("  %-36s %7.3f ms  %6.1f TOPS\n", "  without the rescale (wrong y)", i8n, flop / i8n / 1e9);
    std::printf("  %-36s %7.3f ms  %6.1f TOPS\n", "  groups of 2 token tiles", g2, flop / g2 / 1e9);
    std::printf("  %-36s %7.3f ms  %6.1f TOPS\n", "  groups of 4 token tiles", g4, flop / g4 / 1e9);
    std::printf("  %-36s %7.3f ms  %6.1f TOPS\n", "  groups of 4, without the rescale", g4n, flop / g4n / 1e9);
    std::printf("  %-36s %7.3f ms\n", "quantize x to q8 (once per input)", q);
    std::printf("  int8 + quantize vs f16: %.2fx\n", f16 / (i8 + q));
    return 0;
}
