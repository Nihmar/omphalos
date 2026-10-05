// Tile-GEMM structure bench for #326: the engine's pre-#326 shape (128 rows x 256 tokens per
// workgroup, one 16-row W tile x 16 token tiles per wave) against the 2 x 8 shape (256 rows x 128
// tokens, two row tiles x eight token tiles per wave, all eight waves on distinct row pairs).
// Both decode their weights with the same TD::frag decoders, so every output must be bit-identical;
// the 2 x 8 shape reads a B operand from LDS once per two WMMAs instead of once per WMMA and its x
// buffer is half the size (20 KB against 40 KB, two workgroups per WGP).
//   hipcc -O3 --offload-arch=gfx1200 -std=c++20 -I engine/src -o /tmp/tile2 bench/gemm/tile2.hip
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/quant_decode.hh"

namespace qd = omph::kernels::qd;

namespace {

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr int kXLd = 40;  // halves per staged token row (32 + 8 pad)

void check(hipError_t e, const char * what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

struct T {
    const char * name;
    uint32_t type;
    long long bytes;
};

const T kTypes[] = {
    {"iq3s", 21, omph::format::kIq3sTileBytes},     {"iq3xxs", 18, omph::format::kIq3xxsTileBytes},
    {"iq4xs", 23, omph::format::kIq4xsTileBytes},   {"q4k", 12, omph::format::kQ4kTileBytes},
    {"iq2s", 22, omph::format::kIq2sTileBytes},     {"q2k", 10, omph::format::kQ2kTileBytes},
    {"iq2xs", 17, omph::format::kIq2xsTileBytes},   {"iq2xxs", 16, omph::format::kIq2xxsTileBytes},
    {"q6k", 14, omph::format::kQ6kTileBytes},       {"iq1m", 29, omph::format::kIq1mTileBytes},
};

// The pre-#326 shape: 8 waves x 16 rows (128 rows) x 256 tokens.
template <class TD>
__global__ __launch_bounds__(256) void gemm_t1(const uint8_t * __restrict__ wt,
                                              const _Float16 * __restrict__ x, float * __restrict__ y,
                                              const long long blocks, const int tokens, const long long ldy) {
    constexpr int kTok = 256;
    __shared__ __attribute__((aligned(16))) uint8_t tab[TD::kLds];
    __shared__ _Float16 xs[2][kTok * kXLd];
    const int tid = (int) threadIdx.x;
    TD::init(tab);
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int h = lane >> 4;
    const long long tile = (long long) blockIdx.y * 8 + wv;
    const long long n0 = (long long) blockIdx.x * kTok;
    const long long k = blocks * 256;
    const long long tx = n0 + tid < tokens ? n0 + tid : 0;
    const uint4 * xrow = reinterpret_cast<const uint4 *>(x + tx * k);
    const auto stage = [&](const long long s, _Float16 * buf) {
        const uint4 * src = xrow + s * 4;
        uint4 * dst = reinterpret_cast<uint4 *>(buf + tid * kXLd);
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
    };
    float8 acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        acc[j] = float8{};
    }
    stage(0, xs[0]);
    __syncthreads();
    int buf = 0;
    for (long long b = 0; b < blocks; ++b) {
        const typename TD::Blk kb = TD::load(wt + (tile * blocks + b) * TD::kBytes, lane);
#pragma unroll 1
        for (int sb = 0; sb < 8; ++sb) {
            const long long s = b * 8 + sb;
            if (s + 1 < blocks * 8) {
                stage(s + 1, xs[buf ^ 1]);
            }
            const half8 a[2] = {TD::frag(tab, kb, sb, 0, h), TD::frag(tab, kb, sb, 1, h)};
            const _Float16 * cur = xs[buf];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
#pragma unroll
                for (int gl = 0; gl < 2; ++gl) {
                    const half8 bv = *reinterpret_cast<const half8 *>(
                        cur + (j * 16 + (lane & 15)) * kXLd + h * 16 + gl * 8);
                    acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[gl], bv, acc[j]);
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
            float * dst = y + t * ldy + tile * 16 + h * 8;
            reinterpret_cast<float4 *>(dst)[0] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
            reinterpret_cast<float4 *>(dst)[1] = make_float4(acc[j][4], acc[j][5], acc[j][6], acc[j][7]);
        }
    }
}

// #326: 8 waves x 32 rows (256 rows) x 128 tokens, two row tiles and eight token
// tiles per wave, so each B operand from LDS feeds two WMMAs.
template <class TD>
__global__ __launch_bounds__(256) void gemm_t2(const uint8_t * __restrict__ wt,
                                              const _Float16 * __restrict__ x, float * __restrict__ y,
                                              const long long blocks, const int tokens, const long long ldy) {
    constexpr int kTok = 128;
    __shared__ __attribute__((aligned(16))) uint8_t tab[TD::kLds];
    __shared__ _Float16 xs[2][kTok * kXLd];
    const int tid = (int) threadIdx.x;
    TD::init(tab);
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int h = lane >> 4;
    const long long tile = (long long) blockIdx.y * 16 + wv * 2;
    const long long n0 = (long long) blockIdx.x * kTok;
    const long long k = blocks * 256;
    const long long tx = n0 + tid < tokens ? n0 + tid : 0;
    const uint4 * xrow = reinterpret_cast<const uint4 *>(x + tx * k);
    const auto stage = [&](const long long s, _Float16 * buf) {
        if (tid >= kTok) {
            return;
        }
        const uint4 * src = xrow + s * 4;
        uint4 * dst = reinterpret_cast<uint4 *>(buf + tid * kXLd);
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
    };
    float8 acc[2][8];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[i][j] = float8{};
        }
    }
    stage(0, xs[0]);
    __syncthreads();
    int buf = 0;
    for (long long b = 0; b < blocks; ++b) {
        const typename TD::Blk kb[2] = {TD::load(wt + (tile * blocks + b) * TD::kBytes, lane),
                                        TD::load(wt + ((tile + 1) * blocks + b) * TD::kBytes, lane)};
#pragma unroll 1
        for (int sb = 0; sb < 8; ++sb) {
            const half8 a[2][2] = {{TD::frag(tab, kb[0], sb, 0, h), TD::frag(tab, kb[0], sb, 1, h)},
                                   {TD::frag(tab, kb[1], sb, 0, h), TD::frag(tab, kb[1], sb, 1, h)}};
#pragma unroll
            for (int gl = 0; gl < 2; ++gl) {
                if (gl == 0) {
                    const long long s = b * 8 + sb;
                    if (s + 1 < blocks * 8) {
                        stage(s + 1, xs[buf ^ 1]);
                    }
                }
                const _Float16 * cur = xs[buf];
                half8 bv[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    bv[j] = *reinterpret_cast<const half8 *>(
                        cur + (j * 16 + (lane & 15)) * kXLd + h * 16 + gl * 8);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i][gl], bv[j], acc[i][j]);
                    }
                }
                if (gl == 1) {
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
            const long long t = n0 + j * 16 + (lane & 15);
            if (t < tokens) {
                float * dst = y + t * ldy + (tile + i) * 16 + h * 8;
                reinterpret_cast<float4 *>(dst)[0] = make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
                reinterpret_cast<float4 *>(dst)[1] = make_float4(acc[i][j][4], acc[i][j][5], acc[i][j][6], acc[i][j][7]);
            }
        }
    }
}

#define OMPH_TILE_BENCH(NAME, TID, TD)                                                                       \
    case TID: {                                                                                        \
        const dim3 g1((unsigned) ((tokens + 255) / 256), (unsigned) (rows / 128));                       \
        const dim3 g2((unsigned) ((tokens + 127) / 128), (unsigned) (rows / 256));                       \
        const auto run1 = [&] { gemm_t1<TD><<<g1, 256>>>(w, x, y1, blocks, tokens, rows); };           \
        const auto run2 = [&] { gemm_t2<TD><<<g2, 256>>>(w, x, y2, blocks, tokens, rows); };           \
        run1();                                                                                         \
        run2();                                                                                         \
        check(hipDeviceSynchronize(), #NAME);                                                           \
        std::vector<float> a1((size_t) tokens * rows), a2(a1.size());                                    \
        check(hipMemcpy(a1.data(), y1, a1.size() * 4, hipMemcpyDeviceToHost), "copy");                   \
        check(hipMemcpy(a2.data(), y2, a2.size() * 4, hipMemcpyDeviceToHost), "copy");                   \
        size_t differ = 0;                                                                              \
        for (size_t i = 0; i < a1.size(); ++i) differ += std::memcmp(&a1[i], &a2[i], 4) != 0;            \
        const auto timeit = [&](const auto & fn) {                                                       \
            std::vector<float> v;                                                                        \
            for (int rep = 0; rep < reps; ++rep) {                                                       \
                hipEvent_t e0, e1;                                                                      \
                (void) hipEventCreate(&e0);                                                             \
                (void) hipEventCreate(&e1);                                                             \
                (void) hipEventRecord(e0, nullptr);                                                     \
                for (int i = 0; i < launches; ++i) fn();                                                \
                (void) hipEventRecord(e1, nullptr);                                                     \
                (void) hipEventSynchronize(e1);                                                         \
                float ms = 0;                                                                           \
                (void) hipEventElapsedTime(&ms, e0, e1);                                                \
                v.push_back(ms / launches);                                                             \
            }                                                                                           \
            std::sort(v.begin(), v.end());                                                              \
            return v[v.size() / 2];                                                                     \
        };                                                                                             \
        for (int i = 0; i < 150; ++i) {  /* ~1 s: the clocks ramp first */                               \
            run1();                                                                                     \
            run2();                                                                                     \
        }                                                                                               \
        check(hipDeviceSynchronize(), "warm-up");                                                       \
        const double flops = 2.0 * rows * (double) (blocks * 256) * tokens;                             \
        const float ms1 = timeit(run1);                                                                 \
        const float ms2 = timeit(run2);                                                                 \
        std::printf("%-7s  128x256 %6.1f TFLOPS   2x8 %6.1f TFLOPS  (%+5.1f%%)  %s\n", #NAME,           \
                    flops / ms1 / 1e9, flops / ms2 / 1e9, (ms1 / ms2 - 1) * 100,                         \
                    differ == 0 ? "bit-identical" : "DIFFERS");                                          \
        break;                                                                                          \
    }

} // namespace

int main(int argc, char ** argv) {
    const int tokens = argc > 1 ? std::atoi(argv[1]) : 512;
    const int rows = argc > 2 ? std::atoi(argv[2]) : 17408;
    const int rblocks = argc > 3 ? std::atoi(argv[3]) : 5120;
    const int reps = 5, launches = 20;
    std::mt19937 rng(178);
    for (const T & t : kTypes) {
        const long long blocks = rblocks / 256;
        std::vector<uint8_t> wh((size_t) (rows / 16) * blocks * t.bytes);
        for (auto & c : wh) c = (uint8_t) rng();
        for (long long q = 0; q < (rows / 16) * blocks; ++q) {  // small positive f16 scales at the tail
            for (int r = 0; r < 16; ++r) {
                const _Float16 d = (_Float16) (0.001f * (float) (rng() % 64 + 1));
                std::memcpy(&wh[(size_t) q * t.bytes + t.bytes - 32 + r * 2], &d, 2);
            }
        }
        std::vector<_Float16> xhv((size_t) tokens * blocks * 256);
        for (auto & v : xhv) v = (_Float16) ((float) ((int) (rng() % 2001) - 1000) / 1000.0f);
        uint8_t * w;
        _Float16 * x;
        float * y1;
        float * y2;
        check(hipMalloc(&w, wh.size()), "malloc");
        check(hipMalloc(&x, xhv.size() * 2), "malloc");
        check(hipMalloc(&y1, (size_t) tokens * rows * 4), "malloc");
        check(hipMalloc(&y2, (size_t) tokens * rows * 4), "malloc");
        check(hipMemcpy(w, wh.data(), wh.size(), hipMemcpyHostToDevice), "copy");
        check(hipMemcpy(x, xhv.data(), xhv.size() * 2, hipMemcpyHostToDevice), "copy");
        switch (t.type) {
            OMPH_TILE_BENCH(iq3s, 21, qd::TileIq3s)
            OMPH_TILE_BENCH(iq3xxs, 18, qd::TileIq3xxs)
            OMPH_TILE_BENCH(iq4xs, 23, qd::TileIq4xs)
            OMPH_TILE_BENCH(q4k, 12, qd::TileQ4k)
            OMPH_TILE_BENCH(iq2s, 22, qd::TileIq2s)
            OMPH_TILE_BENCH(q2k, 10, qd::TileQ2k)
            OMPH_TILE_BENCH(iq2xs, 17, qd::TileIq2xs)
            OMPH_TILE_BENCH(iq2xxs, 16, qd::TileIq2xxs)
            OMPH_TILE_BENCH(q6k, 14, qd::TileQ6k)
            OMPH_TILE_BENCH(iq1m, 29, qd::TileIq1m)
            default: break;
        }
        (void) hipFree(w);
        (void) hipFree(x);
        (void) hipFree(y1);
        (void) hipFree(y2);
    }
    return 0;
}
