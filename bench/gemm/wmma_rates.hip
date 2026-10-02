// Sustained WMMA rates on gfx1200 per operand type (#203 phase 0): f16, bf16, iu8,
// fp8, iu4 (16x16x32), on random register operands (4 A x 4 B per wave, 16
// accumulators), ~5 s per type so the power cap (160 W) has settled. PLAN.md
// §10.5's "no int8 matrix path on gfx1200" (M3) came from assembling gfx11's
// mnemonic; the gfx12 builtins below are the ones llama.cpp's MMQ uses on RDNA4.
//   hipcc -O3 --offload-arch=gfx1200 -o /var/tmp/wmma_rates bench/gemm/wmma_rates.hip
//   /var/tmp/wmma_rates [seconds per type]   (sample amd-smi metric -c -p alongside)
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef short s8v __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef int i8v __attribute__((ext_vector_type(8)));
typedef int i2v __attribute__((ext_vector_type(2)));

enum { kF16, kBF16, kIU8, kFP8, kIU4 };

template <int T>
__global__ __launch_bounds__(256) void k(const uint4 * src, float * out, const int iters) {
    const int lane = threadIdx.x & 31;
    uint4 a[4], b[4];
    for (int i = 0; i < 4; ++i) {
        a[i] = src[((blockIdx.x * 8 + i) % 1024) * 32 + lane];
        b[i] = src[((blockIdx.x * 8 + 4 + i) % 1024) * 32 + lane];
    }
    float s = 0.0f;
    if constexpr (T == kF16 || T == kBF16 || T == kFP8) {
        f8 acc[4][4] = {};
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if constexpr (T == kF16) {
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                            __builtin_bit_cast(h8, a[i]), __builtin_bit_cast(h8, b[j]), acc[i][j]);
                    } else if constexpr (T == kBF16) {
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                            __builtin_bit_cast(s8v, a[i]), __builtin_bit_cast(s8v, b[j]), acc[i][j]);
                    } else {
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                            i2v{(int) a[i].x, (int) a[i].y}, i2v{(int) b[j].x, (int) b[j].y}, acc[i][j]);
                    }
                }
            }
        }
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) s += acc[i][j][0];
    } else {
        i8v acc[4][4] = {};
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if constexpr (T == kIU8) {
                        acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                            true, i2v{(int) a[i].x, (int) a[i].y}, true, i2v{(int) b[j].x, (int) b[j].y}, acc[i][j],
                            false);
                    } else {
                        acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                            true, i2v{(int) a[i].x, (int) a[i].y}, true, i2v{(int) b[j].x, (int) b[j].y}, acc[i][j],
                            false);
                    }
                }
            }
        }
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) s += (float) acc[i][j][0];
    }
    if (s == 1234.5f) out[0] = s;
}

template <int T>
void run(const char * name, const uint4 * d, float * o, const double seconds, const double ops_per_wmma) {
    const int wgs = 256, iters = 4000;
    hipEvent_t e0, e1;
    (void) hipEventCreate(&e0);
    (void) hipEventCreate(&e1);
    std::vector<double> rates;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        (void) hipEventRecord(e0);
        k<T><<<wgs, 256>>>(d, o, iters);
        (void) hipEventRecord(e1);
        (void) hipEventSynchronize(e1);
        float ms = 0;
        (void) hipEventElapsedTime(&ms, e0, e1);
        rates.push_back((double) wgs * 8 * iters * 16 * ops_per_wmma / ms / 1e9);
    }
    // the last half of the window: the clocks have settled under the power cap
    double sum = 0;
    const size_t from = rates.size() / 2;
    for (size_t i = from; i < rates.size(); ++i) sum += rates[i];
    std::printf("%-5s %6.1f T(FL)OPS sustained (mean of the last %zu of %zu launches)\n", name,
                sum / (double) (rates.size() - from), rates.size() - from, rates.size());
    std::fflush(stdout);
}

int main(int argc, char ** argv) {
    const double seconds = argc > 1 ? std::atof(argv[1]) : 5.0;
    std::vector<uint32_t> h(1024 * 32 * 4);
    std::mt19937 rng(203);
    for (auto & v : h) {
        // random bytes; per type the bit patterns stay finite (f16 / bf16: exponent
        // bits capped, fp8 E4M3: no 0x7F / 0xFF NaN codes)
        v = rng() & 0x3BFF3BFFu;
    }
    uint4 * d;
    float * o;
    (void) hipMalloc(&d, h.size() * 4);
    (void) hipMalloc(&o, 4);
    (void) hipMemcpy(d, h.data(), h.size() * 4, hipMemcpyHostToDevice);
    run<kF16>("f16", d, o, 1.0, 8192);  // warm-up
    run<kF16>("f16", d, o, seconds, 8192);
    run<kBF16>("bf16", d, o, seconds, 8192);
    run<kIU8>("iu8", d, o, seconds, 8192);
    run<kFP8>("fp8", d, o, seconds, 8192);
    run<kIU4>("iu4", d, o, seconds, 16384);
    return 0;
}
