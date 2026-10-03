// Do WMMAs and VALU work overlap on gfx1200? (#203 P1, the int8 prefill GEMM.)
// An int8 GEMM on IQ weights rescales its int32 accumulators every sub-block
// (32 k = 2 WMMAs): ~16-24 VALU ops per 2 WMMAs. Each wave here runs 4 x 4
// WMMAs on register operands (random data) and N independent f32 FMAs per WMMA
// on registers of its own; if the rate holds as N grows, the VALU work hides
// under the matrix units, if it falls, the two serialize.
//   hipcc -O3 --offload-arch=gfx1200 -o /var/tmp/wmma_valu bench/gemm/wmma_valu.hip && /var/tmp/wmma_valu
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef int i8v __attribute__((ext_vector_type(8)));
typedef int i2v __attribute__((ext_vector_type(2)));

template <bool INT8, int N>
__global__ __launch_bounds__(256) void k(const uint4 * src, float * out, const int iters) {
    const int lane = threadIdx.x & 31;
    uint4 a[4], b[4];
    for (int i = 0; i < 4; ++i) {
        a[i] = src[((blockIdx.x * 8 + i) % 1024) * 32 + lane];
        b[i] = src[((blockIdx.x * 8 + 4 + i) % 1024) * 32 + lane];
    }
    float v[8];
    for (int i = 0; i < 8; ++i) v[i] = (float) (lane + i) * 1e-3f;
    const float m = 0.999f, c = 1e-4f;
    float s = 0.0f;
    if constexpr (INT8) {
        i8v acc[4][4] = {};
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, i2v{(int) a[i].x, (int) a[i].y}, true, i2v{(int) b[j].x, (int) b[j].y}, acc[i][j], false);
#pragma unroll
                    for (int n = 0; n < N; ++n) v[n % 8] = fmaf(v[n % 8], m, c);
                }
            }
        }
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) s += (float) acc[i][j][0];
    } else {
        f8 acc[4][4] = {};
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        __builtin_bit_cast(h8, a[i]), __builtin_bit_cast(h8, b[j]), acc[i][j]);
#pragma unroll
                    for (int n = 0; n < N; ++n) v[n % 8] = fmaf(v[n % 8], m, c);
                }
            }
        }
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) s += acc[i][j][0];
    }
    for (int i = 0; i < 8; ++i) s += v[i];
    if (s == 1234.5f) out[0] = s;
}

template <bool INT8, int N>
void run(const uint4 * d, float * o) {
    const int wgs = 256, iters = 2000;
    hipEvent_t e0, e1;
    (void) hipEventCreate(&e0);
    (void) hipEventCreate(&e1);
    std::vector<double> r;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 3.0) {
        (void) hipEventRecord(e0);
        k<INT8, N><<<wgs, 256>>>(d, o, iters);
        (void) hipEventRecord(e1);
        (void) hipEventSynchronize(e1);
        float ms = 0;
        (void) hipEventElapsedTime(&ms, e0, e1);
        r.push_back((double) wgs * 8 * iters * 16 * 8192 / ms / 1e9);
    }
    double sum = 0;
    for (size_t i = r.size() / 2; i < r.size(); ++i) sum += r[i];
    const double rate = sum / (double) (r.size() - r.size() / 2);
    std::printf("%s + %2d FMAs per WMMA: %6.1f T(FL)OPS of WMMA\n", INT8 ? "iu8" : "f16", N, rate);
    std::fflush(stdout);
}

int main() {
    std::vector<uint32_t> h(1024 * 32 * 4);
    std::mt19937 rng(203);
    for (auto & v : h) v = rng() & 0x3BFF3BFFu;
    uint4 * d;
    float * o;
    (void) hipMalloc(&d, h.size() * 4);
    (void) hipMalloc(&o, 4);
    (void) hipMemcpy(d, h.data(), h.size() * 4, hipMemcpyHostToDevice);
    run<false, 0>(d, o);  // warm-up
    run<false, 0>(d, o);
    run<false, 4>(d, o);
    run<false, 8>(d, o);
    run<false, 16>(d, o);
    run<true, 0>(d, o);
    run<true, 4>(d, o);
    run<true, 8>(d, o);
    run<true, 12>(d, o);
    run<true, 16>(d, o);
    return 0;
}
