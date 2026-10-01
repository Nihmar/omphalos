// raw WMMA throughput on gfx1200: independent 16x16x16 f16 -> f32 (and f16 -> f16) tiles in registers
#include <hip/hip_runtime.h>
#include <cstdio>
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
template <int NACC, int ITERS>
__global__ __launch_bounds__(256) void wmma_f32(float * out, float seed) {
    half8 a, b;
    for (int i = 0; i < 8; ++i) { a[i] = (_Float16) (seed * (threadIdx.x + i)); b[i] = (_Float16) (seed * (i - threadIdx.x)); }
    float8 c[NACC];
    for (int j = 0; j < NACC; ++j) for (int i = 0; i < 8; ++i) c[j][i] = 0.f;
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int j = 0; j < NACC; ++j) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c[j]);
    }
    float s = 0; for (int j = 0; j < NACC; ++j) for (int i = 0; i < 8; ++i) s += c[j][i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <int NACC, int ITERS>
__global__ __launch_bounds__(256) void wmma_f16(float * out, float seed) {
    half8 a, b;
    for (int i = 0; i < 8; ++i) { a[i] = (_Float16) (seed * (threadIdx.x + i)); b[i] = (_Float16) (seed * (i - threadIdx.x)); }
    half8 c[NACC];
    for (int j = 0; j < NACC; ++j) for (int i = 0; i < 8; ++i) c[j][i] = 0;
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int j = 0; j < NACC; ++j) c[j] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a, b, c[j]);
    }
    float s = 0; for (int j = 0; j < NACC; ++j) for (int i = 0; i < 8; ++i) s += (float) c[j][i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <typename F> void run(const char * name, F launch, double flops) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    launch(); hipDeviceSynchronize();
    hipEventRecord(a); for (int i = 0; i < 5; ++i) launch(); hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    printf("%-28s %.1f TFLOPS\n", name, flops * 5 / (ms * 1e-3) / 1e12);
}
int main() {
    float * o; hipMalloc(&o, 4096 * 256 * 4);
    constexpr int IT = 4096, BLOCKS = 4096;
    const double per = 2.0 * 16 * 16 * 16;  // flops per wmma
    const double waves = BLOCKS * 8.0;
    run("f32 acc, 4 chains", [&] { wmma_f32<4, IT><<<BLOCKS, 256>>>(o, 0.001f); }, waves * IT * 4 * per);
    run("f32 acc, 8 chains", [&] { wmma_f32<8, IT><<<BLOCKS, 256>>>(o, 0.001f); }, waves * IT * 8 * per);
    run("f16 acc, 8 chains", [&] { wmma_f16<8, IT><<<BLOCKS, 256>>>(o, 0.001f); }, waves * IT * 8 * per);
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("CUs %d, clock %d MHz\n", p.multiProcessorCount, p.clockRate / 1000);
}
