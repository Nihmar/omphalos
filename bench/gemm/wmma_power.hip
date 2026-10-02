// Pure WMMAs on constant vs random register operands, ~5 s each: with real data the
// power cap (160 W) holds the matrix units under the constant-operand figure.
//   hipcc -O3 --offload-arch=gfx1200 -o /var/tmp/wmma_power bench/gemm/wmma_power.hip && /var/tmp/wmma_power 1
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
// pure WMMAs on 4 A x 4 B register operands read once from `src`
__global__ __launch_bounds__(256) void k(const half8 * src, float * out, int iters) {
    const int lane = threadIdx.x & 31;
    half8 a[4], b[4];
    for (int i = 0; i < 4; ++i) { a[i] = src[(blockIdx.x * 8 + i) % 4096 * 32 + lane]; b[i] = src[(blockIdx.x * 8 + 4 + i) % 4096 * 32 + lane]; }
    float8 acc[4][4] = {};
    for (int it = 0; it < iters; ++it)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
    float s = 0; for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) s += acc[i][j][0];
    if (s == 1234.5f) out[0] = s;
}
int main(int argc, char ** argv) {
    const int mode = atoi(argv[1]);  // 0 constant, 1 random
    std::vector<_Float16> h(4096 * 32 * 8);
    std::mt19937 rng(1);
    for (auto & v : h) v = mode ? (_Float16) (((int) (rng() % 2001) - 1000) / 1000.0f) : (_Float16) 0.5f;
    half8 * d; float * o; (void) hipMalloc(&d, h.size() * 2); (void) hipMalloc(&o, 4);
    (void) hipMemcpy(d, h.data(), h.size() * 2, hipMemcpyHostToDevice);
    hipEvent_t e0, e1; (void) hipEventCreate(&e0); (void) hipEventCreate(&e1);
    const int wgs = 256, iters = 20000;
    for (int rep = 0; rep < 80; ++rep) {
        (void) hipEventRecord(e0);
        k<<<wgs, 256>>>(d, o, iters);
        (void) hipEventRecord(e1); (void) hipEventSynchronize(e1);
        float ms; (void) hipEventElapsedTime(&ms, e0, e1);
        if (rep % 20 == 19) printf("%s operands: %.1f TFLOPS (%.0f ms)\n", mode ? "random" : "constant", (double) wgs * 8 * iters * 16 * 8192 / ms / 1e9, ms);
    }
}
