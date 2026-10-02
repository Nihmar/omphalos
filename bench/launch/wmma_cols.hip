// Is a WMMA output column independent of the other columns of B (#206)?
// D = A x B with B's column 0 fixed and the other 15 columns random vs zero:
// column 0 of D must match bit for bit for a WMMA-based GEMV whose NT-token
// rows equal its single-token rows.
//   hipcc -O3 --offload-arch=gfx1200 -o /var/tmp/wmma_cols bench/launch/wmma_cols.hip && /var/tmp/wmma_cols
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

// A 16 x 32 (two k16 steps), B 32 x 16; lane layouts as in the engine's kernels
__global__ void mm(const _Float16 * a, const _Float16 * b, float * d) {
    const int lane = threadIdx.x;
    float8 acc = {};
    for (int s = 0; s < 2; ++s) {
        half8 av, bv;
        for (int i = 0; i < 8; ++i) {
            av[i] = a[(lane & 15) * 32 + s * 16 + (lane >> 4) * 8 + i];   // row lane%16, k ...
            bv[i] = b[(s * 16 + (lane >> 4) * 8 + i) * 16 + (lane & 15)];  // col lane%16, k ...
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, acc);
    }
    for (int v = 0; v < 8; ++v) d[((lane >> 4) * 8 + v) * 16 + (lane & 15)] = acc[v];
}

int main() {
    std::mt19937 rng(206);
    std::uniform_real_distribution<float> u(-2.0f, 2.0f);
    int bad = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        _Float16 ha[16 * 32], hb1[32 * 16], hb2[32 * 16];
        for (auto & x : ha) x = (_Float16) u(rng);
        for (int k = 0; k < 32; ++k) {
            for (int j = 0; j < 16; ++j) {
                const _Float16 c0 = (_Float16) u(rng);
                hb1[k * 16 + j] = j == 0 ? c0 : (_Float16) u(rng);
                hb2[k * 16 + j] = j == 0 ? hb1[k * 16] : (_Float16) 0.0f;
            }
            hb2[k * 16] = hb1[k * 16];
        }
        _Float16 *a, *b;
        float * d;
        (void) hipMalloc(&a, sizeof ha);
        (void) hipMalloc(&b, sizeof hb1);
        (void) hipMalloc(&d, 256 * 4);
        float d1[256], d2[256];
        (void) hipMemcpy(a, ha, sizeof ha, hipMemcpyHostToDevice);
        (void) hipMemcpy(b, hb1, sizeof hb1, hipMemcpyHostToDevice);
        mm<<<1, 32>>>(a, b, d);
        (void) hipMemcpy(d1, d, sizeof d1, hipMemcpyDeviceToHost);
        (void) hipMemcpy(b, hb2, sizeof hb2, hipMemcpyHostToDevice);
        mm<<<1, 32>>>(a, b, d);
        (void) hipMemcpy(d2, d, sizeof d2, hipMemcpyDeviceToHost);
        for (int i = 0; i < 16; ++i) {
            if (std::memcmp(&d1[i * 16], &d2[i * 16], 4) != 0) ++bad;
        }
        (void) hipFree(a);
        (void) hipFree(b);
        (void) hipFree(d);
    }
    std::printf("column 0 differs in %d of %d rows\n", bad, 2000 * 16);
    return bad != 0;
}
