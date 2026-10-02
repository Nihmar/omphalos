// LDS read bandwidth with ds_load_b128 (#213 context: the prefill GEMM reads one B operand per WMMA).
//   hipcc -O3 --offload-arch=gfx1200 -o /var/tmp/lds_bw bench/gemm/lds_bw.hip
#include <hip/hip_runtime.h>
#include <cstdio>
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
template <int WAVES>
__global__ __launch_bounds__(WAVES * 32) void k(uint4 * out, int iters) {
    __shared__ uint4 buf[2048];  // 32 KB
    const int tid = threadIdx.x, lane = tid & 31, wv = tid >> 5;
    for (int i = tid; i < 2048; i += blockDim.x) buf[i] = make_uint4(i, i + 1, i + 2, i + 3);
    __syncthreads();
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint4 v = buf[((it * 8 + j + wv) & 63) * 32 + lane];
            acc.x ^= v.x; acc.y += v.y; acc.z ^= v.z; acc.w += v.w;
        }
    }
    if (acc.x == 12345 && acc.y == 7) out[blockIdx.x * blockDim.x + tid] = acc;
}
int main() {
    uint4 * o; (void) hipMalloc(&o, 1 << 24);
    hipEvent_t a, b; (void) hipEventCreate(&a); (void) hipEventCreate(&b);
    const int iters = 4096;
    for (int rep = 0; rep < 3; ++rep) {
        for (int wgs : {16, 32, 64, 128}) {
            (void) hipEventRecord(a);
            k<8><<<wgs, 256>>>(o, iters);
            (void) hipEventRecord(b); (void) hipEventSynchronize(b);
            float ms; (void) hipEventElapsedTime(&ms, a, b);
            const double bytes = (double) wgs * 256 * iters * 8 * 16;
            if (rep == 2) printf("%3d WGs x 8 waves: %.2f TB/s LDS reads (b128)\n", wgs, bytes / ms / 1e9);
        }
    }
}
