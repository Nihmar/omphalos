// LDS read bandwidth with ds_load_b128 (#213, #203 phase 0: the prefill GEMM reads one B operand per WMMA).
//   hipcc -O3 --offload-arch=gfx1200 -o /var/tmp/lds_bw bench/gemm/lds_bw.hip
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
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
    // sustained: launches for ~3 s, the mean of the second half (the clocks ramp
    // up first; a cold single launch reads ~30 % low)
    uint4 * o; (void) hipMalloc(&o, 1 << 24);
    hipEvent_t a, b; (void) hipEventCreate(&a); (void) hipEventCreate(&b);
    const int iters = 4096, wgs = 128;
    std::vector<double> v;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 3.0) {
        (void) hipEventRecord(a);
        k<8><<<wgs, 256>>>(o, iters);
        (void) hipEventRecord(b); (void) hipEventSynchronize(b);
        float ms; (void) hipEventElapsedTime(&ms, a, b);
        v.push_back((double) wgs * 256 * iters * 8 * 16 / ms / 1e9);
    }
    double sum = 0;
    for (size_t i = v.size() / 2; i < v.size(); ++i) sum += v[i];
    printf("%d WGs x 8 waves: %.2f TB/s LDS reads (b128), sustained over %zu launches\n", wgs,
           sum / (double) (v.size() - v.size() / 2), v.size());
}
