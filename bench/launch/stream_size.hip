// Fixed cost per launch of a streaming read (#203): back-to-back launches each
// reading N MiB (rotating over a 2 GiB pool, so neither L2 nor the 32 MB
// Infinity Cache holds the data), time per launch t(N) = a + N / B. The decode
// GEMVs lose ~8-18 us per tensor against the bandwidth at their size: if a pure
// streaming kernel pays the same a, that cost is the launch's (fill and drain of
// the memory pipeline), and only fewer launches remove it.
//   hipcc -O3 --offload-arch=gfx1200 -o /var/tmp/stream_size bench/launch/stream_size.hip && /var/tmp/stream_size
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

__global__ __launch_bounds__(256) void rd(const uint4 * __restrict__ p, const long long n, uint4 * out) {
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    if (acc.x == 0x12345678u && acc.y == 7u) {
        out[0] = acc;
    }
}

int main() {
    const size_t pool = 2ull << 30;
    uint4 * p;
    uint4 * o;
    (void) hipMalloc(&p, pool);
    (void) hipMalloc(&o, 64);
    (void) hipMemset(p, 1, pool);
    hipEvent_t a, b;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);
    const unsigned grid = 16 * 16;  // 16 WGs of 8 waves per WGP
    const auto t0 = std::chrono::steady_clock::now();  // warm-up: the clocks ramp first
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 2.0) {
        rd<<<grid, 256>>>(p, (long long) (pool / 16), o);
        (void) hipDeviceSynchronize();
    }
    std::printf("MiB per launch   us per launch   GB/s   us over N/318.3GB/s\n");
    for (const double mib : {1.0, 2.0, 4.0, 8.0, 12.9, 16.0, 21.5, 32.0, 36.5, 64.0}) {
        const size_t bytes = (size_t) (mib * 1048576) / 16 * 16;
        const long long n = (long long) (bytes / 16);
        const size_t slots = pool / bytes;
        const int launches = 400;
        std::vector<float> v;
        for (int rep = 0; rep < 5; ++rep) {
            (void) hipEventRecord(a);
            for (int i = 0; i < launches; ++i) {
                rd<<<grid, 256>>>(p + (i % slots) * n, n, o);
            }
            (void) hipEventRecord(b);
            (void) hipEventSynchronize(b);
            float ms = 0;
            (void) hipEventElapsedTime(&ms, a, b);
            v.push_back(ms * 1000 / launches);
        }
        std::sort(v.begin(), v.end());
        const double us = v[2];
        std::printf("%14.1f %15.1f %6.1f %18.1f\n", mib, us, bytes / us / 1e3, us - bytes / 318.3e3);
    }
    return 0;
}
