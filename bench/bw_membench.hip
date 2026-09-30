// Streaming read-bandwidth microbenchmark for gfx1200 (PLAN.md §7, Milestone 0).
//
// Reads a large buffer with 128-bit vector loads and reports achieved GB/s.
// This number is the real 100% for decode (weights are read once per token).
//
// build: hipcc -O3 --offload-arch=gfx1200 bench/bw_membench.hip -o bench/build/bw_membench
// usage: bench/build/bw_membench [buffer_gib] [passes]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr)                                                                  \
    do {                                                                                 \
        hipError_t err_ = (expr);                                                        \
        if (err_ != hipSuccess) {                                                        \
            std::fprintf(stderr, "HIP error: %s (%s:%d)\n", hipGetErrorString(err_),     \
                         __FILE__, __LINE__);                                            \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

// Grid-stride sum of float4, unrolled 4x to keep several 128-bit loads in flight.
__global__ void read_sum_kernel(const float4* __restrict__ in, float* __restrict__ out,
                                unsigned long long n4) {
    const unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    unsigned long long i = tid;
    for (; i + 3 * stride < n4; i += 4 * stride) {
        const float4 a = in[i];
        const float4 b = in[i + stride];
        const float4 c = in[i + 2 * stride];
        const float4 d = in[i + 3 * stride];
        acc0 += a.x + a.y + a.z + a.w;
        acc1 += b.x + b.y + b.z + b.w;
        acc2 += c.x + c.y + c.z + c.w;
        acc3 += d.x + d.y + d.z + d.w;
    }
    for (; i < n4; i += stride) {
        const float4 a = in[i];
        acc0 += a.x + a.y + a.z + a.w;
    }

    __shared__ float sacc[256];
    sacc[threadIdx.x] = acc0 + acc1 + acc2 + acc3;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int k = 0; k < (int)blockDim.x; ++k) total += sacc[k];
        out[blockIdx.x] = total;  // keeps the loads live
    }
}

int main(int argc, char** argv) {
    const double gib = (argc > 1) ? std::atof(argv[1]) : 10.0;
    const int passes = (argc > 2) ? std::atoi(argv[2]) : 10;
    const unsigned long long bytes = (unsigned long long)(gib * 1ull * (1ull << 30));
    const unsigned long long n4 = bytes / sizeof(float4);

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    float4* buf = nullptr;
    float* out = nullptr;
    HIP_CHECK(hipMalloc(&buf, n4 * sizeof(float4)));
    HIP_CHECK(hipMalloc(&out, 65536 * sizeof(float)));
    HIP_CHECK(hipMemset(buf, 0x3f, n4 * sizeof(float4)));

    const int block = 256;
    const int grid = prop.multiProcessorCount * 16;
    std::fprintf(stderr, "device: %s | CUs: %d | grid: %d x %d | buffer: %.2f GiB\n",
                 prop.name, prop.multiProcessorCount, grid, block, gib);

    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));

    read_sum_kernel<<<grid, block>>>(buf, out, n4);  // warmup
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> ms(passes);
    for (int p = 0; p < passes; ++p) {
        HIP_CHECK(hipEventRecord(t0));
        read_sum_kernel<<<grid, block>>>(buf, out, n4);
        HIP_CHECK(hipEventRecord(t1));
        HIP_CHECK(hipEventSynchronize(t1));
        float elapsed = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed, t0, t1));
        ms[p] = elapsed;
    }
    std::sort(ms.begin(), ms.end());

    const double bytes_read = (double)n4 * sizeof(float4);
    auto gbps = [&](double t_ms) { return bytes_read / (t_ms / 1000.0) / 1e9; };
    const double best = gbps(ms.front());
    const double median = gbps(ms[passes / 2]);
    const double worst = gbps(ms.back());

    std::printf("device,buffer_gib,bytes_read,passes,best_ms,median_ms,worst_ms,"
                "best_gb_s,median_gb_s,worst_gb_s\n");
    std::printf("%s,%.2f,%llu,%d,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f\n", prop.name, gib, bytes,
                passes, ms.front(), ms[passes / 2], ms.back(), best, median, worst);

    HIP_CHECK(hipFree(buf));
    HIP_CHECK(hipFree(out));
    return 0;
}
