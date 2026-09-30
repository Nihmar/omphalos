// HIP device probe and self test (PLAN.md §9, §11).
#include "runtime/device.hh"

#include <hip/hip_runtime.h>

#include <cstdio>

namespace {

__global__ void self_test_kernel(float * out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (float) i * 0.5f;
    }
}

} // namespace

namespace omph::runtime {

int device_count(void) {
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess ? count : 0;
}

int device_info(int index, char * out, size_t out_size) {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, index) != hipSuccess) {
        return -1;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (hipSetDevice(index) != hipSuccess || hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        free_bytes = total_bytes = 0;
    }
    std::snprintf(out, out_size, "%s | %s | %d CUs | VRAM %.2f GiB (%.2f GiB free)", prop.name,
                  prop.gcnArchName, prop.multiProcessorCount, (double) total_bytes / 1073741824.0,
                  (double) free_bytes / 1073741824.0);
    return 0;
}

int self_test(void) {
    constexpr int n = 64;
    float * dev = nullptr;
    if (hipMalloc(&dev, n * sizeof(float)) != hipSuccess) {
        return -1;
    }
    self_test_kernel<<<1, n>>>(dev, n);
    if (hipGetLastError() != hipSuccess) {
        (void) hipFree(dev);
        return -2;
    }
    float host[n] = {};
    const hipError_t err = hipMemcpy(host, dev, sizeof(host), hipMemcpyDeviceToHost);
    (void) hipFree(dev);
    if (err != hipSuccess) {
        return -3;
    }
    for (int i = 0; i < n; ++i) {
        if (host[i] != (float) i * 0.5f) {
            return -4;
        }
    }
    return 0;
}

} // namespace omph::runtime
