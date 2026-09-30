// isa_probe — what integer/fp16 dot hardware is actually reachable on this GPU.
//
// Context (PLAN.md §10.1, §10.5): the scalar builtin __builtin_amdgcn_sdot4
// does not compile for gfx1200 ("needs target feature dot1-insts"), which is
// how LLVM gates it — but the assembler accepts v_dot4_i32_i8 for gfx1200, so
// the instruction may still be there. This probe runs it through inline asm.
//
// It also documents the per-target instruction availability:
//
//   instruction                  gfx1100   gfx1200
//   v_dot4_i32_i8                  yes       yes
//   v_dot2_f32_f16                 yes       yes
//   v_wmma_i32_16x16x16_iu8        yes        NO     <- no int8 matrix tile
//   v_wmma_f32_16x16x16_f16         NO       yes
//
// build: hipcc --offload-arch=gfx1200 -O3 isa_probe.cpp -o isa_probe
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdint>

namespace {

__global__ void dot4_kernel(const uint32_t * a, const uint32_t * b, int * c) {
    const uint32_t va = a[0];
    const uint32_t vb = b[0];
    int acc = 0;
    asm volatile("v_dot4_i32_i8 %0, %1, %2, %3"
                 : "=v"(acc)
                 : "v"(va), "v"(vb), "v"(acc));
    c[0] = acc;
}

// [1,2,3,4] . [5,6,7,8] = 70
constexpr uint32_t pack4(const int8_t a, const int8_t b, const int8_t c, const int8_t d) {
    return (uint32_t) (uint8_t) a | ((uint32_t) (uint8_t) b << 8) | ((uint32_t) (uint8_t) c << 16) |
           ((uint32_t) (uint8_t) d << 24);
}

// --- raw instruction throughput (independent chains for ILP) ---------------

constexpr int kChains = 16;
constexpr int kIters = 200000;

__global__ void tp_dot4(const uint32_t * in, int * out, const int iters) {
    const uint32_t a = in[0];
    const uint32_t b = in[1];
    int acc[kChains] = {};
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kChains; ++j) {
            asm volatile("v_dot4_i32_i8 %0, %1, %2, %3"
                         : "=v"(acc[j])
                         : "v"(a), "v"(b), "v"(acc[j]));
        }
    }
    int s = 0;
#pragma unroll
    for (int j = 0; j < kChains; ++j) s += acc[j];
    out[threadIdx.x] = s;
}

__global__ void tp_f16(const __half2 * in, float * out, const int iters) {
    const __half2 a = in[0];
    const __half2 b = in[1];
    __half2 acc[kChains];
#pragma unroll
    for (int j = 0; j < kChains; ++j) {
        acc[j] = __float2half2_rn(0.0f);
    }
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kChains; ++j) {
            acc[j] = __hfma2(a, b, acc[j]);
        }
    }
    float r = 0.0f;
#pragma unroll
    for (int j = 0; j < kChains; ++j) {
        r += __low2float(acc[j]) + __high2float(acc[j]);
    }
    out[threadIdx.x] = r;
}

__global__ void tp_f32(const float * in, float * out, const int iters) {
    const float a = in[0];
    const float b = in[1];
    float acc[kChains] = {};
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kChains; ++j) {
            acc[j] = fmaf(a, b, acc[j]);
        }
    }
    int s = 0;
#pragma unroll
    for (int j = 0; j < kChains; ++j) s += acc[j];
    out[threadIdx.x] = s;
}

template <typename F>
double run_throughput(F launch, const double ops_per_iter_per_lane, const int lanes,
                      const int blocks) {
    hipEvent_t e0, e1;
    (void) hipEventCreate(&e0);
    (void) hipEventCreate(&e1);
    launch(8);
    (void) hipDeviceSynchronize();
    (void) hipEventRecord(e0, nullptr);
    launch(kIters);
    (void) hipEventRecord(e1, nullptr);
    (void) hipDeviceSynchronize();
    float ms = 0.0f;
    (void) hipEventElapsedTime(&ms, e0, e1);
    return ops_per_iter_per_lane * (double) kIters / (ms * 1e-3) * lanes * blocks / 1e12;
}

} // namespace

int main() {
    hipDeviceProp_t prop{};
    (void) hipGetDeviceProperties(&prop, 0);
    std::printf("device: %s (%s)\n", prop.name, prop.gcnArchName);

    const uint32_t ha = pack4(1, 2, 3, 4);
    const uint32_t hb = pack4(5, 6, 7, 8);
    const uint32_t host[2] = {ha, hb};
    uint32_t * dev = nullptr;
    int * out = nullptr;
    if (hipMalloc(&dev, 2 * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&out, sizeof(int)) != hipSuccess ||
        hipMemcpy(dev, host, sizeof(host), hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("setup failed\n");
        return 1;
    }
    dot4_kernel<<<1, 1>>>(dev, dev + 1, out);
    const hipError_t rc = hipDeviceSynchronize();
    if (rc != hipSuccess) {
        std::printf("v_dot4_i32_i8: kernel failed (%s)\n", hipGetErrorString(rc));
        return 1;
    }
    int got = 0;
    (void) hipMemcpy(&got, out, sizeof(int), hipMemcpyDeviceToHost);
    std::printf("v_dot4_i32_i8 via inline asm: %d (expected 70) -> %s\n", got,
                got == 70 ? "works" : "WRONG");

    // ---- raw instruction throughput
    const int lanes = 256;
    const int blocks = 32 * 8;  // 8 workgroups per CU on 32 CUs
    const double per_iter = (double) kChains * (double) lanes * (double) blocks;
    const double i8_tops =
        run_throughput([&](const int iters) { tp_dot4<<<blocks, lanes>>>(dev, out, iters); },
                       4.0 * per_iter, 1, 1);
    const double f16_inputs[2] = {1.0f, 1.0f};
    float * fdev = nullptr;
    (void) hipMalloc(&fdev, 64);
    (void) hipMemcpy(fdev, f16_inputs, sizeof(f16_inputs), hipMemcpyHostToDevice);
    const double f16_tflops = run_throughput(
        [&](const int iters) {
            tp_f16<<<blocks, lanes>>>((const __half2 *) fdev, (float *) out, iters);
        },
        4.0 * per_iter, 1, 1);
    const double f32_tflops = run_throughput(
        [&](const int iters) { tp_f32<<<blocks, lanes>>>(fdev, (float *) out, iters); },
        2.0 * per_iter, 1, 1);
    std::printf("throughput (raw, %d chains x %d lanes x %d workgroups):\n", kChains, lanes,
                blocks);
    std::printf("  v_dot4_i32_i8     : %6.1f TOPS\n", i8_tops);
    std::printf("  __hfma2 (half2)   : %6.1f TFLOPS\n", f16_tflops);
    std::printf("  fmaf (f32)        : %6.1f TFLOPS\n", f32_tflops);
    return got == 70 ? 0 : 1;
}
