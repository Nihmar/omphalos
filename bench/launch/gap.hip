// Inter-kernel gap on one stream (#205): N dependent tiny kernels launched
// directly vs replayed from a captured HIP graph; the time per kernel.
//   hipcc -O3 --offload-arch=gfx1200 -o /var/tmp/gap bench/launch/gap.hip && /var/tmp/gap
#include <hip/hip_runtime.h>

#include <cstdio>

__global__ void tiny(float * p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = p[i] * 0.999f + 1.0f;
}

int main() {
    const int N = 1000, n = 5120;
    float * p;
    (void) hipMalloc(&p, n * 4);
    hipStream_t s;
    (void) hipStreamCreate(&s);
    hipEvent_t a, b;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);
    const auto run = [&]() { for (int i = 0; i < N; ++i) tiny<<<n / 256, 256, 0, s>>>(p, n); };
    for (int w = 0; w < 20; ++w) run();
    (void) hipStreamSynchronize(s);
    float ms = 0;
    for (int rep = 0; rep < 3; ++rep) {
        (void) hipEventRecord(a, s);
        run();
        (void) hipEventRecord(b, s);
        (void) hipEventSynchronize(b);
        (void) hipEventElapsedTime(&ms, a, b);
        std::printf("stream: %.2f us per kernel\n", ms * 1000 / N);
    }
    hipGraph_t g;
    hipGraphExec_t ge;
    (void) hipStreamBeginCapture(s, hipStreamCaptureModeGlobal);
    run();
    (void) hipStreamEndCapture(s, &g);
    (void) hipGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    for (int w = 0; w < 5; ++w) (void) hipGraphLaunch(ge, s);
    (void) hipStreamSynchronize(s);
    for (int rep = 0; rep < 3; ++rep) {
        (void) hipEventRecord(a, s);
        (void) hipGraphLaunch(ge, s);
        (void) hipEventRecord(b, s);
        (void) hipEventSynchronize(b);
        (void) hipEventElapsedTime(&ms, a, b);
        std::printf("graph:  %.2f us per kernel\n", ms * 1000 / N);
    }
    return 0;
}
