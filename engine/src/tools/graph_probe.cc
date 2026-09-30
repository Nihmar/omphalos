// Measures what a HIP graph is worth for a decode step: the same chain of many
// small, dependent kernels, launched one by one and then replayed as one graph.
//
// A decode step is ~1861 kernels; the trace showed 91 % GPU busy with ~7 ms of
// gaps. This probe answers the question those two numbers cannot: how much of
// the step's cost is per-launch overhead, and how much of *that* a graph
// removes. It is the cheap experiment before the expensive refactor.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#include "kernels/elementwise.hh"

namespace {

double ms_between(hipEvent_t a, hipEvent_t b) {
    float ms = 0.0f;
    (void) hipEventElapsedTime(&ms, a, b);
    return (double) ms;
}

} // namespace

int main(int argc, char ** argv) {
    int64_t n_kernels = 1861;
    int64_t width = 5120;
    int reps = 5;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--kernels") == 0 && i + 1 < argc) {
            n_kernels = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        }
    }

    float * x = nullptr;
    float * zero = nullptr;
    if (hipMalloc(&x, (size_t) width * 4) != hipSuccess ||
        hipMalloc(&zero, (size_t) width * 4) != hipSuccess) {
        std::fprintf(stderr, "hipMalloc failed\n");
        return 1;
    }
    std::vector<float> host((size_t) width, 1.0f);
    (void) hipMemcpy(x, host.data(), (size_t) width * 4, hipMemcpyHostToDevice);
    std::vector<float> zeros((size_t) width, 0.0f);
    (void) hipMemcpy(zero, zeros.data(), (size_t) width * 4, hipMemcpyHostToDevice);

    const auto chain = [&](hipStream_t stream) {
        for (int64_t i = 0; i < n_kernels; ++i) {
            (void) omph::kernels::add_inplace(x, zero, width, stream);
        }
    };

    hipEvent_t a = nullptr;
    hipEvent_t b = nullptr;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);

    // warmup
    chain(nullptr);
    (void) hipDeviceSynchronize();

    double best_free = 1e30;
    for (int r = 0; r < reps; ++r) {
        (void) hipEventRecord(a, nullptr);
        chain(nullptr);
        (void) hipEventRecord(b, nullptr);
        (void) hipDeviceSynchronize();
        const double ms = ms_between(a, b);
        best_free = ms < best_free ? ms : best_free;
    }

    // the same chain, captured
    hipStream_t cap = nullptr;
    if (hipStreamCreate(&cap) != hipSuccess) {
        std::fprintf(stderr, "hipStreamCreate failed\n");
        return 1;
    }
    hipGraph_t graph = nullptr;
    if (hipStreamBeginCapture(cap, hipStreamCaptureModeThreadLocal) != hipSuccess) {
        std::fprintf(stderr, "hipStreamBeginCapture failed\n");
        return 1;
    }
    chain(cap);
    if (hipStreamEndCapture(cap, &graph) != hipSuccess || graph == nullptr) {
        std::fprintf(stderr, "hipStreamEndCapture failed\n");
        return 1;
    }
    hipGraphExec_t exec = nullptr;
    if (hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != hipSuccess) {
        std::fprintf(stderr, "hipGraphInstantiate failed\n");
        return 1;
    }

    double best_graph = 1e30;
    for (int r = 0; r < reps; ++r) {
        (void) hipEventRecord(a, nullptr);
        (void) hipGraphLaunch(exec, nullptr);
        (void) hipEventRecord(b, nullptr);
        (void) hipDeviceSynchronize();
        const double ms = ms_between(a, b);
        best_graph = ms < best_graph ? ms : best_graph;
    }

    std::printf("chain of %lld tiny dependent kernels\n", (long long) n_kernels);
    std::printf("  launched one by one : %8.3f ms  (%.2f us per kernel)\n", best_free,
                1000.0 * best_free / (double) n_kernels);
    std::printf("  replayed as a graph : %8.3f ms  (%.2f us per kernel)\n", best_graph,
                1000.0 * best_graph / (double) n_kernels);
    std::printf("  saved               : %8.3f ms  (%.1f%%)\n", best_free - best_graph,
                100.0 * (best_free - best_graph) / best_free);

    (void) hipGraphExecDestroy(exec);
    (void) hipGraphDestroy(graph);
    (void) hipStreamDestroy(cap);
    (void) hipFree(x);
    (void) hipFree(zero);
    return 0;
}
