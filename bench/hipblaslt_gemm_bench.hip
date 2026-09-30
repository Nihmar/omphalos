// hipBLASLt FP16 GEMM microbenchmark at the model's projection shapes (M = 512).
//
// Why this exists: the distro rocBLAS is built without hipBLASLt and returns
// ~3.7 TFLOPS fp16 flat (unusable as a ceiling), hipblaslt-bench is not
// packaged, and rocWMMA is not installed. hipBLASLt itself IS installed with
// gfx1200 Tensile kernels, so we link it directly.
//
// build:
//   hipcc -O3 --offload-arch=gfx1200 bench/hipblaslt_gemm_bench.hip \
//       -o bench/build/hipblaslt_gemm_bench -L/opt/rocm/lib -lhipblaslt \
//       -Wl,-rpath,/opt/rocm/lib
//
// usage: hipblaslt_gemm_bench <label> <M> <N> <K> [iters]
// prints one CSV row: label,m,n,k,median_ms,best_ms,worst_ms,tflops,workspace_mb

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_HIP(x)                                                                     \
    do {                                                                                 \
        hipError_t err_ = (x);                                                           \
        if (err_ != hipSuccess) {                                                        \
            std::fprintf(stderr, "hip error %s (%s:%d)\n", hipGetErrorString(err_),      \
                         __FILE__, __LINE__);                                            \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

#define CHECK_BLT(x)                                                                     \
    do {                                                                                 \
        hipblasStatus_t st_ = (x);                                                       \
        if (st_ != HIPBLAS_STATUS_SUCCESS) {                                             \
            std::fprintf(stderr, "hipblaslt error %d (%s:%d)\n", (int)st_, __FILE__,     \
                         __LINE__);                                                      \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <label> <M> <N> <K> [iters]\n", argv[0]);
        return 1;
    }
    const char* label = argv[1];
    const int M = std::atoi(argv[2]);
    const int N = std::atoi(argv[3]);
    const int K = std::atoi(argv[4]);
    const int iters = (argc > 5) ? std::atoi(argv[5]) : 20;

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "device: %s | shape %s: M=%d N=%d K=%d\n", prop.name, label, M, N,
                 K);

    hipblasLtHandle_t handle;
    CHECK_BLT(hipblasLtCreate(&handle));

    const size_t aBytes = (size_t)M * K * sizeof(__half);
    const size_t bBytes = (size_t)K * N * sizeof(__half);
    const size_t cBytes = (size_t)M * N * sizeof(__half);

    std::vector<__half> hA((size_t)M * K, __float2half(0.01f));
    std::vector<__half> hB((size_t)K * N, __float2half(0.01f));

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    CHECK_HIP(hipMalloc(&dA, aBytes));
    CHECK_HIP(hipMalloc(&dB, bBytes));
    CHECK_HIP(hipMalloc(&dC, cBytes));
    CHECK_HIP(hipMemcpy(dA, hA.data(), aBytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB.data(), bBytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(dC, 0, cBytes));

    hipblasLtMatmulDesc_t op;
    CHECK_BLT(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t opN = HIPBLAS_OP_N;
    CHECK_BLT(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &opN,
                                              sizeof(opN)));
    CHECK_BLT(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &opN,
                                              sizeof(opN)));

    hipblasLtMatrixLayout_t aDesc, bDesc, cDesc, dDesc;
    CHECK_BLT(hipblasLtMatrixLayoutCreate(&aDesc, HIP_R_16F, M, K, M));
    CHECK_BLT(hipblasLtMatrixLayoutCreate(&bDesc, HIP_R_16F, K, N, K));
    CHECK_BLT(hipblasLtMatrixLayoutCreate(&cDesc, HIP_R_16F, M, N, M));
    CHECK_BLT(hipblasLtMatrixLayoutCreate(&dDesc, HIP_R_16F, M, N, M));

    hipblasLtMatmulPreference_t pref;
    CHECK_BLT(hipblasLtMatmulPreferenceCreate(&pref));
    size_t wsSize = 256ull * 1024 * 1024;
    CHECK_BLT(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));

    hipblasLtMatmulHeuristicResult_t heur{};
    int returned = 0;
    CHECK_BLT(hipblasLtMatmulAlgoGetHeuristic(handle, op, aDesc, bDesc, cDesc, dDesc, pref,
                                              1, &heur, &returned));
    if (returned < 1) {
        std::fprintf(stderr, "no hipblaslt algorithm for this shape\n");
        return 1;
    }

    void* ws = nullptr;
    CHECK_HIP(hipMalloc(&ws, wsSize));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto run = [&]() {
        CHECK_BLT(hipblasLtMatmul(handle, op, &alpha, dA, aDesc, dB, bDesc, &beta, dC, cDesc,
                                  dC, dDesc, &heur.algo, ws, wsSize, nullptr));
    };

    run();
    run();
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    CHECK_HIP(hipEventCreate(&t0));
    CHECK_HIP(hipEventCreate(&t1));
    std::vector<float> ms(iters);
    for (int i = 0; i < iters; ++i) {
        CHECK_HIP(hipEventRecord(t0));
        run();
        CHECK_HIP(hipEventRecord(t1));
        CHECK_HIP(hipEventSynchronize(t1));
        float elapsed = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&elapsed, t0, t1));
        ms[i] = elapsed;
    }
    std::sort(ms.begin(), ms.end());

    const double flop = 2.0 * (double)M * N * K;
    const double median = ms[iters / 2];
    std::printf("%s,%d,%d,%d,%.3f,%.3f,%.3f,%.1f,%.1f\n", label, M, N, K, median, ms.front(),
                ms.back(), flop / (median / 1000.0) / 1e12, heur.workspaceSize / 1048576.0);
    return 0;
}
