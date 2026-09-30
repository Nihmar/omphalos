// hipBLASLt GEMM microbenchmark at the model's projection shapes (M = 512).
//
// Why this exists: the distro rocBLAS is built without hipBLASLt and returns
// ~3.7 TFLOPS flat (unusable as a ceiling), hipblaslt-bench is not packaged,
// and rocWMMA is not installed. hipBLASLt itself IS installed with gfx1200
// Tensile kernels, so we link it directly.
//
// build:
//   hipcc -O3 --offload-arch=gfx1200 bench/hipblaslt_gemm_bench.hip \
//       -o bench/build/hipblaslt_gemm_bench -L/opt/rocm/lib -lhipblaslt \
//       -Wl,-rpath,/opt/rocm/lib
//
// usage: hipblaslt_gemm_bench <label> <M> <N> <K> [iters] [fp16|int8]
// prints one CSV row:
//   label,m,n,k,dtype,median_ms,best_ms,worst_ms,tflops,workspace_mb
// (for int8, "tflops" counts 2*M*N*K integer ops per second)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        std::fprintf(stderr, "usage: %s <label> <M> <N> <K> [iters] [fp16|int8]\n",
                     argv[0]);
        return 1;
    }
    const char* label = argv[1];
    const int M = std::atoi(argv[2]);
    const int N = std::atoi(argv[3]);
    const int K = std::atoi(argv[4]);
    const int iters = (argc > 5) ? std::atoi(argv[5]) : 20;
    const bool int8 = (argc > 6) && std::strcmp(argv[6], "int8") == 0;
    const char* dtype = int8 ? "int8" : "fp16";

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "device: %s | %s %s: M=%d N=%d K=%d\n", prop.name, label, dtype, M,
                 N, K);

    hipblasLtHandle_t handle;
    CHECK_BLT(hipblasLtCreate(&handle));

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    hipblasLtMatmulDesc_t op;
    hipblasLtMatrixLayout_t aDesc, bDesc, cDesc, dDesc;
    float alpha_f = 1.0f;
    float beta_f = 0.0f;
    std::int32_t alpha_i = 1;
    std::int32_t beta_i = 0;
    const void* alpha = nullptr;
    const void* beta = nullptr;
    hipDataType abType = HIP_R_16F;
    hipDataType cdType = HIP_R_16F;

    if (!int8) {
        const size_t aBytes = (size_t)M * K * sizeof(__half);
        const size_t bBytes = (size_t)K * N * sizeof(__half);
        const size_t cBytes = (size_t)M * N * sizeof(__half);
        std::vector<__half> hA((size_t)M * K, __float2half(0.01f));
        std::vector<__half> hB((size_t)K * N, __float2half(0.01f));
        CHECK_HIP(hipMalloc(&dA, aBytes));
        CHECK_HIP(hipMalloc(&dB, bBytes));
        CHECK_HIP(hipMalloc(&dC, cBytes));
        CHECK_HIP(hipMemcpy(dA, hA.data(), aBytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(dB, hB.data(), bBytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemset(dC, 0, cBytes));
        abType = HIP_R_16F;
        cdType = HIP_R_16F;
        alpha = &alpha_f;
        beta = &beta_f;
        CHECK_BLT(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    } else {
        const size_t aBytes = (size_t)M * K;
        const size_t bBytes = (size_t)K * N;
        const size_t cBytes = (size_t)M * N * sizeof(std::int32_t);
        std::vector<std::int8_t> hA((size_t)M * K, 1);
        std::vector<std::int8_t> hB((size_t)K * N, 1);
        CHECK_HIP(hipMalloc(&dA, aBytes));
        CHECK_HIP(hipMalloc(&dB, bBytes));
        CHECK_HIP(hipMalloc(&dC, cBytes));
        CHECK_HIP(hipMemcpy(dA, hA.data(), aBytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(dB, hB.data(), bBytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemset(dC, 0, cBytes));
        abType = HIP_R_8I;
        cdType = HIP_R_32I;
        alpha = &alpha_i;
        beta = &beta_i;
        CHECK_BLT(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32I, HIP_R_32I));
    }

    hipblasOperation_t opN = HIPBLAS_OP_N;
    CHECK_BLT(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &opN,
                                              sizeof(opN)));
    CHECK_BLT(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &opN,
                                              sizeof(opN)));

    CHECK_BLT(hipblasLtMatrixLayoutCreate(&aDesc, abType, M, K, M));
    CHECK_BLT(hipblasLtMatrixLayoutCreate(&bDesc, abType, K, N, K));
    CHECK_BLT(hipblasLtMatrixLayoutCreate(&cDesc, cdType, M, N, M));
    CHECK_BLT(hipblasLtMatrixLayoutCreate(&dDesc, cdType, M, N, M));

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

    auto run = [&]() {
        CHECK_BLT(hipblasLtMatmul(handle, op, alpha, dA, aDesc, dB, bDesc, beta, dC, cDesc,
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
    std::printf("%s,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.1f,%.1f\n", label, M, N, K, dtype, median,
                ms.front(), ms.back(), flop / (median / 1000.0) / 1e12,
                heur.workspaceSize / 1048576.0);
    return 0;
}
