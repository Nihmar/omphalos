#include "runtime/matmul.hh"

#include <cstdio>

namespace omph::runtime {
namespace {

constexpr hipblasOperation_t kTransA = HIPBLAS_OP_T;
constexpr hipblasOperation_t kTransB = HIPBLAS_OP_N;

bool check(const hipblasStatus_t status, const char * what) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "hipblaslt: %s failed (%d)\n", what, (int) status);
        return false;
    }
    return true;
}

} // namespace

Linear::Linear() {
    workspace_size_ = 64ull * 1024 * 1024;
    if (!check(hipblasLtCreate(&handle_), "create")) {
        return;
    }
    if (hipMalloc(&workspace_, workspace_size_) != hipSuccess) {
        workspace_ = nullptr;
    }
}

Linear::~Linear() {
    if (workspace_ != nullptr) {
        (void) hipFree(workspace_);
    }
    if (handle_ != nullptr) {
        (void) hipblasLtDestroy(handle_);
    }
}

bool Linear::run(const void * w, const void * x, float * y, const int64_t out_features,
                 const int64_t in_features, const int64_t tokens) {
    if (handle_ == nullptr || workspace_ == nullptr) {
        return false;
    }

    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;
    hipblasLtMatrixLayout_t d = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    bool ok = check(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F), "desc");

    // A = W (row-major out x in) == column-major (in x out), op(A) = A^T = (out x in)
    if (ok) ok = check(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &kTransA, sizeof(kTransA)), "transA");
    // B = x (row-major tokens x in) == column-major (in x tokens), no transpose
    if (ok) ok = check(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &kTransB, sizeof(kTransB)), "transB");

    if (ok) ok = check(hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, in_features, out_features, in_features), "layout A");
    if (ok) ok = check(hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, in_features, tokens, in_features), "layout B");
    if (ok) ok = check(hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, out_features, tokens, out_features), "layout C");
    if (ok) ok = check(hipblasLtMatrixLayoutCreate(&d, HIP_R_32F, out_features, tokens, out_features), "layout D");
    if (ok) ok = check(hipblasLtMatmulPreferenceCreate(&pref), "preference");
    if (ok) ok = check(hipblasLtMatmulPreferenceSetAttribute(
                           pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_,
                           sizeof(workspace_size_)), "workspace");

    hipblasLtMatmulHeuristicResult_t heur{};
    int found = 0;
    if (ok) ok = check(hipblasLtMatmulAlgoGetHeuristic(handle_, op, a, b, c, d, pref, 1, &heur, &found), "heuristic");
    if (ok && found < 1) {
        std::fprintf(stderr, "hipblaslt: no algorithm for out=%lld in=%lld tokens=%lld\n",
                     (long long) out_features, (long long) in_features, (long long) tokens);
        ok = false;
    }
    if (ok) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        ok = check(hipblasLtMatmul(handle_, op, &alpha, w, a, x, b, &beta, y, c, y, d, &heur.algo,
                                   workspace_, workspace_size_, hipStreamPerThread), "matmul");
    }

    if (pref != nullptr) (void) hipblasLtMatmulPreferenceDestroy(pref);
    if (d != nullptr) (void) hipblasLtMatrixLayoutDestroy(d);
    if (c != nullptr) (void) hipblasLtMatrixLayoutDestroy(c);
    if (b != nullptr) (void) hipblasLtMatrixLayoutDestroy(b);
    if (a != nullptr) (void) hipblasLtMatrixLayoutDestroy(a);
    if (op != nullptr) (void) hipblasLtMatmulDescDestroy(op);
    return ok;
}

} // namespace omph::runtime
