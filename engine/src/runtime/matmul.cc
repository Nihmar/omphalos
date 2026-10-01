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
    for (auto & kv : plans_) {
        destroy(kv.second);
    }
    if (workspace_ != nullptr) {
        (void) hipFree(workspace_);
    }
    if (handle_ != nullptr) {
        (void) hipblasLtDestroy(handle_);
    }
}

void Linear::destroy(Plan & p) {
    if (p.d != nullptr) (void) hipblasLtMatrixLayoutDestroy(p.d);
    if (p.c != nullptr) (void) hipblasLtMatrixLayoutDestroy(p.c);
    if (p.b != nullptr) (void) hipblasLtMatrixLayoutDestroy(p.b);
    if (p.a != nullptr) (void) hipblasLtMatrixLayoutDestroy(p.a);
    if (p.op != nullptr) (void) hipblasLtMatmulDescDestroy(p.op);
    p = Plan{};
}

const Linear::Plan * Linear::plan(const int64_t out_features, const int64_t in_features,
                                  const int64_t tokens, const int64_t ldy) {
    const auto key = std::make_tuple(out_features, in_features, tokens, ldy);
    const auto it = plans_.find(key);
    if (it != plans_.end()) {
        return &it->second;
    }
    Plan p;
    hipblasLtMatmulPreference_t pref = nullptr;
    bool ok = check(hipblasLtMatmulDescCreate(&p.op, HIPBLAS_COMPUTE_32F, HIP_R_32F), "desc");
    // A = W (row-major out x in) == column-major (in x out), op(A) = A^T = (out x in)
    if (ok) ok = check(hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSA, &kTransA, sizeof(kTransA)), "transA");
    // B = x (row-major tokens x in) == column-major (in x tokens), no transpose
    if (ok) ok = check(hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSB, &kTransB, sizeof(kTransB)), "transB");
    if (ok) ok = check(hipblasLtMatrixLayoutCreate(&p.a, HIP_R_16F, in_features, out_features, in_features), "layout A");
    if (ok) ok = check(hipblasLtMatrixLayoutCreate(&p.b, HIP_R_16F, in_features, tokens, in_features), "layout B");
    if (ok) ok = check(hipblasLtMatrixLayoutCreate(&p.c, HIP_R_32F, out_features, tokens, ldy), "layout C");
    if (ok) ok = check(hipblasLtMatrixLayoutCreate(&p.d, HIP_R_32F, out_features, tokens, ldy), "layout D");
    if (ok) ok = check(hipblasLtMatmulPreferenceCreate(&pref), "preference");
    if (ok) ok = check(hipblasLtMatmulPreferenceSetAttribute(
                           pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_,
                           sizeof(workspace_size_)), "workspace");
    hipblasLtMatmulHeuristicResult_t heur{};
    int found = 0;
    if (ok) ok = check(hipblasLtMatmulAlgoGetHeuristic(handle_, p.op, p.a, p.b, p.c, p.d, pref, 1, &heur, &found), "heuristic");
    if (pref != nullptr) (void) hipblasLtMatmulPreferenceDestroy(pref);
    if (ok && found < 1) {
        std::fprintf(stderr, "hipblaslt: no algorithm for out=%lld in=%lld tokens=%lld\n",
                     (long long) out_features, (long long) in_features, (long long) tokens);
        ok = false;
    }
    if (!ok) {
        destroy(p);
        return nullptr;
    }
    p.algo = heur.algo;
    return &plans_.emplace(key, p).first->second;
}

bool Linear::run(const void * w, const void * x, float * y, const int64_t out_features,
                 const int64_t in_features, const int64_t tokens, const int64_t ldy) {
    if (handle_ == nullptr || workspace_ == nullptr) {
        return false;
    }
    const Plan * p = plan(out_features, in_features, tokens, ldy > 0 ? ldy : out_features);
    if (p == nullptr) {
        return false;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return check(hipblasLtMatmul(handle_, p->op, &alpha, w, p->a, x, p->b, &beta, y, p->c, y,
                                 p->d, &p->algo, workspace_, workspace_size_, hipStreamPerThread),
                 "matmul");
}

} // namespace omph::runtime
