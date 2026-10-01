#include "runtime/matmul.hh"

#include <hip/hip_runtime.h>

#include "kernels/gemm.hh"

namespace omph::runtime {

bool Linear::run(const void * w, const void * x, float * y, const int64_t out_features,
                 const int64_t in_features, const int64_t tokens, const int64_t ldy) {
    return omph::kernels::gemm_f16(w, x, y, out_features, in_features, tokens,
                                   ldy > 0 ? ldy : out_features, hipStreamPerThread);
}

} // namespace omph::runtime
