// Runner: matmul dispatch (fused GEMV, four-token GEMV, f16 + WMMA GEMM).
#include "model/runner.hh"

#include "format/repack.hh"
#include "kernels/attn.hh"
#include "kernels/dequant.hh"
#include "kernels/elementwise.hh"
#include "kernels/gdn.hh"
#include "kernels/gemv.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace omph::model {

// One fused GEMV launch for a single token, dispatched on the GGUF type.
bool Runner::gemv_one(const int type, const void * w, const void * x, float * y,
                     const int64_t n_out, const int64_t k, hipStream_t st) {
    switch (type) {
        case 10: return omph::kernels::gemv_q2k(w, x, y, n_out, k, st);
        case 12: return omph::kernels::gemv_q4k(w, x, y, n_out, k, st);
        case 14: return omph::kernels::gemv_q6k(w, x, y, n_out, k, st);
        case 16: return omph::kernels::gemv_iq2_xxs(w, x, y, n_out, k, st);
        case 17: return omph::kernels::gemv_iq2_xs(w, x, y, n_out, k, st);
        case 18: return omph::kernels::gemv_iq3_xxs(w, x, y, n_out, k, st);
        case 21: return omph::kernels::gemv_iq3_s(w, x, y, n_out, k, st);
        case 22: return omph::kernels::gemv_iq2_s(w, x, y, n_out, k, st);
        case 23: return omph::kernels::gemv_iq4_xs(w, x, y, n_out, k, st);
        case 29: return omph::kernels::gemv_iq1_m(w, x, y, n_out, k, st);
        default: return false;
    }
}

// One matmul: the fused GEMV for single-token steps when it is available for
// this tensor, the small-batch GEMV for a few tokens, otherwise the f16
// dequant + GEMM path.
bool Runner::matmul(const Mat & m, const void * x16, float * y, const int64_t n_out,
            const int64_t k, const int64_t T) {
    if (m.t == nullptr) {
        throw std::runtime_error("missing tensor " + m.name);
    }
    const GemvEntry * g = m.gemv;
    if (T == 1) {
        // BF16 weights: bf16 is the top half of an f32, so a direct dot
        // product beats converting to f16 and running a tiny M = 1 GEMM.
        if (m.t->type == 30 && (int64_t) m.t->ne[1] == n_out && (int64_t) m.t->ne[0] == k &&
            !env_.no_bf16_gemv) {
            timer_gemv_.start(gemv_stream_);
            const bool ok = omph::kernels::gemv_bf16(m.dev, x16, y, n_out, k, gemv_stream_);
            timer_gemv_.stop(t_gemv_, gemv_stream_);
            if (ok) {
                return true;
            }
        }
    }
    if (use_gemv_ && T == 1) {
        if (g != nullptr && g->rows == n_out && g->k == k) {
            if (env_.skip_gemv) {
                return true;  // ablation only: wrong results, valid timing
            }
            // Ablation only: skip the GEMVs of one GGUF type. The step time
            // difference is that type's real cost; rocprofv3's per-kernel
            // times inflate some kernels by up to 40 % (#63).
            if (env_.skip_gemv_type >= 0 && (uint32_t) env_.skip_gemv_type == g->type) {
                return true;
            }
            timer_gemv_.start(gemv_stream_);
            const bool gemv_ok = gemv_one(g->type, m.dev, x16, y, n_out, k, gemv_stream_);
            timer_gemv_.stop(t_gemv_, gemv_stream_);
            if (gemv_ok) {
                return true;
            }
            if (env_.trace_f16) {
                std::fprintf(stderr,
                             "f16 path: %s (entry yes, type %u, ne %lld x %lld, %lld B)\n",
                             m.name.c_str(), (unsigned) m.t->type, (long long) m.t->ne[0],
                             (long long) m.t->ne[1], (long long) m.t->nbytes);
            }
        } else if (env_.trace_f16) {
            std::fprintf(stderr, "f16 path: %s (not a gemv tensor, type %u, %lld B)\n",
                         m.name.c_str(), (unsigned) m.t->type, (long long) m.t->nbytes);
        }
    }
    // Long multi-token runs (prefill chunks) take the GEMM path even with
    // --gemv: past gemm_min_ tokens, dequantizing a weight once beats T / 4
    // passes of the NT GEMVs over it (M8).
    if (use_gemv_ && T > 1 && T < gemm_min_ && g != nullptr && g->rows == n_out && g->k == k) {
        // the ablations of the single-token path apply here too
        if (env_.skip_gemv ||
            (env_.skip_gemv_type >= 0 && (uint32_t) env_.skip_gemv_type == g->type)) {
            return true;
        }
        const auto * w = static_cast<const uint8_t *>(m.dev);
        const auto * xb = static_cast<const uint8_t *>(x16);
        // NT-token kernels (#126): groups of up to four tokens per weight read,
        // a remainder of 2 or 3 included; a type without them falls through.
        if (!env_.no_b4) {
            timer_gemv_.start(gemv_stream_);
            bool ok = true;
            bool supported = true;
            for (int64_t t0 = 0; t0 < T && ok; ) {
                const int64_t n = std::min<int64_t>(4, T - t0);
                if (n == 1) {
                    ok = gemv_one(g->type, w, xb + t0 * k * 2, y + t0 * n_out, n_out, k,
                                  gemv_stream_);
                } else {
                    ok = omph::kernels::gemv_multi(g->type, w, xb + t0 * k * 2, y + t0 * n_out,
                                                   n_out, k, (int) n, gemv_stream_);
                    if (!ok && t0 == 0) {
                        supported = false;  // nothing issued yet: fall through
                        (void) hipGetLastError();
                        break;
                    }
                }
                t0 += n;
            }
            timer_gemv_.stop(t_gemv_, gemv_stream_);
            if (supported && ok) {
                return true;
            }
        }
        if (has_gemv_type(g->type)) {
            // Types with a single-token kernel but no NT form (IQ1_M, Q6_K): one
            // launch per token. No staging, so a long prefill stays inside VRAM.
            timer_gemv_.start(gemv_stream_);
            bool ok = true;
            for (int64_t t0 = 0; t0 < T && ok; ++t0) {
                ok = gemv_one(g->type, w, xb + t0 * k * 2, y + t0 * n_out, n_out, k,
                              gemv_stream_);
            }
            timer_gemv_.stop(t_gemv_, gemv_stream_);
            if (ok) {
                return true;
            }
        }
    }
    if (env_.skip_stage) {
        return true;  // ablation only
    }
    // a weight over the scratch budget goes in slices of rows, each one's
    // GEMM writing its columns of y
    const int64_t rows = stage_rows(n_out, k);
    for (int64_t r0 = 0; r0 < n_out; r0 += rows) {
        const int64_t nr = std::min(rows, n_out - r0);
        void * w = stage_w(m.name, r0, nr);
        timer_gemm_.start();
        const bool ok = linear_.run(w, x16, y + r0, nr, k, T, n_out);
        timer_gemm_.stop(t_gemm_);
        if (!ok) {
            return false;
        }
    }
    return true;
}

} // namespace omph::model
