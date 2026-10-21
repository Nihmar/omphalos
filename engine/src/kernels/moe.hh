// The mixture-of-experts FFN of qwen35moe (#493, #490): the router, the weight
// accumulation and the shared expert's sigmoid gate.
//
// The semantics are llama.cpp's build_moe_ffn with norm_w = true and no bias
// (src/llama-graph.cpp at the pin): logits = x . router, softmax over all the
// experts, top-k (argsort order), the selected probabilities renormalized to
// sum 1. The weights are applied to the experts' down outputs, after the FFN.
//
// The routed experts' weights are slices of one 3-D tensor in the engine's
// tile layout (format/repack.hh): every expert is a contiguous run of 16-row
// tiles and of GGUF blocks, so the existing per-type GEMV/GEMM entry points
// take an expert by pointing them at its slice (the layout's offsets are a
// function of the tile/block index alone). No new matmul kernel is needed.
#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace omph::kernels {

// The routing table on the device (#493): the per-expert counts, their
// exclusive prefix sum and the gather (xgi = the token) / slot arrays in
// expert-major order. n_pairs = tokens * topk; at most 256 experts (one
// thread each). `act` (optional, n_expert + 1 ints) receives the number of
// chosen experts, then their ids in increasing order (#551).
bool moe_order(const void * ids, int64_t n_pairs, int64_t n_expert, int64_t topk, void * counts,
               void * offs, void * xgi, void * slot, void * max_count, hipStream_t stream,
               void * act = nullptr);

// The routed gate's selection: the softmax over a token's router logits (the
// [tokens, n_expert] f32 rows of the router GEMM, plus the rows of `logits_lo`
// when it is not null), the top-k ids and their renormalized probabilities.
// Ties keep the lower id. The logits themselves are a plain f16 GEMM (the
// router weights are converted to f16 once at load): the per-expert scalar dot
// this replaced was 57 % of the second model's prefill (#494). The input is
// two f16 halves, the GEMM's on each summed here (#547: an f16 input alone
// flipped 0.14 % of the expert picks).
bool moe_topk(const void * logits, const void * logits_lo, int * ids, float * weights, int64_t tokens,
              int64_t n_expert, int topk, hipStream_t stream);

// The router logits of a few tokens (decode, verification; #509): y[t][e] =
// w_f16[e] . (x_f16[t] + x_lo_f16[t]), one wave per (expert, token) in a fixed
// order, so a token's logits are the same for any token count. Needs k even.
bool moe_router(const void * w_f16, const void * x_f16, const void * x_lo_f16, float * logits, int64_t tokens,
                int64_t n_expert, int64_t k, hipStream_t stream);

// rows x n 32-bit words from src (pitch spitch words) to dst (pitch dpitch),
// either side in VRAM or in pinned host memory: the expert cache's table
// upload and selection log (#513) as kernels on the compute stream, so they do
// not queue on a DMA engine behind the cache's fills.
bool moe_copy_words(void * dst, int64_t dpitch, const void * src, int64_t spitch, int64_t n, int64_t rows,
                    hipStream_t stream);

// out[t][j] += weights[t * stride + slot] * y[t][j]: one expert's down output,
// weighted by its router probability, accumulated into the block's output.
bool moe_acc(float * out, const float * y, const float * weights, int64_t tokens, int64_t stride,
             int slot, int64_t n, hipStream_t stream);

// The prefill path (#494): the rows `ids[0..n)` of x (f16, k values each)
// gathered back to back into dst, so one expert's tokens are one contiguous
// GEMM input.
bool moe_gather(const void * x_f16, int64_t k, const int * ids, int64_t n, void * dst,
                hipStream_t stream);

// ... and the other direction: the gathered row i (an expert's down output)
// stored at slot (dst[i], slot[i]) of the per-slot buffer `xo`
// (tokens x topk x ne). Two rows never target the same slot, so a plain store
// keeps the result deterministic (no atomics).
bool moe_scatter_store(float * xo, const float * y, const int * slot, const int * dst, int64_t n,
                       int64_t ne, int topk, hipStream_t stream);

// The end of a MoE layer (#522): out[t][j] = sum_e weights[t * topk + e] *
// xo[(t * topk + e) * ne + j] in slot order (or out as it is when xo is null)
// + sigmoid(x_t . wg) * ysh[t * ne + j]; wg the shared expert's F32 gate row
// (ffn_gate_inp_shexp), x the layer's f16 FFN input (its rounding moves the
// gate by < 1e-4, #544; the router reads hi + lo, #547). One launch for the
// routed sum, the shared gate and the shared expert's accumulate.
bool moe_finish(float * out, const float * xo, const float * weights, int topk, const float * ysh,
                const void * wg, const void * x_f16, int64_t tokens, int64_t ne, hipStream_t stream);

} // namespace omph::kernels
