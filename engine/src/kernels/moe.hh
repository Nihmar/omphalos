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

// The routed gate for `tokens` hidden states (f16, k values each):
// logits = x . router (F32 [k, n_expert], ne[0] = k), softmax over all of them,
// the top `topk` ids written to `ids` (row-major, tokens x topk) and their
// renormalized probabilities to `weights`. Ties keep the lower id.
bool moe_route(const void * router, const void * x_f16, int * ids, float * weights, int64_t tokens,
               int64_t k, int64_t n_expert, int topk, hipStream_t stream);

// out[t][j] += weights[t * stride + slot] * y[t][j]: one expert's down output,
// weighted by its router probability, accumulated into the block's output.
bool moe_acc(float * out, const float * y, const float * weights, int64_t tokens, int64_t stride,
             int slot, int64_t n, hipStream_t stream);

// The shared expert's gate per token: out[t] = sigmoid(x_t . w), w a F32 row of
// k values (ffn_gate_inp_shexp).
bool moe_shexp_gate(const void * w, const void * x_f16, float * out, int64_t tokens, int64_t k,
                    hipStream_t stream);

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

// out[t][j] = sum_e weights[t * topk + e] * xo[(t * topk + e) * ne + j]: the
// experts' weighted down outputs summed in slot order (deterministic).
bool moe_combine(float * out, const float * xo, const float * weights, int64_t tokens, int topk,
                 int64_t ne, hipStream_t stream);

} // namespace omph::kernels
