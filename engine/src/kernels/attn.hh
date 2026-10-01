// Attention block kernels (PLAN.md §10.4, §13).
#pragma once

#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace omph::kernels {

// --- quantized KV (PLAN.md §13) -------------------------------------------
//
// By default K is stored as Q8 and V as Q4, in 32-element blocks along head_dim
// (Hadamard-rotated) with f16 scales in a separate stream. head_dim is 256, so
// there are 8 blocks per head per token: K is 272 B per head (256 + 16 of
// scales) and V is 144 B, against 1024 B each in f32. With k_q4 (OMPH_KV_K4,
// #81) K uses V's Q4 format too. The last `window` tokens are also kept exactly
// in an FP16 ring.

// --- fused attention prep (#79) -------------------------------------------
//
// One launch for the Q/gate split, the QK RMSNorm, the partial NeoX RoPE on q
// and k, the Hadamard rotation of q, k and v (when `rotate`) and, when k_q8 is
// set, the quantized cache write at pos0 (+ the FP16 ring). head_dim must be
// 256.
struct AttnPrep {
    const float * qg = nullptr;      // (tokens, nh, 2 * hd): q | gate per head
    float * q = nullptr;             // (tokens, nh, hd)
    float * gate = nullptr;          // (tokens, nh, hd)
    float * k = nullptr;             // (tokens, nkv, hd), in place
    float * v = nullptr;             // (tokens, nkv, hd), in place
    const float * q_norm = nullptr;  // (hd)
    const float * k_norm = nullptr;  // (hd)
    long long tokens = 0;
    long long pos0 = 0;
    long long nh = 0;
    long long nkv = 0;
    long long hd = 0;
    int n_rot = 0;
    float freq_base = 10000.0f;
    float eps = 1e-6f;
    bool rotate = false;             // Hadamard on q, k, v (quantized cache)
    uint8_t * k_q8 = nullptr;        // quantized cache, or null for f32
    __half * k_scales = nullptr;
    uint8_t * v_q4 = nullptr;
    __half * v_scales = nullptr;
    __half * k16 = nullptr;          // FP16 ring, or null
    __half * v16 = nullptr;
    long long window = 0;
    bool k_q4 = false;               // quantize K as Q4 into k_q8 (V's format, #81)
};
bool attn_prep(const AttnPrep & a, hipStream_t stream);

// --- GQA-grouped flash attention (issue #59) -------------------------------
//
// One layer's KV cache, in either storage: f32 (k_f32/v_f32, (seq, nkv, hd)) or
// K Q8 / V Q4 with the optional FP16 window (the cache attn_prep writes).
struct KvCache {
    const float * k_f32 = nullptr;
    const float * v_f32 = nullptr;
    const uint8_t * k_q8 = nullptr;
    const void * k_scales = nullptr;
    const uint8_t * v_q4 = nullptr;
    const void * v_scales = nullptr;
    const void * k16 = nullptr;
    const void * v16 = nullptr;
    int64_t window = 0;
    bool k_q4 = false;  // k_q8 holds K in V's Q4 format (#81)
};

// Causal attention with GQA and the sigmoid output gate. q, gate, out:
// (tokens, nh, hd), q post-RoPE (and rotated for a Q8/Q4 cache); query t sits
// at cache row seq - tokens + t. A
// workgroup serves a tile of query tokens times the nh/nkv query heads of one KV
// head, so each K/V block is read and dequantized once for all of them; the key
// range is split so a decode step still fills the GPU, and a second kernel
// merges the splits in a fixed order (deterministic), un-rotates V when
// `v_rotated` (Q8/Q4 cache) and applies the gate. Written for this model:
// head_dim 256 and 6 query heads per KV head. `work` must hold
// attention_gqa_work_bytes(tokens, ...).
bool attention_gqa(const float * q, const KvCache & kv, const float * gate, float * out,
                   int64_t tokens, int64_t seq, int64_t nh, int64_t nkv, int64_t hd, float scale,
                   bool v_rotated, void * work, size_t work_bytes, hipStream_t stream,
                   void * out_f16 = nullptr,  // set: out is ignored, f16 written here
                   bool allow_wmma = true,    // 16+ tokens: the WMMA prefill kernel (#97)
                   int64_t key_chunk = 0);    // > 0: keys per split, fixed for the run (#136)

// Keys per split for a run whose sequences reach max_seq (a multiple of 16).
// With a fixed chunk every query sums its keys in the same order in any call
// (a decode step or a speculative batch), so the two agree bit for bit.
int64_t attention_key_chunk(int64_t max_seq);

// Copies the FP16-ring slots of positions pos_first .. pos_first + count - 1
// (slot = position % window) of every layer from (src_k, src_v) to
// (dst_k, dst_v), all laid out as the ring: layers x window x row_bytes. Saves
// the slots a speculative verification overwrites, and restores those of the
// rejected positions (#98, #122). row_bytes % 16 == 0.
bool kv_ring_copy(const void * src_k, const void * src_v, void * dst_k, void * dst_v,
                  int64_t layers, int64_t window, int64_t row_bytes, int64_t pos_first,
                  int64_t count, hipStream_t stream);

// Workspace for any call with up to `max_tokens` query tokens (and, with a
// key chunk, sequences up to max_seq).
size_t attention_gqa_work_bytes(int64_t max_tokens, int64_t nh, int64_t nkv, int64_t hd,
                                int64_t max_seq = 0, int64_t key_chunk = 0);

} // namespace omph::kernels
