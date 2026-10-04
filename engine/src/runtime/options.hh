// Every OMPH_* environment switch of the runner, parsed once (#100).
//
// They are A/B switches, validation references and ablations, not features:
// the defaults are what the engine runs. Ablations give wrong results with
// valid timings.
#pragma once

#include <cstdint>

namespace omph::runtime {

struct EnvOptions {
    // --- KV cache
    bool kv_host = false;       // OMPH_KV_HOST: exact f32 cache in pinned host RAM (reference)
    bool kv_f32 = false;        // OMPH_KV_F32: exact f32 cache in VRAM instead of K Q8 / V Q4
    bool kv_k4 = false;         // OMPH_KV_K4: K in V's Q4 format too (#81, experiment)
    // K4 on these attention layers (0-based over the 16 attention layers, bit i),
    // the rest Q8 (#175): by default the eight least sensitive ones, measured one
    // layer at a time (bench/results/k4-per-layer-175.txt). OMPH_KV_K4_LAYERS=i,j,...
    // replaces the set, OMPH_KV_K4_LAYERS=none keeps every layer Q8.
    static constexpr uint64_t kDefaultK4Layers = (1u << 2) | (1u << 4) | (1u << 5) | (1u << 6) |
                                                 (1u << 7) | (1u << 9) | (1u << 11) | (1u << 15);
    uint64_t kv_k4_layers = kDefaultK4Layers;
    int64_t kv_window = 128;    // OMPH_KV_WINDOW=N: FP16 ring of the last N tokens (0 = off)

    // --- decode paths (A/B switches)
    bool overlap = false;       // OMPH_OVERLAP: a side stream for sibling GEMVs (#71; off by default, #189)
    int stage_mib = 20;         // OMPH_STAGE_MIB=N: f16 weight slices of ~N MiB (0: whole)
    int gemm_min = 16;          // OMPH_GEMM_MIN=T: --gemv runs of T+ tokens take the GEMM path
    bool no_fused_gemm = false; // OMPH_NO_FUSED_GEMM: dequant to f16, then the GEMM (#141)
    bool attn_scalar = false;   // OMPH_ATTN_SCALAR: prefill attention without WMMA (#97)
    bool attn_dec_scalar = false;  // OMPH_ATTN_DEC_SCALAR: decode / verification attention on the scalar kernel (#169)
    bool gdn_serial = false;    // OMPH_GDN_SERIAL: a chunk's delta rule in one launch (#96)
    bool gdn_exact = false;     // OMPH_GDN_EXACT: prefill delta rule per token, not the chunked WY form (#240)
    bool no_b4 = false;         // OMPH_NO_B4: no NT-token GEMVs (verification, --gemv prefill)
    bool no_group = false;      // OMPH_NO_GROUP: sibling GEMVs as separate launches (#214)
    bool no_swiglu_gemm = false;  // OMPH_NO_SWIGLU_GEMM: the FFN's up GEMM writes f32, then swiglu_f16 (#221)
    int64_t mtp_window = 16384;  // OMPH_MTP_WINDOW=N: the MTP block attends 16 sinks + the last N..2N positions (0: all; #171)
    int64_t draft_vocab = 98304;  // OMPH_DRAFT_VOCAB=N: MTP drafts over the first N token ids (0: all; #217)
    bool no_bf16_gemv = false;  // OMPH_NO_BF16_GEMV: BF16 weights through the f16 path
    bool no_f16_cache = false;  // OMPH_NO_F16_CACHE: re-convert f16-path weights every call
    bool host_argmax = false;   // OMPH_HOST_ARGMAX: greedy argmax on the host (#102)
    bool gdn_per_token = false; // OMPH_GDN_PER_TOKEN: one delta-net launch per token (#96)

    // --- ablations (wrong results, valid timings)
    bool skip_attn = false;     // OMPH_SKIP_ATTN: no attention blocks
    bool skip_ffn = false;      // OMPH_SKIP_FFN: no FFN blocks
    bool skip_blocks = false;   // OMPH_SKIP_BLOCKS: no attention / delta-net / FFN at all
    bool skip_gemv = false;     // OMPH_SKIP_GEMV: no fused GEMVs
    int skip_gemv_type = -1;    // OMPH_SKIP_GEMV_TYPE=T: no GEMVs of GGUF type T (#63)
    bool skip_stage = false;    // OMPH_SKIP_STAGE: no f16 staging + GEMMs

    // --- diagnostics
    bool timing = false;        // OMPH_TIMING: VRAM after load, per-step GPU and wall times
    bool phases = false;        // OMPH_PHASES: per-phase GPU totals (stage / gemm / gemv /
                                // blocks); their events perturb the step, keep apart
    bool trace_alloc = false;   // OMPH_TRACE_ALLOC: every f16 scratch allocation
    bool trace_f16 = false;     // OMPH_TRACE_F16: matmuls that fall back to the f16 path
    bool trace_stage = false;   // OMPH_TRACE_STAGE: every f16 staging / cache hit
    bool spec_check = false;    // OMPH_SPEC_CHECK: verify every speculative rollback (slow)
    bool check_finite = false;  // OMPH_CHECK_FINITE: refuse logits rows with NaN / Inf (#315)
    int test_bad_side = 0;      // OMPH_TEST_BAD_SIDE=N: the rejected side stream back after N steps (#144)
    int test_mrope = 0;         // OMPH_TEST_MROPE=swap|flat: image positions with h / w swapped, or
                                // sequential (validation of the M-RoPE layout, #160)
    int ngram = 1;              // OMPH_NGRAM=0: no n-gram (prompt lookup) drafts (#199)
    int ngram_min = 4;          // OMPH_NGRAM_MIN=N: an n-gram draft needs N proposed tokens (#199)
    float dflash_keep = 0.12f;  // OMPH_DFLASH_KEEP=P: draft position n only while the measured chance
                                // that drafts 1..n are all kept is >= P (0: always all, #245)
    float dflash_pmin = 0.0f;   // OMPH_DFLASH_PMIN=P: DFlash2 drafts stop where the selector's best
                                // candidate has softmax probability < P (#245)
    int test_q8act = 0;         // OMPH_TEST_Q8ACT=B: every prefill GEMM's input quantized to int8 in blocks
                                // of B along k and back: an int8 GEMM's numerics (#213)

    static EnvOptions from_env();
};

} // namespace omph::runtime
