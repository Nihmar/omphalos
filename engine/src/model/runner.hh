// The runner: weights in VRAM (repacked for the fused GEMVs), the activation
// buffers, the KV cache and the delta-net states, and the forward pass of the
// 64-layer stack (PLAN.md §12). One instance per model; not copyable.
//
// Split by concern (#100): runner.cc (load, buffers, weight resolution, the f16
// path), forward.cc (layer loop, lm_head), layers.cc (attention and delta-net
// blocks, KV cache), dispatch.cc (GEMV / GEMM dispatch).
#pragma once

#include "format/gguf.hh"
#include "model/hparams.hh"
#include "runtime/allocations.hh"
#include "runtime/matmul.hh"
#include "runtime/options.hh"
#include "runtime/scratch.hh"
#include "runtime/timing.hh"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace omph::model {

using omph::runtime::PhaseTimer;
using omph::runtime::Scratch;

class Runner {
    // A tensor repacked for a fused GEMV (PLAN.md §8.3).
    struct GemvEntry {
        size_t off = 0;
        size_t bytes = 0;
        int64_t rows = 0;
        int64_t k = 0;
        uint32_t type = 0;
        bool has_b4 = false;  // a small-batch kernel exists for this type
    };

    // A weight matrix resolved once at load (#100): the hot path looks nothing
    // up by name.
    struct Mat {
        std::string name;
        const omph::gguf::TensorInfo * t = nullptr;  // null: not in the file
        const GemvEntry * gemv = nullptr;            // repacked for a fused GEMV
        const void * dev = nullptr;                  // its bytes in the weight image
    };
    struct LayerWeights {
        bool recurrent = false;
        const float * attn_norm = nullptr;
        const float * post_norm = nullptr;
        // full attention
        const float * q_norm = nullptr;
        const float * k_norm = nullptr;
        Mat attn_q, attn_k, attn_v, attn_output;
        // gated delta net
        const float * dt_bias = nullptr;
        const float * ssm_a = nullptr;
        const float * ssm_norm = nullptr;
        const float * conv_w = nullptr;
        const uint16_t * w_beta = nullptr;   // BF16 rows, dotted inside gdn_step
        const uint16_t * w_alpha = nullptr;
        Mat attn_qkv, attn_gate, ssm_out;
        // FFN
        Mat ffn_up, ffn_gate, ffn_down;
    };

public:
    // Loads the model at `path` and sizes the activations for chunks of up to
    // `max_tokens` tokens and the KV cache for `kv_capacity` positions (default:
    // max_tokens). `use_gemv` repacks the weights for the fused GEMVs (decode);
    // `last_logits_only` keeps the lm_head to one row.
    explicit Runner(const std::string & path, const int64_t max_tokens, const bool use_gemv,
                    const omph::runtime::EnvOptions & env, const bool last_logits_only = false,
                    const int64_t kv_capacity = 0);

    // Memory is owned by mem_; the stream and the events are released here.
    ~Runner();

    Runner(const Runner &) = delete;
    Runner & operator=(const Runner &) = delete;

    // want_logits = false runs the layers only (a prefill chunk whose logits
    // nobody reads): no final norm, no lm_head.
    // greedy (decode, --gemv): set to the argmax token computed on the device,
    // and the logits stay there (#102); -1 when this path did not run, and the
    // caller takes the argmax of `logits`.
    bool forward(const std::vector<int32_t> & toks, std::vector<float> & logits,
                 const std::string & trace_dir, const int64_t start_pos = 0,
                 const bool want_logits = true, int32_t * greedy = nullptr);

    const HParams & hparams() const { return h_; }

private:
    // Final norm and lm_head of the T rows in h16_, into `logits` on the host
    // (or, with `greedy` on a single-token GEMV step, only the argmax). Returns
    // once the results are on the host.
    bool lm_head(const int64_t T, std::vector<float> & logits, int32_t * greedy);

    // The first `rows` rows of logits_ to the host (synchronous).
    bool copy_logits(float * dst, const int64_t rows);

    // lm head in vocab chunks, through the f16 dequant + hipBLASLt path: the
    // whole f16 head (248k x 5120, 2.5 GB) would not fit. Writes (T, n_vocab)
    // to `out` on the host, one vocab chunk at a time.
    bool head_chunked(const void * x16, const int64_t T, float * out);

    // OMPH_PHASES: GPU time per phase since the last report. Each GEMV is timed
    // on its own stream, so GEMVs overlapped on the side stream count in full
    // and the gemv total can exceed the step.
    void report_phases();

    void step_event(hipEvent_t a, hipEvent_t b);

    // Returns false: every caller is a bool function that reports failure.
    static bool fail(const char * msg);

    // False for the tensors of the blocks past the stack (the MTP block).
    bool in_stack(const std::string & name) const;


    bool attn_layer(const int64_t il, const LayerWeights & L, const int64_t T,
                    const int64_t pos0);

    bool gdn_layer(const int64_t il, const LayerWeights & L, const int64_t T);

    // One attention layer's slice of the quantized cache (indexed by attention
    // layer, kv_index_, not by model layer: 16 of the 64 layers have one).
    struct QuantKv {
        uint8_t * kq;
        void * ksc;
        uint8_t * vq;
        void * vsc;
        void * k16;
        void * v16;
    };
    QuantKv quant_kv(const int64_t il) const;

    // KV write + attention, on either the f32 cache (memcpy) or the Q8/Q4 one
    // (quantize); attention_gqa then reads either, dequantizing on the fly.
    bool attn_impl(const int64_t il, float * k_cache, float * v_cache, const int64_t pos0,
                   const int64_t T);

    // Runs `side` on the side stream and `main` on the default one, both after
    // everything issued so far; the default stream continues once both are done.
    // Without overlap (prefill, or OMPH_NO_OVERLAP) they simply run in order.
    template <typename Side, typename Main>
    bool fork_join(const int64_t T, Side && side, Main && main) {
        if (!overlap_ || T != 1) {
            return side() && main();
        }
        if (hipEventRecord(ev_fork_, nullptr) != hipSuccess ||
            hipStreamWaitEvent(side_, ev_fork_, 0) != hipSuccess) {
            return false;
        }
        gemv_stream_ = side_;
        const bool ok_side = side();
        gemv_stream_ = nullptr;
        const bool ok_main = main();
        return ok_side && ok_main && hipEventRecord(ev_join_, side_) == hipSuccess &&
               hipStreamWaitEvent(nullptr, ev_join_, 0) == hipSuccess;
    }

    // One fused GEMV launch for a single token, dispatched on the GGUF type.
    static bool gemv_one(const int type, const void * w, const void * x, float * y,
                         const int64_t n_out, const int64_t k, hipStream_t st);

    // Small-batch GEMV: one weight read per four tokens. Only the types that have
    // this form answer true; the others stay on the f16 path.
    static bool gemv_batch4(const int type, const void * w, const void * x, float * y,
                            const int64_t n_out, const int64_t k, hipStream_t st);

    // One matmul: the fused GEMV for single-token steps when it is available for
    // this tensor, the small-batch GEMV for a few tokens, otherwise the f16
    // dequant + hipBLASLt path.
    bool matmul(const Mat & m, const void * x16, float * y, const int64_t n_out,
                const int64_t k, const int64_t T);

    // Allocates *p on first use (the f16-path buffers, #86).
    void * lazy(void ** p, const size_t bytes);

    // raw_stage_ holds the original bytes of any one repacked tensor.
    size_t raw_stage_bytes() const;

    // Device pointer to the original GGUF bytes: they live in the image unless
    // the tensor was repacked, in which case they are staged from the file.
    const void * raw_bytes(const std::string & name);

    // Types that have a fused GEMV kernel: for these the f16 form is never worth
    // keeping (it would be two to four times the quantized size, and they are the
    // bulk of the model).
    static bool has_gemv_type(const uint32_t type);

    void * stage_w(const std::string & name);

    Mat resolve(const std::string & name) const;

    // An f32 vector in the image, or null when the file does not have it.
    const float * resolve_f32(const std::string & name) const;

    void resolve_layers();

    const omph::runtime::EnvOptions env_;  // first: the other members' setup reads it
    omph::runtime::Allocations mem_;       // every device / pinned buffer below
    omph::gguf::File file_;
    HParams h_;
    bool use_gemv_ = false;
    omph::runtime::Linear linear_;  // one hipBLASLt handle for the whole run
    PhaseTimer timer_stage_;
    PhaseTimer timer_gemm_;
    PhaseTimer timer_gemv_;
    PhaseTimer timer_block_;
    std::vector<std::pair<hipEvent_t, hipEvent_t>> t_gemv_;
    std::vector<std::pair<hipEvent_t, hipEvent_t>> t_stage_;
    std::vector<std::pair<hipEvent_t, hipEvent_t>> t_gemm_;
    std::vector<std::pair<hipEvent_t, hipEvent_t>> t_block_;
    std::unordered_map<std::string, size_t> off_;
    std::unordered_map<std::string, GemvEntry> gems_;
    std::vector<LayerWeights> layers_;  // resolved at load, by resolve_layers()
    Mat head_;
    const float * out_norm_ = nullptr;
    uint32_t embd_type_ = 0;
    int64_t embd_row_bytes_ = 0;
    Scratch scratch_;
    std::vector<void *> states_;  // per layer, into state_pool_; null for attention
    void * state_pool_ = nullptr;
    void * embd_host_ = nullptr;   // token_embd, raw GGUF bytes, pinned host memory
    bool last_logits_only_ = false;
    static constexpr int64_t kHeadRows = 32;  // logits_ rows (multiple of 4)
    std::vector<int64_t> kv_index_;
    std::vector<char> conv_flip_;
    int64_t max_seq_ = 0;
    int64_t max_tokens_ = 0;
    void * kv_k_ = nullptr;
    void * kv_v_ = nullptr;
    void * kv_kq_ = nullptr;
    void * kv_ks_ = nullptr;
    void * kv_vq_ = nullptr;
    void * kv_vs_ = nullptr;
    void * kv_k16_ = nullptr;
    void * kv_v16_ = nullptr;
    int64_t kv_window_ = 0;  // Q8/Q4 mode: 128 unless OMPH_KV_WINDOW says otherwise
    bool kv_q8q4_ = false;
    bool kv_k4_ = false;  // K stored as Q4 (experiment, #81)
    bool kv_host_ = false;  // f32 KV in pinned host RAM (validation reference)
    void * kv_stage_k_ = nullptr;
    // Overlap of independent GEMVs (#71): the decode forks a layer's sibling
    // projections onto side_ and joins before their consumers. Consecutive
    // kernels on one stream leave a gap between one's tail and the next one's
    // start that caps the GEMVs at ~80 % of the bandwidth; overlapped, they
    // reach ~96 % (omph-gemv-bench).
    hipStream_t side_ = nullptr;
    hipEvent_t ev_fork_{};
    hipEvent_t ev_join_{};
    hipStream_t gemv_stream_ = nullptr;  // where matmul's GEMVs go
    bool overlap_ = false;
    void * attn_work_ = nullptr;  // split-K partials of attention_gqa
    size_t attn_work_bytes_ = 0;
    void * kv_stage_v_ = nullptr;
    void * dev_weights_ = nullptr;
    void * x_ = nullptr;
    void * cur_ = nullptr;
    void * resid_ = nullptr;
    void * blk_ = nullptr;
    void * h16_ = nullptr;
    void * fused_ = nullptr;
    void * z_ = nullptr;
    void * q_ = nullptr;
    void * gate_ = nullptr;
    void * k_ = nullptr;
    void * v_ = nullptr;
    void * ffn1_ = nullptr;
    void * ffn2_ = nullptr;
    void * ffn16_ = nullptr;
    void * logits_ = nullptr;
    void * argmax_key_ = nullptr;  // 8 bytes: the greedy decode's packed argmax (#102)
    void * tmp_logits_ = nullptr;
    void * head16_ = nullptr;
    void * raw_stage_ = nullptr;
    size_t tmp_logits_bytes_ = 0;
    size_t head16_bytes_ = 0;
    std::map<std::string, void *> f16_cache_;  // into mem_
};

} // namespace omph::model
