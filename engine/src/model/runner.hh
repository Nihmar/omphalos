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

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace omph::model {

using omph::runtime::PhaseTimer;
using omph::runtime::Scratch;

// What a forward reads besides token ids (#160): embedding rows that replace
// some tokens (an image's), and M-RoPE positions.
struct ForwardInputs {
    std::vector<int64_t> rows;     // indices into the tokens, ascending
    const float * embd = nullptr;  // rows.size() x n_embd, host memory
    std::vector<int32_t> mpos;     // tokens x 3 (t, h, w), or empty: text positions
};

class Runner {
    // A tensor repacked for a fused GEMV (PLAN.md §8.3).
    struct GemvEntry {
        size_t off = 0;
        size_t bytes = 0;
        int64_t rows = 0;
        int64_t k = 0;
        uint32_t type = 0;
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
                    const int64_t kv_capacity = 0, bool mtp = false, const std::string & dflash = "");

    // Memory is owned by mem_; the stream and the events are released here.
    ~Runner();

    Runner(const Runner &) = delete;
    Runner & operator=(const Runner &) = delete;

    // want_logits = false runs the layers only (a prefill chunk whose logits
    // nobody reads): no final norm, no lm_head.
    // greedy (decode, --gemv): set to the argmax token computed on the device,
    // and the logits stay there (#102); -1 when this path did not run, and the
    // caller takes the argmax of `logits`.
    // `in` (#160): image embedding rows and M-RoPE positions; the MTP KV fill
    // after it reads them too.
    bool forward(const std::vector<int32_t> & toks, std::vector<float> & logits,
                 const std::string & trace_dir, const int64_t start_pos = 0,
                 const bool want_logits = true, int32_t * greedy = nullptr,
                 const ForwardInputs * in = nullptr);
    // Text RoPE position = cache position + delta (<= 0 after images, #160),
    // for every forward without M-RoPE positions, verification and draft.
    void set_rope_delta(int64_t delta) { rope_delta_ = delta; }
    // MTP window (#171): a prompt of n tokens fills the MTP KV for its last
    // window only; after a restore to position pos, the window holds nothing
    // past it (and starts over when pos is before it).
    void mtp_prompt(const int64_t n) { mtp_fill_from_ = mtp_window_ > 0 ? std::max(n - mtp_window_, kMtpSink) : 0; }
    void mtp_rewind(int64_t pos);

    const HParams & hparams() const { return h_; }

    // Starts a new sequence at position 0 without reloading: zeroes the
    // delta-net states, the conv tails and the MTP block's h (#152). The KV
    // caches need nothing: positions restart, and attention reads only the
    // positions the new sequence has written.
    bool reset_sequence();

    // --- sequence checkpoints (#158) ---
    // What a checkpoint holds: the state after the kept sequence that its next
    // tokens read and cannot rebuild from the KV caches — every delta-net
    // state and conv tail, the FP16 KV ring, the MTP block's h. The Q8/Q4 KV
    // (and the MTP KV) stay in VRAM: restoring is valid while the positions
    // below the checkpoint hold the same tokens as when it was saved.
    size_t checkpoint_bytes() const;
    // A pinned host buffer of checkpoint_bytes(), freed with the runner.
    void * checkpoint_alloc();
    bool checkpoint_save(void * host);
    bool checkpoint_restore(const void * host);

    // --- saved conversations (#179) ---
    // The quantized KV of positions [0, n) (the attention layers' and the MTP
    // block's) and the DFlash2 drafter's ring: with a checkpoint at m <= n, the
    // whole state of the sequence's first m tokens, whatever the caches held
    // since. kv_prefix_bytes is 0 when the KV is not the quantized cache
    // (OMPH_KV_F32 / OMPH_KV_HOST). The ring's entries past m are rewritten
    // before the drafter reads them (positions are injected in order).
    size_t kv_prefix_bytes(int64_t n) const;
    bool kv_prefix_save(void * host, int64_t n);
    // the first m of the n positions a buffer of kv_prefix_bytes(n) holds
    bool kv_prefix_restore(const void * host, int64_t n, int64_t m);

    // --- speculative decoding (#122) ---
    // Allocates what a verification of up to `max_tokens` tokens needs to be
    // rolled back (the alternate delta-net states, +151 MB, the replay records,
    // the conv histories, the FP16-ring backup). Call once, after construction.
    void enable_speculation(int64_t max_tokens);
    // Runs `toks` at positions pos0.. as one verification and returns every
    // row's greedy token in `argmax`. The caches and states then hold all of
    // them until commit() says how many to keep. With `logits` (speculative
    // sampling, #197) every row's logits come back instead (T x n_vocab, the
    // decode step's values bit for bit) and `argmax` stays empty.
    bool verify(const std::vector<int32_t> & toks, int64_t pos0, std::vector<int32_t> & argmax,
                std::vector<float> * logits = nullptr);
    // Keeps the first `accepted` (1 .. T) tokens of the last verification:
    // swaps in the advanced states when all are kept, else replays the kept
    // prefix onto the old ones and restores the FP16-ring slots of the rest.
    bool commit(int64_t accepted);

    // --- MTP drafting (#124), with `mtp` set at construction ---
    // The MTP KV is filled automatically: after every forward the runner keeps
    // (prefill chunks, decode steps) and at commit(), for the kept tokens.
    // Drafts `k` tokens after `token` at position `pos` (the next position to
    // fill, i.e. the length of the kept sequence): the MTP block on
    // (h_{pos-1}, token), then chained on its own output. With `logits`, every
    // draft's logits row is appended to it (validation).
    bool mtp_draft(int32_t token, int64_t pos, int64_t k, std::vector<int32_t> & drafts,
                   std::vector<float> * logits = nullptr);

    // --- DFlash2 drafting (#245), with a drafter .omph given at construction ---
    // Its KV ring gets the kept tokens' target features like the MTP KV (after
    // every forward the runner keeps, and at commit()). Drafts up to
    // dflash_block() - 1 tokens after `token` at position `pos`: one pass of
    // the block [token, MASK...] and the selector's greedy path.
    bool dflash_on() const { return dfl_block_ > 0; }
    int64_t dflash_block() const { return dfl_block_; }
    bool dflash_draft(int32_t token, int64_t pos, int64_t k, std::vector<int32_t> & drafts);
    // The next verification's drafts came from elsewhere (n-gram, #199): no length statistics.
    void dflash_no_draft() { dfl_drafted_ = 0; }
    int64_t spec_max() const { return spec_max_; }

private:
    bool checkpoint_copy(void * host, bool save);
    bool kv_prefix_copy(void * host, int64_t n, int64_t m, bool save);
    // Final norm and lm_head of the T rows in h16_, into `logits` on the host
    // (or, with `greedy` on a single-token GEMV step, only the argmax). Returns
    // once the results are on the host.
    bool lm_head(const int64_t T, std::vector<float> & logits, int32_t * greedy);

    // The first `rows` rows of logits_ to the host (synchronous).
    bool copy_logits(float * dst, const int64_t rows);

    // OMPH_CHECK_FINITE: scans the rows this call produced -- the host ones when
    // the head wrote them, else the device ones (the greedy argmax and the
    // verification paths) -- and fails on the first value that is not finite.
    bool check_logits_finite(const int64_t T, const std::vector<float> & logits);

    // lm head in vocab chunks, through the f16 dequant + GEMM path: the
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


    // il == n_layer is the MTP block (#124); kv_only writes K / V and stops.
    bool attn_layer(const int64_t il, const LayerWeights & L, const int64_t T,
                    const int64_t pos0, bool kv_only = false);

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

    // The cache an attention layer uses: Q8/Q4 (+ ring) or f32.
    struct KvView {
        bool quant = false;
        QuantKv q{};
        int64_t window = 0;
        int64_t ring = 0;
        bool k_q4 = false;
        float * k_f32 = nullptr;
        float * v_f32 = nullptr;
    };
    KvView kv_view(int64_t il) const;

    // KV write + attention, on either the f32 cache (memcpy) or the Q8/Q4 one
    // (written by attn_prep); attention_gqa then reads either.
    bool attn_impl(const KvView & kv, int64_t pos0, int64_t T);

    // Runs `side` on the side stream and `main` on the default one, both after
    // everything issued so far; the default stream continues once both are done.
    // Without overlap (the default, and prefill) they simply run in order.
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

    // The GEMVs of one input (a layer's siblings: ffn_gate + ffn_up, attn_qkv +
    // attn_gate, attn_q + attn_k + attn_v) in one launch for a verification's
    // 4 tokens (#214): bit-identical to the members' own launches, ~6.6 us less
    // per launch removed. False, with nothing issued, when it does not apply
    // (another token count, a member without a grouped body, OMPH_NO_GROUP or
    // an ablation); the caller then runs the members itself.
    struct Sibling {
        const Mat * m;
        void * y;
        int64_t n_out;
    };
    bool grouped(const void * x16, int64_t k, int64_t T, std::initializer_list<Sibling> s);

    // An FFN whose up projection takes the fused GEMM (a prefill chunk): up's
    // GEMM writes f16(silu(gate) * up) into out16 directly, bit-identical to
    // up in f32 then swiglu_f16 (#221). up_swiglu_ok says whether it applies.
    bool up_swiglu_ok(const Mat & up, int64_t n_out, int64_t k, int64_t T) const;
    bool up_swiglu(const Mat & up, const void * x16, const float * gate, void * out16, int64_t n_out, int64_t k,
                   int64_t T);

    // One fused GEMV launch for a single token, dispatched on the GGUF type.
    static bool gemv_one(const int type, const void * w, const void * x, float * y,
                         const int64_t n_out, const int64_t k, hipStream_t st);

    // One matmul: the fused GEMV for single-token steps when it is available for
    // this tensor, the small-batch GEMV for a few tokens, otherwise the f16
    // dequant + GEMM path.
    bool matmul(const Mat & m, const void * x16, float * y, const int64_t n_out,
                const int64_t k, const int64_t T);

    // Keeps the side stream only if overlapping beats running in order (#132).
    void calibrate_overlap();
    // Times single-token steps and re-runs the calibration when they turn slow (#144).
    void watch_step_begin(int64_t T);
    void watch_step_end();

    // Allocates *p on first use (the f16-path buffers, #86).
    void * lazy(void ** p, const size_t bytes);
    // x16 through OMPH_TEST_Q8ACT's int8 round trip (#213), or x16 itself
    const void * q8_input(const void * x16, int64_t k, int64_t T);
    void * q8x_ = nullptr;

    // Device pointer to the original GGUF bytes of a tensor that was not
    // repacked (repacked ones are dequantized from their layout).
    const void * raw_bytes(const std::string & name);

    // Types that have a fused GEMV kernel: for these the f16 form is never worth
    // keeping (it would be two to four times the quantized size, and they are the
    // bulk of the model).
    static bool has_gemv_type(const uint32_t type);

    void * stage_scratch(size_t bytes);
    void * stage_w(const std::string & name, int64_t row0 = 0, int64_t nrows = -1);
    int64_t stage_rows(int64_t n_out, int64_t k) const;

    Mat resolve(const std::string & name) const;

    // An f32 vector in the image, or null when the file does not have it.
    const float * resolve_f32(const std::string & name) const;

    void resolve_layers();

    const omph::runtime::EnvOptions env_;  // first: the other members' setup reads it
    omph::runtime::Allocations mem_;       // every device / pinned buffer below
    omph::gguf::File file_;
    HParams h_;
    bool use_gemv_ = false;
    omph::runtime::Linear linear_;  // the f16 WMMA GEMM (#132)
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
    // embed()'s token ids and gathered rows on the device, grown on demand (#223)
    int32_t * embd_ids_ = nullptr;
    void * embd_stage_ = nullptr;
    int64_t embd_cap_ = 0;
    bool last_logits_only_ = false;
    static constexpr int64_t kHeadRows = 32;  // logits_ rows (multiple of 4)
    std::vector<int64_t> kv_index_;
    std::vector<char> conv_flip_;
    // --- MTP (#124) ---
    bool mtp_ = false;
    LayerWeights mtp_L_;        // blk.<n_layer>: an attention block + FFN
    Mat mtp_eh_;                // nextn.eh_proj: [enorm(embed); hnorm(h)] -> n_embd
    const float * mtp_enorm_ = nullptr;
    const float * mtp_hnorm_ = nullptr;
    const float * mtp_head_norm_ = nullptr;
    void * mtp_kq_ = nullptr;   // its own Q8/Q4 cache, no FP16 ring
    // The MTP cache as a window (#171): mtp_cap_ rows. The first kMtpSink rows
    // keep positions [0, kMtpSink) for good (the attention sinks: without them
    // drafts at 100k lost 7 % of the decode speed); row kMtpSink + r holds
    // position mtp_base_ + r (mtp_base_ >= kMtpSink), and positions up to
    // mtp_next_ are filled. When a write would pass the last row, the last
    // mtp_window positions move to the front of the window (no overlap: it holds
    // two windows). A prompt fills the MTP KV only for its sinks and from
    // mtp_fill_from_ (its last window); the drafts attend over sinks + window.
    static constexpr int64_t kMtpSink = 16;
    int64_t mtp_window_ = 0;  // 0: the whole context
    int64_t mtp_cap_ = 0;
    int64_t mtp_base_ = 0;
    int64_t mtp_next_ = 0;
    int64_t mtp_fill_from_ = 0;
    bool mtp_rows(int64_t pos0, int64_t T, bool kv_only);
    int64_t mtp_row(int64_t pos0) const;
    static constexpr size_t kMtpHeader = 3 * sizeof(int64_t);
    // the MTP rows a saved conversation of n positions holds (#179, #171)
    int64_t mtp_saved_rows(const int64_t n) const { return mtp_window_ > 0 ? mtp_cap_ : n; }
    void * mtp_ks_ = nullptr;
    void * mtp_vq_ = nullptr;
    void * mtp_vs_ = nullptr;
    void * mtp_hlast_ = nullptr;    // output_norm(x) rows of the last forward (the h_p)
    void * mtp_hin_ = nullptr;      // the h rows an MTP pass reads: h_{p-1} per position
    void * mtp_pending_ = nullptr;  // h of the last kept position (zeros at the start)
    void * mtp_g_ = nullptr;        // the MTP block's own normed output (chaining)
    std::vector<int32_t> last_toks_;
    // #217: the share of recent tokens (the prompt, the kept ones) outside the
    // drafts' vocabulary prefix, averaged over ~64 tokens. The drafts use the
    // prefix while it stays under kDraftOovMax: English and code sit at 0-2 %,
    // Italian at ~16 % (where a prefix draft misses too often to pay off).
    static constexpr double kDraftOovMax = 0.05;
    double draft_oov_ = 0.0;
    void observe_draft(const int32_t * toks, int64_t n);
    int64_t last_pos0_ = 0;
    // The MTP block over T positions: embeddings of `toks`, h rows `h_in`,
    // positions pos0..; kv_only stops after the K / V write, `head` also runs
    // shared_head_norm + lm_head and returns the greedy token.
    bool mtp_block(const int32_t * toks, const float * h_in, int64_t pos0, int64_t T,
                   bool kv_only, int32_t * argmax = nullptr, std::vector<float> * logits = nullptr,
                   const ForwardInputs * in = nullptr);
    // the inputs of the forward being run (#160); its M-RoPE positions on the device
    const ForwardInputs * inputs_ = nullptr;
    void * mpos_dev_ = nullptr;
    int64_t rope_delta_ = 0;
    // embeddings of tokens [0, T) into dst (n_embd f32 rows): token rows from
    // the table, image rows from `in`
    bool embed(const int32_t * toks, int64_t T, const ForwardInputs * in, float * dst);
    // The KV fill of the first `keep` tokens of the last forward.
    bool mtp_fill(int64_t keep);

    // DFlash2 (#245; model/dflash.cc). The drafter's tensors sit in the weight
    // image under "dflash." + their name.
    struct DflashLayer {
        const float * attn_norm = nullptr;
        const float * ffn_norm = nullptr;
        const float * q_norm = nullptr;
        const float * k_norm = nullptr;
        const float * attn_conv_base = nullptr;  // (n_embd, 2 taps, 2 sides)
        const float * ffn_conv_base = nullptr;
        Mat q, k, v, o, attn_conv, ffn_conv, gate, up, down;
    };
    void dflash_load();  // hyperparameters, tensors, buffers
    // The target features of rows [0, T) of the last forward into the ring at pos0..
    bool dflash_inject(int64_t pos0, int64_t T);
    bool dflash_layer(const DflashLayer & L, int64_t l, int64_t B, int64_t pos);
    std::unique_ptr<omph::gguf::File> dft_file_;
    int64_t dfl_block_ = 0;             // 0: no drafter
    std::vector<int64_t> dfl_capture_;  // target layers whose output is a feature, in feature order
    std::vector<DflashLayer> dfl_;
    Mat dfl_fc_, dfl_gate_w_;
    const float * dfl_enc_norm_ = nullptr;
    const float * dfl_out_norm_ = nullptr;
    const void * dfl_sel_prev_ = nullptr;  // repacked Q4_K, one 256-weight row per token
    const void * dfl_sel_next_ = nullptr;
    int32_t dfl_mask_ = 0;
    int64_t dfl_ne_ = 0;       // drafter width (5120)
    int64_t dfl_ff_ = 0;       // its FFN (17408)
    int64_t dfl_swa_ = 0;      // sliding window (2048) = ring slots
    size_t dfl_ring_bytes_ = 0;  // each of dfl_ring_k_ / dfl_ring_v_
    int64_t dfl_group_ = 0;    // conv group size (16)
    float dfl_eps_ = 0.0f;
    float dfl_theta_ = 0.0f;
    void * dfl_feat_ = nullptr;    // (rows, n_capture * n_embd) f16: the last forward's features
    void * dfl_ring_k_ = nullptr;  // layers x slots x (8 x 128) f16
    void * dfl_ring_v_ = nullptr;
    int32_t * dfl_tags_ = nullptr; // slots: the position a slot holds (-1: none)
    void * dfl_bk_ = nullptr;      // the block's K / V (B, 8 x 128) f32
    void * dfl_bv_ = nullptr;
    void * dfl_dyn_ = nullptr;     // conv coefficients (B, 4 x groups) f32
    void * dfl_sgate_ = nullptr;   // selector gate (B, 256) f32
    void * dfl_attn_ws_ = nullptr; // attention partials
    void * dfl_logits_ = nullptr;  // (B - 1, n_vocab) f32
    int32_t * dfl_ids_ = nullptr;  // (B - 1, 16)
    float * dfl_vals_ = nullptr;
    float * dfl_scores_ = nullptr; // (B - 1, 16, 16)
    // Adaptive draft length: per draft position n, an average of "the first n
    // drafts were all kept" over the steps that drafted n; a step drafts the
    // positions above OMPH_DFLASH_KEEP, all of them every 8th step (exploring).
    std::vector<double> dfl_keep_;
    int64_t dfl_drafted_ = 0;      // drafts of the last dflash_draft
    int64_t dfl_steps_ = 0;
    void dflash_observe(int64_t kept_drafts);
    std::vector<__half *> state_cur_;  // recurrent layers: the state the next step reads (f16, #273)
    std::vector<__half *> state_alt_;  // speculation: the buffer a verification writes
    std::vector<int64_t> rec_index_;  // recurrent layer -> 0.. (-1 for attention)
    int64_t spec_max_ = 0;            // 0: speculation off
    bool verifying_ = false;
    int64_t verify_tokens_ = 0;
    int64_t verify_pos0_ = 0;
    std::vector<int32_t> * verify_argmax_ = nullptr;  // verify(): rows' tokens from lm_head
    void * spec_keys_ = nullptr;                      // their packed argmax keys (device)
    static constexpr int64_t kVerifyRowsMax = 32;
    void * replay_pool_ = nullptr;
    void * conv_hist_pool_ = nullptr;
    void * ring_backup_k_ = nullptr;
    void * ring_backup_v_ = nullptr;
    std::vector<uint8_t> check_ring_k_;  // OMPH_SPEC_CHECK: the rings before a verification
    std::vector<uint8_t> check_ring_v_;
    // OMPH_SPEC_CHECK: replaying all recorded tokens reproduces gdn_step's state.
    bool check_replay();
    int64_t check_tokens_ = 0;           // the T of the verification being checked
    void * spec_check_scratch_ = nullptr;
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
    int64_t kv_ring_ = 0;    // the FP16 ring's slots: window + kKvRingExtra (#161)
    bool kv_q8q4_ = false;
    // K stored as Q4 per attention layer (#81, #175), and each layer's offset
    // into kv_kq_ (a Q4 layer's K takes half the bytes)
    std::vector<uint8_t> kv_k4_;
    std::vector<size_t> kv_kq_off_;
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
    size_t tmp_logits_bytes_ = 0;
    size_t head16_bytes_ = 0;
    std::map<std::string, void *> f16_cache_;  // into mem_
    size_t scratch_bytes_ = 0;    // the largest f16 weight slice (stage_rows)
    bool scratch_ready_ = false;  // allocated on first use (a weight without a decoder, #141)
    int64_t gemm_min_ = 0;        // tokens from which a --gemv runner uses the GEMM path
    // T tokens through the multi-token GEMVs: below gemm_min_, and always in a
    // speculative verification (up to 16 tokens), whose rows must equal the
    // single-token GEMV's bit for bit (#161, #244)
    bool gemv_path(const int64_t T) const { return use_gemv_ && T > 1 && (T < gemm_min_ || verifying_); }
    int64_t key_chunk_ = 0;       // attention keys per split, fixed for the run (#136)
    // side-stream watchdog (#144)
    static constexpr double kSlowStep = 1.35;
    static constexpr int kMaxRecalibrations = 8;
    hipEvent_t mon_a_ = nullptr;
    hipEvent_t mon_b_ = nullptr;
    bool mon_open_ = false;
    bool recal_pending_ = false;
    double step_ref_ = 0.0;       // slow average of the normal single-token steps, ms
    int slow_steps_ = 0;
    int recalibrations_ = 0;
    int64_t watched_steps_ = 0;
    hipStream_t test_bad_side_ = nullptr;  // OMPH_TEST_BAD_SIDE
};

} // namespace omph::model
