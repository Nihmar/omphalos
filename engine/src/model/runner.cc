// Runner: load (upload + repack), buffers, KV cache and states, weight
// resolution, the f16 staging path, diagnostics.
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

Runner::Runner(const std::string & path, const int64_t max_tokens, const bool use_gemv,
                const omph::runtime::EnvOptions & env, const bool last_logits_only,
                const int64_t kv_capacity)
    : env_(env), file_(path), h_(read_hparams(file_)), use_gemv_(use_gemv) {
    timer_stage_.enable(env_.phases);
    timer_gemm_.enable(env_.phases);
    timer_gemv_.enable(env_.phases);
    timer_block_.enable(env_.phases);
    if (const int w = omph::kernels::kernel_wave_size(); w != 32) {
        throw std::runtime_error("kernels built for wave size " + std::to_string(w) +
                                 "; they need wave32");
    }
    const int64_t ne = h_.n_embd;
    const int64_t T = max_tokens;
    max_tokens_ = max_tokens;
    const int64_t ssm_q = h_.ssm_n_kh * h_.ssm_s;              // 2048
    const int64_t ssm_v = h_.ssm_n_vh * (h_.ssm_inner / h_.ssm_n_vh);  // 6144
    const int64_t attn_q = h_.n_head * h_.head_dim;            // 6144
    const int64_t attn_kv = h_.n_head_kv * h_.head_dim;        // 1024
    const int64_t ssm_channels = ssm_q * 2 + ssm_v;            // 10240 (gdn qkv)
    const int64_t attn_fused = h_.n_head * 2 * h_.head_dim;    // 12288 (q | gate)
    const int64_t fused = std::max(ssm_channels, attn_fused);

    // Per-tensor upload: the types with a fused GEMV kernel are repacked on
    // the host first (PLAN.md §8.3), everything else goes up verbatim. Only
    // one copy of each tensor is kept in VRAM.
    {
        struct Place {
            std::string name;
            size_t off = 0;
        };
        std::vector<Place> places;
        size_t total = 0;
        for (const omph::gguf::TensorInfo & t : file_.tensors()) {
            // The MTP block (blk.<n_layer>.*) is not run until M6 (#92).
            if (!in_stack(t.name)) {
                continue;
            }
            // The token embedding is a row gather, not a matmul: it stays in
            // pinned host memory and the dequant kernel reads the one row a
            // token needs (10 KB at IQ2_S) over PCIe, which keeps 388 MiB out
            // of VRAM (#92).
            if (t.name == "token_embd.weight") {
                embd_host_ = mem_.host((size_t) t.nbytes, "cannot allocate the host embedding");
                std::memcpy(embd_host_, file_.tensor_data(t), (size_t) t.nbytes);
                continue;
            }
            Place p;
            p.name = t.name;
            p.off = total;
            const int64_t bb = omph::format::quant_block_bytes(t.type);
            // Sized from the layout: the repack itself runs once, at upload.
            const int64_t packed_bytes =
                use_gemv_ && bb > 0 && t.nbytes % (uint64_t) bb == 0
                    ? omph::format::repacked_bytes(t.type, (int64_t) (t.nbytes / (uint64_t) bb))
                    : 0;
            if (packed_bytes > 0) {
                GemvEntry e;
                e.off = total;
                e.bytes = (size_t) packed_bytes;
                e.rows = (int64_t) t.ne[1];
                e.k = (int64_t) t.ne[0];
                e.type = t.type;
                e.has_b4 = !env_.no_b4 &&
                           (t.type == 12 || t.type == 18 || t.type == 21 || t.type == 23);
                gems_[t.name] = e;
                total += ((size_t) packed_bytes + 255) & ~(size_t) 255;
                places.push_back(p);
                continue;
            }
            total += ((size_t) t.nbytes + 255) & ~(size_t) 255;
            places.push_back(p);
        }
        dev_weights_ = mem_.device(total, "cannot allocate the weight image");
        auto * base = static_cast<uint8_t *>(dev_weights_);
        for (const Place & p : places) {
            const omph::gguf::TensorInfo * t = file_.tensor(p.name);
            const int64_t bb = omph::format::quant_block_bytes(t->type);
            const bool repacked = gems_.count(p.name) != 0 && bb > 0;
            std::vector<uint8_t> packed;
            const void * src = file_.tensor_data(*t);
            size_t bytes = (size_t) t->nbytes;
            if (repacked) {
                if (!omph::format::repack_any(t->type, src,
                                              (int64_t) (t->nbytes / (uint64_t) bb), packed) ||
                    packed.size() != gems_.at(p.name).bytes) {
                    throw std::runtime_error("repack failed for " + p.name);
                }
                src = packed.data();
                bytes = packed.size();
            }
            if (hipMemcpy(base + p.off, src, bytes, hipMemcpyHostToDevice) != hipSuccess) {
                throw std::runtime_error("cannot upload " + p.name);
            }
            off_[p.name] = p.off;
        }
    }

    resolve_layers();

    // The f16 staging scratch holds one layer's weights on the dequant +
    // hipBLASLt path, so it is sized to the largest per-layer sum of the
    // tensors that can take it: every 2D weight of a block (~770 MiB, #93),
    // and with --gemv only the ones without a GEMV (the BF16 beta / alpha
    // projections in a prefill, ~1 MB a layer, #86).
    size_t scratch_bytes = 1 << 20;
    {
        std::unordered_map<std::string, size_t> per_layer;
        for (const omph::gguf::TensorInfo & t : file_.tensors()) {
            if (t.ne.size() < 2 || t.type == 0 || gems_.count(t.name) != 0 ||
                t.name.rfind("blk.", 0) != 0 || !in_stack(t.name)) {
                continue;  // f32 vectors, GEMV tensors, embedding / head
            }
            const std::string layer = t.name.substr(0, t.name.find('.', 4));
            per_layer[layer] += (((size_t) numel(t) * 2) + 255) & ~(size_t) 255;
        }
        for (const auto & kv : per_layer) {
            scratch_bytes = std::max(scratch_bytes, kv.second);
        }
    }
    scratch_.init(mem_.device(scratch_bytes, "cannot allocate the weight scratch"),
                  scratch_bytes, env_.trace_alloc);
    const auto alloc = [&](void ** p, const size_t bytes) { *p = mem_.device(bytes); };
    alloc(&x_, T * ne * 4);
    alloc(&cur_, T * ne * 4);
    alloc(&resid_, T * ne * 4);
    alloc(&blk_, T * ne * 4);
    alloc(&h16_, T * ne * 2);
    alloc(&fused_, T * fused * 4);
    alloc(&z_, T * ssm_v * 4);
    alloc(&q_, T * attn_q * 4);
    alloc(&gate_, T * attn_q * 4);
    alloc(&k_, T * attn_kv * 4);  // attention only: the delta-net reads qkv itself
    alloc(&v_, T * attn_kv * 4);
    alloc(&ffn1_, T * h_.n_ff * 4);
    alloc(&ffn2_, T * h_.n_ff * 4);
    alloc(&ffn16_, T * h_.n_ff * 2);
    last_logits_only_ = last_logits_only;
    // The head's rows go to the host in batches of kHeadRows: a whole chunk
    // of f32 logits would be T x 248320 x 4 = 508 MB at T = 512 (#93).
    alloc(&logits_, (size_t) (last_logits_only ? 1 : std::min<int64_t>(T, kHeadRows)) *
                         h_.n_vocab * 4);
    alloc(&argmax_key_, sizeof(unsigned long long));
    // tmp_logits_, head16_ (the vocab-chunked f16 lm_head) and raw_stage_ (raw
    // bytes of a repacked tensor for the f16 path) are allocated on first
    // use: the GEMV decode never touches them (#86).
    tmp_logits_bytes_ = (size_t) T * std::min<int64_t>(h_.n_vocab, 32768) * 4;
    head16_bytes_ = (size_t) std::min<int64_t>(h_.n_vocab, 32768) * ne * 2;
    const int64_t kvcap = kv_capacity > 0 ? kv_capacity : T;
    max_seq_ = kvcap;
    int64_t n_kv = 0;
    for (int64_t il = 0; il < h_.n_layer; ++il) {
        kv_index_.push_back(layers_[(size_t) il].recurrent ? -1 : n_kv++);
    }
    // The KV cache is sized by the whole sequence, the activations by the
    // chunk: that is what lets a long prompt run in pieces.
    // Q8/Q4 by default (#69): validated to 32k under llama.cpp's q8_0/q4_0
    // budget at 4.9x less VRAM than f32 (#58, #61).
    kv_host_ = env_.kv_host;
    kv_q8q4_ = !kv_host_ && !env_.kv_f32;
    if (kv_host_) {
        // Validation reference only: the exact f32 cache in pinned host RAM
        // (4.29 GB at 32k does not fit beside the weights), and one layer's
        // worth of it staged into VRAM before each attention.
        const size_t bytes = (size_t) n_kv * kvcap * attn_kv * 4;
        kv_k_ = mem_.host(bytes, "cannot allocate the host KV cache");
        kv_v_ = mem_.host(bytes, "cannot allocate the host KV cache");
        alloc(&kv_stage_k_, (size_t) kvcap * attn_kv * 4);
        alloc(&kv_stage_v_, (size_t) kvcap * attn_kv * 4);
    } else if (!kv_q8q4_) {
        alloc(&kv_k_, (size_t) n_kv * kvcap * attn_kv * 4);
        alloc(&kv_v_, (size_t) n_kv * kvcap * attn_kv * 4);
    } else {
        // K Q8 + V Q4 with 32-element blocks: 272 B and 144 B per head per
        // token, against 1024 B each in f32 (PLAN.md §13).
        const int64_t nblk = h_.head_dim / 32;
        // OMPH_KV_K4=1: K in V's Q4 format too (#81, experiment).
        kv_k4_ = env_.kv_k4;
        alloc(&kv_kq_, (size_t) n_kv * kvcap * attn_kv / (kv_k4_ ? 2 : 1));
        alloc(&kv_ks_, (size_t) n_kv * kvcap * h_.n_head_kv * nblk * 2);
        alloc(&kv_vq_, (size_t) n_kv * kvcap * attn_kv / 2);
        alloc(&kv_vs_, (size_t) n_kv * kvcap * h_.n_head_kv * nblk * 2);
        // FP16 window: the last `kv_window_` tokens exactly, in the same
        // rotated basis, in a ring (PLAN §13.4). 128 by default: it keeps the
        // KL under llama.cpp's q8_0/q4_0 up to 32k for 8.4 MB (#61);
        // OMPH_KV_WINDOW=0 turns it off.
        kv_window_ = env_.kv_window;
        alloc(&kv_k16_, (size_t) n_kv * kv_window_ * attn_kv * 2);
        alloc(&kv_v16_, (size_t) n_kv * kv_window_ * attn_kv * 2);
    }
    overlap_ = use_gemv_ && !env_.no_overlap;
    if (overlap_ && (hipStreamCreateWithFlags(&side_, hipStreamNonBlocking) != hipSuccess ||
                     hipEventCreateWithFlags(&ev_fork_, hipEventDisableTiming) != hipSuccess ||
                     hipEventCreateWithFlags(&ev_join_, hipEventDisableTiming) != hipSuccess)) {
        throw std::runtime_error("cannot create the side stream");
    }
    attn_work_bytes_ = omph::kernels::attention_gqa_work_bytes(T, h_.n_head, h_.n_head_kv,
                                                               h_.head_dim);
    alloc(&attn_work_, attn_work_bytes_);
    // Delta-net state + the two conv tails, for the recurrent layers only
    // (the attention layers have none), in one allocation (#92).
    conv_flip_.assign((size_t) h_.n_layer, 0);
    {
        const int64_t n_state = h_.ssm_n_vh * h_.ssm_s * h_.ssm_s;
        const int64_t n_conv = (h_.ssm_conv_k - 1) * ssm_channels;
        const size_t per_layer = (((size_t) (n_state + 2 * n_conv) * 4) + 255) & ~(size_t) 255;
        const size_t n_rec = (size_t) std::count(kv_index_.begin(), kv_index_.end(), -1);
        alloc(&state_pool_, std::max<size_t>(n_rec, 1) * per_layer);
        if (hipMemset(state_pool_, 0, std::max<size_t>(n_rec, 1) * per_layer) != hipSuccess) {
            throw std::runtime_error("cannot clear the delta-net state");
        }
        size_t at = 0;
        for (int64_t il = 0; il < h_.n_layer; ++il) {
            if (kv_index_[il] >= 0) {
                states_.push_back(nullptr);
                continue;
            }
            states_.push_back(static_cast<uint8_t *>(state_pool_) + at);
            at += per_layer;
        }
        // The recurrent state proper sits after the two conv tails; speculation
        // swaps it with a second buffer (#122).
        state_cur_.assign((size_t) h_.n_layer, nullptr);
        state_alt_.assign((size_t) h_.n_layer, nullptr);
        for (int64_t il = 0; il < h_.n_layer; ++il) {
            if (states_[(size_t) il] != nullptr) {
                state_cur_[(size_t) il] = reinterpret_cast<float *>(
                    static_cast<uint8_t *>(states_[(size_t) il]) + 2 * n_conv * 4);
            }
        }
    }
    if (env_.timing) {
        size_t free_b = 0;
        size_t total_b = 0;
        (void) hipMemGetInfo(&free_b, &total_b);
        std::fprintf(stderr, "vram: %zu MiB used of %zu MiB after load\n",
                     (total_b - free_b) >> 20, total_b >> 20);
    }
}

// Memory is owned by mem_; the stream and the events are released here.
Runner::~Runner() {
    if (side_ != nullptr) {
        (void) hipStreamDestroy(side_);
        (void) hipEventDestroy(ev_fork_);
        (void) hipEventDestroy(ev_join_);
    }
}

// OMPH_PHASES: GPU time per phase since the last report. Each GEMV is timed
// on its own stream, so GEMVs overlapped on the side stream count in full
// and the gemv total can exceed the step.
void Runner::report_phases() {
    if (!env_.phases) {
        return;
    }
    (void) hipDeviceSynchronize();
    const size_t n_stage = t_stage_.size();
    const size_t n_gemm = t_gemm_.size();
    const size_t n_gemv = t_gemv_.size();
    const size_t n_block = t_block_.size();
    const double ms_stage = PhaseTimer::total_ms(t_stage_);
    const double ms_gemm = PhaseTimer::total_ms(t_gemm_);
    const double ms_gemv = PhaseTimer::total_ms(t_gemv_);
    const double ms_block = PhaseTimer::total_ms(t_block_);
    std::fprintf(stderr,
                 "phases: stage_w %.1f ms  gemm(f16) %.1f ms  gemv %.1f ms  blocks %.1f ms "
                 "| calls %zu/%zu/%zu/%zu\n",
                 ms_stage, ms_gemm, ms_gemv, ms_block, n_stage, n_gemm, n_gemv, n_block);
}

void Runner::step_event(hipEvent_t a, hipEvent_t b) {
    (void) hipEventRecord(b, nullptr);
    (void) hipEventSynchronize(b);
    float ms = 0.0f;
    (void) hipEventElapsedTime(&ms, a, b);
    std::fprintf(stderr, "step gpu %.2f ms\n", ms);
    (void) hipEventDestroy(a);
    (void) hipEventDestroy(b);
}

// Returns false: every caller is a bool function that reports failure.
bool Runner::fail(const char * msg) {
    // The pending HIP error (if any) is usually the actual cause.
    const hipError_t err = hipGetLastError();
    std::fprintf(stderr, "%s%s%s\n", msg, err != hipSuccess ? ": " : "",
                 err != hipSuccess ? hipGetErrorString(err) : "");
    return false;
}

// False for the tensors of the blocks past the stack (the MTP block).
bool Runner::in_stack(const std::string & name) const {
    if (name.rfind("blk.", 0) != 0) {
        return true;
    }
    return std::atoll(name.c_str() + 4) < h_.n_layer;
}

// Allocates *p on first use (the f16-path buffers, #86).
void * Runner::lazy(void ** p, const size_t bytes) {
    if (*p == nullptr) {
        *p = mem_.device(bytes, "out of VRAM (f16-path buffer)");
    }
    return *p;
}

// raw_stage_ holds the original bytes of any one repacked tensor.
size_t Runner::raw_stage_bytes() const {
    size_t n = 0;
    for (const auto & kv : gems_) {
        n = std::max(n, (size_t) file_.tensor(kv.first)->nbytes);
    }
    return n;
}

// Device pointer to the original GGUF bytes: they live in the image unless
// the tensor was repacked, in which case they are staged from the file.
const void * Runner::raw_bytes(const std::string & name) {
    const auto it = gems_.find(name);
    if (it == gems_.end()) {
        return static_cast<const uint8_t *>(dev_weights_) + off_.at(name);
    }
    const omph::gguf::TensorInfo * t = file_.tensor(name);
    if (hipMemcpy(lazy(&raw_stage_, raw_stage_bytes()), file_.tensor_data(*t),
                  (size_t) t->nbytes,
                  hipMemcpyHostToDevice) != hipSuccess) {
        throw std::runtime_error("cannot stage " + name);
    }
    return raw_stage_;
}

// Types that have a fused GEMV kernel: for these the f16 form is never worth
// keeping (it would be two to four times the quantized size, and they are the
// bulk of the model).
bool Runner::has_gemv_type(const uint32_t type) {
    switch (type) {
        case 10:
        case 12:
        case 14:
        case 16:
        case 17:
        case 18:
        case 21:
        case 22:
        case 23:
        case 29:
        case 30: return true;
        default: return false;
    }
}

void * Runner::stage_w(const std::string & name) {
    const omph::gguf::TensorInfo * t = file_.tensor(name);
    if (t == nullptr) {
        throw std::runtime_error("missing tensor " + name);
    }
    // Tensors of a type without a fused GEMV were converted to f16 on every
    // call (1.6 ms of every decode step for the 18.6 MiB IQ1_M tensor before
    // it got one). Convert once and keep it, up to 256 MiB of f16 per
    // tensor; the big GEMV types (whose f16 form would not fit VRAM) never
    // take this path.
    const int64_t n_elems = numel(*t);
    bool cacheable = !env_.no_f16_cache &&
                     !has_gemv_type(t->type) && n_elems * 2 <= (256 << 20);
    if (cacheable) {
        const auto it = f16_cache_.find(name);
        if (it != f16_cache_.end()) {
            if (env_.trace_stage) {
                std::fprintf(stderr, "stage %-40s CACHED\n", name.c_str());
            }
            return it->second;
        }
    }
    if (env_.trace_stage) {
        std::fprintf(stderr, "stage %-40s type %u %lld B cacheable %d\n", name.c_str(),
                     (unsigned) t->type, (long long) t->nbytes, cacheable ? 1 : 0);
    }
    timer_stage_.start();
    const int64_t n = numel(*t);
    void * dst = nullptr;
    if (cacheable) {
        // Long contexts fill VRAM: caching is an optimization, never a
        // reason to fail. Fall back to the per-call staging buffer
        // (try_device clears the out-of-memory error, which the next kernel
        // wrapper's hipGetLastError() would otherwise report as its own).
        dst = mem_.try_device((size_t) n * 2);
        if (dst == nullptr) {
            cacheable = false;
            dst = scratch_.alloc((size_t) n * 2);
        }
    } else {
        dst = scratch_.alloc((size_t) n * 2);
    }
    if (env_.trace_alloc) {
        std::fprintf(stderr, "stage %-40s %10lld elems  ne=[", name.c_str(), (long long) n);
        for (const uint64_t d : t->ne) {
            std::fprintf(stderr, "%llu,", (unsigned long long) d);
        }
        std::fprintf(stderr, "] type=%u\n", t->type);
    }
    if (!omph::kernels::dequantize(t->type, raw_bytes(name), dst, n, true, nullptr)) {
        throw std::runtime_error("dequant failed for " + name);
    }
    timer_stage_.stop(t_stage_);
    if (cacheable) {
        f16_cache_[name] = dst;
    }
    return dst;
}

Runner::Mat Runner::resolve(const std::string & name) const {
    Mat m;
    m.name = name;
    m.t = file_.tensor(name);
    const auto off = off_.find(name);
    if (m.t != nullptr && off != off_.end()) {
        m.dev = static_cast<const uint8_t *>(dev_weights_) + off->second;
    }
    const auto g = gems_.find(name);
    if (g != gems_.end()) {
        m.gemv = &g->second;
    }
    return m;
}

// An f32 vector in the image, or null when the file does not have it.
const float * Runner::resolve_f32(const std::string & name) const {
    return static_cast<const float *>(resolve(name).dev);
}

void Runner::resolve_layers() {
    layers_.resize((size_t) h_.n_layer);
    for (int64_t il = 0; il < h_.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        LayerWeights & L = layers_[(size_t) il];
        L.recurrent = file_.tensor(p + "ssm_a") != nullptr;
        L.attn_norm = resolve_f32(p + "attn_norm.weight");
        L.post_norm = resolve_f32(p + "post_attention_norm.weight");
        if (L.attn_norm == nullptr || L.post_norm == nullptr) {
            throw std::runtime_error("missing layer norms in " + p);
        }
        L.ffn_up = resolve(p + "ffn_up.weight");
        L.ffn_gate = resolve(p + "ffn_gate.weight");
        L.ffn_down = resolve(p + "ffn_down.weight");
        if (L.recurrent) {
            L.dt_bias = resolve_f32(p + "ssm_dt.bias");
            L.ssm_a = resolve_f32(p + "ssm_a");
            L.ssm_norm = resolve_f32(p + "ssm_norm.weight");
            L.conv_w = resolve_f32(p + "ssm_conv1d.weight");
            const Mat wb = resolve(p + "ssm_beta.weight");
            const Mat wa = resolve(p + "ssm_alpha.weight");
            if (L.dt_bias == nullptr || L.ssm_a == nullptr || L.ssm_norm == nullptr ||
                L.conv_w == nullptr || wb.t == nullptr || wa.t == nullptr ||
                wb.t->type != 30 || wa.t->type != 30) {
                throw std::runtime_error("delta-net tensors missing in " + p +
                                         " (ssm_beta / ssm_alpha must be BF16)");
            }
            L.w_beta = static_cast<const uint16_t *>(wb.dev);
            L.w_alpha = static_cast<const uint16_t *>(wa.dev);
            L.attn_qkv = resolve(p + "attn_qkv.weight");
            L.attn_gate = resolve(p + "attn_gate.weight");
            L.ssm_out = resolve(p + "ssm_out.weight");
        } else {
            L.q_norm = resolve_f32(p + "attn_q_norm.weight");
            L.k_norm = resolve_f32(p + "attn_k_norm.weight");
            if (L.q_norm == nullptr || L.k_norm == nullptr) {
                throw std::runtime_error("QK norms missing in " + p);
            }
            L.attn_q = resolve(p + "attn_q.weight");
            L.attn_k = resolve(p + "attn_k.weight");
            L.attn_v = resolve(p + "attn_v.weight");
            L.attn_output = resolve(p + "attn_output.weight");
        }
    }
    head_ = resolve("output.weight");
    out_norm_ = resolve_f32("output_norm.weight");
    const omph::gguf::TensorInfo * te = file_.tensor("token_embd.weight");
    if (head_.t == nullptr || out_norm_ == nullptr || te == nullptr || embd_host_ == nullptr) {
        throw std::runtime_error("output.weight / output_norm / token_embd missing");
    }
    embd_type_ = te->type;
    embd_row_bytes_ = (int64_t) (te->nbytes / te->ne[1]);
}

// --- speculative decoding (#122) ---

void Runner::enable_speculation(const int64_t max_tokens) {
    if (spec_max_ > 0 || max_tokens <= 1) {
        return;
    }
    if (max_tokens > max_tokens_) {
        throw std::runtime_error("speculation: verification longer than the activations");
    }
    spec_max_ = max_tokens;
    const int64_t n_kh = h_.ssm_n_kh;
    const int64_t n_vh = h_.ssm_n_vh;
    const int64_t channels = 2 * n_kh * h_.ssm_s + h_.ssm_inner;
    const int64_t n_state = n_vh * h_.ssm_s * h_.ssm_s;
    int64_t n_rec = 0;
    rec_index_.assign((size_t) h_.n_layer, -1);
    for (int64_t il = 0; il < h_.n_layer; ++il) {
        if (state_cur_[(size_t) il] != nullptr) {
            rec_index_[(size_t) il] = n_rec++;
        }
    }
    auto * alt = static_cast<float *>(
        mem_.device((size_t) std::max<int64_t>(n_rec, 1) * n_state * 4,
                    "out of VRAM (speculation states)"));
    for (int64_t il = 0; il < h_.n_layer; ++il) {
        if (rec_index_[(size_t) il] >= 0) {
            state_alt_[(size_t) il] = alt + rec_index_[(size_t) il] * n_state;
        }
    }
    replay_pool_ = mem_.device((size_t) std::max<int64_t>(n_rec, 1) * max_tokens *
                               omph::kernels::gdn_replay_floats(n_vh, n_kh) * 4);
    conv_hist_pool_ = mem_.device((size_t) std::max<int64_t>(n_rec, 1) *
                                  (h_.ssm_conv_k - 1 + max_tokens) * channels * 4);
    if (kv_q8q4_ && kv_window_ > 0) {
        const int64_t n_kv = (int64_t) std::count_if(kv_index_.begin(), kv_index_.end(),
                                                     [](int64_t k) { return k >= 0; });
        const size_t ring = (size_t) n_kv * kv_window_ * h_.n_head_kv * h_.head_dim * 2;
        ring_backup_k_ = mem_.device(ring);
        ring_backup_v_ = mem_.device(ring);
    }
    // A verification brings every row back, also with --last-logits.
    if (last_logits_only_ || max_tokens > kHeadRows) {
        logits_ = mem_.device((size_t) std::max<int64_t>(max_tokens, kHeadRows) * h_.n_vocab * 4);
    }
}

bool Runner::verify(const std::vector<int32_t> & toks, const int64_t pos0,
                    std::vector<int32_t> & argmax) {
    const int64_t T = (int64_t) toks.size();
    if (spec_max_ == 0 || T < 1 || T > spec_max_) {
        return fail("verify: speculation not enabled, or too many tokens");
    }
    const int64_t n_kv = (int64_t) std::count_if(kv_index_.begin(), kv_index_.end(),
                                                 [](int64_t k) { return k >= 0; });
    const int64_t row_bytes = h_.n_head_kv * h_.head_dim * 2;
    if (ring_backup_k_ != nullptr &&
        !omph::kernels::kv_ring_copy(kv_k16_, kv_v16_, ring_backup_k_, ring_backup_v_, n_kv,
                                     kv_window_, row_bytes, pos0, T, nullptr)) {
        return fail("verify: ring backup failed");
    }
    if (env_.spec_check && ring_backup_k_ != nullptr) {
        const size_t ring = (size_t) n_kv * kv_window_ * row_bytes;
        check_ring_k_.resize(ring);
        check_ring_v_.resize(ring);
        if (hipMemcpy(check_ring_k_.data(), kv_k16_, ring, hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(check_ring_v_.data(), kv_v16_, ring, hipMemcpyDeviceToHost) != hipSuccess) {
            return fail("verify: ring check copy failed");
        }
    }
    verifying_ = true;
    verify_tokens_ = T;
    verify_pos0_ = pos0;
    std::vector<float> logits;
    const bool ok = forward(toks, logits, std::string(), pos0, true, nullptr);
    verifying_ = false;
    if (!ok) {
        return false;
    }
    argmax.resize((size_t) T);
    for (int64_t t = 0; t < T; ++t) {
        const float * row = logits.data() + t * h_.n_vocab;
        int64_t best = 0;
        for (int64_t i = 1; i < h_.n_vocab; ++i) {
            if (row[i] > row[best]) {
                best = i;
            }
        }
        argmax[(size_t) t] = (int32_t) best;
    }
    return true;
}

bool Runner::commit(const int64_t accepted) {
    const int64_t T = verify_tokens_;
    if (T == 0 || accepted < 1 || accepted > T) {
        return fail("commit: no verification, or accepted out of range");
    }
    check_tokens_ = T;
    verify_tokens_ = 0;
    if (env_.spec_check && !check_replay()) {
        return false;
    }
    if (accepted == T) {
        for (int64_t il = 0; il < h_.n_layer; ++il) {
            if (rec_index_[(size_t) il] >= 0) {
                std::swap(state_cur_[(size_t) il], state_alt_[(size_t) il]);
            }
        }
        return true;
    }
    const int64_t n_kh = h_.ssm_n_kh;
    const int64_t n_vh = h_.ssm_n_vh;
    const int64_t channels = 2 * n_kh * h_.ssm_s + h_.ssm_inner;
    const int64_t n_conv_f = (h_.ssm_conv_k - 1) * channels;
    for (int64_t il = 0; il < h_.n_layer; ++il) {
        const int64_t r = rec_index_[(size_t) il];
        if (r < 0) {
            continue;
        }
        const float * rec = static_cast<const float *>(replay_pool_) +
                            r * spec_max_ * omph::kernels::gdn_replay_floats(n_vh, n_kh);
        const float * hist = static_cast<const float *>(conv_hist_pool_) +
                             r * (h_.ssm_conv_k - 1 + spec_max_) * channels;
        // the verification flipped the conv buffers: the current one is its tail
        auto * conv_a = static_cast<float *>(states_[(size_t) il]);
        float * conv_now = conv_flip_[(size_t) il] ? conv_a + n_conv_f : conv_a;
        if (!omph::kernels::gdn_replay(state_cur_[(size_t) il], rec, accepted, n_vh, n_kh,
                                       nullptr) ||
            !omph::kernels::gdn_conv_select(hist, conv_now, accepted, h_.ssm_conv_k, channels,
                                            nullptr)) {
            return fail("commit: delta-net rollback failed");
        }
    }
    if (ring_backup_k_ != nullptr) {
        const int64_t n_kv = (int64_t) std::count_if(kv_index_.begin(), kv_index_.end(),
                                                     [](int64_t k) { return k >= 0; });
        if (!omph::kernels::kv_ring_copy(ring_backup_k_, ring_backup_v_, kv_k16_, kv_v16_, n_kv,
                                         kv_window_, h_.n_head_kv * h_.head_dim * 2,
                                         verify_pos0_ + accepted, T - accepted, nullptr)) {
            return fail("commit: ring restore failed");
        }
        if (env_.spec_check) {
            // Every slot but those of the kept positions is back to its state
            // before the verification, byte for byte.
            const int64_t row = h_.n_head_kv * h_.head_dim * 2;
            std::vector<uint8_t> k(check_ring_k_.size()), v(check_ring_v_.size());
            if (hipMemcpy(k.data(), kv_k16_, k.size(), hipMemcpyDeviceToHost) != hipSuccess ||
                hipMemcpy(v.data(), kv_v16_, v.size(), hipMemcpyDeviceToHost) != hipSuccess) {
                return fail("commit: ring check copy failed");
            }
            for (int64_t l = 0; l < n_kv; ++l) {
                for (int64_t slot = 0; slot < kv_window_; ++slot) {
                    bool kept = false;
                    for (int64_t i = 0; i < accepted; ++i) {
                        kept = kept || (verify_pos0_ + i) % kv_window_ == slot;
                    }
                    const size_t off = (size_t) ((l * kv_window_ + slot) * row);
                    if (!kept && (std::memcmp(k.data() + off, check_ring_k_.data() + off, row) != 0 ||
                                  std::memcmp(v.data() + off, check_ring_v_.data() + off, row) != 0)) {
                        std::fprintf(stderr, "spec check: ring slot %lld of layer %lld not restored\n",
                                     (long long) slot, (long long) l);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

// OMPH_SPEC_CHECK: the replay of all T recorded tokens onto the state the
// verification started from must give, bit for bit, the state gdn_step reached
// (the alternate buffer). Checked on every recurrent layer before commit()
// changes anything.
bool Runner::check_replay() {
    const int64_t n_kh = h_.ssm_n_kh;
    const int64_t n_vh = h_.ssm_n_vh;
    const size_t n_state = (size_t) n_vh * h_.ssm_s * h_.ssm_s;
    float * scratch = static_cast<float *>(lazy(&spec_check_scratch_, n_state * 4));
    std::vector<float> a(n_state), b(n_state);
    int64_t mismatched = 0;
    for (int64_t il = 0; il < h_.n_layer; ++il) {
        const int64_t r = rec_index_[(size_t) il];
        if (r < 0) {
            continue;
        }
        const float * rec = static_cast<const float *>(replay_pool_) +
                            r * spec_max_ * omph::kernels::gdn_replay_floats(n_vh, n_kh);
        if (hipMemcpy(scratch, state_cur_[(size_t) il], n_state * 4, hipMemcpyDeviceToDevice) !=
                hipSuccess ||
            !omph::kernels::gdn_replay(scratch, rec, check_tokens_, n_vh, n_kh, nullptr) ||
            hipMemcpy(a.data(), scratch, n_state * 4, hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(b.data(), state_alt_[(size_t) il], n_state * 4, hipMemcpyDeviceToHost) !=
                hipSuccess) {
            return fail("spec check: replay failed");
        }
        if (std::memcmp(a.data(), b.data(), n_state * 4) != 0) {
            ++mismatched;
        }
    }
    if (mismatched > 0) {
        std::fprintf(stderr, "spec check: replay differs from gdn_step in %lld layers\n",
                     (long long) mismatched);
        return false;
    }
    return true;
}

} // namespace omph::model
