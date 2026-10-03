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
#include <thread>
#include <vector>

namespace omph::model {

Runner::Runner(const std::string & path, const int64_t max_tokens, const bool use_gemv,
                const omph::runtime::EnvOptions & env, const bool last_logits_only,
                const int64_t kv_capacity, const bool mtp, const std::string & dflash)
    : env_(env), file_(path), h_(read_hparams(file_)), use_gemv_(use_gemv), mtp_(mtp) {
    if (!file_.omph()) {
        throw std::runtime_error(path + " is not an .omph file: convert the GGUF with omph-convert (#178)");
    }
    if (!dflash.empty()) {  // the DFlash2 drafter (#245): its tensors join the weight image
        dft_file_ = std::make_unique<omph::gguf::File>(dflash);
        if (!dft_file_->omph()) {
            throw std::runtime_error(dflash + " is not an .omph file: convert the drafter GGUF with omph-convert");
        }
    }
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

    // The weights come from the .omph file (#178) already in the layouts the
    // kernels read (omph-convert repacked and verified them): one copy of each
    // tensor goes up as it is stored. The token embedding is a row gather, not
    // a matmul: it stays in pinned host memory and the dequant kernel reads
    // the one row a token needs (10 KB at IQ2_S) over PCIe, which keeps 388 MiB
    // out of VRAM (#92).
    {
        struct Place {
            const omph::gguf::File * f;
            const omph::gguf::TensorInfo * t;
            std::string name;
            size_t off;
        };
        std::vector<Place> places;
        size_t total = 0;
        std::vector<std::pair<const omph::gguf::File *, const omph::gguf::TensorInfo *>> all;
        for (const omph::gguf::TensorInfo & t : file_.tensors()) {
            all.emplace_back(&file_, &t);
        }
        if (dft_file_ != nullptr) {
            for (const omph::gguf::TensorInfo & t : dft_file_->tensors()) {
                all.emplace_back(dft_file_.get(), &t);
            }
        }
        for (const auto & [f, tp] : all) {
            const omph::gguf::TensorInfo & t = *tp;
            const std::string name = f == &file_ ? t.name : "dflash." + t.name;
            if (f == &file_ && !in_stack(t.name)) {
                continue;
            }
            if (f == &file_ && t.name == "token_embd.weight") {
                if (t.layout != omph::gguf::kLayoutGguf) {
                    throw std::runtime_error("token_embd.weight is not stored in the GGUF layout");
                }
                embd_host_ = mem_.host((size_t) t.nbytes, "cannot allocate the host embedding");
                std::memcpy(embd_host_, file_.tensor_data(t), (size_t) t.nbytes);
                continue;
            }
            if (t.layout != omph::gguf::kLayoutGguf) {
                // the engine's layout of the type (format/repack.hh): IQ3_S tiles, else the repack
                const int64_t k = (int64_t) t.ne[0];
                const int64_t rows = t.ne.size() >= 2 ? (int64_t) t.ne[1] : 1;
                if (t.layout != omph::format::engine_layout(t.type) ||
                    t.stored != (uint64_t) omph::format::engine_layout_bytes(t.type, rows, k)) {
                    throw std::runtime_error(name + ": not in this engine's layout (reconvert with omph-convert)");
                }
                GemvEntry e;
                e.off = total;
                e.bytes = (size_t) t.stored;
                e.rows = (int64_t) t.ne[1];
                e.k = (int64_t) t.ne[0];
                e.type = t.type;
                gems_[name] = e;
            }
            places.push_back({f, &t, name, total});
            total += ((size_t) t.stored + 255) & ~(size_t) 255;
        }
        dev_weights_ = mem_.device(total, "cannot allocate the weight image");
        auto * base = static_cast<uint8_t *>(dev_weights_);
        for (const Place & p : places) {
            if (hipMemcpy(base + p.off, p.f->tensor_data(*p.t), (size_t) p.t->stored, hipMemcpyHostToDevice) !=
                hipSuccess) {
                throw std::runtime_error("cannot upload " + p.name);
            }
            off_[p.name] = p.off;
        }
    }

    resolve_layers();
    if (dft_file_ != nullptr) {
        dflash_load();
    }

    // The f16 staging scratch holds the slice of rows of one weight that a
    // GEMM is about to read (it is reset for every slice: the dequant and the
    // GEMM run in stream order), at most ~OMPH_STAGE_MIB (M8; it held a whole
    // layer, ~770 MiB, #93). A --gemv runner only needs it for long
    // multi-token runs (prefill chunks), so it allocates it on first use.
    size_t scratch_bytes = 1 << 20;
    for (const omph::gguf::TensorInfo & t : file_.tensors()) {
        if (t.ne.size() < 2 || t.type == 0 || t.name.rfind("blk.", 0) != 0 || !in_stack(t.name)) {
            continue;  // f32 vectors, embedding / head
        }
        const int64_t rows = stage_rows((int64_t) t.ne[1], (int64_t) t.ne[0]);
        scratch_bytes = std::max(scratch_bytes, (((size_t) rows * t.ne[0] * 2) + 255) & ~(size_t) 255);
    }
    // Allocated on first use: with the fused GEMM (#141) only a weight of a
    // type without a decoder ever needs it.
    scratch_bytes_ = scratch_bytes;
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
    // tmp_logits_ and head16_ (the vocab-chunked f16 lm_head) are allocated
    // on first use: the GEMV decode never touches them (#86).
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
        // OMPH_KV_K4=1: K in V's Q4 format too (#81, experiment); or only on the
        // attention layers of OMPH_KV_K4_LAYERS (#175).
        kv_k4_.assign((size_t) n_kv, 0);
        kv_kq_off_.assign((size_t) n_kv, 0);
        size_t kq_bytes = 0;
        for (int64_t i = 0; i < n_kv; ++i) {
            kv_k4_[(size_t) i] = env_.kv_k4 || (i < 64 && ((env_.kv_k4_layers >> i) & 1));
            kv_kq_off_[(size_t) i] = kq_bytes;
            kq_bytes += (size_t) kvcap * attn_kv / (kv_k4_[(size_t) i] ? 2 : 1);
        }
        alloc(&kv_kq_, kq_bytes);
        alloc(&kv_ks_, (size_t) n_kv * kvcap * h_.n_head_kv * nblk * 2);
        alloc(&kv_vq_, (size_t) n_kv * kvcap * attn_kv / 2);
        alloc(&kv_vs_, (size_t) n_kv * kvcap * h_.n_head_kv * nblk * 2);
        // FP16 window: the last `kv_window_` tokens exactly, in the same
        // rotated basis, in a ring (PLAN §13.4). 128 by default: it keeps the
        // KL under llama.cpp's q8_0/q4_0 up to 32k for 8.4 MB (#61);
        // OMPH_KV_WINDOW=0 turns it off.
        kv_window_ = env_.kv_window;
        // kKvRingExtra slots more: every query of a short chunk (a
        // verification) reads its own last kv_window_ keys exactly (#161)
        kv_ring_ = kv_window_ > 0 ? kv_window_ + omph::kernels::kKvRingExtra : 0;
        alloc(&kv_k16_, (size_t) n_kv * kv_ring_ * attn_kv * 2);
        alloc(&kv_v16_, (size_t) n_kv * kv_ring_ * attn_kv * 2);
    }
    // Off by default (#189): with the persistent-warp GEMVs (#63) one GEMV
    // fills the GPU, and the measured step is 0.7 ms faster in order (44.2 vs
    // 44.9 ms), while a side stream the calibration keeps can still cost
    // ~9 ms per step (53.6) below the watchdog's threshold. OMPH_OVERLAP=1
    // brings it back, with the calibration and the watchdog.
    overlap_ = use_gemv_ && env_.overlap;
    if (overlap_ && (hipStreamCreateWithFlags(&side_, hipStreamNonBlocking) != hipSuccess ||
                     hipEventCreateWithFlags(&ev_fork_, hipEventDisableTiming) != hipSuccess ||
                     hipEventCreateWithFlags(&ev_join_, hipEventDisableTiming) != hipSuccess)) {
        throw std::runtime_error("cannot create the side stream");
    }
    key_chunk_ = omph::kernels::attention_key_chunk(max_seq_);
    attn_work_bytes_ = omph::kernels::attention_gqa_work_bytes(T, h_.n_head, h_.n_head_kv,
                                                               h_.head_dim, max_seq_, key_chunk_);
    alloc(&attn_work_, attn_work_bytes_);
    // MTP (#124): the block's own cache (Q8/Q4, no ring) and its h buffers.
    if (mtp_) {
        const int64_t nblk = h_.head_dim / 32;
        alloc(&mtp_kq_, (size_t) kvcap * attn_kv);
        alloc(&mtp_ks_, (size_t) kvcap * h_.n_head_kv * nblk * 2);
        alloc(&mtp_vq_, (size_t) kvcap * attn_kv / 2);
        alloc(&mtp_vs_, (size_t) kvcap * h_.n_head_kv * nblk * 2);
        alloc(&mtp_hlast_, (size_t) T * ne * 4);
        alloc(&mtp_hin_, (size_t) T * ne * 4);
        alloc(&mtp_pending_, (size_t) ne * 4);
        alloc(&mtp_g_, (size_t) ne * 4);
        if (hipMemset(mtp_pending_, 0, (size_t) ne * 4) != hipSuccess) {
            throw std::runtime_error("cannot clear the MTP h");
        }
    }
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
    gemm_min_ = env_.gemm_min;
    if (overlap_) {
        calibrate_overlap();
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
    // with MTP on, its block (blk.<n_layer>) is loaded too (#124)
    return std::atoll(name.c_str() + 4) < h_.n_layer + (mtp_ ? 1 : 0);
}

// The side stream overlaps sibling GEMVs (#71), ~0.9 ms of a 46 ms step. But
// whether two of the process's hardware queues really run side by side is
// decided when the queues are created, by the driver's queue scheduler, and
// varies from run to run: on a bad pair every small kernel after a join waited
// ~15 us and the step took 72 ms instead of 46.7 without overlap (#132; it
// had been hidden by hipBLASLt taking a queue first). So the pattern of a
// decode step's FFN is timed both ways at load; a losing side stream is
// replaced by a new one (the next hardware queue), and after three tries the
// overlap is turned off.
void Runner::calibrate_overlap() {
    const LayerWeights & L = layers_.front();
    const int64_t ne = h_.n_embd;
    const auto pattern = [&]() {
        for (int i = 0; i < 16; ++i) {
            if (!omph::kernels::add_rms_norm_f16(static_cast<const float *>(blk_),
                                                 static_cast<const float *>(x_),
                                                 static_cast<float *>(resid_), L.post_norm,
                                                 h16_, 1, ne, (float) h_.eps, nullptr) ||
                !fork_join(
                    1,
                    [&] {
                        return matmul(L.ffn_up, h16_, static_cast<float *>(ffn2_), h_.n_ff, ne, 1);
                    },
                    [&] {
                        return matmul(L.ffn_gate, h16_, static_cast<float *>(ffn1_), h_.n_ff, ne,
                                      1);
                    }) ||
                !omph::kernels::swiglu_f16(static_cast<const float *>(ffn1_),
                                           static_cast<const float *>(ffn2_), ffn16_, h_.n_ff,
                                           nullptr)) {
                return false;
            }
        }
        return true;
    };
    hipEvent_t e0 = nullptr;
    hipEvent_t e1 = nullptr;
    if (hipEventCreate(&e0) != hipSuccess || hipEventCreate(&e1) != hipSuccess) {
        throw std::runtime_error("cannot create the calibration events");
    }
    // the median of five timings of the pattern, overlapped or not
    const auto time = [&](const bool overlap) {
        overlap_ = overlap;
        std::vector<float> ms;
        for (int rep = 0; rep < 6; ++rep) {
            float t = 0.0f;
            if (hipEventRecord(e0, nullptr) != hipSuccess || !pattern() ||
                hipEventRecord(e1, nullptr) != hipSuccess || hipEventSynchronize(e1) != hipSuccess ||
                hipEventElapsedTime(&t, e0, e1) != hipSuccess) {
                throw std::runtime_error("overlap calibration failed");
            }
            if (rep > 0) {  // the first one warms up
                ms.push_back(t);
            }
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    };
    const float serial = time(false);
    bool keep = false;
    for (int attempt = 0; attempt < 3 && !keep; ++attempt) {
        if (attempt > 0) {
            hipStream_t fresh = nullptr;
            if (hipStreamCreateWithFlags(&fresh, hipStreamNonBlocking) != hipSuccess) {
                break;
            }
            if (env_.test_bad_side > 0 && test_bad_side_ == nullptr) {
                test_bad_side_ = side_;  // kept for the watchdog's test hook
            } else {
                (void) hipStreamDestroy(side_);
            }
            side_ = fresh;
        }
        const float overlapped = time(true);
        keep = overlapped < serial;
        if (env_.timing) {
            std::fprintf(stderr, "overlap calibration: serial %.3f ms, side stream %d %.3f ms%s\n",
                         serial, attempt, overlapped, keep ? " (kept)" : "");
        }
    }
    overlap_ = keep;
    (void) hipEventDestroy(e0);
    (void) hipEventDestroy(e1);
}

// The side stream can turn bad during a run, after a calibration that kept a
// good one (#144: twice, ~72 ms per step instead of 45.8). Every
// single-token step is timed; three in a row at 1.35x a slow average of the
// normal ones re-run the calibration at the start of the next step, when the
// activation buffers it uses are free. A step's time grows only slowly with
// the context, so this does not fire on long runs.
void Runner::watch_step_begin(const int64_t T) {
    if (T != 1 || side_ == nullptr || !env_.overlap) {
        return;
    }
    if (recal_pending_ && recalibrations_ < kMaxRecalibrations) {
        recal_pending_ = false;
        ++recalibrations_;
        if (env_.timing) {
            std::fprintf(stderr, "side stream: steps %.1fx slower than %.2f ms, recalibrating\n",
                         kSlowStep, step_ref_);
        }
        calibrate_overlap();
    }
    if (env_.test_bad_side > 0 && ++watched_steps_ == env_.test_bad_side &&
        test_bad_side_ != nullptr) {
        side_ = test_bad_side_;  // test hook: the stream the calibration rejected
        test_bad_side_ = nullptr;
        overlap_ = true;
        if (env_.timing) {
            std::fprintf(stderr, "side stream: test hook, rejected stream back in\n");
        }
    }
    if (mon_a_ == nullptr && (hipEventCreate(&mon_a_) != hipSuccess ||
                              hipEventCreate(&mon_b_) != hipSuccess)) {
        return;
    }
    (void) hipEventRecord(mon_a_, nullptr);
    mon_open_ = true;
}

void Runner::watch_step_end() {
    if (!mon_open_) {
        return;
    }
    mon_open_ = false;
    float ms = 0.0f;
    if (hipEventRecord(mon_b_, nullptr) != hipSuccess || hipEventSynchronize(mon_b_) != hipSuccess ||
        hipEventElapsedTime(&ms, mon_a_, mon_b_) != hipSuccess) {
        return;
    }
    if (step_ref_ <= 0.0) {
        step_ref_ = ms;
    } else if (ms > kSlowStep * step_ref_) {
        if (++slow_steps_ >= 3) {
            slow_steps_ = 0;
            recal_pending_ = true;
        }
    } else {
        slow_steps_ = 0;
        step_ref_ = 0.9 * step_ref_ + 0.1 * ms;
    }
}

bool Runner::reset_sequence() {
    if (dfl_tags_ != nullptr && hipMemsetAsync(dfl_tags_, 0xFF, (size_t) dfl_swa_ * 4, nullptr) != hipSuccess) {
        return fail("reset: drafter ring failed");
    }
    const int64_t ssm_q = h_.ssm_n_kh * h_.ssm_s;
    const int64_t ssm_channels = 2 * ssm_q + h_.ssm_inner;
    const size_t n_state = (size_t) h_.ssm_n_vh * h_.ssm_s * h_.ssm_s;
    const size_t n_conv = (size_t) (h_.ssm_conv_k - 1) * ssm_channels;
    for (int64_t il = 0; il < h_.n_layer; ++il) {
        if (states_[(size_t) il] == nullptr) {
            continue;
        }
        // the two conv tails sit at the start of the layer's block; the state
        // in use may be the alternate buffer after a speculative commit
        if (hipMemsetAsync(states_[(size_t) il], 0, 2 * n_conv * 4, nullptr) != hipSuccess ||
            hipMemsetAsync(state_cur_[(size_t) il], 0, n_state * 4, nullptr) != hipSuccess) {
            return fail("reset: cannot clear the delta-net state");
        }
    }
    conv_flip_.assign((size_t) h_.n_layer, 0);
    if (mtp_pending_ != nullptr &&
        hipMemsetAsync(mtp_pending_, 0, (size_t) h_.n_embd * 4, nullptr) != hipSuccess) {
        return fail("reset: cannot clear the MTP h");
    }
    last_toks_.clear();
    last_pos0_ = 0;
    rope_delta_ = 0;
    return hipDeviceSynchronize() == hipSuccess || fail("reset failed");
}

size_t Runner::checkpoint_bytes() const {
    const int64_t channels = 2 * h_.ssm_n_kh * h_.ssm_s + h_.ssm_inner;
    const size_t per_layer = ((size_t) (h_.ssm_conv_k - 1) * channels + (size_t) h_.ssm_n_vh * h_.ssm_s * h_.ssm_s) * 4;
    size_t bytes = 0;
    for (int64_t il = 0; il < h_.n_layer; ++il) {
        if (states_[(size_t) il] != nullptr) bytes += per_layer;
    }
    if (kv_k16_ != nullptr) {
        const int64_t n_kv = (int64_t) std::count_if(kv_index_.begin(), kv_index_.end(),
                                                     [](int64_t k) { return k >= 0; });
        bytes += 2 * (size_t) n_kv * kv_ring_ * h_.n_head_kv * h_.head_dim * 2;
    }
    if (mtp_pending_ != nullptr) bytes += (size_t) h_.n_embd * 4;
    return bytes;
}

void * Runner::checkpoint_alloc() {
    return mem_.host(checkpoint_bytes(), "out of pinned host memory (checkpoint)");
}

// The checkpoint's parts in order: per recurrent layer the current conv tail
// and state, then the K and V rings, then the MTP h. `save` copies them to
// the host buffer, else from it (into the first conv slot).
bool Runner::checkpoint_copy(void * host, const bool save) {
    const int64_t channels = 2 * h_.ssm_n_kh * h_.ssm_s + h_.ssm_inner;
    const size_t conv_bytes = (size_t) (h_.ssm_conv_k - 1) * channels * 4;
    const size_t state_bytes = (size_t) h_.ssm_n_vh * h_.ssm_s * h_.ssm_s * 4;
    auto * at = static_cast<uint8_t *>(host);
    const auto copy = [&](void * dev, const size_t bytes) {
        const hipError_t e = save ? hipMemcpyAsync(at, dev, bytes, hipMemcpyDeviceToHost, nullptr)
                                  : hipMemcpyAsync(dev, at, bytes, hipMemcpyHostToDevice, nullptr);
        at += bytes;
        return e == hipSuccess;
    };
    if (hipDeviceSynchronize() != hipSuccess) {
        return fail("checkpoint: device error");
    }
    bool ok = true;
    for (int64_t il = 0; il < h_.n_layer && ok; ++il) {
        if (states_[(size_t) il] == nullptr) continue;
        auto * conv = static_cast<uint8_t *>(states_[(size_t) il]);
        ok = copy(conv + (save && conv_flip_[(size_t) il] ? conv_bytes : 0), conv_bytes) &&
             copy(state_cur_[(size_t) il], state_bytes);
    }
    if (ok && kv_k16_ != nullptr) {
        const int64_t n_kv = (int64_t) std::count_if(kv_index_.begin(), kv_index_.end(),
                                                     [](int64_t k) { return k >= 0; });
        const size_t ring = (size_t) n_kv * kv_ring_ * h_.n_head_kv * h_.head_dim * 2;
        ok = copy(kv_k16_, ring) && copy(kv_v16_, ring);
    }
    if (ok && mtp_pending_ != nullptr) {
        ok = copy(mtp_pending_, (size_t) h_.n_embd * 4);
    }
    if (!ok || hipDeviceSynchronize() != hipSuccess) {
        return fail("checkpoint: copy failed");
    }
    if (!save) {
        conv_flip_.assign((size_t) h_.n_layer, 0);
        last_toks_.clear();
        last_pos0_ = 0;
    }
    return true;
}

bool Runner::checkpoint_save(void * host) { return checkpoint_copy(host, true); }

bool Runner::checkpoint_restore(const void * host) { return checkpoint_copy(const_cast<void *>(host), false); }

// Allocates *p on first use (the f16-path buffers, #86).
void * Runner::lazy(void ** p, const size_t bytes) {
    if (*p == nullptr) {
        *p = mem_.device(bytes, "out of VRAM (f16-path buffer)");
    }
    return *p;
}

const void * Runner::q8_input(const void * x16, const int64_t k, const int64_t T) {
    constexpr int64_t kMaxK = 32768;
    if (env_.test_q8act <= 0) {
        return x16;
    }
    if (k > kMaxK || T > max_tokens_) {
        throw std::runtime_error("OMPH_TEST_Q8ACT: input larger than its buffer");
    }
    void * q = lazy(&q8x_, (size_t) max_tokens_ * kMaxK * 2);
    if (!omph::kernels::q8_roundtrip_f16(x16, q, T, k, env_.test_q8act, hipStreamPerThread)) {
        throw std::runtime_error("OMPH_TEST_Q8ACT: bad block size");
    }
    return q;
}

// Device pointer to the original GGUF bytes of a tensor that was not
// repacked (or whose repacked layout is the GGUF bytes, IQ1_M).
const void * Runner::raw_bytes(const std::string & name) {
    const auto it = gems_.find(name);
    if (it != gems_.end() && it->second.type != 29) {
        throw std::runtime_error("raw bytes of the repacked " + name + " are not kept");
    }
    return static_cast<const uint8_t *>(dev_weights_) + off_.at(name);
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

// The f16 scratch for one weight slice. It holds one slice at a time: the
// GEMM that reads it runs, in stream order, before the next dequant
// overwrites it.
void * Runner::stage_scratch(const size_t bytes) {
    if (!scratch_ready_) {
        scratch_.init(mem_.device(scratch_bytes_, "out of VRAM (weight scratch)"), scratch_bytes_,
                      env_.trace_alloc);
        scratch_ready_ = true;
    }
    scratch_.reset();
    return scratch_.alloc(bytes);
}

// Rows per f16 slice of an (n_out x k) weight: the whole weight if it fits
// the OMPH_STAGE_MIB budget, else equal slices (multiples of 256 rows) of
// about that size. A slice's GEMM writes its columns of y in place.
int64_t Runner::stage_rows(const int64_t n_out, const int64_t k) const {
    const int64_t budget = (int64_t) env_.stage_mib << 20;
    const int64_t bytes = n_out * k * 2;
    if (budget <= 0 || bytes <= budget) {
        return n_out;
    }
    const int64_t n = (bytes + budget - 1) / budget;
    return std::min(n_out, ((n_out + n - 1) / n + 255) & ~(int64_t) 255);
}

void * Runner::stage_w(const std::string & name, const int64_t row0, int64_t nrows) {
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
    const int64_t k = (int64_t) t->ne[0];
    const int64_t n_rows = n_elems / k;
    if (nrows < 0) {
        nrows = n_rows - row0;
    }
    bool cacheable = row0 == 0 && nrows == n_rows && !env_.no_f16_cache &&
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
    const int64_t n = nrows * k;
    void * dst = nullptr;
    if (cacheable) {
        // Long contexts fill VRAM: caching is an optimization, never a
        // reason to fail. Fall back to the per-call staging buffer
        // (try_device clears the out-of-memory error, which the next kernel
        // wrapper's hipGetLastError() would otherwise report as its own).
        dst = mem_.try_device((size_t) n * 2);
        if (dst == nullptr) {
            cacheable = false;
            dst = stage_scratch((size_t) n * 2);
        }
    } else {
        dst = stage_scratch((size_t) n * 2);
    }
    if (env_.trace_alloc) {
        std::fprintf(stderr, "stage %-40s %10lld elems  ne=[", name.c_str(), (long long) n);
        for (const uint64_t d : t->ne) {
            std::fprintf(stderr, "%llu,", (unsigned long long) d);
        }
        std::fprintf(stderr, "] type=%u\n", t->type);
    }
    // repacked weights straight from their layout, at the bandwidth bound and
    // bit-identical to dequantizing the GGUF bytes (M8)
    const auto g = gems_.find(name);
    const bool ok = g != gems_.end() && g->second.type != 29
                        ? omph::kernels::dequant_repacked(g->second.type,
                                                          static_cast<const uint8_t *>(dev_weights_) +
                                                              g->second.off,
                                                          dst, g->second.rows, g->second.k, nullptr,
                                                          row0, nrows)
                        : omph::kernels::dequantize(t->type,
                                                    static_cast<const uint8_t *>(raw_bytes(name)) +
                                                        row0 * (int64_t) (t->nbytes / n_rows),
                                                    dst, n, true, nullptr);
    if (!ok) {
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
    const bool dft = name.rfind("dflash.", 0) == 0;
    m.t = dft ? (dft_file_ != nullptr ? dft_file_->tensor(name.substr(7)) : nullptr) : file_.tensor(name);
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
    if (mtp_) {
        const std::string p = "blk." + std::to_string(h_.n_layer) + ".";
        LayerWeights & L = mtp_L_;
        L.attn_norm = resolve_f32(p + "attn_norm.weight");
        L.post_norm = resolve_f32(p + "post_attention_norm.weight");
        L.q_norm = resolve_f32(p + "attn_q_norm.weight");
        L.k_norm = resolve_f32(p + "attn_k_norm.weight");
        L.attn_q = resolve(p + "attn_q.weight");
        L.attn_k = resolve(p + "attn_k.weight");
        L.attn_v = resolve(p + "attn_v.weight");
        L.attn_output = resolve(p + "attn_output.weight");
        L.ffn_up = resolve(p + "ffn_up.weight");
        L.ffn_gate = resolve(p + "ffn_gate.weight");
        L.ffn_down = resolve(p + "ffn_down.weight");
        mtp_eh_ = resolve(p + "nextn.eh_proj.weight");
        mtp_enorm_ = resolve_f32(p + "nextn.enorm.weight");
        mtp_hnorm_ = resolve_f32(p + "nextn.hnorm.weight");
        mtp_head_norm_ = resolve_f32(p + "nextn.shared_head_norm.weight");
        if (L.attn_norm == nullptr || L.post_norm == nullptr || L.q_norm == nullptr ||
            L.k_norm == nullptr || mtp_eh_.t == nullptr || mtp_enorm_ == nullptr ||
            mtp_hnorm_ == nullptr || mtp_head_norm_ == nullptr || L.attn_q.t == nullptr ||
            L.ffn_down.t == nullptr) {
            throw std::runtime_error("the model has no complete MTP block (" + p + "nextn.*)");
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
        const size_t ring = (size_t) n_kv * kv_ring_ * h_.n_head_kv * h_.head_dim * 2;
        ring_backup_k_ = mem_.device(ring);
        ring_backup_v_ = mem_.device(ring);
    }
    spec_keys_ = mem_.device((size_t) std::max<int64_t>(max_tokens, 1) * 8);
    // A verification brings every row back, also with --last-logits.
    if (last_logits_only_ || max_tokens > kHeadRows) {
        logits_ = mem_.device((size_t) std::max<int64_t>(max_tokens, kHeadRows) * h_.n_vocab * 4);
    }
}

bool Runner::verify(const std::vector<int32_t> & toks, const int64_t pos0,
                    std::vector<int32_t> & argmax, std::vector<float> * rows) {
    const int64_t T = (int64_t) toks.size();
    if (spec_max_ == 0 || T < 1 || T > spec_max_) {
        return fail("verify: speculation not enabled, or too many tokens");
    }
    const int64_t n_kv = (int64_t) std::count_if(kv_index_.begin(), kv_index_.end(),
                                                 [](int64_t k) { return k >= 0; });
    const int64_t row_bytes = h_.n_head_kv * h_.head_dim * 2;
    if (ring_backup_k_ != nullptr &&
        !omph::kernels::kv_ring_copy(kv_k16_, kv_v16_, ring_backup_k_, ring_backup_v_, n_kv,
                                     kv_ring_, row_bytes, pos0, T, nullptr)) {
        return fail("verify: ring backup failed");
    }
    if (env_.spec_check && ring_backup_k_ != nullptr) {
        const size_t ring = (size_t) n_kv * kv_ring_ * row_bytes;
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
    argmax.clear();
    verify_argmax_ = rows == nullptr ? &argmax : nullptr;
    std::vector<float> logits;
    const bool ok = forward(toks, logits, std::string(), pos0, true, nullptr);
    verifying_ = false;
    verify_argmax_ = nullptr;
    if (!ok) {
        return false;
    }
    if (rows != nullptr) {
        *rows = std::move(logits);
        return (int64_t) rows->size() == T * h_.n_vocab || fail("verify: logits missing");
    }
    if ((int64_t) argmax.size() == T) {
        return true;  // the device argmax ran (the GEMV Q4_K head)
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

void Runner::observe_draft(const int32_t * toks, const int64_t n) {
    if (env_.draft_vocab <= 0) {
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        draft_oov_ += ((toks[i] >= env_.draft_vocab ? 1.0 : 0.0) - draft_oov_) / 64.0;
    }
}

bool Runner::commit(const int64_t accepted) {
    const int64_t T = verify_tokens_;
    if (T == 0 || accepted < 1 || accepted > T) {
        return fail("commit: no verification, or accepted out of range");
    }
    if ((int64_t) last_toks_.size() >= accepted) {
        observe_draft(last_toks_.data(), accepted);  // the verified tokens kept
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
        dflash_observe(accepted - 1);
        return (!mtp_ || mtp_fill(accepted)) && dflash_inject(verify_pos0_, accepted);
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
                                         kv_ring_, h_.n_head_kv * h_.head_dim * 2,
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
                for (int64_t slot = 0; slot < kv_ring_; ++slot) {
                    bool kept = false;
                    for (int64_t i = 0; i < accepted; ++i) {
                        kept = kept || (verify_pos0_ + i) % kv_ring_ == slot;
                    }
                    const size_t off = (size_t) ((l * kv_ring_ + slot) * row);
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
    dflash_observe(accepted - 1);
    return (!mtp_ || mtp_fill(accepted)) && dflash_inject(verify_pos0_, accepted);
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
