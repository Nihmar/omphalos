// omph-run — naive full forward pass of the 64-layer stack (PLAN.md §12, M2).
//
// usage: omph-run <model.gguf> <tokens.txt> <out-logits.f32> [--trace-dir DIR]
//   tokens.txt: token ids separated by whitespace.
//   Logits are written as tokens x n_vocab f32 for the whole prompt (one row
//   with --last-logits, the last N with --logits-tail N); with --trace-dir the
//   per-layer outputs are dumped as l_out-<layer>.f32 (single-chunk prompts only).
//   The KV cache is K Q8 / V Q4 with an FP16 ring of the last 128 tokens
//   (OMPH_KV_WINDOW=N changes it, 0 disables it). OMPH_KV_F32=1 keeps it in f32
//   instead (the exact reference); OMPH_KV_HOST=1 is that f32 cache in host RAM,
//   for long contexts where it does not fit in VRAM.
//
// The whole quantized tensor block lives in VRAM; the weights of the current
// layer are dequantized to f16 into a reusable scratch buffer, and every op is
// a separate kernel launch. Naive by design — this is the M2 correctness path.
#include "format/gguf.hh"
#include "format/repack.hh"
#include "kernels/attn.hh"
#include "kernels/dequant.hh"
#include "kernels/elementwise.hh"
#include "kernels/gdn.hh"
#include "kernels/gemv.hh"
#include "runtime/matmul.hh"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

class Scratch {
public:
    bool init(const size_t bytes) {
        cap_ = bytes;
        return hipMalloc(&base_, bytes) == hipSuccess;
    }
    void reset() { used_ = 0; }
    void * alloc(const size_t bytes) {
        const size_t off = (used_ + 255) & ~(size_t) 255;
        if (std::getenv("OMPH_TRACE_ALLOC") != nullptr) {
            std::fprintf(stderr, "  alloc %8zu MiB (used %8zu MiB)\n", bytes >> 20,
                         (off + bytes) >> 20);
        }
        if (off + bytes > cap_) {
            std::fprintf(stderr, "scratch: need %zu MiB, have %zu MiB (used %zu MiB)\n",
                         (off + bytes) >> 20, cap_ >> 20, used_ >> 20);
            throw std::runtime_error("weight scratch exhausted");
        }
        used_ = off + bytes;
        return static_cast<uint8_t *>(base_) + off;
    }

private:
    void * base_ = nullptr;
    size_t cap_ = 0;
    size_t used_ = 0;
};

struct HParams {
    int64_t n_embd = 0;
    int64_t n_layer = 0;
    int64_t n_vocab = 0;
    int64_t n_head = 0;
    int64_t n_head_kv = 0;
    int64_t head_dim = 0;
    int64_t n_rot = 0;
    int64_t n_ff = 0;
    int64_t ssm_n_kh = 0;
    int64_t ssm_n_vh = 0;
    int64_t ssm_s = 0;
    int64_t ssm_inner = 0;
    int64_t ssm_conv_k = 0;
    double eps = 1e-6;
    double freq_base = 10000.0;
};

bool meta_int(const omph::gguf::File & f, const char * key, int64_t & out) {
    const omph::gguf::Value * v = f.find(key);
    uint64_t u = 0;
    if (v == nullptr || !v->as_u64(u)) {
        return false;
    }
    out = (int64_t) u;
    return true;
}

double meta_float(const omph::gguf::File & f, const char * key, const double fallback) {
    const omph::gguf::Value * v = f.find(key);
    if (v == nullptr) {
        return fallback;
    }
    if (v->type == omph::gguf::ValueType::FLOAT32 || v->type == omph::gguf::ValueType::FLOAT64) {
        return v->f;
    }
    uint64_t u = 0;
    return v->as_u64(u) ? (double) u : fallback;
}

int64_t numel(const omph::gguf::TensorInfo & t) {
    int64_t n = 1;
    for (const uint64_t d : t.ne) {
        n *= (int64_t) d;
    }
    return n;
}

void write_f32(const std::string & path, const std::vector<float> & data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * 4));
}

// Phase timing: record a fresh event pair per call, resolve them all after the
// run (reusing one pair made every total `calls x last interval`, #64).
class PhaseTimer {
public:
    PhaseTimer() : on_(std::getenv("OMPH_TIMING") != nullptr) {}
    void start() {
        if (on_) {
            (void) hipEventCreate(&a_);
            (void) hipEventRecord(a_, nullptr);
        }
    }
    void stop(std::vector<std::pair<hipEvent_t, hipEvent_t>> & sink) {
        if (on_) {
            hipEvent_t b{};
            (void) hipEventCreate(&b);
            (void) hipEventRecord(b, nullptr);
            sink.emplace_back(a_, b);
        }
    }
    // Sums and destroys the recorded pairs.
    static double total_ms(std::vector<std::pair<hipEvent_t, hipEvent_t>> & v) {
        double ms = 0.0;
        for (const auto & p : v) {
            float d = 0.0f;
            if (hipEventElapsedTime(&d, p.first, p.second) == hipSuccess) {
                ms += d;
            }
            (void) hipEventDestroy(p.first);
            (void) hipEventDestroy(p.second);
        }
        v.clear();
        return ms;
    }

private:
    bool on_ = false;
    hipEvent_t a_{};
};

class Runner {
public:
    explicit Runner(const std::string & path, const int64_t max_tokens, const bool use_gemv,
                    const bool last_logits_only = false, const int64_t kv_capacity = 0)
        : file_(path), h_(read_hparams(file_)), use_gemv_(use_gemv) {
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
                Place p;
                p.name = t.name;
                p.off = total;
                const int64_t bb = omph::format::quant_block_bytes(t.type);
                // The token embedding is a row gather, not a matmul: keep its raw
                // bytes in the image so the per-token lookup never stages 388 MiB
                // over PCIe.
                const bool embed = t.name == "token_embd.weight";
                // Sized from the layout: the repack itself runs once, at upload.
                const int64_t packed_bytes =
                    use_gemv_ && !embed && bb > 0 && t.nbytes % (uint64_t) bb == 0
                        ? omph::format::repacked_bytes(t.type, (int64_t) (t.nbytes / (uint64_t) bb))
                        : 0;
                if (packed_bytes > 0) {
                    GemvEntry e;
                    e.off = total;
                    e.bytes = (size_t) packed_bytes;
                    e.rows = (int64_t) t.ne[1];
                    e.k = (int64_t) t.ne[0];
                    e.type = t.type;
                    e.has_b4 = std::getenv("OMPH_NO_B4") == nullptr &&
                               (t.type == 12 || t.type == 18 || t.type == 21 || t.type == 23);
                    gems_[t.name] = e;
                    total += ((size_t) packed_bytes + 255) & ~(size_t) 255;
                    places.push_back(p);
                    continue;
                }
                total += ((size_t) t.nbytes + 255) & ~(size_t) 255;
                places.push_back(p);
            }
            if (hipMalloc(&dev_weights_, total) != hipSuccess) {
                throw std::runtime_error("cannot allocate the weight image");
            }
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

        if (!scratch_.init((size_t) 1280 * 1024 * 1024)) {
            throw std::runtime_error("cannot allocate the weight scratch");
        }
        const auto alloc = [&](void ** p, const size_t bytes) {
            if (hipMalloc(p, bytes) != hipSuccess) {
                throw std::runtime_error("out of VRAM");
            }
        };
        alloc(&x_, T * ne * 4);
        alloc(&cur_, T * ne * 4);
        alloc(&resid_, T * ne * 4);
        alloc(&blk_, T * ne * 4);
        alloc(&h16_, T * ne * 2);
        alloc(&fused_, T * fused * 4);
        alloc(&conv_out_, T * fused * 4);
        alloc(&z_, T * ssm_v * 4);
        alloc(&q_, T * attn_q * 4);
        alloc(&gate_, T * attn_q * 4);
        alloc(&k_, T * std::max(ssm_q, attn_kv) * 4);
        alloc(&v_, T * std::max(ssm_v, attn_kv) * 4);
        alloc(&sk_, h_.ssm_n_vh * h_.ssm_s * 4);
        alloc(&dvec_, h_.ssm_n_vh * h_.ssm_s * 4);
        alloc(&beta_, T * h_.ssm_n_vh * 4);
        alloc(&alpha_, T * h_.ssm_n_vh * 4);
        alloc(&ffn1_, T * h_.n_ff * 4);
        alloc(&ffn2_, T * h_.n_ff * 4);
        alloc(&ffn16_, T * h_.n_ff * 2);
        last_logits_only_ = last_logits_only;
        alloc(&logits_, (last_logits_only ? 1 : T) * h_.n_vocab * 4);
        alloc(&tmp_logits_, T * std::min<int64_t>(h_.n_vocab, 32768) * 4);
        alloc(&head16_, (size_t) std::min<int64_t>(h_.n_vocab, 32768) * ne * 2);
        alloc(&raw_stage_, (size_t) 768 * 1024 * 1024);  // raw bytes for the f16 fallback
        const int64_t kvcap = kv_capacity > 0 ? kv_capacity : T;
        max_seq_ = kvcap;
        int64_t n_kv = 0;
        for (int64_t il = 0; il < h_.n_layer; ++il) {
            const bool recurrent = file_.tensor("blk." + std::to_string(il) + ".ssm_a") != nullptr;
            kv_index_.push_back(recurrent ? -1 : n_kv++);
        }
        // The KV cache is sized by the whole sequence, the activations by the
        // chunk: that is what lets a long prompt run in pieces.
        // Q8/Q4 by default (#69): validated to 32k under llama.cpp's q8_0/q4_0
        // budget at 4.9x less VRAM than f32 (#58, #61).
        kv_host_ = std::getenv("OMPH_KV_HOST") != nullptr;
        kv_q8q4_ = !kv_host_ && std::getenv("OMPH_KV_F32") == nullptr;
        if (kv_host_) {
            // Validation reference only: the exact f32 cache in pinned host RAM
            // (4.29 GB at 32k does not fit beside the weights), and one layer's
            // worth of it staged into VRAM before each attention.
            const size_t bytes = (size_t) n_kv * kvcap * attn_kv * 4;
            if (hipHostMalloc(&kv_k_, bytes) != hipSuccess ||
                hipHostMalloc(&kv_v_, bytes) != hipSuccess) {
                throw std::runtime_error("cannot allocate the host KV cache");
            }
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
            kv_k4_ = std::getenv("OMPH_KV_K4") != nullptr;
            alloc(&kv_kq_, (size_t) n_kv * kvcap * attn_kv / (kv_k4_ ? 2 : 1));
            alloc(&kv_ks_, (size_t) n_kv * kvcap * h_.n_head_kv * nblk * 2);
            alloc(&kv_vq_, (size_t) n_kv * kvcap * attn_kv / 2);
            alloc(&kv_vs_, (size_t) n_kv * kvcap * h_.n_head_kv * nblk * 2);
            // FP16 window: the last `kv_window_` tokens exactly, in the same
            // rotated basis, in a ring (PLAN §13.4). 128 by default: it keeps the
            // KL under llama.cpp's q8_0/q4_0 up to 32k for 8.4 MB (#61);
            // OMPH_KV_WINDOW=0 turns it off.
            kv_window_ = 128;
            if (const char * w = std::getenv("OMPH_KV_WINDOW")) {
                kv_window_ = std::atoll(w);
            }
            alloc(&kv_k16_, (size_t) n_kv * kv_window_ * attn_kv * 2);
            alloc(&kv_v16_, (size_t) n_kv * kv_window_ * attn_kv * 2);
        }
        overlap_ = use_gemv_ && std::getenv("OMPH_NO_OVERLAP") == nullptr;
        if (overlap_ && (hipStreamCreateWithFlags(&side_, hipStreamNonBlocking) != hipSuccess ||
                         hipEventCreateWithFlags(&ev_fork_, hipEventDisableTiming) != hipSuccess ||
                         hipEventCreateWithFlags(&ev_join_, hipEventDisableTiming) != hipSuccess)) {
            throw std::runtime_error("cannot create the side stream");
        }
        attn_work_bytes_ = omph::kernels::attention_gqa_work_bytes(T, h_.n_head, h_.n_head_kv,
                                                                   h_.head_dim);
        alloc(&attn_work_, attn_work_bytes_);
        conv_flip_.assign((size_t) h_.n_layer, 0);
        for (int64_t il = 0; il < h_.n_layer; ++il) {
            void * st = nullptr;
            const int64_t n_state = h_.ssm_n_vh * h_.ssm_s * h_.ssm_s;
            const int64_t n_conv = (h_.ssm_conv_k - 1) * ssm_channels;
            if (hipMalloc(&st, (n_state + 2 * n_conv) * 4) != hipSuccess ||
                hipMemset(st, 0, (n_state + 2 * n_conv) * 4) != hipSuccess) {
                throw std::runtime_error("out of VRAM (state)");
            }
            states_.push_back(st);
        }
    }

    ~Runner() {
        if (kv_host_) {
            (void) hipHostFree(kv_k_);
            (void) hipHostFree(kv_v_);
        }
        for (void * p : f16_cache_owned_) {
            (void) hipFree(p);
        }
        if (dev_weights_ != nullptr) {
            (void) hipFree(dev_weights_);
        }
    }

    // want_logits = false runs the layers only (a prefill chunk whose logits
    // nobody reads): no final norm, no lm_head.
    bool forward(const std::vector<int32_t> & toks, std::vector<float> & logits,
                 const std::string & trace_dir, const int64_t start_pos = 0,
                 const bool want_logits = true) {
        const int64_t T = (int64_t) toks.size();
        // The activations hold max_tokens_ rows and the KV cache max_seq_
        // positions: anything past either is an out-of-bounds write.
        if (T <= 0 || T > max_tokens_ || start_pos < 0 || start_pos + T > max_seq_) {
            return fail("forward: tokens exceed the activation or KV capacity");
        }
        hipEvent_t step_a{}, step_b{};
        const bool time_step = T == 1 && std::getenv("OMPH_TIMING") != nullptr;
        if (time_step) {
            (void) hipEventCreate(&step_a);
            (void) hipEventCreate(&step_b);
            (void) hipEventRecord(step_a, nullptr);
        }
        const int64_t ne = h_.n_embd;

        // token embeddings, one row at a time (the table is quantized)
        const omph::gguf::TensorInfo * te = file_.tensor("token_embd.weight");
        if (te == nullptr) return fail("token_embd.weight missing");
        const int64_t row_bytes = (int64_t) (te->nbytes / te->ne[1]);
        for (int64_t t = 0; t < T; ++t) {
            const uint8_t * src =
                static_cast<const uint8_t *>(raw_bytes(te->name)) + (size_t) toks[t] * row_bytes;
            if (!omph::kernels::dequantize(te->type, src,
                                           static_cast<uint8_t *>(x_) + t * ne * 4, ne, false,
                                           nullptr)) {
                return fail("embedding dequant failed");
            }
        }

        const auto vec = [&](const std::string & name) -> const float * {
            const omph::gguf::TensorInfo * t = file_.tensor(name);
            return t == nullptr ? nullptr : reinterpret_cast<const float *>(
                                                static_cast<const uint8_t *>(dev_weights_) +
                                                off_.at(t->name));
        };

        // Set when the previous layer's final residual add already wrote this
        // layer's attn_norm(x) into h16_ (one fused launch instead of two).
        bool h16_normed = false;
        for (int64_t il = 0; il < h_.n_layer; ++il) {
            const std::string p = "blk." + std::to_string(il) + ".";
            const bool recurrent = file_.tensor(p + "ssm_a") != nullptr;
            const bool skip_attn = !recurrent && std::getenv("OMPH_SKIP_ATTN") != nullptr;
            const bool skip_ffn = std::getenv("OMPH_SKIP_FFN") != nullptr;
            const float * attn_norm = vec(p + "attn_norm.weight");
            const float * post_norm = vec(p + "post_attention_norm.weight");
            if (attn_norm == nullptr || post_norm == nullptr) {
                return fail("missing layer norms");
            }
            if (!h16_normed &&
                !omph::kernels::add_rms_norm_f16(static_cast<const float *>(x_), nullptr, nullptr,
                                                 attn_norm, h16_, T, ne, (float) h_.eps,
                                                 nullptr)) {
                return fail("attn_norm failed");
            }
            h16_normed = false;
            scratch_.reset();
            if (std::getenv("OMPH_SKIP_BLOCKS") != nullptr) {
                continue;  // ablation only: the layer output is the normed input
            }
            timer_block_.start();
            if (recurrent) {
                if (!gdn_layer(il, p, T)) return false;
            } else if (!skip_attn) {
                if (!attn_layer(il, p, T, start_pos)) return false;
            }
            timer_block_.stop(t_block_);
            if (skip_ffn) {
                continue;
            }
            // x = ffn(rms_norm(block + x)) + (block + x)
            if (!omph::kernels::add_rms_norm_f16(static_cast<const float *>(blk_),
                                                 static_cast<const float *>(x_),
                                                 static_cast<float *>(resid_), post_norm, h16_,
                                                 T, ne, (float) h_.eps, nullptr)) {
                return fail("residual/norm failed");
            }
            const bool gate_up_ok = fork_join(
                T,
                [&] {
                    return matmul(p + "ffn_up.weight", h16_, static_cast<float *>(ffn2_), h_.n_ff,
                                  ne, T);
                },
                [&] {
                    return matmul(p + "ffn_gate.weight", h16_, static_cast<float *>(ffn1_),
                                  h_.n_ff, ne, T);
                });
            if (!gate_up_ok ||
                !omph::kernels::swiglu_f16(static_cast<const float *>(ffn1_),
                                           static_cast<const float *>(ffn2_), ffn16_, T * h_.n_ff,
                                           nullptr) ||
                !matmul(p + "ffn_down.weight", ffn16_, static_cast<float *>(cur_), ne, h_.n_ff, T)) {
                return fail("ffn failed");
            }
            // x = ffn + resid; for all but the last layer, the same launch also
            // writes the next layer's attn_norm(x) into h16_.
            const float * next_norm =
                il + 1 < h_.n_layer ? vec("blk." + std::to_string(il + 1) + ".attn_norm.weight")
                                    : nullptr;
            if (next_norm != nullptr) {
                if (!omph::kernels::add_rms_norm_f16(static_cast<const float *>(cur_),
                                                     static_cast<const float *>(resid_),
                                                     static_cast<float *>(x_), next_norm, h16_, T,
                                                     ne, (float) h_.eps, nullptr)) {
                    return fail("residual/norm failed");
                }
                h16_normed = true;
            } else if (!omph::kernels::add_out(static_cast<const float *>(cur_),
                                               static_cast<const float *>(resid_),
                                               static_cast<float *>(x_), T * ne, nullptr)) {
                return fail("residual failed");
            }
            if (!trace_dir.empty()) {
                std::vector<float> host((size_t) T * ne);
                (void) hipMemcpy(host.data(), x_, host.size() * 4, hipMemcpyDeviceToHost);
                write_f32(trace_dir + "/l_out-" + std::to_string(il) + ".f32", host);
            }
        }

        if (!want_logits) {
            logits.clear();
            if (hipDeviceSynchronize() != hipSuccess) {
                return fail("forward failed");
            }
            report_phases();
            if (time_step) {
                step_event(step_a, step_b);
            }
            return true;
        }
        const float * out_norm = vec("output_norm.weight");
        if (out_norm == nullptr ||
            !omph::kernels::rms_norm(static_cast<const float *>(x_), out_norm,
                                     static_cast<float *>(cur_), T, ne, (float) h_.eps, 1.0f,
                                     nullptr) ||
            !omph::kernels::cast_f32_to_f16(static_cast<const float *>(cur_), h16_, T * ne,
                                            nullptr)) {
            return fail("output_norm failed");
        }
        if (use_gemv_ && T == 1 && gems_.count("output.weight") != 0) {
            if (!matmul("output.weight", h16_, static_cast<float *>(logits_), h_.n_vocab, ne, T) ||
                hipDeviceSynchronize() != hipSuccess) {
                return fail("lm_head (gemv) failed");
            }
            logits.resize((size_t) T * h_.n_vocab);
            (void) hipMemcpy(logits.data(), logits_, logits.size() * 4, hipMemcpyDeviceToHost);
            report_phases();
            if (time_step) {
                step_event(step_a, step_b);
            }
            return true;
        }
        // Only the last token's logits: at 32k the full (T, 248320) f32 buffer is
        // 33 GB, and the long-context KV validation only needs the final row.
        if (last_logits_only_) {
            const uint8_t * xlast = static_cast<const uint8_t *>(h16_) + (T - 1) * ne * 2;
            const omph::gguf::TensorInfo * ht = file_.tensor("output.weight");
            if (ht == nullptr) {
                return fail("output.weight missing");
            }
            if (use_gemv_ && gems_.count("output.weight") != 0) {
                const auto & e = gems_.at("output.weight");
                const void * w = static_cast<const uint8_t *>(dev_weights_) + e.off;
                if (!gemv_one(e.type, w, xlast, static_cast<float *>(logits_), h_.n_vocab, ne, nullptr)) {
                    return fail("lm_head gemv failed");
                }
            } else if (!head_chunked(xlast, 1)) {
                return fail("lm_head failed");
            }
            if (hipDeviceSynchronize() != hipSuccess) {
                return fail("lm_head failed");
            }
            logits.resize((size_t) h_.n_vocab);
            (void) hipMemcpy(logits.data(), logits_, logits.size() * 4, hipMemcpyDeviceToHost);
            report_phases();
            if (time_step) {
                step_event(step_a, step_b);
            }
            return true;
        }
        // Small-batch head: for a Q4_K head with 2+ tokens, run four tokens per
        // weight read instead of materializing f16 (PLAN.md §10.1's N = 1..5).
        if (use_gemv_ && T > 1 && gems_.count("output.weight") != 0 &&
            gems_.at("output.weight").type == 12) {
            const auto & e = gems_.at("output.weight");
            const void * w = static_cast<const uint8_t *>(dev_weights_) + e.off;
            int64_t t0 = 0;
            for (; t0 + 4 <= T; t0 += 4) {
                if (!omph::kernels::gemv_q4k_b4(w, static_cast<const uint8_t *>(h16_) + t0 * ne * 2,
                                                static_cast<float *>(logits_) + t0 * h_.n_vocab,
                                                h_.n_vocab, ne, nullptr)) {
                    return fail("lm_head batch4 failed");
                }
            }
            for (; t0 < T; ++t0) {
                if (!omph::kernels::gemv_q4k(
                        w, static_cast<const uint8_t *>(h16_) + t0 * ne * 2,
                        static_cast<float *>(logits_) + t0 * h_.n_vocab, h_.n_vocab, ne,
                        nullptr)) {
                    return fail("lm_head tail failed");
                }
            }
            if (hipDeviceSynchronize() != hipSuccess) {
                return fail("lm_head failed");
            }
            logits.resize((size_t) T * h_.n_vocab);
            (void) hipMemcpy(logits.data(), logits_, logits.size() * 4, hipMemcpyDeviceToHost);
            report_phases();
            return true;
        }
        if (!head_chunked(h16_, T)) {
            return false;
        }
        if (hipDeviceSynchronize() != hipSuccess) {
            return fail("lm_head failed");
        }
        logits.resize((size_t) T * h_.n_vocab);
        (void) hipMemcpy(logits.data(), logits_, logits.size() * 4, hipMemcpyDeviceToHost);
        report_phases();
        if (time_step) {
            step_event(step_a, step_b);
        }
        return true;
    }

    // lm head in vocab chunks, through the f16 dequant + hipBLASLt path: the
    // whole f16 head (248k x 5120, 2.5 GB) would not fit. Writes (T, n_vocab)
    // into logits_.
    bool head_chunked(const void * x16, const int64_t T) {
        const int64_t ne = h_.n_embd;
        const omph::gguf::TensorInfo * head = file_.tensor("output.weight");
        if (head == nullptr) {
            return fail("output.weight missing");
        }
        const int64_t head_row_bytes = (int64_t) (head->nbytes / head->ne[1]);
        if (gems_.count(head->name) != 0) {
            (void) hipMemcpy(raw_stage_, file_.tensor_data(*head), (size_t) head->nbytes,
                             hipMemcpyHostToDevice);
        }
        const uint8_t * head_src = gems_.count(head->name) != 0
                                       ? static_cast<const uint8_t *>(raw_stage_)
                                       : static_cast<const uint8_t *>(dev_weights_) +
                                             off_.at(head->name);
        const int64_t chunk = std::min<int64_t>(h_.n_vocab, 32768);
        for (int64_t v0 = 0; v0 < h_.n_vocab; v0 += chunk) {
            const int64_t rows = std::min(chunk, h_.n_vocab - v0);
            void * wh = head16_;
            timer_stage_.start();
            const bool head_dq = omph::kernels::dequantize(head->type,
                                                           head_src + v0 * head_row_bytes, wh,
                                                           rows * ne, true, nullptr);
            timer_stage_.stop(t_stage_);
            timer_gemm_.start();
            const bool head_gm =
                linear_.run(wh, x16, static_cast<float *>(tmp_logits_), rows, ne, T);
            timer_gemm_.stop(t_gemm_);
            if (!head_dq || !head_gm ||
                hipMemcpy2D(static_cast<uint8_t *>(logits_) + v0 * 4, (size_t) h_.n_vocab * 4,
                            tmp_logits_, (size_t) rows * 4, (size_t) rows * 4, (size_t) T,
                            hipMemcpyDeviceToDevice) != hipSuccess) {
                return fail("lm_head failed");
            }
        }
        return true;
    }

    void report_phases() {
        if (std::getenv("OMPH_TIMING") == nullptr) {
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

    void step_event(hipEvent_t a, hipEvent_t b) {
        (void) hipEventRecord(b, nullptr);
        (void) hipEventSynchronize(b);
        float ms = 0.0f;
        (void) hipEventElapsedTime(&ms, a, b);
        std::fprintf(stderr, "step gpu %.2f ms\n", ms);
        (void) hipEventDestroy(a);
        (void) hipEventDestroy(b);
    }

    const HParams & hparams() const { return h_; }

private:
    // Returns false: every caller is a bool function that reports failure.
    static bool fail(const char * msg) {
        // The pending HIP error (if any) is usually the actual cause.
        const hipError_t err = hipGetLastError();
        std::fprintf(stderr, "%s%s%s\n", msg, err != hipSuccess ? ": " : "",
                     err != hipSuccess ? hipGetErrorString(err) : "");
        return false;
    }

    static HParams read_hparams(const omph::gguf::File & f) {
        HParams h;
        meta_int(f, "qwen35.embedding_length", h.n_embd);
        meta_int(f, "qwen35.block_count", h.n_layer);
        meta_int(f, "qwen35.attention.head_count", h.n_head);
        meta_int(f, "qwen35.attention.head_count_kv", h.n_head_kv);
        meta_int(f, "qwen35.attention.key_length", h.head_dim);
        meta_int(f, "qwen35.rope.dimension_count", h.n_rot);
        meta_int(f, "qwen35.feed_forward_length", h.n_ff);
        meta_int(f, "qwen35.ssm.group_count", h.ssm_n_kh);
        meta_int(f, "qwen35.ssm.time_step_rank", h.ssm_n_vh);
        meta_int(f, "qwen35.ssm.state_size", h.ssm_s);
        meta_int(f, "qwen35.ssm.inner_size", h.ssm_inner);
        meta_int(f, "qwen35.ssm.conv_kernel", h.ssm_conv_k);
        h.eps = meta_float(f, "qwen35.attention.layer_norm_rms_epsilon", 1e-6);
        h.freq_base = meta_float(f, "qwen35.rope.freq_base", 10000.0);
        // the vocab comes from the embedding table; trailing MTP blocks (the
        // nextn.* group, ignored by the normal decode path) are not part of the
        // stack we run here
        if (const omph::gguf::TensorInfo * te = f.tensor("token_embd.weight")) {
            h.n_vocab = (int64_t) te->ne[1];
        }
        while (h.n_layer > 0 &&
               f.tensor("blk." + std::to_string(h.n_layer - 1) + ".nextn.eh_proj.weight") !=
                   nullptr) {
            --h.n_layer;
        }
        if (h.n_embd <= 0 || h.n_layer <= 0 || h.n_vocab <= 0 || h.n_head <= 0 ||
            h.head_dim <= 0 || h.n_ff <= 0) {
            throw std::runtime_error("incomplete hyperparameters");
        }
        return h;
    }

    bool attn_layer(const int64_t il, const std::string & p, const int64_t T,
                    const int64_t pos0) {
        const int64_t ne = h_.n_embd;
        const int64_t q_out = h_.n_head * 2 * h_.head_dim;
        const int64_t kv_out = h_.n_head_kv * h_.head_dim;
        // Null in the quantized-KV mode, where the f32 cache is never allocated.
        float * k_cache = kv_k_ == nullptr
                              ? nullptr
                              : static_cast<float *>(kv_k_) + kv_index_[il] * max_seq_ * kv_out;
        float * v_cache = kv_v_ == nullptr
                              ? nullptr
                              : static_cast<float *>(kv_v_) + kv_index_[il] * max_seq_ * kv_out;
        const float * q_norm = f32_ref(p + "attn_q_norm.weight");
        const float * k_norm = f32_ref(p + "attn_k_norm.weight");

        const bool proj_ok = fork_join(
            T,
            [&] {
                return matmul(p + "attn_k.weight", h16_, static_cast<float *>(k_), kv_out, ne, T) &&
                       matmul(p + "attn_v.weight", h16_, static_cast<float *>(v_), kv_out, ne, T);
            },
            [&] {
                return matmul(p + "attn_q.weight", h16_, static_cast<float *>(fused_), q_out, ne,
                              T);
            });
        // split, QK-norm, RoPE, the Hadamard rotation and (quantized cache)
        // the KV write, in one launch (#79).
        omph::kernels::AttnPrep prep;
        prep.qg = static_cast<const float *>(fused_);
        prep.q = static_cast<float *>(q_);
        prep.gate = static_cast<float *>(gate_);
        prep.k = static_cast<float *>(k_);
        prep.v = static_cast<float *>(v_);
        prep.q_norm = q_norm;
        prep.k_norm = k_norm;
        prep.tokens = T;
        prep.pos0 = pos0;
        prep.nh = h_.n_head;
        prep.nkv = h_.n_head_kv;
        prep.hd = h_.head_dim;
        prep.n_rot = (int) h_.n_rot;
        prep.freq_base = (float) h_.freq_base;
        prep.eps = (float) h_.eps;
        prep.rotate = kv_q8q4_;
        if (kv_q8q4_) {
            const QuantKv c = quant_kv(il);
            prep.k_q8 = c.kq;
            prep.k_scales = reinterpret_cast<__half *>(c.ksc);
            prep.v_q4 = c.vq;
            prep.v_scales = reinterpret_cast<__half *>(c.vsc);
            prep.k16 = kv_window_ > 0 ? reinterpret_cast<__half *>(c.k16) : nullptr;
            prep.v16 = kv_window_ > 0 ? reinterpret_cast<__half *>(c.v16) : nullptr;
            prep.window = kv_window_;
            prep.k_q4 = kv_k4_;
        }
        if (!proj_ok || !omph::kernels::attn_prep(prep, nullptr) ||
            !attn_impl(il, k_cache, v_cache, pos0, T) ||
            !matmul(p + "attn_output.weight", ffn16_, static_cast<float *>(blk_), ne,
                    h_.n_head * h_.head_dim, T)) {
            return fail("attention layer failed");
        }
        return true;
    }

    bool gdn_layer(const int64_t il, const std::string & p, const int64_t T) {
        const int64_t ne = h_.n_embd;
        const int64_t n_kh = h_.ssm_n_kh;
        const int64_t n_vh = h_.ssm_n_vh;
        const int64_t s = h_.ssm_s;
        const int64_t v_dim = h_.ssm_inner / n_vh;
        const int64_t q_dims = n_kh * s;
        const int64_t k_dims = n_kh * s;
        const int64_t v_dims = n_vh * v_dim;
        const int64_t channels = q_dims + k_dims + v_dims;
        const float l2_scale = 1.0f / std::sqrt((float) s);

        const float * dt_bias = f32_ref(p + "ssm_dt.bias");
        const float * ssm_a = f32_ref(p + "ssm_a");
        const float * ssm_norm = f32_ref(p + "ssm_norm.weight");
        const float * conv_w = f32_ref(p + "ssm_conv1d.weight");
        uint8_t * st = static_cast<uint8_t *>(states_[il]);
        const int64_t n_conv_f = (h_.ssm_conv_k - 1) * channels;
        float * conv_a = reinterpret_cast<float *>(st);
        float * conv_b = conv_a + n_conv_f;
        float * conv_cur = conv_flip_[il] ? conv_b : conv_a;
        float * conv_new = conv_flip_[il] ? conv_a : conv_b;
        float * seq_state = reinterpret_cast<float *>(st + 2 * n_conv_f * 4);


        const bool proj_ok = fork_join(
            T,
            [&] {
                return matmul(p + "attn_gate.weight", h16_, static_cast<float *>(z_), v_dims, ne,
                              T) &&
                       matmul(p + "ssm_beta.weight", h16_, static_cast<float *>(beta_), n_vh, ne,
                              T) &&
                       matmul(p + "ssm_alpha.weight", h16_, static_cast<float *>(alpha_), n_vh, ne,
                              T);
            },
            [&] {
                return matmul(p + "attn_qkv.weight", h16_, static_cast<float *>(fused_), channels,
                              ne, T);
            });
        if (!proj_ok) {
            return fail("gdn projection failed");
        }
        // conv, L2 norms, gates, delta rule and gated norm: one launch per token
        // (#78, #83). The L2 norm is rms_norm(x, eps/s) / sqrt(s).
        omph::kernels::GdnStep step;
        step.state = seq_state;
        step.qkv = static_cast<const float *>(fused_);
        step.conv_w = conv_w;
        step.conv_cur = conv_cur;
        step.conv_new = conv_new;
        step.dt_bias = dt_bias;
        step.ssm_a = ssm_a;
        step.norm_w = ssm_norm;
        step.tokens = T;
        step.channels = channels;
        step.q_dims = q_dims;
        step.kv_dims = k_dims;
        step.conv_k = h_.ssm_conv_k;
        step.n_kh = n_kh;
        step.eps_l2 = (float) (h_.eps / (double) s);
        step.l2_scale = l2_scale;
        step.eps_norm = (float) h_.eps;
        for (int64_t t = 0; t < T; ++t) {
            step.t = t;
            step.beta = static_cast<const float *>(beta_) + t * n_vh;
            step.alpha = static_cast<const float *>(alpha_) + t * n_vh;
            step.z = static_cast<const float *>(z_) + t * v_dims;
            step.out16 = reinterpret_cast<__half *>(static_cast<uint8_t *>(ffn16_) + t * v_dims * 2);
            if (!omph::kernels::gdn_step(step, n_vh, nullptr)) {
                return fail("delta rule failed");
            }
        }
        if (!matmul(p + "ssm_out.weight", ffn16_, static_cast<float *>(blk_), ne, v_dims, T)) {
            return fail("gdn output failed");
        }
        conv_flip_[il] ^= 1;
        return true;
    }

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
    QuantKv quant_kv(const int64_t il) const {
        const int64_t kv_out = h_.n_head_kv * h_.head_dim;
        const int64_t nblk = h_.head_dim / 32;
        const int64_t kvl = kv_index_[il];
        return {static_cast<uint8_t *>(kv_kq_) + kvl * max_seq_ * kv_out / (kv_k4_ ? 2 : 1),
                static_cast<uint8_t *>(kv_ks_) + kvl * max_seq_ * h_.n_head_kv * nblk * 2,
                static_cast<uint8_t *>(kv_vq_) + kvl * max_seq_ * kv_out / 2,
                static_cast<uint8_t *>(kv_vs_) + kvl * max_seq_ * h_.n_head_kv * nblk * 2,
                static_cast<uint8_t *>(kv_k16_) + kvl * kv_window_ * kv_out * 2,
                static_cast<uint8_t *>(kv_v16_) + kvl * kv_window_ * kv_out * 2};
    }

    // KV write + attention, on either the f32 cache (memcpy) or the Q8/Q4 one
    // (quantize); attention_gqa then reads either, dequantizing on the fly.
    bool attn_impl(const int64_t il, float * k_cache, float * v_cache, const int64_t pos0,
                   const int64_t T) {
        const int64_t kv_out = h_.n_head_kv * h_.head_dim;
        const float scale = 1.0f / std::sqrt((float) h_.head_dim);
        if (kv_q8q4_) {
            // The rows were quantized into the cache by attn_prep.
            const QuantKv c = quant_kv(il);
            omph::kernels::KvCache kv;
            kv.k_q8 = c.kq;
            kv.k_scales = c.ksc;
            kv.v_q4 = c.vq;
            kv.v_scales = c.vsc;
            kv.k16 = kv_window_ > 0 ? c.k16 : nullptr;
            kv.v16 = kv_window_ > 0 ? c.v16 : nullptr;
            kv.window = kv_window_;
            kv.k_q4 = kv_k4_;
            return omph::kernels::attention_gqa(
                static_cast<const float *>(q_), kv, static_cast<const float *>(gate_),
                nullptr, T, pos0 + T, h_.n_head, h_.n_head_kv, h_.head_dim, scale, true,
                attn_work_, attn_work_bytes_, nullptr, ffn16_);
        }
        if (k_cache == nullptr || v_cache == nullptr) {
            return false;
        }
        // hipMemcpyDefault: the cache is device memory, or pinned host memory
        // with OMPH_KV_HOST.
        if (hipMemcpy(k_cache + pos0 * kv_out, k_, (size_t) T * kv_out * 4, hipMemcpyDefault) !=
                hipSuccess ||
            hipMemcpy(v_cache + pos0 * kv_out, v_, (size_t) T * kv_out * 4, hipMemcpyDefault) !=
                hipSuccess) {
            return false;
        }
        if (kv_host_) {
            const size_t bytes = (size_t) (pos0 + T) * kv_out * 4;
            if (hipMemcpy(kv_stage_k_, k_cache, bytes, hipMemcpyHostToDevice) != hipSuccess ||
                hipMemcpy(kv_stage_v_, v_cache, bytes, hipMemcpyHostToDevice) != hipSuccess) {
                return false;
            }
            k_cache = static_cast<float *>(kv_stage_k_);
            v_cache = static_cast<float *>(kv_stage_v_);
        }
        omph::kernels::KvCache kv;
        kv.k_f32 = k_cache;
        kv.v_f32 = v_cache;
        return omph::kernels::attention_gqa(static_cast<const float *>(q_), kv,
                                            static_cast<const float *>(gate_),
                                            nullptr, T, pos0 + T, h_.n_head, h_.n_head_kv,
                                            h_.head_dim, scale, false, attn_work_,
                                            attn_work_bytes_, nullptr, ffn16_);
    }

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

    // Small-batch GEMV: one weight read per four tokens. Only the types that have
    // this form answer true; the others stay on the f16 path.
    static bool gemv_batch4(const int type, const void * w, const void * x, float * y,
                            const int64_t n_out, const int64_t k, hipStream_t st) {
        switch (type) {
            case 12: return omph::kernels::gemv_q4k_b4(w, x, y, n_out, k, st);
            case 21: return omph::kernels::gemv_iq3s_b4(w, x, y, n_out, k, st);
            case 23: return omph::kernels::gemv_iq4_xs_b4(w, x, y, n_out, k, st);
            case 18: return omph::kernels::gemv_iq3_xxs_b4(w, x, y, n_out, k, st);
            default: return false;
        }
    }

    // One matmul: the fused GEMV for single-token steps when it is available for
    // this tensor, the small-batch GEMV for a few tokens, otherwise the f16
    // dequant + hipBLASLt path.
    bool matmul(const std::string & name, const void * x16, float * y, const int64_t n_out,
                const int64_t k, const int64_t T) {
        if (T == 1) {
            // BF16 weights: bf16 is the top half of an f32, so a direct dot
            // product beats converting to f16 and running a tiny M = 1 GEMM.
            const omph::gguf::TensorInfo * ti = file_.tensor(name);
            if (ti != nullptr && ti->type == 30 && ti->ne[1] == n_out && ti->ne[0] == k &&
                std::getenv("OMPH_NO_BF16_GEMV") == nullptr) {
                const void * w = static_cast<const uint8_t *>(dev_weights_) + off_.at(name);
                timer_gemv_.start();
                const bool ok = omph::kernels::gemv_bf16(w, x16, y, n_out, k, gemv_stream_);
                timer_gemv_.stop(t_gemv_);
                if (ok) {
                    return true;
                }
            }
        }
        if (use_gemv_ && T == 1) {
            const auto it = gems_.find(name);
            if (it != gems_.end() && it->second.rows == n_out && it->second.k == k) {
                const void * w = static_cast<const uint8_t *>(dev_weights_) + it->second.off;
                if (std::getenv("OMPH_SKIP_GEMV") != nullptr) {
                    return true;  // ablation only: wrong results, valid timing
                }
                // Ablation only: skip the GEMVs of one GGUF type. The step time
                // difference is that type's real cost; rocprofv3's per-kernel
                // times inflate some kernels by up to 40 % (#63).
                if (const char * st = std::getenv("OMPH_SKIP_GEMV_TYPE")) {
                    if ((uint32_t) std::atoi(st) == it->second.type) {
                        return true;
                    }
                }
                timer_gemv_.start();
                const bool gemv_ok = gemv_one(it->second.type, w, x16, y, n_out, k, gemv_stream_);
                timer_gemv_.stop(t_gemv_);
                if (gemv_ok) {
                    return true;
                }
                if (std::getenv("OMPH_TRACE_F16") != nullptr) {
                    const omph::gguf::TensorInfo * ti = file_.tensor(name);
                    std::fprintf(stderr,
                                 "f16 path: %s (entry %s, type %u, ne %lld x %lld, %lld B)\n",
                                 name.c_str(), it != gems_.end() ? "yes" : "no",
                                 ti != nullptr ? (unsigned) ti->type : 0u,
                                 ti != nullptr ? (long long) ti->ne[0] : 0,
                                 ti != nullptr ? (long long) ti->ne[1] : 0,
                                 ti != nullptr ? (long long) ti->nbytes : 0);
                }
            } else if (std::getenv("OMPH_TRACE_F16") != nullptr) {
                const omph::gguf::TensorInfo * ti = file_.tensor(name);
                std::fprintf(stderr, "f16 path: %s (not a gemv tensor, type %u, %lld B)\n",
                             name.c_str(), ti != nullptr ? (unsigned) ti->type : 0u,
                             ti != nullptr ? (long long) ti->nbytes : 0);
            }
        }
        if (use_gemv_ && T > 1) {
            const auto it = gems_.find(name);
            if (it != gems_.end() && it->second.rows == n_out && it->second.k == k &&
                !it->second.has_b4 && has_gemv_type(it->second.type)) {
                // Types with a single-token kernel but no small-batch form: run one
                // per token. No staging, so a long prefill stays inside VRAM — it
                // just does not amortize the weight read (the b4 port is the fix).
                const uint8_t * w = static_cast<const uint8_t *>(dev_weights_) + it->second.off;
                const auto * xb = static_cast<const uint8_t *>(x16);
                timer_gemv_.start();
                bool ok = true;
                for (int64_t t0 = 0; t0 < T && ok; ++t0) {
                    ok = gemv_one(it->second.type, w, xb + t0 * k * 2, y + t0 * n_out, n_out, k,
                                  gemv_stream_);
                }
                timer_gemv_.stop(t_gemv_);
                if (ok) {
                    return true;
                }
            }
            if (it != gems_.end() && it->second.rows == n_out && it->second.k == k &&
                it->second.has_b4) {
                const uint8_t * w = static_cast<const uint8_t *>(dev_weights_) + it->second.off;
                const auto * xb = static_cast<const uint8_t *>(x16);
                timer_gemv_.start();
                bool ok = true;
                int64_t t0 = 0;
                for (; t0 + 4 <= T && ok; t0 += 4) {
                    ok = gemv_batch4(it->second.type, w, xb + t0 * k * 2, y + t0 * n_out, n_out, k,
                                     gemv_stream_);
                }
                for (; t0 < T && ok; ++t0) {
                    ok = gemv_one(it->second.type, w, xb + t0 * k * 2, y + t0 * n_out, n_out, k,
                                  gemv_stream_);
                }
                timer_gemv_.stop(t_gemv_);
                if (ok) {
                    return true;
                }
            }
        }
        if (std::getenv("OMPH_SKIP_STAGE") != nullptr) {
            return true;  // ablation only
        }
        void * w = stage_w(name);
        timer_gemm_.start();
        const bool ok = linear_.run(w, x16, y, n_out, k, T);
        timer_gemm_.stop(t_gemm_);
        return ok;
    }

    // Device pointer to the original GGUF bytes: they live in the image unless
    // the tensor was repacked, in which case they are staged from the file.
    const void * raw_bytes(const std::string & name) {
        const auto it = gems_.find(name);
        if (it == gems_.end()) {
            return static_cast<const uint8_t *>(dev_weights_) + off_.at(name);
        }
        const omph::gguf::TensorInfo * t = file_.tensor(name);
        if (hipMemcpy(raw_stage_, file_.tensor_data(*t), (size_t) t->nbytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            throw std::runtime_error("cannot stage " + name);
        }
        return raw_stage_;
    }

    // Types that have a fused GEMV kernel: for these the f16 form is never worth
    // keeping (it would be two to four times the quantized size, and they are the
    // bulk of the model).
    static bool has_gemv_type(const uint32_t type) {
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

    void * stage_w(const std::string & name) {
        const omph::gguf::TensorInfo * t = file_.tensor(name);
        if (t == nullptr) {
            throw std::runtime_error("missing tensor " + name);
        }
        // Small tensors that never got a quantized kernel (IQ1_M) were being
        // converted to f16 on every call — 1.6 ms of every decode step for one
        // 18.6 MiB tensor. Convert once, keep it: at most 64 MiB, which keeps the
        // big repacked weights (whose f16 form would not fit VRAM) out.
        const int64_t n_elems = numel(*t);
        bool cacheable = std::getenv("OMPH_NO_F16_CACHE") == nullptr &&
                         !has_gemv_type(t->type) && n_elems * 2 <= (256 << 20);
        if (cacheable) {
            const auto it = f16_cache_.find(name);
            if (it != f16_cache_.end()) {
                if (std::getenv("OMPH_TRACE_STAGE") != nullptr) {
                    std::fprintf(stderr, "stage %-40s CACHED\n", name.c_str());
                }
                return it->second;
            }
        }
        if (std::getenv("OMPH_TRACE_STAGE") != nullptr) {
            std::fprintf(stderr, "stage %-40s type %u %lld B cacheable %d\n", name.c_str(),
                         (unsigned) t->type, (long long) t->nbytes, cacheable ? 1 : 0);
        }
        timer_stage_.start();
        const int64_t n = numel(*t);
        void * dst = nullptr;
        if (cacheable) {
            if (hipMalloc(&dst, (size_t) n * 2) != hipSuccess) {
                // Long contexts fill VRAM: caching is an optimization, never a
                // reason to fail. Fall back to the per-call staging buffer, and
                // clear the out-of-memory error, which the next kernel wrapper's
                // hipGetLastError() would otherwise report as its own failure.
                (void) hipGetLastError();
                cacheable = false;
                dst = scratch_.alloc((size_t) n * 2);
            }
        } else {
            dst = scratch_.alloc((size_t) n * 2);
        }
        if (std::getenv("OMPH_TRACE_ALLOC") != nullptr) {
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
            f16_cache_owned_.push_back(dst);
        }
        return dst;
    }

    const float * f32_ref(const std::string & name) {
        const omph::gguf::TensorInfo * t = file_.tensor(name);
        if (t == nullptr) {
            throw std::runtime_error("missing tensor " + name);
        }
        return reinterpret_cast<const float *>(static_cast<const uint8_t *>(dev_weights_) +
                                               off_.at(name));
    }

    struct GemvEntry {
        size_t off = 0;
        size_t bytes = 0;
        int64_t rows = 0;
        int64_t k = 0;
        uint32_t type = 0;
        bool has_b4 = false;  // a small-batch kernel exists for this type
    };

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
    Scratch scratch_;
    std::vector<void *> states_;
    bool last_logits_only_ = false;
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
    void * conv_out_ = nullptr;
    void * z_ = nullptr;
    void * q_ = nullptr;
    void * gate_ = nullptr;
    void * k_ = nullptr;
    void * v_ = nullptr;
    void * sk_ = nullptr;
    void * dvec_ = nullptr;
    void * beta_ = nullptr;
    void * alpha_ = nullptr;
    void * ffn1_ = nullptr;
    void * ffn2_ = nullptr;
    void * ffn16_ = nullptr;
    void * logits_ = nullptr;
    void * tmp_logits_ = nullptr;
    void * head16_ = nullptr;
    void * raw_stage_ = nullptr;
    std::map<std::string, void *> f16_cache_;
    std::vector<void *> f16_cache_owned_;
};

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tokens.txt> <out-logits.f32> "
                             "[--trace-dir DIR] [--tokens N] [--last-logits | --logits-tail N] "
                             "[--generate N --gen-out FILE] [--gemv]\n",
                     argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    const std::string tok_path = argv[2];
    const std::string logits_path = argv[3];
    std::string trace_dir;
    std::string gen_path;
    int64_t max_tokens = 0;
    bool last_logits = false;
    int64_t logits_tail = 0;
    int64_t generate = 0;
    bool use_gemv = false;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace-dir") == 0 && i + 1 < argc) {
            trace_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--last-logits") == 0) {
            last_logits = true;
        } else if (std::strcmp(argv[i], "--logits-tail") == 0 && i + 1 < argc) {
            logits_tail = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            max_tokens = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            generate = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--gemv") == 0) {
            use_gemv = true;
        } else if (std::strcmp(argv[i], "--gen-out") == 0 && i + 1 < argc) {
            gen_path = argv[++i];
        }
    }
    try {
        std::vector<int32_t> toks;
        {
            std::ifstream in(tok_path);
            int64_t v = 0;
            while (in >> v) {
                toks.push_back((int32_t) v);
            }
        }
        if (toks.empty()) {
            std::fprintf(stderr, "no tokens read from %s\n", tok_path.c_str());
            return 1;
        }
        if (max_tokens > 0 && (int64_t) toks.size() > max_tokens) {
            toks.resize((size_t) max_tokens);
        }
        if (generate > 0 && gen_path.empty()) {
            std::fprintf(stderr, "--generate also needs --gen-out\n");
            return 2;
        }
        // Repacking only pays off when the decode runs: a prefill-only run is
        // better off with the raw bytes in place (no staging fallback).
        // Long prompts run in chunks: the activation buffers are sized by the
        // chunk, the KV cache by the whole sequence. Without this, a 8k prompt
        // needs ~3 GB of activations on top of the weights and does not fit.
        const int64_t total_len = (int64_t) toks.size() + generate + 8;
        int64_t act_chunk = total_len < 512 ? total_len : 512;
        if (max_tokens > 0 && max_tokens < act_chunk) {
            act_chunk = max_tokens;
        }
        // The per-layer trace is one file per layer for one forward call: with
        // several chunks it would hold the last chunk only.
        if (!trace_dir.empty() && (int64_t) toks.size() > act_chunk) {
            std::fprintf(stderr, "--trace-dir needs the prompt in one chunk (<= %lld tokens)\n",
                         (long long) act_chunk);
            return 2;
        }
        Runner runner(model, act_chunk, use_gemv && generate > 0, last_logits, total_len);
        const HParams & h = runner.hparams();
        // Every chunk's rows are kept, so the file is the whole prompt's logits;
        // with --last-logits only the last row, with --logits-tail N the last N.
        // Chunks that do not reach the rows being kept skip the lm_head.
        const int64_t n_toks = (int64_t) toks.size();
        const int64_t keep = last_logits ? 1 : (logits_tail > 0 ? std::min(logits_tail, n_toks)
                                                                : n_toks);
        std::vector<float> logits;
        std::vector<float> part_logits;
        for (int64_t off = 0; off < n_toks; off += act_chunk) {
            const int64_t n = std::min<int64_t>(act_chunk, n_toks - off);
            const std::vector<int32_t> part(toks.begin() + (size_t) off,
                                            toks.begin() + (size_t) (off + n));
            const bool want = off + n > n_toks - keep;
            if (!runner.forward(part, part_logits, trace_dir, off, want)) {
                return 1;
            }
            if (last_logits) {
                logits.swap(part_logits);
            } else if (want) {
                // rows of this chunk inside the tail
                const int64_t first = std::max<int64_t>(0, (n_toks - keep) - off);
                logits.insert(logits.end(), part_logits.begin() + (size_t) (first * h.n_vocab),
                              part_logits.end());
            }
        }
        write_f32(logits_path, logits);
        const auto argmax = [&](const float * row) {
            int64_t best = 0;
            for (int64_t i = 1; i < h.n_vocab; ++i) {
                if (row[i] > row[best]) {
                    best = i;
                }
            }
            return (int32_t) best;
        };
        if (generate > 0) {
            // greedy decode: one token per step, reusing the KV cache, the conv
            // state and the delta-net state
            std::vector<int32_t> gen;
            int32_t next = argmax(logits.data() + (logits.size() - h.n_vocab));
            for (int64_t i = 0; i < generate; ++i) {
                gen.push_back(next);
                if (i + 1 == generate) {
                    break;
                }
                const std::vector<int32_t> one{next};
                if (!runner.forward(one, logits, std::string(), (int64_t) toks.size() + i)) {
                    return 1;
                }
                next = argmax(logits.data());
            }
            if (FILE * f = std::fopen(gen_path.c_str(), "w")) {
                for (const int32_t t : gen) {
                    std::fprintf(f, "%d\n", t);
                }
                std::fclose(f);
            }
            std::printf("generated:");
            for (const int32_t t : gen) {
                std::printf(" %d", t);
            }
            std::printf("\n");
        }
        // greedy tokens for the report
        const size_t n_rows = logits.size() / (size_t) h.n_vocab;
        std::printf("logits: %lld x %lld -> %s\ngreedy:", (long long) n_rows,
                    (long long) h.n_vocab, logits_path.c_str());
        for (size_t t = 0; t < n_rows; ++t) {
            const float * row = logits.data() + t * (size_t) h.n_vocab;
            std::printf(" %lld", (long long) argmax(row));
        }
        std::printf("\n");
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
    return 0;
}
