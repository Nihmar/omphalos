// omph-run — naive full forward pass of the 64-layer stack (PLAN.md §12, M2).
//
// usage: omph-run <model.gguf> <tokens.txt> <out-logits.f32> [--trace-dir DIR]
//   tokens.txt: token ids separated by whitespace.
//   Logits are written as tokens x n_vocab f32; with --trace-dir the per-layer
//   outputs are dumped as l_out-<layer>.f32.
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

class Runner {
public:
    explicit Runner(const std::string & path, const int64_t max_tokens, const bool use_gemv)
        : file_(path), h_(read_hparams(file_)), use_gemv_(use_gemv) {
        const int64_t ne = h_.n_embd;
        const int64_t T = max_tokens;
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
                if (use_gemv_ && bb > 0 && t.nbytes % (uint64_t) bb == 0) {
                    std::vector<uint8_t> packed;
                    if (omph::format::repack_any(t.type, file_.tensor_data(t),
                                                 (int64_t) (t.nbytes / (uint64_t) bb), packed)) {
                        GemvEntry e;
                        e.off = total;
                        e.bytes = packed.size();
                        e.rows = (int64_t) t.ne[1];
                        e.k = (int64_t) t.ne[0];
                        e.type = t.type;
                        gems_[t.name] = e;
                        total += (packed.size() + 255) & ~(size_t) 255;
                        places.push_back(p);
                        continue;
                    }
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
                    (void) omph::format::repack_any(t->type, src,
                                                    (int64_t) (t->nbytes / (uint64_t) bb), packed);
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
        alloc(&o_, T * ssm_v * 4);
        alloc(&q_, T * attn_q * 4);
        alloc(&gate_, T * attn_q * 4);
        alloc(&attn_, T * attn_q * 4);
        alloc(&k_, T * std::max(ssm_q, attn_kv) * 4);
        alloc(&v_, T * std::max(ssm_v, attn_kv) * 4);
        alloc(&sk_, h_.ssm_n_vh * h_.ssm_s * 4);
        alloc(&dvec_, h_.ssm_n_vh * h_.ssm_s * 4);
        alloc(&beta_, T * h_.ssm_n_vh * 4);
        alloc(&alpha_, T * h_.ssm_n_vh * 4);
        alloc(&ffn1_, T * h_.n_ff * 4);
        alloc(&ffn2_, T * h_.n_ff * 4);
        alloc(&ffn16_, T * h_.n_ff * 2);
        alloc(&logits_, T * h_.n_vocab * 4);
        alloc(&tmp_logits_, T * std::min<int64_t>(h_.n_vocab, 32768) * 4);
        alloc(&head16_, (size_t) std::min<int64_t>(h_.n_vocab, 32768) * ne * 2);
        alloc(&raw_stage_, (size_t) 768 * 1024 * 1024);  // raw bytes for the f16 fallback
        max_seq_ = T;
        int64_t n_kv = 0;
        for (int64_t il = 0; il < h_.n_layer; ++il) {
            const bool recurrent = file_.tensor("blk." + std::to_string(il) + ".ssm_a") != nullptr;
            kv_index_.push_back(recurrent ? -1 : n_kv++);
        }
        alloc(&kv_k_, (size_t) n_kv * T * attn_kv * 4);
        alloc(&kv_v_, (size_t) n_kv * T * attn_kv * 4);
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
        if (dev_weights_ != nullptr) {
            (void) hipFree(dev_weights_);
        }
    }

    bool forward(const std::vector<int32_t> & toks, std::vector<float> & logits,
                 const std::string & trace_dir, const int64_t start_pos = 0) {
        const int64_t T = (int64_t) toks.size();
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

        for (int64_t il = 0; il < h_.n_layer; ++il) {
            const std::string p = "blk." + std::to_string(il) + ".";
            const bool recurrent = file_.tensor(p + "ssm_a") != nullptr;
            const float * attn_norm = vec(p + "attn_norm.weight");
            const float * post_norm = vec(p + "post_attention_norm.weight");
            if (attn_norm == nullptr || post_norm == nullptr) {
                return fail("missing layer norms");
            }
            if (!omph::kernels::rms_norm(static_cast<const float *>(x_), attn_norm,
                                         static_cast<float *>(cur_), T, ne, (float) h_.eps, 1.0f,
                                         nullptr)) {
                return fail("attn_norm failed");
            }
            scratch_.reset();
            if (recurrent) {
                if (!gdn_layer(il, p, T)) return false;
            } else {
                if (!attn_layer(il, p, T, start_pos)) return false;
            }
            // x = ffn(rms_norm(block + x)) + (block + x)
            if (!omph::kernels::add_out(static_cast<const float *>(blk_),
                                        static_cast<const float *>(x_),
                                        static_cast<float *>(resid_), T * ne, nullptr) ||
                !omph::kernels::rms_norm(static_cast<const float *>(resid_), post_norm,
                                         static_cast<float *>(cur_), T, ne, (float) h_.eps, 1.0f,
                                         nullptr) ||
                !omph::kernels::cast_f32_to_f16(static_cast<const float *>(cur_), h16_, T * ne,
                                                nullptr)) {
                return fail("residual/norm failed");
            }
            if (!matmul(p + "ffn_gate.weight", h16_, static_cast<float *>(ffn1_), h_.n_ff, ne, T) ||
                !matmul(p + "ffn_up.weight", h16_, static_cast<float *>(ffn2_), h_.n_ff, ne, T) ||
                !omph::kernels::silu_inplace(static_cast<float *>(ffn1_), T * h_.n_ff, nullptr) ||
                !omph::kernels::mul_inplace(static_cast<float *>(ffn1_),
                                            static_cast<const float *>(ffn2_), T * h_.n_ff,
                                            nullptr) ||
                !omph::kernels::cast_f32_to_f16(static_cast<const float *>(ffn1_), ffn16_,
                                                T * h_.n_ff, nullptr) ||
                !matmul(p + "ffn_down.weight", ffn16_, static_cast<float *>(cur_), ne, h_.n_ff, T) ||
                !omph::kernels::add_out(static_cast<const float *>(cur_),
                                        static_cast<const float *>(resid_),
                                        static_cast<float *>(x_), T * ne, nullptr)) {
                return fail("ffn failed");
            }
            if (!trace_dir.empty()) {
                std::vector<float> host((size_t) T * ne);
                (void) hipMemcpy(host.data(), x_, host.size() * 4, hipMemcpyDeviceToHost);
                write_f32(trace_dir + "/l_out-" + std::to_string(il) + ".f32", host);
            }
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
            return true;
        }
        // lm head in vocab chunks: the whole f16 head (248k x 5120) would not fit
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
        omph::runtime::Linear linear;
        for (int64_t v0 = 0; v0 < h_.n_vocab; v0 += chunk) {
            const int64_t rows = std::min(chunk, h_.n_vocab - v0);
            void * wh = head16_;
            if (!omph::kernels::dequantize(
                    head->type,
                    head_src + v0 * head_row_bytes,
                    wh, rows * ne, true, nullptr) ||
                !linear.run(wh, h16_, static_cast<float *>(tmp_logits_), rows, ne, T) ||
                hipMemcpy2D(static_cast<uint8_t *>(logits_) + v0 * 4, (size_t) h_.n_vocab * 4,
                            tmp_logits_, (size_t) rows * 4, (size_t) rows * 4, (size_t) T,
                            hipMemcpyDeviceToDevice) != hipSuccess) {
                return fail("lm_head failed");
            }
        }
        if (hipDeviceSynchronize() != hipSuccess) {
            return fail("lm_head failed");
        }
        logits.resize((size_t) T * h_.n_vocab);
        (void) hipMemcpy(logits.data(), logits_, logits.size() * 4, hipMemcpyDeviceToHost);
        return true;
    }

    const HParams & hparams() const { return h_; }

private:
    static int fail(const char * msg) {
        std::fprintf(stderr, "%s\n", msg);
        return 1;
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
        float * k_cache = static_cast<float *>(kv_k_) + kv_index_[il] * max_seq_ * kv_out;
        float * v_cache = static_cast<float *>(kv_v_) + kv_index_[il] * max_seq_ * kv_out;
        const float * q_norm = f32_ref(p + "attn_q_norm.weight");
        const float * k_norm = f32_ref(p + "attn_k_norm.weight");
        if (!omph::kernels::cast_f32_to_f16(static_cast<const float *>(cur_), h16_, T * ne, nullptr)) {
            return fail("cast failed");
        }
        if (!matmul(p + "attn_q.weight", h16_, static_cast<float *>(fused_), q_out, ne, T) ||
            !matmul(p + "attn_k.weight", h16_, static_cast<float *>(k_), kv_out, ne, T) ||
            !matmul(p + "attn_v.weight", h16_, static_cast<float *>(v_), kv_out, ne, T) ||
            !omph::kernels::split_qg(static_cast<const float *>(fused_),
                                     static_cast<float *>(q_), static_cast<float *>(gate_), T,
                                     h_.n_head, h_.head_dim, nullptr) ||
            !omph::kernels::rms_norm(static_cast<const float *>(q_), q_norm,
                                     static_cast<float *>(q_), T * h_.n_head, h_.head_dim,
                                     (float) h_.eps, 1.0f, nullptr) ||
            !omph::kernels::rms_norm(static_cast<const float *>(k_), k_norm,
                                     static_cast<float *>(k_), T * h_.n_head_kv, h_.head_dim,
                                     (float) h_.eps, 1.0f, nullptr) ||
            !omph::kernels::rope_neox(static_cast<float *>(q_), T, h_.n_head, h_.head_dim, h_.n_rot,
                                      (float) h_.freq_base, pos0, nullptr) ||
            !omph::kernels::rope_neox(static_cast<float *>(k_), T, h_.n_head_kv, h_.head_dim,
                                      h_.n_rot, (float) h_.freq_base, pos0, nullptr) ||
            hipMemcpy(k_cache + pos0 * kv_out, k_, (size_t) T * kv_out * 4,
                      hipMemcpyDeviceToDevice) != hipSuccess ||
            hipMemcpy(v_cache + pos0 * kv_out, v_, (size_t) T * kv_out * 4,
                      hipMemcpyDeviceToDevice) != hipSuccess ||
            !omph::kernels::attention(static_cast<const float *>(q_), k_cache, v_cache,
                                      static_cast<const float *>(gate_),
                                      static_cast<float *>(attn_), T, pos0 + T, h_.n_head,
                                      h_.n_head_kv, h_.head_dim,
                                      1.0f / std::sqrt((float) h_.head_dim), nullptr) ||
            !omph::kernels::cast_f32_to_f16(static_cast<const float *>(attn_), ffn16_,
                                            T * h_.n_head * h_.head_dim, nullptr) ||
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

        if (!omph::kernels::cast_f32_to_f16(static_cast<const float *>(cur_), h16_, T * ne, nullptr)) {
            return fail("cast failed");
        }
        if (!matmul(p + "attn_qkv.weight", h16_, static_cast<float *>(fused_), channels, ne, T) ||
            !matmul(p + "attn_gate.weight", h16_, static_cast<float *>(z_), v_dims, ne, T) ||
            !matmul(p + "ssm_beta.weight", h16_, static_cast<float *>(beta_), n_vh, ne, T) ||
            !matmul(p + "ssm_alpha.weight", h16_, static_cast<float *>(alpha_), n_vh, ne, T)) {
            return fail("gdn projection failed");
        }
        if (!omph::kernels::sigmoid_inplace(static_cast<float *>(beta_), T * n_vh, nullptr) ||
            !omph::kernels::softplus_bias_inplace(static_cast<float *>(alpha_), dt_bias, T, n_vh,
                                                  nullptr) ||
            !omph::kernels::mul_row_inplace(static_cast<float *>(alpha_), ssm_a, T, n_vh,
                                            nullptr) ||
            !omph::kernels::conv1d_state(static_cast<const float *>(fused_), conv_w, conv_cur,
                                         static_cast<float *>(conv_out_), T, channels,
                                         h_.ssm_conv_k, nullptr) ||
            !omph::kernels::conv_state_update(static_cast<const float *>(fused_), conv_cur,
                                              conv_new, T, channels, h_.ssm_conv_k, nullptr) ||
            !omph::kernels::silu_inplace(static_cast<float *>(conv_out_), T * channels, nullptr) ||
            !omph::kernels::split_qkv(static_cast<const float *>(conv_out_),
                                      static_cast<float *>(q_), static_cast<float *>(k_),
                                      static_cast<float *>(v_), T, q_dims, k_dims, v_dims,
                                      nullptr) ||
            !omph::kernels::rms_norm(static_cast<const float *>(q_), nullptr,
                                     static_cast<float *>(q_), T * n_kh, s,
                                     (float) (h_.eps / (double) s), l2_scale, nullptr) ||
            !omph::kernels::rms_norm(static_cast<const float *>(k_), nullptr,
                                     static_cast<float *>(k_), T * n_kh, s,
                                     (float) (h_.eps / (double) s), l2_scale, nullptr)) {
            return fail("gdn preprocessing failed");
        }
        for (int64_t t = 0; t < T; ++t) {
            if (!omph::kernels::delta_decay(seq_state,
                                            static_cast<const float *>(alpha_) + t * n_vh, n_vh, s,
                                            nullptr) ||
                !omph::kernels::delta_sk(seq_state, static_cast<const float *>(k_) + t * k_dims,
                                         static_cast<float *>(sk_), n_vh, n_kh, s, nullptr) ||
                !omph::kernels::delta_d(static_cast<const float *>(v_) + t * v_dims,
                                        static_cast<const float *>(sk_),
                                        static_cast<const float *>(beta_) + t * n_vh,
                                        static_cast<float *>(dvec_), n_vh, s, nullptr) ||
                !omph::kernels::delta_update(seq_state, static_cast<const float *>(k_) + t * k_dims,
                                             static_cast<const float *>(dvec_), n_vh, n_kh, s,
                                             nullptr) ||
                !omph::kernels::delta_o(seq_state, static_cast<const float *>(q_) + t * q_dims,
                                        static_cast<float *>(o_) + t * v_dims, n_vh, n_kh, s,
                                        l2_scale, nullptr)) {
                return fail("delta rule failed");
            }
        }
        if (!omph::kernels::gated_norm(static_cast<const float *>(o_), ssm_norm,
                                       static_cast<const float *>(z_), static_cast<float *>(o_),
                                       T * n_vh, v_dim, (float) h_.eps, nullptr) ||
            !omph::kernels::cast_f32_to_f16(static_cast<const float *>(o_), ffn16_, T * v_dims,
                                            nullptr) ||
            !matmul(p + "ssm_out.weight", ffn16_, static_cast<float *>(blk_), ne, v_dims, T)) {
            return fail("gdn output failed");
        }
        conv_flip_[il] ^= 1;
        return true;
    }

    // One matmul: the fused GEMV for single-token steps when it is available for
    // this tensor, otherwise the f16 dequant + hipBLASLt path.
    bool matmul(const std::string & name, const void * x16, float * y, const int64_t n_out,
                const int64_t k, const int64_t T) {
        if (use_gemv_ && T == 1) {
            const auto it = gems_.find(name);
            if (it != gems_.end() && it->second.rows == n_out && it->second.k == k) {
                const void * w = static_cast<const uint8_t *>(dev_weights_) + it->second.off;
                switch (it->second.type) {
                    case 10: return omph::kernels::gemv_q2k(w, x16, y, n_out, k, nullptr);
                    case 12: return omph::kernels::gemv_q4k(w, x16, y, n_out, k, nullptr);
                    case 14: return omph::kernels::gemv_q6k(w, x16, y, n_out, k, nullptr);
                    case 16: return omph::kernels::gemv_iq2_xxs(w, x16, y, n_out, k, nullptr);
                    case 17: return omph::kernels::gemv_iq2_xs(w, x16, y, n_out, k, nullptr);
                    case 18: return omph::kernels::gemv_iq3_xxs(w, x16, y, n_out, k, nullptr);
                    case 22: return omph::kernels::gemv_iq2_s(w, x16, y, n_out, k, nullptr);
                    case 21: return omph::kernels::gemv_iq3_s(w, x16, y, n_out, k, nullptr);
                    case 23: return omph::kernels::gemv_iq4_xs(w, x16, y, n_out, k, nullptr);
                    default: break;
                }
            }
        }
        void * w = stage_w(name);
        omph::runtime::Linear linear;
        return linear.run(w, x16, y, n_out, k, T);
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

    void * stage_w(const std::string & name) {
        const omph::gguf::TensorInfo * t = file_.tensor(name);
        if (t == nullptr) {
            throw std::runtime_error("missing tensor " + name);
        }
        const int64_t n = numel(*t);
        void * dst = scratch_.alloc((size_t) n * 2);
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
    };

    omph::gguf::File file_;
    HParams h_;
    bool use_gemv_ = false;
    std::unordered_map<std::string, size_t> off_;
    std::unordered_map<std::string, GemvEntry> gems_;
    Scratch scratch_;
    std::vector<void *> states_;
    std::vector<int64_t> kv_index_;
    std::vector<char> conv_flip_;
    int64_t max_seq_ = 0;
    void * kv_k_ = nullptr;
    void * kv_v_ = nullptr;
    void * dev_weights_ = nullptr;
    void * x_ = nullptr;
    void * cur_ = nullptr;
    void * resid_ = nullptr;
    void * blk_ = nullptr;
    void * h16_ = nullptr;
    void * fused_ = nullptr;
    void * conv_out_ = nullptr;
    void * z_ = nullptr;
    void * o_ = nullptr;
    void * q_ = nullptr;
    void * gate_ = nullptr;
    void * attn_ = nullptr;
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
};

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tokens.txt> <out-logits.f32> "
                             "[--trace-dir DIR] [--tokens N] [--generate N --gen-out FILE]\n",
                     argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    const std::string tok_path = argv[2];
    const std::string logits_path = argv[3];
    std::string trace_dir;
    std::string gen_path;
    int64_t max_tokens = 0;
    int64_t generate = 0;
    bool use_gemv = false;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace-dir") == 0 && i + 1 < argc) {
            trace_dir = argv[++i];
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
        Runner runner(model, (int64_t) toks.size() + generate + 8, use_gemv);
        const HParams & h = runner.hparams();
        std::vector<float> logits;
        if (!runner.forward(toks, logits, trace_dir, 0)) {
            return 1;
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
            int32_t next = argmax(logits.data() + (toks.size() - 1) * (size_t) h.n_vocab);
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
        std::printf("logits: %lld x %lld -> %s\ngreedy:", (long long) toks.size(),
                    (long long) h.n_vocab, logits_path.c_str());
        for (size_t t = 0; t < toks.size(); ++t) {
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
