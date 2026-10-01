// omph-attn — run one full-attention layer through the kernels the engine runs
// (attn_prep + attention_gqa, #99).
//
// usage: omph-attn <model.gguf> <layer> <in.f32> <out-prefix> <tokens> [--trace]
//                  [--kv f32|q8q4|q4q4] [--window N] [--chunk N]
//   in.f32: tokens x n_embd, row-major float32 (the layer's attn_norm input)
//   --kv: the cache, as in omph-run (f32 = OMPH_KV_F32, q8q4 = default,
//         q4q4 = OMPH_KV_K4); --window: the FP16 window of the quantized cache
//         (128 as in omph-run; 0 makes every key go through the quantized
//         blocks); --chunk: tokens per call (1 = the decode path, with the
//         split-K attention), all of them by default.
//   writes <out-prefix>.out.f32 and, with --trace, the intermediate tensors.
#include "format/gguf.hh"
#include "kernels/attn.hh"
#include "kernels/dequant.hh"
#include "kernels/elementwise.hh"
#include "runtime/matmul.hh"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int fail(const char * msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

bool meta_int(const omph::gguf::File & f, const char * key, int64_t & out) {
    const omph::gguf::Value * v = f.find(key);
    uint64_t u = 0;
    if (v == nullptr || !v->as_u64(u)) {
        return false;
    }
    out = (int64_t) u;
    return true;
}

double meta_float(const omph::gguf::File & f, const char * key, double fallback) {
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

void write_f32(const std::string & path, const std::vector<float> & data) {
    FILE * f = std::fopen(path.c_str(), "wb");
    const bool ok = f != nullptr && std::fwrite(data.data(), 4, data.size(), f) == data.size();
    if (f == nullptr || std::fclose(f) != 0 || !ok) {
        throw std::runtime_error("cannot write " + path);
    }
}

void read_back(std::vector<float> & host, const void * dev, const size_t n) {
    host.resize(n);
    if (hipMemcpy(host.data(), dev, n * 4, hipMemcpyDeviceToHost) != hipSuccess) {
        throw std::runtime_error("read-back failed");
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <layer> <in.f32> <out-prefix> <tokens> [--trace] "
                     "[--kv f32|q8q4|q4q4] [--window N] [--chunk N]\n",
                     argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const int layer = std::atoi(argv[2]);
    const char * in_path = argv[3];
    const std::string prefix = argv[4];
    const int64_t tokens = std::atoll(argv[5]);
    bool trace = false;
    std::string kv_mode = "f32";
    int64_t window = 128;
    int64_t chunk = 0;
    for (int i = 6; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace") == 0) {
            trace = true;
        } else if (std::strcmp(argv[i], "--kv") == 0 && i + 1 < argc) {
            kv_mode = argv[++i];
        } else if (std::strcmp(argv[i], "--window") == 0 && i + 1 < argc) {
            window = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            chunk = std::atoll(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }
    if (kv_mode != "f32" && kv_mode != "q8q4" && kv_mode != "q4q4") {
        return fail("--kv must be f32, q8q4 or q4q4");
    }
    if (tokens <= 0 || window < 0 || chunk < 0) {
        return fail("bad token count, window or chunk");
    }
    if (chunk == 0 || chunk > tokens) {
        chunk = tokens;
    }
    const bool quant = kv_mode != "f32";
    const bool k_q4 = kv_mode == "q4q4";

    try {
        if (omph::kernels::kernel_wave_size() != 32) {
            return fail("kernels not built for wave32");
        }
        omph::gguf::File file(model_path);
        int64_t n_head = 0;
        int64_t n_head_kv = 0;
        int64_t head_dim = 0;
        int64_t n_rot = 0;
        int64_t n_embd = 0;
        meta_int(file, "qwen35.attention.head_count", n_head);
        meta_int(file, "qwen35.attention.head_count_kv", n_head_kv);
        meta_int(file, "qwen35.attention.key_length", head_dim);
        meta_int(file, "qwen35.rope.dimension_count", n_rot);
        meta_int(file, "qwen35.embedding_length", n_embd);
        const double freq_base = meta_float(file, "qwen35.rope.freq_base", 10000.0);
        const double eps = meta_float(file, "qwen35.attention.layer_norm_rms_epsilon", 1e-6);
        if (n_head <= 0 || n_head_kv <= 0 || head_dim != 256 || n_embd <= 0 || n_rot <= 0) {
            return fail("missing or unsupported hyperparameters");
        }
        const int64_t q_out = n_head * 2 * head_dim;
        const int64_t kv_out = n_head_kv * head_dim;
        const int64_t att = n_head * head_dim;

        const std::string p = "blk." + std::to_string(layer) + ".";
        if (file.tensor(p + "attn_q_norm.weight") == nullptr) {
            return fail("not an attention layer");
        }
        bool ok = true;
        const auto dev_alloc = [&](void ** dst, const size_t bytes) {
            ok = ok && hipMalloc(dst, bytes) == hipSuccess && hipMemset(*dst, 0, bytes) == hipSuccess;
        };
        const auto dequant_to = [&](const char * name, void ** dev) -> bool {
            const omph::gguf::TensorInfo * t = file.tensor(p + name);
            if (t == nullptr) {
                return false;
            }
            int64_t elems = 1;
            for (const uint64_t d : t->ne) {
                elems *= (int64_t) d;
            }
            void * q = nullptr;
            if (hipMalloc(&q, t->nbytes) != hipSuccess ||
                hipMalloc(dev, (size_t) elems * 2) != hipSuccess ||
                hipMemcpy(q, file.tensor_data(*t), t->nbytes, hipMemcpyHostToDevice) != hipSuccess) {
                return false;
            }
            const bool done = omph::kernels::dequantize(t->type, q, *dev, elems, true, nullptr);
            (void) hipFree(q);
            return done;
        };
        const auto upload_f32 = [&](const char * name, void ** dev) -> bool {
            const omph::gguf::TensorInfo * t = file.tensor(p + name);
            if (t == nullptr || t->type != 0) {
                return false;
            }
            return hipMalloc(dev, t->nbytes) == hipSuccess &&
                   hipMemcpy(*dev, file.tensor_data(*t), t->nbytes, hipMemcpyHostToDevice) ==
                       hipSuccess;
        };

        void * wq = nullptr;
        void * wk = nullptr;
        void * wv = nullptr;
        void * wo = nullptr;
        void * qn = nullptr;
        void * kn = nullptr;
        if (!dequant_to("attn_q.weight", &wq) || !dequant_to("attn_k.weight", &wk) ||
            !dequant_to("attn_v.weight", &wv) || !dequant_to("attn_output.weight", &wo) ||
            !upload_f32("attn_q_norm.weight", &qn) || !upload_f32("attn_k_norm.weight", &kn)) {
            return fail("failed to load the layer's weights");
        }

        // input
        std::vector<float> host_in((size_t) tokens * n_embd);
        if (FILE * fi = std::fopen(in_path, "rb")) {
            const size_t got = std::fread(host_in.data(), 4, host_in.size(), fi);
            std::fclose(fi);
            if (got != host_in.size()) {
                return fail("short read of the input file");
            }
        } else {
            return fail("cannot open the input file");
        }
        void * ex32 = nullptr;
        void * ex16 = nullptr;
        dev_alloc(&ex32, host_in.size() * 4);
        dev_alloc(&ex16, host_in.size() * 2);
        void * qf = nullptr;    // (T, nh * 2 * hd): q | gate per head
        void * kf = nullptr;    // (T, nkv * hd); normed + roped in place = the f32 cache
        void * vf = nullptr;
        void * q = nullptr;     // (T, nh, hd)
        void * gate = nullptr;
        void * attn16 = nullptr;  // (T, nh, hd) f16, as the engine writes it
        void * out = nullptr;   // (T, n_embd)
        dev_alloc(&qf, (size_t) tokens * q_out * 4);
        dev_alloc(&kf, (size_t) tokens * kv_out * 4);
        dev_alloc(&vf, (size_t) tokens * kv_out * 4);
        dev_alloc(&q, (size_t) tokens * att * 4);
        dev_alloc(&gate, (size_t) tokens * att * 4);
        dev_alloc(&attn16, (size_t) tokens * att * 2);
        dev_alloc(&out, (size_t) tokens * n_embd * 4);
        // quantized cache, laid out as omph-run's per-layer slice
        const int64_t nblk = head_dim / 32;
        void * kq = nullptr;
        void * ks = nullptr;
        void * vq = nullptr;
        void * vs = nullptr;
        void * k16 = nullptr;
        void * v16 = nullptr;
        if (quant) {
            dev_alloc(&kq, (size_t) tokens * kv_out / (k_q4 ? 2 : 1));
            dev_alloc(&ks, (size_t) tokens * n_head_kv * nblk * 2);
            dev_alloc(&vq, (size_t) tokens * kv_out / 2);
            dev_alloc(&vs, (size_t) tokens * n_head_kv * nblk * 2);
            if (window > 0) {
                dev_alloc(&k16, (size_t) window * kv_out * 2);
                dev_alloc(&v16, (size_t) window * kv_out * 2);
            }
        }
        const size_t work_bytes =
            omph::kernels::attention_gqa_work_bytes(chunk, n_head, n_head_kv, head_dim);
        void * work = nullptr;
        dev_alloc(&work, work_bytes);
        if (!ok) {
            return fail("out of VRAM");
        }
        if (hipMemcpy(ex32, host_in.data(), host_in.size() * 4, hipMemcpyHostToDevice) !=
                hipSuccess ||
            !omph::kernels::cast_f32_to_f16((const float *) ex32, ex16, (int64_t) host_in.size(),
                                            nullptr)) {
            return fail("input upload failed");
        }

        // projections (the f16 GEMM, all tokens: not what this tool checks)
        omph::runtime::Linear linear;
        if (!linear.run(wq, ex16, (float *) qf, q_out, n_embd, tokens) ||
            !linear.run(wk, ex16, (float *) kf, kv_out, n_embd, tokens) ||
            !linear.run(wv, ex16, (float *) vf, kv_out, n_embd, tokens)) {
            return fail("projection failed");
        }

        // attn_prep + attention_gqa, `chunk` tokens per call as omph-run does
        const float scale = 1.0f / std::sqrt((float) head_dim);
        for (int64_t t0 = 0; t0 < tokens; t0 += chunk) {
            const int64_t n = std::min(chunk, tokens - t0);
            omph::kernels::AttnPrep prep;
            prep.qg = static_cast<const float *>(qf) + t0 * q_out;
            prep.q = static_cast<float *>(q) + t0 * att;
            prep.gate = static_cast<float *>(gate) + t0 * att;
            prep.k = static_cast<float *>(kf) + t0 * kv_out;
            prep.v = static_cast<float *>(vf) + t0 * kv_out;
            prep.q_norm = static_cast<const float *>(qn);
            prep.k_norm = static_cast<const float *>(kn);
            prep.tokens = n;
            prep.pos0 = t0;
            prep.nh = n_head;
            prep.nkv = n_head_kv;
            prep.hd = head_dim;
            prep.n_rot = (int) n_rot;
            prep.freq_base = (float) freq_base;
            prep.eps = (float) eps;
            prep.rotate = quant;
            omph::kernels::KvCache kv;
            if (quant) {
                prep.k_q8 = static_cast<uint8_t *>(kq);
                prep.k_scales = static_cast<__half *>(ks);
                prep.v_q4 = static_cast<uint8_t *>(vq);
                prep.v_scales = static_cast<__half *>(vs);
                prep.k16 = static_cast<__half *>(k16);
                prep.v16 = static_cast<__half *>(v16);
                prep.window = window;
                prep.k_q4 = k_q4;
                kv.k_q8 = static_cast<const uint8_t *>(kq);
                kv.k_scales = ks;
                kv.v_q4 = static_cast<const uint8_t *>(vq);
                kv.v_scales = vs;
                kv.k16 = k16;
                kv.v16 = v16;
                kv.window = window;
                kv.k_q4 = k_q4;
            } else {
                kv.k_f32 = static_cast<const float *>(kf);
                kv.v_f32 = static_cast<const float *>(vf);
            }
            if (!omph::kernels::attn_prep(prep, nullptr) ||
                !omph::kernels::attention_gqa(
                    static_cast<const float *>(q) + t0 * att, kv,
                    static_cast<const float *>(gate) + t0 * att, nullptr, n, t0 + n, n_head,
                    n_head_kv, head_dim, scale, quant, work, work_bytes, nullptr,
                    static_cast<uint8_t *>(attn16) + t0 * att * 2)) {
                return fail("attention failed");
            }
        }
        if (!linear.run(wo, attn16, (float *) out, n_embd, att, tokens) ||
            hipDeviceSynchronize() != hipSuccess) {
            return fail("output projection failed");
        }

        std::vector<float> host_out;
        read_back(host_out, out, (size_t) tokens * n_embd);
        write_f32(prefix + ".out.f32", host_out);
        if (trace) {
            // q / k are post-RoPE (and Hadamard-rotated with a quantized cache)
            std::vector<float> t1;
            read_back(t1, qf, (size_t) tokens * q_out);
            write_f32(prefix + "-q_full.f32", t1);
            read_back(t1, q, (size_t) tokens * att);
            write_f32(prefix + "-q.f32", t1);
            read_back(t1, kf, (size_t) tokens * kv_out);
            write_f32(prefix + "-k.f32", t1);
            read_back(t1, gate, (size_t) tokens * att);
            write_f32(prefix + "-gate.f32", t1);
            std::vector<__half> h((size_t) tokens * att);
            if (hipMemcpy(h.data(), attn16, h.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) {
                return fail("read-back failed");
            }
            t1.resize(h.size());
            for (size_t i = 0; i < h.size(); ++i) {
                t1[i] = __half2float(h[i]);
            }
            write_f32(prefix + "-attn_gated.f32", t1);
        }
        std::printf("attn layer %d (%s, window %lld, chunk %lld): %lld tokens -> %s.out.f32\n",
                    layer, kv_mode.c_str(), (long long) (quant ? window : 0), (long long) chunk,
                    (long long) tokens, prefix.c_str());
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
    return 0;
}
