// omph-gdn — run one gated-delta-net layer through the kernel the engine runs
// (gdn_step, #99).
//
// usage: omph-gdn <model.gguf> <layer> <in.f32> <out-prefix> <tokens> [--trace]
//                 [--chunk N]
//   in.f32: tokens x n_embd, row-major float32 (the layer's attn_norm input)
//   --chunk: tokens per call, the conv state carried across calls as omph-run
//            does (1 = the decode path); all of them by default.
//   writes <out-prefix>.out.f32 and, with --trace, the projections and the
//   gated-norm output.
#include "format/gguf.hh"
#include "kernels/dequant.hh"
#include "kernels/elementwise.hh"
#include "kernels/gdn.hh"
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
                     "[--chunk N]\n",
                     argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const int layer = std::atoi(argv[2]);
    const char * in_path = argv[3];
    const std::string prefix = argv[4];
    const int64_t tokens = std::atoll(argv[5]);
    bool trace = false;
    int64_t chunk = 0;
    for (int i = 6; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace") == 0) {
            trace = true;
        } else if (std::strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            chunk = std::atoll(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }
    if (tokens <= 0 || chunk < 0) {
        return fail("bad token count or chunk");
    }
    if (chunk == 0 || chunk > tokens) {
        chunk = tokens;
    }

    try {
        if (omph::kernels::kernel_wave_size() != 32) {
            return fail("kernels not built for wave32");
        }
        omph::gguf::File file(model_path);
        int64_t n_kh = 0;
        int64_t n_vh = 0;
        int64_t s = 0;
        int64_t inner = 0;
        int64_t conv_k = 0;
        int64_t n_embd = 0;
        meta_int(file, "qwen35.ssm.group_count", n_kh);
        meta_int(file, "qwen35.ssm.time_step_rank", n_vh);
        meta_int(file, "qwen35.ssm.state_size", s);
        meta_int(file, "qwen35.ssm.inner_size", inner);
        meta_int(file, "qwen35.ssm.conv_kernel", conv_k);
        meta_int(file, "qwen35.embedding_length", n_embd);
        const double eps = meta_float(file, "qwen35.attention.layer_norm_rms_epsilon", 1e-6);
        if (n_kh <= 0 || n_vh <= 0 || s != 128 || inner <= 0 || conv_k <= 1 || n_embd <= 0) {
            return fail("missing or unsupported hyperparameters");
        }
        const int64_t v_dim = inner / n_vh;
        const int64_t q_dims = n_kh * s;
        const int64_t k_dims = n_kh * s;
        const int64_t v_dims = n_vh * v_dim;
        const int64_t channels = q_dims + k_dims + v_dims;

        const std::string p = "blk." + std::to_string(layer) + ".";
        if (file.tensor(p + "ssm_a") == nullptr) {
            return fail("not a delta-net layer");
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
        // Raw bytes: the f32 vectors, and the BF16 beta / alpha rows gdn_step
        // dots itself.
        const auto upload = [&](const char * name, const uint32_t type, void ** dev) -> bool {
            const omph::gguf::TensorInfo * t = file.tensor(p + name);
            if (t == nullptr || t->type != type) {
                return false;
            }
            return hipMalloc(dev, t->nbytes) == hipSuccess &&
                   hipMemcpy(*dev, file.tensor_data(*t), t->nbytes, hipMemcpyHostToDevice) ==
                       hipSuccess;
        };

        void * wqkv = nullptr;
        void * wz = nullptr;
        void * wout = nullptr;
        void * wbeta = nullptr;
        void * walpha = nullptr;
        void * dt = nullptr;
        void * ssm_a = nullptr;
        void * norm_w = nullptr;
        void * conv_w = nullptr;
        if (!dequant_to("attn_qkv.weight", &wqkv) || !dequant_to("attn_gate.weight", &wz) ||
            !dequant_to("ssm_out.weight", &wout) || !upload("ssm_beta.weight", 30, &wbeta) ||
            !upload("ssm_alpha.weight", 30, &walpha) || !upload("ssm_dt.bias", 0, &dt) ||
            !upload("ssm_a", 0, &ssm_a) || !upload("ssm_norm.weight", 0, &norm_w) ||
            !upload("ssm_conv1d.weight", 0, &conv_w)) {
            return fail("failed to load the layer's weights (beta / alpha must be BF16)");
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
        void * qkv = nullptr;
        void * z = nullptr;
        void * final16 = nullptr;
        void * out = nullptr;
        void * conv_a = nullptr;
        void * conv_b = nullptr;
        void * state = nullptr;
        dev_alloc(&ex32, host_in.size() * 4);
        dev_alloc(&ex16, host_in.size() * 2);
        dev_alloc(&qkv, (size_t) tokens * channels * 4);
        dev_alloc(&z, (size_t) tokens * v_dims * 4);
        dev_alloc(&final16, (size_t) tokens * v_dims * 2);
        dev_alloc(&out, (size_t) tokens * n_embd * 4);
        dev_alloc(&conv_a, (size_t) (conv_k - 1) * channels * 4);
        dev_alloc(&conv_b, (size_t) (conv_k - 1) * channels * 4);
        dev_alloc(&state, (size_t) n_vh * s * s * 2);  // f16 (#273)
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
        if (!linear.run(wqkv, ex16, (float *) qkv, channels, n_embd, tokens) ||
            !linear.run(wz, ex16, (float *) z, v_dims, n_embd, tokens)) {
            return fail("projection failed");
        }

        // gdn_step, one launch per token, `chunk` tokens per call: the conv
        // state flips between its two buffers after each call, as in omph-run
        omph::kernels::GdnStep step;
        step.state = static_cast<__half *>(state);
        step.conv_w = static_cast<const float *>(conv_w);
        step.w_beta = static_cast<const uint16_t *>(wbeta);
        step.w_alpha = static_cast<const uint16_t *>(walpha);
        step.k_in = n_embd;
        step.dt_bias = static_cast<const float *>(dt);
        step.ssm_a = static_cast<const float *>(ssm_a);
        step.norm_w = static_cast<const float *>(norm_w);
        step.channels = channels;
        step.q_dims = q_dims;
        step.kv_dims = k_dims;
        step.conv_k = conv_k;
        step.n_kh = n_kh;
        step.eps_l2 = (float) (eps / (double) s);
        step.l2_scale = 1.0f / std::sqrt((float) s);
        step.eps_norm = (float) eps;
        bool flip = false;
        for (int64_t t0 = 0; t0 < tokens; t0 += chunk) {
            const int64_t n = std::min(chunk, tokens - t0);
            step.qkv = static_cast<const float *>(qkv) + t0 * channels;
            step.conv_cur = static_cast<const float *>(flip ? conv_b : conv_a);
            step.conv_new = static_cast<float *>(flip ? conv_a : conv_b);
            step.tokens = n;
            if (n > 1) {  // what the engine runs for several tokens (#96)
                step.x16 = static_cast<const __half *>(ex16) + t0 * n_embd;
                step.z = static_cast<const float *>(z) + t0 * v_dims;
                step.out16 = static_cast<__half *>(final16) + t0 * v_dims;
                if (!omph::kernels::gdn_chunk(step, n_vh, nullptr)) {
                    return fail("gdn_chunk failed");
                }
                flip = !flip;
                continue;
            }
            for (int64_t t = 0; t < n; ++t) {
                step.t = t;
                step.x16 = static_cast<const __half *>(ex16) + (t0 + t) * n_embd;
                step.z = static_cast<const float *>(z) + (t0 + t) * v_dims;
                step.out16 = static_cast<__half *>(final16) + (t0 + t) * v_dims;
                if (!omph::kernels::gdn_step(step, n_vh, nullptr)) {
                    return fail("gdn_step failed");
                }
            }
            flip = !flip;
        }
        if (!linear.run(wout, final16, (float *) out, n_embd, v_dims, tokens) ||
            hipDeviceSynchronize() != hipSuccess) {
            return fail("output projection failed");
        }

        std::vector<float> host_out;
        read_back(host_out, out, (size_t) tokens * n_embd);
        write_f32(prefix + ".out.f32", host_out);
        if (trace) {
            std::vector<float> t1;
            read_back(t1, qkv, (size_t) tokens * channels);
            write_f32(prefix + "-qkv.f32", t1);
            read_back(t1, z, (size_t) tokens * v_dims);
            write_f32(prefix + "-z.f32", t1);
            std::vector<__half> h((size_t) tokens * v_dims);
            if (hipMemcpy(h.data(), final16, h.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) {
                return fail("read-back failed");
            }
            t1.resize(h.size());
            for (size_t i = 0; i < h.size(); ++i) {
                t1[i] = __half2float(h[i]);
            }
            write_f32(prefix + "-final.f32", t1);
        }
        std::printf("gdn layer %d (chunk %lld): %lld tokens -> %s.out.f32\n", layer,
                    (long long) chunk, (long long) tokens, prefix.c_str());
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
    return 0;
}
