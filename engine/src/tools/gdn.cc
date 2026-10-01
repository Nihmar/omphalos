// omph-gdn — run one gated-delta-net layer through the naive GPU path.
//
// usage: omph-gdn <model.gguf> <layer> <in.f32> <out-prefix> <tokens> [--trace]
//   in.f32: tokens x n_embd, row-major float32 (the layer's attn_norm input)
#include "format/gguf.hh"
#include "kernels/dequant.hh"
#include "kernels/elementwise.hh"
#include "kernels/gdn.hh"
#include "runtime/matmul.hh"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    (void) hipMemcpy(host.data(), dev, n * 4, hipMemcpyDeviceToHost);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <layer> <in.f32> <out-prefix> <tokens> [--trace]\n",
                     argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const int layer = std::atoi(argv[2]);
    const char * in_path = argv[3];
    const std::string prefix = argv[4];
    const int64_t tokens = std::atoll(argv[5]);
    const bool trace = argc > 6 && std::strcmp(argv[6], "--trace") == 0;

    try {
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
        if (n_kh <= 0 || n_vh <= 0 || s <= 0 || inner <= 0 || conv_k <= 1 || n_embd <= 0) {
            return fail("missing hyperparameters");
        }
        const int64_t v_dim = inner / n_vh;
        const int64_t q_dims = n_kh * s;
        const int64_t k_dims = n_kh * s;
        const int64_t v_dims = n_vh * v_dim;
        const int64_t channels = q_dims + k_dims + v_dims;
        const float l2_scale = 1.0f / std::sqrt((float) s);

        const std::string p = "blk." + std::to_string(layer) + ".";
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
                hipMalloc(dev, (size_t) elems * 2) != hipSuccess) {
                return false;
            }
            if (hipMemcpy(q, file.tensor_data(*t), t->nbytes, hipMemcpyHostToDevice) != hipSuccess) {
                return false;
            }
            const bool ok = omph::kernels::dequantize(t->type, q, *dev, elems, true, nullptr);
            (void) hipFree(q);
            return ok;
        };

        void * wqkv = nullptr;
        void * wz = nullptr;
        void * wbeta = nullptr;
        void * walpha = nullptr;
        void * wout = nullptr;
        if (!dequant_to("attn_qkv.weight", &wqkv) || !dequant_to("attn_gate.weight", &wz) ||
            !dequant_to("ssm_beta.weight", &wbeta) || !dequant_to("ssm_alpha.weight", &walpha) ||
            !dequant_to("ssm_out.weight", &wout)) {
            return fail("failed to dequantize weights");
        }

        const auto vec_host = [&](const char * name) {
            const omph::gguf::TensorInfo * t = file.tensor(p + name);
            const float * src = reinterpret_cast<const float *>(file.tensor_data(*t));
            return std::vector<float>(src, src + t->ne[0]);
        };
        const std::vector<float> dt_bias = vec_host("ssm_dt.bias");
        const std::vector<float> ssm_a = vec_host("ssm_a");
        const std::vector<float> ssm_norm_w = vec_host("ssm_norm.weight");
        // conv1d.weight is (kernel, channels) — flat layout w[c*kernel + j]
        const omph::gguf::TensorInfo * cw = file.tensor(p + "ssm_conv1d.weight");
        int64_t cw_n = 1;
        for (const uint64_t d : cw->ne) {
            cw_n *= (int64_t) d;
        }
        const float * cw_src = reinterpret_cast<const float *>(file.tensor_data(*cw));
        const std::vector<float> conv_w(cw_src, cw_src + cw_n);

        void * dev_dt = nullptr;
        void * dev_a = nullptr;
        void * dev_norm = nullptr;
        void * dev_convw = nullptr;
        (void) hipMalloc(&dev_dt, dt_bias.size() * 4);
        (void) hipMalloc(&dev_a, ssm_a.size() * 4);
        (void) hipMalloc(&dev_norm, ssm_norm_w.size() * 4);
        (void) hipMalloc(&dev_convw, conv_w.size() * 4);
        (void) hipMemcpy(dev_dt, dt_bias.data(), dt_bias.size() * 4, hipMemcpyHostToDevice);
        (void) hipMemcpy(dev_a, ssm_a.data(), ssm_a.size() * 4, hipMemcpyHostToDevice);
        (void) hipMemcpy(dev_norm, ssm_norm_w.data(), ssm_norm_w.size() * 4, hipMemcpyHostToDevice);
        (void) hipMemcpy(dev_convw, conv_w.data(), conv_w.size() * 4, hipMemcpyHostToDevice);

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
        (void) hipMalloc(&ex32, host_in.size() * 4);
        (void) hipMalloc(&ex16, host_in.size() * 2);
        (void) hipMemcpy(ex32, host_in.data(), host_in.size() * 4, hipMemcpyHostToDevice);
        (void) omph::kernels::cast_f32_to_f16((const float *) ex32, ex16, (int64_t) host_in.size(),
                                              nullptr);

        omph::runtime::Linear linear;
        void * qkv = nullptr;
        void * z = nullptr;
        void * beta = nullptr;
        void * alpha = nullptr;
        void * conv_raw = nullptr;
        void * q = nullptr;
        void * k = nullptr;
        void * v = nullptr;
        void * sk = nullptr;
        void * dvec = nullptr;
        void * o = nullptr;
        void * final = nullptr;
        void * out = nullptr;
        void * conv_state = nullptr;
        void * state = nullptr;
        void * final16 = nullptr;
        (void) hipMalloc(&qkv, (size_t) tokens * channels * 4);
        (void) hipMalloc(&z, (size_t) tokens * v_dims * 4);
        (void) hipMalloc(&beta, (size_t) tokens * n_vh * 4);
        (void) hipMalloc(&alpha, (size_t) tokens * n_vh * 4);
        (void) hipMalloc(&conv_raw, (size_t) tokens * channels * 4);
        (void) hipMalloc(&q, (size_t) tokens * q_dims * 4);
        (void) hipMalloc(&k, (size_t) tokens * k_dims * 4);
        (void) hipMalloc(&v, (size_t) tokens * v_dims * 4);
        (void) hipMalloc(&sk, (size_t) n_vh * s * 4);
        (void) hipMalloc(&dvec, (size_t) n_vh * s * 4);
        (void) hipMalloc(&o, (size_t) tokens * v_dims * 4);
        (void) hipMalloc(&final, (size_t) tokens * v_dims * 4);
        (void) hipMalloc(&out, (size_t) tokens * n_embd * 4);
        (void) hipMalloc(&conv_state, (size_t)(conv_k - 1) * channels * 4);
        (void) hipMalloc(&state, (size_t) n_vh * s * s * 4);
        (void) hipMalloc(&final16, (size_t) tokens * v_dims * 2);
        (void) hipMemset(conv_state, 0, (size_t)(conv_k - 1) * channels * 4);
        (void) hipMemset(state, 0, (size_t) n_vh * s * s * 4);

        if (!linear.run(wqkv, ex16, (float *) qkv, channels, n_embd, tokens) ||
            !linear.run(wz, ex16, (float *) z, v_dims, n_embd, tokens) ||
            !linear.run(wbeta, ex16, (float *) beta, n_vh, n_embd, tokens) ||
            !linear.run(walpha, ex16, (float *) alpha, n_vh, n_embd, tokens)) {
            return fail("projection failed");
        }
        if (!omph::kernels::sigmoid_inplace((float *) beta, tokens * n_vh, nullptr)) {
            return fail("sigmoid failed");
        }
        if (trace) {
            std::vector<float> t1;
            read_back(t1, qkv, (size_t) tokens * channels);
            write_f32(prefix + "-qkv.f32", t1);
            read_back(t1, z, (size_t) tokens * v_dims);
            write_f32(prefix + "-z.f32", t1);
            read_back(t1, beta, (size_t) tokens * n_vh);
            write_f32(prefix + "-beta.f32", t1);
            read_back(t1, alpha, (size_t) tokens * n_vh);
            write_f32(prefix + "-alpha.f32", t1);
        }
        if (!omph::kernels::softplus_bias_inplace((float *) alpha, (const float *) dev_dt, tokens,
                                                  n_vh, nullptr)) {
            return fail("softplus failed");
        }
        if (trace) {
            std::vector<float> t1;
            read_back(t1, alpha, (size_t) tokens * n_vh);
            write_f32(prefix + "-a_softplus.f32", t1);
        }
        if (!omph::kernels::mul_row_inplace((float *) alpha, (const float *) dev_a, tokens, n_vh,
                                            nullptr)) {
            return fail("gate mul failed");
        }
        if (trace) {
            std::vector<float> t1;
            read_back(t1, alpha, (size_t) tokens * n_vh);
            write_f32(prefix + "-gate.f32", t1);
        }

        // conv1d + silu, then split q/k/v
        if (!omph::kernels::conv1d_state((const float *) qkv, (const float *) dev_convw,
                                         (const float *) conv_state, (float *) conv_raw, tokens,
                                         channels, conv_k, nullptr)) {
            return fail("conv failed");
        }
        if (trace) {
            std::vector<float> t1;
            read_back(t1, conv_raw, (size_t) tokens * channels);
            write_f32(prefix + "-conv_raw.f32", t1);
        }
        if (!omph::kernels::silu_inplace((float *) conv_raw, tokens * channels, nullptr)) {
            return fail("silu failed");
        }
        if (trace) {
            std::vector<float> t1;
            read_back(t1, conv_raw, (size_t) tokens * channels);
            write_f32(prefix + "-conv_silu.f32", t1);
        }
        if (!omph::kernels::split_qkv((const float *) conv_raw, (float *) q, (float *) k,
                                      (float *) v, tokens, q_dims, k_dims, v_dims, nullptr)) {
            return fail("split failed");
        }
        // L2 normalization of q and k (per head): rms_norm(x, eps/n) * 1/sqrt(n)
        if (!omph::kernels::rms_norm((const float *) q, nullptr, (float *) q, tokens * n_kh, s,
                                     (float) (eps / (double) s), l2_scale, nullptr) ||
            !omph::kernels::rms_norm((const float *) k, nullptr, (float *) k, tokens * n_kh, s,
                                     (float) (eps / (double) s), l2_scale, nullptr)) {
            return fail("l2 norm failed");
        }
        if (trace) {
            std::vector<float> t1;
            read_back(t1, q, (size_t) tokens * q_dims);
            write_f32(prefix + "-q_l2.f32", t1);
            read_back(t1, k, (size_t) tokens * k_dims);
            write_f32(prefix + "-k_l2.f32", t1);
            read_back(t1, v, (size_t) tokens * v_dims);
            write_f32(prefix + "-v.f32", t1);
        }

        // delta rule, one token at a time
        for (int64_t t = 0; t < tokens; ++t) {
            if (!omph::kernels::delta_decay((float *) state, (const float *) alpha + t * n_vh, n_vh,
                                            s, nullptr) ||
                !omph::kernels::delta_sk((const float *) state, (const float *) k + t * k_dims, (float *) sk,
                                         n_vh, n_kh, s, nullptr) ||
                !omph::kernels::delta_d((const float *) v + t * v_dims, (const float *) sk,
                                        (const float *) beta + t * n_vh, (float *) dvec, n_vh, s,
                                        nullptr) ||
                !omph::kernels::delta_update((float *) state, (const float *) k + t * k_dims,
                                             (const float *) dvec, n_vh, n_kh, s, nullptr) ||
                !omph::kernels::delta_o((const float *) state, (const float *) q + t * q_dims,
                                        (float *) o + t * v_dims, n_vh, n_kh, s, l2_scale,
                                        nullptr)) {
                return fail("delta rule failed");
            }
        }
        if (trace) {
            std::vector<float> t1;
            read_back(t1, o, (size_t) tokens * v_dims);
            write_f32(prefix + "-attn_output.f32", t1);
        }

        if (!omph::kernels::gated_norm((const float *) o, (const float *) dev_norm, (const float *) z,
                                       (float *) final, tokens * n_vh, v_dim, (float) eps,
                                       nullptr)) {
            return fail("gated norm failed");
        }
        if (trace) {
            std::vector<float> t1;
            read_back(t1, final, (size_t) tokens * v_dims);
            write_f32(prefix + "-final.f32", t1);
        }
        if (!omph::kernels::cast_f32_to_f16((const float *) final, final16, tokens * v_dims,
                                            nullptr) ||
            !linear.run(wout, final16, (float *) out, n_embd, v_dims, tokens)) {
            return fail("output projection failed");
        }
        if (hipDeviceSynchronize() != hipSuccess) {
            return fail("kernel failed");
        }

        std::vector<float> host_out;
        read_back(host_out, out, (size_t) tokens * n_embd);
        write_f32(prefix + ".out.f32", host_out);
        std::printf("gdn layer %d: %lld tokens -> %s.out.f32\n", layer, (long long) tokens,
                    prefix.c_str());
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
    return 0;
}
