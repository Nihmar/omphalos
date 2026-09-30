// omph-attn — run one full-attention layer through the naive GPU path.
//
// usage: omph-attn <model.gguf> <layer> <in.f32> <out-prefix> <tokens> [--trace]
//   in.f32: tokens x n_embd, row-major float32 (the layer's attn_norm input)
//   writes <out-prefix>.out.f32 and, with --trace, the intermediate tensors.
#include "format/gguf.hh"
#include "kernels/attn.hh"
#include "kernels/dequant.hh"
#include "kernels/elementwise.hh"
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
    if (FILE * f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(data.data(), 4, data.size(), f);
        std::fclose(f);
    }
}

void read_back(std::vector<float> & host, void * dev, const size_t n) {
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
        int64_t n_head = 0;
        int64_t n_head_kv = 0;
        int64_t head_dim = 0;
        int64_t n_rot = 0;
        meta_int(file, "qwen35.attention.head_count", n_head);
        meta_int(file, "qwen35.attention.head_count_kv", n_head_kv);
        meta_int(file, "qwen35.attention.key_length", head_dim);
        meta_int(file, "qwen35.rope.dimension_count", n_rot);
        const double freq_base = meta_float(file, "qwen35.rope.freq_base", 10000.0);
        const double eps = meta_float(file, "qwen35.attention.layer_norm_rms_epsilon", 1e-6);
        int64_t n_embd = 0;
        meta_int(file, "qwen35.embedding_length", n_embd);
        if (n_head <= 0 || n_head_kv <= 0 || head_dim <= 0 || n_embd <= 0) {
            return fail("missing hyperparameters");
        }
        const int64_t q_out = n_head * 2 * head_dim;
        const int64_t kv_out = n_head_kv * head_dim;

        const std::string p = "blk." + std::to_string(layer) + ".";
        const auto dequant_to = [&](const char * name, void ** dev, int64_t & elems) -> bool {
            const omph::gguf::TensorInfo * t = file.tensor(p + name);
            if (t == nullptr) {
                return false;
            }
            elems = 1;
            for (const uint64_t d : t->ne) {
                elems *= (int64_t) d;
            }
            void * q = nullptr;
            if (hipMalloc(&q, t->nbytes) != hipSuccess || hipMalloc(dev, (size_t) elems * 2) != hipSuccess) {
                return false;
            }
            if (hipMemcpy(q, file.tensor_data(*t), t->nbytes, hipMemcpyHostToDevice) != hipSuccess) {
                return false;
            }
            const bool ok = omph::kernels::dequantize(t->type, q, *dev, elems, true, nullptr);
            (void) hipFree(q);
            return ok;
        };

        void * wq = nullptr;
        void * wk = nullptr;
        void * wv = nullptr;
        void * wo = nullptr;
        int64_t n = 0;
        if (!dequant_to("attn_q.weight", &wq, n) || !dequant_to("attn_k.weight", &wk, n) ||
            !dequant_to("attn_v.weight", &wv, n) || !dequant_to("attn_output.weight", &wo, n)) {
            return fail("failed to dequantize weights");
        }

        // host-side norm weights (F32)
        const auto read_vec = [&](const char * name) {
            const omph::gguf::TensorInfo * t = file.tensor(p + name);
            const float * src = reinterpret_cast<const float *>(file.tensor_data(*t));
            return std::vector<float>(src, src + t->ne[0]);
        };
        const std::vector<float> q_norm_w = read_vec("attn_q_norm.weight");
        const std::vector<float> k_norm_w = read_vec("attn_k_norm.weight");

        void * dev_qn = nullptr;
        void * dev_kn = nullptr;
        (void) hipMalloc(&dev_qn, q_norm_w.size() * 4);
        (void) hipMalloc(&dev_kn, k_norm_w.size() * 4);
        (void) hipMemcpy(dev_qn, q_norm_w.data(), q_norm_w.size() * 4, hipMemcpyHostToDevice);
        (void) hipMemcpy(dev_kn, k_norm_w.data(), k_norm_w.size() * 4, hipMemcpyHostToDevice);

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

        // projections
        void * qf = nullptr;  // (T, n_head*2*head_dim) f32
        void * kf = nullptr;  // (T, n_head_kv*head_dim)
        void * vf = nullptr;
        void * out = nullptr; // (T, n_embd)
        (void) hipMalloc(&qf, (size_t) tokens * q_out * 4);
        (void) hipMalloc(&kf, (size_t) tokens * kv_out * 4);
        (void) hipMalloc(&vf, (size_t) tokens * kv_out * 4);
        (void) hipMalloc(&out, (size_t) tokens * n_embd * 4);

        omph::runtime::Linear linear;
        if (!linear.run(wq, ex16, (float *) qf, q_out, n_embd, tokens) ||
            !linear.run(wk, ex16, (float *) kf, kv_out, n_embd, tokens) ||
            !linear.run(wv, ex16, (float *) vf, kv_out, n_embd, tokens)) {
            return fail("projection failed");
        }

        // split q / gate, qk-norm, rope, attention
        void * q = nullptr;
        void * gate = nullptr;
        void * attn = nullptr;
        void * attn16 = nullptr;
        (void) hipMalloc(&q, (size_t) tokens * n_head * head_dim * 4);
        (void) hipMalloc(&gate, (size_t) tokens * n_head * head_dim * 4);
        (void) hipMalloc(&attn, (size_t) tokens * n_head * head_dim * 4);
        (void) hipMalloc(&attn16, (size_t) tokens * n_head * head_dim * 2);

        if (!omph::kernels::split_qg((const float *) qf, (float *) q, (float *) gate, tokens, n_head,
                                     head_dim, nullptr)) {
            return fail("split_qg failed");
        }
        if (!omph::kernels::rms_norm((const float *) q, (const float *) dev_qn, (float *) q,
                                     tokens * n_head, head_dim, (float) eps, 1.0f, nullptr) ||
            !omph::kernels::rms_norm((const float *) kf, (const float *) dev_kn, (float *) kf,
                                     tokens * n_head_kv, head_dim, (float) eps, 1.0f, nullptr)) {
            return fail("qk-norm failed");
        }
        if (trace) {
            std::vector<float> t1;
            read_back(t1, qf, (size_t) tokens * q_out);
            write_f32(prefix + "-q_full.f32", t1);
            read_back(t1, q, (size_t) tokens * n_head * head_dim);
            write_f32(prefix + "-q_normed.f32", t1);
            read_back(t1, kf, (size_t) tokens * kv_out);
            write_f32(prefix + "-k_normed.f32", t1);
            read_back(t1, gate, (size_t) tokens * n_head * head_dim);
            write_f32(prefix + "-gate.f32", t1);
        }
        if (!omph::kernels::rope_neox((float *) q, tokens, n_head, head_dim, n_rot,
                                      (float) freq_base, 0, nullptr) ||
            !omph::kernels::rope_neox((float *) kf, tokens, n_head_kv, head_dim, n_rot,
                                      (float) freq_base, 0, nullptr)) {
            return fail("rope failed");
        }
        if (trace) {
            std::vector<float> t1;
            read_back(t1, q, (size_t) tokens * n_head * head_dim);
            write_f32(prefix + "-q.f32", t1);
            read_back(t1, kf, (size_t) tokens * kv_out);
            write_f32(prefix + "-k.f32", t1);
        }
        omph::kernels::KvCache kv;
        kv.k_f32 = (const float *) kf;
        kv.v_f32 = (const float *) vf;
        const size_t work_bytes =
            omph::kernels::attention_gqa_work_bytes(tokens, n_head, n_head_kv, head_dim);
        void * work = nullptr;
        if (hipMalloc(&work, work_bytes) != hipSuccess ||
            !omph::kernels::attention_gqa((const float *) q, kv, (const float *) gate,
                                          (float *) attn, tokens, tokens, n_head, n_head_kv,
                                          head_dim, 1.0f / std::sqrt((float) head_dim), false,
                                          work, work_bytes, nullptr)) {
            return fail("attention failed");
        }

        // output projection
        if (!omph::kernels::cast_f32_to_f16((const float *) attn, attn16,
                                            tokens * n_head * head_dim, nullptr)) {
            return fail("cast (attn) failed");
        }
        if (!linear.run(wo, attn16, (float *) out, n_embd, n_head * head_dim, tokens)) {
            return fail("output projection failed");
        }
        if (hipDeviceSynchronize() != hipSuccess) {
            return fail("kernel failed");
        }

        std::vector<float> host_out;
        read_back(host_out, out, (size_t) tokens * n_embd);
        write_f32(prefix + ".out.f32", host_out);
        if (trace) {
            std::vector<float> t1;
            read_back(t1, attn, (size_t) tokens * n_head * head_dim);
            write_f32(prefix + "-attn_gated.f32", t1);
        }
        std::printf("attn layer %d: %lld tokens -> %s.out.f32\n", layer, (long long) tokens,
                    prefix.c_str());
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
    return 0;
}
