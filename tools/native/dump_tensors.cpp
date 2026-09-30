// Golden-tensor dumper for the omphalos reference implementation (PLAN.md §9).
//
// Loads a GGUF with libllama, evaluates one fixed prompt and writes every float
// activation of the graph to <out>/<NNNN>-<name>.<f32|f16|bf16>, with an
// index.jsonl describing each tensor (name, op, type, ne, file). The prompt
// tokens are saved to tokens.txt.
//
// build: tools/native/build.sh <llama.cpp-dir>
// usage: dump_tensors -m model.gguf -p "prompt" -o outdir [-ngl 0] [-c 512]

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct dump_state {
    std::string outdir;
    FILE * index = nullptr;
    int counter = 0;
    long long saved = 0;
    long long bytes = 0;
};

static std::string sanitize(const char * name) {
    std::string s(name);
    for (char & c : s) {
        if (!(std::isalnum((unsigned char) c) || c == '.' || c == '_' || c == '-')) {
            c = '_';
        }
    }
    return s;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;

    if (ask) {
        return true;  // observe every node
    }
    if (t == nullptr || t->name[0] == '\0') {
        return true;
    }
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
        return true;  // activations only
    }
    if (t->buffer == nullptr) {
        return true;
    }
    // Pack the logical content (ne0 fastest, rows contiguous): views can have
    // arbitrary strides between rows, so a flat memcpy would mix rows up.
    const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
    const size_t logical = row_bytes * (size_t) t->ne[1] * (size_t) t->ne[2] * (size_t) t->ne[3];
    if (row_bytes == 0 || logical == 0) {
        return true;
    }

    std::vector<uint8_t> packed(logical);
    if (ggml_backend_buffer_is_host(t->buffer) && ggml_nbytes(t) == logical) {
        std::memcpy(packed.data(), t->data, logical);
    } else if (ggml_backend_buffer_is_host(t->buffer)) {
        const char * base = (const char *) t->data;
        size_t off = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                    std::memcpy(packed.data() + off,
                                base + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], row_bytes);
                    off += row_bytes;
                }
            }
        }
    } else {
        size_t off = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                    const size_t byte_off = i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    ggml_backend_tensor_get(t, packed.data() + off, byte_off, row_bytes);
                    off += row_bytes;
                }
            }
        }
    }

    const char * ext = t->type == GGML_TYPE_F32  ? "f32"
                     : t->type == GGML_TYPE_F16  ? "f16"
                     : "bf16";
    char fname[1024];
    std::snprintf(fname, sizeof(fname), "%s/%04d-%s.%s", st->outdir.c_str(), st->counter++,
                  sanitize(t->name).c_str(), ext);
    if (FILE * f = std::fopen(fname, "wb")) {
        std::fwrite(packed.data(), 1, logical, f);
        std::fclose(f);
    }
    const char * base = std::strrchr(fname, '/');
    std::fprintf(st->index,
                 "{\"name\": \"%s\", \"op\": \"%s\", \"type\": \"%s\","
                 " \"ne\": [%lld, %lld, %lld, %lld], \"bytes\": %lld, \"file\": \"%s\"}\n",
                 t->name, ggml_op_name(t->op), ggml_type_name(t->type), (long long) t->ne[0],
                 (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                 (long long) logical, base ? base + 1 : fname);
    st->saved++;
    st->bytes += (long long) logical;
    return true;
}

static void log_cb(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

static void usage(const char * prog) {
    std::fprintf(stderr, "usage: %s -m model.gguf -p \"prompt\" -o outdir [-ngl N] [-c N]\n",
                 prog);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt;
    std::string outdir = "golden";
    int ngl = 0;
    int n_ctx = 512;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after %s\n", key.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (key == "-m" || key == "--model") {
            model_path = value();
        } else if (key == "-p" || key == "--prompt") {
            prompt = value();
        } else if (key == "-o" || key == "--out") {
            outdir = value();
        } else if (key == "-ngl" || key == "--n-gpu-layers") {
            ngl = std::atoi(value().c_str());
        } else if (key == "-c" || key == "--ctx-size") {
            n_ctx = std::atoi(value().c_str());
        } else if (key == "-h" || key == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", key.c_str());
            usage(argv[0]);
            return 2;
        }
    }
    if (model_path.empty() || prompt.empty()) {
        usage(argv[0]);
        return 2;
    }

    const std::string mkdir_cmd = "mkdir -p '" + outdir + "'";
    if (std::system(mkdir_cmd.c_str()) != 0) {
        std::fprintf(stderr, "cannot create output dir %s\n", outdir.c_str());
        return 1;
    }

    dump_state st;
    st.outdir = outdir;
    st.index = std::fopen((outdir + "/index.jsonl").c_str(), "w");
    if (st.index == nullptr) {
        std::fprintf(stderr, "cannot write index.jsonl\n");
        return 1;
    }

    llama_backend_init();
    llama_log_set(log_cb, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_special = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens(prompt.size() + 2);
    int32_t n_tok = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                                   tokens.data(), (int32_t) tokens.size(), add_special, true);
    if (n_tok < 0) {  // buffer too small: returns negative of the required size
        tokens.resize(-n_tok);
        n_tok = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(),
                               (int32_t) tokens.size(), add_special, true);
    }
    if (n_tok <= 0) {
        std::fprintf(stderr, "tokenization failed\n");
        return 1;
    }
    tokens.resize(n_tok);
    if (FILE * tf = std::fopen((outdir + "/tokens.txt").c_str(), "w")) {
        for (int32_t i = 0; i < n_tok; ++i) {
            std::fprintf(tf, "%d\n", tokens[i]);
        }
        std::fclose(tf);
    }
    if (FILE * pf = std::fopen((outdir + "/prompt.txt").c_str(), "w")) {
        std::fputs(prompt.c_str(), pf);
        std::fclose(pf);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) n_ctx;
    cparams.n_batch = (uint32_t) n_ctx;
    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        return 1;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tok);
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    std::fclose(st.index);
    std::fprintf(stderr, "dumped %lld tensors (%lld bytes) to %s\n", st.saved, st.bytes,
                 outdir.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
