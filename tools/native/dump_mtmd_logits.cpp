// The reference for images (#160): llama.cpp evaluates a prompt with images
// through mtmd and writes the logits of its last position (f32), and the
// prompt's token ids with each image's rows as -1 (to check that omphalos
// tokenizes the same prompt).
//
// usage: dump_mtmd_logits <model.gguf> <mmproj.gguf> <prompt.txt> <out.f32> <out-tokens.txt> [image ...]
//   prompt.txt: text with one <__media__> marker per image (mtmd adds
//   <|vision_start|> / <|vision_end|> around each image itself)
// The images are encoded on the CPU (as omphalos does), the text model runs
// with all layers on the GPU and an f16 KV cache.
// DUMP_TAIL=<ids file>: then those tokens but the last, in one batch, and
// the logits of every position after the prompt (the prompt's last row
// first): the reference for omph-generate --force ... --logits-out.
#include <llama.h>
#include <mtmd-helper.h>
#include <mtmd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s <model> <mmproj> <prompt.txt> <out.f32> <out-tokens.txt> [image ...]\n",
                     argv[0]);
        return 2;
    }
    std::ifstream pf(argv[3], std::ios::binary);
    std::stringstream ss;
    ss << pf.rdbuf();
    const std::string prompt = ss.str();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) return 1;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    llama_context * lctx = llama_init_from_model(model, cp);
    if (lctx == nullptr) return 1;

    mtmd_context_params vp = mtmd_context_params_default();
    vp.use_gpu = false;
    vp.print_timings = false;
    mtmd_context * vctx = mtmd_init_from_file(argv[2], model, vp);
    if (vctx == nullptr) return 1;

    std::vector<mtmd_bitmap *> bitmaps;
    for (int i = 6; i < argc; ++i) {
        mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_file(vctx, argv[i], false,
                                                                          mtmd_helper_init_opt_default());
        if (w.bitmap == nullptr) {
            std::fprintf(stderr, "cannot load %s\n", argv[i]);
            return 1;
        }
        bitmaps.push_back(w.bitmap);
    }
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_input_text text{prompt.c_str(), prompt.size(), false, true};
    std::vector<const mtmd_bitmap *> cb(bitmaps.begin(), bitmaps.end());
    if (mtmd_tokenize(vctx, chunks, &text, cb.data(), cb.size()) != 0) {
        std::fprintf(stderr, "mtmd_tokenize failed\n");
        return 1;
    }
    std::ofstream tf(argv[5]);
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token * t = mtmd_input_chunk_get_tokens_text(c, &n);
            for (size_t k = 0; k < n; ++k) tf << t[k] << "\n";
        } else {
            for (size_t k = 0; k < mtmd_input_chunk_get_n_tokens(c); ++k) tf << -1 << "\n";
        }
    }
    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(vctx, lctx, chunks, 0, 0, (int32_t) cp.n_batch, true, &n_past) != 0) {
        std::fprintf(stderr, "eval failed\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> rows(llama_get_logits_ith(lctx, -1), llama_get_logits_ith(lctx, -1) + n_vocab);
    if (const char * tail = std::getenv("DUMP_TAIL")) {
        std::ifstream idf(tail);
        std::vector<llama_token> ids;
        for (long long v; idf >> v;) ids.push_back((llama_token) v);
        if (ids.size() > 1) {
            const int n = (int) ids.size() - 1;
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) {
                b.token[i] = ids[(size_t) i];
                b.pos[i] = n_past + i;
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = true;
            }
            b.n_tokens = n;
            if (llama_decode(lctx, b) != 0) {
                std::fprintf(stderr, "tail decode failed\n");
                return 1;
            }
            for (int i = 0; i < n; ++i) {
                const float * l = llama_get_logits_ith(lctx, i);
                rows.insert(rows.end(), l, l + n_vocab);
            }
            llama_batch_free(b);
        }
    }
    FILE * f = std::fopen(argv[4], "wb");
    if (f == nullptr || std::fwrite(rows.data(), 4, rows.size(), f) != rows.size()) return 1;
    std::fclose(f);
    std::fprintf(stderr, "%zu tokens, n_past %d, logits written\n", mtmd_helper_get_n_tokens(chunks), (int) n_past);
    for (mtmd_bitmap * b : bitmaps) mtmd_bitmap_free(b);
    mtmd_input_chunks_free(chunks);
    mtmd_free(vctx);
    llama_free(lctx);
    llama_model_free(model);
    return 0;
}
