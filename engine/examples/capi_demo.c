/* The generation C ABI from plain C (#154): load, render a chat request,
 * tokenize, generate with a streaming callback, detokenize, a second turn
 * that continues the cached sequence.
 *
 * usage: omph-capi-demo <model.omph> [<mmproj.gguf> <image>]
 * (with an mmproj and an image: one more turn about the image, #180)
 */
#include "omphalos.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int on_token(int32_t token, const char * piece, size_t len, void * user) {
    (void) token;
    (void) user;
    fwrite(piece, 1, len, stdout);
    fflush(stdout);
    return 1;
}

static int run_turn(omph_engine * e, const char * request, const char * image, size_t image_size) {
    char text[16384];
    const int64_t nt = omph_chat_render(e, request, text, sizeof(text));
    if (nt < 0) {
        fprintf(stderr, "render: %s\n", omph_engine_last_error(e));
        return 1;
    }
    int32_t ids[8192];
    const int64_t n = omph_tokenize(e, text, (size_t) nt, 1, ids, 8192);
    if (n < 0) {
        fprintf(stderr, "tokenize: %s\n", omph_engine_last_error(e));
        return 1;
    }
    omph_generate_params gp;
    omph_generate_params_default(&gp);
    gp.max_tokens = 64;
    if (image != NULL) {
        gp.images = &image;
        gp.image_sizes = &image_size;
        gp.n_images = 1;
    }
    omph_generate_result r;
    if (omph_generate(e, ids, (size_t) n, &gp, on_token, NULL, &r) != 0) {
        fprintf(stderr, "generate: %s\n", omph_engine_last_error(e));
        return 1;
    }
    printf("\n[%lld prompt tokens, %lld cached; %lld generated in %.1f ms, drafts %lld/%lld, stop %d]\n",
           (long long) r.prompt_tokens, (long long) r.cached_tokens, (long long) r.n_tokens, r.decode_ms,
           (long long) r.accepted, (long long) r.drafted, r.stop_reason);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.omph> [<mmproj.gguf> <image>]\n", argv[0]);
        return 2;
    }
    printf("omphalos %s\n", omph_version());
    omph_engine_params p;
    omph_engine_params_default(&p);
    p.model_path = argv[1];
    p.context = 4096;
    if (argc >= 4) p.mmproj_path = argv[2];
    char err[256];
    omph_engine * e = omph_engine_load(&p, err, sizeof(err));
    if (e == NULL) {
        fprintf(stderr, "load: %s\n", err);
        return 1;
    }
    /* round trip */
    const char * s = "Ciao, mondo! \xe2\x98\x95";
    int32_t ids[64];
    const int64_t n = omph_tokenize(e, s, strlen(s), 1, ids, 64);
    char back[128];
    const int64_t m = omph_detokenize(e, ids, (size_t) n, 1, back, sizeof(back));
    printf("round trip: %lld tokens, %s\n", (long long) n,
           m == (int64_t) strlen(s) && memcmp(back, s, (size_t) m) == 0 ? "identical" : "DIFFERENT");
    /* a too small buffer reports the size it needs */
    const int64_t need = omph_tokenize(e, s, strlen(s), 1, ids, 1);
    printf("small buffer: %lld (needs %lld)\n", (long long) need, (long long) n);
    /* two turns: the second continues the cached sequence */
    int rc = run_turn(e, "{\"messages\":[{\"role\":\"user\",\"content\":\"Name three primary colors.\"}],"
                         "\"add_generation_prompt\":true,\"enable_thinking\":false}", NULL, 0);
    if (rc == 0) {
        rc = run_turn(e, "{\"messages\":[{\"role\":\"user\",\"content\":\"Name three primary colors.\"},"
                         "{\"role\":\"assistant\",\"content\":\"Red, yellow and blue.\"},"
                         "{\"role\":\"user\",\"content\":\"And the secondary ones?\"}],"
                         "\"add_generation_prompt\":true,\"enable_thinking\":false}", NULL, 0);
    }
    if (rc == 0 && argc >= 4) {  /* an image, as file bytes */
        FILE * f = fopen(argv[3], "rb");
        if (f == NULL) {
            fprintf(stderr, "cannot open %s\n", argv[3]);
            return 1;
        }
        fseek(f, 0, SEEK_END);
        const long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        char * img = malloc((size_t) size);
        const size_t got = img != NULL ? fread(img, 1, (size_t) size, f) : 0;
        fclose(f);
        rc = got == (size_t) size
                 ? run_turn(e, "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image\"},"
                               "{\"type\":\"text\",\"text\":\"Describe the image in one sentence.\"}]}],"
                               "\"add_generation_prompt\":true,\"enable_thinking\":false}",
                            img, (size_t) size)
                 : 1;
        free(img);
    }
    /* errors are codes, not exceptions */
    char out[16];
    const int64_t bad = omph_chat_render(e, "{\"messages\":[]}", out, sizeof(out));
    printf("bad request: %lld (%s)\n", (long long) bad, omph_engine_last_error(e));
    omph_engine_free(e);
    return rc;
}
