/* omphalos — public C ABI (PLAN.md §6.1).
 *
 * Deliberately small: the inference API grows here as milestones land
 * (C ABI is callable from anything: Python ctypes, Delphi, the HTTP server).
 */
#ifndef OMPHALOS_H
#define OMPHALOS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OMPHALOS_VERSION_MAJOR 0
#define OMPHALOS_VERSION_MINOR 1
#define OMPHALOS_VERSION_PATCH 0

/* Library version, "major.minor.patch". */
const char * omph_version(void);

/* Number of HIP devices visible to the runtime (0 when none). */
int omph_device_count(void);

/* Writes a human-readable description of device `index` into `out`.
 * Returns 0 on success, negative on error. */
int omph_device_info(int index, char * out, size_t out_size);

/* Runs a trivial GPU kernel end to end (self test). Returns 0 on success. */
int omph_self_test(void);

/* ---- generation (#154) -------------------------------------------------
 *
 * An engine holds the model, its tokenizer and one sequence in its caches.
 * One engine is used by one thread at a time. Functions return 0 (or a
 * count) on success and a negative OMPH_E* code on error; the message is
 * then in omph_engine_last_error(). Functions that fill a buffer return the
 * number of elements written, or -(needed) when `cap` is too small.
 */

#define OMPH_E_ARG (-1)      /* bad argument */
#define OMPH_E_LOAD (-2)     /* the model could not be loaded */
#define OMPH_E_RUN (-3)      /* the GPU run failed */
#define OMPH_E_TEMPLATE (-4) /* the chat request is invalid / rejected by the template */

typedef struct omph_engine omph_engine;

typedef struct {
    const char * model_path; /* .omph file (omph-convert writes it from the GGUF, #178) */
    int64_t context;         /* KV capacity in tokens (default 8192) */
    int64_t chunk;           /* prefill chunk (default 512) */
    int mtp;                 /* load the MTP block for speculative decoding (default 1) */
    int64_t cache_mib;       /* pinned host RAM for sequence checkpoints: a prompt that
                                diverges from the cached one resumes from the latest
                                checkpoint before the difference (default 2048, 0: none) */
    int64_t kv_ram_mib;      /* pinned host RAM for whole conversations: a prompt that leaves
                                the cached conversation saves it, one that continues a saved
                                one restores it without a prefill (default 8192, 0: none) */
} omph_engine_params;

void omph_engine_params_default(omph_engine_params * p);
/* NULL on failure, with the reason in `err` (when not NULL). */
omph_engine * omph_engine_load(const omph_engine_params * p, char * err, size_t err_size);
void omph_engine_free(omph_engine * e);
const char * omph_engine_last_error(const omph_engine * e);
int64_t omph_engine_context(const omph_engine * e);

/* text -> ids; parse_special: control tokens written in the text become ids */
int64_t omph_tokenize(omph_engine * e, const char * text, size_t len, int parse_special,
                      int32_t * ids, size_t cap);
/* ids -> bytes (not NUL-terminated); special: write control tokens' text */
int64_t omph_detokenize(omph_engine * e, const int32_t * ids, size_t n, int special, char * out,
                        size_t cap);
/* a chat request (JSON: messages, tools, add_generation_prompt, enable_thinking,
 * reasoning_effort, ...) -> the prompt text (not NUL-terminated) */
int64_t omph_chat_render(omph_engine * e, const char * request_json, char * out, size_t cap);

typedef struct {
    int64_t max_tokens;   /* default 256 */
    float temperature;    /* 0: greedy (default) */
    int top_k;            /* 0: off */
    float top_p;          /* 1: off */
    float min_p;          /* 0: off */
    uint64_t seed;
    int speculative;      /* MTP drafts, greedy or sampled (default 1) */
    const int32_t * stop; /* extra stop tokens */
    size_t n_stop;
} omph_generate_params;

#define OMPH_STOP_LENGTH 0
#define OMPH_STOP_EOG 1          /* an end-of-generation token */
#define OMPH_STOP_TOKEN 2        /* one of params.stop */
#define OMPH_STOP_CALLBACK 3
#define OMPH_STOP_CONTEXT 4
#define OMPH_STOP_ERROR 5

typedef struct {
    int64_t n_tokens;      /* generated (a final stop token included) */
    int64_t prompt_tokens;
    int64_t cached_tokens; /* of the prompt, reused from the previous sequence */
    double prefill_ms;
    double decode_ms;
    int64_t drafted;
    int64_t accepted;
    int stop_reason;       /* OMPH_STOP_* */
} omph_generate_result;

/* Called for every generated token with its bytes (control tokens: empty);
 * return 0 to stop. */
typedef int (*omph_token_callback)(int32_t token, const char * piece, size_t piece_len, void * user);

void omph_generate_params_default(omph_generate_params * p);
/* Generates after `prompt`, continuing the cached sequence when the prompt
 * extends it. `cb` and `result` may be NULL. */
int omph_generate(omph_engine * e, const int32_t * prompt, size_t n, const omph_generate_params * p,
                  omph_token_callback cb, void * user, omph_generate_result * result);

#ifdef __cplusplus
}
#endif

#endif /* OMPHALOS_H */
