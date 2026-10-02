// The generation C ABI (#154) on model::Generator. No C++ exception crosses
// it: every entry point catches and returns an OMPH_E* code.
#include "omphalos.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/generator.hh"
#include "runtime/options.hh"
#include "text/chat.hh"
#include "text/json.hh"

static_assert((int) omph::model::GenerateResult::Stop::Length == OMPH_STOP_LENGTH &&
                  (int) omph::model::GenerateResult::Stop::EndOfGeneration == OMPH_STOP_EOG &&
                  (int) omph::model::GenerateResult::Stop::StopToken == OMPH_STOP_TOKEN &&
                  (int) omph::model::GenerateResult::Stop::Callback == OMPH_STOP_CALLBACK &&
                  (int) omph::model::GenerateResult::Stop::ContextFull == OMPH_STOP_CONTEXT &&
                  (int) omph::model::GenerateResult::Stop::Error == OMPH_STOP_ERROR,
              "the C stop reasons mirror GenerateResult::Stop");

struct omph_engine {
    std::unique_ptr<omph::model::Generator> gen;
    std::string error;
};

namespace {

int64_t fail(omph_engine * e, const int64_t code, const std::string & msg) {
    if (e != nullptr) {
        e->error = msg;
    }
    return code;
}

// copies n elements, or returns -(n) when they do not fit
template <typename T>
int64_t copy_out(const T * src, const size_t n, T * dst, const size_t cap) {
    if (n > cap || (dst == nullptr && n > 0)) {
        return -(int64_t) n;
    }
    if (n > 0) {
        std::memcpy(dst, src, n * sizeof(T));
    }
    return (int64_t) n;
}

} // namespace

extern "C" {

void omph_engine_params_default(omph_engine_params * p) {
    if (p == nullptr) return;
    p->model_path = nullptr;
    p->context = 8192;
    p->chunk = 512;
    p->mtp = 1;
    p->cache_mib = 2048;
}

omph_engine * omph_engine_load(const omph_engine_params * p, char * err, const size_t err_size) {
    const auto report = [&](const std::string & m) {
        if (err != nullptr && err_size > 0) {
            std::snprintf(err, err_size, "%s", m.c_str());
        }
    };
    if (p == nullptr || p->model_path == nullptr) {
        report("no model path");
        return nullptr;
    }
    try {
        auto e = std::make_unique<omph_engine>();
        omph::model::Generator::Config c;
        c.model = p->model_path;
        c.context = p->context > 0 ? p->context : 8192;
        c.chunk = p->chunk > 0 ? p->chunk : 512;
        c.mtp = p->mtp != 0;
        c.cache_mib = p->cache_mib > 0 ? p->cache_mib : 0;
        e->gen = std::make_unique<omph::model::Generator>(c, omph::runtime::EnvOptions::from_env());
        return e.release();
    } catch (const std::exception & ex) {
        report(ex.what());
        return nullptr;
    }
}

void omph_engine_free(omph_engine * e) { delete e; }

const char * omph_engine_last_error(const omph_engine * e) { return e != nullptr ? e->error.c_str() : ""; }

int64_t omph_engine_context(const omph_engine * e) { return e != nullptr ? e->gen->context() : 0; }

int64_t omph_tokenize(omph_engine * e, const char * text, const size_t len, const int parse_special,
                      int32_t * ids, const size_t cap) {
    if (e == nullptr || (text == nullptr && len > 0)) return fail(e, OMPH_E_ARG, "bad argument");
    try {
        const std::vector<int32_t> v =
            e->gen->tokenizer().encode(std::string_view(text != nullptr ? text : "", len), parse_special != 0);
        return copy_out(v.data(), v.size(), ids, cap);
    } catch (const std::exception & ex) {
        return fail(e, OMPH_E_ARG, ex.what());
    }
}

int64_t omph_detokenize(omph_engine * e, const int32_t * ids, const size_t n, const int special, char * out,
                        const size_t cap) {
    if (e == nullptr || (ids == nullptr && n > 0)) return fail(e, OMPH_E_ARG, "bad argument");
    try {
        const std::string s = e->gen->tokenizer().decode(std::vector<int32_t>(ids, ids + n), special != 0);
        return copy_out(s.data(), s.size(), out, cap);
    } catch (const std::exception & ex) {
        return fail(e, OMPH_E_ARG, ex.what());
    }
}

int64_t omph_chat_render(omph_engine * e, const char * request_json, char * out, const size_t cap) {
    if (e == nullptr || request_json == nullptr) return fail(e, OMPH_E_ARG, "bad argument");
    try {
        const std::string s = omph::text::render_chat(omph::text::Json::parse(request_json));
        return copy_out(s.data(), s.size(), out, cap);
    } catch (const std::exception & ex) {
        return fail(e, OMPH_E_TEMPLATE, ex.what());
    }
}

void omph_generate_params_default(omph_generate_params * p) {
    if (p == nullptr) return;
    p->max_tokens = 256;
    p->temperature = 0.0f;
    p->top_k = 0;
    p->top_p = 1.0f;
    p->min_p = 0.0f;
    p->seed = 0;
    p->speculative = 1;
    p->stop = nullptr;
    p->n_stop = 0;
}

int omph_generate(omph_engine * e, const int32_t * prompt, const size_t n, const omph_generate_params * p,
                  omph_token_callback cb, void * user, omph_generate_result * result) {
    if (e == nullptr || prompt == nullptr || n == 0) return (int) fail(e, OMPH_E_ARG, "empty prompt");
    omph_generate_params def;
    omph_generate_params_default(&def);
    const omph_generate_params & gp = p != nullptr ? *p : def;
    try {
        omph::model::GenerateRequest req;
        req.max_tokens = gp.max_tokens;
        req.sampling.temperature = gp.temperature;
        req.sampling.top_k = gp.top_k;
        req.sampling.top_p = gp.top_p;
        req.sampling.min_p = gp.min_p;
        req.sampling.seed = gp.seed;
        req.speculative = gp.speculative != 0;
        if (gp.stop != nullptr) req.stop.assign(gp.stop, gp.stop + gp.n_stop);
        const omph::text::Tokenizer & tok = e->gen->tokenizer();
        const auto res = e->gen->generate(std::vector<int32_t>(prompt, prompt + n), req, [&](const int32_t t) {
            if (cb == nullptr) return true;
            const std::string piece = tok.piece(t, false);
            return cb(t, piece.data(), piece.size(), user) != 0;
        });
        if (result != nullptr) {
            result->n_tokens = (int64_t) res.tokens.size();
            result->prompt_tokens = res.prompt_tokens;
            result->cached_tokens = res.cached_tokens;
            result->prefill_ms = res.prefill_ms;
            result->decode_ms = res.decode_ms;
            result->drafted = res.drafted;
            result->accepted = res.accepted;
            result->stop_reason = (int) res.stop;
        }
        if (res.stop == omph::model::GenerateResult::Stop::Error) {
            return (int) fail(e, OMPH_E_RUN, "generation failed (see stderr)");
        }
        return 0;
    } catch (const std::exception & ex) {
        return (int) fail(e, OMPH_E_RUN, ex.what());
    }
}

} // extern "C"
