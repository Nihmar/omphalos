// omph-server — an OpenAI-compatible HTTP server on the engine's Generator (#156).
//
// usage: omph-server <model.gguf> [options]
//   --host H          address to bind (default 127.0.0.1)
//   --port P          port (default 8080)
//   --ctx N           KV capacity (default 8192)
//   --chunk N         prefill chunk (default 512)
//   --no-mtp          do not load the MTP block (no speculative decoding, -352 MiB of VRAM)
//   --dflash FILE     draft with a DFlash2 drafter .omph instead of the MTP block (#245)
//   --kv-ram MIB      pinned host RAM for whole conversations (default 8192, 0: none, #179):
//                     a prompt that leaves the cached conversation saves it first, one that
//                     continues a saved conversation restores it instead of prefilling
//   --cache-ram MIB   pinned host RAM for sequence checkpoints (default 2048, 0: none): a
//                     retried answer or a history without the reasoning resumes from one
//   --mmproj FILE     the vision encoder: images as base64 data: URLs in image_url
//                     items, encoded on the CPU (builds with OMPH_LLAMA_DIR, #160)
//   --alias NAME      the model id in the API (default: the file name without .gguf)
//   --api-key KEY     require "Authorization: Bearer KEY"
//   --cors ORIGIN     allow browser requests from ORIGIN (e.g. "*")
//   --temp T, --top-k K, --top-p P, --min-p M, --max-tokens N
//                     defaults for requests that leave them out (default: greedy,
//                     until the context is full); greedy and sampled requests both
//                     decode speculatively with the MTP block (#197)
//   --repeat-penalty P, --repeat-last-n N, --frequency-penalty P, --presence-penalty P
//                     llama.cpp's penalties sampler (#298): the last N tokens of
//                     the sequence are scaled by P (sign-aware) and pushed down
//                     by freq/pres (defaults 1.0 / 64 / 0.0 / 0.0, i.e. off);
//                     a penalized greedy step gives up the speculative drafts
//
// Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions,
// POST /v1/completions (stream or not). One request at a time; a chat's next
// turn continues the cached sequence when its prompt extends it.
#include "model/generator.hh"
#include "runtime/options.hh"
#include "runtime/timing.hh"
#include "server/access.hh"
#include "server/http.hh"
#include "server/openai.hh"
#include "tools/cli.hh"
#include "text/json.hh"
#ifdef OMPH_VISION
#include "vision/encoder.hh"
#endif

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <vector>

namespace {

using omph::model::GenerateResult;
using omph::server::Delta;
using omph::text::Json;

// The sampling the request actually runs with: the body's, or the server's
// default when it leaves it out. Printed in the per-request summary so that
// "which sampling did that client get" is answerable from the log (#284).
std::string sampling_label(const omph::server::Job & job) {
    std::string label = job.temperature <= 0.0f ? "greedy" : "";
    if (job.temperature > 0.0f) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "temp %.2f top-k %d top-p %.2f min-p %.2f", (double) job.temperature,
                      job.top_k, (double) job.top_p, (double) job.min_p);
        label = buf;
    }
    if (job.repeat_penalty != 1.0f || job.frequency_penalty != 0.0f || job.presence_penalty != 0.0f) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), " repeat %.2f freq %.2f pres %.2f last-n %d", (double) job.repeat_penalty,
                      (double) job.frequency_penalty, (double) job.presence_penalty, job.penalty_last_n);
        label += buf;
    }
    if (job.seeded) {
        label += " seed " + std::to_string(job.seed);
    }
    return label;
}

struct Server {
    omph::model::Generator & gen;
    std::string model_id;
    std::string host;
    int port = 0;
    std::string api_key;
    std::string cors;
    omph::server::Defaults defaults;
#ifdef OMPH_VISION
    omph::vision::Encoder * vision = nullptr;
#endif

    static Json num(const double v) { return Json::number(v, true); }
    static Json str(const std::string & s) { return Json::string(s); }

    bool reply(omph::server::Connection & c, const int status, const Json & body) {
        // every string that reaches a response is valid UTF-8, whatever byte
        // sequence a tokenizer or a JSON escape put in it (#338)
        return c.respond(status, "application/json", omph::server::sanitize_utf8(body.dump()));
    }
    bool fail(omph::server::Connection & c, const int status, const std::string & message,
              const std::string & type = "invalid_request_error", const std::string & param = "") {
        return reply(c, status, omph::server::error_body(message, type, param));
    }

    Json model_object() const {
        Json m = Json::object();
        m.set("id", str(model_id));
        m.set("object", str("model"));
        m.set("created", num(0));
        m.set("owned_by", str("omphalos"));
        return m;
    }

    void handle(omph::server::Connection & c, const omph::server::Request & req) {
        if (!cors.empty()) {
            c.extra_headers = {{"Access-Control-Allow-Origin", cors}};
        }
        if (const std::string * h = req.header("host"); h != nullptr && !omph::server::host_allowed(*h, host)) {
            fail(c, 403, "the Host header does not match where this server is bound (" + host +
                             "), so a page that resolved its own name here cannot drive it",
                 "invalid_request_error");
            return;
        }
        if (req.method == "OPTIONS") {  // CORS preflight
            if (!cors.empty()) {
                c.extra_headers.emplace_back("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                c.extra_headers.emplace_back("Access-Control-Allow-Headers", "Content-Type, Authorization");
            }
            c.respond(204, "", "");
            return;
        }
        std::string path = req.path;
        if (path.rfind("/v1/", 0) == 0) path.erase(0, 3);
        if (path == "/health") {
            Json ok = Json::object();
            ok.set("status", str("ok"));
            reply(c, 200, ok);
            return;
        }
        if (!api_key.empty()) {
            const std::string * auth = req.header("authorization");
            if (auth == nullptr || !omph::server::bearer_key_ok(*auth, api_key)) {
                fail(c, 401, "invalid API key", "authentication_error");
                return;
            }
        }
        if (path == "/models" || path == "/models/" + model_id) {
            if (req.method != "GET") {
                fail(c, 405, "use GET");
                return;
            }
            if (path != "/models") {
                reply(c, 200, model_object());
                return;
            }
            Json list = Json::object();
            list.set("object", str("list"));
            Json data = Json::array();
            data.push(model_object());
            list.set("data", std::move(data));
            reply(c, 200, list);
            return;
        }
        if (path == "/chat/completions" || path == "/completions") {
            if (req.method != "POST") {
                fail(c, 405, "use POST");
                return;
            }
            // A text/plain POST from a web page is a CORS-simple request: it
            // would start a generation and evict or save conversations (#338).
            const std::string * ct = req.header("content-type");
            if (ct == nullptr || !omph::server::json_content_type(*ct)) {
                fail(c, 415, "Content-Type: application/json is required");
                return;
            }
            complete(c, req, path == "/chat/completions");
            return;
        }
        fail(c, 404, "no such endpoint: " + req.method + " " + req.path, "not_found_error");
    }

    void complete(omph::server::Connection & c, const omph::server::Request & req, const bool chat) {
        omph::server::Job job;
        try {
            job = omph::server::parse_request(Json::parse(req.body), chat, defaults);
        } catch (const omph::server::BadRequest & e) {
            fail(c, 400, e.message, "invalid_request_error", e.param);
            return;
        } catch (const std::exception & e) {
            fail(c, 400, std::string("invalid JSON body: ") + e.what());
            return;
        }
        const omph::text::Tokenizer & tok = gen.tokenizer();
        std::vector<int32_t> prompt;
        if (!job.prompt_ids.empty()) {
            prompt = job.prompt_ids;
        } else if (!job.prompt_segments.empty()) {
            prompt = omph::text::tokenize_chat(job.prompt_segments, tok);  // #292: text stays text
        } else {
            prompt = tok.encode(job.prompt, true);  // a raw completion: the caller owns the tokens
        }
        for (const int32_t t : prompt) {
            if (t < 0 || t >= tok.size()) {
                fail(c, 400, "token id " + std::to_string(t) + " out of range", "invalid_request_error", "prompt");
                return;
            }
        }
        if (prompt.empty()) {
            fail(c, 400, "the prompt is empty", "invalid_request_error", "prompt");
            return;
        }
        omph::model::GenerateRequest greq;
        const int32_t image_pad = tok.find("<|image_pad|>");
        const auto placeholders = std::count(prompt.begin(), prompt.end(), image_pad);
        if (placeholders != (std::ptrdiff_t) job.images.size()) {
            // A literal image-pad string in a message's text is tokenized as the
            // token itself (#292): say so instead of leaving the caller with a
            // count that does not match anything they sent.
            fail(c, 400,
                 "the prompt carries " + std::to_string((long long) placeholders) + " image placeholders and "
                     "the request " + std::to_string(job.images.size()) + " images; a literal image-pad token "
                     "in a message's text is tokenized as the token itself (#292)",
                 "invalid_request_error", "messages");
            return;
        }
        if (!job.images.empty()) {
#ifdef OMPH_VISION
            if (vision == nullptr) {
                fail(c, 400, "this server has no vision encoder (start it with --mmproj)", "invalid_request_error",
                     "messages");
                return;
            }
            // the CPU encodes while the GPU prefills the text before the first image (#180)
            const std::vector<int32_t> before(prompt.begin(), std::find(prompt.begin(), prompt.end(), image_pad));
            const double t0 = omph::runtime::now_ms();
            double t_text = 0.0;
            try {
                greq.images = vision->encode_all(job.images, [&] {
                    gen.prefill(before);
                    t_text = omph::runtime::now_ms() - t0;
                });
            } catch (const std::runtime_error & e) {
                fail(c, 400, e.what(), "invalid_request_error", "messages");
                return;
            }
            std::fprintf(stderr, "%zu image(s) ready in %.0f ms (CPU), the %zu tokens before them in %.0f ms (GPU)\n",
                         job.images.size(), omph::runtime::now_ms() - t0, before.size(), t_text);
#else
            fail(c, 400, "this server was built without vision", "invalid_request_error", "messages");
            return;
#endif
        }
        int64_t prompt_len = (int64_t) prompt.size();  // with the images' rows
        for (const auto & im : greq.images) prompt_len += im->n_tokens() - 1;
        if (prompt_len >= gen.context()) {
            fail(c, 400,
                 "the prompt is " + std::to_string(prompt_len) + " tokens; the context holds " +
                     std::to_string(gen.context()),
                 "invalid_request_error", chat ? "messages" : "prompt");
            return;
        }
        greq.max_tokens = job.max_tokens > 0 ? job.max_tokens : gen.context();
        greq.sampling.temperature = job.temperature;
        greq.sampling.top_k = job.top_k;
        greq.sampling.top_p = job.top_p;
        greq.sampling.min_p = job.min_p;
        greq.sampling.seed = job.seeded ? job.seed : std::random_device{}() * 0x100000000ull + std::random_device{}();
        greq.sampling.repeat_penalty = job.repeat_penalty;
        greq.sampling.frequency_penalty = job.frequency_penalty;
        greq.sampling.presence_penalty = job.presence_penalty;
        greq.sampling.penalty_last_n = job.penalty_last_n;
        bool client_gone = false;  // the socket says the client is gone (#338)
        // A 20 s - 2.5 min prefill with no output looks like a hang (#310): one
        // line every 3 s, the same shape as the decode progress below.
        {
            const double t0 = omph::runtime::now_ms();
            double t_last = t0;
            greq.on_prefill = [&c, &client_gone, t0, t_last](const int64_t done,
                                                             const int64_t total) mutable {
                if (c.client_gone()) {  // the abandoned request stops here, not an hour later (#338)
                    client_gone = true;
                    return false;
                }
                const double now = omph::runtime::now_ms();
                if (done < total && now - t_last < 3000.0) {
                    return true;
                }
                t_last = now;
                std::fprintf(stderr, "  prefill %lld / %lld tokens, %.1f t/s\n", (long long) done,
                             (long long) total, done > 0 && now > t0 ? 1000.0 * (double) done / (now - t0) : 0.0);
                return true;
            };
        }

        const std::string id = omph::server::random_id(chat ? "chatcmpl-" : "cmpl-", 24);
        const auto created = (double) std::time(nullptr);
        // a chunk of the stream (or the response's frame)
        const auto frame = [&](const char * object) {
            Json f = Json::object();
            f.set("id", str(id));
            f.set("object", str(object));
            f.set("created", num(created));
            f.set("model", str(model_id));
            return f;
        };
        const auto chunk = [&](Json delta_or_text, const Json & finish) {
            Json f = frame(chat ? "chat.completion.chunk" : "text_completion");
            Json choice = Json::object();
            choice.set("index", num(0));
            if (chat) {
                choice.set("delta", std::move(delta_or_text));
            } else {
                choice.set("text", std::move(delta_or_text));
            }
            choice.set("logprobs", Json());
            choice.set("finish_reason", finish);
            Json choices = Json::array();
            choices.push(std::move(choice));
            f.set("choices", std::move(choices));
            return f;
        };
        const auto call_json = [&](const omph::server::ToolCall & call, int index, bool streamed) {
            Json fn = Json::object();
            fn.set("name", str(call.name));
            fn.set("arguments", str(call.arguments));
            Json j = Json::object();
            if (streamed) j.set("index", num(index));
            j.set("id", str(call.id));
            j.set("type", str("function"));
            j.set("function", std::move(fn));
            return j;
        };

        omph::server::OutputParser parser(job);
        std::string reasoning, content, echo;
        std::vector<omph::server::ToolCall> calls;
        if (job.echo) {
            // a decode of arbitrary ids can end mid-character: the response must stay JSON (#338)
            echo = omph::server::sanitize_utf8(job.prompt_ids.empty() ? job.prompt
                                                                     : tok.decode(job.prompt_ids, false));
        }
        bool gone = false;         // the client closed the stream
        int n_streamed_calls = 0;
        // every SSE payload is a JSON body too: the same UTF-8 guarantee
        const auto event = [&c](const Json & body) {
            return c.event(omph::server::sanitize_utf8(body.dump()));
        };
        const auto send = [&](const Delta & d) {
            if (!job.stream) {
                reasoning += d.reasoning;
                content += d.content;
                calls.insert(calls.end(), d.calls.begin(), d.calls.end());
                return;
            }
            if (gone || d.empty()) return;
            if (!chat) {
                gone = !event(chunk(str(d.content), Json()));
                return;
            }
            if (!d.reasoning.empty()) {
                Json delta = Json::object();
                delta.set("reasoning_content", str(d.reasoning));
                gone = gone || !event(chunk(std::move(delta), Json()));
            }
            if (!d.content.empty()) {
                Json delta = Json::object();
                delta.set("content", str(d.content));
                gone = gone || !event(chunk(std::move(delta), Json()));
            }
            for (const auto & call : d.calls) {
                Json delta = Json::object();
                Json arr = Json::array();
                arr.push(call_json(call, n_streamed_calls++, true));
                delta.set("tool_calls", std::move(arr));
                gone = gone || !event(chunk(std::move(delta), Json()));
            }
        };
        if (job.stream) {
            gone = !c.begin_events();
            if (!gone && chat) {
                Json delta = Json::object();
                delta.set("role", str("assistant"));
                delta.set("content", str(""));
                gone = !event(chunk(std::move(delta), Json()));
            } else if (!gone && !echo.empty()) {
                gone = !event(chunk(str(echo), Json()));
            }
        }
        GenerateResult res;
        if (!gone) {
            // a progress line every 3 s of decoding, as llama-server's (#231)
            double t_first = 0.0, t_last = 0.0;
            int64_t n = 0, n_last = 0;
            res = gen.generate(prompt, greq, [&](const int32_t t) {
                if (!client_gone && c.client_gone()) {
                    client_gone = true;  // a non-streamed request has no send() to fail (#338)
                }
                const double now = omph::runtime::now_ms();
                if (n++ == 0) t_first = t_last = now;
                if (now - t_last >= 3000.0) {
                    const GenerateResult & r = gen.running();
                    std::fprintf(stderr, "  %lld tokens, %.1f t/s (last 3 s: %.1f t/s)", (long long) n,
                                 1000.0 * (double) (n - 1) / (now - t_first),
                                 1000.0 * (double) (n - n_last) / (now - t_last));
                    if (r.drafted > 0) {
                        std::fprintf(stderr, ", drafts accepted %.0f %%", 100.0 * (double) r.accepted / (double) r.drafted);
                    }
                    std::fprintf(stderr, "\n");
                    t_last = now;
                    n_last = n;
                }
                send(parser.push(tok.piece(t, false)));
                return !gone && !client_gone && !parser.stopped();
            });
            send(parser.finish());
        }
        const bool error = res.stop == GenerateResult::Stop::Error;
        const bool length = !parser.stopped() &&
                            (res.stop == GenerateResult::Stop::Length || res.stop == GenerateResult::Stop::ContextFull);
        const Json finish = str(omph::server::finish_reason(length, parser.any_tool_call()));
        const auto completion_tokens = (double) res.tokens.size();
        Json usage = Json::object();
        usage.set("prompt_tokens", num((double) prompt_len));
        usage.set("completion_tokens", num(completion_tokens));
        usage.set("total_tokens", num((double) prompt_len + completion_tokens));
        Json details = Json::object();
        details.set("cached_tokens", num((double) res.cached_tokens));
        usage.set("prompt_tokens_details", std::move(details));
        // llama.cpp's timings, for clients that show them
        Json timings = Json::object();
        timings.set("prompt_n", num((double) (res.prompt_tokens - res.cached_tokens)));
        timings.set("prompt_ms", Json::number(res.prefill_ms, false));
        timings.set("predicted_n", num(completion_tokens));
        timings.set("predicted_ms", Json::number(res.decode_ms, false));
        timings.set("draft_n", num((double) res.drafted));
        timings.set("draft_n_accepted", num((double) res.accepted));

        if (job.stream) {
            if (!gone) {
                if (error) {
                    event(omph::server::error_body("generation failed", "server_error"));
                } else {
                    Json last = chunk(chat ? Json::object() : str(""), finish);
                    last.set("timings", timings);
                    event(last);
                    if (job.include_usage) {
                        Json u = frame(chat ? "chat.completion.chunk" : "text_completion");
                        u.set("choices", Json::array());
                        u.set("usage", usage);
                        event(u);
                    }
                }
                c.event("[DONE]");
            }
        } else if (error) {
            fail(c, 500, "generation failed", "server_error");
        } else {
            Json r = frame(chat ? "chat.completion" : "text_completion");
            Json choice = Json::object();
            choice.set("index", num(0));
            if (chat) {
                Json msg = Json::object();
                msg.set("role", str("assistant"));
                msg.set("content", content.empty() && !calls.empty() ? Json() : str(content));
                if (!reasoning.empty()) msg.set("reasoning_content", str(reasoning));
                if (!calls.empty()) {
                    Json arr = Json::array();
                    for (size_t i = 0; i < calls.size(); ++i) arr.push(call_json(calls[i], (int) i, false));
                    msg.set("tool_calls", std::move(arr));
                }
                choice.set("message", std::move(msg));
            } else {
                choice.set("text", str(echo + content));
            }
            choice.set("logprobs", Json());
            choice.set("finish_reason", finish);
            Json choices = Json::array();
            choices.push(std::move(choice));
            r.set("choices", std::move(choices));
            r.set("usage", std::move(usage));
            r.set("timings", std::move(timings));
            reply(c, 200, r);
        }
        static const char * kStop[] = {"length", "end of generation", "stop token", "stopped", "context full",
                                       "error"};
        std::fprintf(stderr,
                     "%s %s: sampling %s; prompt %zu tokens (%lld cached%s%s) in %.0f ms "
                     "(checkpoints %.0f ms, %.1f t/s); %zu "
                     "tokens in %.0f ms "
                     "(%.1f t/s, drafts accepted %lld / %lld); stop: %s%s\n",
                     req.method.c_str(), req.path.c_str(), sampling_label(job).c_str(), (size_t) prompt_len,
                     (long long) res.cached_tokens,
                     res.restored ? ", restored" : "", res.saved ? ", previous conversation saved" : "",
                     res.prefill_ms,
                     res.checkpoint_ms,
                     res.prefill_ms > 0 ? 1000.0 * (double) (res.prompt_tokens - res.cached_tokens) / res.prefill_ms
                                        : 0.0,
                     res.tokens.size(), res.decode_ms,
                     res.decode_ms > 0 ? 1000.0 * completion_tokens / res.decode_ms : 0.0,
                     (long long) res.accepted, (long long) res.drafted,
                     parser.stopped() ? "stop string" : kStop[(int) res.stop],
                     (gone || client_gone) ? " (client gone)" : "");
    }
};

std::string default_id(const std::string & path) {
    std::string name = path.substr(path.find_last_of('/') + 1);
    if (name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) name.resize(name.size() - 5);
    return name;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [options]\n", argv[0]);
        return 2;
    }
    omph::model::Generator::Config cfg;
    cfg.model = argv[1];
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string alias, api_key, cors, mmproj;
    omph::server::Defaults defaults;
    for (int i = 2; i < argc; ++i) {
        const auto val = [&]() -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", argv[i]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--host")) host = val();
        else if (!std::strcmp(argv[i], "--port")) port = omph::cli::integer(val(), "--port");
        else if (!std::strcmp(argv[i], "--ctx")) cfg.context = omph::cli::integer(val(), "--ctx");
        else if (!std::strcmp(argv[i], "--chunk")) cfg.chunk = omph::cli::integer(val(), "--chunk");
        else if (!std::strcmp(argv[i], "--no-mtp")) cfg.mtp = false;
        else if (!std::strcmp(argv[i], "--dflash")) cfg.dflash = val();
        else if (!std::strcmp(argv[i], "--cache-ram")) cfg.cache_mib = omph::cli::integer(val(), "--cache-ram");
        else if (!std::strcmp(argv[i], "--kv-ram")) cfg.kv_ram_mib = omph::cli::integer(val(), "--kv-ram");
        else if (!std::strcmp(argv[i], "--alias")) alias = val();
        else if (!std::strcmp(argv[i], "--mmproj")) mmproj = val();
        else if (!std::strcmp(argv[i], "--api-key")) api_key = val();
        else if (!std::strcmp(argv[i], "--cors")) cors = val();
        else if (!std::strcmp(argv[i], "--temp")) defaults.temperature = (float) omph::cli::number(val(), "--temp");
        else if (!std::strcmp(argv[i], "--top-k")) defaults.top_k = (int) omph::cli::integer(val(), "--top-k");
        else if (!std::strcmp(argv[i], "--top-p")) defaults.top_p = (float) omph::cli::number(val(), "--top-p");
        else if (!std::strcmp(argv[i], "--min-p")) defaults.min_p = (float) omph::cli::number(val(), "--min-p");
        else if (!std::strcmp(argv[i], "--repeat-penalty")) defaults.repeat_penalty = (float) omph::cli::number(val(), "--repeat-penalty");
        else if (!std::strcmp(argv[i], "--repeat-last-n")) defaults.penalty_last_n = (int) omph::cli::integer(val(), "--repeat-last-n");
        else if (!std::strcmp(argv[i], "--frequency-penalty")) defaults.frequency_penalty = (float) omph::cli::number(val(), "--frequency-penalty");
        else if (!std::strcmp(argv[i], "--presence-penalty")) defaults.presence_penalty = (float) omph::cli::number(val(), "--presence-penalty");
        else if (!std::strcmp(argv[i], "--max-tokens")) defaults.max_tokens = omph::cli::integer(val(), "--max-tokens");
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    try {
        omph::server::Listener listener(host, port);  // fail before the minute of loading
        omph::model::Generator gen(cfg, omph::runtime::EnvOptions::from_env());
        Server server{gen, alias.empty() ? default_id(cfg.model) : alias, host, port, api_key, cors, defaults};
#ifdef OMPH_VISION
        std::unique_ptr<omph::vision::Encoder> vision;
        if (!mmproj.empty()) {
            vision = std::make_unique<omph::vision::Encoder>(mmproj, cfg.model);
            server.vision = vision.get();
        }
#else
        if (!mmproj.empty()) throw std::runtime_error("--mmproj: built without vision (OMPH_LLAMA_DIR)");
#endif
        std::fprintf(stderr, "omph-server: %s on http://%s:%d (context %lld)\n", server.model_id.c_str(),
                     host.c_str(), port, (long long) gen.context());
        while (true) {
            const int fd = listener.accept_one();
            if (fd < 0) {
                std::perror("accept");
                continue;
            }
            omph::server::Connection conn(fd);
            omph::server::Request req;
            if (!conn.read(req)) continue;
            try {
                server.handle(conn, req);
            } catch (const std::exception & e) {  // the next request still gets served
                std::fprintf(stderr, "%s %s: %s\n", req.method.c_str(), req.path.c_str(), e.what());
                server.fail(conn, 500, e.what(), "server_error");
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
