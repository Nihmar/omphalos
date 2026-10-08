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
//   --alias NAME      the model id in the API (default: the file name without .omph)
//   --api-key KEY     require "Authorization: Bearer KEY"
//   --cors ORIGIN     allow browser requests from ORIGIN (e.g. "*")
//   --tools LIST      llama.cpp's server-side agent tools the web UI lists:
//                     read_file, file_glob_search, grep_search,
//                     exec_shell_command, write_file, edit_file, get_info, or
//                     "all". They run with this process's permissions (files,
//                     shell): enable only where that is trusted (#380)
//   --agent           --tools all, llama.cpp's shortcut
//   --log-json        one JSON object per line on stderr for the ready line,
//                     the requests' start and progress and the summary,
//                     instead of the human lines (#304; the TUI reads both)
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
#include "server/tools.hh"
#include "server/webui.hh"
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
    std::string model_path;
    std::string host;
    int port = 0;
    std::string api_key;
    std::string cors;
    omph::server::Defaults defaults;
    // llama.cpp's server-side agent tools the web UI lists (#380): the names
    // from --tools/--agent, empty when off (the route then answers 403).
    std::vector<std::string> tools;
    bool log_json = false;
    int64_t request_seq = 0;  // --log-json: correlates a request's three events
#ifdef OMPH_VISION
    omph::vision::Encoder * vision = nullptr;
#endif

    static Json num(const double v) { return Json::number(v, true); }
    static Json str(const std::string & s) { return Json::string(s); }

    // --log-json (#304): the events of tools/tui/log.py, one JSON object per
    // line on stderr, replacing the human line of the same event. Everything
    // else (a startup error, the ablation banner, the images' line) stays
    // human, and the TUI passes it through.
    static Json event_object(const char * name, const int64_t id) {
        Json e = Json::object();
        e.set("event", str(name));
        e.set("id", Json::integer(id));
        return e;
    }
    void log_event(const Json & e) const {
        if (log_json) {
            std::fprintf(stderr, "%s\n", e.dump().c_str());
        }
    }

    bool reply(omph::server::Connection & c, const int status, const Json & body) {
        // every string that reaches a response is valid UTF-8, whatever byte
        // sequence a tokenizer or a JSON escape put in it (#338)
        return c.respond(status, "application/json", omph::server::sanitize_utf8(body.dump()));
    }

    // One web UI asset (#378): the resolution webui::resolve made, turned into
    // the response llama.cpp sends (ETag, cache policy, gzip, COEP/COOP).
    static bool serve_webui(omph::server::Connection & c,
                            const omph::server::webui::Resolution & res) {
        using S = omph::server::webui::Resolution::Status;
        if (res.status == S::GzipRequired) {
            c.extra_headers.emplace_back("Vary", "Accept-Encoding");
            return c.respond(415, "text/plain", "Error: gzip is not supported by this browser\n");
        }
        const omph::server::webui::Asset & a = *res.asset;
        c.extra_headers.emplace_back("ETag", "\"" + std::string(a.etag) + "\"");
        if (res.status == S::NotModified) {
            return c.respond(304, "", "");
        }
        c.extra_headers.emplace_back("Cache-Control", res.cache_control);
        if (a.gzip) {
            c.extra_headers.emplace_back("Vary", "Accept-Encoding");
            c.extra_headers.emplace_back("Content-Encoding", "gzip");
        }
        if (res.isolate) {
            c.extra_headers.emplace_back("Cross-Origin-Embedder-Policy", "require-corp");
            c.extra_headers.emplace_back("Cross-Origin-Opener-Policy", "same-origin");
        }
        return c.respond(200, std::string(a.type),
                         std::string(reinterpret_cast<const char *>(a.data), a.size));
    }

    // llama.cpp's timings object (server-common.cpp), the fields its web UI
    // reads for the live statistics (#382): the running counters during the
    // stream and the final numbers in the last chunk.
    static Json timings_json(const GenerateResult & r) {
        const double prompt_n = (double) (r.prompt_tokens - r.cached_tokens);
        const double pred_n = (double) r.tokens.size();
        Json t = Json::object();
        t.set("cache_n", Json::integer(r.cached_tokens));
        t.set("prompt_n", Json::integer((int64_t) prompt_n));
        t.set("prompt_ms", Json::number(r.prefill_ms, false));
        if (prompt_n > 0 && r.prefill_ms > 0) {
            t.set("prompt_per_token_ms", Json::number(r.prefill_ms / prompt_n, false));
            t.set("prompt_per_second", Json::number(1000.0 * prompt_n / r.prefill_ms, false));
        }
        t.set("predicted_n", Json::integer((int64_t) pred_n));
        t.set("predicted_ms", Json::number(r.decode_ms, false));
        if (pred_n > 0 && r.decode_ms > 0) {
            t.set("predicted_per_token_ms", Json::number(r.decode_ms / pred_n, false));
            t.set("predicted_per_second", Json::number(1000.0 * pred_n / r.decode_ms, false));
        }
        t.set("draft_n", Json::integer(r.drafted));
        t.set("draft_n_accepted", Json::integer(r.accepted));
        return t;
    }

    // llama.cpp's /props shape (tools/server/server-context.cpp: get_res_props),
    // the fields its web UI reads (#378).
    Json props_json() const {
        const omph::text::Tokenizer & tok = gen.tokenizer();
        Json params = Json::object();
        params.set("temperature", Json::number((double) defaults.temperature, false));
        params.set("top_k", Json::integer(defaults.top_k));
        params.set("top_p", Json::number((double) defaults.top_p, false));
        params.set("min_p", Json::number((double) defaults.min_p, false));
        params.set("seed", Json::integer(-1));  // -1: drawn per request
        params.set("n_predict", Json::integer(defaults.max_tokens));
        params.set("max_tokens", Json::integer(defaults.max_tokens));
        params.set("stream", Json::boolean(false));
        params.set("repeat_last_n", Json::integer(defaults.penalty_last_n));
        params.set("repeat_penalty", Json::number((double) defaults.repeat_penalty, false));
        params.set("presence_penalty", Json::number((double) defaults.presence_penalty, false));
        params.set("frequency_penalty", Json::number((double) defaults.frequency_penalty, false));
        Json dgs = Json::object();
        dgs.set("params", std::move(params));
        dgs.set("n_ctx", Json::integer(gen.context()));
        Json props = Json::object();
        props.set("default_generation_settings", std::move(dgs));
        props.set("total_slots", Json::integer(1));
        props.set("model_alias", str(model_id));
        props.set("model_path", str(model_path));
        Json modalities = Json::object();
#ifdef OMPH_VISION
        modalities.set("vision", Json::boolean(vision != nullptr));
#else
        modalities.set("vision", Json::boolean(false));
#endif
        modalities.set("video", Json::boolean(false));
        modalities.set("audio", Json::boolean(false));
        props.set("modalities", std::move(modalities));
        props.set("media_marker", str("<__media__>"));
        props.set("endpoint_slots", Json::boolean(true));
        props.set("endpoint_props", Json::boolean(false));   // no POST /props
        props.set("endpoint_metrics", Json::boolean(false));  // no /metrics
        props.set("ui", Json::boolean(!omph::server::webui::assets().empty()));
        props.set("ui_settings", Json::object());
        props.set("chat_template", str(gen.chat_template()));
        props.set("bos_token", str(tok.piece(tok.bos(), true)));
        props.set("eos_token", str(tok.piece(tok.eos(), true)));
        props.set("build_info", str("omphalos"));
        props.set("is_sleeping", Json::boolean(false));
        props.set("cors_proxy_enabled", Json::boolean(false));
        return props;
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
        // The web UI (#378): llama.cpp's static bundle, one exact route per
        // embedded file, served before the API key so the page itself is what
        // asks the client for the key (llama-server does the same).
        if (req.method == "GET") {
            const auto web = omph::server::webui::resolve(
                path, req.header("if-none-match") != nullptr ? *req.header("if-none-match") : "",
                req.header("accept-encoding") != nullptr ? *req.header("accept-encoding") : "");
            if (web.status != omph::server::webui::Resolution::Status::NotFound) {
                serve_webui(c, web);
                return;
            }
        }
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
        if (path == "/props" && req.method == "GET") {  // the web UI (#378)
            reply(c, 200, props_json());
            return;
        }
        if (path == "/slots" && req.method == "GET") {
            Json arr = Json::array();
            Json slot = Json::object();
            slot.set("id", Json::integer(0));
            slot.set("n_ctx", Json::integer(gen.context()));
            slot.set("speculative", Json::boolean(true));
            slot.set("is_processing", Json::boolean(false));
            arr.push(std::move(slot));
            reply(c, 200, arr);
            return;
        }
        // The web UI's tool page (#380): the list and the invocations, in
        // llama.cpp's shapes. Off unless --tools/--agent enabled them.
        if (path == "/tools") {
            if (tools.empty()) {
                fail(c, 403, "server tools are not enabled (start omph-server with --tools all)",
                     "not_supported_error");
                return;
            }
            if (req.method == "GET") {
                reply(c, 200, omph::server::tools::list(tools));
                return;
            }
            if (req.method != "POST") {
                fail(c, 405, "use GET or POST");
                return;
            }
            Json body;
            try {
                body = Json::parse(req.body);
            } catch (const std::exception & e) {
                fail(c, 400, std::string("invalid JSON body: ") + e.what());
                return;
            }
            const Json & tool = body.get("tool");
            if (!tool.is_string()) {
                fail(c, 400, "tool is required and must be a string", "invalid_request_error", "tool");
                return;
            }
            const std::string * tool_cwd = req.header("x-tool-cwd");
            try {
                reply(c, 200, omph::server::tools::invoke(tool.as_string(), body.get("params"),
                                                          tool_cwd != nullptr ? *tool_cwd : ""));
            } catch (const std::exception & e) {
                fail(c, 500, e.what(), "server_error");
            }
            return;
        }
        // llama-server routes omph-server does not implement: answer a 501 in
        // the OpenAI shape (the UI's optional features hide behind the
        // endpoint_* flags of /props and fail loudly if ever called).
        for (const char * p : {"/metrics", "/models/load", "/models/unload", "/models/sse", "/stream",
                               "/streams/lookup", "/chat/completions/control"}) {
            if (path == p) {
                fail(c, 501, std::string("omph-server does not implement ") + p + " (llama.cpp's route)",
                     "not_supported_error");
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
        const int64_t req_id = ++request_seq;
        if (log_json) {
            Json e = event_object("request_start", req_id);
            e.set("method", str(req.method));
            e.set("path", str(req.path));
            log_event(e);
        }
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
        double last_write = omph::runtime::now_ms();
        // every SSE payload is a JSON body too: the same UTF-8 guarantee
        const auto event = [&c, &last_write](const Json & body) {
            last_write = omph::runtime::now_ms();
            return c.event(omph::server::sanitize_utf8(body.dump()));
        };
        bool in_generate = false;
        // llama.cpp's per-chunk `timings` (#382): the web UI's live statistics
        // read it from every chunk while a reply is being written.
        const auto timed = [&](Json body) {
            if (job.timings_per_token && in_generate) {
                body.set("timings", timings_json(gen.running()));
            }
            return body;
        };
        // llama.cpp's `sse_ping_interval`: a comment keeps a silent stream
        // (a long prefill) from being dropped by an impatient client. Only a
        // streamed response: on a non-streamed one the ping would be written
        // before the HTTP status line (#390).
        const auto ping_if_due = [&]() {
            if (job.stream && job.sse_ping_interval > 0 && !gone &&
                omph::runtime::now_ms() - last_write >= job.sse_ping_interval * 1000.0) {
                gone = !c.ping();
                last_write = omph::runtime::now_ms();
            }
        };
        const auto send = [&](const Delta & d) {
            ping_if_due();
            if (!job.stream) {
                reasoning += d.reasoning;
                content += d.content;
                calls.insert(calls.end(), d.calls.begin(), d.calls.end());
                return;
            }
            if (gone || d.empty()) return;
            if (!chat) {
                gone = !event(timed(chunk(str(d.content), Json())));
                return;
            }
            if (!d.reasoning.empty()) {
                Json delta = Json::object();
                delta.set("reasoning_content", str(d.reasoning));
                gone = gone || !event(timed(chunk(std::move(delta), Json())));
            }
            if (!d.content.empty()) {
                Json delta = Json::object();
                delta.set("content", str(d.content));
                gone = gone || !event(timed(chunk(std::move(delta), Json())));
            }
            for (const auto & call : d.calls) {
                Json delta = Json::object();
                Json arr = Json::array();
                arr.push(call_json(call, n_streamed_calls++, true));
                delta.set("tool_calls", std::move(arr));
                gone = gone || !event(timed(chunk(std::move(delta), Json())));
            }
        };
        if (job.stream) {
            gone = !c.begin_events();
            if (!gone && chat) {
                Json delta = Json::object();
                delta.set("role", str("assistant"));
                delta.set("content", str(""));
                gone = !event(chunk(std::move(delta), Json()));  // no timings yet: the prefill starts now
            } else if (!gone && !echo.empty()) {
                gone = !event(chunk(str(echo), Json()));
            }
        }
        // The prefill's progress (#382): llama.cpp's `prompt_progress`, one
        // chunk per prefill chunk, on top of the stderr line of #310.
        {
            const double t0 = omph::runtime::now_ms();
            double t_last = t0;
            greq.on_prefill = [this, &c, &client_gone, &gone, &chunk, &event, &timed, &ping_if_due, &job, chat,
                               req_id, t0, t_last](const int64_t done, const int64_t total) mutable {
                if (c.client_gone()) {  // the abandoned request stops here, not an hour later (#338)
                    client_gone = true;
                    return false;
                }
                const double now = omph::runtime::now_ms();
                const bool report = done >= total || now - t_last >= 3000.0;
                if (report) {
                    t_last = now;
                    const double t_s = done > 0 && now > t0 ? 1000.0 * (double) done / (now - t0) : 0.0;
                    if (log_json) {
                        Json e = event_object("progress", req_id);
                        e.set("phase", str("prefill"));
                        e.set("tokens", Json::integer(done));
                        e.set("total", Json::integer(total));
                        e.set("t_s", num(t_s));
                        log_event(e);
                    } else {
                        std::fprintf(stderr, "  prefill %lld / %lld tokens, %.1f t/s\n", (long long) done,
                                     (long long) total, t_s);
                    }
                }
                if (job.stream && !gone && (job.return_progress || job.timings_per_token)) {
                    Json body = timed(chunk(chat ? Json::object() : str(""), Json()));
                    if (job.return_progress) {
                        Json progress = Json::object();
                        progress.set("total", Json::integer(total));
                        progress.set("cache", Json::integer(gen.running().cached_tokens));
                        progress.set("processed", Json::integer(done));
                        progress.set("time_ms", Json::number(now - t0, false));
                        body.set("prompt_progress", std::move(progress));
                    }
                    gone = gone || !event(body);
                }
                ping_if_due();
                return true;
            };
        }

        GenerateResult res;
        if (!gone) {
            in_generate = true;
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
                    const double t_s = 1000.0 * (double) (n - 1) / (now - t_first);
                    const double last_s = 1000.0 * (double) (n - n_last) / (now - t_last);
                    if (log_json) {
                        Json e = event_object("progress", req_id);
                        e.set("phase", str("decode"));
                        e.set("tokens", Json::integer(n));
                        e.set("t_s", num(t_s));
                        e.set("last_t_s", num(last_s));
                        if (r.drafted > 0) {
                            e.set("drafted", Json::integer(r.drafted));
                            e.set("accepted", Json::integer(r.accepted));
                        }
                        log_event(e);
                    } else {
                        std::fprintf(stderr, "  %lld tokens, %.1f t/s (last 3 s: %.1f t/s)", (long long) n, t_s,
                                     last_s);
                        if (r.drafted > 0) {
                            std::fprintf(stderr, ", drafts accepted %.0f %%",
                                         100.0 * (double) r.accepted / (double) r.drafted);
                        }
                        std::fprintf(stderr, "\n");
                    }
                    t_last = now;
                    n_last = n;
                }
                send(parser.push(tok.piece(t, false)));
                return !gone && !client_gone && !parser.stopped();
            });
            in_generate = false;
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
        Json timings = timings_json(res);

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
        const char * stop = parser.stopped() ? "stop string" : kStop[(int) res.stop];
        const bool client_left = gone || client_gone;
        if (log_json) {
            Json e = event_object("request", req_id);
            e.set("method", str(req.method));
            e.set("path", str(req.path));
            e.set("sampling_line", str(sampling_label(job)));
            e.set("prompt_tokens", Json::integer(prompt_len));
            e.set("cached_tokens", Json::integer(res.cached_tokens));
            e.set("restored", Json::boolean(res.restored));
            e.set("saved", Json::boolean(res.saved));
            e.set("prefill_ms", num(res.prefill_ms));
            e.set("checkpoint_ms", num(res.checkpoint_ms));
            e.set("completion_tokens", Json::integer((int64_t) res.tokens.size()));
            e.set("decode_ms", num(res.decode_ms));
            e.set("drafts", Json::integer(res.drafted));
            e.set("accepted", Json::integer(res.accepted));
            e.set("stop", str(stop));
            e.set("client_gone", Json::boolean(client_left));
            log_event(e);
        } else {
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
                         res.prefill_ms > 0
                             ? 1000.0 * (double) (res.prompt_tokens - res.cached_tokens) / res.prefill_ms
                             : 0.0,
                         res.tokens.size(), res.decode_ms,
                         res.decode_ms > 0 ? 1000.0 * completion_tokens / res.decode_ms : 0.0,
                         (long long) res.accepted, (long long) res.drafted, stop, client_left ? " (client gone)" : "");
        }
    }
};

std::string default_id(const std::string & path) {
    std::string name = path.substr(path.find_last_of('/') + 1);
    // the server loads the .omph omph-convert writes (#178): its id lost the
    // conversion suffix only for .gguf and kept ".omph" (#364)
    for (const char * suffix : {".omph", ".gguf"}) {
        const size_t n = std::strlen(suffix);
        if (name.size() > n && name.compare(name.size() - n, n, suffix) == 0) {
            name.resize(name.size() - n);
            break;
        }
    }
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
    std::string alias, api_key, cors, mmproj, tools_spec;
    bool log_json = false;
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
        else if (!std::strcmp(argv[i], "--tools")) tools_spec = val();
        else if (!std::strcmp(argv[i], "--agent")) tools_spec = "all";  // llama.cpp's shortcut
        else if (!std::strcmp(argv[i], "--log-json")) log_json = true;
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
    std::vector<std::string> enabled_tools;
    try {
        enabled_tools = omph::server::tools::parse(tools_spec);
    } catch (const std::exception & e) {  // an unknown --tools name, before the model loads
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    try {
        omph::server::Listener listener(host, port);  // fail before the minute of loading
        const double t_load = omph::runtime::now_ms();
        omph::model::Generator gen(cfg, omph::runtime::EnvOptions::from_env());
        Server server{gen, alias.empty() ? default_id(cfg.model) : alias, cfg.model, host, port, api_key, cors,
                      defaults, enabled_tools};
        server.log_json = log_json;
#ifdef OMPH_VISION
        std::unique_ptr<omph::vision::Encoder> vision;
        if (!mmproj.empty()) {
            vision = std::make_unique<omph::vision::Encoder>(mmproj, cfg.model);
            server.vision = vision.get();
        }
#else
        if (!mmproj.empty()) throw std::runtime_error("--mmproj: built without vision (OMPH_LLAMA_DIR)");
#endif
        if (log_json) {
            Json e = Json::object();
            e.set("event", Server::str("ready"));
            e.set("model", Server::str(server.model_id));
            e.set("host", Server::str(host));
            e.set("port", Server::num(port));
            e.set("context", Json::integer(gen.context()));
            e.set("load_ms", Server::num(omph::runtime::now_ms() - t_load));
            server.log_event(e);
        } else {
            std::fprintf(stderr, "omph-server: %s on http://%s:%d (context %lld)\n", server.model_id.c_str(),
                         host.c_str(), port, (long long) gen.context());
        }
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
                if (log_json) {
                    Json j = Json::object();
                    j.set("event", Server::str("http"));
                    j.set("status", Server::num(500));
                    j.set("method", Server::str(req.method));
                    j.set("path", Server::str(req.path));
                    j.set("error", Server::str(e.what()));
                    server.log_event(j);
                } else {
                    std::fprintf(stderr, "%s %s: %s\n", req.method.c_str(), req.path.c_str(), e.what());
                }
                if (conn.streaming()) {
                    // The event stream's headers are out: an error event and the
                    // end of the stream, never a second HTTP response in it (#337).
                    conn.event(omph::server::sanitize_utf8(
                        omph::server::error_body(e.what(), "server_error").dump()));
                    conn.event("[DONE]");
                } else {
                    server.fail(conn, 500, e.what(), "server_error");
                }
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
