#include "server/openai.hh"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include "text/chat.hh"

namespace omph::server {
namespace {

using text::Json;

[[noreturn]] void bad(const std::string & message, const std::string & param = "") {
    throw BadRequest{message, param};
}

bool is_space(const char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// the longest proper prefix of `tag` that `s` ends with
size_t partial_suffix(const std::string & s, const std::string & tag) {
    for (size_t n = std::min(s.size(), tag.size() - 1); n > 0; --n) {
        if (s.compare(s.size() - n, n, tag, 0, n) == 0) {
            return n;
        }
    }
    return 0;
}

// invalid UTF-8 (a byte-level token sequence that never completes a
// character) to U+FFFD, so that responses stay valid JSON text
std::string sanitize_utf8(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const auto b = (uint8_t) s[i];
        size_t len = b < 0x80 ? 1 : (b >> 5) == 0x6 ? 2 : (b >> 4) == 0xE ? 3 : (b >> 3) == 0x1E ? 4 : 0;
        bool ok = len > 0 && i + len <= s.size();
        for (size_t k = 1; ok && k < len; ++k) {
            ok = ((uint8_t) s[i + k] & 0xC0) == 0x80;
        }
        if (ok && len > 1) {  // overlong forms and surrogates
            uint32_t cp = b & (0x7F >> len);
            for (size_t k = 1; k < len; ++k) cp = (cp << 6) | ((uint8_t) s[i + k] & 0x3F);
            static const uint32_t kMin[5] = {0, 0, 0x80, 0x800, 0x10000};
            ok = cp >= kMin[len] && cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF);
        }
        if (ok) {
            out.append(s, i, len);
            i += len;
        } else {
            out += "\xEF\xBF\xBD";
            ++i;
        }
    }
    return out;
}

double number(const Json & body, const char * key, double lo, double hi, double fallback) {
    const Json * v = body.find(key);
    if (v == nullptr || v->is_null()) return fallback;
    if (!v->is_number() || v->as_number() < lo || v->as_number() > hi) {
        bad(std::string(key) + " must be a number in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]",
            key);
    }
    return v->as_number();
}

bool flag(const Json & body, const char * key, bool fallback) {
    const Json * v = body.find(key);
    if (v == nullptr || v->is_null()) return fallback;
    if (!v->is_bool()) bad(std::string(key) + " must be a boolean", key);
    return v->as_bool();
}

// OpenAI's reasoning_effort to the template's (xhigh / medium / low), or
// "off" for "none" / "minimal"
std::string effort(const Json & v) {
    if (!v.is_string()) bad("reasoning_effort must be a string", "reasoning_effort");
    const std::string & e = v.as_string();
    if (e == "none" || e == "minimal") return "off";
    if (e == "low" || e == "medium" || e == "xhigh") return e;
    if (e == "high") return "xhigh";
    bad("unsupported reasoning_effort " + e, "reasoning_effort");
}

// A message's images, in order, appended to `images` as file bytes: only
// data: URLs (the server fetches nothing and reads no local file). Audio,
// video and files are refused.
void collect_images(const Json & content, std::vector<std::string> & images) {
    if (!content.is_array()) return;
    for (const Json & item : content.items()) {
        const Json & type = item.get("type");
        const std::string t = type.is_string() ? type.as_string() : "";
        if (t == "input_audio" || t == "video" || t == "file" || t == "video_url" || item.has("video")) {
            bad("audio, video and file inputs are not supported", "messages");
        }
        if (!(t == "image_url" || t == "image" || item.has("image_url") || item.has("image"))) continue;
        const Json & iu = item.has("image_url") ? item.get("image_url") : item.get("image");
        const Json & url = iu.is_object() ? iu.get("url") : iu;
        const std::string u = url.is_string() ? url.as_string() : "";
        const size_t comma = u.find(',');
        if (u.rfind("data:", 0) != 0 || comma == std::string::npos || comma < 12 ||
            u.compare(comma - 7, 7, ";base64") != 0) {
            bad("images must be base64 data: URLs (data:image/png;base64,...)", "messages");
        }
        std::string bytes;
        if (!base64_decode(u.substr(comma + 1), bytes) || bytes.empty()) {
            bad("an image's base64 data is invalid", "messages");
        }
        images.push_back(std::move(bytes));
    }
}

void parse_sampling(const Json & body, const Defaults & d, Job & job) {
    job.temperature = (float) number(body, "temperature", 0.0, 2.0, d.temperature);
    job.top_p = (float) number(body, "top_p", 0.0, 1.0, d.top_p);
    if (job.top_p <= 0.0f) job.top_p = 1.0f;
    const double k = number(body, "top_k", -1.0, 1e9, d.top_k);
    job.top_k = k < 0 ? 0 : (int) k;
    job.min_p = (float) number(body, "min_p", 0.0, 1.0, d.min_p);
    // Sampling fields the engine does not implement are refused, not dropped
    // in silence (#284): a client that asks for a repetition penalty to stop
    // a loop would otherwise never learn that nothing happened. A no-op value
    // (0, 1.0, or an empty bias) is accepted, so clients that always send the
    // field are unaffected.
    const auto unimplemented = [&](const char * key, const double no_op, const char * no_op_text) {
        const Json * v = body.find(key);
        if (v == nullptr || v->is_null()) {
            return;
        }
        if (!v->is_number() || v->as_number() != no_op) {
            bad(std::string(key) + " is not implemented: only " + no_op_text + " is accepted", key);
        }
    };
    unimplemented("frequency_penalty", 0.0, "0");
    unimplemented("presence_penalty", 0.0, "0");
    unimplemented("repetition_penalty", 1.0, "1.0");
    unimplemented("repeat_penalty", 1.0, "1.0");
    if (const Json * bias = body.find("logit_bias"); bias != nullptr && !bias->is_null() &&
        !(bias->is_object() && bias->size() == 0)) {
        bad("logit_bias is not implemented: omit it, or send {}", "logit_bias");
    }
    const Json * seed = body.find("seed");
    if (seed != nullptr && !seed->is_null()) {
        if (!seed->is_number() || !seed->number_is_integer()) bad("seed must be an integer", "seed");
        job.seeded = true;
        job.seed = (uint64_t) (int64_t) seed->as_number();
    }
    job.max_tokens = d.max_tokens;
    for (const char * key : {"max_tokens", "max_completion_tokens"}) {
        const Json * v = body.find(key);
        if (v == nullptr || v->is_null()) continue;
        if (!v->is_number() || !v->number_is_integer() || v->as_number() < 1) {
            bad(std::string(key) + " must be a positive integer", key);
        }
        job.max_tokens = (int64_t) v->as_number();
    }
    const Json & stop = body.get("stop");
    if (stop.is_string()) {
        job.stop.push_back(stop.as_string());
    } else if (stop.is_array()) {
        for (const Json & s : stop.items()) {
            if (!s.is_string()) bad("stop must be a string or an array of strings", "stop");
            job.stop.push_back(s.as_string());
        }
    } else if (!stop.is_null()) {
        bad("stop must be a string or an array of strings", "stop");
    }
    job.stop.erase(std::remove(job.stop.begin(), job.stop.end(), std::string()), job.stop.end());
    const Json * n = body.find("n");
    if (n != nullptr && !n->is_null() && !(n->is_number() && n->as_number() == 1.0)) {
        bad("only n = 1 is supported", "n");
    }
    job.stream = flag(body, "stream", false);
    const Json & so = body.get("stream_options");
    if (so.is_object()) job.include_usage = flag(so, "include_usage", false);
}

} // namespace

Job parse_request(const Json & body, const bool chat, const Defaults & defaults) {
    if (!body.is_object()) bad("the request body must be a JSON object");
    Job job;
    job.chat = chat;
    parse_sampling(body, defaults, job);
    if (!chat) {
        job.thinking = false;
        job.echo = flag(body, "echo", false);
        const Json * p = body.find("prompt");
        if (p == nullptr) bad("prompt is required", "prompt");
        const Json * one = p;
        if (p->is_array() && p->size() == 1 && (p->items()[0].is_string() || p->items()[0].is_array())) {
            one = &p->items()[0];  // a batch of one prompt
        }
        if (one->is_string()) {
            job.prompt = one->as_string();
        } else if (one->is_array() && one->size() > 0) {
            for (const Json & t : one->items()) {
                if (!t.is_number() || !t.number_is_integer()) {
                    bad("prompt must be a string, token ids, or a batch of one of them", "prompt");
                }
                job.prompt_ids.push_back((int32_t) t.as_number());
            }
        } else {
            bad("prompt must be a string, token ids, or a batch of one of them", "prompt");
        }
        return job;
    }
    const Json & messages = body.get("messages");
    if (!messages.is_array() || messages.size() == 0) bad("messages must be a non-empty array", "messages");
    // the chat template's request: the messages ("developer" is the system
    // role), the tools, the template switches
    Json req = Json::object();
    Json msgs = Json::array();
    for (const Json & m : messages.items()) {
        if (!m.is_object()) bad("every message must be an object", "messages");
        collect_images(m.get("content"), job.images);
        Json copy = Json::object();
        for (const auto & [k, v] : m.members()) {
            if (k == "role" && v.is_string() && v.as_string() == "developer") {
                copy.set(k, Json::string("system"));
            } else {
                copy.set(k, v);
            }
        }
        msgs.push(std::move(copy));
    }
    req.set("messages", std::move(msgs));
    const Json & tools = body.get("tools");
    const Json & choice = body.get("tool_choice");
    const bool no_tools = choice.is_string() && choice.as_string() == "none";
    if (!tools.is_null() && !tools.is_array()) bad("tools must be an array", "tools");
    if (tools.is_array() && tools.size() > 0 && !no_tools) {
        job.parse_tools = true;
        job.tools = tools;
        req.set("tools", tools);
    }
    req.set("add_generation_prompt", Json::boolean(true));
    // switches: chat_template_kwargs (as in vLLM / llama.cpp), then the
    // top-level reasoning_effort
    const Json & kwargs = body.get("chat_template_kwargs");
    if (kwargs.is_object()) {
        for (const char * key : {"enable_thinking", "reasoning_effort", "preserve_thinking"}) {
            if (const Json * v = kwargs.find(key)) req.set(key, *v);
        }
    }
    if (const Json * v = body.find("reasoning_effort"); v != nullptr && !v->is_null()) {
        const std::string e = effort(*v);
        if (e == "off") {
            req.set("enable_thinking", Json::boolean(false));
        } else {
            req.set("reasoning_effort", Json::string(e));
        }
    }
    if (const Json * v = req.find("enable_thinking"); v != nullptr && !v->is_bool()) {
        bad("enable_thinking must be a boolean", "chat_template_kwargs");
    }
    const Json * et = req.find("enable_thinking");
    job.thinking = et == nullptr || et->as_bool();
    try {
        job.prompt_segments = text::render_chat_segments(req);
        job.prompt = text::join_segments(job.prompt_segments);
    } catch (const std::runtime_error & e) {
        bad(e.what(), "messages");
    }
    return job;
}

OutputParser::OutputParser(const Job & job)
    : job_(job), state_(job.chat && job.thinking ? State::Reasoning : State::Content) {}

Delta OutputParser::push(const std::string & text) {
    Delta d;
    if (stopped_) return d;
    raw_ += text;
    // the earliest stop string; none can start before fed_ (the text that may
    // begin one is held back)
    size_t stop_at = std::string::npos;
    for (const std::string & s : job_.stop) {
        stop_at = std::min(stop_at, raw_.find(s, fed_));
    }
    if (stop_at != std::string::npos) {
        raw_.resize(stop_at);
        stopped_ = true;
        feed(raw_.substr(fed_), d);
        fed_ = raw_.size();
        return d;
    }
    size_t hold = 0;
    for (const std::string & s : job_.stop) {
        hold = std::max(hold, partial_suffix(raw_, s));
    }
    const size_t limit = std::min(raw_.size() - hold, utf8_complete(raw_));
    if (limit > fed_) {
        feed(raw_.substr(fed_, limit - fed_), d);
        fed_ = limit;
    }
    return d;
}

Delta OutputParser::finish() {
    Delta d;
    if (fed_ < raw_.size()) {
        feed(raw_.substr(fed_), d);
        fed_ = raw_.size();
    }
    switch (state_) {
        case State::Reasoning: emit(State::Reasoning, buf_, d); break;
        case State::Content: emit(State::Content, buf_, d); break;
        case State::ToolCall: emit(State::Content, "<tool_call>" + buf_, d); break;  // unterminated
    }
    buf_.clear();
    state_ = State::Content;
    return d;
}

void OutputParser::feed(const std::string & text, Delta & d) {
    static const std::string kThinkEnd = "</think>", kCall = "<tool_call>", kCallEnd = "</tool_call>";
    buf_ += text;
    while (true) {
        const std::string * tag = state_ == State::Reasoning ? &kThinkEnd
                                  : state_ == State::ToolCall ? &kCallEnd
                                  : job_.parse_tools          ? &kCall
                                                              : nullptr;
        if (tag == nullptr) {
            emit(state_, buf_, d);
            buf_.clear();
            return;
        }
        const size_t pos = buf_.find(*tag);
        if (pos == std::string::npos) {
            if (state_ != State::ToolCall) {  // a call is parsed whole
                const size_t hold = partial_suffix(buf_, *tag);
                emit(state_, buf_.substr(0, buf_.size() - hold), d);
                buf_.erase(0, buf_.size() - hold);
            }
            return;
        }
        const std::string before = buf_.substr(0, pos);
        buf_.erase(0, pos + tag->size());
        switch (state_) {
            case State::Reasoning:
                emit(State::Reasoning, before, d);
                space_[0].clear();
                state_ = State::Content;
                break;
            case State::Content:
                emit(State::Content, before, d);
                space_[1].clear();
                state_ = State::ToolCall;
                break;
            case State::ToolCall: {
                ToolCall call;
                if (parse_call(before, call)) {
                    d.calls.push_back(std::move(call));
                    ++n_calls_;
                } else {
                    emit(State::Content, "<tool_call>" + before + "</tool_call>", d);
                }
                state_ = State::Content;
                break;
            }
        }
    }
}

void OutputParser::emit(const State channel, std::string text, Delta & d) {
    if (!job_.chat) {  // a raw completion: the text as generated
        d.content += sanitize_utf8(text);
        return;
    }
    const int c = channel == State::Reasoning ? 0 : 1;
    if (!started_[c]) {
        size_t b = 0;
        while (b < text.size() && is_space(text[b])) ++b;
        text.erase(0, b);
        if (text.empty()) return;
        started_[c] = true;
    }
    text = space_[c] + text;
    size_t e = text.size();
    while (e > 0 && is_space(text[e - 1])) --e;
    space_[c] = text.substr(e);
    text.resize(e);
    if (text.empty()) return;
    (c == 0 ? d.reasoning : d.content) += sanitize_utf8(text);
}

// <function=NAME> <parameter=KEY>\nVALUE\n</parameter>... </function>, the
// chat template's format for a call
bool OutputParser::parse_call(const std::string & block, ToolCall & call) {
    size_t i = 0;
    const auto skip = [&] {
        while (i < block.size() && is_space(block[i])) ++i;
    };
    const auto take = [&](const std::string & lit) {
        if (block.compare(i, lit.size(), lit) != 0) return false;
        i += lit.size();
        return true;
    };
    skip();
    if (!take("<function=")) return false;
    size_t e = block.find('>', i);
    if (e == std::string::npos || e == i) return false;
    call.name = sanitize_utf8(block.substr(i, e - i));
    i = e + 1;
    Json args = Json::object();
    while (true) {
        skip();
        if (take("</function>")) break;
        if (!take("<parameter=")) return false;
        e = block.find('>', i);
        if (e == std::string::npos) return false;
        const std::string key = sanitize_utf8(block.substr(i, e - i));
        i = e + 1;
        e = block.find("</parameter>", i);
        if (e == std::string::npos) return false;
        std::string value = block.substr(i, e - i);
        i = e + std::string("</parameter>").size();
        if (!value.empty() && value.front() == '\n') value.erase(0, 1);
        if (!value.empty() && value.back() == '\n') value.pop_back();
        args.set(key, Json::parse(argument_value(call.name, key, sanitize_utf8(value))));
    }
    skip();
    if (i != block.size()) return false;
    call.id = random_id("call_", 24);
    call.arguments = args.dump();
    return true;
}

// A parameter's value as JSON text: a string, unless the tool's schema gives
// the parameter another type and the value parses as JSON (the template
// writes non-string arguments with tojson).
std::string OutputParser::argument_value(const std::string & function, const std::string & key,
                                         const std::string & value) const {
    std::string type;
    for (const Json & t : job_.tools.items()) {
        const Json & f = t.has("function") ? t.get("function") : t;
        if (!f.get("name").is_string() || f.get("name").as_string() != function) continue;
        const Json & prop = f.get("parameters").get("properties").get(key);
        const Json & ty = prop.get("type");
        if (ty.is_string()) {
            type = ty.as_string();
        } else if (ty.is_array()) {  // ["integer", "null"]: the first non-null type
            for (const Json & x : ty.items()) {
                if (x.is_string() && x.as_string() != "null") {
                    type = x.as_string();
                    break;
                }
            }
        } else if (prop.has("anyOf") || prop.has("oneOf") || prop.has("enum") || prop.has("const")) {
            type = "any";
        }
        break;
    }
    if (!type.empty() && type != "string") {
        try {
            return Json::parse(value).dump();
        } catch (const std::runtime_error &) {
        }
    }
    return Json::string(value).dump();
}

const char * finish_reason(const bool length, const bool tool_calls) {
    return tool_calls ? "tool_calls" : length ? "length" : "stop";
}

size_t utf8_complete(const std::string & s) {
    // the last lead byte within the final 3 bytes, if its character is cut
    for (size_t back = 1; back <= 3 && back <= s.size(); ++back) {
        const auto b = (uint8_t) s[s.size() - back];
        if ((b & 0xC0) == 0x80) continue;  // a continuation byte
        const size_t len = (b >> 5) == 0x6 ? 2 : (b >> 4) == 0xE ? 3 : (b >> 3) == 0x1E ? 4 : 1;
        return len > back ? s.size() - back : s.size();
    }
    return s.size();
}

bool base64_decode(const std::string & in, std::string & out) {
    out.clear();
    out.reserve(in.size() / 4 * 3);
    uint32_t acc = 0;
    int bits = 0;
    size_t pad = 0;
    for (const char c : in) {
        int v = -1;
        if (c >= 'A' && c <= 'Z') v = c - 'A';
        else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
        else if (c >= '0' && c <= '9') v = c - '0' + 52;
        else if (c == '+' || c == '-') v = 62;
        else if (c == '/' || c == '_') v = 63;
        else if (c == '=') { ++pad; continue; }
        else if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        else return false;
        if (pad > 0) return false;  // data after the padding
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += (char) ((acc >> bits) & 0xFF);
        }
    }
    return pad <= 2;
}

std::string random_id(const std::string & prefix, const size_t n) {
    static const char kAlpha[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    static std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<int> pick(0, (int) sizeof(kAlpha) - 2);
    std::string id = prefix;
    for (size_t i = 0; i < n; ++i) id += kAlpha[pick(rng)];
    return id;
}

Json error_body(const std::string & message, const std::string & type, const std::string & param) {
    Json err = Json::object();
    err.set("message", Json::string(message));
    err.set("type", Json::string(type));
    err.set("param", param.empty() ? Json() : Json::string(param));
    err.set("code", Json());
    Json body = Json::object();
    body.set("error", std::move(err));
    return body;
}

} // namespace omph::server
