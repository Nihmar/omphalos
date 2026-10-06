// The OpenAI-compatible API on the engine (#156), without the engine: request
// bodies to a prompt and generation parameters, the generated text back to
// OpenAI's message fields (reasoning, content, tool calls), incrementally so
// that streamed and whole responses carry the same text. CPU-only: the CI
// tests exercise it (engine/tests/test_openai.cc).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "text/chat.hh"
#include "text/json.hh"

namespace omph::server {

// The server's defaults for what a request leaves out (omph-server options).
struct Defaults {
    float temperature = 0.0f;  // greedy: the speculative MTP path
    int top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    // llama.cpp's penalties sampler (--repeat-penalty/-last-n/--*-penalty):
    // the same window for all three, off by default
    float repeat_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int penalty_last_n = 64;
    int64_t max_tokens = -1;   // -1: until the context is full
};

// A request, validated: what to generate and how to answer.
struct Job {
    bool chat = true;             // /v1/chat/completions, else /v1/completions
    std::string prompt;           // the text to tokenize (special tokens parsed when chat)
    // The chat rendering as structure and content (#292): the server tokenizes
    // each segment with its own rule, so a message's text cannot inject one of
    // the template's special tokens.
    std::vector<text::Segment> prompt_segments;
    std::vector<int32_t> prompt_ids;  // /v1/completions with token ids instead of text
    bool thinking = true;         // the generation opens a <think> block
    bool parse_tools = false;     // tools given (and tool_choice is not "none")
    text::Json tools;             // their declarations (parameter types for the arguments)
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    bool seeded = false;
    uint64_t seed = 0;
    float repeat_penalty = 1.0f;    // llama.cpp's --repeat-penalty
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int penalty_last_n = 64;        // llama.cpp's --repeat-last-n (0: off)
    int64_t max_tokens = -1;
    std::vector<std::string> stop;
    bool stream = false;
    // The web UI's live statistics (#382): llama.cpp's fields. With
    // timings_per_token every streamed chunk carries the running `timings`;
    // return_progress adds a top-level `prompt_progress` per prefill chunk;
    // sse_ping_interval (seconds, 0 = off) keeps a silent stream alive with
    // `:` comments.
    bool timings_per_token = false;
    bool return_progress = false;
    double sse_ping_interval = 0.0;
    bool include_usage = false;   // stream_options.include_usage
    bool echo = false;            // /v1/completions: the prompt before the completion
    std::vector<std::string> images;  // the image files' bytes, in prompt order (#160)
};

// An invalid request: the message of a 400 answer.
struct BadRequest {
    std::string message;
    std::string param;
};

// Throws BadRequest. `chat` selects the endpoint.
Job parse_request(const text::Json & body, bool chat, const Defaults & defaults);

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;  // a JSON object, as a string (OpenAI's format)
};

// What a piece of generated text adds to the answer.
struct Delta {
    std::string reasoning;
    std::string content;
    std::vector<ToolCall> calls;
    bool empty() const { return reasoning.empty() && content.empty() && calls.empty(); }
};

// The generated text to message fields, as it arrives. The text is split into
// reasoning (up to </think>) and content; <tool_call> blocks in the content
// become tool calls; each part is trimmed of surrounding whitespace, as the
// chat template trims them (a raw completion's text stays as generated).
// Text that may still turn into a tag, a stop string or a whole UTF-8
// character is held back until it is decided.
class OutputParser {
public:
    OutputParser(const Job & job);

    // Appends generated text; returns what is now decided. stopped() turns
    // true when a stop string appears (the text from it on is dropped).
    Delta push(const std::string & text);
    // The generation ended: everything held back is decided.
    Delta finish();

    bool stopped() const { return stopped_; }
    bool any_tool_call() const { return n_calls_ > 0; }

private:
    enum class State { Reasoning, Content, ToolCall };
    void feed(const std::string & text, Delta & d);
    void emit(State channel, std::string text, Delta & d);
    bool parse_call(const std::string & block, ToolCall & call);
    std::string argument_value(const std::string & function, const std::string & key,
                               const std::string & value) const;

    const Job & job_;
    std::string raw_;       // the generated text so far (up to a stop string)
    size_t fed_ = 0;        // raw_[0, fed_) went through feed()
    std::string buf_;       // fed text waiting for its state's tag
    State state_;
    bool stopped_ = false;
    bool started_[2] = {false, false};  // reasoning / content: something emitted
    std::string space_[2];              // their trailing whitespace, held back
    int n_calls_ = 0;
};

// OpenAI's finish_reason from how the generation ended.
const char * finish_reason(bool length, bool tool_calls);

// The bytes of `text` that end in a whole UTF-8 character (the rest may be
// completed by the next token).
size_t utf8_complete(const std::string & text);

// Invalid UTF-8 (a byte-level token sequence that never completes a character,
// or a lone surrogate the JSON parser combined) to U+FFFD, so that every
// string written into a response keeps it valid JSON text (#338).
std::string sanitize_utf8(const std::string & s);

// Base64 (RFC 4648, padding optional, whitespace ignored) to bytes; false
// on any other character.
bool base64_decode(const std::string & in, std::string & out);

// A random identifier: prefix + n letters and digits (chatcmpl-..., call_...).
std::string random_id(const std::string & prefix, size_t n);

// An OpenAI error object.
text::Json error_body(const std::string & message, const std::string & type, const std::string & param = "");

} // namespace omph::server
