// The OpenAI-compatible API on the engine (#156), without the engine: request
// bodies to a prompt and generation parameters, the generated text back to
// OpenAI's message fields (reasoning, content, tool calls), incrementally so
// that streamed and whole responses carry the same text. CPU-only: the CI
// tests exercise it (engine/tests/test_openai.cc).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "text/json.hh"

namespace omph::server {

// The server's defaults for what a request leaves out (omph-server options).
struct Defaults {
    float temperature = 0.0f;  // greedy: the speculative MTP path
    int top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    int64_t max_tokens = -1;   // -1: until the context is full
};

// A request, validated: what to generate and how to answer.
struct Job {
    bool chat = true;             // /v1/chat/completions, else /v1/completions
    std::string prompt;           // the text to tokenize (special tokens parsed when chat)
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
    int64_t max_tokens = -1;
    std::vector<std::string> stop;
    bool stream = false;
    bool include_usage = false;   // stream_options.include_usage
    bool echo = false;            // /v1/completions: the prompt before the completion
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

// A random identifier: prefix + n letters and digits (chatcmpl-..., call_...).
std::string random_id(const std::string & prefix, size_t n);

// An OpenAI error object.
text::Json error_body(const std::string & message, const std::string & type, const std::string & param = "");

} // namespace omph::server
