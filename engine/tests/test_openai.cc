// The server's OpenAI mapping (#156), without a model: requests to prompts
// and parameters, and the generated text to reasoning / content / tool calls,
// the same whether it arrives whole or a byte at a time (what streaming
// relies on). The end-to-end check is tools/check_server.py.
#include "check.hh"
#include "server/openai.hh"
#include "text/json.hh"

#include <cstdio>
#include <string>
#include <vector>

using omph::server::Delta;
using omph::server::Job;
using omph::server::OutputParser;
using omph::text::Json;

namespace {

struct Out {
    std::string reasoning, content;
    std::vector<omph::server::ToolCall> calls;
    bool stopped = false;
    void add(const Delta & d) {
        reasoning += d.reasoning;
        content += d.content;
        calls.insert(calls.end(), d.calls.begin(), d.calls.end());
    }
};

// the text pushed in pieces of `step` bytes (0: whole)
Out run(const Job & job, const std::string & text, size_t step) {
    OutputParser p(job);
    Out o;
    if (step == 0) step = text.size();
    for (size_t i = 0; i < text.size() && !p.stopped(); i += step) {
        o.add(p.push(text.substr(i, step)));
    }
    o.add(p.finish());
    o.stopped = p.stopped();
    return o;
}

// the same result for every piece size
bool consistent(const Job & job, const std::string & text, Out & whole) {
    whole = run(job, text, 0);
    for (size_t step = 1; step <= 7; ++step) {
        const Out o = run(job, text, step);
        if (o.reasoning != whole.reasoning || o.content != whole.content || o.calls.size() != whole.calls.size() ||
            o.stopped != whole.stopped) {
            std::printf("step %zu: [%s] [%s] %zu\n", step, o.reasoning.c_str(), o.content.c_str(), o.calls.size());
            return false;
        }
        for (size_t i = 0; i < o.calls.size(); ++i) {
            if (o.calls[i].name != whole.calls[i].name || o.calls[i].arguments != whole.calls[i].arguments) {
                return false;
            }
        }
    }
    return true;
}

Job chat_job(const std::string & body) {
    return omph::server::parse_request(Json::parse(body), true, omph::server::Defaults{});
}

bool rejects(const std::string & body, bool chat = true) {
    try {
        (void) omph::server::parse_request(Json::parse(body), chat, omph::server::Defaults{});
    } catch (const omph::server::BadRequest &) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    const std::string tools = R"("tools":[{"type":"function","function":{"name":"get_weather",)"
                              R"("parameters":{"type":"object","properties":{"city":{"type":"string"},)"
                              R"("days":{"type":"integer"},"units":{"enum":["c","f"]}}}}}])";
    // requests
    Job j = chat_job(R"({"messages":[{"role":"user","content":"Hi"}],"temperature":0.6,"top_p":0.95,)"
                     R"("top_k":20,"seed":7,"max_completion_tokens":64,"stop":"END","stream":true,)"
                     R"("stream_options":{"include_usage":true}})");
    const std::string tail = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n";
    CHECK(j.prompt.rfind("<|im_start|>system\nReasoning effort is set to xhigh.", 0) == 0 &&
              j.prompt.size() > tail.size() && j.prompt.compare(j.prompt.size() - tail.size(), tail.size(), tail) == 0,
          "the generation prompt (the template's default effort)");
    CHECK(j.thinking && !j.parse_tools && j.stream && j.include_usage, "flags");
    CHECK(!j.timings_per_token && !j.return_progress && j.sse_ping_interval == 0.0,
          "the live-timing fields default off (#382)");
    const Job live = chat_job(
        R"({"messages":[{"role":"user","content":"Hi"}],"stream":true,"timings_per_token":true,)"
        R"("return_progress":true,"sse_ping_interval":1})");
    CHECK(live.timings_per_token && live.return_progress && live.sse_ping_interval == 1.0,
          "timings_per_token, return_progress and sse_ping_interval are parsed (#382)");
    CHECK(j.temperature == 0.6f && j.top_p == 0.95f && j.top_k == 20 && j.seeded && j.seed == 7 &&
              j.max_tokens == 64 && j.stop == std::vector<std::string>{"END"},
          "sampling");
    j = chat_job(R"({"messages":[{"role":"developer","content":"Be brief."},{"role":"user","content":"Hi"}],)"
                 R"("chat_template_kwargs":{"enable_thinking":false}})");
    CHECK(j.prompt == "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n\n</think>\n\n",
          "developer role, thinking off");
    CHECK(!j.thinking, "thinking off");
    CHECK(!chat_job(R"({"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"minimal"})").thinking,
          "reasoning_effort minimal turns thinking off");
    j = chat_job(R"({"messages":[{"role":"user","content":"Hi"}],)" + tools + "}");
    CHECK(j.parse_tools && j.prompt.find("<tools>") != std::string::npos, "tools");
    j = chat_job(R"({"messages":[{"role":"user","content":"Hi"}],"tool_choice":"none",)" + tools + "}");
    CHECK(!j.parse_tools && j.prompt.find("<tools>") == std::string::npos, "tool_choice none");
    CHECK(rejects(R"({"messages":[]})"), "no messages");
    CHECK(rejects(R"({"messages":[{"role":"user","content":"Hi"}],"n":2})"), "n > 1");
    CHECK(rejects(R"({"messages":[{"role":"user","content":"Hi"}],"temperature":"hot"})"), "bad temperature");
    CHECK(!rejects(R"({"messages":[{"role":"user","content":"Hi"}],"frequency_penalty":0,)"
                   R"("presence_penalty":0,"repetition_penalty":1,"repeat_penalty":1.0,"logit_bias":{}})"),
          "the no-op values of every sampling field");
    CHECK(rejects(R"({"messages":[{"role":"user","content":"Hi"}],"logit_bias":{"123":1}})") &&
              rejects(R"({"messages":[{"role":"user","content":"Hi"}],"frequency_penalty":"no"})") &&
              rejects(R"({"messages":[{"role":"user","content":"Hi"}],"repeat_penalty":0})"),
          "unimplemented sampling fields are refused, not dropped in silence");
    j = chat_job(R"({"messages":[{"role":"user","content":"Hi"}],"repeat_penalty":1.1,)"
                 R"("repetition_penalty":1.2,"frequency_penalty":0.3,"presence_penalty":0.4,)"
                 R"("repeat_last_n":1024})");
    CHECK(j.repeat_penalty == 1.2f && j.frequency_penalty == 0.3f && j.presence_penalty == 0.4f &&
              j.penalty_last_n == 1024,
          "llama.cpp's penalties sampler (repetition_penalty is the vLLM spelling)");
    CHECK(rejects(R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]})") &&
              rejects(R"({"messages":[{"role":"user","content":[{"type":"image_url",)"
                      R"("image_url":{"url":"https://example.com/a.png"}}]}]})") &&
              rejects(R"({"messages":[{"role":"user","content":[{"type":"image_url",)"
                      R"("image_url":{"url":"data:,x"}}]}]})") &&
              rejects(R"({"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{}}]}]})"),
          "images other than base64 data: URLs, audio");
    j = chat_job(R"({"messages":[{"role":"user","content":[{"type":"text","text":"What is it?"},)"
                 R"({"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8gd29ybGQ="}}]}]})");
    CHECK(j.images == std::vector<std::string>{"hello world"} &&
              j.prompt.find("What is it?<|vision_start|><|image_pad|><|vision_end|><|im_end|>") != std::string::npos,
          "an image: its bytes, its place in the prompt");
    std::string bytes;
    CHECK(omph::server::base64_decode("SGk-_w", bytes) && bytes == "Hi>\xff" &&
              !omph::server::base64_decode("SGk=SGk", bytes) && !omph::server::base64_decode("S*Gk", bytes),
          "base64 (URL-safe alphabet, padding)");
    CHECK(rejects(R"({"messages":[{"role":"assistant","content":"Hi"}]})"), "the template's own errors");
    j = omph::server::parse_request(Json::parse(R"({"prompt":[[1,2,3]],"echo":true})"), false, {});
    CHECK(!j.chat && j.prompt_ids == std::vector<int32_t>({1, 2, 3}) && j.echo, "completions: token ids");
    CHECK(rejects(R"({"prompt":["a","b"]})", false), "completions: a batch of two");

    // output: thinking, then content, trimmed
    Out o;
    Job think = chat_job(R"({"messages":[{"role":"user","content":"Hi"}]})");
    CHECK(consistent(think, "The user greets.\n</think>\n\nHello! 😀 How can I help?", o), "think split: pieces");
    CHECK(o.reasoning == "The user greets." && o.content == "Hello! 😀 How can I help?", "think split: [%s] [%s]",
          o.reasoning.c_str(), o.content.c_str());
    CHECK(consistent(think, "cut off before the end of thinking  ", o) && o.reasoning ==
              "cut off before the end of thinking" && o.content.empty(),
          "no </think>");
    // a stop string, also across pieces and inside the reasoning
    Job stop = think;
    stop.stop = {"STOP", "\xc3\xa9t"};
    CHECK(consistent(stop, "a</think>b STOP c", o) && o.stopped && o.content == "b", "stop string");
    CHECK(consistent(stop, "a STO STOP", o) && o.stopped && o.reasoning == "a STO", "a partial stop string");
    CHECK(consistent(stop, "</think>caf\xc3\xa9 \xc3\xa9t\xc3\xa9", o) && o.content == "caf\xc3\xa9", "UTF-8 stop");
    // tool calls: typed by the schema, content before them
    Job tool = chat_job(R"({"messages":[{"role":"user","content":"Hi"}],)" + tools + "}");
    const std::string call = "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n"
                             "</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=units>\n\"c\"\n"
                             "</parameter>\n</function>\n</tool_call>";
    CHECK(consistent(tool, "plan\n</think>\n\n" + call + "\n" + call, o), "tool calls: pieces");
    CHECK(o.reasoning == "plan" && o.content == "Let me check.\nLet me check." && o.calls.size() == 2,
          "tool calls: [%s] [%s] %zu", o.reasoning.c_str(), o.content.c_str(), o.calls.size());
    CHECK(!o.calls.empty() && o.calls[0].name == "get_weather" &&
              o.calls[0].arguments == R"({"city": "Rome", "days": 3, "units": "c"})" &&
              o.calls[0].id.rfind("call_", 0) == 0,
          "tool call arguments: %s", o.calls.empty() ? "" : o.calls[0].arguments.c_str());
    CHECK(consistent(tool, "</think><tool_call>\n<function=get_weather>\n<parameter=city>\nRo", o) &&
              o.calls.empty() && o.content == "<tool_call>\n<function=get_weather>\n<parameter=city>\nRo",
          "an unterminated call stays content");
    CHECK(consistent(tool, "</think><tool_call>garbage</tool_call>", o) && o.calls.empty() &&
              o.content == "<tool_call>garbage</tool_call>",
          "a malformed call stays content");
    CHECK(consistent(think, "</think><tool_call>x</tool_call>", o) && o.content == "<tool_call>x</tool_call>",
          "no tools: no parsing");
    // raw completions: as generated; invalid UTF-8 replaced
    Job raw = omph::server::parse_request(Json::parse(R"({"prompt":"x"})"), false, {});
    CHECK(consistent(raw, "  two  spaces\n", o) && o.content == "  two  spaces\n", "completion text untrimmed");
    CHECK(consistent(raw, "a\xff" "b\xe2\x82", o) && o.content == "a\xef\xbf\xbd" "b\xef\xbf\xbd\xef\xbf\xbd",
          "invalid UTF-8");
    CHECK(omph::server::utf8_complete("ab\xe2\x82") == 2 && omph::server::utf8_complete("ab\xe2\x82\xac") == 5,
          "utf8_complete");
    // The public sanitizer (#338): the same rule, on strings the parser never saw
    CHECK(omph::server::sanitize_utf8("ok") == "ok", "sanitize keeps valid text");
    CHECK(omph::server::sanitize_utf8("a\xff" "b") == "a\xef\xbf\xbd" "b", "sanitize replaces a bad byte");
    CHECK(omph::server::sanitize_utf8("\xed\xa0\x80") == "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd",
          "sanitize rejects a surrogate (one replacement per byte)");
    CHECK(omph::server::sanitize_utf8("\xf0\x9f\x98\x80") == "\xf0\x9f\x98\x80", "a 4-byte character survives");

    if (omph_test::failures == 0) {
        std::printf("test_openai: all checks passed\n");
    }
    return omph_test::failures;
}
