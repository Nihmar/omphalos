// The JSON value and the chat template (#150), without a model: dump() writes
// what Python's json.dumps(ensure_ascii=False) writes, parse() reads what it
// must, and one conversation renders as the GGUF's template renders it with
// jinja2 (the full comparison is tools/check_chat_template.py).
#include "check.hh"
#include "text/chat.hh"
#include "text/json.hh"

#include <cstdio>
#include <stdexcept>
#include <string>

using omph::text::Json;

namespace {

std::string roundtrip(const std::string & s) { return Json::parse(s).dump(); }

bool throws(const std::string & s) {
    try {
        (void) Json::parse(s);
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    // json.dumps(json.loads(s), ensure_ascii=False) for each s
    CHECK(roundtrip(R"({"a":1,"b":[true,false,null],"c":"x\"y\\z\n\t\u0001é"})") ==
              "{\"a\": 1, \"b\": [true, false, null], \"c\": \"x\\\"y\\\\z\\n\\t\\u0001\xc3\xa9\"}",
          "separators, escapes, raw UTF-8");
    CHECK(roundtrip("[2.5, 1e16, 1.5e-05, 0.0001, 100.0, 1e-07, -0.5, 123456789]") ==
              "[2.5, 1e+16, 1.5e-05, 0.0001, 100.0, 1e-07, -0.5, 123456789]",
          "Python float repr");
    CHECK(roundtrip(R"("😀\/")") == "\"\xf0\x9f\x98\x80/\"", "surrogate pair, escaped slash");
    CHECK(roundtrip(R"({"k":1,"k":2})") == "{\"k\": 2}", "a duplicate key: the last value wins");
    CHECK(roundtrip(" { } ") == "{}" && roundtrip("[]") == "[]", "empty containers");
    CHECK(throws("{\"a\":}") && throws("[1,]") && throws("\"x") && throws("1 2") && throws("tru"),
          "malformed input throws");

    // the template's "thinking off" generation prompt (jinja2 reference)
    const Json req = Json::parse(
        R"({"messages":[{"role":"system","content":" Be brief. "},{"role":"user","content":"Hi"}],)"
        R"("add_generation_prompt":true,"enable_thinking":false})");
    CHECK(omph::text::render_chat(req) ==
              "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n",
          "a rendered prompt");
    bool raised = false;
    try {
        (void) omph::text::render_chat(Json::parse(R"({"messages":[]})"));
    } catch (const std::runtime_error &) {
        raised = true;
    }
    CHECK(raised, "the template's errors are raised");
    if (omph_test::failures == 0) {
        std::printf("test_chat: all checks passed\n");
    }
    return omph_test::failures;
}
