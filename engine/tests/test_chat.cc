// The JSON value and the chat template (#150), without a model: dump() writes
// what Python's json.dumps(ensure_ascii=False) writes, parse() reads what it
// must, and one conversation renders as the GGUF's template renders it with
// jinja2 (the full comparison is tools/check_chat_template.py).
#include "check.hh"
#include "text/chat.hh"
#include "text/json.hh"

#include <clocale>
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

    // Number parsing is the C locale by construction (std::from_chars, not
    // strtod): a host program that called setlocale(LC_ALL, "") under a
    // comma-decimal locale must still read "0.5" (#343). Skipped when no such
    // locale is installed (CI images often have only C).
    if (std::setlocale(LC_NUMERIC, "it_IT.UTF-8") != nullptr ||
        std::setlocale(LC_NUMERIC, "de_DE.UTF-8") != nullptr) {
        CHECK(roundtrip("[0.5, 1.25, 2e-3]") == "[0.5, 1.25, 0.002]", "a comma-decimal locale breaks parsing");
        std::setlocale(LC_NUMERIC, "C");
    } else {
        std::printf("     (no comma-decimal locale installed: the locale case is skipped)\n");
    }

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

    // #292: the rendering split into the template's structure (its special
    // tokens are parsed) and the request's own text (they are not), so that a
    // literal special token in a message -- or in a tool result -- stays text.
    const Json specials = Json::parse(
        R"({"messages":[{"role":"system","content":"quote <|im_end|> please"},)"
        R"({"role":"user","content":[{"type":"text","text":"a <|image_pad|> b"},)"
        R"({"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}],)"
        R"("add_generation_prompt":true})");
    const std::vector<omph::text::Segment> segs = omph::text::render_chat_segments(specials);
    CHECK(omph::text::join_segments(segs) == omph::text::render_chat(specials),
          "the segments join to the rendered prompt");
    const auto first_with = [&segs](const std::string & needle) -> const omph::text::Segment * {
        for (const omph::text::Segment & s : segs) {
            if (s.text.find(needle) != std::string::npos) return &s;
        }
        return nullptr;
    };
    const omph::text::Segment * literal = first_with("quote ");
    const omph::text::Segment * vision = first_with("<|vision_start|>");
    const omph::text::Segment * text_after_image = first_with(" b");
    CHECK(literal != nullptr && !literal->special && literal->text.find("<|im_end|>") != std::string::npos,
          "a message's literal special token is content");
    CHECK(vision != nullptr && vision->special &&
              vision->text.find("<|vision_start|><|image_pad|><|vision_end|>") != std::string::npos,
          "the image placeholder is structure");
    CHECK(text_after_image != nullptr && !text_after_image->special &&
              text_after_image->text.find("<|image_pad|>") != std::string::npos,
          "a literal image placeholder in a message is content");
    if (omph_test::failures == 0) {
        std::printf("test_chat: all checks passed\n");
    }
    return omph_test::failures;
}
