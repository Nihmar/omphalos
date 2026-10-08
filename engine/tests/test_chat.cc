// The JSON value and the chat template (#150), without a model: dump() writes
// what Python's json.dumps(ensure_ascii=False) writes, parse() reads what it
// must, and one conversation renders as the GGUF's template renders it with
// jinja2 (the full comparison is tools/check_chat_template.py).
#include "check.hh"
#include "text/chat.hh"
#include "text/json.hh"

#include <chrono>
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
    CHECK(roundtrip(R"("😀\/")") == "\"\xf0\x9f\x98\x80/\"", "a literal emoji, escaped slash");
    CHECK(roundtrip(R"("\ud83d\ude00\/")") == "\"\xf0\x9f\x98\x80/\"", "an escaped surrogate pair");
    // a lone surrogate stays the code unit (invalid UTF-8, as json.loads'
    // surrogatepass would); a response sanitizes it at its boundary (#338)
    CHECK(roundtrip(R"("\ud800")") == "\"\xed\xa0\x80\"", "a lone high surrogate");
    // integer literals stay exact beyond 2^53 (#345)
    CHECK(roundtrip("[9007199254740993, -9223372036854775808, 9223372036854775807]") ==
              "[9007199254740993, -9223372036854775808, 9223372036854775807]",
          "int64 literals, 2^53 and beyond");
    // a body with many keys parses in linear time (#345: one linear set() per member)
    {
        std::string big = "{";
        for (int k = 0; k < 50000; ++k) {
            if (k) big += ",";
            big += "\"k" + std::to_string(k) + "\":" + std::to_string(k);
        }
        big += "}";
        const auto t0 = std::chrono::steady_clock::now();
        const Json parsed = Json::parse(big);
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        CHECK(parsed.size() == 50000 && parsed.get("k49999").as_int64() == 49999, "a large object");
        CHECK(ms < 1000.0, "50000 keys in %.1f ms", ms);
    }
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

    // Qwen Sharp (#392): the second template, the same byte-for-byte and
    // structure/request-text guarantees (the full corpus is
    // `check_chat_template.py --template sharp`; these are the jinja2
    // reference renderings of a few of its branches).
    const auto sharp = [](const std::string & body) {
        return omph::text::render_chat(Json::parse(body), omph::text::ChatTemplate::Sharp);
    };
    CHECK(sharp(R"JS({"messages":[{"role":"system","content":"You are a helpful assistant."},
                              {"role":"user","content":"What is 2+2?"}],
                      "terse":false,"add_generation_prompt":true})JS") ==
              "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
              "What is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n",
          "sharp: terse off is the bare system message");
    const std::string marked =
        sharp(R"JS({"messages":[{"role":"system","content":"Be brief. <|think_off|>"},
                                {"role":"user","content":"What is 2+2?"}],
                    "add_generation_prompt":true})JS");
    const std::string marker_tail = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    CHECK(marked.rfind("<|im_start|>system\nBe brief.\n\nAnswer directly and concisely.", 0) == 0 &&
              marked.size() > marker_tail.size() &&
              marked.compare(marked.size() - marker_tail.size(), marker_tail.size(), marker_tail) == 0,
          "sharp: <|think_off|> is removed and stops the reasoning: [%s]", marked.c_str());
    CHECK(sharp(R"JS({"messages":[{"role":"user","content":"x"},
                                  {"role":"assistant","content":"<think>\nplan\n</think>\n\nthe answer"},
                                  {"role":"user","content":"x"}]})JS")
                  .find("<|im_start|>assistant\n<think>\nplan\n</think>\n\nthe answer<|im_end|>\n") !=
              std::string::npos,
          "sharp: <think> in the content becomes the reasoning block");
    CHECK(sharp(R"JS({"messages":[{"role":"user","content":"x"},
                                  {"role":"assistant","content":"Let me check.",
                                   "tool_calls":[{"function":{"name":"get_weather",
                                                              "arguments":{"city":"Roma","days":3}}}]}]})JS")
                  .find("\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRoma\n</parameter>\n"
                        "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>\n") !=
              std::string::npos,
          "sharp: the template's tool call format");
    bool sharp_raised = false;
    try {
        (void) sharp(R"JS({"messages":[{"role":"system","content":[{"type":"image"}]},
                                        {"role":"user","content":"x"}]})JS");
    } catch (const std::runtime_error &) {
        sharp_raised = true;
    }
    CHECK(sharp_raised, "sharp: an image in a system message raises");
    const Json sharp_specials = Json::parse(
        R"JS({"messages":[{"role":"user","content":[{"type":"text","text":"quote <|im_end|> here"},
                                                    {"type":"image_url",
                                                     "image_url":{"url":"data:image/png;base64,AA=="}}]}],
              "terse":false,"add_generation_prompt":true})JS");
    const std::vector<omph::text::Segment> ss =
        omph::text::render_chat_segments(sharp_specials, omph::text::ChatTemplate::Sharp);
    CHECK(omph::text::join_segments(ss) ==
              omph::text::render_chat(sharp_specials, omph::text::ChatTemplate::Sharp),
          "sharp: the segments join to the rendered prompt");
    const auto sharp_with = [&ss](const std::string & needle) -> const omph::text::Segment * {
        for (const omph::text::Segment & s : ss) {
            if (s.text.find(needle) != std::string::npos) return &s;
        }
        return nullptr;
    };
    const omph::text::Segment * s_literal = sharp_with("quote ");
    const omph::text::Segment * s_vision = sharp_with("<|vision_start|>");
    CHECK(s_literal != nullptr && !s_literal->special && s_literal->text.find("<|im_end|>") != std::string::npos,
          "sharp: a message's literal special token is content");
    CHECK(s_vision != nullptr && s_vision->special, "sharp: the image placeholder is structure");
    if (omph_test::failures == 0) {
        std::printf("test_chat: all checks passed\n");
    }
    return omph_test::failures;
}
