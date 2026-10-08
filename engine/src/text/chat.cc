#include "text/chat.hh"

#include <stdexcept>
#include <string_view>
#include <utility>

#include "text/tokenizer.hh"

namespace omph::text {
namespace {

[[noreturn]] void raise(const std::string & msg) {
    throw std::runtime_error("chat template: " + msg);
}

// Python's str.isspace() code points (what jinja2's |trim strips)
bool py_space(const uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

// the code point starting at s[i] and its length (invalid bytes: themselves)
uint32_t cp_at(const std::string & s, const size_t i, size_t & len) {
    const auto b0 = (uint8_t) s[i];
    const auto cont = [&](size_t k) { return k < s.size() && ((uint8_t) s[k] & 0xC0) == 0x80; };
    if (b0 >= 0xC0 && b0 < 0xE0 && cont(i + 1)) {
        len = 2;
        return ((b0 & 0x1Fu) << 6) | ((uint8_t) s[i + 1] & 0x3Fu);
    }
    if (b0 >= 0xE0 && b0 < 0xF0 && cont(i + 1) && cont(i + 2)) {
        len = 3;
        return ((b0 & 0x0Fu) << 12) | (((uint8_t) s[i + 1] & 0x3Fu) << 6) | ((uint8_t) s[i + 2] & 0x3Fu);
    }
    len = 1;
    return b0 < 0x80 ? b0 : 0xFFFD;
}

std::string trim(const std::string & s) {
    size_t b = 0;
    while (b < s.size()) {
        size_t len = 0;
        if (!py_space(cp_at(s, b, len))) break;
        b += len;
    }
    size_t e = b;  // end of the last non-space code point
    for (size_t i = b; i < s.size();) {
        size_t len = 0;
        const uint32_t cp = cp_at(s, i, len);
        i += len;
        if (!py_space(cp)) e = i;
    }
    return s.substr(b, e - b);
}

bool starts_with(const std::string & s, std::string_view p) { return s.rfind(p, 0) == 0; }
bool ends_with(const std::string & s, std::string_view p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

std::string join(const std::vector<Segment> & segments) {
    std::string out;
    for (const Segment & s : segments) {
        out += s.text;
    }
    return out;
}

// jinja truthiness of a JSON value (a missing key is null here: falsy)
bool truthy(const Json & j) {
    switch (j.type()) {
        case Json::Type::Null: return false;
        case Json::Type::Bool: return j.as_bool();
        case Json::Type::Number: return j.as_number() != 0.0;
        case Json::Type::String: return !j.as_string().empty();
        case Json::Type::Array:
        case Json::Type::Object: return j.size() != 0;
    }
    return false;
}

// how jinja prints a value with {{ }} (strings as is)
std::string print(const Json & j) {
    if (j.is_string()) return j.as_string();
    if (j.is_null()) return "None";
    if (j.is_bool()) return j.as_bool() ? "True" : "False";
    return j.dump();
}

struct Renderer {
    const Json & req;
    bool add_vision_id = false;
    int image_count = 0;
    int video_count = 0;
    std::vector<Segment> segs;

    explicit Renderer(const Json & request) : req(request) {}

    // Appends a piece of the prompt: the template's own text (`special`), or a
    // string the request supplied (not special, #292). Adjacent pieces with
    // the same flag merge, so the text the tokenizer sees stays as close as
    // possible to the whole prompt.
    void add(std::string text, const bool special = true) {
        if (text.empty()) {
            return;
        }
        if (!segs.empty() && segs.back().special == special) {
            segs.back().text += text;
            return;
        }
        segs.push_back({std::move(text), special});
    }

    // The template trims a message's content (jinja's |trim) before writing
    // it; the trimming must not lose which pieces are the request's text and
    // which are the template's own placeholders, so the segments are clipped
    // to the trimmed span instead of being re-joined (#292).
    void add_trimmed(const std::vector<Segment> & segments) {
        const std::string all = join(segments);
        const std::string trimmed = trim(all);
        if (trimmed.empty()) {
            return;
        }
        const size_t start = all.find(trimmed);
        const size_t end = start + trimmed.size();
        size_t at = 0;
        for (const Segment & seg : segments) {
            const size_t b = at;
            const size_t e = at + seg.text.size();
            at = e;
            if (e <= start || b >= end) {
                continue;
            }
            const size_t from = b >= start ? 0 : start - b;
            const size_t to = e <= end ? seg.text.size() : end - b;
            add(seg.text.substr(from, to - from), seg.special);
        }
    }

    std::vector<Segment> render_content(const Json * content, bool do_vision_count, bool is_system = false) {
        std::vector<Segment> out;
        if (content == nullptr || content->is_null()) {
            return out;
        }
        if (content->is_string()) {
            out.push_back({content->as_string(), false});
            return out;
        }
        if (!content->is_array()) {
            raise("Unexpected content type.");
        }
        for (const Json & item : content->items()) {
            const std::string type = item.get("type").is_string() ? item.get("type").as_string() : "";
            if (item.has("image") || item.has("image_url") || type == "image") {
                if (is_system) raise("System message cannot contain images.");
                if (do_vision_count) ++image_count;
                if (add_vision_id) out.push_back({"Picture " + std::to_string(image_count) + ": ", true});
                out.push_back({"<|vision_start|><|image_pad|><|vision_end|>", true});
            } else if (item.has("video") || type == "video") {
                if (is_system) raise("System message cannot contain videos.");
                if (do_vision_count) ++video_count;
                if (add_vision_id) out.push_back({"Video " + std::to_string(video_count) + ": ", true});
                out.push_back({"<|vision_start|><|video_pad|><|vision_end|>", true});
            } else if (item.has("text")) {
                out.push_back({print(item.get("text")), false});
            } else {
                raise("Unexpected item type in content.");
            }
        }
        return out;
    }

    std::vector<Segment> run() {
        const Json & messages = req.get("messages");
        if (!truthy(messages) || !messages.is_array()) {
            raise("No messages provided.");
        }
        const std::vector<Json> & msgs = messages.items();
        add_vision_id = truthy(req.get("add_vision_id"));
        const Json * enable_thinking = req.find("enable_thinking");
        std::string reasoning;
        if (enable_thinking == nullptr || (enable_thinking->is_bool() && enable_thinking->as_bool())) {
            const Json * effort = req.find("reasoning_effort");
            const std::string e = effort == nullptr ? "xhigh" : print(*effort);
            if (e != "xhigh" && e != "medium" && e != "low") {
                raise("Unexpected reasoning effort " + e +
                      ". Supported types are xhigh (default), medium, and low.");
            }
            if (e == "xhigh") {
                reasoning = "Reasoning effort is set to xhigh. Please think carefully through the task, "
                            "validate key assumptions, consider plausible alternatives, and prioritize "
                            "correctness, consistency, and clarity in the final answer.";
            } else if (e == "low") {
                reasoning = "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
                            "directly to the conclusion without unnecessary elaboration.";
            }
        }
        const auto content_segments = [&](const Json & m, const bool count_vision) {
            // a system message may not carry images / videos (the template
            // raises): the flag was never passed, so the check never fired (#344)
            return render_content(m.find("content"), count_vision, print(m.get("role")) == "system");
        };
        const Json & tools = req.get("tools");
        const auto role_of = [](const Json & m) { return print(m.get("role")); };
        if (tools.is_array() && tools.size() > 0) {
            add("<|im_start|>system\n");
            if (!reasoning.empty()) add(reasoning + "\n\n");
            add("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for (const Json & t : tools.items()) {
                add("\n");
                add(t.dump(), false);  // a tool definition is the client's own text (#292)
            }
            add("\n</tools>");
            add("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
                "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
                "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
                "Reminder:\n- Function calls MUST follow the specified format: an inner <function=...>"
                "</function> block must be nested within <tool_call></tool_call> XML tags\n- Required "
                "parameters MUST be specified\n- You may provide optional reasoning for your function call "
                "in natural language BEFORE the function call, but NOT after\n- If there is no function call "
                "available, answer the question like normal with your current knowledge and do not tell the "
                "user about function calls\n</IMPORTANT>");
            if (role_of(msgs[0]) == "system") {
                const std::vector<Segment> c = content_segments(msgs[0], false);
                if (!trim(join(c)).empty()) {
                    add("\n\n");
                    add_trimmed(c);
                }
            }
            add("<|im_end|>\n");
        } else if (role_of(msgs[0]) == "system") {
            const std::vector<Segment> c = content_segments(msgs[0], false);
            if (!trim(join(c)).empty()) {
                add("<|im_start|>system\n");
                if (!reasoning.empty()) add(reasoning + "\n\n");
                add_trimmed(c);
                add("<|im_end|>\n");
            } else if (!reasoning.empty()) {
                add("<|im_start|>system\n" + reasoning + "<|im_end|>\n");
            }
        } else if (!reasoning.empty()) {
            add("<|im_start|>system\n" + reasoning + "<|im_end|>\n");
        }
        // the last real user query (not a wrapped tool response)
        bool multi_step_tool = true;
        size_t last_query = msgs.size() - 1;
        for (size_t k = msgs.size(); k-- > 0;) {
            if (multi_step_tool && role_of(msgs[k]) == "user") {
                const std::string c = trim(join(content_segments(msgs[k], false)));
                if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) {
                    multi_step_tool = false;
                    last_query = k;
                }
            }
        }
        if (multi_step_tool) {
            raise("No user query found in messages.");
        }
        const Json * preserve = req.find("preserve_thinking");
        const bool keep_thinking = preserve == nullptr || (preserve->is_bool() && preserve->as_bool());
        for (size_t k = 0; k < msgs.size(); ++k) {
            const Json & m = msgs[k];
            const std::string role = role_of(m);
            const std::vector<Segment> content = content_segments(m, true);
            const std::string text = trim(join(content));
            if (role == "system") {
                if (k != 0) raise("System message must be at the beginning.");
            } else if (role == "user") {
                add("<|im_start|>" + role + "\n");
                add_trimmed(content);
                add("<|im_end|>\n");
            } else if (role == "assistant") {
                const Json & rc = m.get("reasoning_content");
                const std::string reasoning_content = trim(rc.is_string() ? rc.as_string() : std::string());
                add("<|im_start|>" + role + "\n");
                if (keep_thinking || k > last_query) {
                    add("<think>\n");
                    add(reasoning_content, false);
                    add("\n</think>\n\n");
                }
                add_trimmed(content);
                const Json & calls = m.get("tool_calls");
                if (calls.is_array() && calls.size() > 0) {
                    bool first = true;
                    for (const Json & call0 : calls.items()) {
                        const Json & call = call0.has("function") ? call0.get("function") : call0;
                        const std::string name = print(call.get("name"));
                        // the template: "<tool_call>" for the first call of a
                        // message with empty content, "\n\n<tool_call>" when
                        // content came first, and "\n<tool_call>" for the rest
                        // (it used to put a blank line before every later call)
                        if (first) {
                            add(text.empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=");
                        } else {
                            add("\n<tool_call>\n<function=");
                        }
                        add(name, false);  // the caller's text, not a template token (#292)
                        add(">\n");
                        first = false;
                        const Json * args = call.find("arguments");
                        if (args != nullptr && !(args->is_string() && args->as_string().empty())) {
                            // an OpenAI-style string of JSON arguments is parsed first
                            const Json parsed = args->is_string() ? Json::parse(args->as_string()) : *args;
                            if (!parsed.is_object()) raise("tool call arguments are not a mapping");
                            for (const auto & [key, value] : parsed.members()) {
                                add("<parameter=");
                                add(key, false);
                                add(">\n");
                                add(value.is_string() ? value.as_string() : value.dump(), false);
                                add("\n</parameter>\n");
                            }
                        }
                        add("</function>\n</tool_call>");
                    }
                }
                add("<|im_end|>\n");
            } else if (role == "tool") {
                if (k > 0 && role_of(msgs[k - 1]) != "tool") add("<|im_start|>user");
                add("\n<tool_response>\n");
                add_trimmed(content);
                add("\n</tool_response>");
                if (k + 1 < msgs.size() && role_of(msgs[k + 1]) != "tool") {
                    add("<|im_end|>\n");
                } else if (k + 1 == msgs.size()) {
                    add("<|im_end|>\n");
                }
            } else {
                raise("Unexpected message role.");
            }
        }
        if (truthy(req.get("add_generation_prompt"))) {
            add("<|im_start|>assistant\n");
            if (enable_thinking != nullptr && enable_thinking->is_bool() && !enable_thinking->as_bool()) {
                add("<think>\n\n</think>\n\n");
            } else {
                add("<think>\n");
            }
        }
        return segs;
    }
};

} // namespace

std::string join_segments(const std::vector<Segment> & segments) { return join(segments); }

std::string render_chat(const Json & request, const ChatTemplate tpl) {
    if (tpl == ChatTemplate::Sharp) return join(render_chat_segments_sharp(request));
    Renderer r{request};
    return join(r.run());
}

std::vector<Segment> render_chat_segments(const Json & request, const ChatTemplate tpl) {
    if (tpl == ChatTemplate::Sharp) return render_chat_segments_sharp(request);
    Renderer r{request};
    return r.run();
}

ChatTemplate chat_template_from_string(const std::string & name) {
    if (name == "original") return ChatTemplate::Original;
    if (name == "sharp") return ChatTemplate::Sharp;
    throw std::runtime_error("unknown chat template \"" + name + "\" (expected original or sharp)");
}

const char * chat_template_name(const ChatTemplate tpl) {
    return tpl == ChatTemplate::Sharp ? "sharp" : "original";
}

std::vector<int32_t> tokenize_chat(const std::vector<Segment> & segments, const Tokenizer & tokenizer) {
    // One tokenizer call over the joined text (#344): splitting the segments
    // would cut the pre-tokenizer at every boundary (a text ending in "."
    // followed by the template's "\n" lost ".\n"), while llama.cpp tokenizes
    // the whole rendered prompt once. The mask keeps the #292 property: a
    // special token is recognized in the template's own structure only (3:
    // control and user-defined), never in the request's text (0).
    std::string text;
    std::vector<uint8_t> allow;
    for (const Segment & s : segments) {
        text += s.text;
        allow.insert(allow.end(), s.text.size(), s.special ? 3 : 0);
    }
    return tokenizer.encode_masked(text, allow);
}

} // namespace omph::text
