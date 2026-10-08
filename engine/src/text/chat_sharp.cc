// The Qwen Sharp chat template (#392) as code: engine/templates/
// qwen-sharp-v22.5.0.jinja (peculiar-ragdoll/Qwen-Sharp-Chat-Templates,
// Apache-2.0) rewritten in C++, byte for byte as jinja2 renders it
// (tools/check_chat_template.py --template sharp) and with the same
// structure/request-text split as the GGUF template (#292, #344): the
// template's own literals are `special`, the request's strings are not.
#include "text/chat.hh"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace omph::text {
namespace {

[[noreturn]] void raise(const std::string & msg) { throw std::runtime_error("chat template: " + msg); }

// Python's str.isspace() code points (what jinja2's |trim strips)
bool py_space(const uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

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
    if (b0 >= 0xF0 && b0 < 0xF8 && cont(i + 1) && cont(i + 2) && cont(i + 3)) {
        len = 4;
        return ((b0 & 0x07u) << 18) | (((uint8_t) s[i + 1] & 0x3Fu) << 12) |
               (((uint8_t) s[i + 2] & 0x3Fu) << 6) | ((uint8_t) s[i + 3] & 0x3Fu);
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

// Python's len(): the number of code points (an invalid byte counts as one)
size_t py_len(const std::string & s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size(); ++n) {
        size_t len = 0;
        (void) cp_at(s, i, len);
        i += len;
    }
    return n;
}

// The byte length of the first `max` code points
size_t cp_prefix_len(const std::string & s, const size_t max) {
    size_t i = 0;
    for (size_t n = 0; i < s.size() && n < max; ++n) {
        size_t len = 0;
        (void) cp_at(s, i, len);
        i += len;
    }
    return i;
}

// Python's str.lower() for the checks the template makes (all ASCII markers;
// the two non-ASCII code points that lowercase into ASCII are handled)
std::string py_lower(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        size_t len = 0;
        const uint32_t cp = cp_at(s, i, len);
        i += len;
        if (cp < 128) {
            out += (char) (cp >= 'A' && cp <= 'Z' ? cp + 32 : cp);
        } else if (cp == 0x212A) {  // KELVIN SIGN -> k
            out += 'k';
        } else if (cp == 0x0130) {  // LATIN CAPITAL I WITH DOT ABOVE -> "i" + U+0307
            out += "i\xCC\x87";
        } else {
            out.append(s, i - len, len);
        }
    }
    return out;
}

bool starts_with(const std::string & s, const std::string & p) { return s.rfind(p, 0) == 0; }
bool ends_with(const std::string & s, const std::string & p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

bool contains(const std::string & s, const std::string & needle) {
    return s.find(needle) != std::string::npos;
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

// Python str() of a value: the |string filter. Arrays and objects use Python's
// repr (single quotes), strings themselves.
std::string py_repr(const Json & j) {
    if (j.is_string()) return "'" + j.as_string() + "'";
    if (j.is_null()) return "None";
    if (j.is_bool()) return j.as_bool() ? "True" : "False";
    if (j.is_number()) return j.dump();
    if (j.is_array()) {
        std::string out = "[";
        bool first = true;
        for (const Json & e : j.items()) {
            if (!first) out += ", ";
            first = false;
            out += py_repr(e);
        }
        return out + "]";
    }
    std::string out = "{";
    bool first = true;
    for (const auto & [k, v] : j.members()) {
        if (!first) out += ", ";
        first = false;
        out += "'" + k + "': " + py_repr(v);
    }
    return out + "}";
}

std::string py_str(const Json & j) { return j.is_string() ? j.as_string() : py_repr(j); }

// A string under construction with a per-byte flag: 1 = the template's own
// structure (a special token written literally is parsed), 0 = the request's
// text (#292). The Python-level string operations the Sharp template makes on
// a message's content (trim, split, truncate to N characters, ...) keep the
// flags, so the split survives them.
struct Piece {
    std::string text;
    std::vector<uint8_t> special;

    void add(std::string s, const bool special_flag) {
        if (s.empty()) return;
        special.insert(special.end(), s.size(), special_flag ? 1 : 0);
        text += s;
    }
    void add(const Piece & p) {
        text += p.text;
        special.insert(special.end(), p.special.begin(), p.special.end());
    }
    bool empty() const { return text.empty(); }

    static Piece from(const std::vector<Segment> & segs) {
        Piece p;
        for (const Segment & s : segs) {
            p.add(s.text, s.special);
        }
        return p;
    }
    std::vector<Segment> segments() const {
        std::vector<Segment> out;
        size_t at = 0;
        while (at < text.size()) {
            const bool sp = special[at] != 0;
            size_t to = at;
            while (to < text.size() && (special[to] != 0) == sp) ++to;
            out.push_back({text.substr(at, to - at), sp});
            at = to;
        }
        return out;
    }
    Piece slice(const size_t b, const size_t e) const {
        Piece p;
        p.text = text.substr(b, e - b);
        p.special.assign(special.begin() + (std::ptrdiff_t) b, special.begin() + (std::ptrdiff_t) e);
        return p;
    }
    Piece trimmed() const {
        const std::string t = trim(text);
        if (t.empty()) return {};
        const size_t b = text.find(t);
        return slice(b, b + t.size());
    }
    // Python's .split(needle)[-1] / [0]
    Piece after_last(const std::string & needle) const {
        const size_t at = text.rfind(needle);
        return at == std::string::npos ? *this : slice(at + needle.size(), text.size());
    }
    Piece before_first(const std::string & needle) const {
        const size_t at = text.find(needle);
        return at == std::string::npos ? *this : slice(0, at);
    }
    void lstrip_char(const char c) {
        size_t at = 0;
        while (at < text.size() && text[at] == c) ++at;
        text.erase(0, at);
        special.erase(special.begin(), special.begin() + (std::ptrdiff_t) at);
    }
    void rstrip_char(const char c) {
        size_t e = text.size();
        while (e > 0 && text[e - 1] == c) --e;
        text.resize(e);
        special.resize(e);
    }
    // content[:max] by characters
    void truncate_cp(const size_t max) {
        const size_t n = cp_prefix_len(text, max);
        text.resize(n);
        special.resize(n);
    }
    // Every occurrence of `needle` removed (the template's split()|join(''))
    void erase_all(const std::string & needle) {
        if (needle.empty()) return;
        std::vector<std::pair<size_t, size_t>> ranges;
        for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size())) {
            ranges.emplace_back(at, at + needle.size());
        }
        for (auto it = ranges.rbegin(); it != ranges.rend(); ++it) {
            text.erase(it->first, it->second - it->first);
            special.erase(special.begin() + (std::ptrdiff_t) it->first,
                          special.begin() + (std::ptrdiff_t) it->second);
        }
    }
};

class SharpRenderer {
public:
    explicit SharpRenderer(const Json & request) : req_(request) {}

    std::vector<Segment> run() {
        const Json & messages = req_.get("messages");
        if (!truthy(messages) || !messages.is_array()) raise("No messages provided.");

        tool_format_ = req_.get("tool_call_format").is_string() ? req_.get("tool_call_format").as_string() : "xml";
        add_vision_id_ = truthy(req_.get("add_vision_id"));
        // `max_tool_arg_chars if defined else 0`: a positive number, anything
        // else is treated as 0 (the template only compares it to 0)
        const Json & mtac = req_.get("max_tool_arg_chars");
        max_tool_arg_chars_ = mtac.is_number() && mtac.as_number() > 0 ? (int64_t) mtac.as_number() : 0;
        const Json & mtrc = req_.get("max_tool_response_chars");
        max_tool_response_chars_ = mtrc.is_number() && mtrc.as_number() > 0 ? (int64_t) mtrc.as_number() : 0;
        const Json & et = req_.get("enable_thinking");
        bool thinking = et.is_null() || truthy(et);  // `enable_thinking if defined else true`
        // `_effort_raw = (reasoning_effort | string | lower) if defined and not none else 'medium'`
        const Json & eff_j = req_.get("reasoning_effort");
        const std::string effort_raw = eff_j.is_null() ? "medium" : py_lower(py_str(eff_j));
        std::string effort;
        if (effort_raw == "none" || effort_raw == "off") {
            thinking = false;
            effort = "medium";
        } else if (effort_raw == "minimal" || effort_raw == "low") {
            effort = "low";
        } else if (effort_raw == "high" || effort_raw == "xhigh" || effort_raw == "max" ||
                   effort_raw == "ultracode" || effort_raw == "extreme") {
            effort = "xhigh";
        } else {
            effort = "medium";
        }
        const Json & auto_disable = req_.get("auto_disable_thinking_with_tools");
        const bool has_tools = tools().is_array() && tools().size() > 0;
        if ((auto_disable.is_null() ? false : truthy(auto_disable)) && has_tools) {
            thinking = false;
        }
        // The `<|think_*|>` markers in system/user content re-set the state as
        // the conversation is scanned; this pass is the whole conversation.
        for (const Json & m : messages.items()) {
            const std::string role = role_of(m);
            if (role != "system" && role != "developer" && role != "user") continue;
            const Json & content = m.get("content");
            if (content.is_string()) {
                apply_marker(content.as_string(), thinking, effort);
            } else if (content.is_array()) {
                for (const Json & item : content.items()) {
                    std::string t;
                    if (item.is_string()) {
                        t = item.as_string();
                    } else if (item.is_object() && item.get("text").is_string()) {
                        t = item.get("text").as_string();
                    }
                    if (!t.empty()) apply_marker(t, thinking, effort);
                }
            }
        }
        std::string reasoning_instructions;
        if (thinking) {
            if (effort == "xhigh") {
                reasoning_instructions =
                    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
                    "clarity in the final answer.";
            } else if (effort == "low") {
                reasoning_instructions =
                    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                    "conclusion without unnecessary elaboration.";
            }
        }

        // The leading system/developer block, joined with blank lines, marker
        // free (`sys_state.content`)
        size_t head_count = 0;
        {
            bool seen_non_system = false;
            for (const Json & m : messages.items()) {
                const std::string role = role_of(m);
                const bool is_sys = role == "system" || role == "developer";
                if (is_sys && !seen_non_system) {
                    ++head_count;
                } else {
                    seen_non_system = true;
                }
            }
        }
        Piece sys_content;
        for (size_t k = 0; k < head_count; ++k) {
            Piece part = render_content(messages.items()[k].find("content"), false, true).trimmed();
            for (const char * marker : kThinkMarkers) part.erase_all(marker), part = part.trimmed();
            if (part.empty()) continue;
            if (!sys_content.empty()) {
                sys_content.add("\n\n", true);
                sys_content.add(part);
            } else {
                sys_content.add(part);
            }
        }
        Piece sc = sys_content;
        const std::string terse_lead =
            thinking ? "Answer directly, after thinking. Lead with the answer, then only what it needs to be "
                       "correct and usable."
                     : "Answer directly and concisely. Give the answer with only what it needs to be correct and "
                       "usable.";
        const std::string terse_core_raw =
            "\nNever: open with preamble or pleasantries; restate the question; add filler transitions; hedge with "
            "niceties; or repeat a point you've already made.\nAlways: keep essential steps, caveats, uncertainties, "
            "and specifics — never drop correctness or a needed warning for brevity. Keep the final answer lean. Use "
            "the least structure that conveys it (plain prose when short; lists or code only when they earn their "
            "place). If genuinely uncertain, say so and explain why — never omit uncertainty for the sake of "
            "brevity.\nIf a user request is genuinely ambiguous, ask a sharp question, don't guess.\n";
        const std::string terse = terse_lead + "\n" + trim(terse_core_raw);
        const Json & terse_kw = req_.get("terse");
        const bool terse_on = terse_kw.is_null() ? true : truthy(terse_kw);
        if (terse_on) {
            Piece t;
            t.add(terse, true);  // the template's own instruction text
            if (sc.empty()) {
                sc = t.trimmed();
            } else {
                sc = sc.trimmed();
                sc.add("\n\n", true);
                sc.add(t.trimmed());
            }
        }
        const Json & suppress_kw = req_.get("suppress_tool_instructions");
        const bool suppress_tools =
            suppress_kw.is_null() ? (!sc.empty() && contains(sc.text, "[TOOL_REQUEST]")) : truthy(suppress_kw);

        if (has_tools && !suppress_tools) {
            add("<|im_start|>system\n");
            if (!reasoning_instructions.empty()) add(reasoning_instructions + "\n\n");
            add("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for (const Json & t : tools().items()) {
                add("\n");
                add(t.dump(), false);  // a tool definition is the client's own text (#292)
            }
            add("\n</tools>");
            if (tool_format_ == "json") {
                add("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n");
                if (thinking) add("<think>\nBrief explanation of tool call\n</think>\n");
                add("<tool_call>\n{\"name\": \"example_function_name\", \"arguments\": {\"example_parameter_1\": "
                    "\"value_1\", \"example_parameter_2\": \"This is the value for the second parameter\"}}\n"
                    "</tool_call>\n\n<IMPORTANT>\nReminder:\n");
                if (thinking) {
                    add("- You can use the <think></think> block to plan your next tool call OR to synthesize data "
                        "and formulate your final response to the user.\n- ALL explanation and reasoning MUST be "
                        "placed strictly inside the <think></think> block.\n");
                }
                add("- Function calls MUST follow the specified format: a single JSON object with \"name\" and "
                    "\"arguments\" keys inside <tool_call></tool_call> XML tags.\n");
                if (thinking) {
                    add("- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after "
                        "thinking, with NO conversational text before it.\n");
                } else {
                    add("- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY, with NO "
                        "conversational text before it.\n");
                }
                add("- The <tool_call> tag MUST be at the very beginning of a new line, with NO spaces or "
                    "indentation before it.\n- To call multiple functions, output a separate, completely closed "
                    "<tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.\n- If you have "
                    "all necessary data, provide your final answer directly to the user without any tool call.\n"
                    "</IMPORTANT>");
            } else {
                add("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n");
                if (thinking) add("<think>\nBrief explanation of tool call\n</think>\n");
                add("<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
                    "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
                    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
                    "Reminder:\n");
                if (thinking) {
                    add("- You can use the <think></think> block to plan your next tool call OR to synthesize data "
                        "and formulate your final response to the user.\n- ALL explanation and reasoning MUST be "
                        "placed strictly inside the <think></think> block.\n");
                }
                add("- Function calls MUST follow the specified format: an inner <function=...></function> block "
                    "must be nested within <tool_call></tool_call> XML tags.\n");
                if (thinking) {
                    add("- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after "
                        "thinking, with NO conversational text before it.\n");
                } else {
                    add("- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY, with NO "
                        "conversational text before it.\n");
                }
                add("- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with NO "
                    "spaces or indentation before them.\n- To call multiple functions, output a separate, "
                    "completely closed <tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> "
                    "blocks.\n- If you have all necessary data, provide your final answer directly to the user "
                    "without any tool call.\n</IMPORTANT>");
            }
            if (!sc.empty()) {
                add("\n\n");
                add(sc);
            }
            add("<|im_end|>\n");
        } else if (!sc.empty()) {
            add("<|im_start|>system\n");
            if (!reasoning_instructions.empty()) add(reasoning_instructions + "\n\n");
            add(sc);
            add("<|im_end|>\n");
        } else if (!reasoning_instructions.empty()) {
            add("<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n");
        }

        // `_msgs = messages[head.count:]`
        std::vector<const Json *> msgs;
        for (size_t k = head_count; k < messages.size(); ++k) msgs.push_back(&messages.items()[k]);
        const int64_t last_idx = (int64_t) msgs.size() - 1;
        // The last real user query (not a wrapped tool response), from the end
        int64_t last_query_index = last_idx;
        {
            bool multi_step_tool = true;
            for (int64_t k = (int64_t) msgs.size() - 1; k >= 0; --k) {
                const Json & m = *msgs[(size_t) k];
                if (multi_step_tool && role_of(m) == "user") {
                    const std::string rc = py_trim(render_content(m.find("content"), false, false));
                    if (!(starts_with(rc, "<tool_response>") && ends_with(rc, "</tool_response>"))) {
                        multi_step_tool = false;
                        last_query_index = k;
                    }
                }
            }
            if (multi_step_tool) {
                last_query_index = last_idx > 50 ? last_idx : 0;
            }
        }

        // `_preserve_thinking`: preserve_reasoning, then preserve_thinking, else true
        const Json & pr = req_.get("preserve_reasoning");
        const Json & pt = req_.get("preserve_thinking");
        const bool preserve_thinking =
            !pr.is_null() ? truthy(pr) : (!pt.is_null() ? truthy(pt) : true);

        std::string prev_role;
        int64_t consecutive_failures = 0;
        for (size_t k = 0; k < msgs.size(); ++k) {
            const Json & m = *msgs[k];
            const std::string role = role_of(m);
            const bool is_system = role == "system" || role == "developer";
            Piece content = render_content(m.find("content"), true, is_system).trimmed();
            if (is_system || role == "user") {
                for (const char * marker : kThinkMarkers) {
                    if (contains(content.text, marker)) {
                        content.erase_all(marker);
                        content = content.trimmed();
                    }
                }
            }
            if (is_system) {
                add("<|im_start|>system\n");
                add(content);
                add("<|im_end|>\n");
            } else if (role == "user") {
                consecutive_failures = 0;
                add("<|im_start|>user\n");
                add(content);
                add("<|im_end|>\n");
            } else if (role == "assistant") {
                Piece reasoning;
                // the first defined reasoning_source: reasoning_content, thinking, reasoning
                for (const char * key : {"reasoning_content", "thinking", "reasoning"}) {
                    const Json & v = m.get(key);
                    if (!v.is_null()) {
                        reasoning.add(py_str(v), false);
                        break;
                    }
                }
                if (!reasoning.empty()) {
                    std::string lead_end;
                    const std::string & c = content.text;
                    if (starts_with(c, "<think>") && contains(c, "</think>")) {
                        lead_end = "</think>";
                    } else if (starts_with(c, "<thinking>") && contains(c, "</thinking>")) {
                        lead_end = "</thinking>";
                    } else if (starts_with(c, "</think>")) {
                        lead_end = "</think>";
                    } else if (starts_with(c, "</thinking>")) {
                        lead_end = "</thinking>";
                    }
                    if (!lead_end.empty()) {
                        content = content.after_last(lead_end);
                        content.lstrip_char('\n');
                    }
                } else {
                    std::string think_end;
                    const std::string & c = content.text;
                    if (starts_with(c, "</think>")) {
                        think_end = "</think>";
                    } else if (starts_with(c, "</thinking>")) {
                        think_end = "</thinking>";
                    } else if (contains(c, "\n</think>")) {
                        think_end = "\n</think>";
                    } else if (contains(c, "\n</thinking>")) {
                        think_end = "\n</thinking>";
                    } else if (contains(c, "\n</ think>")) {
                        think_end = "\n</ think>";
                    } else if (contains(c, "\n</think >")) {
                        think_end = "\n</think >";
                    } else if (starts_with(c, "<think>") && contains(c, "</think>")) {
                        think_end = "</think>";
                    } else if (starts_with(c, "<thinking>") && contains(c, "</thinking>")) {
                        think_end = "</thinking>";
                    }
                    if (!think_end.empty()) {
                        const std::string think_start = contains(think_end, "thinking") ? "<thinking>" : "<think>";
                        reasoning = content.before_first(think_end);
                        reasoning.rstrip_char('\n');
                        if (contains(reasoning.text, think_start)) {
                            reasoning = reasoning.after_last(think_start);
                            reasoning.lstrip_char('\n');
                        }
                        content = content.after_last(think_end);
                        content.lstrip_char('\n');
                    }
                }
                reasoning = reasoning.trimmed();
                if (preserve_thinking || (int64_t) k > last_query_index) {
                    add("<|im_start|>assistant\n<think>\n");
                    add(reasoning);
                    add("\n</think>\n\n");
                } else {
                    add("<|im_start|>assistant\n");
                }
                add(content);
                const Json & calls = m.get("tool_calls");
                if (calls.is_array() && calls.size() > 0) {
                    bool first = true;
                    for (const Json & call0 : calls.items()) {
                        const Json & call = call0.is_object() && !call0.get("function").is_null()
                                                ? call0.get("function")
                                                : call0;
                        const Json & name_j = call.get("name");
                        const std::string tc_name = name_j.is_null() ? "" : print(name_j);
                        if (tool_format_ == "json") {
                            if (first) {
                                if (!trim(content.text).empty()) add("\n\n");
                            } else {
                                add("\n");
                            }
                            std::string args = "{}";
                            const Json & a = call.get("arguments");
                            if (!a.is_null()) {
                                if (a.is_object() || a.is_array()) {
                                    args = a.dump();
                                } else if (a.is_string()) {
                                    if (!a.as_string().empty()) args = a.as_string();
                                } else {
                                    args = a.dump();
                                }
                            }
                            add("<tool_call>\n{\"name\": ");
                            add(Json::string(tc_name).dump(), false);
                            add(", \"arguments\": ");
                            add(args, false);
                            add("}\n</tool_call>");
                        } else {
                            if (first) {
                                if (!trim(content.text).empty()) {
                                    add("\n\n<tool_call>\n<function=");
                                } else {
                                    add("<tool_call>\n<function=");
                                }
                            } else {
                                add("\n<tool_call>\n<function=");
                            }
                            add(tc_name, false);
                            add(">\n");
                            const Json & a = call.get("arguments");
                            if (!a.is_null()) {
                                if (a.is_object()) {
                                    for (const auto & [key, value] : a.members()) {
                                        add("<parameter=");
                                        add(key, false);
                                        add(">\n");
                                        const std::string av = value.is_string() ? value.as_string() : value.dump();
                                        add_truncated(av, max_tool_arg_chars_);
                                        add("\n</parameter>\n");
                                    }
                                } else {
                                    const std::string raw = a.is_string() ? a.as_string() : a.dump();
                                    if (!raw.empty()) add_truncated(raw, max_tool_arg_chars_);
                                }
                            }
                            add("</function>\n</tool_call>");
                        }
                        first = false;
                    }
                }
                add("<|im_end|>\n");
            } else if (role == "tool") {
                const std::string lower = py_lower(content.text);
                const std::string lower_head = lower.substr(0, cp_prefix_len(lower, 120));
                const bool code_or_grep =
                    contains(lower, "throw new ") || contains(lower, "throw error") ||
                    contains(lower, "console.error") || contains(lower, "logger.error") ||
                    contains(lower, "logging.error") || contains(lower_head, "import ") ||
                    contains(lower_head, "def ") || contains(lower_head, "function ");
                const bool exit_code_zero = contains(lower_head, "exit code: 0") ||
                                            contains(lower_head, "process exited with code 0");
                const bool error_field_ok =
                    contains(lower_head, "\"error\": null") || contains(lower_head, "\"error\":null") ||
                    contains(lower_head, "\"error\": false") || contains(lower_head, "\"error\":false") ||
                    contains(lower_head, "\"error\": \"\"") || contains(lower_head, "\"error\":\"\"");
                const bool strong_error =
                    (contains(lower_head, "\"error\":") && !error_field_ok) ||
                    contains(lower_head, "\"status\": \"error\"") || contains(lower_head, "\"status\":\"error\"") ||
                    contains(lower_head, "traceback (most recent call last):") ||
                    contains(lower_head, "command not found") || contains(lower_head, "invalid syntax") ||
                    contains(lower_head, "fatal:") ||
                    ((contains(lower_head, "exit code: ") || contains(lower_head, "process exited with code")) &&
                     !exit_code_zero) ||
                    starts_with(lower_head, "exception:") || starts_with(lower_head, "failed to ");
                const bool weak_error = contains(lower_head, "error:") || contains(lower_head, "err!");
                const bool weak_suppressed =
                    contains(lower_head, "$ ") || contains(lower_head, "took ") || py_len(content.text) >= 600;
                if (!code_or_grep && (strong_error || (weak_error && !weak_suppressed))) {
                    ++consecutive_failures;
                } else {
                    consecutive_failures = 0;
                }
                if (prev_role != "tool") add("<|im_start|>user");
                const std::string trimmed = trim(content.text);
                const bool json_payload = tool_format_ == "json" && !trimmed.empty() &&
                                          (trimmed[0] == '{' || trimmed[0] == '[');
                if (!json_payload && max_tool_response_chars_ > 0 &&
                    (int64_t) py_len(content.text) > max_tool_response_chars_) {
                    const size_t n = py_len(content.text);
                    content.truncate_cp(max_tool_response_chars_);
                    content.add("\n[TRUNCATED - original length " + std::to_string(n) + " chars]", true);
                }
                add("\n<tool_response>\n");
                add(content);
                if (consecutive_failures >= 2) {
                    add("\n\n⚠️ SYSTEM WARNING: " + std::to_string(consecutive_failures) +
                        " consecutive tool errors detected. Your previous approach is incorrect. You MUST use a "
                        "fundamentally different approach or corrected arguments.");
                } else if (consecutive_failures == 1) {
                    add("\n\n⚠️ SYSTEM WARNING: The previous tool call returned an error. Diagnose the failure and "
                        "retry with completely corrected arguments.");
                }
                add("\n</tool_response>");
                if (k + 1 == msgs.size()) {
                    add("<|im_end|>\n");
                } else if (role_of(*msgs[k + 1]) != "tool") {
                    add("<|im_end|>\n");
                }
            } else {
                add("<|im_start|>user\n[" + role + "]: ");
                add(content);
                add("<|im_end|>\n");
            }
            prev_role = role;
        }
        if (truthy(req_.get("add_generation_prompt"))) {
            add("<|im_start|>assistant\n");
            if (!thinking) {
                add("<think>\n\n</think>\n\n");
            } else {
                add("<think>\n");
            }
        }
        return out_.segments();
    }

private:
    static constexpr const char * kThinkMarkers[] = {
        "<|think_off|>", "<|think_on|>", "<|think_xhigh|>", "<|think_high|>", "<|think_ultracode|>",
        "<|think_extreme|>", "<|think_max|>", "<|think_medium|>", "<|think_low|>", "<|think_minimal|>"};

    const Json & req_;
    std::string tool_format_ = "xml";
    Piece out_;
    int64_t max_tool_arg_chars_ = 0;
    int64_t max_tool_response_chars_ = 0;
    bool add_vision_id_ = false;
    int image_count_ = 0;
    int video_count_ = 0;

    const Json & tools() const {
        static const Json kNull;
        const Json * t = req_.find("tools");
        return t == nullptr ? kNull : *t;
    }
    static std::string role_of(const Json & m) { return m.get("role").is_string() ? m.get("role").as_string() : ""; }

    void add(std::string text, const bool special = true) { out_.add(std::move(text), special); }
    void add(const Piece & p) { out_.add(p); }
    void add_truncated(const std::string & s, const int64_t max) {
        if (max > 0 && (int64_t) py_len(s) > max) {
            Piece p;
            p.add(s, false);
            const size_t n = py_len(s);
            p.truncate_cp((size_t) max);
            add(p);
            add("\n[TRUNCATED - original length " + std::to_string(n) + " chars]");
        } else {
            add(s, false);
        }
    }

    void apply_marker(const std::string & text, bool & thinking, std::string & effort) const {
        if (contains(text, "<|think_off|>")) {
            thinking = false;
        } else if (contains(text, "<|think_on|>")) {
            thinking = true;
        } else if (contains(text, "<|think_xhigh|>") || contains(text, "<|think_high|>") ||
                   contains(text, "<|think_ultracode|>") || contains(text, "<|think_extreme|>") ||
                   contains(text, "<|think_max|>")) {
            thinking = true;
            effort = "xhigh";
        } else if (contains(text, "<|think_low|>") || contains(text, "<|think_minimal|>")) {
            thinking = true;
            effort = "low";
        } else if (contains(text, "<|think_medium|>")) {
            thinking = true;
            effort = "medium";
        }
    }

    // The template's render_content macro: a string as is; an array of
    // text/image/video parts; null as nothing; anything else raises.
    Piece render_content(const Json * content, const bool do_vision_count, const bool is_system) {
        Piece p;
        if (content == nullptr || content->is_null()) return p;
        if (content->is_string()) {
            p.add(content->as_string(), false);
            return p;
        }
        if (!content->is_array()) raise("Unexpected content type.");
        for (const Json & item : content->items()) {
            const bool image = item.is_object()
                                   ? (item.get("type").is_string() && item.get("type").as_string() == "image") ||
                                         item.has("image") || item.has("image_url")
                                   : (item.is_string() && contains(item.as_string(), "image"));
            const bool video = item.is_object()
                                   ? (item.get("type").is_string() && item.get("type").as_string() == "video") ||
                                         item.has("video") || item.has("video_url")
                                   : (item.is_string() && contains(item.as_string(), "video"));
            if (image) {
                if (is_system) raise("System message cannot contain images.");
                if (do_vision_count) ++image_count_;
                if (add_vision_id_) p.add("Picture " + std::to_string(image_count_) + ": ", true);
                p.add("<|vision_start|><|image_pad|><|vision_end|>", true);
            } else if (video) {
                if (is_system) raise("System message cannot contain videos.");
                if (do_vision_count) ++video_count_;
                if (add_vision_id_) p.add("Video " + std::to_string(video_count_) + ": ", true);
                p.add("<|vision_start|><|video_pad|><|vision_end|>", true);
            } else if (item.is_object() && item.has("text")) {
                p.add(print(item.get("text")), false);
            } else if (!item.is_object()) {
                p.add(item.is_string() ? item.as_string() : print(item), false);
            } else {
                raise("Unexpected item type in content.");
            }
        }
        return p;
    }

    // The template joins content segments with no separator; `render_content`
    // in the Sharp template returns a string, `| trim` follows.
    static std::string py_trim(const Piece & p) { return trim(p.text); }
};

} // namespace

std::vector<Segment> render_chat_segments_sharp(const Json & request) {
    SharpRenderer r{request};
    return r.run();
}

} // namespace omph::text
