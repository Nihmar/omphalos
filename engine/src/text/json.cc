#include "text/json.hh"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <system_error>
#include <unordered_map>

namespace omph::text {
namespace {

const Json kNull;

[[noreturn]] void fail(const std::string & what, size_t at) {
    throw std::runtime_error("json: " + what + " at byte " + std::to_string(at));
}

void append_utf8(std::string & s, uint32_t cp) {
    if (cp < 0x80) {
        s += (char) cp;
    } else if (cp < 0x800) {
        s += (char) (0xC0 | (cp >> 6));
        s += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char) (0xE0 | (cp >> 12));
        s += (char) (0x80 | ((cp >> 6) & 0x3F));
        s += (char) (0x80 | (cp & 0x3F));
    } else {
        s += (char) (0xF0 | (cp >> 18));
        s += (char) (0x80 | ((cp >> 12) & 0x3F));
        s += (char) (0x80 | ((cp >> 6) & 0x3F));
        s += (char) (0x80 | (cp & 0x3F));
    }
}

struct Parser {
    std::string_view s;
    size_t i = 0;

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
            ++i;
        }
    }
    bool lit(std::string_view w) {
        if (s.substr(i, w.size()) == w) {
            i += w.size();
            return true;
        }
        return false;
    }
    uint32_t hex4() {
        if (i + 4 > s.size()) {
            fail("short \\u escape", i);
        }
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t) (c - 'A' + 10);
            else fail("bad \\u escape", i - 1);
        }
        return v;
    }
    std::string str() {
        if (i >= s.size() || s[i] != '"') {
            fail("expected a string", i);
        }
        ++i;
        std::string out;
        while (true) {
            if (i >= s.size()) {
                fail("unterminated string", i);
            }
            const char c = s[i++];
            if (c == '"') {
                return out;
            }
            if ((unsigned char) c < 0x20) {
                fail("control character in a string", i - 1);
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i >= s.size()) {
                fail("unterminated escape", i);
            }
            const char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && s.substr(i, 2) == "\\u") {
                        const size_t save = i;
                        i += 2;
                        const uint32_t lo = hex4();
                        if (lo >= 0xDC00 && lo < 0xE000) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else {
                            i = save;
                        }
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("bad escape", i - 1);
            }
        }
    }
    Json value(int depth) {
        if (depth > 256) {
            fail("nesting too deep", i);
        }
        ws();
        if (i >= s.size()) {
            fail("unexpected end", i);
        }
        const char c = s[i];
        if (c == '{') {
            ++i;
            std::vector<std::pair<std::string, Json>> members;
            ws();
            if (i < s.size() && s[i] == '}') {
                ++i;
                return Json::object_of(std::move(members));
            }
            while (true) {
                ws();
                std::string k = str();
                ws();
                if (i >= s.size() || s[i] != ':') {
                    fail("expected ':'", i);
                }
                ++i;
                members.emplace_back(std::move(k), value(depth + 1));
                ws();
                if (i < s.size() && s[i] == ',') {
                    ++i;
                    continue;
                }
                if (i < s.size() && s[i] == '}') {
                    ++i;
                    return Json::object_of(std::move(members));
                }
                fail("expected ',' or '}'", i);
            }
        }
        if (c == '[') {
            ++i;
            Json a = Json::array();
            ws();
            if (i < s.size() && s[i] == ']') {
                ++i;
                return a;
            }
            while (true) {
                a.push(value(depth + 1));
                ws();
                if (i < s.size() && s[i] == ',') {
                    ++i;
                    continue;
                }
                if (i < s.size() && s[i] == ']') {
                    ++i;
                    return a;
                }
                fail("expected ',' or ']'", i);
            }
        }
        if (c == '"') {
            return Json::string(str());
        }
        if (lit("true")) return Json::boolean(true);
        if (lit("false")) return Json::boolean(false);
        if (lit("null")) return Json();
        // number
        const size_t b = i;
        if (i < s.size() && s[i] == '-') ++i;
        bool integer = true;
        while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' ||
                                s[i] == 'E' || s[i] == '+' || s[i] == '-')) {
            if (s[i] == '.' || s[i] == 'e' || s[i] == 'E') integer = false;
            ++i;
        }
        if (i == b) {
            fail("unexpected character", i);
        }
        const std::string num(s.substr(b, i - b));
        // std::from_chars, not strtod: locale-independent by construction, so
        // a host program that called setlocale(LC_ALL, "") cannot make "0.5" a
        // bad number (#343). The grammar is strtod's minus the locale (JSON
        // has no leading '+' / "inf" anyway)
        double v = 0.0;
        const auto r = std::from_chars(num.data(), num.data() + num.size(), v);
        if (r.ec != std::errc() || r.ptr != num.data() + num.size()) {
            fail("bad number", b);
        }
        if (integer) {
            // keep the literal exactly while it fits an int64 (#345): a dumped
            // tool-call id of 19 digits must reach the client unrounded
            int64_t iv = 0;
            const auto ri = std::from_chars(num.data(), num.data() + num.size(), iv);
            if (ri.ec == std::errc() && ri.ptr == num.data() + num.size()) {
                return Json::integer(iv);
            }
        }
        return Json::number(v, integer);
    }
};

void dump_string(std::string & out, const std::string & s) {
    out += '"';
    for (const char ch : s) {
        const auto c = (unsigned char) ch;
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {  // Python escapes only the C0 controls (not 0x7F)
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += ch;  // ensure_ascii=False: UTF-8 as is
                }
        }
    }
    out += '"';
}

// Python's repr of a float: the shortest round trip, with ".0" on integral
// values and exponent form outside [1e-4, 1e16).
void dump_number(std::string & out, const Json & j) {
    if (j.number_is_exact()) {
        out += std::to_string(j.as_int64());  // no '.' ever, so the C locale
        return;
    }
    const double v = j.as_number();
    const bool integer = j.number_is_integer();
    if (integer && std::fabs(v) < 9.007199254740992e15) {
        out += std::to_string((long long) v);
        return;
    }
    if (std::isnan(v)) {
        out += "NaN";
        return;
    }
    if (std::isinf(v)) {
        out += v > 0 ? "Infinity" : "-Infinity";
        return;
    }
    char buf[64];
    const double a = std::fabs(v);
    if (a != 0.0 && (a < 1e-4 || a >= 1e16)) {
        auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
        std::string t(buf, r.ptr);
        // Python writes e+16 / e-05 (two-digit exponent with sign)
        const size_t e = t.find('e');
        std::string mant = t.substr(0, e);
        std::string ex = t.substr(e + 1);
        const char sign = ex[0] == '-' ? '-' : '+';
        if (ex[0] == '-' || ex[0] == '+') ex = ex.substr(1);
        while (ex.size() < 2) ex = "0" + ex;
        out += mant + "e" + sign + ex;
        return;
    }
    auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::fixed);
    std::string t(buf, r.ptr);
    if (t.find('.') == std::string::npos) t += ".0";
    out += t;
}

void dump_value(std::string & out, const Json & j) {
    switch (j.type()) {
        case Json::Type::Null: out += "null"; break;
        case Json::Type::Bool: out += j.as_bool() ? "true" : "false"; break;
        case Json::Type::Number: dump_number(out, j); break;
        case Json::Type::String: dump_string(out, j.as_string()); break;
        case Json::Type::Array: {
            out += '[';
            bool first = true;
            for (const Json & e : j.items()) {
                if (!first) out += ", ";
                first = false;
                dump_value(out, e);
            }
            out += ']';
            break;
        }
        case Json::Type::Object: {
            out += '{';
            bool first = true;
            for (const auto & [k, v] : j.members()) {
                if (!first) out += ", ";
                first = false;
                dump_string(out, k);
                out += ": ";
                dump_value(out, v);
            }
            out += '}';
            break;
        }
    }
}

} // namespace

Json Json::boolean(const bool b) {
    Json j;
    j.type_ = Type::Bool;
    j.b_ = b;
    return j;
}
Json Json::number(const double v, const bool integer) {
    Json j;
    j.type_ = Type::Number;
    j.num_ = v;
    j.integer_ = integer;
    return j;
}
Json Json::integer(const int64_t v) {
    Json j;
    j.type_ = Type::Number;
    j.num_ = (double) v;
    j.integer_ = true;
    j.exact_ = true;
    j.int_ = v;
    return j;
}
Json Json::object_of(std::vector<std::pair<std::string, Json>> members) {
    Json j = Json::object();
    std::unordered_map<std::string, size_t> index;
    index.reserve(members.size());
    for (auto & m : members) {
        const auto it = index.find(m.first);
        if (it != index.end()) {
            j.obj_[(size_t) it->second].second = std::move(m.second);  // last value, first position
        } else {
            index.emplace(m.first, j.obj_.size());
            j.obj_.push_back(std::move(m));
        }
    }
    return j;
}
Json Json::string(std::string s) {
    Json j;
    j.type_ = Type::String;
    j.str_ = std::move(s);
    return j;
}
Json Json::array() {
    Json j;
    j.type_ = Type::Array;
    return j;
}
Json Json::object() {
    Json j;
    j.type_ = Type::Object;
    return j;
}

Json Json::parse(const std::string_view text) {
    Parser p{text};
    Json v = p.value(0);
    p.ws();
    if (p.i != text.size()) {
        fail("trailing characters", p.i);
    }
    return v;
}

std::string Json::dump() const {
    std::string out;
    dump_value(out, *this);
    return out;
}

const Json * Json::find(const std::string_view key) const {
    if (type_ != Type::Object) {
        return nullptr;
    }
    // the last duplicate wins, as in Python's json.loads
    for (auto it = obj_.rbegin(); it != obj_.rend(); ++it) {
        if (it->first == key) {
            return &it->second;
        }
    }
    return nullptr;
}

const Json & Json::get(const std::string_view key) const {
    const Json * v = find(key);
    return v != nullptr ? *v : kNull;
}

void Json::set(std::string key, Json v) {
    for (auto & kv : obj_) {
        if (kv.first == key) {
            kv.second = std::move(v);
            return;
        }
    }
    obj_.emplace_back(std::move(key), std::move(v));
}

} // namespace omph::text
