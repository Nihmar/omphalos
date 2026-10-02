#include "text/tokenizer.hh"

#include <algorithm>
#include <stdexcept>

#include "text/unicode_tables.hh"

namespace omph::text {
namespace {

struct Cp {
    uint32_t cp;
    uint32_t off;  // byte offset in the text
    uint8_t len;   // its UTF-8 length
    uint8_t fl;    // kCp* flags
};

uint8_t cp_flags(const uint32_t cp) {
    // binary search over the sorted, disjoint ranges
    size_t lo = 0;
    size_t hi = sizeof(kCpRanges) / sizeof(kCpRanges[0]);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < kCpRanges[mid].first) {
            hi = mid;
        } else if (cp > kCpRanges[mid].last) {
            lo = mid + 1;
        } else {
            return kCpRanges[mid].flags;
        }
    }
    return 0;
}

// UTF-8 to code points; an invalid byte becomes U+FFFD of length 1 (class: none)
std::vector<Cp> decode_utf8(const std::string_view s) {
    std::vector<Cp> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const auto b0 = (uint8_t) s[i];
        uint32_t cp = 0xFFFD;
        int len = 1;
        const auto cont = [&](const size_t k) {
            return k < s.size() && ((uint8_t) s[k] & 0xC0) == 0x80;
        };
        if (b0 < 0x80) {
            cp = b0;
        } else if ((b0 & 0xE0) == 0xC0 && cont(i + 1)) {
            cp = ((b0 & 0x1Fu) << 6) | ((uint8_t) s[i + 1] & 0x3Fu);
            len = 2;
        } else if ((b0 & 0xF0) == 0xE0 && cont(i + 1) && cont(i + 2)) {
            cp = ((b0 & 0x0Fu) << 12) | (((uint8_t) s[i + 1] & 0x3Fu) << 6) |
                 ((uint8_t) s[i + 2] & 0x3Fu);
            len = 3;
        } else if ((b0 & 0xF8) == 0xF0 && cont(i + 1) && cont(i + 2) && cont(i + 3)) {
            cp = ((b0 & 0x07u) << 18) | (((uint8_t) s[i + 1] & 0x3Fu) << 12) |
                 (((uint8_t) s[i + 2] & 0x3Fu) << 6) | ((uint8_t) s[i + 3] & 0x3Fu);
            len = 4;
        }
        out.push_back({cp, (uint32_t) i, (uint8_t) len, len == 1 && b0 >= 0x80 ? (uint8_t) 0 : cp_flags(cp)});
        i += (size_t) len;
    }
    return out;
}

void append_utf8(std::string & s, const uint32_t cp) {
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

const omph::gguf::Value & need(const omph::gguf::File & f, const char * key) {
    const omph::gguf::Value * v = f.find(key);
    if (v == nullptr) {
        throw std::runtime_error(std::string("tokenizer: missing ") + key);
    }
    return *v;
}

} // namespace

// The qwen35 pre-tokenizer (llama.cpp, from tokenizer.json), alternatives in order:
//   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//   | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   | \p{N}
//   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
// as a scanner: at each position the first alternative that matches wins, with
// the length a backtracking matcher would give it.
std::vector<std::pair<size_t, size_t>> Tokenizer::split_words(const std::string_view text) {
    const std::vector<Cp> c = decode_utf8(text);
    const size_t n = c.size();
    const auto L = [&](size_t i) { return (c[i].fl & kCpLetter) != 0; };
    const auto LM = [&](size_t i) { return (c[i].fl & (kCpLetter | kCpMark)) != 0; };
    const auto N = [&](size_t i) { return (c[i].fl & kCpNumber) != 0; };
    const auto WS = [&](size_t i) { return (c[i].fl & kCpSpace) != 0; };
    const auto crlf = [&](size_t i) { return c[i].cp == '\r' || c[i].cp == '\n'; };
    const auto other = [&](size_t i) { return !WS(i) && !LM(i) && !N(i); };
    const auto lower = [&](size_t i) { return c[i].cp < 0x80 ? (uint32_t) (c[i].cp | 0x20) : 0u; };

    std::vector<std::pair<size_t, size_t>> words;
    size_t i = 0;
    while (i < n) {
        size_t len = 0;
        // 1. contractions
        if (c[i].cp == '\'' && i + 1 < n) {
            const uint32_t a = lower(i + 1);
            const uint32_t b = i + 2 < n ? lower(i + 2) : 0u;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') {
                len = 2;
            } else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
                len = 3;
            }
        }
        // 2. an optional non-letter / non-number / non-newline, then letters or marks
        if (len == 0) {
            size_t j = i;
            if (!LM(i) && !crlf(i) && !L(i) && !N(i) && i + 1 < n && LM(i + 1)) {
                j = i + 1;
            }
            if (LM(j)) {
                while (j < n && LM(j)) {
                    ++j;
                }
                len = j - i;
            }
        }
        // 3. one number code point
        if (len == 0 && N(i)) {
            len = 1;
        }
        // 4. an optional space, a run of anything else, trailing newlines
        if (len == 0) {
            size_t j = i;
            if (c[i].cp == ' ' && i + 1 < n && other(i + 1)) {
                j = i + 1;
            }
            if (other(j)) {
                while (j < n && other(j)) {
                    ++j;
                }
                while (j < n && crlf(j)) {
                    ++j;
                }
                len = j - i;
            }
        }
        // 5-7. whitespace
        if (len == 0 && WS(i)) {
            size_t j = i;
            while (j < n && WS(j)) {
                ++j;
            }
            size_t last_nl = n;
            for (size_t k = i; k < j; ++k) {
                if (crlf(k)) {
                    last_nl = k;
                }
            }
            if (last_nl < n) {
                len = last_nl + 1 - i;  // \s*[\r\n]+: up to the run's last newline
            } else if (j == n) {
                len = j - i;            // \s+(?!\S) at the end of the text
            } else if (j - i >= 2) {
                len = j - i - 1;        // \s+(?!\S): all but the last, which goes with the next word
            } else {
                len = 1;                // \s+
            }
        }
        if (len == 0) {
            len = 1;  // not reachable: every code point belongs to some alternative
        }
        const size_t b = c[i].off;
        const size_t e = i + len < n ? c[i + len].off : text.size();
        words.emplace_back(b, e);
        i += len;
    }
    return words;
}

Tokenizer::Tokenizer(const omph::gguf::File & file) {
    std::string_view model;
    std::string_view pre;
    if (!need(file, "tokenizer.ggml.model").as_str(model) || model != "gpt2" ||
        !need(file, "tokenizer.ggml.pre").as_str(pre) || pre != "qwen35") {
        throw std::runtime_error("tokenizer: only the gpt2 / qwen35 tokenizer is implemented");
    }
    tokens_ = file.string_array("tokenizer.ggml.tokens");
    const std::vector<int64_t> types = file.int_array("tokenizer.ggml.token_type");
    if (types.size() != tokens_.size()) {
        throw std::runtime_error("tokenizer: token_type and tokens differ in length");
    }
    types_.assign(types.begin(), types.end());
    ids_.reserve(tokens_.size());
    for (size_t i = 0; i < tokens_.size(); ++i) {
        ids_.emplace(tokens_[i], (int32_t) i);
        if (types_[i] == 3 || types_[i] == 4) {
            special_.push_back((int32_t) i);
        }
    }
    // longest first, as llama.cpp partitions them
    std::stable_sort(special_.begin(), special_.end(), [&](const int32_t a, const int32_t b) {
        return tokens_[(size_t) a].size() > tokens_[(size_t) b].size();
    });
    const std::vector<std::string> merges = file.string_array("tokenizer.ggml.merges");
    merge_rank_.reserve(merges.size());
    for (size_t r = 0; r < merges.size(); ++r) {
        merge_rank_.emplace(merges[r], (int32_t) r);
    }
    uint64_t v = 0;
    if (const omph::gguf::Value * e = file.find("tokenizer.ggml.eos_token_id"); e && e->as_u64(v)) {
        eos_ = (int32_t) v;
    }
    if (const omph::gguf::Value * b = file.find("tokenizer.ggml.bos_token_id"); b && b->as_u64(v)) {
        bos_ = (int32_t) v;
    }
    if (const omph::gguf::Value * a = file.find("tokenizer.ggml.add_bos_token"); a && a->as_u64(v)) {
        add_bos_ = v != 0;
    }
    // GPT-2's bytes_to_unicode: printable bytes map to themselves, the others
    // to 256 + n in byte order
    uint32_t next = 256;
    for (uint32_t b = 0; b < 256; ++b) {
        const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        const uint32_t cp = printable ? b : next++;
        append_utf8(byte_sym_[b], cp);
        sym_byte_[cp] = (uint8_t) b;
    }
}

int32_t Tokenizer::find(const std::string_view text) const {
    const auto it = ids_.find(std::string(text));
    return it == ids_.end() ? -1 : it->second;
}

// Byte-level BPE of one word: its bytes as symbols, then the adjacent pair of
// lowest merge rank merged, the leftmost on ties, until none is in the merges
// (llama.cpp's priority-queue order). A final symbol outside the vocabulary
// falls back to its bytes' tokens.
void Tokenizer::bpe(const std::string_view word, std::vector<int32_t> & out) const {
    std::vector<std::string> sym;
    sym.reserve(word.size());
    for (const char ch : word) {
        sym.push_back(byte_sym_[(uint8_t) ch]);
    }
    const auto rank = [&](const size_t k) {
        const auto it = merge_rank_.find(sym[k] + " " + sym[k + 1]);
        return it == merge_rank_.end() ? INT32_MAX : it->second;
    };
    std::vector<int32_t> ranks(sym.size() > 1 ? sym.size() - 1 : 0);
    for (size_t k = 0; k + 1 < sym.size(); ++k) {
        ranks[k] = rank(k);
    }
    while (!ranks.empty()) {
        size_t best = 0;
        for (size_t k = 1; k < ranks.size(); ++k) {
            if (ranks[k] < ranks[best]) {
                best = k;
            }
        }
        if (ranks[best] == INT32_MAX) {
            break;
        }
        sym[best] += sym[best + 1];
        sym.erase(sym.begin() + (std::ptrdiff_t) best + 1);
        ranks.erase(ranks.begin() + (std::ptrdiff_t) best);
        if (best + 1 < sym.size()) {
            ranks[best] = rank(best);
        }
        if (best > 0) {
            ranks[best - 1] = rank(best - 1);
        }
    }
    for (const std::string & s : sym) {
        const auto it = ids_.find(s);
        if (it != ids_.end()) {
            out.push_back(it->second);
            continue;
        }
        // a symbol of several bytes not in the vocabulary: its bytes one by one
        const std::vector<Cp> cps = decode_utf8(s);
        for (const Cp & cp : cps) {
            const auto b = sym_byte_.find(cp.cp);
            if (b != sym_byte_.end()) {
                const auto t = ids_.find(byte_sym_[b->second]);
                if (t != ids_.end()) {
                    out.push_back(t->second);
                }
            }
        }
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string_view text, const bool parse_special) const {
    // fragments: raw text ranges, or special token ids
    struct Frag {
        bool raw;
        size_t b;
        size_t e;
        int32_t id;
    };
    std::vector<Frag> frags{{true, 0, text.size(), -1}};
    for (const int32_t sid : special_) {
        if (!parse_special && types_[(size_t) sid] == 3) {
            continue;
        }
        const std::string & st = tokens_[(size_t) sid];
        if (st.empty()) {
            continue;
        }
        std::vector<Frag> next;
        next.reserve(frags.size());
        for (const Frag & f : frags) {
            if (!f.raw) {
                next.push_back(f);
                continue;
            }
            size_t pos = f.b;
            while (pos < f.e) {
                const size_t m = text.substr(0, f.e).find(st, pos);
                if (m == std::string_view::npos) {
                    break;
                }
                if (m > pos) {
                    next.push_back({true, pos, m, -1});
                }
                next.push_back({false, m, m + st.size(), sid});
                pos = m + st.size();
            }
            if (pos < f.e) {
                next.push_back({true, pos, f.e, -1});
            }
        }
        frags.swap(next);
    }
    std::vector<int32_t> out;
    if (add_bos_ && bos_ >= 0) {
        out.push_back(bos_);
    }
    for (const Frag & f : frags) {
        if (!f.raw) {
            out.push_back(f.id);
            continue;
        }
        const std::string_view frag = text.substr(f.b, f.e - f.b);
        for (const auto & [b, e] : split_words(frag)) {
            bpe(frag.substr(b, e - b), out);
        }
    }
    return out;
}

std::string Tokenizer::piece(const int32_t id, const bool special) const {
    if (id < 0 || id >= size()) {
        return std::string();
    }
    const std::string & t = tokens_[(size_t) id];
    const int32_t type = types_[(size_t) id];
    if (type == 3) {
        return special ? t : std::string();
    }
    if (type == 4) {
        return t;
    }
    std::string out;
    for (const Cp & cp : decode_utf8(t)) {
        const auto b = sym_byte_.find(cp.cp);
        if (b != sym_byte_.end()) {
            out += (char) b->second;
        } else {
            out.append(t, cp.off, cp.len);
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids, const bool special) const {
    std::string out;
    for (const int32_t id : ids) {
        out += piece(id, special);
    }
    return out;
}

} // namespace omph::text
