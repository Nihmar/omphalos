// The tokenizer on a synthetic GGUF (#148): the qwen35 pre-tokenizer's word
// splits, byte-level BPE by merge rank, special tokens with and without
// parsing, decoding back to the bytes, and the GGUF reader's full arrays (more
// than the 256 elements the metadata keeps). The real vocabulary is checked
// against llama.cpp by tools/check_tokenizer.py.
#include "check.hh"
#include "format/gguf.hh"
#include "text/tokenizer.hh"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct Writer {
    std::vector<uint8_t> b;
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back((uint8_t) (v >> (8 * i)));
    }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) b.push_back((uint8_t) (v >> (8 * i)));
    }
    void str(const std::string & s) {
        u64(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
    void kv_str(const std::string & k, const std::string & v) {
        str(k);
        u32(8);
        str(v);
    }
    void kv_strs(const std::string & k, const std::vector<std::string> & v) {
        str(k);
        u32(9);
        u32(8);
        u64(v.size());
        for (const std::string & s : v) str(s);
    }
    void kv_i32s(const std::string & k, const std::vector<int32_t> & v) {
        str(k);
        u32(9);
        u32(5);
        u64(v.size());
        for (const int32_t x : v) u32((uint32_t) x);
    }
};

void append_utf8(std::string & s, uint32_t cp) {
    if (cp < 0x80) {
        s += (char) cp;
    } else {
        s += (char) (0xC0 | (cp >> 6));
        s += (char) (0x80 | (cp & 0x3F));
    }
}

// GPT-2's bytes_to_unicode: every byte's symbol (all below U+0800)
std::vector<std::string> byte_symbols() {
    std::vector<std::string> out(256);
    uint32_t next = 256;
    for (uint32_t b = 0; b < 256; ++b) {
        const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        append_utf8(out[b], printable ? b : next++);
    }
    return out;
}

std::vector<std::string> words(const std::string & text) {
    std::vector<std::string> out;
    for (const auto & [b, e] : omph::text::Tokenizer::split_words(text)) {
        out.push_back(text.substr(b, e - b));
    }
    return out;
}

} // namespace

int main() {
    // --- the pre-tokenizer's splits (the regex's alternatives and backtracking)
    using V = std::vector<std::string>;
    CHECK(words("Hello world") == (V{"Hello", " world"}), "letters with a space prefix");
    CHECK(words("It's we'LL") == (V{"It", "'s", " we", "'LL"}), "contractions, any case");
    CHECK(words("a  b") == (V{"a", " ", " b"}), "\\s+(?!\\S) leaves the last space to the word");
    CHECK(words("x\n\n\ny") == (V{"x", "\n\n\n", "y"}), "newline runs");
    CHECK(words("  \n  b") == (V{"  \n", " ", " b"}), "\\s*[\\r\\n]+ stops at the last newline");
    CHECK(words("123") == (V{"1", "2", "3"}), "one digit per word");
    CHECK(words("hi!!\n\nok") == (V{"hi", "!!\n\n", "ok"}), "punctuation keeps trailing newlines");
    CHECK(words(" ?x") == (V{" ?", "x"}), "an optional space before punctuation");
    CHECK(words("end  ") == (V{"end", "  "}), "trailing whitespace at the end of the text");
    CHECK(words("caf\xc3\xa9 e\xcc\x81") == (V{"caf\xc3\xa9", " e\xcc\x81"}), "letters and marks");
    CHECK(words("\xe6\x97\xa5\xe6\x9c\xac") == (V{"\xe6\x97\xa5\xe6\x9c\xac"}), "CJK letters");

    // --- a synthetic tokenizer: the 256 byte symbols, three merges, two specials
    const std::vector<std::string> bs = byte_symbols();
    std::vector<std::string> tokens = bs;
    std::vector<int32_t> types(256, 1);
    const std::string G = bs[' '];  // the space symbol
    for (const std::string & t : {G + "t", std::string("he"), G + "the"}) {
        tokens.push_back(t);
        types.push_back(1);
    }
    tokens.push_back("<|im_start|>");
    types.push_back(3);
    tokens.push_back("<think>");
    types.push_back(4);
    // #345: a merge chain over '=' (the 10 levels of a 1 MB run) so the BPE's
    // merge loop is exercised by one enormous pre-tokenizer word; after the
    // specials, whose ids the checks below pin
    std::vector<std::string> merges = {G + " t", "h e", G + "t he"};
    const int32_t id_eq2 = (int32_t) tokens.size();
    for (int k = 1; k <= 10; ++k) {
        tokens.push_back(std::string(1u << k, '='));
        types.push_back(1);
        merges.push_back(std::string(1u << (k - 1), '=') + " " + std::string(1u << (k - 1), '='));
    }
    const int32_t id_gt = 256, id_he = 257, id_gthe = 258, id_ims = 259, id_think = 260;
    Writer w;
    w.u32(0x46554747u);
    w.u32(3);
    w.u64(0);  // tensors
    w.u64(5);  // keys
    w.kv_str("tokenizer.ggml.model", "gpt2");
    w.kv_str("tokenizer.ggml.pre", "qwen35");
    w.kv_strs("tokenizer.ggml.tokens", tokens);
    w.kv_i32s("tokenizer.ggml.token_type", types);
    w.kv_strs("tokenizer.ggml.merges", merges);
    while (w.b.size() % 32 != 0) w.b.push_back(0);
    char path[] = "/tmp/omph_test_tokenizer_XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp");
    CHECK(write(fd, w.b.data(), w.b.size()) == (ssize_t) w.b.size(), "write");
    close(fd);
    try {
        const omph::gguf::File file(path);
        CHECK(file.string_array("tokenizer.ggml.tokens") == tokens, "every string of a 261-element array");
        const std::vector<int64_t> t64 = file.int_array("tokenizer.ggml.token_type");
        CHECK(t64.size() == types.size() && t64[259] == 3 && t64[260] == 4, "every integer of the array");

        const omph::text::Tokenizer tok(file);
        CHECK(tok.encode(" the") == (std::vector<int32_t>{id_gthe}), "merges by rank: Gt, he, then Gthe");
        CHECK(tok.encode("the") == (std::vector<int32_t>{(int32_t) 't', id_he}), "no Gt without the space");
        CHECK(tok.encode(" tea") == (std::vector<int32_t>{id_gt, (int32_t) 'e', (int32_t) 'a'}),
              "partial merges");
        CHECK(tok.encode("<|im_start|>x") == (std::vector<int32_t>{id_ims, (int32_t) 'x'}),
              "a control token parsed");
        CHECK(tok.encode("<|im_start|>", false).size() > 1, "a control token left as text");
        CHECK(tok.encode("a<think>b", false) == (std::vector<int32_t>{(int32_t) 'a', id_think, (int32_t) 'b'}),
              "a user-defined token is always parsed");
        const std::string text = "caf\xc3\xa9 \xf0\x9f\x98\x80 \t\r\n<|im_start|> the end";
        CHECK(tok.decode(tok.encode(text)) == text, "decode gives the bytes back");
        CHECK(tok.decode(tok.encode(text), false).find("<|im_start|>") == std::string::npos,
              "control tokens dropped without `special`");
        CHECK(tok.find("<think>") == id_think && tok.find("nope") == -1, "find by text");
        // #345: a 1 MB run of one character is one pre-tokenizer word; the old
        // merge loop rescanned every pair after every merge (O(n^2), minutes)
        {
            const std::string run(1 << 20, '=');
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<int32_t> ids = tok.encode(run);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            CHECK(ids.size() == 1024 && ids.front() == id_eq2 + 9,
                  "a 1 MB '=' run merges to 1024 symbols (%zu ids)", ids.size());
            CHECK(ms < 1000.0, "1 MB of '=' took %.1f ms", ms);
        }
    } catch (const std::exception & e) {
        CHECK(false, "exception: %s", e.what());
    }
    unlink(path);
    if (omph_test::failures == 0) {
        std::printf("test_tokenizer: all checks passed\n");
    }
    return omph_test::failures;
}
