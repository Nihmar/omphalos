// Text <-> token ids for the model's tokenizer (#148): GPT-2 style byte-level
// BPE with the `qwen35` pre-tokenizer, the vocabulary and merges read from
// the GGUF. Matches llama.cpp's tokenizer token for token.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "format/gguf.hh"

namespace omph::text {

class Tokenizer {
public:
    // Reads tokenizer.ggml.{tokens, token_type, merges, bos/eos ids}; throws on
    // a tokenizer it does not implement.
    explicit Tokenizer(const omph::gguf::File & file);

    // parse_special: control tokens written literally in the text (e.g.
    // "<|im_start|>") become their ids; user-defined ones always do.
    std::vector<int32_t> encode(std::string_view text, bool parse_special = true) const;

    // encode(), but a special token is recognized only where `allow` permits
    // it, as bits: 1 = user-defined (#think), 2 = control, 3 = both, 0 = none.
    // The chat template (#292) marks its structure both and the request's own
    // text none, so one call tokenizes the joined text: the pre-tokenizer runs
    // across the segment boundaries (#344), while no text can inject a token.
    // `allow` may be shorter than `text`: the bytes past it allow nothing.
    std::vector<int32_t> encode_masked(std::string_view text, const std::vector<uint8_t> & allow) const;

    // The bytes of the tokens, concatenated (they may end in the middle of a
    // UTF-8 sequence). Control tokens are written as their text when
    // `special`, else dropped.
    std::string decode(const std::vector<int32_t> & ids, bool special = true) const;
    std::string piece(int32_t id, bool special = true) const;

    int32_t size() const { return (int32_t) tokens_.size(); }
    int32_t eos() const { return eos_; }
    int32_t bos() const { return bos_; }
    bool add_bos() const { return add_bos_; }
    // The id of a token by its exact text (special tokens), -1 if none.
    int32_t find(std::string_view text) const;

    // The pre-tokenizer alone: byte ranges [begin, end) of the words of
    // `text` (exposed for the tests).
    static std::vector<std::pair<size_t, size_t>> split_words(std::string_view text);

private:
    void bpe(std::string_view word, std::vector<int32_t> & out) const;

    std::vector<std::string> tokens_;   // as in the GGUF (byte-level encoded)
    std::vector<int32_t> types_;        // 1 normal, 3 control, 4 user-defined, 5 unused, 6 byte
    std::unordered_map<std::string, int32_t> ids_;
    std::unordered_map<std::string, int32_t> merge_rank_;  // "left right"
    std::vector<int32_t> special_;      // control + user-defined, longest text first
    std::string byte_sym_[256];         // byte -> its byte-level symbol (UTF-8)
    std::unordered_map<uint32_t, uint8_t> sym_byte_;  // symbol code point -> byte
    int32_t eos_ = -1;
    int32_t bos_ = -1;
    bool add_bos_ = false;
};

} // namespace omph::text
