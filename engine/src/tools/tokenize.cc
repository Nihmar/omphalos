// omph-tokenize — the engine's tokenizer on the command line (#148).
//
// usage: omph-tokenize <model.gguf> [--no-parse-special] [--decode | --chat | --chat-ids] < input
//   text on stdin -> token ids on stdout (space-separated, one line)
//   --decode: token ids on stdin -> text on stdout (control tokens included)
//   --chat: a JSON chat request on stdin (text/chat.hh) -> the rendered prompt (#150)
//   --chat-ids: the same, tokenized
//   --no-parse-special: control tokens written in the text stay text
//   --chat-template original|sharp: the GGUF's template (default) or the
//   vendored Qwen Sharp one (#392)
#include "format/gguf.hh"
#include "text/chat.hh"
#include "text/json.hh"
#include "text/tokenizer.hh"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <optional>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [--no-parse-special] [--decode] < input\n", argv[0]);
        return 2;
    }
    bool parse_special = true;
    bool decode = false;
    int chat = 0;  // 1: render, 2: render + tokenize
    omph::text::ChatTemplate tpl = omph::text::ChatTemplate::Original;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-parse-special") == 0) {
            parse_special = false;
        } else if (std::strcmp(argv[i], "--decode") == 0) {
            decode = true;
        } else if (std::strcmp(argv[i], "--chat") == 0) {
            chat = 1;
        } else if (std::strcmp(argv[i], "--chat-ids") == 0) {
            chat = 2;
        } else if (std::strcmp(argv[i], "--chat-template") == 0 && i + 1 < argc) {
            try {
                tpl = omph::text::chat_template_from_string(argv[++i]);
            } catch (const std::exception & e) {
                std::fprintf(stderr, "%s\n", e.what());
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    try {
        const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
        if (chat == 1) {
            const std::string prompt = omph::text::render_chat(omph::text::Json::parse(in), tpl);
            std::fwrite(prompt.data(), 1, prompt.size(), stdout);
            return 0;
        }
        const omph::gguf::File file(argv[1]);
        const omph::text::Tokenizer tok(file);
        // a chat prompt is rendered as segments: its structure parsed for
        // special tokens, the request's own text not (#292)
        const std::optional<std::vector<omph::text::Segment>> segments =
            chat == 2 ? std::optional(omph::text::render_chat_segments(omph::text::Json::parse(in), tpl))
                      : std::nullopt;
        const std::string text = chat == 1 ? omph::text::render_chat(omph::text::Json::parse(in), tpl) : in;
        if (decode) {
            std::istringstream ss(in);
            std::vector<int32_t> ids;
            long long v = 0;
            while (ss >> v) {
                ids.push_back((int32_t) v);
            }
            const std::string decoded = tok.decode(ids);
            std::fwrite(decoded.data(), 1, decoded.size(), stdout);
            return 0;
        }
        const std::vector<int32_t> ids =
            segments.has_value() ? omph::text::tokenize_chat(*segments, tok) : tok.encode(text, parse_special);
        for (size_t i = 0; i < ids.size(); ++i) {
            std::printf(i ? " %d" : "%d", ids[i]);
        }
        std::printf("\n");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
