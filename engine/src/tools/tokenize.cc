// omph-tokenize — the engine's tokenizer on the command line (#148).
//
// usage: omph-tokenize <model.gguf> [--no-parse-special] [--decode] < input
//   text on stdin -> token ids on stdout (space-separated, one line)
//   --decode: token ids on stdin -> text on stdout (control tokens included)
//   --no-parse-special: control tokens written in the text stay text
#include "format/gguf.hh"
#include "text/tokenizer.hh"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [--no-parse-special] [--decode] < input\n", argv[0]);
        return 2;
    }
    bool parse_special = true;
    bool decode = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-parse-special") == 0) {
            parse_special = false;
        } else if (std::strcmp(argv[i], "--decode") == 0) {
            decode = true;
        } else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    try {
        const omph::gguf::File file(argv[1]);
        const omph::text::Tokenizer tok(file);
        const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
        if (decode) {
            std::istringstream ss(in);
            std::vector<int32_t> ids;
            long long v = 0;
            while (ss >> v) {
                ids.push_back((int32_t) v);
            }
            const std::string text = tok.decode(ids);
            std::fwrite(text.data(), 1, text.size(), stdout);
            return 0;
        }
        const std::vector<int32_t> ids = tok.encode(in, parse_special);
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
