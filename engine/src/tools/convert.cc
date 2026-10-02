// omph-convert — GGUF -> .omph, the only file the engine loads (#178).
//
// usage: omph-convert <model.gguf> [out.omph]
//   out defaults to the input path with .gguf replaced by .omph. Every tensor a
//   fused kernel reads is repacked and checked to rebuild the GGUF bytes bit for
//   bit; the GGUF's metadata (hyperparameters, tokenizer, chat template) is
//   copied verbatim with its SHA-256 recorded (format/omph.hh).
#include "format/omph.hh"

#include <cstdio>
#include <exception>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> [out.omph]\n", argv[0]);
        return 2;
    }
    const std::string in = argv[1];
    std::string out;
    if (argc == 3) {
        out = argv[2];
    } else {
        const std::string ext = ".gguf";
        out = in.size() > ext.size() && in.compare(in.size() - ext.size(), ext.size(), ext) == 0
                  ? in.substr(0, in.size() - ext.size()) + ".omph"
                  : in + ".omph";
    }
    try {
        int last = -1;
        const auto st = omph::format::convert_to_omph(in, out, [&](uint64_t done, uint64_t total) {
            const int pct = (int) (100 * done / total);
            if (pct / 10 != last / 10) {
                std::fprintf(stderr, "\r%3d %% (%llu / %llu tensors)", pct, (unsigned long long) done,
                             (unsigned long long) total);
                last = pct;
            }
        });
        std::fprintf(stderr, "\n");
        std::printf("%s -> %s: %llu tensors (%llu repacked and verified bit-exact), %.2f -> %.2f GiB, "
                    "source sha256 %s\n",
                    in.c_str(), out.c_str(), (unsigned long long) st.tensors, (unsigned long long) st.repacked,
                    st.bytes_in / 1073741824.0, st.bytes_out / 1073741824.0, st.source_sha256.c_str());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "\n%s\n", e.what());
        return 1;
    }
    return 0;
}
