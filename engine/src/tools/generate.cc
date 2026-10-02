// omph-generate — text in, text out, on the engine's Generator (#152).
//
// usage: omph-generate <model.gguf> [options] < input
//   input: a prompt (text), a chat request with --chat (JSON, text/chat.hh), or
//          token ids with --prompt-ids
//   output: the generated text, streamed (--ids: the token ids instead);
//           timing and speculation statistics on stderr
//   --max N           tokens to generate (default 512)
//   --temp T          sampling temperature (default 0: greedy)
//   --top-k K, --top-p P, --min-p M, --seed S
//   --no-spec         no MTP speculative decoding (greedy only uses it)
//   --no-mtp          do not load the MTP block (-352 MiB of VRAM)
//   --ctx N           KV capacity (default 8192)
//   --cache-mib N     host RAM for sequence checkpoints (default 2048, 0: none)
//   --repeat N        run the same request N times (the second and later
//                     restore the checkpoint before the generation prompt, #158)
//   --then FILE       then a second chat request from FILE (a conversation's
//                     next turn: it continues the cached sequence when it extends it)
#include "model/generator.hh"
#include "runtime/options.hh"
#include "text/chat.hh"
#include "text/json.hh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [options] < input\n", argv[0]);
        return 2;
    }
    omph::model::Generator::Config cfg;
    cfg.model = argv[1];
    omph::model::GenerateRequest req;
    req.max_tokens = 512;
    bool chat = false;
    bool prompt_ids = false;
    bool out_ids = false;
    int repeat = 1;
    std::string then_path;
    for (int i = 2; i < argc; ++i) {
        const auto val = [&]() -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", argv[i]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--chat")) chat = true;
        else if (!std::strcmp(argv[i], "--prompt-ids")) prompt_ids = true;
        else if (!std::strcmp(argv[i], "--ids")) out_ids = true;
        else if (!std::strcmp(argv[i], "--no-spec")) req.speculative = false;
        else if (!std::strcmp(argv[i], "--no-mtp")) cfg.mtp = false;
        else if (!std::strcmp(argv[i], "--max")) req.max_tokens = std::atoll(val());
        else if (!std::strcmp(argv[i], "--temp")) req.sampling.temperature = (float) std::atof(val());
        else if (!std::strcmp(argv[i], "--top-k")) req.sampling.top_k = std::atoi(val());
        else if (!std::strcmp(argv[i], "--top-p")) req.sampling.top_p = (float) std::atof(val());
        else if (!std::strcmp(argv[i], "--min-p")) req.sampling.min_p = (float) std::atof(val());
        else if (!std::strcmp(argv[i], "--seed")) req.sampling.seed = std::strtoull(val(), nullptr, 10);
        else if (!std::strcmp(argv[i], "--ctx")) cfg.context = std::atoll(val());
        else if (!std::strcmp(argv[i], "--cache-mib")) cfg.cache_mib = std::atoll(val());
        else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(val());
        else if (!std::strcmp(argv[i], "--then")) then_path = val();
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    try {
        const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
        omph::model::Generator gen(cfg, omph::runtime::EnvOptions::from_env());
        const omph::text::Tokenizer & tok = gen.tokenizer();
        std::vector<int32_t> prompt;
        if (prompt_ids) {
            std::istringstream ss(in);
            long long v = 0;
            while (ss >> v) prompt.push_back((int32_t) v);
        } else {
            prompt = tok.encode(chat ? omph::text::render_chat(omph::text::Json::parse(in)) : in);
        }
        if (!then_path.empty()) {
            repeat = 2;
        }
        for (int r = 0; r < repeat; ++r) {
            if (r == 1 && !then_path.empty()) {
                FILE * f = std::fopen(then_path.c_str(), "rb");
                if (f == nullptr) {
                    std::fprintf(stderr, "cannot open %s\n", then_path.c_str());
                    return 1;
                }
                std::string body;
                char buf[4096];
                size_t n = 0;
                while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) body.append(buf, n);
                std::fclose(f);
                prompt = tok.encode(omph::text::render_chat(omph::text::Json::parse(body)));
            }
            bool first = true;
            const auto res = gen.generate(prompt, req, [&](const int32_t t) {
                if (out_ids) {
                    std::printf(first ? "%d" : " %d", t);
                } else {
                    const std::string p = tok.piece(t, false);
                    std::fwrite(p.data(), 1, p.size(), stdout);
                }
                first = false;
                std::fflush(stdout);
                return true;
            });
            std::printf("\n");
            std::fflush(stdout);
            static const char * kStop[] = {"length", "end of generation", "stop token", "callback",
                                           "context full", "error"};
            const double tps = res.decode_ms > 0 ? 1000.0 * (double) res.tokens.size() / res.decode_ms : 0.0;
            std::fprintf(stderr,
                         "prompt %lld tokens (%lld cached%s) in %.1f ms (checkpoints %.1f ms); %zu tokens in "
                         "%.1f ms (%.2f ms/token, %.1f t/s); drafts %lld / %lld accepted; stop: %s\n",
                         (long long) res.prompt_tokens, (long long) res.cached_tokens,
                         res.restored ? ", restored" : "", res.prefill_ms, res.checkpoint_ms,
                         res.tokens.size(), res.decode_ms,
                         res.tokens.empty() ? 0.0 : res.decode_ms / (double) res.tokens.size(), tps,
                         (long long) res.accepted, (long long) res.drafted, kStop[(int) res.stop]);
            if (res.stop == omph::model::GenerateResult::Stop::Error) return 1;
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
