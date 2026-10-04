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
//   --no-spec         no MTP speculative decoding (greedy, and sampling since #197)
//   --no-mtp          do not load the MTP block (-352 MiB of VRAM)
//   --dflash FILE     draft with a DFlash2 drafter .omph instead of the MTP block (#245)
//   --ctx N           KV capacity (default 8192)
//   --cache-mib N     host RAM for sequence checkpoints (default 2048, 0: none)
//   --kv-ram N        host RAM for whole conversations (default 8192, 0: none, #179)
//   --repeat N        run the same request N times (the second and later
//                     restore the checkpoint before the generation prompt, #158)
//   --then FILE       then a second chat request from FILE (a conversation's
//                     next turn: it continues the cached sequence when it extends it)
//   --mmproj FILE     the vision encoder (builds with OMPH_LLAMA_DIR, #160)
//   --image FILE      an image for the next <|image_pad|> of the prompt (repeat
//                     for several; with --chat, one per image item of the request)
//   --logits-out FILE the logits after the prompt, f32 (validation)
//   --force FILE      decode these token ids instead (one per step); --logits-out
//                     then gets the prompt's row and every step's (validation)
#include "model/generator.hh"
#include "tools/cli.hh"
#include "runtime/options.hh"
#include "runtime/timing.hh"
#include "text/chat.hh"
#include "text/json.hh"
#ifdef OMPH_VISION
#include "vision/encoder.hh"
#endif

#include <algorithm>
#include <cstdio>
#include <memory>
#include <cstdlib>
#include <stdexcept>
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
    std::string then_path, mmproj, logits_out, force_path;
    std::vector<std::string> image_paths;
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
        else if (!std::strcmp(argv[i], "--dflash")) cfg.dflash = val();
        else if (!std::strcmp(argv[i], "--max")) req.max_tokens = (int64_t) omph::cli::integer(val(), "--max");
        else if (!std::strcmp(argv[i], "--temp")) req.sampling.temperature = (float) omph::cli::number(val(), "--temp");
        else if (!std::strcmp(argv[i], "--top-k")) req.sampling.top_k = omph::cli::integer(val(), "--top-k");
        else if (!std::strcmp(argv[i], "--top-p")) req.sampling.top_p = (float) omph::cli::number(val(), "--top-p");
        else if (!std::strcmp(argv[i], "--min-p")) req.sampling.min_p = (float) omph::cli::number(val(), "--min-p");
        else if (!std::strcmp(argv[i], "--seed")) req.sampling.seed = omph::cli::uinteger(val(), "--seed");
        else if (!std::strcmp(argv[i], "--repeat-penalty")) req.sampling.repeat_penalty = (float) omph::cli::number(val(), "--repeat-penalty");
        else if (!std::strcmp(argv[i], "--repeat-last-n")) req.sampling.penalty_last_n = (int) omph::cli::integer(val(), "--repeat-last-n");
        else if (!std::strcmp(argv[i], "--frequency-penalty")) req.sampling.frequency_penalty = (float) omph::cli::number(val(), "--frequency-penalty");
        else if (!std::strcmp(argv[i], "--presence-penalty")) req.sampling.presence_penalty = (float) omph::cli::number(val(), "--presence-penalty");
        else if (!std::strcmp(argv[i], "--ctx")) cfg.context = omph::cli::integer(val(), "--ctx");
        else if (!std::strcmp(argv[i], "--cache-mib")) cfg.cache_mib = omph::cli::integer(val(), "--cache-mib");
        else if (!std::strcmp(argv[i], "--kv-ram")) cfg.kv_ram_mib = omph::cli::integer(val(), "--kv-ram");
        else if (!std::strcmp(argv[i], "--repeat")) repeat = (int) omph::cli::integer(val(), "--repeat");
        else if (!std::strcmp(argv[i], "--then")) then_path = val();
        else if (!std::strcmp(argv[i], "--mmproj")) mmproj = val();
        else if (!std::strcmp(argv[i], "--image")) image_paths.push_back(val());
        else if (!std::strcmp(argv[i], "--logits-out")) logits_out = val();
        else if (!std::strcmp(argv[i], "--force")) force_path = val();
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    try {
        const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
        omph::model::Generator gen(cfg, omph::runtime::EnvOptions::from_env());
        const omph::text::Tokenizer & tok = gen.tokenizer();
        const auto read_file = [](const std::string & path) {
            FILE * f = std::fopen(path.c_str(), "rb");
            if (f == nullptr) throw std::runtime_error("cannot open " + path);
            std::string body;
            char buf[65536];
            size_t n = 0;
            while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) body.append(buf, n);
            std::fclose(f);
            return body;
        };
        std::vector<std::string> image_bytes;
#ifdef OMPH_VISION
        std::unique_ptr<omph::vision::Encoder> vision;
#endif
        if (!image_paths.empty()) {
#ifdef OMPH_VISION
            if (mmproj.empty()) throw std::runtime_error("--image needs --mmproj");
            vision = std::make_unique<omph::vision::Encoder>(mmproj, cfg.model);
            for (const std::string & path : image_paths) image_bytes.push_back(read_file(path));
#else
            throw std::runtime_error("built without vision (configure with OMPH_LLAMA_DIR)");
#endif
        }
        std::vector<float> first_logits;
        std::vector<int32_t> forced;
        if (!force_path.empty()) {
            std::istringstream fs(read_file(force_path));
            long long v = 0;
            while (fs >> v) forced.push_back((int32_t) v);
            req.force = &forced;
            req.max_tokens = (int64_t) forced.size();
            if (!logits_out.empty()) req.forced_logits = &first_logits;
        } else if (!logits_out.empty()) {
            req.prefill_logits = &first_logits;
        }
        std::vector<int32_t> prompt;
        if (prompt_ids) {
            std::istringstream ss(in);
            long long v = 0;
            while (ss >> v) prompt.push_back((int32_t) v);
        } else {
            prompt = chat ? omph::text::tokenize_chat(omph::text::render_chat_segments(omph::text::Json::parse(in)),
                                                      tok)
                          : tok.encode(in);
        }
#ifdef OMPH_VISION
        if (!image_bytes.empty()) {  // encoded on the CPU while the GPU prefills the text before them (#180)
            const int32_t pad = tok.find("<|image_pad|>");
            const std::vector<int32_t> before(prompt.begin(), std::find(prompt.begin(), prompt.end(), pad));
            const double t0 = omph::runtime::now_ms();
            double t_text = 0.0;
            req.images = vision->encode_all(image_bytes, [&] {
                if (!gen.prefill(before)) throw std::runtime_error("prefill failed");
                t_text = omph::runtime::now_ms() - t0;
            });
            for (size_t i = 0; i < req.images.size(); ++i) {
                std::fprintf(stderr, "image %s: %dx%d tokens\n", image_paths[i].c_str(), req.images[i]->nx,
                             req.images[i]->ny);
            }
            std::fprintf(stderr, "images encoded in %.0f ms (CPU), the %zu tokens before them in %.0f ms (GPU)\n",
                         omph::runtime::now_ms() - t0, before.size(), t_text);
        }
#endif
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
                prompt = omph::text::tokenize_chat(omph::text::render_chat_segments(omph::text::Json::parse(body)),
                                                   tok);
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
            if (!logits_out.empty() && r == 0) {
                FILE * f = std::fopen(logits_out.c_str(), "wb");
                if (f == nullptr || std::fwrite(first_logits.data(), 4, first_logits.size(), f) != first_logits.size()) {
                    std::fprintf(stderr, "cannot write %s\n", logits_out.c_str());
                    return 1;
                }
                std::fclose(f);
            }
            static const char * kStop[] = {"length", "end of generation", "stop token", "callback",
                                           "context full", "error"};
            const double tps = res.decode_ms > 0 ? 1000.0 * (double) res.tokens.size() / res.decode_ms : 0.0;
            std::fprintf(stderr,
                         "prompt %lld tokens (%lld cached%s%s) in %.1f ms (checkpoints %.1f ms); %zu tokens in "
                         "%.1f ms (%.2f ms/token, %.1f t/s); drafts %lld / %lld accepted; stop: %s\n",
                         (long long) res.prompt_tokens, (long long) res.cached_tokens,
                         res.restored ? ", restored" : "", res.saved ? ", previous saved" : "", res.prefill_ms,
                         res.checkpoint_ms,
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
