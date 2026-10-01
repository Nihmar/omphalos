// omph-run — the engine's command-line runner: prefill a prompt, write its
// logits, optionally decode greedily (PLAN.md §12).
//
// usage: omph-run <model.gguf> <tokens.txt> <out-logits.f32> [options]
//   tokens.txt          token ids separated by whitespace
//   --last-logits       write only the last row of logits (long prompts)
//   --logits-tail N     write only the last N rows
//   --tokens N          use only the first N prompt tokens
//   --generate N        greedy-decode N tokens after the prompt ...
//   --gen-out FILE      ... and write them here (one id per line); required with --generate
//   --gemv              decode with the fused GEMVs on repacked weights (the fast
//                       path; only effective with --generate, a prefill-only run
//                       stays on the f16 + hipBLASLt path)
//   --trace-dir DIR     dump each layer's output as l_out-<layer>.f32 (one-chunk
//                       prompts only)
// Without --last-logits / --logits-tail the file holds tokens x n_vocab f32.
// Long prompts run in chunks of 512 tokens. The KV cache is K Q8 / V Q4 with an
// FP16 ring of the last 128 tokens. The OMPH_* environment switches (KV
// variants, A/B switches, ablations, timing) are listed in runtime/options.hh
// and AGENTS.md.
#include "model/runner.hh"
#include "runtime/options.hh"
#include "runtime/timing.hh"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_f32(const std::string & path, const std::vector<float> & data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * 4));
    out.close();
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
}

} // namespace

int main(int argc, char ** argv) {
    const auto usage = [&]() {
        std::fprintf(stderr, "usage: %s <model.gguf> <tokens.txt> <out-logits.f32> "
                             "[--trace-dir DIR] [--tokens N] [--last-logits | --logits-tail N] "
                             "[--generate N --gen-out FILE] [--gemv]\n",
                     argv[0]);
        return 2;
    };
    if (argc < 4) {
        return usage();
    }
    const std::string model = argv[1];
    const std::string tok_path = argv[2];
    const std::string logits_path = argv[3];
    std::string trace_dir;
    std::string gen_path;
    int64_t max_tokens = 0;
    bool last_logits = false;
    int64_t logits_tail = 0;
    int64_t generate = 0;
    bool use_gemv = false;
    std::string oracle_path;   // --draft-oracle: drafts from a token file (#122)
    int64_t draft_k = 3;
    int64_t draft_corrupt = 0;  // corrupt every N-th draft (0: never)
    bool mtp = false;           // --mtp: load the MTP block (#124)
    bool draft_mtp = false;     // --draft-mtp K: speculative decode with MTP drafts
    std::string mtp_out;        // --mtp-out: the first two drafts' logits (validation)
    // A flag's numeric value: a whole non-negative number or nothing.
    const auto count = [](const char * s, int64_t & out) {
        char * end = nullptr;
        const long long v = std::strtoll(s, &end, 10);
        out = v;
        return end != s && *end == '\0' && v >= 0;
    };
    for (int i = 4; i < argc; ++i) {
        const bool has_value = i + 1 < argc;
        if (std::strcmp(argv[i], "--trace-dir") == 0 && has_value) {
            trace_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--last-logits") == 0) {
            last_logits = true;
        } else if (std::strcmp(argv[i], "--logits-tail") == 0 && has_value) {
            if (!count(argv[++i], logits_tail)) return usage();
        } else if (std::strcmp(argv[i], "--tokens") == 0 && has_value) {
            if (!count(argv[++i], max_tokens)) return usage();
        } else if (std::strcmp(argv[i], "--generate") == 0 && has_value) {
            if (!count(argv[++i], generate)) return usage();
        } else if (std::strcmp(argv[i], "--gemv") == 0) {
            use_gemv = true;
        } else if (std::strcmp(argv[i], "--gen-out") == 0 && has_value) {
            gen_path = argv[++i];
        } else if (std::strcmp(argv[i], "--draft-oracle") == 0 && has_value) {
            oracle_path = argv[++i];
        } else if (std::strcmp(argv[i], "--mtp") == 0) {
            mtp = true;
        } else if (std::strcmp(argv[i], "--draft-mtp") == 0 && has_value) {
            if (!count(argv[++i], draft_k) || draft_k < 1) return usage();
            mtp = true;
            draft_mtp = true;
        } else if (std::strcmp(argv[i], "--mtp-out") == 0 && has_value) {
            mtp_out = argv[++i];
            mtp = true;
        } else if (std::strcmp(argv[i], "--draft-k") == 0 && has_value) {
            if (!count(argv[++i], draft_k) || draft_k < 1) return usage();
        } else if (std::strcmp(argv[i], "--draft-corrupt") == 0 && has_value) {
            if (!count(argv[++i], draft_corrupt)) return usage();
        } else {
            std::fprintf(stderr, "unknown option or missing value: %s\n", argv[i]);
            return usage();
        }
    }
    if (use_gemv && generate == 0) {
        std::fprintf(stderr, "note: --gemv only applies with --generate; prefill-only runs "
                             "use the f16 path\n");
    }
    try {
        std::vector<int32_t> toks;
        {
            std::ifstream in(tok_path);
            int64_t v = 0;
            while (in >> v) {
                if (v < 0 || v > INT32_MAX) {
                    std::fprintf(stderr, "token id %lld out of range in %s\n", (long long) v,
                                 tok_path.c_str());
                    return 1;
                }
                toks.push_back((int32_t) v);
            }
            if (!in.eof()) {
                std::fprintf(stderr, "%s: not a list of token ids\n", tok_path.c_str());
                return 1;
            }
        }
        if (toks.empty()) {
            std::fprintf(stderr, "no tokens read from %s\n", tok_path.c_str());
            return 1;
        }
        if (max_tokens > 0 && (int64_t) toks.size() > max_tokens) {
            toks.resize((size_t) max_tokens);
        }
        if (generate > 0 && gen_path.empty()) {
            std::fprintf(stderr, "--generate also needs --gen-out\n");
            return 2;
        }
        // Repacking only pays off when the decode runs: a prefill-only run is
        // better off with the raw bytes in place (no staging fallback).
        // Long prompts run in chunks: the activation buffers are sized by the
        // chunk, the KV cache by the whole sequence. Without this, a 8k prompt
        // needs ~3 GB of activations on top of the weights and does not fit.
        const int64_t total_len = (int64_t) toks.size() + generate + 8;
        int64_t act_chunk = total_len < 512 ? total_len : 512;
        if (max_tokens > 0 && max_tokens < act_chunk) {
            act_chunk = max_tokens;
        }
        // The per-layer trace is one file per layer for one forward call: with
        // several chunks it would hold the last chunk only.
        if (!trace_dir.empty() && (int64_t) toks.size() > act_chunk) {
            std::fprintf(stderr, "--trace-dir needs the prompt in one chunk (<= %lld tokens)\n",
                         (long long) act_chunk);
            return 2;
        }
        const omph::runtime::EnvOptions env = omph::runtime::EnvOptions::from_env();
        if (mtp && !(use_gemv && generate > 0)) {
            std::fprintf(stderr, "--mtp needs --gemv and --generate\n");
            return 2;
        }
        omph::model::Runner runner(model, act_chunk, use_gemv && generate > 0, env, last_logits,
                                   total_len, mtp);
        std::vector<int32_t> oracle;
        if (!oracle_path.empty()) {
            std::ifstream in(oracle_path);
            int64_t v = 0;
            while (in >> v) {
                oracle.push_back((int32_t) v);
            }
            if (oracle.empty() || generate == 0) {
                std::fprintf(stderr, "--draft-oracle needs a token file and --generate\n");
                return 2;
            }
            runner.enable_speculation(draft_k + 1);
        }
        if (draft_mtp) {
            if (!oracle.empty()) {
                std::fprintf(stderr, "--draft-mtp and --draft-oracle are exclusive\n");
                return 2;
            }
            runner.enable_speculation(draft_k + 1);
        }
        const omph::model::HParams & h = runner.hparams();
        // An id past the vocabulary would index the embedding out of bounds.
        for (const int32_t t : toks) {
            if (t >= h.n_vocab) {
                std::fprintf(stderr, "token id %d >= n_vocab %lld\n", t, (long long) h.n_vocab);
                return 1;
            }
        }
        // Every chunk's rows are kept, so the file is the whole prompt's logits;
        // with --last-logits only the last row, with --logits-tail N the last N.
        // Chunks that do not reach the rows being kept skip the lm_head.
        const int64_t n_toks = (int64_t) toks.size();
        const int64_t keep = last_logits ? 1 : (logits_tail > 0 ? std::min(logits_tail, n_toks)
                                                                : n_toks);
        std::vector<float> logits;
        std::vector<float> part_logits;
        for (int64_t off = 0; off < n_toks; off += act_chunk) {
            const int64_t n = std::min<int64_t>(act_chunk, n_toks - off);
            const std::vector<int32_t> part(toks.begin() + (size_t) off,
                                            toks.begin() + (size_t) (off + n));
            const bool want = off + n > n_toks - keep;
            if (!runner.forward(part, part_logits, trace_dir, off, want)) {
                return 1;
            }
            if (last_logits) {
                logits.swap(part_logits);
            } else if (want) {
                // rows of this chunk inside the tail
                const int64_t first = std::max<int64_t>(0, (n_toks - keep) - off);
                logits.insert(logits.end(), part_logits.begin() + (size_t) (first * h.n_vocab),
                              part_logits.end());
            }
        }
        write_f32(logits_path, logits);
        const auto argmax = [&](const float * row) {
            int64_t best = 0;
            for (int64_t i = 1; i < h.n_vocab; ++i) {
                if (row[i] > row[best]) {
                    best = i;
                }
            }
            return (int32_t) best;
        };
        if (generate > 0) {
            // greedy decode: one token per step, reusing the KV cache, the conv
            // state and the delta-net state
            const bool timing = env.timing;
            std::vector<float> step_logits;  // `logits` keeps the file's rows for the report
            // OMPH_HOST_ARGMAX=1: copy the logits back and take the argmax on the
            // host, as before #102 (A/B and validation).
            const bool host_argmax = env.host_argmax;
            std::vector<int32_t> gen;
            int32_t next = argmax(logits.data() + (logits.size() - h.n_vocab));
            if (!mtp_out.empty()) {
                // MTP validation (#124): two chained drafts after the prompt and
                // its greedy token, their logits rows to the file.
                std::vector<int32_t> drafts;
                std::vector<float> dl;
                if (!runner.mtp_draft(next, (int64_t) toks.size(), 2, drafts, &dl)) {
                    return 1;
                }
                write_f32(mtp_out, dl);
                std::printf("mtp drafts after %d:", next);
                for (const int32_t d : drafts) {
                    std::printf(" %d", d);
                }
                std::printf("\n");
            }
            if (!oracle.empty() || draft_mtp) {
                // Speculative greedy decode (#122, #124), the drafts from the MTP
                // head or, for validation, an oracle file. Oracle: the drafts
                // for generated token g are oracle[g ..], every draft_corrupt-th
                // one deliberately wrong. Each step verifies [next, drafts] in one
                // forward, keeps the drafts that match the verifier's own greedy
                // tokens plus the verifier's next one, and rolls back the rest.
                // The output must equal the plain greedy decode's.
                int64_t pos = (int64_t) toks.size();
                int64_t n_drafted = 0;
                int64_t n_accepted = 0;
                int64_t n_steps = 0;
                const double t0 = omph::runtime::now_ms();
                while (true) {
                    gen.push_back(next);
                    if ((int64_t) gen.size() >= generate) {
                        break;
                    }
                    std::vector<int32_t> batch{next};
                    if (draft_mtp) {
                        std::vector<int32_t> drafts;
                        if (!runner.mtp_draft(next, pos, draft_k, drafts)) {
                            return 1;
                        }
                        n_drafted += (int64_t) drafts.size();
                        batch.insert(batch.end(), drafts.begin(), drafts.end());
                    }
                    for (int64_t j = 0; j < draft_k && !draft_mtp; ++j) {
                        const size_t g = gen.size() + (size_t) j;
                        if (g >= oracle.size()) {
                            break;
                        }
                        ++n_drafted;
                        int32_t d = oracle[g];
                        if (draft_corrupt > 0 && n_drafted % draft_corrupt == 0) {
                            d = (int32_t) ((d + 1) % h.n_vocab);
                        }
                        batch.push_back(d);
                    }
                    std::vector<int32_t> am;
                    if (!runner.verify(batch, pos, am)) {
                        return 1;
                    }
                    int64_t a = 0;
                    while (a + 1 < (int64_t) batch.size() && batch[(size_t) a + 1] == am[(size_t) a]) {
                        ++a;
                    }
                    if (!runner.commit(a + 1)) {
                        return 1;
                    }
                    ++n_steps;
                    n_accepted += a;
                    for (int64_t j = 0; j < a && (int64_t) gen.size() < generate; ++j) {
                        gen.push_back(batch[(size_t) j + 1]);
                    }
                    if ((int64_t) gen.size() >= generate) {
                        break;
                    }
                    next = am[(size_t) a];
                    pos += a + 1;
                }
                std::fprintf(stderr,
                             "speculative: %lld steps, %lld / %lld drafts accepted, %.2f ms "
                             "per generated token\n",
                             (long long) n_steps, (long long) n_accepted, (long long) n_drafted,
                             (omph::runtime::now_ms() - t0) / (double) (gen.size() - 1));
            }
            for (int64_t i = (int64_t) gen.size(); i < generate; ++i) {
                gen.push_back(next);
                if (i + 1 == generate) {
                    break;
                }
                const double w0 = omph::runtime::now_ms();
                const std::vector<int32_t> one{next};
                int32_t on_device = -1;
                if (!runner.forward(one, step_logits, std::string(), (int64_t) toks.size() + i,
                                    true, host_argmax ? nullptr : &on_device)) {
                    return 1;
                }
                next = on_device >= 0 ? on_device : argmax(step_logits.data());
                if (timing) {
                    std::fprintf(stderr, "step wall %.2f ms\n", omph::runtime::now_ms() - w0);
                }
            }
            FILE * f = std::fopen(gen_path.c_str(), "w");
            bool ok = f != nullptr;
            for (const int32_t t : gen) {
                ok = ok && std::fprintf(f, "%d\n", t) > 0;
            }
            if (f == nullptr || std::fclose(f) != 0 || !ok) {
                std::fprintf(stderr, "cannot write %s\n", gen_path.c_str());
                return 1;
            }
            std::printf("generated:");
            for (const int32_t t : gen) {
                std::printf(" %d", t);
            }
            std::printf("\n");
        }
        // greedy tokens for the report
        const size_t n_rows = logits.size() / (size_t) h.n_vocab;
        std::printf("logits: %lld x %lld -> %s\ngreedy:", (long long) n_rows,
                    (long long) h.n_vocab, logits_path.c_str());
        for (size_t t = 0; t < n_rows; ++t) {
            const float * row = logits.data() + t * (size_t) h.n_vocab;
            std::printf(" %lld", (long long) argmax(row));
        }
        std::printf("\n");
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
    return 0;
}
