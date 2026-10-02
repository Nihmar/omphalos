#include "model/generator.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "runtime/timing.hh"

namespace omph::model {

Generator::Generator(const Config & config, const omph::runtime::EnvOptions & env)
    : config_(config), env_(env) {
    if (config_.chunk <= 0 || config_.context <= 0) {
        throw std::runtime_error("generator: chunk and context must be positive");
    }
    file_ = std::make_unique<omph::gguf::File>(config_.model);
    tokenizer_ = std::make_unique<omph::text::Tokenizer>(*file_);
    // The decode configuration: repacked weights for the GEMVs, one logits
    // row, the KV for the whole context.
    runner_ = std::make_unique<Runner>(config_.model, std::min(config_.chunk, config_.context),
                                       /*use_gemv=*/true, env_, /*last_logits_only=*/true,
                                       config_.context, config_.mtp);
    if (config_.mtp) {
        runner_->enable_speculation(config_.draft_k + 1);
    }
    for (const char * t : {"<|im_end|>", "<|endoftext|>"}) {
        const int32_t id = tokenizer_->find(t);
        if (id >= 0) eog_.push_back(id);
    }
    if (tokenizer_->eos() >= 0 && !is_eog(tokenizer_->eos())) {
        eog_.push_back(tokenizer_->eos());
    }
    im_start_ = tokenizer_->find("<|im_start|>");
    image_pad_ = tokenizer_->find("<|image_pad|>");
    if (config_.cache_mib > 0) {
        max_checkpoints_ = (size_t) (config_.cache_mib << 20) / runner_->checkpoint_bytes();
    }
}

bool Generator::is_eog(const int32_t id) const {
    return std::find(eog_.begin(), eog_.end(), id) != eog_.end();
}

// Runs toks[from..] at positions from.. in prefill chunks; the last chunk's
// logits row ends in last_logits. A chunk also ends at each of `cuts`, where
// a checkpoint is saved.
bool Generator::feed(const Expanded & p, const int64_t from, std::vector<float> & last_logits,
                     const std::vector<int64_t> & cuts, GenerateResult & res) {
    const std::vector<int32_t> & toks = p.tokens;
    const int64_t n = (int64_t) toks.size();
    const int64_t ne = runner_->hparams().n_embd;
    size_t ci = 0;
    std::vector<float> rows;  // the chunk's image rows
    for (int64_t off = from; off < n;) {
        while (ci < cuts.size() && cuts[ci] <= off) ++ci;
        int64_t end = std::min<int64_t>(off + config_.chunk, n);
        const bool cut = ci < cuts.size() && cuts[ci] <= end;
        if (cut) end = cuts[ci];
        const std::vector<int32_t> part(toks.begin() + off, toks.begin() + end);
        ForwardInputs in;
        rows.clear();
        for (int64_t j = off; j < end; ++j) {
            if (const Image * im = p.image_of[(size_t) j]) {
                in.rows.push_back(j - off);
                const float * src = im->embd.data() + p.image_row[(size_t) j] * ne;
                rows.insert(rows.end(), src, src + ne);
            }
        }
        in.embd = rows.data();
        if (!p.mpos.empty()) in.mpos.assign(p.mpos.begin() + 3 * off, p.mpos.begin() + 3 * end);
        const bool with_inputs = !in.rows.empty() || !in.mpos.empty();
        if (!runner_->forward(part, last_logits, std::string(), off, end == n, nullptr,
                              with_inputs ? &in : nullptr)) {
            return false;
        }
        if (cut) {
            const double t0 = omph::runtime::now_ms();
            if (!save_checkpoint(toks, end)) return false;
            res.checkpoint_ms += omph::runtime::now_ms() - t0;
        }
        off = end;
    }
    return true;
}

// The cached sequence still holds the checkpoint's tokens: its KV positions
// are the ones the checkpoint's state was computed with.
bool Generator::valid(const Checkpoint & c) const {
    return c.tokens.size() <= seq_.size() && std::equal(c.tokens.begin(), c.tokens.end(), seq_.begin());
}

// Where to save checkpoints in toks[from..]: before <|im_start|> tokens (a
// special token: the tokens before it do not depend on what follows), the
// first and the last one -- the first message of the new part (a shared
// system prompt, an edited message) and the generation prompt (a retried
// answer, a history whose reasoning the client dropped). Only where the
// prefill it saves is at least kMinSaved tokens: each cut costs one more pass
// over the weights for the tokens after it, plus the copy (~150 ms, #158).
std::vector<int64_t> Generator::checkpoint_positions(const std::vector<int32_t> & toks, const int64_t from) const {
    constexpr int64_t kMinSaved = 512;
    std::vector<int64_t> cand;
    if (max_checkpoints_ == 0 || im_start_ < 0) return cand;
    for (int64_t p = std::max<int64_t>(from + 1, 1); p < (int64_t) toks.size(); ++p) {
        if (toks[(size_t) p] == im_start_) cand.push_back(p);
    }
    std::vector<int64_t> cuts;
    int64_t base = from;  // where a restore of this prompt would start without these cuts
    for (const int64_t p : {cand.empty() ? -1 : cand.front(), cand.empty() ? -1 : cand.back()}) {
        if (p > base && p - base >= kMinSaved && (cuts.empty() || cuts.back() != p)) {
            cuts.push_back(p);
            base = p;
        }
    }
    return cuts;
}

bool Generator::save_checkpoint(const std::vector<int32_t> & toks, const int64_t pos) {
    void * buf = nullptr;
    if (checkpoints_.size() >= max_checkpoints_) {  // the least recently used goes
        auto lru = std::min_element(checkpoints_.begin(), checkpoints_.end(),
                                    [](const Checkpoint & a, const Checkpoint & b) { return a.used < b.used; });
        buf = lru->host;
        checkpoints_.erase(lru);
    } else if (!spare_.empty()) {
        buf = spare_.back();
        spare_.pop_back();
    } else {
        try {
            buf = runner_->checkpoint_alloc();
        } catch (const std::exception &) {  // no more pinned memory: keep what we have
            max_checkpoints_ = checkpoints_.size();
            return true;
        }
    }
    if (!runner_->checkpoint_save(buf)) {
        spare_.push_back(buf);
        return false;
    }
    checkpoints_.push_back({std::vector<int32_t>(toks.begin(), toks.begin() + pos), buf, ++clock_});
    return true;
}

// Drops the checkpoints that cannot become valid again (the cached sequence
// differs within their tokens), or all of them.
void Generator::drop_checkpoints(const bool all) {
    for (size_t i = checkpoints_.size(); i-- > 0;) {
        const Checkpoint & c = checkpoints_[i];
        const size_t m = std::min(c.tokens.size(), seq_.size());
        if (all || !std::equal(c.tokens.begin(), c.tokens.begin() + m, seq_.begin())) {
            spare_.push_back(c.host);
            checkpoints_.erase(checkpoints_.begin() + (std::ptrdiff_t) i);
        }
    }
}

// Brings the caches to the longest usable prefix of `prompt`: the cached
// sequence when the prompt extends it, else the latest valid checkpoint
// within their common prefix, else nothing. Returns its length.
int64_t Generator::resume(const std::vector<int32_t> & prompt, GenerateResult & res) {
    size_t common = 0;
    while (common < seq_.size() && common < prompt.size() && seq_[common] == prompt[common]) ++common;
    if (!seq_.empty() && common == seq_.size() && common < prompt.size()) {
        return (int64_t) common;
    }

    Checkpoint * best = nullptr;
    for (Checkpoint & c : checkpoints_) {
        if (c.tokens.size() <= common && c.tokens.size() < prompt.size() && valid(c) &&
            (best == nullptr || c.tokens.size() > best->tokens.size())) {
            best = &c;
        }
    }
    if (best != nullptr) {
        const double t0 = omph::runtime::now_ms();
        if (runner_->checkpoint_restore(best->host)) {
            res.checkpoint_ms += omph::runtime::now_ms() - t0;
            res.restored = true;
            best->used = ++clock_;
            seq_.resize(best->tokens.size());
            return (int64_t) seq_.size();
        }
    }
    seq_.clear();
    return runner_->reset_sequence() ? 0 : -1;
}

// Host sampling: temperature, then top-k, min-p and top-p over the tokens
// within e^-30 of the best (the rest cannot matter), from a seeded generator.
int32_t Generator::sample(const std::vector<float> & logits, const Sampling & s) {
    const int64_t nv = (int64_t) logits.size();
    if (s.temperature <= 0.0f) {
        return (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    const float inv_t = 1.0f / s.temperature;
    const float best = *std::max_element(logits.begin(), logits.end());
    std::vector<std::pair<float, int32_t>> cand;  // (scaled logit, id)
    for (int64_t i = 0; i < nv; ++i) {
        const float z = (logits[(size_t) i] - best) * inv_t;
        if (z > -30.0f) {
            cand.emplace_back(z, (int32_t) i);
        }
    }
    std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    if (s.top_k > 0 && (size_t) s.top_k < cand.size()) {
        cand.resize((size_t) s.top_k);
    }
    std::vector<double> p(cand.size());
    double sum = 0.0;
    for (size_t i = 0; i < cand.size(); ++i) {
        p[i] = std::exp((double) cand[i].first);
        sum += p[i];
    }
    size_t keep = cand.size();
    if (s.min_p > 0.0f) {  // relative to the best (p[0] / sum)
        const double floor = (double) s.min_p * p[0];
        keep = 1;
        while (keep < cand.size() && p[keep] >= floor) ++keep;
    }
    if (s.top_p < 1.0f) {
        double acc = 0.0;
        size_t k = 0;
        while (k < keep) {
            acc += p[k] / sum;
            ++k;
            if (acc >= (double) s.top_p) break;
        }
        keep = k;
    }
    double total = 0.0;
    for (size_t i = 0; i < keep; ++i) total += p[i];
    std::uniform_real_distribution<double> u(0.0, total);
    double r = u(rng_);
    for (size_t i = 0; i < keep; ++i) {
        r -= p[i];
        if (r <= 0.0) return cand[i].second;
    }
    return cand[keep - 1].second;
}

// The prompt with each <|image_pad|> replaced by its image's rows, and the
// positions: text advances by 1, an image starting at p0 gives its row i
// (t, h, w) = (p0, p0 + i / nx, p0 + i % nx) and advances by max(nx, ny), as
// llama.cpp's mtmd does for M-RoPE models.
bool Generator::expand(const std::vector<int32_t> & prompt, const GenerateRequest & req, Expanded & out) const {
    const int64_t ne = runner_->hparams().n_embd;
    const bool mrope = !req.images.empty();
    size_t next = 0;
    int64_t r = 0;
    for (const int32_t t : prompt) {
        if (t < 0 || t >= tokenizer_->size()) return false;
        if (t != image_pad_ || image_pad_ < 0) {
            out.tokens.push_back(t);
            out.image_of.push_back(nullptr);
            out.image_row.push_back(0);
            if (mrope) out.mpos.insert(out.mpos.end(), {(int32_t) r, (int32_t) r, (int32_t) r});
            ++r;
            continue;
        }
        if (next >= req.images.size()) return false;  // a placeholder without an image
        const Image * im = req.images[next++].get();
        if (im == nullptr || im->nx <= 0 || im->ny <= 0 || (int64_t) im->embd.size() != im->n_tokens() * ne) {
            return false;
        }
        const int32_t id = -1 - (int32_t) (im->hash & 0x3fffffff);
        for (int64_t i = 0; i < im->n_tokens(); ++i) {
            out.tokens.push_back(id);
            out.image_of.push_back(im);
            out.image_row.push_back(i);
            int32_t h = (int32_t) (r + i / im->nx), w = (int32_t) (r + i % im->nx);
            if (env_.test_mrope == 1) std::swap(h, w);           // validation ablations
            if (env_.test_mrope == 2) h = w = (int32_t) (r + i);
            out.mpos.insert(out.mpos.end(), {env_.test_mrope == 2 ? h : (int32_t) r, h, w});
        }
        r += env_.test_mrope == 2 ? im->n_tokens() : std::max(im->nx, im->ny);
    }
    out.rope_end = r;
    return next == req.images.size();
}

GenerateResult Generator::generate(const std::vector<int32_t> & prompt_ids, const GenerateRequest & req,
                                   const std::function<bool(int32_t)> & on_token) {
    GenerateResult res;
    Expanded ex;
    if (prompt_ids.empty() || !expand(prompt_ids, req, ex)) {
        res.stop = GenerateResult::Stop::Error;
        return res;
    }
    const std::vector<int32_t> & prompt = ex.tokens;
    res.prompt_tokens = (int64_t) prompt.size();
    if ((int64_t) prompt.size() >= config_.context) {
        res.stop = GenerateResult::Stop::ContextFull;
        return res;
    }
    rng_.seed(req.sampling.seed);
    const double t0 = omph::runtime::now_ms();
    const int64_t from = resume(prompt, res);
    if (from < 0) {
        drop_checkpoints(true);
        res.stop = GenerateResult::Stop::Error;
        return res;
    }
    res.cached_tokens = from;
    std::vector<float> logits;
    runner_->set_rope_delta(0);  // text-only prompts; with images every chunk has its positions
    if (!feed(ex, from, logits, checkpoint_positions(prompt, from), res)) {
        seq_.clear();
        drop_checkpoints(true);
        res.stop = GenerateResult::Stop::Error;
        return res;
    }
    seq_ = prompt;
    drop_checkpoints(false);
    // generated tokens: RoPE position = cache position + delta (0 without images)
    runner_->set_rope_delta(ex.rope_end - (int64_t) prompt.size());
    if (req.prefill_logits != nullptr) *req.prefill_logits = logits;
    const double t1 = omph::runtime::now_ms();
    res.prefill_ms = t1 - t0;

    const bool forcing = req.force != nullptr && !req.force->empty();
    const bool greedy = req.sampling.temperature <= 0.0f && !forcing;
    int32_t next = forcing ? (*req.force)[0] : sample(logits, req.sampling);
    if (forcing && req.forced_logits != nullptr) {
        req.forced_logits->insert(req.forced_logits->end(), logits.begin(), logits.end());
    }
    // a token decided: report it, and say whether to go on
    const auto emit = [&](const int32_t t) {
        res.tokens.push_back(t);
        const bool cont = on_token ? on_token(t) : true;
        if (is_eog(t)) {
            res.stop = GenerateResult::Stop::EndOfGeneration;
            return false;
        }
        if (std::find(req.stop.begin(), req.stop.end(), t) != req.stop.end()) {
            res.stop = GenerateResult::Stop::StopToken;
            return false;
        }
        if (!cont) {
            res.stop = GenerateResult::Stop::Callback;
            return false;
        }
        if ((int64_t) res.tokens.size() >= req.max_tokens) {
            res.stop = GenerateResult::Stop::Length;
            return false;
        }
        if ((int64_t) seq_.size() + 1 >= config_.context) {
            res.stop = GenerateResult::Stop::ContextFull;
            return false;
        }
        return true;
    };
    if (req.max_tokens <= 0) {
        res.stop = GenerateResult::Stop::Length;
        return res;
    }
    bool go = emit(next);
    if (greedy && req.speculative && config_.mtp && !forcing) {
        // Speculative greedy (#122, #124): draft k tokens with the MTP block,
        // verify [next, drafts] in one forward, keep the drafts the model
        // agrees with plus its own next token. Same tokens as plain greedy.
        while (go) {
            const int64_t pos = (int64_t) seq_.size();
            const int64_t k = std::min<int64_t>(config_.draft_k, config_.context - pos - 1);
            std::vector<int32_t> batch{next};
            if (k > 0) {
                std::vector<int32_t> drafts;
                if (!runner_->mtp_draft(next, pos, k, drafts)) {
                    res.stop = GenerateResult::Stop::Error;
                    break;
                }
                res.drafted += (int64_t) drafts.size();
                batch.insert(batch.end(), drafts.begin(), drafts.end());
            }
            std::vector<int32_t> am;
            if (!runner_->verify(batch, pos, am)) {
                res.stop = GenerateResult::Stop::Error;
                break;
            }
            int64_t a = 0;
            while (a + 1 < (int64_t) batch.size() && batch[(size_t) a + 1] == am[(size_t) a]) {
                ++a;
            }
            res.accepted += a;
            // The accepted drafts in order; when one ends the generation, the
            // caches keep the tokens before it, as the plain path never feeds
            // its last token (else an end-of-generation and what the model
            // drafted after it would sit in the cached sequence, #160).
            int64_t keep = a + 1;
            for (int64_t j = 1; j <= a; ++j) {
                if (!emit(batch[(size_t) j])) {
                    keep = j;
                    go = false;
                    break;
                }
            }
            if (!runner_->commit(keep)) {
                res.stop = GenerateResult::Stop::Error;
                break;
            }
            seq_.insert(seq_.end(), batch.begin(), batch.begin() + keep);
            if (go) {
                next = am[(size_t) a];
                go = emit(next);
            }
        }
    } else {
        std::vector<float> step;
        while (go) {
            const std::vector<int32_t> one{next};
            int32_t on_device = -1;
            if (!runner_->forward(one, step, std::string(), (int64_t) seq_.size(), true,
                                  greedy ? &on_device : nullptr)) {
                res.stop = GenerateResult::Stop::Error;
                break;
            }
            seq_.push_back(next);
            if (forcing) {
                if (req.forced_logits != nullptr) {
                    req.forced_logits->insert(req.forced_logits->end(), step.begin(), step.end());
                }
                if (res.tokens.size() >= req.force->size()) break;
                next = (*req.force)[res.tokens.size()];
            } else {
                next = on_device >= 0 ? on_device : sample(step, req.sampling);
            }
            go = emit(next);
        }
    }
    if (res.stop == GenerateResult::Stop::Error) {
        seq_.clear();  // the caches are in an unknown state: start over next time
        drop_checkpoints(true);
    }
    res.decode_ms = omph::runtime::now_ms() - t1;
    return res;
}

} // namespace omph::model
