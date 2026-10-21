// Runner: the forward pass (embedding, layer loop, final norm, lm_head).
#include "model/runner.hh"

#include "format/repack.hh"
#include "kernels/attn.hh"
#include "kernels/dflash.hh"
#include "kernels/dequant.hh"
#include "kernels/elementwise.hh"
#include "kernels/gdn.hh"
#include "kernels/gemv.hh"
#include "kernels/gemm.hh"
#include "kernels/moe.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>

namespace omph::model {

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

// want_logits = false runs the layers only (a prefill chunk whose logits
// nobody reads): no final norm, no lm_head.
// greedy (decode, --gemv): set to the argmax token computed on the device,
// and the logits stay there (#102); -1 when this path did not run, and the
// caller takes the argmax of `logits`.
// Token rows dequantized from the table (pinned host memory: the kernel reads
// the row over PCIe), image rows copied from `in`.
bool Runner::embed(const int32_t * toks, const int64_t T, const ForwardInputs * in, float * dst) {
    const int64_t ne = h_.n_embd;
    // a decode step or a verification: a dequant per row is cheaper than the
    // id upload plus the gather
    const bool gather = T > 4;
    if (gather && T > embd_cap_) {  // grown geometrically; the old buffers stay with mem_
        embd_cap_ = std::max<int64_t>(T, 2 * embd_cap_);
        embd_ids_ = static_cast<int32_t *>(mem_.device((size_t) embd_cap_ * 4, "out of VRAM (embedding ids)"));
        embd_stage_ = mem_.device((size_t) (embd_cap_ * embd_row_bytes_), "out of VRAM (embedding rows)");
    }
    // (pageable source: copied to staging before the call returns, no wait on the GPU)
    if (gather && hipMemcpyAsync(embd_ids_, toks, (size_t) T * 4, hipMemcpyHostToDevice, nullptr) != hipSuccess) {
        return fail("embedding ids upload failed");
    }
    // the token rows of [t0, t): one gather + one dequant (#223), or one dequant per row
    const auto rows = [&](const int64_t t0, const int64_t t) {
        if (gather) {
            return t == t0 || omph::kernels::dequantize_rows(embd_type_, embd_host_, embd_row_bytes_,
                                                             embd_ids_ + t0, t - t0, ne, embd_stage_,
                                                             dst + t0 * ne, nullptr);
        }
        for (int64_t i = t0; i < t; ++i) {
            const uint8_t * src = static_cast<const uint8_t *>(embd_host_) + (size_t) toks[i] * embd_row_bytes_;
            if (!omph::kernels::dequantize(embd_type_, src, dst + i * ne, ne, false, nullptr)) {
                return false;
            }
        }
        return true;
    };
    size_t next = 0;  // the next image row of `in`
    int64_t run0 = 0;  // the first token row not yet embedded
    for (int64_t t = 0; t < T; ++t) {
        if (in != nullptr && next < in->rows.size() && in->rows[next] == t) {
            if (!rows(run0, t)) {
                return fail("embedding dequant failed");
            }
            size_t run = 1;  // a run of consecutive image rows: one copy
            while (next + run < in->rows.size() && in->rows[next + run] == t + (int64_t) run) ++run;
            if (hipMemcpyAsync(dst + t * ne, in->embd + next * ne, run * ne * 4, hipMemcpyHostToDevice,
                               nullptr) != hipSuccess) {
                return fail("image embedding upload failed");
            }
            t += (int64_t) run - 1;
            next += run;
            run0 = t + 1;
            continue;
        }
    }
    return rows(run0, T) || fail("embedding dequant failed");
}


// The mixture-of-experts FFN (#493): the routed part through the selected
// experts, then the shared expert gated by a sigmoid, accumulated into cur_.
//
// The routed gate runs once for the whole chunk (one block per token). v1 reads
// the selection back to the host: the expert slices are picked by pointer, so
// the dispatch needs their ids here. A device-side indexed GEMV removes the
// round trip later (#493 step 2).
// The prefill shape (#494): one router pass for the chunk, the rows gathered
// per expert so each expert's tokens are one contiguous GEMM input, then the
// slot-ordered weighted sum. The order of the sums is fixed (a store per
// (token, slot) and one combine in slot order): the result does not depend on
// the atomics' order.
// The expert cache (#512), before a forward's kernels: the selections the
// last logged step made (once their copy is done) touch their experts, and
// each host-read expert of a layer takes the least recently used slot whose
// expert that step did not choose. All on the compute stream, in order: the
// fills (a copy kernel from the pinned copy), then the table (#513). The
// victims' last readers are earlier kernels of the same stream, the new
// experts' first readers later ones, and every write goes through the GPU's
// L2. An asynchronous version - fills on a copy stream, the table by DMA -
// changed the text of one n-gram run in a few (a kernel read a slot while its
// fill wrote it) and, with the fills on a DMA engine, decoded half the runs at
// 22 ms/step instead of 12.
bool Runner::moe_cache_begin() {
    if (moe_tabs_.empty() || moe_log_T_ == 0 || hipEventQuery(moe_ev_log_) != hipSuccess) {
        return true;
    }
    const int64_t n = h_.n_expert;
    const int64_t picks = moe_log_T_ * h_.n_expert_used;
    ++moe_clock_;
    if (moe_trace_ != nullptr) {
        const int32_t t32 = (int32_t) moe_log_T_;
        std::fwrite(&t32, 4, 1, moe_trace_);
        for (int64_t tab = 0; tab < moe_log_tabs_; ++tab) {
            std::fwrite(moe_log_ + (size_t) tab * (size_t) (kMoeLogTokens * h_.n_expert_used), 4, (size_t) picks,
                        moe_trace_);
        }
    }
    bool changed = false;
    for (int64_t tab = 0; tab < moe_log_tabs_; ++tab) {
        MoeTab & c = moe_tabs_[(size_t) tab];
        const int * ids = moe_log_ + (size_t) tab * (size_t) (kMoeLogTokens * h_.n_expert_used);
        for (int64_t i = 0; i < picks; ++i) {
            c.last[(size_t) ids[i]] = moe_clock_;
            ++c.count[(size_t) ids[i]];
        }
        for (int64_t i = 0; i < picks; ++i) {
            const int e = ids[i];
            if (c.slot_of[(size_t) e] >= 0) {
                continue;
            }
            ++moe_misses_;
            // admitted on its second miss within kMoeAdmitSteps steps (#522): a
            // fill costs the PCIe read the miss already paid, so an expert seen
            // once is not worth it. Measured (400 tokens, step gpu): no
            // admission 12.17 / 11.47 / 11.58 ms (code / ita / tech), 4 steps
            // 11.50 / 11.07 / 11.29, 8 11.52 / 11.05, 16 11.62 / 11.10 / 11.27,
            // never 13.22 / 11.18 / 13.18
            const int64_t prev = c.missed[(size_t) e];
            c.missed[(size_t) e] = moe_clock_;
            if (prev == 0 || moe_clock_ - prev > kMoeAdmitSteps) {
                continue;
            }
            // the least recently used slot whose expert this step did not choose
            int victim = -1;
            for (int64_t s = moe_fixed_; s < moe_cap_; ++s) {
                const int o = c.owner[(size_t) s];
                if (c.last[(size_t) o] < moe_clock_ &&
                    (victim < 0 || c.last[(size_t) o] < c.last[(size_t) c.owner[(size_t) victim]])) {
                    victim = (int) s;
                }
            }
            if (victim < 0) {
                continue;
            }
            const int old = c.owner[(size_t) victim];
            c.slot_of[(size_t) old] = -1;
            c.owner[(size_t) victim] = e;
            c.slot_of[(size_t) e] = victim;
            for (int k = 0; k < 3; ++k) {
                const void ** row = moe_eptr_host_.data() + ((size_t) tab * 3 + (size_t) k) * (size_t) n;
                uint8_t * slot = const_cast<uint8_t *>(c.dev[k]) + (size_t) victim * c.pe[k];
                row[old] = c.host[k] + (size_t) c.hidx[(size_t) old] * c.pe[k];
                row[e] = slot;
                if (!omph::kernels::moe_copy_words(slot, 0, c.host[k] + (size_t) c.hidx[(size_t) e] * c.pe[k], 0,
                                                   (int64_t) (c.pe[k] / 4), 1, hipStreamPerThread)) {
                    return fail("expert fill failed");
                }
            }
            changed = true;
        }
        moe_selects_ += picks;
    }
    moe_log_T_ = 0;
    // through a pinned buffer (a pageable source is pinned by the runtime
    // during the copy, which under host memory pressure stalled the compute
    // stream for seconds, #512), read by a kernel
    if (changed) {
        const size_t words = moe_eptr_host_.size() * sizeof(void *) / 4;
        if (hipEventSynchronize(moe_ev_up_) != hipSuccess) {
            return fail("expert table upload wait failed");
        }
        std::memcpy(moe_eptr_pin_, moe_eptr_host_.data(), words * 4);
        if (!omph::kernels::moe_copy_words(moe_eptr_, (int64_t) words, moe_eptr_pin_, (int64_t) words,
                                           (int64_t) words, 1, hipStreamPerThread) ||
            hipEventRecord(moe_ev_up_, hipStreamPerThread) != hipSuccess) {
            return fail("expert table upload failed");
        }
    }
    return true;
}

// A step's selections back to the host (#512): one strided copy of every
// logged layer's T x topk ids after the layers, its event polled by the next
// moe_cache_begin. A chunk (T > kMoeLogTokens) is not logged, nor is a step
// while the previous log is still unread.
bool Runner::moe_cache_log(const int64_t T) {
    if (moe_tabs_.empty() || T > kMoeLogTokens || moe_log_T_ > 0) {
        return true;
    }
    if (!omph::kernels::moe_copy_words(moe_log_, kMoeLogTokens * h_.n_expert_used, moe_ids_, moe_ids_stride_,
                                       T * h_.n_expert_used, moe_log_tabs_, hipStreamPerThread) ||
        hipEventRecord(moe_ev_log_, hipStreamPerThread) != hipSuccess) {
        return fail("expert log copy failed");
    }
    moe_log_T_ = T;
    return true;
}

// The routed gate's selection into moe_ids_ / moe_w_: the logits from a GEMV
// for a step or a verification (one token's logits do not depend on T, #509)
// and from the f16 GEMM for a prefill chunk (#494), then the top-k. The input
// is h16_ + h16lo_ (#547): rounded to f16 alone it flipped 0.14 % of the picks
// against an exact router (llama.cpp's f32 one: 0.005 %); the chunk's GEMM
// runs on each half and the top-k sums the two.
bool Runner::moe_route(const LayerWeights & L, const int64_t T) {
    const int64_t ne = h_.n_embd;
    const int64_t n_expert = h_.n_expert;
    auto * logits = static_cast<float *>(moe_logits_);
    auto * logits_lo = logits + T * n_expert;
    const bool gemv = T <= kMoeRouterGemvMax;
    const bool ok = gemv ? omph::kernels::moe_router(L.gate_inp16, h16_, h16lo_, logits, T, n_expert, ne,
                                                     hipStreamPerThread)
                         : omph::kernels::gemm_f16(L.gate_inp16, h16_, logits, n_expert, ne, T, n_expert,
                                                   hipStreamPerThread) &&
                               omph::kernels::gemm_f16(L.gate_inp16, h16lo_, logits_lo, n_expert, ne, T, n_expert,
                                                       hipStreamPerThread);
    return ok && omph::kernels::moe_topk(logits, gemv ? nullptr : logits_lo, moe_ids(L),
                                         static_cast<float *>(moe_w_), T, n_expert,
                                         (int) h_.n_expert_used, hipStreamPerThread);
}

bool Runner::moe_ffn_batched(const LayerWeights & L, const int64_t T) {
    const int64_t ne = h_.n_embd;
    const int64_t ffn = h_.n_ff_exp;
    const int topk = (int) h_.n_expert_used;
    const int n_expert = (int) h_.n_expert;
    auto * ids = moe_ids(L);

    if (!moe_route(L, T)) {
        return fail("moe route failed");
    }
    // The routing table, the gather / store indices and the per-expert offsets
    // are all built on the device (#493): a batched step (a prefill chunk or a
    // verification) has no host round trip left.
    auto * xgi = static_cast<int *>(moe_xgi_);
    auto * slot = static_cast<int *>(moe_slot_);
    if (!omph::kernels::moe_order(ids, T * topk, n_expert, topk, moe_cnt_, moe_off_, xgi, slot,
                                  moe_cnt_max_, hipStreamPerThread)) {
        return fail("moe order failed");
    }
    // Every expert's rows gathered once (the counting sort already ordered
    // them by expert), then one grouped GEMM per matrix kind where the type has
    // a tile kernel, and the per-expert shape on the gathered rows where it
    // does not (#494).
    const int64_t n_rows = T * topk;
    auto * xg = static_cast<__half *>(moe_xg_);
    auto * xo = static_cast<float *>(moe_xo_);
    auto * ffn1 = static_cast<float *>(ffn1_);
    auto * ffn2 = static_cast<float *>(ffn2_);
    auto * down_out = static_cast<float *>(moe_down_);
    if (!omph::kernels::moe_gather(h16_, ne, xgi, n_rows, xg, hipStreamPerThread)) {
        return fail("moe gather (batched) failed");
    }
    // The grid's row bound: the largest per-expert count, read from the device
    // for a chunk (one 4-byte copy per layer, amortized over hundreds of rows);
    // for a step or a verification the bound is the token count itself, so no
    // host round trip happens
    int64_t max_rows = T;
    if (T > 64) {
        int mx = 0;
        if (hipMemcpy(&mx, moe_cnt_max_, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
            return fail("moe max count readback failed");
        }
        max_rows = mx > 0 ? mx : 1;
    }
    // The expert slices (the expert dim is the outermost in the 3-D tensor)
    // and the x of a grouped GEMM (the gathered rows for gate / up, the
    // swiglu output on the same rows for the down)
    const int64_t gate_rows = (int64_t) L.ffn_gate_exps.t->ne[1];
    const int64_t up_rows = (int64_t) L.ffn_up_exps.t->ne[1];
    const int64_t down_rows = (int64_t) L.ffn_down_exps.t->ne[1];
    // The expert slices: the expert dim is the outermost in the 3-D tensor, so
    // a slice is a pointer plus a GemvEntry with the slice's rows/k
    const auto slice = [this](const Mat & base, const int64_t e, const int64_t rows,
                              const int64_t k) -> Mat {
        Mat m = base;
        m.dev = expert_ptr(base, e);
        m.sliced = true;
        if (base.gemv != nullptr) {
            slice_gemv_ = *base.gemv;
            slice_gemv_.rows = rows;
            slice_gemv_.k = k;
            m.gemv = &slice_gemv_;
        }
        return m;
    };
    // The x of a grouped GEMM is the gathered rows (gate / up) or the swiglu
    // output on the same rows (down): both are laid out per gathered row.
    const bool no_group = std::getenv("OMPH_MOE_NO_GROUP") != nullptr;  // A/B (#494)
    const auto grouped = [&](const Mat & w, const void * x,
                             const int64_t n_out, const int64_t k, float * y, const int64_t ldy) {
        if (no_group) {
            return false;
        }
        return omph::kernels::gemm_q_grouped(w.t->type, w.experts, x,
                                             static_cast<const int *>(moe_off_),
                                             static_cast<const int *>(moe_cnt_), n_expert, max_rows, y,
                                             n_out, k, ldy, hipStreamPerThread);
    };
    // The per-expert fallback (a type without a tile kernel) walks the experts
    // in order, so it needs the counts on the host: one readback, once.
    bool host_counts = false;
    const auto ensure_host_counts = [&]() {
        if (host_counts) {
            return true;
        }
        const bool ok =
            hipMemcpy(moe_counts_.data(), moe_cnt_, (size_t) n_expert * sizeof(int),
                      hipMemcpyDeviceToHost) == hipSuccess &&
            hipMemcpy(moe_host_ids_.data(), ids, (size_t) T * topk * sizeof(int),
                      hipMemcpyDeviceToHost) == hipSuccess;
        host_counts = ok;
        return ok;
    };
    // gate and up: grouped when the type has a tile kernel, else one expert at a
    // time, both writing at the gathered row offsets
    if (!grouped(L.ffn_gate_exps, xg, ffn, ne, ffn1, ffn) ||
        !grouped(L.ffn_up_exps, xg, ffn, ne, ffn2, ffn)) {
        if (!ensure_host_counts()) {
            return fail("moe fallback readback failed");
        }
        for (int e = 0, at = 0; e < n_expert; ++e) {
            const int64_t n_e = moe_counts_[(size_t) e];
            const int64_t off = at;
            at += n_e;
            if (n_e > 0 &&
                (!matmul(slice(L.ffn_gate_exps, e, gate_rows, ne),
                         xg + off * ne, ffn1 + off * ffn, ffn, ne, n_e) ||
                 !matmul(slice(L.ffn_up_exps, e, up_rows, ne), xg + off * ne,
                         ffn2 + off * ffn, ffn, ne, n_e))) {
                return fail("moe gate/up (batched) failed");
            }
        }
    }
    // the swiglu first: the down projections read its output
    if (!omph::kernels::swiglu_f16(static_cast<const float *>(ffn1_),
                                   static_cast<const float *>(ffn2_), ffn16_, n_rows * ffn,
                                   hipStreamPerThread)) {
        return fail("moe swiglu (batched) failed");
    }
    if (!grouped(L.ffn_down_exps, ffn16_, ne, ffn, down_out, ne)) {
        if (!ensure_host_counts()) {
            return fail("moe fallback readback failed");
        }
        for (int e = 0, at = 0; e < n_expert; ++e) {
            const int64_t n_e = moe_counts_[(size_t) e];
            const int64_t off = at;
            at += n_e;
            if (n_e > 0 &&
                !matmul(slice(L.ffn_down_exps, e, down_rows, ffn),
                        reinterpret_cast<const __half *>(ffn16_) + off * ffn, down_out + off * ne, ne,
                        ffn, n_e)) {
                return fail("moe down (batched) failed");
            }
        }
    }
    if (!omph::kernels::moe_scatter_store(xo, down_out, slot, xgi, n_rows, ne, topk,
                                          hipStreamPerThread)) {
        return fail("moe store (batched) failed");
    }
    // the shared expert into moe_down_ (the routed rows are in xo now), then the
    // slot-ordered weighted sum plus the gated shared output
    return moe_shared(L, T, static_cast<float *>(moe_down_)) &&
           moe_finish(L, T, xo, static_cast<float *>(moe_down_));
}

bool Runner::moe_ffn(const LayerWeights & L, const int64_t T) {
    const int64_t ne = h_.n_embd;
    // A verification (or a short --gemv run) takes the decode's shape over
    // T x topk slots: every token's result is its decode result bit for bit,
    // so speculation cannot change the text (#496); a chunk is batched.
    if (T > 1 && !(T <= kMoeRouterGemvMax && gemv_path(T))) {
        return moe_ffn_batched(L, T);
    }
    const int64_t ffn = h_.n_ff_exp;
    const int topk = (int) h_.n_expert_used;
    auto * ids = moe_ids(L);
    auto * weights = static_cast<float *>(moe_w_);

    if (!moe_route(L, T)) {
        return fail("moe route failed");
    }
    // A verification (#551): the routing table orders the block's picks by
    // expert, and each chosen expert's weights are read once for all of its
    // tokens (the per-slot shape below read them once per (token, expert)).
    // Every (token, expert) is that slot's decode result bit for bit, and the
    // combine runs in slot order as below, so the text cannot change.
    const auto & ge = L.ffn_gate_exps;
    const auto & de = L.ffn_down_exps;
    if (T > 1 && !env_.moe_no_rows && ge.t->type == L.ffn_up_exps.t->type &&
        ge.t->ne[1] == L.ffn_up_exps.t->ne[1] && omph::kernels::gemv_moe_rows_type(ge.t->type) &&
        omph::kernels::gemv_moe_rows_type(de.t->type)) {
        const int64_t n_rows = T * topk;
        const int64_t max_act = std::min<int64_t>(n_rows, h_.n_expert);
        auto * xgi = static_cast<int *>(moe_xgi_);
        auto * slot = static_cast<int *>(moe_slot_);
        auto * down_out = static_cast<float *>(moe_down_);
        auto * xo = static_cast<float *>(moe_xo_);
        if (!omph::kernels::moe_order(ids, n_rows, h_.n_expert, topk, moe_cnt_, moe_off_, xgi, slot, moe_cnt_max_,
                                      hipStreamPerThread, moe_act_) ||
            !omph::kernels::gemv_moe_rows_gu(ge.t->type, ge.experts, L.ffn_up_exps.experts, moe_act_, moe_off_,
                                             moe_cnt_, xgi, h16_, ffn16_, (int64_t) ge.t->ne[1], ne, max_act,
                                             hipStreamPerThread) ||
            !omph::kernels::gemv_moe_rows(de.t->type, de.experts, moe_act_, moe_off_, moe_cnt_, nullptr, ffn16_,
                                          down_out, (int64_t) de.t->ne[1], ffn, max_act, hipStreamPerThread) ||
            !omph::kernels::moe_scatter_store(xo, down_out, slot, xgi, n_rows, ne, topk, hipStreamPerThread)) {
            return fail("moe grouped verification failed");
        }
        // the routed rows are in xo: the shared expert takes moe_down_
        return moe_shared(L, T, down_out) && moe_finish(L, T, xo, down_out);
    }
    // The device-side shape (#497): the selection never leaves the device, and
    // each matrix kind is one launch over every selected expert of every token.
    // It needs all three to be tile-layout types; the per-expert path below
    // falls back.
    {
        const int64_t slots = T * topk;
        const int64_t gate_rows = (int64_t) L.ffn_gate_exps.t->ne[1];
        const int64_t up_rows = (int64_t) L.ffn_up_exps.t->ne[1];
        const int64_t down_rows = (int64_t) L.ffn_down_exps.t->ne[1];
        auto * ffn1 = static_cast<float *>(ffn1_);
        auto * ffn2 = static_cast<float *>(ffn2_);
        auto * down_out = static_cast<float *>(moe_down_);
        // gate, up and SwiGLU in one launch when both are of one tile type
        // (#522), else the two GEMVs and swiglu_f16 - the same values
        const bool gu = L.ffn_gate_exps.t->type == L.ffn_up_exps.t->type && gate_rows == up_rows &&
                        omph::kernels::gemv_moe_gu(L.ffn_gate_exps.t->type, L.ffn_gate_exps.experts,
                                                   L.ffn_up_exps.experts, ids, h16_, ffn16_, gate_rows, ne,
                                                   slots, topk, hipStreamPerThread);
        if ((gu ||
             (omph::kernels::gemv_moe(L.ffn_gate_exps.t->type, L.ffn_gate_exps.experts, ids,
                                      h16_, ffn1, gate_rows, ne, slots, topk, hipStreamPerThread) &&
              omph::kernels::gemv_moe(L.ffn_up_exps.t->type, L.ffn_up_exps.experts, ids, h16_,
                                      ffn2, up_rows, ne, slots, topk, hipStreamPerThread) &&
              omph::kernels::swiglu_f16(static_cast<const float *>(ffn1_),
                                        static_cast<const float *>(ffn2_), ffn16_, slots * h_.n_ff_exp,
                                        hipStreamPerThread))) &&
            omph::kernels::gemv_moe(L.ffn_down_exps.t->type, L.ffn_down_exps.experts, ids,
                                    ffn16_, down_out, down_rows, h_.n_ff_exp, slots, 1,
                                    hipStreamPerThread)) {
            // the routed rows stay in moe_down_: the shared expert goes to moe_xo_
            return moe_shared(L, T, static_cast<float *>(moe_xo_)) &&
                   moe_finish(L, T, down_out, static_cast<float *>(moe_xo_));
        }
        (void) hipGetLastError();
    }
    // the per-expert fallback accumulates the experts' weighted down outputs
    // into cur_ (the device-side shape's combine stores them)
    if (hipMemsetAsync(cur_, 0, (size_t) T * ne * 4, gemv_stream_) != hipSuccess) {
        return fail("moe memset failed");
    }
    if (hipStreamSynchronize(hipStreamPerThread) != hipSuccess ||
        hipMemcpy(moe_host_ids_.data(), ids, (size_t) T * topk * sizeof(int),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return fail("moe_route readback failed");
    }

    // The expert dim is the outermost: one expert is a contiguous run of tiles
    // (and of GGUF blocks), so a slice is a pointer plus the entries' rows/k.
    const auto slice = [this](const Mat & base, const int64_t e, const int64_t rows,
                              const int64_t k) -> Mat {
        Mat m = base;
        m.dev = expert_ptr(base, e);
        m.sliced = true;
        if (base.gemv != nullptr) {
            // the entry's rows/k describe the whole tensor: the slice needs its
            // own, kept alive in a member for the call that follows
            slice_gemv_ = *base.gemv;
            slice_gemv_.rows = rows;
            slice_gemv_.k = k;
            m.gemv = &slice_gemv_;
        }
        return m;
    };
    const int64_t gate_rows = (int64_t) L.ffn_gate_exps.t->ne[1];
    const int64_t up_rows = (int64_t) L.ffn_up_exps.t->ne[1];
    const int64_t down_rows = (int64_t) L.ffn_down_exps.t->ne[1];

    for (int64_t t = 0; t < T; ++t) {
        const auto * x16 = static_cast<const __half *>(h16_) + t * ne;
        float * out_t = static_cast<float *>(cur_) + t * ne;
        for (int s = 0; s < topk; ++s) {
            const int e = moe_host_ids_[(size_t) t * topk + s];
            if (e < 0 || e >= h_.n_expert) {
                return fail("moe_route chose an out-of-range expert");
            }
            const char * what = nullptr;
            if (!matmul(slice(L.ffn_gate_exps, e, gate_rows, ne), x16,
                        static_cast<float *>(ffn1_), ffn, ne, 1)) {
                what = "gate projection";
            } else if (!matmul(slice(L.ffn_up_exps, e, up_rows, ne), x16,
                               static_cast<float *>(ffn2_), ffn, ne, 1)) {
                what = "up projection";
            } else if (!omph::kernels::swiglu_f16(static_cast<const float *>(ffn1_),
                                                  static_cast<const float *>(ffn2_), ffn16_, ffn,
                                                  gemv_stream_)) {
                what = "swiglu";
            } else if (!matmul(slice(L.ffn_down_exps, e, down_rows, ffn), ffn16_,
                               static_cast<float *>(moe_down_), ne, ffn, 1)) {
                what = "down projection";
            } else if (!omph::kernels::moe_acc(out_t, static_cast<const float *>(moe_down_),
                                               weights + t * topk, 1, topk, s, ne, gemv_stream_)) {
                what = "accumulate";
            }
            if (what != nullptr) {
                return fail((std::string("moe expert ") + what + " failed").c_str());
            }
        }
    }
    // cur_ holds the routed sum already
    return moe_shared(L, T, static_cast<float *>(moe_down_)) &&
           moe_finish(L, T, nullptr, static_cast<float *>(moe_down_));
}

// The end of a MoE layer (#522): cur_ = the slot-ordered weighted sum of the
// routed rows xo (or cur_ itself when xo is null) + sigmoid(x . gate) * ysh.
bool Runner::moe_finish(const LayerWeights & L, const int64_t T, const float * xo, const float * ysh) {
    if (!omph::kernels::moe_finish(static_cast<float *>(cur_), xo, static_cast<const float *>(moe_w_),
                                   (int) h_.n_expert_used, ysh, L.ffn_gate_inp_shexp, h16_, T, h_.n_embd,
                                   hipStreamPerThread)) {
        return fail("moe finish failed");
    }
    return true;
}

// The shared expert of a MoE layer (#493): SwiGLU 2048 -> 512 -> 2048 into y;
// moe_finish gates it with sigmoid(x . ffn_gate_inp_shexp) and adds it. One
// pass for the whole chunk: the shared expert does not depend on the routing.
bool Runner::moe_shared(const LayerWeights & L, const int64_t T, float * y) {
    const int64_t ne = h_.n_embd;
    const int64_t shexp = h_.n_ff_shexp;
    // one token of a GGUF-block gate / up pair: one launch with the SwiGLU (#522)
    const Mat & g = L.ffn_gate_shexp;
    const Mat & u = L.ffn_up_shexp;
    const bool fits = T == 1 && use_gemv_ && g.gemv != nullptr && u.gemv != nullptr &&
                      g.gemv->type == u.gemv->type && g.gemv->rows == shexp && u.gemv->rows == shexp &&
                      g.gemv->k == ne && u.gemv->k == ne;
    // ... or of a tile type: a one-slot gemv_moe_gu over its {gate, up} pair
    const bool gu = fits && (g.gemv->gguf ? omph::kernels::gemv_legacy_gu(g.gemv->type, g.dev, u.dev, h16_, ffn16_,
                                                                          shexp, ne, gemv_stream_)
                             : L.shexp_gu != nullptr &&
                                   omph::kernels::gemv_moe_gu(g.gemv->type, L.shexp_gu, L.shexp_gu + 1, moe_zero_id_,
                                                              h16_, ffn16_, shexp, ne, 1, 1, hipStreamPerThread));
    if ((!gu && (!matmul(g, h16_, static_cast<float *>(ffn1_), shexp, ne, T) ||
                 !matmul(u, h16_, static_cast<float *>(ffn2_), shexp, ne, T) ||
                 !omph::kernels::swiglu_f16(static_cast<const float *>(ffn1_),
                                            static_cast<const float *>(ffn2_), ffn16_, T * shexp,
                                            hipStreamPerThread))) ||
        !matmul(L.ffn_down_shexp, ffn16_, y, ne, shexp, T)) {
        return fail("moe shared expert failed");
    }
    return true;
}

bool Runner::forward(const std::vector<int32_t> & toks, std::vector<float> & logits,
             const std::string & trace_dir, const int64_t start_pos,
             const bool want_logits, int32_t * greedy, const ForwardInputs * in) {
    const int64_t T = (int64_t) toks.size();
    if (!verifying_) {
        observe_draft(toks.data(), T);  // a prompt or a plain step: real tokens (#217)
    }
    // the inputs stay visible to the layers (M-RoPE) and the MTP fill until
    // this forward returns
    struct Scope {
        const ForwardInputs *& slot;
        ~Scope() { slot = nullptr; }
    } scope{inputs_};
    if (in != nullptr) {
        if (!in->mpos.empty() && (int64_t) in->mpos.size() != 3 * T) {
            return fail("forward: M-RoPE positions do not match the tokens");
        }
        for (size_t i = 0; i < in->rows.size(); ++i) {
            if (in->rows[i] < 0 || in->rows[i] >= T || (i > 0 && in->rows[i] <= in->rows[i - 1]) ||
                in->embd == nullptr) {
                return fail("forward: bad embedding rows");
            }
        }
        if (!in->mpos.empty()) {
            if (mpos_dev_ == nullptr) mpos_dev_ = mem_.device((size_t) max_tokens_ * 3 * 4);
            if (T > max_tokens_ || hipMemcpy(mpos_dev_, in->mpos.data(), (size_t) T * 3 * 4,
                                             hipMemcpyHostToDevice) != hipSuccess) {
                return fail("forward: M-RoPE position upload failed");
            }
        }
        inputs_ = in;
    }
    if (greedy != nullptr) {
        *greedy = -1;
    }
    // The activations hold max_tokens_ rows and the KV cache max_seq_
    // positions: anything past either is an out-of-bounds write.
    if (T <= 0 || T > max_tokens_ || start_pos < 0 || start_pos + T > max_seq_) {
        return fail("forward: tokens exceed the activation or KV capacity");
    }
    // Every id must index the embedding table: a negative id is an image
    // placeholder (those rows come from `in`), and the n-gram drafts copy ids
    // out of the sequence (#334). The embedding lookup itself has no bounds
    // check.
    for (int64_t t = 0, next_row = 0; t < T; ++t) {
        if (in != nullptr && next_row < (int64_t) in->rows.size() && in->rows[(size_t) next_row] == t) {
            ++next_row;
            continue;
        }
        if (toks[(size_t) t] < 0 || toks[(size_t) t] >= h_.n_vocab) {
            return fail("forward: token id outside the vocabulary");
        }
    }
    watch_step_begin(T);
    hipEvent_t step_a{}, step_b{};
    const bool time_step = T == 1 && env_.timing;
    if (time_step) {
        (void) hipEventCreate(&step_a);
        (void) hipEventCreate(&step_b);
        (void) hipEventRecord(step_a, nullptr);
    }
    const int64_t ne = h_.n_embd;
    if (!moe_cache_begin()) {
        return false;
    }

    if (!embed(toks.data(), T, in, static_cast<float *>(x_))) {
        return false;
    }

    // Set when the previous layer's final residual add already wrote this
    // layer's attn_norm(x) into h16_ (one fused launch instead of two).
    bool h16_normed = false;
    for (int64_t il = 0; il < h_.n_layer; ++il) {
        const LayerWeights & L = layers_[(size_t) il];
        const bool skip_attn = !L.recurrent && env_.skip_attn;
        const bool skip_ffn = env_.skip_ffn;
        const float * attn_norm = L.attn_norm;
        const float * post_norm = L.post_norm;
        if (!h16_normed &&
            !omph::kernels::add_rms_norm_f16(static_cast<const float *>(x_), nullptr, nullptr,
                                             attn_norm, h16_, T, ne, (float) h_.eps,
                                             nullptr)) {
            return fail("attn_norm failed");
        }
        h16_normed = false;
        if (env_.skip_blocks) {
            continue;  // ablation only: the layer output is the normed input
        }
        timer_block_.start();
        if (L.recurrent) {
            if (!gdn_layer(il, L, T)) return false;
        } else if (!skip_attn) {
            if (!attn_layer(il, L, T, start_pos)) return false;
        }
        timer_block_.stop(t_block_);
        if (skip_ffn) {
            continue;
        }
        // x = ffn(rms_norm(block + x)) + (block + x)
        if (!omph::kernels::add_rms_norm_f16(static_cast<const float *>(blk_),
                                             static_cast<const float *>(x_),
                                             static_cast<float *>(resid_), post_norm, h16_,
                                             T, ne, (float) h_.eps, nullptr, h16lo_)) {
            return fail("residual/norm failed");
        }
        if (h_.moe()) {
            // the second model's FFN (#493): chosen once, at load time
            if (!moe_ffn(L, T)) {
                return fail("moe ffn failed");
            }
        } else {
            // a prefill chunk: gate, then up with SwiGLU in its GEMM's epilogue (#221)
            const bool fused = up_swiglu_ok(L.ffn_up, h_.n_ff, ne, T);
            const bool act_ok =
                fused ? matmul(L.ffn_gate, h16_, static_cast<float *>(ffn1_), h_.n_ff, ne, T) &&
                            up_swiglu(L.ffn_up, h16_, static_cast<const float *>(ffn1_), ffn16_, h_.n_ff, ne, T)
                      : (grouped(h16_, ne, T, {{&L.ffn_up, ffn2_, h_.n_ff}, {&L.ffn_gate, ffn1_, h_.n_ff}}) ||
                         fork_join(
                             T,
                             [&] { return matmul(L.ffn_up, h16_, static_cast<float *>(ffn2_), h_.n_ff, ne, T); },
                             [&] {
                                 return matmul(L.ffn_gate, h16_, static_cast<float *>(ffn1_), h_.n_ff, ne, T);
                             })) &&
                            omph::kernels::swiglu_f16(static_cast<const float *>(ffn1_),
                                                      static_cast<const float *>(ffn2_), ffn16_, T * h_.n_ff, nullptr);
            if (!act_ok || !matmul(L.ffn_down, ffn16_, static_cast<float *>(cur_), ne, h_.n_ff, T)) {
                return fail("ffn failed");
            }
        }
        // x = ffn + resid; for all but the last layer, the same launch also
        // writes the next layer's attn_norm(x) into h16_.
        const float * next_norm =
            il + 1 < h_.n_layer ? layers_[(size_t) il + 1].attn_norm : nullptr;
        if (next_norm != nullptr) {
            if (!omph::kernels::add_rms_norm_f16(static_cast<const float *>(cur_),
                                                 static_cast<const float *>(resid_),
                                                 static_cast<float *>(x_), next_norm, h16_, T,
                                                 ne, (float) h_.eps, nullptr)) {
                return fail("residual/norm failed");
            }
            h16_normed = true;
        } else if (!omph::kernels::add_out(static_cast<const float *>(cur_),
                                           static_cast<const float *>(resid_),
                                           static_cast<float *>(x_), T * ne, nullptr)) {
            return fail("residual failed");
        }
        // the drafter's features (#245): this layer's output, when it names it
        for (size_t f = 0; f < dfl_capture_.size(); ++f) {
            if (dfl_capture_[f] == il &&
                !omph::kernels::dflash_capture(static_cast<const float *>(x_), T, ne, ne,
                                               static_cast<__half *>(dfl_feat_), dfl_fc_.gemv->k,
                                               (int64_t) f * ne, nullptr)) {
                return fail("drafter feature capture failed");
            }
        }
        if (!trace_dir.empty()) {
            std::vector<float> host((size_t) T * ne);
            (void) hipMemcpy(host.data(), x_, host.size() * 4, hipMemcpyDeviceToHost);
            write_f32(trace_dir + "/l_out-" + std::to_string(il) + ".f32", host);
        }
    }
    if (!moe_cache_log(T)) {
        return false;
    }

    // MTP (#124): the h_p the block pairs with the next tokens, every row.
    if (mtp_) {
        if (!omph::kernels::rms_norm(static_cast<const float *>(x_), out_norm_,
                                     static_cast<float *>(mtp_hlast_), T, ne, (float) h_.eps,
                                     1.0f, nullptr)) {
            return fail("mtp h failed");
        }
    }
    // The tokens this forward carried: commit() reads them to update the
    // draft-vocabulary switch. MTP and DFlash2 both need it (#340: the DFlash
    // path never set last_toks_, so its switch saw the prompt only).
    if (mtp_ || dflash_on()) {
        last_toks_ = toks;
        last_pos0_ = start_pos;
    }
    if (want_logits) {
        if (!lm_head(T, logits, greedy)) {
            return false;
        }
        if (env_.check_finite && !check_logits_finite(T, logits)) {
            return false;
        }
    } else {
        logits.clear();
        if (hipDeviceSynchronize() != hipSuccess) {
            return fail("forward failed");
        }
    }
    // A verification's tokens are filled at commit(), only those kept.
    if (mtp_ && !verifying_ && !mtp_fill(T)) {
        return false;
    }
    if (!verifying_ && !dflash_inject(start_pos, T)) {
        return false;
    }
    report_phases();
    watch_step_end();
    if (time_step) {
        step_event(step_a, step_b);
    }
    return true;
}

// Final norm and lm_head of the T rows in h16_, into `logits` on the host
// (or, with `greedy` on a single-token GEMV step, only the argmax). Returns
// once the results are on the host.
bool Runner::lm_head(const int64_t T, std::vector<float> & logits, int32_t * greedy) {
    const int64_t ne = h_.n_embd;
    // The fast paths (the device argmax of a decode step, four verification rows
    // per weight read) are written for a Q4_K head, GGUF type 12; any other type
    // goes through head_chunked, which dequantizes 32768 rows per vocabulary
    // chunk on every call. Say it once instead of looking like a mystery (#315).
    if (head_.gemv != nullptr && head_.gemv->type != 12 && !head_type_warned_) {
        head_type_warned_ = true;
        std::fprintf(stderr,
                     "lm_head: type %u takes the chunked f16 path (the fast paths are for Q4_K)\n",
                     head_.gemv->type);
    }
    if (!omph::kernels::rms_norm(static_cast<const float *>(x_), out_norm_,
                                 static_cast<float *>(cur_), T, ne, (float) h_.eps, 1.0f,
                                 nullptr) ||
        !omph::kernels::cast_f32_to_f16(static_cast<const float *>(cur_), h16_, T * ne,
                                        nullptr)) {
        return fail("output_norm failed");
    }
    // Decode: the fused GEMV, and on the greedy path only the argmax comes back.
    if (use_gemv_ && T == 1 && head_.gemv != nullptr) {
        if (!matmul(head_, h16_, static_cast<float *>(logits_), h_.n_vocab, ne, T)) {
            return fail("lm_head (gemv) failed");
        }
        if (greedy != nullptr) {
            // 8 bytes back instead of 1 MB of logits and a host argmax (#102)
            unsigned long long key = 0;
            if (!omph::kernels::argmax_f32(static_cast<const float *>(logits_), h_.n_vocab,
                                           static_cast<unsigned long long *>(argmax_key_),
                                           nullptr) ||
                hipMemcpy(&key, argmax_key_, sizeof(key), hipMemcpyDeviceToHost) !=
                    hipSuccess) {
                return fail("argmax failed");
            }
            *greedy = omph::kernels::argmax_key_index(key);
            return true;
        }
        logits.resize((size_t) h_.n_vocab);
        return copy_logits(logits.data(), 1);
    }
    // Only the last token's logits: at 32k the full (T, 248320) f32 buffer is
    // 33 GB, and the long-context KV validation only needs the final row.
    if (last_logits_only_ && !verifying_) {
        const uint8_t * xlast = static_cast<const uint8_t *>(h16_) + (T - 1) * ne * 2;
        logits.resize((size_t) h_.n_vocab);
        if (use_gemv_ && head_.gemv != nullptr) {
            if (!gemv_one(head_.gemv->type, head_.dev, xlast, static_cast<float *>(logits_),
                          h_.n_vocab, ne, nullptr)) {
                return fail("lm_head gemv failed");
            }
            return copy_logits(logits.data(), 1);
        }
        return head_chunked(xlast, 1, logits.data());
    }
    logits.resize((size_t) T * h_.n_vocab);
    // Small-batch head: for a Q4_K head with 2+ tokens, run four tokens per
    // weight read instead of materializing f16 (PLAN.md §10.1's N = 1..5),
    // kHeadRows (a multiple of 4) rows at a time into logits_, then out.
    // A verification (#122, #126): every row's greedy token on the device, 8
    // bytes per row back instead of 1 MB of logits.
    if (verifying_ && verify_argmax_ != nullptr && use_gemv_ && T <= kVerifyRowsMax &&
        head_.gemv != nullptr && has_gemv_type(head_.gemv->type)) {
        const uint32_t ht = head_.gemv->type;
        // 5..16 tokens: one read of the 682 MB head (#244); fewer: four at a time.
        // Another tile type (the second model's Q6_K head, #496): its 1..16-token
        // tile GEMV, else one launch per row - either way each row is the
        // decode's GEMV result, never the chunked f16 path (~10 ms a verification)
        bool one = (ht != 12 || T > 4) && omph::kernels::gemv_tokens(ht, head_.dev, h16_,
                                                                     static_cast<float *>(logits_),
                                                                     h_.n_vocab, ne, (int) T, nullptr);
        for (int64_t t = 0; t < T && !one && ht != 12; ++t) {
            if (!gemv_one(ht, head_.dev, static_cast<const uint8_t *>(h16_) + t * ne * 2,
                          static_cast<float *>(logits_) + t * h_.n_vocab, h_.n_vocab, ne, nullptr)) {
                return fail("lm_head (verify) failed");
            }
        }
        one = one || ht != 12;
        for (int64_t t0 = 0; t0 < T && !one;) {
            const int64_t n = std::min<int64_t>(4, T - t0);
            const uint8_t * x = static_cast<const uint8_t *>(h16_) + t0 * ne * 2;
            float * y = static_cast<float *>(logits_) + t0 * h_.n_vocab;
            const bool ok = n > 1 ? omph::kernels::gemv_multi(12, head_.dev, x, y, h_.n_vocab, ne,
                                                              (int) n, nullptr)
                                  : omph::kernels::gemv_q4k(head_.dev, x, y, h_.n_vocab, ne,
                                                            nullptr);
            if (!ok) {
                return fail("lm_head (verify) failed");
            }
            t0 += n;
        }
        auto * keys = static_cast<unsigned long long *>(spec_keys_);
        for (int64_t t = 0; t < T; ++t) {
            if (!omph::kernels::argmax_f32(static_cast<const float *>(logits_) + t * h_.n_vocab,
                                           h_.n_vocab, keys + t, nullptr)) {
                return fail("verify argmax failed");
            }
        }
        std::vector<unsigned long long> host((size_t) T);
        if (hipMemcpy(host.data(), keys, (size_t) T * 8, hipMemcpyDeviceToHost) != hipSuccess) {
            return fail("verify argmax copy failed");
        }
        verify_argmax_->resize((size_t) T);
        for (int64_t t = 0; t < T; ++t) {
            (*verify_argmax_)[(size_t) t] = omph::kernels::argmax_key_index(host[(size_t) t]);
        }
        logits.clear();
        return true;
    }
    if (gemv_path(T) && head_.gemv != nullptr && head_.gemv->type == 12) {
        for (int64_t r0 = 0; r0 < T; r0 += kHeadRows) {
            const int64_t rows = std::min<int64_t>(kHeadRows, T - r0);
            for (int64_t t0 = 0; t0 < rows;) {
                const int64_t n = std::min<int64_t>(4, rows - t0);
                const uint8_t * x = static_cast<const uint8_t *>(h16_) + (r0 + t0) * ne * 2;
                float * y = static_cast<float *>(logits_) + t0 * h_.n_vocab;
                // up to four tokens per read of the 682 MB head (#126)
                const bool ok =
                    n > 1 ? omph::kernels::gemv_multi(12, head_.dev, x, y, h_.n_vocab, ne, (int) n,
                                                      nullptr)
                          : omph::kernels::gemv_q4k(head_.dev, x, y, h_.n_vocab, ne, nullptr);
                t0 += n;
                if (!ok) {
                    return fail("lm_head batch4 failed");
                }
            }
            if (!copy_logits(logits.data() + r0 * h_.n_vocab, rows)) {
                return false;
            }
        }
        return true;
    }
    return head_chunked(h16_, T, logits.data());
}

bool Runner::check_logits_finite(const int64_t T, const std::vector<float> & logits) {
    const int64_t rows = !logits.empty() ? (int64_t) (logits.size() / h_.n_vocab)
                                         : (verifying_ ? T : 1);
    std::vector<float> staged;
    const float * host = logits.data();
    if (logits.empty()) {
        // the greedy argmax and the verification paths never bring the rows
        // back: 1 MB per row, and only under the switch
        staged.resize((size_t) rows * h_.n_vocab);
        if (!copy_logits(staged.data(), rows)) {
            return false;
        }
        host = staged.data();
    }
    for (int64_t r = 0; r < rows; ++r) {
        const float * row = host + r * h_.n_vocab;
        for (int64_t j = 0; j < h_.n_vocab; ++j) {
            if (!std::isfinite(row[j])) {
                std::fprintf(stderr,
                             "OMPH_CHECK_FINITE: logits row %lld of %lld is not finite at token "
                             "id %lld\n",
                             (long long) r, (long long) rows, (long long) j);
                return fail("logits are not finite");
            }
        }
    }
    return true;
}

// The first `rows` rows of logits_ to the host (synchronous).
bool Runner::copy_logits(float * dst, const int64_t rows) {
    if (hipMemcpy(dst, logits_, (size_t) rows * h_.n_vocab * 4, hipMemcpyDeviceToHost) !=
        hipSuccess) {
        return fail("lm_head failed");
    }
    return true;
}

// lm head in vocab chunks, through the f16 dequant + GEMM path: the
// whole f16 head (248k x 5120, 2.5 GB) would not fit. Writes (T, n_vocab)
// to `out` on the host, one vocab chunk at a time.
bool Runner::head_chunked(const void * x16, const int64_t T, float * out) {
    const int64_t ne = h_.n_embd;
    const omph::gguf::TensorInfo * head = head_.t;
    const int64_t head_row_bytes = (int64_t) (head->nbytes / head->ne[1]);
    void * head16 = lazy(&head16_, head16_bytes_);
    void * tmp_logits = lazy(&tmp_logits_, tmp_logits_bytes_);
    const auto * head_src = static_cast<const uint8_t *>(head_.dev);
    const int64_t chunk = std::min<int64_t>(h_.n_vocab, 32768);
    for (int64_t v0 = 0; v0 < h_.n_vocab; v0 += chunk) {
        const int64_t rows = std::min(chunk, h_.n_vocab - v0);
        void * wh = head16;
        timer_stage_.start();
        // the repacked head's rows v0.. straight from its layout (M8)
        // a GemvEntry's bytes are a repacked layout *unless* it is one of the
        // legacy types, which keep the GGUF blocks (#492)
        const bool head_dq =
            head_.gemv != nullptr && !head_.gemv->gguf
                ? omph::kernels::dequant_repacked(head_.gemv->type, head_src, wh, h_.n_vocab, ne,
                                                  nullptr, v0, rows)
                : omph::kernels::dequantize(head->type, head_src + v0 * head_row_bytes, wh,
                                            rows * ne, true, nullptr);
        timer_stage_.stop(t_stage_);
        timer_gemm_.start();
        const bool head_gm =
            linear_.run(wh, x16, static_cast<float *>(tmp_logits), rows, ne, T);
        timer_gemm_.stop(t_gemm_);
        if (!head_dq || !head_gm ||
            hipMemcpy2D(out + v0, (size_t) h_.n_vocab * 4, tmp_logits, (size_t) rows * 4,
                        (size_t) rows * 4, (size_t) T, hipMemcpyDeviceToHost) != hipSuccess) {
            return fail("lm_head failed");
        }
    }
    return true;
}

} // namespace omph::model
