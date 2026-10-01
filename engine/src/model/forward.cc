// Runner: the forward pass (embedding, layer loop, final norm, lm_head).
#include "model/runner.hh"

#include "format/repack.hh"
#include "kernels/attn.hh"
#include "kernels/dequant.hh"
#include "kernels/elementwise.hh"
#include "kernels/gdn.hh"
#include "kernels/gemv.hh"

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
bool Runner::forward(const std::vector<int32_t> & toks, std::vector<float> & logits,
             const std::string & trace_dir, const int64_t start_pos,
             const bool want_logits, int32_t * greedy) {
    const int64_t T = (int64_t) toks.size();
    if (greedy != nullptr) {
        *greedy = -1;
    }
    // The activations hold max_tokens_ rows and the KV cache max_seq_
    // positions: anything past either is an out-of-bounds write.
    if (T <= 0 || T > max_tokens_ || start_pos < 0 || start_pos + T > max_seq_) {
        return fail("forward: tokens exceed the activation or KV capacity");
    }
    hipEvent_t step_a{}, step_b{};
    const bool time_step = T == 1 && env_.timing;
    if (time_step) {
        (void) hipEventCreate(&step_a);
        (void) hipEventCreate(&step_b);
        (void) hipEventRecord(step_a, nullptr);
    }
    const int64_t ne = h_.n_embd;

    // token embeddings, one row at a time (the table is quantized and lives
    // in pinned host memory: the kernel reads the row over PCIe)
    for (int64_t t = 0; t < T; ++t) {
        const uint8_t * src =
            static_cast<const uint8_t *>(embd_host_) + (size_t) toks[t] * embd_row_bytes_;
        if (!omph::kernels::dequantize(embd_type_, src,
                                       static_cast<uint8_t *>(x_) + t * ne * 4, ne, false,
                                       nullptr)) {
            return fail("embedding dequant failed");
        }
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
        scratch_.reset();
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
                                             T, ne, (float) h_.eps, nullptr)) {
            return fail("residual/norm failed");
        }
        const bool gate_up_ok = fork_join(
            T,
            [&] {
                return matmul(L.ffn_up, h16_, static_cast<float *>(ffn2_), h_.n_ff, ne, T);
            },
            [&] {
                return matmul(L.ffn_gate, h16_, static_cast<float *>(ffn1_), h_.n_ff, ne, T);
            });
        if (!gate_up_ok ||
            !omph::kernels::swiglu_f16(static_cast<const float *>(ffn1_),
                                       static_cast<const float *>(ffn2_), ffn16_, T * h_.n_ff,
                                       nullptr) ||
            !matmul(L.ffn_down, ffn16_, static_cast<float *>(cur_), ne, h_.n_ff, T)) {
            return fail("ffn failed");
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
        if (!trace_dir.empty()) {
            std::vector<float> host((size_t) T * ne);
            (void) hipMemcpy(host.data(), x_, host.size() * 4, hipMemcpyDeviceToHost);
            write_f32(trace_dir + "/l_out-" + std::to_string(il) + ".f32", host);
        }
    }

    // MTP (#124): the h_p the block pairs with the next tokens, every row.
    if (mtp_) {
        if (!omph::kernels::rms_norm(static_cast<const float *>(x_), out_norm_,
                                     static_cast<float *>(mtp_hlast_), T, ne, (float) h_.eps,
                                     1.0f, nullptr)) {
            return fail("mtp h failed");
        }
        last_toks_ = toks;
        last_pos0_ = start_pos;
    }
    if (want_logits) {
        if (!lm_head(T, logits, greedy)) {
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
    report_phases();
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
        head_.gemv != nullptr && head_.gemv->type == 12) {
        for (int64_t t0 = 0; t0 < T;) {
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
    if (use_gemv_ && T > 1 && head_.gemv != nullptr && head_.gemv->type == 12) {
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

// The first `rows` rows of logits_ to the host (synchronous).
bool Runner::copy_logits(float * dst, const int64_t rows) {
    if (hipMemcpy(dst, logits_, (size_t) rows * h_.n_vocab * 4, hipMemcpyDeviceToHost) !=
        hipSuccess) {
        return fail("lm_head failed");
    }
    return true;
}

// lm head in vocab chunks, through the f16 dequant + hipBLASLt path: the
// whole f16 head (248k x 5120, 2.5 GB) would not fit. Writes (T, n_vocab)
// to `out` on the host, one vocab chunk at a time.
bool Runner::head_chunked(const void * x16, const int64_t T, float * out) {
    const int64_t ne = h_.n_embd;
    const omph::gguf::TensorInfo * head = head_.t;
    const int64_t head_row_bytes = (int64_t) (head->nbytes / head->ne[1]);
    if (head_.gemv != nullptr &&
        hipMemcpy(lazy(&raw_stage_, raw_stage_bytes()), file_.tensor_data(*head),
                  (size_t) head->nbytes, hipMemcpyHostToDevice) != hipSuccess) {
        return fail("cannot stage the lm_head");
    }
    void * head16 = lazy(&head16_, head16_bytes_);
    void * tmp_logits = lazy(&tmp_logits_, tmp_logits_bytes_);
    const uint8_t * head_src = head_.gemv != nullptr ? static_cast<const uint8_t *>(raw_stage_)
                                                     : static_cast<const uint8_t *>(head_.dev);
    const int64_t chunk = std::min<int64_t>(h_.n_vocab, 32768);
    for (int64_t v0 = 0; v0 < h_.n_vocab; v0 += chunk) {
        const int64_t rows = std::min(chunk, h_.n_vocab - v0);
        void * wh = head16;
        timer_stage_.start();
        const bool head_dq = omph::kernels::dequantize(head->type,
                                                       head_src + v0 * head_row_bytes, wh,
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
