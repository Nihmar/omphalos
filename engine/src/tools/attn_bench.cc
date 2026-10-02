// omph-attn-bench — the decode / verification attention alone, on a synthetic
// Q8/Q4 cache with the FP16 ring (#169): no model, no prefill, seconds.
//
// usage: omph-attn-bench [--seq N[,N...]] [--tokens T[,T...]] [--iters N] [--ctx N] [--chunk K] [--k4]
//   For each sequence length and token count: the median kernel time over
//   --iters calls (attention + merge, one layer, hipEvent timing) of the WMMA
//   decode kernel and of the scalar kernel, and the KV bytes read per second.
//   Then the identity check speculation relies on: every query of a T-token
//   call must equal, bit for bit, the same query run alone (T = 1) with the
//   cache ending at its position (calls of up to 8 tokens; longer ones take the
//   prefill kernel, timed only). --k4: K in V's Q4 format (OMPH_KV_K4).
#include "kernels/attn.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int64_t kNh = 24;
constexpr int64_t kNkv = 4;
constexpr int64_t kHd = 256;
constexpr int64_t kWindow = 128;

std::vector<int64_t> list(const char * s) {
    std::vector<int64_t> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) v.push_back(std::atoll(item.c_str()));
    return v;
}

template <class T>
T * upload(const std::vector<T> & h) {
    T * d = nullptr;
    if (hipMalloc(&d, h.size() * sizeof(T)) != hipSuccess ||
        hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess) {
        std::fprintf(stderr, "allocation failed\n");
        std::exit(1);
    }
    return d;
}

} // namespace

int main(int argc, char ** argv) {
    std::vector<int64_t> seqs = {4096, 16384, 32768, 65536, 100000};
    std::vector<int64_t> toks = {1, 4};
    int iters = 20;
    int64_t ctx = 0;
    int64_t chunk_arg = 0;
    bool k4 = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--seq") && i + 1 < argc) seqs = list(argv[++i]);
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) toks = list(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--chunk") && i + 1 < argc) chunk_arg = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--k4")) k4 = true;
        else {
            std::fprintf(stderr, "usage: %s [--seq N,...] [--tokens T,...] [--iters N] [--ctx N] [--chunk K] [--k4]\n",
                         argv[0]);
            return 2;
        }
    }
    const int64_t max_seq = ctx > 0 ? ctx : *std::max_element(seqs.begin(), seqs.end());
    const int64_t max_t = *std::max_element(toks.begin(), toks.end());
    const int64_t ring = kWindow + omph::kernels::kKvRingExtra;
    const int64_t nblk = kHd / 32;
    // random cache contents: K Q8 values with f16 scales, V Q4 nibbles, the ring in f16
    std::mt19937 rng(169);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    // K: Q8 values, or Q4 nibbles in V's format (--k4)
    std::vector<uint8_t> kq((size_t) max_seq * kNkv * (k4 ? kHd / 2 : kHd)), vq((size_t) max_seq * kNkv * kHd / 2);
    for (auto & b : kq) b = k4 ? (uint8_t) byte(rng) : (uint8_t) (int8_t) (byte(rng) % 255 - 127);
    for (auto & b : vq) b = (uint8_t) byte(rng);
    std::vector<__half> ksc((size_t) max_seq * kNkv * nblk), vsc(ksc.size());
    for (auto & x : ksc) x = __float2half(0.02f + 0.01f * std::fabs(uni(rng)));
    for (auto & x : vsc) x = __float2half(0.1f + 0.05f * std::fabs(uni(rng)));
    std::vector<__half> k16((size_t) ring * kNkv * kHd), v16(k16.size());
    for (auto & x : k16) x = __float2half(uni(rng) * 2.0f);
    for (auto & x : v16) x = __float2half(uni(rng));
    std::vector<float> qh((size_t) max_t * kNh * kHd), gh(qh.size());
    for (auto & x : qh) x = uni(rng);
    for (auto & x : gh) x = uni(rng);

    omph::kernels::KvCache kv;
    kv.k_q8 = upload(kq);
    kv.k_scales = upload(ksc);
    kv.v_q4 = upload(vq);
    kv.v_scales = upload(vsc);
    kv.k16 = upload(k16);
    kv.v16 = upload(v16);
    kv.window = kWindow;
    kv.k_q4 = k4;
    kv.ring = ring;
    const float * q = upload(qh);
    const float * gate = upload(gh);
    const int64_t chunk = chunk_arg > 0 ? chunk_arg : omph::kernels::attention_key_chunk(max_seq);
    const size_t work_bytes = omph::kernels::attention_gqa_work_bytes(max_t, kNh, kNkv, kHd, max_seq, chunk);
    void * work = nullptr;
    float * out = nullptr;
    if (hipMalloc(&work, work_bytes) != hipSuccess || hipMalloc(&out, (size_t) max_t * kNh * kHd * 4) != hipSuccess) {
        std::fprintf(stderr, "allocation failed\n");
        return 1;
    }
    const float scale = 1.0f / std::sqrt((float) kHd);
    const auto call = [&](int64_t t, int64_t seq, bool wmma, float * dst) {
        return omph::kernels::attention_gqa(q, kv, gate, dst, t, seq, kNh, kNkv, kHd, scale, true, work, work_bytes,
                                            nullptr, nullptr, true, chunk, wmma);
    };
    hipEvent_t e0, e1;
    (void) hipEventCreate(&e0);
    (void) hipEventCreate(&e1);
    // ~1 s of calls first: the GPU clock ramps up under load (#141)
    {
        hipEvent_t w0, w1;
        (void) hipEventCreate(&w0);
        (void) hipEventCreate(&w1);
        float spent = 0.0f;
        while (spent < 1000.0f) {
            (void) hipEventRecord(w0, nullptr);
            for (int i = 0; i < 20; ++i) (void) call(1, max_seq, true, out);
            (void) hipEventRecord(w1, nullptr);
            (void) hipEventSynchronize(w1);
            float ms = 0.0f;
            (void) hipEventElapsedTime(&ms, w0, w1);
            spent += ms;
        }
    }
    std::printf("key chunk %lld (capacity %lld); K%d/V4, KV read per call and layer: %d B per position\n",
                (long long) chunk, (long long) max_seq, k4 ? 4 : 8, k4 ? 1152 : 1664);
    std::printf("%8s %3s %12s %10s %12s %10s %8s\n", "seq", "T", "wmma us", "GB/s", "scalar us", "GB/s", "ratio");
    for (const int64_t seq : seqs) {
        for (const int64_t t : toks) {
            double us[2];
            for (int m = 0; m < 2; ++m) {
                std::vector<float> times;
                for (int it = 0; it < iters + 3; ++it) {
                    (void) hipEventRecord(e0, nullptr);
                    if (!call(t, seq, m == 0, out)) {
                        std::fprintf(stderr, "attention_gqa failed (seq %lld, T %lld)\n", (long long) seq,
                                     (long long) t);
                        return 1;
                    }
                    (void) hipEventRecord(e1, nullptr);
                    (void) hipEventSynchronize(e1);
                    float ms = 0.0f;
                    (void) hipEventElapsedTime(&ms, e0, e1);
                    if (it >= 3) times.push_back(ms * 1000.0f);  // after a warm-up
                }
                std::sort(times.begin(), times.end());
                us[m] = times[times.size() / 2];
            }
            const double bytes = (double) seq * kNkv * ((k4 ? kHd / 2 : kHd) + 2 * nblk + kHd / 2 + 2 * nblk);
            std::printf("%8lld %3lld %12.1f %10.1f %12.1f %10.1f %8.2f\n", (long long) seq, (long long) t, us[0],
                        bytes / us[0] / 1e3, us[1], bytes / us[1] / 1e3, us[1] / us[0]);
        }
    }
    // identity: each row of a T-token call vs that query alone (WMMA path)
    int bad = 0;
    int checked = 0;
    std::vector<float> batch((size_t) max_t * kNh * kHd), single((size_t) kNh * kHd);
    float * one = nullptr;
    (void) hipMalloc(&one, (size_t) kNh * kHd * 4);
    for (const int64_t seq : seqs) {
        for (const int64_t t : toks) {
            // the identity is the decode / verification kernel's (#161): steps of up to 8 tokens
            if (t < 2 || t > 8 || !call(t, seq, true, out)) continue;
            (void) hipMemcpy(batch.data(), out, (size_t) t * kNh * kHd * 4, hipMemcpyDeviceToHost);
            for (int64_t r = 0; r < t; ++r) {
                // the query of row r alone: q row r is the first row of q + r * kNh * kHd
                if (!omph::kernels::attention_gqa(q + r * kNh * kHd, kv, gate + r * kNh * kHd, one, 1,
                                                  seq - t + r + 1, kNh, kNkv, kHd, scale, true, work, work_bytes,
                                                  nullptr, nullptr, true, chunk, true)) {
                    return 1;
                }
                (void) hipMemcpy(single.data(), one, single.size() * 4, hipMemcpyDeviceToHost);
                ++checked;
                if (std::memcmp(single.data(), batch.data() + r * kNh * kHd, single.size() * 4) != 0) ++bad;
            }
        }
    }
    std::printf("identity: %d of %d rows of multi-token calls differ from the single-token call\n", bad, checked);
    return bad == 0 ? 0 : 1;
}
