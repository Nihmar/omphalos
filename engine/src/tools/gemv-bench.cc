// omph-gemv-bench — effective bandwidth of the fused quantized GEMVs against
// the f16 dequant path, on real tensors of the model (PLAN.md §10.1, §17).
//
// usage: omph-gemv-bench <model.gguf> [tensor] [--iters N] [--iq4-all]
#include "format/gguf.hh"
#include "format/repack.hh"
#include "kernels/dequant.hh"
#include "kernels/elementwise.hh"
#include "kernels/gemv.hh"
#include "runtime/matmul.hh"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int64_t kCanary = 1024;  // floats of sentinel after the largest y
constexpr double kMeasuredBandwidthGBs = 318.3;  // bench/results/m0-bandwidth.csv

int fail(const char * msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

bool repack_tensor(const uint32_t type, const void * src, const int64_t n_blocks,
                    std::vector<uint8_t> & dst) {
    return omph::format::repack_any(type, src, n_blocks, dst);
}

bool unrepack_tensor(const uint32_t type, const void * src, const int64_t n_blocks,
                     std::vector<uint8_t> & dst) {
    return omph::format::unrepack_any(type, src, n_blocks, dst);
}

struct Case {
    const omph::gguf::TensorInfo * t = nullptr;
    void * dev = nullptr;  // repacked tensor on the device
    int64_t k = 0;
    int64_t rows = 0;
    int64_t bytes = 0;  // bytes the kernel reads
};

bool launch(const Case & c, const void * x, float * y, hipStream_t stream) {
    if (c.t->type == 12) {
        return omph::kernels::gemv_q4k(c.dev, x, y, c.rows, c.k, stream);
    }
    if (c.t->type == 29) {
        return omph::kernels::gemv_iq1_m(c.dev, x, y, c.rows, c.k, stream);
    }
    if (c.t->type == 23) {
        return omph::kernels::gemv_iq4_xs(c.dev, x, y, c.rows, c.k, stream);
    }
    if (c.t->type == 18) {
        return omph::kernels::gemv_iq3_xxs(c.dev, x, y, c.rows, c.k, stream);
    }
    if (c.t->type == 21) {
        return omph::kernels::gemv_iq3_s(c.dev, x, y, c.rows, c.k, stream);
    }
    if (c.t->type == 10) {
        return omph::kernels::gemv_q2k(c.dev, x, y, c.rows, c.k, stream);
    }
    if (c.t->type == 14) {
        return omph::kernels::gemv_q6k(c.dev, x, y, c.rows, c.k, stream);
    }
    if (c.t->type == 17) {
        return omph::kernels::gemv_iq2_xs(c.dev, x, y, c.rows, c.k, stream);
    }
    if (c.t->type == 16) {
        return omph::kernels::gemv_iq2_xxs(c.dev, x, y, c.rows, c.k, stream);
    }
    if (c.t->type == 22) {
        return omph::kernels::gemv_iq2_s(c.dev, x, y, c.rows, c.k, stream);
    }
    return false;
}

// Repacks a tensor on the host, verifies the rebuild is byte-identical, and
// uploads it.
bool prepare(const omph::gguf::File & file, const omph::gguf::TensorInfo * t, Case & c) {
    const int64_t k = (int64_t) t->ne[0];
    const int64_t rows = (int64_t) t->ne[1];
    const int64_t n_blocks = rows * (k / 256);
    std::vector<uint8_t> host;
    std::vector<uint8_t> rebuilt((size_t) t->nbytes);
    if (!repack_tensor(t->type, file.tensor_data(*t), n_blocks, host) ||
        !unrepack_tensor(t->type, host.data(), n_blocks, rebuilt)) {
        return false;
    }
    if (std::memcmp(rebuilt.data(), file.tensor_data(*t), (size_t) t->nbytes) != 0) {
        std::fprintf(stderr, "repack of %s is not lossless\n", t->name.c_str());
        return false;
    }
    if (hipMalloc(&c.dev, host.size()) != hipSuccess ||
        hipMemcpy(c.dev, host.data(), host.size(), hipMemcpyHostToDevice) != hipSuccess) {
        return false;
    }
    c.t = t;
    c.k = k;
    c.rows = rows;
    // Every byte of the repacked tensor is read once (the Q4_K formula used here
    // before overstated the low-bit types, e.g. IQ2_S by 1.7x; #63).
    c.bytes = (int64_t) host.size();
    return true;
}

double time_ms(const std::vector<Case> & cases, const void * x, float * y, const int iters,
               hipStream_t stream) {
    // Launches alternate over `streams` non-blocking streams (OMPH_BENCH_STREAMS,
    // default 1): with 2 they overlap as the decode's sibling GEMVs do (#71), so
    // the figure is the kernel's own rate rather than the launch-gap-bound one.
    int ns = 1;
    if (const char * e = std::getenv("OMPH_BENCH_STREAMS")) {
        ns = std::atoi(e) > 0 ? std::atoi(e) : 1;
    }
    std::vector<hipStream_t> st((size_t) ns, stream);
    for (int k = 1; k < ns; ++k) {
        (void) hipStreamCreateWithFlags(&st[(size_t) k], hipStreamNonBlocking);
    }
    size_t n = 0;
    for (int i = 0; i < 3; ++i) {
        for (const Case & c : cases) {
            (void) launch(c, x, y, st[n++ % st.size()]);
        }
    }
    (void) hipDeviceSynchronize();
    hipEvent_t e0, e1;
    (void) hipEventCreate(&e0);
    (void) hipEventCreate(&e1);
    (void) hipEventRecord(e0, st[0]);
    for (int k = 1; k < ns; ++k) {
        (void) hipStreamWaitEvent(st[(size_t) k], e0, 0);
    }
    for (int i = 0; i < iters; ++i) {
        for (const Case & c : cases) {
            (void) launch(c, x, y, st[n++ % st.size()]);
        }
    }
    for (int k = 1; k < ns; ++k) {
        hipEvent_t done;
        (void) hipEventCreate(&done);
        (void) hipEventRecord(done, st[(size_t) k]);
        (void) hipStreamWaitEvent(st[0], done, 0);
        (void) hipEventDestroy(done);
    }
    (void) hipEventRecord(e1, st[0]);
    (void) hipDeviceSynchronize();
    float ms = 0.0f;
    (void) hipEventElapsedTime(&ms, e0, e1);
    (void) hipEventDestroy(e0);
    (void) hipEventDestroy(e1);
    for (int k = 1; k < ns; ++k) {
        (void) hipStreamDestroy(st[(size_t) k]);
    }
    return (double) ms / iters;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> [tensor] [--iters N] [--iq4-all] "
                     "[--all-of-type T] [--batch4] [--repack-only T]\n",
                     argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    std::string name = "output.weight";
    int iters = 50;
    bool batch4 = false;
    int all_type = -1;
    int repack_only = -1;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iq4-all") == 0) {
            all_type = 23;
        } else if (std::strcmp(argv[i], "--all-of-type") == 0 && i + 1 < argc) {
            all_type = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--batch4") == 0) {
            batch4 = true;
        } else if (std::strcmp(argv[i], "--repack-only") == 0 && i + 1 < argc) {
            repack_only = std::atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            name = argv[i];
        }
    }
    try {
        omph::gguf::File file(model);
        if (repack_only >= 0) {
            int64_t n_tensors = 0;
            int64_t n_src = 0;
            int64_t n_repacked = 0;
            for (const omph::gguf::TensorInfo & t : file.tensors()) {
                if ((int) t.type != repack_only) {
                    continue;
                }
                const int64_t bb = omph::format::quant_block_bytes(t.type);
                if (bb <= 0) {
                    std::fprintf(stderr, "unsupported type for %s\n", t.name.c_str());
                    return 1;
                }
                const int64_t n_blocks = (int64_t) (t.nbytes / (uint64_t) bb);
                std::vector<uint8_t> packed;
                std::vector<uint8_t> rebuilt((size_t) t.nbytes);
                if (!repack_tensor(t.type, file.tensor_data(t), n_blocks, packed) ||
                    !unrepack_tensor(t.type, packed.data(), n_blocks, rebuilt)) {
                    std::fprintf(stderr, "unsupported type for %s\n", t.name.c_str());
                    return 1;
                }
                if (std::memcmp(rebuilt.data(), file.tensor_data(t), (size_t) t.nbytes) != 0) {
                    std::printf("NOT lossless: %s\n", t.name.c_str());
                    return 1;
                }
                ++n_tensors;
                n_src += (int64_t) t.nbytes;
                n_repacked += (int64_t) packed.size();
            }
            std::printf("repack-only type %d: %lld tensors, source %.1f MiB -> repacked %.1f MiB "
                        "(%.1f%%), all byte-identical rebuilds\n",
                        repack_only, (long long) n_tensors, (double) n_src / (1024 * 1024),
                        (double) n_repacked / (1024 * 1024),
                        100.0 * (double) n_repacked / (double) n_src);
            return 0;
        }
        std::vector<Case> cases;
        double total_bytes = 0.0;

        if (all_type >= 0) {
            // All tensors of one type summed are well past L2, so the aggregate
            // number is a real streaming measurement.
            for (const omph::gguf::TensorInfo & t : file.tensors()) {
                if ((int) t.type != all_type || t.name.rfind("blk.", 0) != 0) {
                    continue;
                }
                Case c;
                if (!prepare(file, &t, c)) {
                    return fail("repack failed");
                }
                cases.push_back(c);
                total_bytes += (double) c.bytes;
            }
            std::printf("type %d: %zu tensors, %.1f MiB read per pass\n", all_type, cases.size(),
                        total_bytes / (1024 * 1024));
        } else {
            const omph::gguf::TensorInfo * t = file.tensor(name);
            if (t == nullptr) {
                return fail("tensor not found");
            }

            Case c;
            if (!prepare(file, t, c)) {
                return fail("repack failed");
            }
            std::printf("tensor %s: %lld x %lld, type %u, %.1f MiB, %lld blocks\n", name.c_str(),
                        (long long) c.rows, (long long) c.k, t->type,
                        (double) t->nbytes / (1024 * 1024),
                        (long long) (c.rows * (c.k / 256)));
            cases.push_back(c);
            total_bytes = (double) c.bytes;
        }

        if (cases.empty()) {
            return fail("no tensor of that type");
        }
        // One x and one y serve every case: size them to the largest (#90).
        int64_t kmax = 0;
        int64_t rmax = 0;
        for (const Case & c : cases) {
            kmax = std::max(kmax, c.k);
            rmax = std::max(rmax, c.rows);
        }
        // Four distinct activation vectors: token 0 is the one the single-token
        // runs use, all four feed the --batch4 check.
        void * dev_x = nullptr;
        void * dev_y = nullptr;
        void * dev_act32 = nullptr;
        if (hipMalloc(&dev_x, (size_t) kmax * 4 * 2) != hipSuccess ||
            hipMalloc(&dev_y, (size_t) (rmax + kCanary) * 4) != hipSuccess ||
            hipMalloc(&dev_act32, (size_t) kmax * 4 * 4) != hipSuccess) {
            return fail("out of VRAM");
        }
        std::vector<float> act((size_t) kmax * 4);
        uint32_t rng = 12345u;
        for (float & a : act) {
            rng = rng * 1664525u + 1013904223u;
            a = ((float) (rng >> 8) / (float) (1u << 24) - 0.5f) * 2.0f;
        }
        (void) hipMemcpy(dev_act32, act.data(), act.size() * 4, hipMemcpyHostToDevice);
        (void) omph::kernels::cast_f32_to_f16((const float *) dev_act32, dev_x,
                                              (int64_t) act.size(), nullptr);
        // Canary past the largest y: a GEMV writing out of its rows trips it.
        const std::vector<float> canary((size_t) kCanary, 1234.5f);
        (void) hipMemcpy(static_cast<float *>(dev_y) + rmax, canary.data(), canary.size() * 4,
                         hipMemcpyHostToDevice);
        const auto canary_ok = [&]() {
            std::vector<float> got((size_t) kCanary);
            (void) hipDeviceSynchronize();
            (void) hipMemcpy(got.data(), static_cast<float *>(dev_y) + rmax, got.size() * 4,
                             hipMemcpyDeviceToHost);
            return got == canary;
        };

        if (const char * occ = std::getenv("OMPH_OCCUPANCY")) {
            (void) occ;
            const omph::kernels::GemvOccupancy o = omph::kernels::gemv_occupancy();
            std::printf("occupancy  : q4k %d  iq4 %d  iq3xxs %d  iq3s %d workgroups/CU "
                        "(block %d, smem %zu/%zu/%zu/%zu)\n",
                        o.q4k, o.iq4, o.iq3, o.iq3s, o.block, o.smem_q4k, o.smem_iq4, o.smem_iq3,
                        o.smem_iq3s);
        }
        if (batch4) {
            // Small-batch path: four tokens per weight read, verified against four
            // single-token GEMVs and reported per token.
            const Case & c = cases.front();
            if (c.t->type != 12 && c.t->type != 21 && c.t->type != 23 && c.t->type != 18) {
                return fail("--batch4 needs a Q4_K, IQ3_S, IQ4_XS or IQ3_XXS tensor");
            }
            // The b4 kernels read token t at x + t * k: pack the four distinct
            // vectors at this case's k.
            void * x4 = nullptr;
            void * y1 = nullptr;
            void * y4 = nullptr;
            if (hipMalloc(&x4, (size_t) c.k * 4 * 2) != hipSuccess ||
                hipMalloc(&y1, (size_t) c.rows * 4) != hipSuccess ||
                hipMalloc(&y4, (size_t) c.rows * 4 * 4) != hipSuccess) {
                return fail("out of VRAM (batch4)");
            }
            for (int t = 0; t < 4; ++t) {
                (void) hipMemcpy(static_cast<uint8_t *>(x4) + (size_t) t * c.k * 2,
                                 static_cast<uint8_t *>(dev_x) + (size_t) t * kmax * 2,
                                 (size_t) c.k * 2, hipMemcpyDeviceToDevice);
            }
            const auto launch_b4 = [&](void * xa, float * ya) {
                if (c.t->type == 12) {
                    return omph::kernels::gemv_q4k_b4(c.dev, xa, ya, c.rows, c.k, nullptr);
                }
                if (c.t->type == 21) {
                    return omph::kernels::gemv_iq3s_b4(c.dev, xa, ya, c.rows, c.k, nullptr);
                }
                if (c.t->type == 23) {
                    return omph::kernels::gemv_iq4_xs_b4(c.dev, xa, ya, c.rows, c.k, nullptr);
                }
                return omph::kernels::gemv_iq3_xxs_b4(c.dev, xa, ya, c.rows, c.k, nullptr);
            };
            for (int i = 0; i < 5; ++i) {
                (void) launch_b4(x4, (float *) y4);
            }
            hipEvent_t e0, e1;
            (void) hipEventCreate(&e0);
            (void) hipEventCreate(&e1);
            (void) hipEventRecord(e0, nullptr);
            for (int i = 0; i < iters; ++i) {
                (void) launch_b4(x4, (float *) y4);
            }
            (void) hipEventRecord(e1, nullptr);
            (void) hipDeviceSynchronize();
            float ms4 = 0.0f;
            (void) hipEventElapsedTime(&ms4, e0, e1);
            (void) hipEventDestroy(e0);
            (void) hipEventDestroy(e1);
            const double per_batch = ms4 / iters;
            const double bytes = (double) c.bytes;
            const double per_token = per_batch / 4.0;
            std::printf("batch4     : %.3f ms per 4 tokens (%.3f ms/token) -> %.1f GB/s, "
                        "%.1f GB/s effective per token\n",
                        per_batch, per_token, bytes / (per_batch * 1e6),
                        bytes / (per_token * 1e6));
            // Token t against the single-token GEMV on token t's own vector, so
            // a swapped x or y offset fails.
            double maxd = 0.0;
            for (int t = 0; t < 4; ++t) {
                (void) launch(c, static_cast<uint8_t *>(x4) + (size_t) t * c.k * 2, (float *) y1,
                              nullptr);
                (void) hipDeviceSynchronize();
                std::vector<float> a((size_t) c.rows);
                std::vector<float> b((size_t) c.rows);
                (void) hipMemcpy(a.data(), static_cast<uint8_t *>(y4) + (size_t) t * c.rows * 4,
                                 (size_t) c.rows * 4, hipMemcpyDeviceToHost);
                (void) hipMemcpy(b.data(), y1, (size_t) c.rows * 4, hipMemcpyDeviceToHost);
                for (int64_t i = 0; i < c.rows; ++i) {
                    maxd = std::max(maxd, (double) std::fabs(a[(size_t) i] - b[(size_t) i]));
                }
            }
            std::printf("batch4 acc : max|d| vs 4 single-token GEMVs = %.3e\n", maxd);
            return maxd < 1e-3 ? 0 : 1;
        }
        const double ms = time_ms(cases, dev_x, (float *) dev_y, iters, nullptr);
        if (!canary_ok()) {
            return fail("a GEMV wrote past its output rows");
        }
        const double gbs = total_bytes / ms / 1e6;
        std::printf("fused      : %.3f ms  (%.1f MiB read)  -> %.1f GB/s  (%.1f%% of %.1f)\n", ms,
                    total_bytes / (1024 * 1024), gbs, 100.0 * gbs / kMeasuredBandwidthGBs,
                    kMeasuredBandwidthGBs);

        // ---- accuracy against the f16 dequant + hipBLASLt path
        const Case & c = cases.front();
        const omph::gguf::TensorInfo * t = c.t;
        void * quantized = nullptr;
        void * dev_f16 = nullptr;
        void * dev_y_f16 = nullptr;
        if (hipMalloc(&quantized, (size_t) t->nbytes) != hipSuccess ||
            hipMalloc(&dev_f16, (size_t) c.rows * c.k * 2) != hipSuccess ||
            hipMalloc(&dev_y_f16, (size_t) c.rows * 4) != hipSuccess ||
            hipMemcpy(quantized, file.tensor_data(*t), (size_t) t->nbytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return fail("upload failed");
        }
        omph::runtime::Linear linear;
        if (!omph::kernels::dequantize(t->type, quantized, dev_f16, c.rows * c.k, true, nullptr) ||
            !linear.run(dev_f16, dev_x, (float *) dev_y_f16, c.rows, c.k, 1)) {
            return fail("f16 reference failed");
        }
        for (int i = 0; i < 5; ++i) {
            (void) launch(c, dev_x, (float *) dev_y, nullptr);
        }
        (void) hipDeviceSynchronize();
        std::vector<float> got((size_t) c.rows);
        std::vector<float> want((size_t) c.rows);
        (void) hipMemcpy(got.data(), dev_y, (size_t) c.rows * 4, hipMemcpyDeviceToHost);
        (void) hipMemcpy(want.data(), dev_y_f16, (size_t) c.rows * 4, hipMemcpyDeviceToHost);
        double max_abs = 0.0;
        double mean_abs = 0.0;
        double ref_max = 1e-12;
        for (int64_t i = 0; i < c.rows; ++i) {
            max_abs = std::max(max_abs, (double) std::fabs(got[i] - want[i]));
            mean_abs += std::fabs(got[i] - want[i]);
            ref_max = std::max(ref_max, (double) std::fabs(want[i]));
        }
        std::printf("accuracy   : %s  max|d| %.4f (%.2e rel), mean|d| %.4f, |ref| up to %.1f\n",
                    t->name.c_str(), max_abs, max_abs / ref_max, mean_abs / c.rows, ref_max);
        int shown = 0;
        for (int64_t i = 0; i < c.rows && shown < 4; ++i) {
            if (std::fabs(got[i] - want[i]) > 1e-3) {
                std::printf("  row %lld: got %.5f want %.5f (idx_in_row? no) d=%.5f\n",
                            (long long) i, got[i], want[i], got[i] - want[i]);
                ++shown;
            }
        }
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
    return 0;
}
