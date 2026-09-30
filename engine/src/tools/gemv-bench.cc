// omph-gemv-bench — effective bandwidth of the fused Q4_K GEMV against the
// f16 dequant path, on a real tensor of the model (PLAN.md §10.1, §17).
//
// usage: omph-gemv-bench <model.gguf> [tensor] [--iters N]
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

constexpr double kMeasuredBandwidthGBs = 318.3;  // bench/results/m0-bandwidth.csv

int fail(const char * msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

double ms_between(const hipEvent_t a, const hipEvent_t b) {
    float ms = 0.0f;
    (void) hipEventElapsedTime(&ms, a, b);
    return (double) ms;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [tensor] [--iters N]\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    std::string name = "output.weight";
    int iters = 100;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            name = argv[i];
        }
    }
    try {
        omph::gguf::File file(model);
        const omph::gguf::TensorInfo * t = file.tensor(name);
        if (t == nullptr) {
            return fail("tensor not found");
        }
        if (t->type != 12) {
            std::fprintf(stderr, "tensor %s is not Q4_K (type %u)\n", name.c_str(), t->type);
            return 1;
        }
        const int64_t k = (int64_t) t->ne[0];
        const int64_t rows = (int64_t) t->ne[1];
        const int64_t n_blocks = rows * (k / 256);
        std::printf("tensor %s: %lld x %lld, Q4_K, %.1f MiB, %lld blocks\n", name.c_str(),
                    (long long) rows, (long long) k, (double) t->nbytes / (1024 * 1024),
                    (long long) n_blocks);

        // ---- lossless repacking, verified by rebuilding the source bytes
        const omph::format::Q4kLayout layout = omph::format::q4k_layout(n_blocks);
        std::vector<uint8_t> repacked((size_t) layout.total);
        omph::format::repack_q4k(file.tensor_data(*t), n_blocks, repacked.data());
        std::vector<uint8_t> rebuilt((size_t) t->nbytes);
        omph::format::unrepack_q4k(repacked.data(), n_blocks, rebuilt.data());
        const bool identical = std::memcmp(rebuilt.data(), file.tensor_data(*t), (size_t) t->nbytes) == 0;
        std::printf("repack: %lld -> %lld bytes, byte-identical rebuild: %s\n",
                    (long long) t->nbytes, (long long) layout.total, identical ? "yes" : "NO");
        if (!identical) {
            return fail("repacking is not lossless");
        }

        // ---- device buffers
        void * dev_repacked = nullptr;
        void * dev_f16 = nullptr;
        void * dev_x = nullptr;
        void * dev_y = nullptr;
        void * dev_y_f16 = nullptr;
        if (hipMalloc(&dev_repacked, (size_t) layout.total) != hipSuccess ||
            hipMalloc(&dev_f16, (size_t) rows * k * 2) != hipSuccess ||
            hipMalloc(&dev_x, (size_t) k * 2) != hipSuccess ||
            hipMalloc(&dev_y, (size_t) rows * 4) != hipSuccess ||
            hipMalloc(&dev_y_f16, (size_t) rows * 4) != hipSuccess) {
            return fail("out of VRAM");
        }
        if (hipMemcpy(dev_repacked, repacked.data(), (size_t) layout.total,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return fail("upload failed");
        }

        // activation: deterministic pseudo-random, cast to f16 on the device
        std::vector<float> act((size_t) k);
        uint32_t rng = 12345u;
        for (int64_t i = 0; i < k; ++i) {
            rng = rng * 1664525u + 1013904223u;
            act[(size_t) i] = ((float) (rng >> 8) / (float) (1u << 24) - 0.5f) * 2.0f;
        }
        void * dev_act32 = nullptr;
        if (hipMalloc(&dev_act32, (size_t) k * 4) != hipSuccess ||
            hipMemcpy(dev_act32, act.data(), (size_t) k * 4, hipMemcpyHostToDevice) != hipSuccess ||
            !omph::kernels::cast_f32_to_f16((const float *) dev_act32, dev_x, k, nullptr)) {
            return fail("activation setup failed");
        }

        // ---- reference: f16 dequant + hipBLASLt GEMV
        void * quantized = nullptr;
        if (hipMalloc(&quantized, (size_t) t->nbytes) != hipSuccess ||
            hipMemcpy(quantized, file.tensor_data(*t), (size_t) t->nbytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return fail("upload failed");
        }
        omph::runtime::Linear linear;
        hipEvent_t e0, e1;
        (void) hipEventCreate(&e0);
        (void) hipEventCreate(&e1);
        (void) hipEventRecord(e0, nullptr);
        if (!omph::kernels::dequantize(t->type, quantized, dev_f16, rows * k, true, nullptr)) {
            return fail("dequant failed");
        }
        for (int i = 0; i < 5; ++i) {
            (void) linear.run(dev_f16, dev_x, (float *) dev_y_f16, rows, k, 1);
        }
        (void) hipEventRecord(e1, nullptr);
        (void) hipDeviceSynchronize();
        const double f16_gemv_ms = ms_between(e0, e1) / 5.0;
        const double f16_bytes = (double) rows * k * 2.0;
        std::printf("f16 path   : %.3f ms  (weights %.1f MiB)  -> %.1f GB/s\n", f16_gemv_ms,
                    f16_bytes / (1024 * 1024), f16_bytes / f16_gemv_ms / 1e6);

        // ---- fused kernel
        for (int i = 0; i < 5; ++i) {
            (void) omph::kernels::gemv_q4k(dev_repacked, dev_x, (float *) dev_y, rows, k, nullptr);
        }
        (void) hipDeviceSynchronize();
        (void) hipEventRecord(e0, nullptr);
        for (int i = 0; i < iters; ++i) {
            (void) omph::kernels::gemv_q4k(dev_repacked, dev_x, (float *) dev_y, rows, k, nullptr);
        }
        (void) hipEventRecord(e1, nullptr);
        if (hipDeviceSynchronize() != hipSuccess) {
            return fail("gemv failed");
        }
        const double fused_ms = ms_between(e0, e1) / (double) iters;
        const double fused_bytes =
            (double) rows * ((double) k * 0.5 + (double) (k / 256) * 16.0 + (double) (k / 256) * 4.0);
        const double gbs = fused_bytes / fused_ms / 1e6;
        std::printf("fused q4k  : %.3f ms  (weights %.1f MiB)  -> %.1f GB/s  (%.1f%% of %.1f)\n",
                    fused_ms, fused_bytes / (1024 * 1024), gbs, 100.0 * gbs / kMeasuredBandwidthGBs,
                    kMeasuredBandwidthGBs);

        // ---- accuracy vs the f16 path
        std::vector<float> got((size_t) rows);
        std::vector<float> want((size_t) rows);
        (void) hipMemcpy(got.data(), dev_y, (size_t) rows * 4, hipMemcpyDeviceToHost);
        (void) hipMemcpy(want.data(), dev_y_f16, (size_t) rows * 4, hipMemcpyDeviceToHost);
        double max_abs = 0.0;
        double sum_abs = 0.0;
        double ref_max = 1e-12;
        for (int64_t i = 0; i < rows; ++i) {
            max_abs = std::max(max_abs, (double) std::fabs(got[i] - want[i]));
            sum_abs += std::fabs(got[i] - want[i]);
            ref_max = std::max(ref_max, (double) std::fabs(want[i]));
        }
        std::printf("accuracy   : max|d| %.4f (%.2e rel), mean|d| %.4f, |ref| up to %.1f\n", max_abs,
                    max_abs / ref_max, sum_abs / rows, ref_max);
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
    return 0;
}
