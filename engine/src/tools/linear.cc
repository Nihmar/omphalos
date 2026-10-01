// omph-linear — y = x @ W^T through the naive GPU path:
// dequantize W to f16, cast x to f16, hipBLASLt GEMM with f32 accumulate.
//
// usage: omph-linear <model.gguf> <weight-tensor> <in.f32> <out.f32> <tokens>
//   in.f32:  tokens x in_features, row-major float32
//   out.f32: tokens x out_features, row-major float32
#include "format/gguf.hh"
#include "kernels/dequant.hh"
#include "kernels/elementwise.hh"
#include "runtime/matmul.hh"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int fail(const char * msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s <model.gguf> <weight-tensor> <in.f32> <out.f32> <tokens>\n",
                     argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * weight_name = argv[2];
    const char * in_path = argv[3];
    const char * out_path = argv[4];
    const int64_t tokens = std::atoll(argv[5]);

    try {
        omph::gguf::File file(model_path);
        const omph::gguf::TensorInfo * weight = file.tensor(weight_name);
        if (weight == nullptr) {
            return fail("weight tensor not found");
        }
        if (weight->ne.size() != 2) {
            return fail("weight tensor must be 2D");
        }
        const int64_t in_features = (int64_t) weight->ne[0];
        const int64_t out_features = (int64_t) weight->ne[1];
        const int64_t w_elems = in_features * out_features;
        const size_t x_elems = (size_t) tokens * in_features;
        const size_t y_elems = (size_t) tokens * out_features;

        std::vector<float> host_in(x_elems);
        if (FILE * fi = std::fopen(in_path, "rb")) {
            const size_t got = std::fread(host_in.data(), sizeof(float), x_elems, fi);
            std::fclose(fi);
            if (got != x_elems) {
                return fail("short read of the input file");
            }
        } else {
            return fail("cannot open the input file");
        }

        void * dev_wq = nullptr;
        void * dev_w16 = nullptr;
        void * dev_x32 = nullptr;
        void * dev_x16 = nullptr;
        void * dev_y32 = nullptr;
        if (hipMalloc(&dev_wq, weight->nbytes) != hipSuccess ||
            hipMalloc(&dev_w16, (size_t) w_elems * 2) != hipSuccess ||
            hipMalloc(&dev_x32, x_elems * 4) != hipSuccess ||
            hipMalloc(&dev_x16, x_elems * 2) != hipSuccess ||
            hipMalloc(&dev_y32, y_elems * 4) != hipSuccess) {
            return fail("hipMalloc failed");
        }

        if (hipMemcpy(dev_wq, file.tensor_data(*weight), weight->nbytes, hipMemcpyHostToDevice) !=
                hipSuccess ||
            hipMemcpy(dev_x32, host_in.data(), x_elems * 4, hipMemcpyHostToDevice) != hipSuccess) {
            return fail("hipMemcpy (host->device) failed");
        }

        if (!omph::kernels::dequantize(weight->type, dev_wq, dev_w16, w_elems, /*out_f16=*/true,
                                       nullptr)) {
            return fail("dequantize failed");
        }
        if (!omph::kernels::cast_f32_to_f16(static_cast<const float *>(dev_x32), dev_x16,
                                            (int64_t) x_elems, nullptr)) {
            return fail("cast failed");
        }

        omph::runtime::Linear linear;
        if (!linear.run(dev_w16, dev_x16, static_cast<float *>(dev_y32), out_features, in_features,
                        tokens)) {
            return fail("linear failed");
        }
        if (hipDeviceSynchronize() != hipSuccess) {
            return fail("kernel failed");
        }

        std::vector<float> host_out(y_elems);
        if (hipMemcpy(host_out.data(), dev_y32, y_elems * 4, hipMemcpyDeviceToHost) != hipSuccess) {
            return fail("hipMemcpy (device->host) failed");
        }
        (void) hipFree(dev_wq);
        (void) hipFree(dev_w16);
        (void) hipFree(dev_x32);
        (void) hipFree(dev_x16);
        (void) hipFree(dev_y32);

        if (FILE * fo = std::fopen(out_path, "wb")) {
            const bool written = std::fwrite(host_out.data(), 4, y_elems, fo) == (size_t) y_elems;
            if (std::fclose(fo) != 0 || !written) {
                return fail("cannot write the output file");
            }
        } else {
            return fail("cannot open the output file");
        }

        std::printf("linear %s: (%lld x %lld) x (%lld x %lld) -> %s\n", weight_name,
                    (long long) tokens, (long long) in_features, (long long) in_features,
                    (long long) out_features, out_path);
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
    return 0;
}
