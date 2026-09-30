// omph-dequant — dequantize one GGUF tensor through the GPU path and dump it.
//
// usage: omph-dequant <model.gguf> <tensor> <out.raw> [--f16]
#include "format/gguf.hh"
#include "kernels/dequant.hh"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int fail(const char * msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tensor> <out.raw> [--f16]\n", argv[0]);
        return 2;
    }
    const bool f16 = (argc > 4) && std::strcmp(argv[4], "--f16") == 0;

    try {
        omph::gguf::File file(argv[1]);
        const omph::gguf::TensorInfo * tensor = file.tensor(argv[2]);
        if (tensor == nullptr) {
            std::fprintf(stderr, "tensor not found: %s\n", argv[2]);
            return 1;
        }
        int64_t n = 1;
        for (const uint64_t dim : tensor->ne) {
            n *= (int64_t) dim;
        }
        const size_t out_bytes = (size_t) n * (f16 ? 2 : 4);

        void * dev_src = nullptr;
        void * dev_dst = nullptr;
        if (hipMalloc(&dev_src, tensor->nbytes) != hipSuccess ||
            hipMalloc(&dev_dst, out_bytes) != hipSuccess) {
            return fail("hipMalloc failed");
        }
        if (hipMemcpy(dev_src, file.tensor_data(*tensor), tensor->nbytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return fail("hipMemcpy (host->device) failed");
        }
        if (!omph::kernels::dequantize(tensor->type, dev_src, dev_dst, n, f16, nullptr)) {
            std::fprintf(stderr, "unsupported type id %u\n", tensor->type);
            return 1;
        }
        if (hipDeviceSynchronize() != hipSuccess) {
            return fail("kernel failed");
        }
        std::vector<uint8_t> host(out_bytes);
        if (hipMemcpy(host.data(), dev_dst, out_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
            return fail("hipMemcpy (device->host) failed");
        }
        (void) hipFree(dev_src);
        (void) hipFree(dev_dst);

        FILE * out = std::fopen(argv[3], "wb");
        if (out == nullptr) {
            return fail("cannot open output file");
        }
        std::fwrite(host.data(), 1, out_bytes, out);
        std::fclose(out);

        const omph::gguf::TypeInfo * info = omph::gguf::type_info(tensor->type);
        std::printf("dequantized %s (%s, %lld elements) -> %s (%zu bytes, %s)\n", argv[2],
                    info ? info->name : "?", (long long) n, argv[3], out_bytes,
                    f16 ? "f16" : "f32");
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
    return 0;
}
