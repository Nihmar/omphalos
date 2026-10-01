// Bump allocator over one device buffer: the f16 staging of one layer's weights
// on the dequant + GEMM path, reset per layer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace omph::runtime {

class Scratch {
public:
    void init(void * base, const size_t bytes, const bool trace) {
        base_ = base;
        cap_ = bytes;
        trace_ = trace;
    }
    void reset() { used_ = 0; }
    void * alloc(const size_t bytes) {
        const size_t off = (used_ + 255) & ~(size_t) 255;
        if (trace_) {
            std::fprintf(stderr, "  alloc %8zu MiB (used %8zu MiB)\n", bytes >> 20,
                         (off + bytes) >> 20);
        }
        if (off + bytes > cap_) {
            std::fprintf(stderr, "scratch: need %zu MiB, have %zu MiB (used %zu MiB)\n",
                         (off + bytes) >> 20, cap_ >> 20, used_ >> 20);
            throw std::runtime_error("weight scratch exhausted");
        }
        used_ = off + bytes;
        return static_cast<uint8_t *>(base_) + off;
    }

private:
    void * base_ = nullptr;
    size_t cap_ = 0;
    size_t used_ = 0;
    bool trace_ = false;
};

} // namespace omph::runtime
