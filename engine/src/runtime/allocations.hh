// Owner of a set of device and pinned-host allocations, freed together (#100).
//
// The runner holds raw pointers into its buffers (they are passed to kernels
// as such); this records every allocation so that all of them are released
// when the owner goes away, including when its constructor throws halfway.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace omph::runtime {

class Allocations {
public:
    Allocations() = default;
    Allocations(const Allocations &) = delete;
    Allocations & operator=(const Allocations &) = delete;
    ~Allocations() {
        for (void * p : device_) {
            (void) hipFree(p);
        }
        for (void * p : host_) {
            (void) hipHostFree(p);
        }
    }

    // Device memory; throws `what` when it cannot be allocated.
    void * device(const size_t bytes, const char * what = "out of VRAM") {
        void * p = nullptr;
        if (hipMalloc(&p, bytes) != hipSuccess) {
            (void) hipGetLastError();  // do not leave the OOM for the next check
            throw std::runtime_error(what);
        }
        device_.push_back(p);
        return p;
    }

    // Device memory, or null (and no pending error) when it does not fit.
    void * try_device(const size_t bytes) {
        void * p = nullptr;
        if (hipMalloc(&p, bytes) != hipSuccess) {
            (void) hipGetLastError();
            return nullptr;
        }
        device_.push_back(p);
        return p;
    }

    // Pinned host memory, mapped into the device address space.
    void * host(const size_t bytes, const char * what = "out of pinned host memory") {
        void * p = nullptr;
        if (hipHostMalloc(&p, bytes) != hipSuccess) {
            (void) hipGetLastError();
            throw std::runtime_error(what);
        }
        host_.push_back(p);
        return p;
    }

private:
    std::vector<void *> device_;
    std::vector<void *> host_;
};

} // namespace omph::runtime
