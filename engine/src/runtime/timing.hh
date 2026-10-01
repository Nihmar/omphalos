// Host wall clock and per-phase GPU event timing (OMPH_TIMING / OMPH_PHASES).
#pragma once

#include <hip/hip_runtime.h>

#include <ctime>
#include <utility>
#include <vector>

namespace omph::runtime {

// Monotonic wall clock in ms (<chrono> pulls in <format>, which this
// clang + libstdc++ pair does not compile).
inline double now_ms() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec * 1e-6;
}

// Phase timing: record a fresh event pair per call, resolve them all after the
// run (reusing one pair made every total `calls x last interval`, #64).
class PhaseTimer {
public:
    void enable(const bool on) { on_ = on; }
    // On the stream the timed work is issued to: the GEMVs forked onto the
    // side stream would otherwise be timed by the default stream (#100).
    void start(hipStream_t stream = nullptr) {
        if (on_) {
            (void) hipEventCreate(&a_);
            (void) hipEventRecord(a_, stream);
        }
    }
    void stop(std::vector<std::pair<hipEvent_t, hipEvent_t>> & sink,
              hipStream_t stream = nullptr) {
        if (on_) {
            hipEvent_t b{};
            (void) hipEventCreate(&b);
            (void) hipEventRecord(b, stream);
            sink.emplace_back(a_, b);
        }
    }
    // Sums and destroys the recorded pairs.
    static double total_ms(std::vector<std::pair<hipEvent_t, hipEvent_t>> & v) {
        double ms = 0.0;
        for (const auto & p : v) {
            float d = 0.0f;
            if (hipEventElapsedTime(&d, p.first, p.second) == hipSuccess) {
                ms += d;
            }
            (void) hipEventDestroy(p.first);
            (void) hipEventDestroy(p.second);
        }
        v.clear();
        return ms;
    }

private:
    bool on_ = false;
    hipEvent_t a_{};
};

} // namespace omph::runtime
