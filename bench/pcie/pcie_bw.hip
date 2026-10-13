// Host <-> device copy bandwidth.
//
// #176 probe (the default, unchanged): 151 MiB (a sequence checkpoint's size,
// #158) through hipMemcpyAsync, pinned host memory (coherent and non-coherent),
// in one copy or in 96 pieces; the fourth of four repetitions is reported. Also
// prints the GPU link speed read from sysfs during the run.
//
// #409 modes: the shapes the KV streaming design actually issues. The quantized
// KV lives in pinned host RAM and is paged into a small device ring in chunks of
// whole rows, one attention layer at a time: per chunk, one H2D of the K rows
// and one of the V rows of a layer. These modes measure that traffic on its own
// (--rows), concurrently with a VRAM read that mimics the decode step's weight
// traffic (--overlap), and the cost of the ready/consumed fences per chunk
// (--fence).
//
// Model shapes (PLAN.md §3): 16 attention layers, 4 KV heads, head_dim 256 ->
// attn_kv = 1024 B and scales = 4 * 8 * 2 = 64 B per row; K Q8 row 1088 B,
// K Q4 row 576 B, V Q4 row 576 B. The #175 default is K4 on 8 of the 16 layers,
// so a token costs 22.5 KB over the stack (19.5 KB with K4/V4).
//
//   hipcc -O2 --offload-arch=gfx1200 -o bench/build/pcie_bw bench/pcie/pcie_bw.hip
//   ./bench/build/pcie_bw              # the #176 probe
//   ./bench/build/pcie_bw --rows       # 1 GiB of KV-shaped H2D per configuration
//   ./bench/build/pcie_bw --overlap 1 # the copies under a VRAM read (gib of copies)
//   ./bench/build/pcie_bw --fence      # event record/wait cost per chunk
//   ./bench/build/pcie_bw --all        # rows + overlap + fence
//
// --rows takes (rows per chunk, GiB streamed) to keep the run short:
//   ./bench/build/pcie_bw --rows 512 1
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                                  \
    do {                                                                                 \
        hipError_t err_ = (expr);                                                        \
        if (err_ != hipSuccess) {                                                        \
            std::fprintf(stderr, "HIP error: %s (%s:%d)\n", hipGetErrorString(err_),     \
                         __FILE__, __LINE__);                                            \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

namespace {

constexpr size_t kK8Row = 1088;  // 4 * 256 B + 4 * 8 * 2 B scales
constexpr size_t kK4Row = 576;   // 4 * 256 / 2 B + scales
constexpr size_t kV4Row = 576;

double ms_since(const std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string read_file(const char * path) {
    std::ifstream f(path);
    std::string s;
    if (f) std::getline(f, s);
    return s.empty() ? "?" : s;
}

// The root port #176 watched, plus the card's own link (the probe may run after
// a resume or through a different slot).
std::string links() {
    const std::string root = read_file("/sys/devices/pci0000:00/0000:00:01.0/current_link_speed");
    const std::string gpu = read_file("/sys/bus/pci/devices/0000:03:00.0/current_link_speed");
    const std::string width = read_file("/sys/bus/pci/devices/0000:03:00.0/current_link_width");
    return "gpu " + gpu + " x" + width + ", root " + root;
}

// --- #176: the checkpoint-shaped probe -------------------------------------
void checkpoint_probe() {
    const size_t n = size_t(151) << 20;
    void *d = nullptr, *h = nullptr, *h2 = nullptr;
    if (hipMalloc(&d, n) != hipSuccess || hipHostMalloc(&h, n) != hipSuccess ||
        hipHostMalloc(&h2, n, hipHostMallocNonCoherent) != hipSuccess) {
        std::fprintf(stderr, "allocation failed\n");
        std::exit(1);
    }
    const auto run = [&](const char * what, void * host, bool d2h, int chunks) {
        double ms = 0;
        std::string speed;
        for (int rep = 0; rep < 4; ++rep) {
            (void) hipDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            const size_t c = n / chunks;
            for (int i = 0; i < chunks; ++i) {
                char * hp = static_cast<char *>(host) + i * c;
                char * dp = static_cast<char *>(d) + i * c;
                (void) (d2h ? hipMemcpyAsync(hp, dp, c, hipMemcpyDeviceToHost, nullptr)
                            : hipMemcpyAsync(dp, hp, c, hipMemcpyHostToDevice, nullptr));
            }
            speed = links();
            (void) hipDeviceSynchronize();
            ms = ms_since(t0);
        }
        std::printf("%-12s %s %3d piece(s): %7.2f ms %6.1f GB/s  (%s)\n", what, d2h ? "D2H" : "H2D",
                    chunks, ms, n / ms / 1e6, speed.c_str());
    };
    for (int chunks : {1, 96}) {
        run("coherent", h, true, chunks);
        run("coherent", h, false, chunks);
        run("noncoherent", h2, true, chunks);
        run("noncoherent", h2, false, chunks);
    }
    (void) hipFree(d);
    (void) hipFreeHost(h);
    (void) hipFreeHost(h2);
}

// --- #409: KV-row-shaped streaming -----------------------------------------

// One layer's K and V rows of a chunk, from the pinned host store into the
// device ring: H2D pairs on `stream`. The host source advances by a whole chunk
// per pair (a real store is read once per pass), the device ring is the small
// buffer that gets reused.
struct StreamResult {
    double ms = 0;
    int64_t pairs = 0;
    int64_t rows = 0;
    size_t k_bytes = 0, v_bytes = 0;
    double bytes() const { return (double)pairs * (double)(k_bytes + v_bytes); }
    double gb_s() const { return ms <= 0 ? 0 : bytes() / (ms / 1000.0) / 1e9; }
    double us_per_pair() const { return pairs == 0 ? 0 : ms * 1000.0 / (double)pairs; }
    double us_per_row() const { return pairs == 0 ? 0 : us_per_pair() / (double)rows; }
};

// Enqueues `rounds` passes over a `store_bytes` host store in `rows`-row chunks
// (K rows then V rows per chunk). With `sync`, waits for the stream and returns
// the wall time.
StreamResult stream_rows(void * hk, void * hv, void * dk, void * dv, hipStream_t s, int64_t rows,
                         size_t k_row, size_t v_row, size_t store_bytes, int rounds, bool sync) {
    const size_t k_bytes = (size_t)rows * k_row;
    const size_t v_bytes = (size_t)rows * v_row;
    const int64_t per_round = std::max<int64_t>(1, (int64_t)store_bytes / (int64_t)(k_bytes + v_bytes));
    StreamResult r;
    r.rows = rows;
    r.k_bytes = k_bytes;
    r.v_bytes = v_bytes;
    r.pairs = per_round * rounds;
    HIP_CHECK(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    for (int round = 0; round < rounds; ++round) {
        for (int64_t i = 0; i < per_round; ++i) {
            HIP_CHECK(hipMemcpyAsync(static_cast<char *>(dk), static_cast<char *>(hk) + i * k_bytes,
                                     k_bytes, hipMemcpyHostToDevice, s));
            HIP_CHECK(hipMemcpyAsync(static_cast<char *>(dv), static_cast<char *>(hv) + i * v_bytes,
                                     v_bytes, hipMemcpyHostToDevice, s));
        }
    }
    if (sync) {
        HIP_CHECK(hipStreamSynchronize(s));
        r.ms = ms_since(t0);
    }
    return r;
}

void rows_probe(const int64_t rows_arg, const double gib) {
    const int64_t rows = rows_arg > 0 ? rows_arg : 512;
    const size_t total = (size_t)(gib * (1ull << 30));
    const size_t chunk_max = (size_t)rows * std::max(kK8Row, kV4Row);
    void * hk = nullptr, * hv = nullptr, * dk = nullptr, * dv = nullptr;
    HIP_CHECK(hipHostMalloc(&hk, total));
    HIP_CHECK(hipHostMalloc(&hv, total));
    HIP_CHECK(hipMalloc(&dk, chunk_max));
    HIP_CHECK(hipMalloc(&dv, chunk_max));
    std::memset(hk, 0x5a, total);
    std::memset(hv, 0x5a, total);
    hipStream_t s = nullptr;
    HIP_CHECK(hipStreamCreate(&s));

    std::printf("# KV-row-shaped H2D: %lld-row chunks, %.2f GiB per configuration (%s)\n",
                (long long)rows, gib, links().c_str());
    for (const char * name : {"K8+V4", "K4+V4"}) {
        const size_t k_row = std::strcmp(name, "K8+V4") == 0 ? kK8Row : kK4Row;
        (void) stream_rows(hk, hv, dk, dv, s, rows, k_row, kV4Row, total, 1, true);  // warm-up
        std::vector<StreamResult> runs;
        for (int rep = 0; rep < 5; ++rep) {
            runs.push_back(stream_rows(hk, hv, dk, dv, s, rows, k_row, kV4Row, total, 1, true));
        }
        std::sort(runs.begin(), runs.end(),
                  [](const StreamResult & a, const StreamResult & b) { return a.ms < b.ms; });
        const StreamResult & m = runs[runs.size() / 2];
        std::printf("%-6s %4lld rows/chunk  K %4zu B  V %3zu B  %7.3f ms  %6.1f GB/s  "
                    "%8.2f us/pair  %6.3f us/row\n",
                    name, (long long)rows, k_row, kV4Row, m.ms, m.gb_s(), m.us_per_pair(), m.us_per_row());
    }
    HIP_CHECK(hipStreamDestroy(s));
    (void) hipFree(dk);
    (void) hipFree(dv);
    (void) hipFreeHost(hk);
    (void) hipFreeHost(hv);
}

// --- #409: the copies under a decode-like VRAM read ------------------------

// Grid-stride 128-bit read; the sum keeps the loads live (bench/bw_membench.hip).
__global__ void read_kernel(const ulonglong2 * __restrict__ in, float * __restrict__ out,
                            unsigned long long n) {
    const unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long acc = 0;
    for (unsigned long long i = tid; i < n; i += stride) {
        const ulonglong2 v = in[i];
        acc += v.x ^ v.y;
    }
    if (acc == 0x123456789ull) out[0] = 1.0f;
}

void overlap_probe(const double gib, const int64_t rows_arg) {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    const size_t weights = (size_t)(2.0 * (1ull << 30));  // read side: 2 GiB per pass
    const int64_t rows = rows_arg > 0 ? rows_arg : 512;
    const size_t store = (size_t)(gib * (1ull << 30));
    const int rounds = 3;  // 3 GiB of copies, ~100 ms at 28 GB/s
    ulonglong2 * w = nullptr;
    float * out = nullptr;
    HIP_CHECK(hipMalloc(&w, weights));
    HIP_CHECK(hipMalloc(&out, 4096));
    HIP_CHECK(hipMemset(w, 0x3f, weights));
    const unsigned long long n = weights / sizeof(ulonglong2);
    const int block = 256;
    const int grid = prop.multiProcessorCount * 16;
    const auto pass = [&](hipStream_t s) { read_kernel<<<grid, block, 0, s>>>(w, out, n); };
    // Both streams non-blocking: on HIP the legacy default stream (nullptr)
    // waits for every blocking stream, which would serialize the copies and
    // the reads instead of overlapping them.
    hipStream_t compute = nullptr, s = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&compute, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    pass(compute);
    HIP_CHECK(hipStreamSynchronize(compute));
    const auto timed_passes = [&](int count, std::vector<double> * out_ms) {
        for (int i = 0; i < count; ++i) {
            const auto p0 = std::chrono::steady_clock::now();
            pass(compute);
            HIP_CHECK(hipStreamSynchronize(compute));
            if (out_ms) out_ms->push_back(ms_since(p0));
        }
    };
    std::vector<double> base;
    timed_passes(2, nullptr);  // warm-up
    timed_passes(15, &base);
    std::sort(base.begin(), base.end());
    const double base_ms = base[base.size() / 2];

    void * hk = nullptr, * hv = nullptr, * dk = nullptr, * dv = nullptr;
    const size_t chunk_max = (size_t)rows * kK8Row;
    HIP_CHECK(hipHostMalloc(&hk, store));
    HIP_CHECK(hipHostMalloc(&hv, store));
    HIP_CHECK(hipMalloc(&dk, chunk_max));
    HIP_CHECK(hipMalloc(&dv, chunk_max));
    std::memset(hk, 0x5a, store);
    std::memset(hv, 0x5a, store);

    // Copies alone, in the same shape the overlapped run uses.
    const StreamResult alone = stream_rows(hk, hv, dk, dv, s, rows, kK8Row, kV4Row, store, rounds, true);
    std::printf("# overlap: %4lld rows/chunk, %d rounds of %.2f GiB, read side %.1f GiB/pass "
                "(%s)\n",
                (long long)rows, rounds, gib, (double)weights / (1ull << 30), links().c_str());
    std::printf("  copies alone:      %7.3f ms  %6.1f GB/s  %8.2f us/pair  %6.3f us/row\n",
                alone.ms, alone.gb_s(), alone.us_per_pair(), alone.us_per_row());

    // Overlapped: enqueue every copy, then run read passes until the copy
    // stream is idle; a pass that ends after the last copy is dropped.
    std::vector<double> overlapped;
    const auto t0 = std::chrono::steady_clock::now();
    (void) stream_rows(hk, hv, dk, dv, s, rows, kK8Row, kV4Row, store, rounds, false);
    while (hipStreamQuery(s) == hipErrorNotReady) {
        const auto p0 = std::chrono::steady_clock::now();
        pass(compute);
        HIP_CHECK(hipStreamSynchronize(compute));
        overlapped.push_back(ms_since(p0));
    }
    HIP_CHECK(hipStreamSynchronize(s));
    const double copy_ms = ms_since(t0);
    if (overlapped.size() > 1) overlapped.pop_back();  // the tail pass, partly past the copies
    std::sort(overlapped.begin(), overlapped.end());
    const double copy_bytes = alone.bytes();  // pairs already counts every round
    if (!overlapped.empty()) {
        const double med = overlapped[overlapped.size() / 2];
        std::printf("  under the copies:  %7.3f ms  %6.1f GB/s   read pass %7.3f ms (%+.1f%%), "
                    "%zu passes overlapped\n",
                    copy_ms, copy_bytes / (copy_ms / 1000.0) / 1e9, med, 100.0 * (med - base_ms) / base_ms,
                    overlapped.size());
        std::printf("  read side alone:   %7.3f ms  %6.1f GB/s  (single-op benchmark)\n", base_ms,
                    (double)weights / (base_ms / 1000.0) / 1e9);
    }
    HIP_CHECK(hipStreamDestroy(s));
    HIP_CHECK(hipStreamDestroy(compute));
    (void) hipFree(dk);
    (void) hipFree(dv);
    (void) hipFreeHost(hk);
    (void) hipFreeHost(hv);
    (void) hipFree(w);
    (void) hipFree(out);
}

// --- #409: ready/consumed fence cost ---------------------------------------

void fence_probe() {
    hipStream_t s1 = nullptr, s2 = nullptr;
    hipEvent_t e = nullptr;
    HIP_CHECK(hipStreamCreate(&s1));
    HIP_CHECK(hipStreamCreate(&s2));
    HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
    const int n = 20000;
    HIP_CHECK(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        HIP_CHECK(hipEventRecord(e, s1));
        HIP_CHECK(hipStreamWaitEvent(s2, e, 0));
    }
    HIP_CHECK(hipStreamSynchronize(s2));
    const double enqueue_us = ms_since(t0) * 1000.0 / n;
    // A dependency that is not yet satisfied: alternate a small kernel and two
    // fences between the two streams, against the same kernels without fences.
    const int m = 500;
    const auto chain = [&](bool fence) {
        HIP_CHECK(hipDeviceSynchronize());
        const auto c0 = std::chrono::steady_clock::now();
        for (int i = 0; i < m; ++i) {
            read_kernel<<<4, 64, 0, s1>>>(nullptr, nullptr, 0);
            if (fence) {
                HIP_CHECK(hipEventRecord(e, s1));
                HIP_CHECK(hipStreamWaitEvent(s2, e, 0));
                read_kernel<<<4, 64, 0, s2>>>(nullptr, nullptr, 0);
                HIP_CHECK(hipEventRecord(e, s2));
                HIP_CHECK(hipStreamWaitEvent(s1, e, 0));
            }
        }
        HIP_CHECK(hipStreamSynchronize(s1));
        HIP_CHECK(hipStreamSynchronize(s2));
        return ms_since(c0) / m;
    };
    (void) chain(false);
    (void) chain(true);
    const double no_fence = chain(false), fence = chain(true);
    std::printf("# fence: %5.2f us per record+wait enqueued (host side), chain %.3f ms per iteration "
                "without vs %.3f with two fences (%+.2f us per hop)\n",
                enqueue_us, no_fence, fence, (fence - no_fence) * 1000.0 / 2.0);
    HIP_CHECK(hipEventDestroy(e));
    HIP_CHECK(hipStreamDestroy(s1));
    HIP_CHECK(hipStreamDestroy(s2));
}

}  // namespace

int main(int argc, char ** argv) {
    bool rows = false, overlap = false, fence = false;
    int64_t rows_arg = 0;
    double gib = 1.0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--rows") == 0) {
            rows = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') rows_arg = std::atoll(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') gib = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--overlap") == 0) {
            overlap = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') gib = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--fence") == 0) {
            fence = true;
        } else if (std::strcmp(argv[i], "--all") == 0) {
            rows = overlap = fence = true;
        } else {
            std::fprintf(stderr, "usage: %s [--rows [rows_per_chunk] [gib]] [--overlap [gib]] "
                                 "[--fence] [--all]\n", argv[0]);
            return 1;
        }
    }
    if (!rows && !overlap && !fence) {
        checkpoint_probe();
        return 0;
    }
    if (rows) rows_probe(rows_arg, gib);
    if (overlap) overlap_probe(gib, rows_arg > 0 ? rows_arg : 512);
    if (fence) fence_probe();
    return 0;
}
