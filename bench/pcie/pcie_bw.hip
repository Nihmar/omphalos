// Host <-> device copy bandwidth (#176): 151 MiB (a sequence checkpoint's
// size, #158) through hipMemcpyAsync, pinned host memory (coherent and
// non-coherent), in one copy or in 96 pieces; the fourth of four repetitions
// is reported. Also prints the GPU link speed read from sysfs during the run.
//
//   hipcc -O2 --offload-arch=gfx1200 -o pcie_bw bench/pcie/pcie_bw.hip && ./pcie_bw
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

int main() {
    const size_t n = size_t(151) << 20;
    void *d = nullptr, *h = nullptr, *h2 = nullptr;
    if (hipMalloc(&d, n) != hipSuccess || hipHostMalloc(&h, n) != hipSuccess ||
        hipHostMalloc(&h2, n, hipHostMallocNonCoherent) != hipSuccess) {
        std::fprintf(stderr, "allocation failed\n");
        return 1;
    }
    const auto link = [] {
        std::ifstream f("/sys/devices/pci0000:00/0000:00:01.0/current_link_speed");
        std::string s;
        std::getline(f, s);
        return s;
    };
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
            speed = link();
            (void) hipDeviceSynchronize();
            ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        std::printf("%-12s %s %3d piece(s): %7.2f ms %6.1f GB/s  (root port link during the copy: %s)\n", what,
                    d2h ? "D2H" : "H2D", chunks, ms, n / ms / 1e6, speed.c_str());
    };
    for (int chunks : {1, 96}) {
        run("coherent", h, true, chunks);
        run("coherent", h, false, chunks);
        run("noncoherent", h2, true, chunks);
        run("noncoherent", h2, false, chunks);
    }
    return 0;
}
