// omph-device-info — HIP device probe + self test through the C ABI.
#include "omphalos.h"

#include <cstdio>
#include <cstring>

int main() {
    std::printf("omphalos %s\n", omph_version());

    const int devices = omph_device_count();
    std::printf("devices: %d\n", devices);
    for (int i = 0; i < devices; ++i) {
        char buf[512] = {};
        if (omph_device_info(i, buf, sizeof(buf)) == 0) {
            std::printf("  device %d: %s\n", i, buf);
        } else {
            std::printf("  device %d: <error>\n", i);
        }
    }

    const int rc = omph_self_test();
    std::printf("self test: %s (rc=%d)\n", rc == 0 ? "ok" : "FAILED", rc);
    return rc == 0 ? 0 : 1;
}
