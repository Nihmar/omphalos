// C ABI implementation (PLAN.md §6.1): thin wrappers over the runtime.
#include "omphalos.h"

#include "runtime/device.hh"

extern "C" {

const char * omph_version(void) {
    return "0.1.0";
}

int omph_device_count(void) {
    return omph::runtime::device_count();
}

int omph_device_info(int index, char * out, size_t out_size) {
    return omph::runtime::device_info(index, out, out_size);
}

int omph_self_test(void) {
    return omph::runtime::self_test();
}

} // extern "C"
