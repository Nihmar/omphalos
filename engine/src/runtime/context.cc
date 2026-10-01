// C ABI implementation (PLAN.md §6.1): thin wrappers over the runtime.
#include "omphalos.h"

#include "runtime/device.hh"

extern "C" {

#define OMPH_STR2(x) #x
#define OMPH_STR(x) OMPH_STR2(x)

const char * omph_version(void) {
    // one source of truth: the header's version macros
    return OMPH_STR(OMPHALOS_VERSION_MAJOR) "." OMPH_STR(OMPHALOS_VERSION_MINOR) "." OMPH_STR(
        OMPHALOS_VERSION_PATCH);
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
