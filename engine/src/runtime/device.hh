// Internal runtime declarations (HIP side).
#pragma once

#include <cstddef>

namespace omph::runtime {

int device_count(void);
int device_info(int index, char * out, size_t out_size);
int self_test(void);

} // namespace omph::runtime
