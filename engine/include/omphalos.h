/* omphalos — public C ABI (PLAN.md §6.1).
 *
 * Deliberately small: the inference API grows here as milestones land
 * (C ABI is callable from anything: Python ctypes, Delphi, the HTTP server).
 */
#ifndef OMPHALOS_H
#define OMPHALOS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OMPHALOS_VERSION_MAJOR 0
#define OMPHALOS_VERSION_MINOR 1
#define OMPHALOS_VERSION_PATCH 0

/* Library version, "major.minor.patch". */
const char * omph_version(void);

/* Number of HIP devices visible to the runtime (0 when none). */
int omph_device_count(void);

/* Writes a human-readable description of device `index` into `out`.
 * Returns 0 on success, negative on error. */
int omph_device_info(int index, char * out, size_t out_size);

/* Runs a trivial GPU kernel end to end (self test). Returns 0 on success. */
int omph_self_test(void);

#ifdef __cplusplus
}
#endif

#endif /* OMPHALOS_H */
