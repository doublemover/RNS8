#ifndef RNS8_ABI_H
#define RNS8_ABI_H

#include <stdint.h>

/* Public enum fields use 32-bit storage on supported ABIs. Fixed C++ backing
 * permits safely validating every 32-bit value received from C/FFI callers;
 * an unfixed enum can trigger undefined behavior before the rejection check.
 * C clients must use the platform's normal 32-bit enum ABI, not -fshort-enums.
 */
#ifdef __cplusplus
#  define RNS8_ENUM_BASE : uint32_t
#else
#  define RNS8_ENUM_BASE
#endif

#endif
