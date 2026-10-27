#pragma once
#include "rns8/rns8.h"
namespace rns8::detail {
// Scratch is caller-owned and disjoint; one ordered trial, synchronous readback.
rns8_status hip_finite_verify_trial(int device, const void* a, const void* b,
    const void* c, uint64_t m, uint64_t n, uint64_t k, uint32_t modulus,
    const void* r, void* br, void* flag, uint32_t& result);
}
