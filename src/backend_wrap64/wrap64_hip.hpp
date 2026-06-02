#ifndef RNS8_BACKEND_WRAP64_HIP_HPP
#define RNS8_BACKEND_WRAP64_HIP_HPP

#include <cstdint>

#include "rns8/rns8.h"

namespace rns8::detail {

rns8_status wrap64_hip_gemm_byte_limbs(
    int device_id,
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    uint8_t* c_limbs,
    int64_t m,
    int64_t n,
    int64_t k);

}  // namespace rns8::detail

#endif
