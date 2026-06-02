#ifndef RNS8_BACKEND_WRAP64_HIP_HPP
#define RNS8_BACKEND_WRAP64_HIP_HPP

#include <cstddef>
#include <cstdint>

#include "rns8/rns8.h"

namespace rns8::detail {

rns8_status wrap64_hip_pack_u64_device(
    int device_id,
    const uint64_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_byte_limbs,
    int64_t rows,
    int64_t cols,
    int64_t ld);

rns8_status wrap64_hip_gemm_byte_limbs_device_resident(
    int device_id,
    const void* device_a_limbs,
    const void* device_b_limbs,
    void* device_c_limbs,
    int64_t m,
    int64_t n,
    int64_t k);

rns8_status wrap64_hip_export_u64_device(
    int device_id,
    const void* device_byte_limbs,
    void** export_buffer,
    std::size_t* export_bytes,
    int64_t rows,
    int64_t cols,
    uint64_t* dst,
    int64_t ld);

}  // namespace rns8::detail

#endif
