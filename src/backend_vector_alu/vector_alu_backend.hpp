#ifndef RNS8_BACKEND_VECTOR_ALU_HPP
#define RNS8_BACKEND_VECTOR_ALU_HPP

#include <cstdint>

#include "rns8/rns8.h"

namespace rns8::detail {

bool vector_alu_compiled();
rns8_status vector_alu_probe(int device_id, rns8_device_info& out);
rns8_status vector_alu_gemm_i64_device(
    int device_id,
    const void* device_a,
    const void* device_b,
    void* device_c,
    void* device_status,
    int64_t m,
    int64_t n,
    int64_t k);
rns8_status vector_alu_gemm_u64_device(
    int device_id,
    const void* device_a,
    const void* device_b,
    void* device_c,
    void* device_status,
    int64_t m,
    int64_t n,
    int64_t k);

}  // namespace rns8::detail

#endif
