#ifndef RNS8_BACKEND_HIP_DIRECT_HPP
#define RNS8_BACKEND_HIP_DIRECT_HPP

#include <cstdint>

#include "rns8/rns8.h"

namespace rns8::detail {

bool hip_direct_compiled();
rns8_status hip_direct_probe(int device_id, rns8_device_info& out);
rns8_status hip_direct_pack_i64(
    int device_id,
    const int64_t* src,
    int8_t* residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix);
rns8_status hip_direct_pack_u64(
    int device_id,
    const uint64_t* src,
    int8_t* residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix);
rns8_status hip_direct_ring_gemm_i8(
    int device_id,
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus);

}  // namespace rns8::detail

#endif
