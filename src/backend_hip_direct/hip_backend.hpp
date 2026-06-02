#ifndef RNS8_BACKEND_HIP_DIRECT_HPP
#define RNS8_BACKEND_HIP_DIRECT_HPP

#include <cstddef>
#include <cstdint>

#include "rns8/rns8.h"

namespace rns8::detail {

bool hip_direct_compiled();
rns8_status hip_direct_probe(int device_id, rns8_device_info& out);
rns8_status hip_direct_allocate(int device_id, std::size_t bytes, void** out);
rns8_status hip_direct_free(int device_id, void* ptr);
rns8_status hip_direct_zero(int device_id, void* ptr, std::size_t bytes);
rns8_status hip_direct_copy_device_to_host(int device_id, void* dst, const void* src, std::size_t bytes);
rns8_status hip_direct_copy_host_to_device(int device_id, void* dst, const void* src, std::size_t bytes);
rns8_status hip_direct_ensure_upload_buffer(int device_id, std::size_t bytes, void** buffer, std::size_t* capacity);
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
rns8_status hip_direct_pack_i64_device(
    int device_id,
    const int64_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix);
rns8_status hip_direct_pack_u64_device(
    int device_id,
    const uint64_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_residues,
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
rns8_status hip_direct_gemm_rns_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix);
rns8_status hip_direct_export_i64_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t bound,
    int64_t* dst,
    int64_t ld);
rns8_status hip_direct_export_u64_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t bound,
    uint64_t* dst,
    int64_t ld);
rns8_status hip_direct_synchronize(int device_id);

}  // namespace rns8::detail

#endif
