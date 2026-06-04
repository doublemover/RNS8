#ifndef RNS8_BACKEND_CK_HPP
#define RNS8_BACKEND_CK_HPP

#include <cstddef>
#include <cstdint>

#include "rns8/rns8.h"

namespace rns8::detail {

bool ck_compiled();
rns8_status ck_probe(int device_id, rns8_device_info& out);
bool ck_workspace_requirements(
    int64_t max_m,
    int64_t max_n,
    int64_t k,
    std::size_t& a_pack_bytes,
    std::size_t& b_pack_bytes,
    std::size_t& temp_c_bytes,
    std::size_t& total_bytes);
rns8_status ck_gemm_rns_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix);
rns8_status ck_gemm_rns_tiled_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count);
rns8_status ck_gemm_finite_u8_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus);

}  // namespace rns8::detail

#endif
