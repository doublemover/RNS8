#ifndef RNS8_BACKEND_AMDGPU_BUILTINS_HPP
#define RNS8_BACKEND_AMDGPU_BUILTINS_HPP

#include <cstddef>
#include <cstdint>

#include "rns8/rns8.h"

namespace rns8::detail {

bool amdgpu_builtins_compiled();
rns8_status amdgpu_builtins_probe(int device_id, rns8_device_info& out);
bool amdgpu_builtins_workspace_requirements(int64_t max_m, int64_t max_n, int64_t k, std::size_t& total_bytes);
bool amdgpu_builtins_use_cdna3_mfma_32x32x16(int64_t m, int64_t n, int64_t k);

enum class amdgpu_builtins_dense_kernel_variant : int {
  auto_select = 0,
  cdna3_mfma_16x16x32 = 1,
  cdna3_mfma_32x32x16 = 2,
};

void amdgpu_builtins_set_dense_kernel_variant_override(amdgpu_builtins_dense_kernel_variant variant);

rns8_status amdgpu_builtins_gemm_rns_device(
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

rns8_status amdgpu_builtins_gemm_finite_u8_device(
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
    uint16_t modulus);

rns8_status amdgpu_builtins_gemm_rns_sparse_a_device(
    int device_id,
    const void* device_a_packed_values,
    const void* device_a_packed_indices,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix);

rns8_status amdgpu_builtins_gemm_finite_u8_sparse_a_device(
    int device_id,
    const void* device_a_packed_values,
    const void* device_a_packed_indices,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus);

}  // namespace rns8::detail

#endif
