#include "backend_hip_direct/hip_backend.hpp"

#include "core/api_internal.hpp"
#include "core/backend_common.hpp"
#include "core/hip_resources.hpp"
#include "core/internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#  include <hip/hip_runtime_api.h>

extern "C" int rns8_hip_direct_pack_i64_device(
    const int64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix);

extern "C" int rns8_hip_direct_pack_u64_device(
    const uint64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix);

extern "C" int rns8_hip_direct_pack_i64_grouped_device(
    const int64_t* d_src,
    int8_t* const* d_residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld,
    int prefix);

extern "C" int rns8_hip_direct_pack_u64_grouped_device(
    const uint64_t* d_src,
    int8_t* const* d_residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld,
    int prefix);

extern "C" int rns8_hip_direct_pack_u8_modulus_device(
    const uint8_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int modulus);

extern "C" int rns8_hip_direct_ring_gemm_i8_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int modulus_index,
    int selected_prefix,
    int safe_k_block);
extern "C" int rns8_hip_direct_ring_gemm_i8_device_on_stream(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int modulus_index,
    int selected_prefix,
    int safe_k_block,
    void* stream);

extern "C" int rns8_hip_direct_ring_gemm_i8_grouped_prefix_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block);
extern "C" int rns8_hip_direct_ring_gemm_i8_grouped_prefix_device_on_stream(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block,
    void* stream);

extern "C" int rns8_hip_direct_ring_gemm_i8_grouped_task_prefix_device(
    const int8_t* const* d_a_ptrs,
    const int8_t* const* d_b_ptrs,
    int8_t* const* d_c_ptrs,
    int task_count,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_i64_native_prefix9_device(
    const int64_t* d_a,
    const int64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_i64_native_prefix9_colpair_device(
    const int64_t* d_a,
    const int64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_i64_native_a_resident_b_prefix9_device(
    const int64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_i64_uniform_small_native_a_resident_b_prefix9_device(
    const int64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_u64_native_prefix9_device(
    const uint64_t* d_a,
    const uint64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_u64_native_prefix9_colpair_device(
    const uint64_t* d_a,
    const uint64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_u64_native_a_resident_b_prefix9_device(
    const uint64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_u64_native_a_resident_b_prefix9_colpair_device(
    const uint64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_u64_resident_a_native_b_prefix9_colpair_device(
    const int8_t* d_a,
    const uint64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_u64_uniform_small_native_a_resident_b_prefix9_device(
    const uint64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_uniform_small_i8_ab_resident_b_prefix9_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_uniform_small_i8_ab_resident_b_prefix9_colpair_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_finite_ring_gemm_i8_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int safe_k_block);

extern "C" int rns8_hip_direct_finite_ring_gemm_u8_native_device(
    const uint8_t* d_a,
    const uint8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int safe_k_block);

extern "C" int rns8_hip_direct_finite_ring_gemm_u8_native_a_i8_b_device(
    const uint8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_i8_scheduled_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    const rns8_plan_tile_schedule_entry* d_schedule,
    const uint8_t* d_zero_a_rows,
    const uint8_t* d_zero_b_cols,
    int entry_count,
    int max_tile_row_blocks,
    int max_tile_col_blocks,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int modulus_index,
    int selected_prefix,
    int safe_k_block);

extern "C" int rns8_hip_direct_zero_scheduled_residue_tiles_device(
    int8_t* d_c,
    const rns8_plan_tile_schedule_entry* d_schedule,
    int entry_count,
    int max_tile_elements,
    int rows,
    int ldc);

extern "C" int rns8_hip_direct_export_u8_modulus_device(
    const int8_t* d_residues,
    uint8_t* d_dst,
    int rows,
    int cols,
    int ld,
    int modulus);

extern "C" int rns8_hip_direct_export_i64_device(
    const int8_t* d_residues,
    int64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status);

extern "C" int rns8_hip_direct_export_i64_scheduled_device(
    const int8_t* d_residues,
    int64_t* d_dst,
    const rns8_plan_tile_schedule_entry* d_schedule,
    const uint64_t* d_bounds,
    const uint8_t* d_zero_a_rows,
    const uint8_t* d_zero_b_cols,
    int entry_count,
    int max_tile_elements,
    int rows,
    int cols,
    int* d_status);

extern "C" int rns8_hip_direct_export_u64_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status);

extern "C" int rns8_hip_direct_export_u64_scheduled_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    const rns8_plan_tile_schedule_entry* d_schedule,
    const uint64_t* d_bounds,
    const uint8_t* d_zero_a_rows,
    const uint8_t* d_zero_b_cols,
    int entry_count,
    int max_tile_elements,
    int rows,
    int cols,
    int* d_status);

extern "C" int rns8_hip_direct_export_exact_wide_signed_limbs_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* d_status);

extern "C" int rns8_hip_direct_export_exact_wide_signed_grouped_limbs_device(
    const int8_t* const* d_residue_ptrs,
    uint64_t* d_dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    int limb_count);

extern "C" int rns8_hip_direct_export_exact_wide_unsigned_limbs_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* d_status);

extern "C" int rns8_hip_direct_export_exact_wide_unsigned_grouped_limbs_device(
    const int8_t* const* d_residue_ptrs,
    uint64_t* d_dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    int limb_count);
#endif

namespace rns8::detail {

#include "hip_backend_state_counters.inc"
#include "hip_backend_runtime_launch_helpers.inc"

#include "hip_backend_resources.inc"

#include "hip_backend_pack.inc"

#include "hip_backend_gemm.inc"

#include "hip_backend_export.inc"

rns8_status hip_direct_synchronize(int device_id) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = hipDeviceSynchronize();
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

}  // namespace rns8::detail
