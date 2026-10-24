#include "backend_common/instruction_roof_plan.hpp"
#include "backend_common/split_k_plan.hpp"
#include "backend_common/word_mod_plan.hpp"
#include "backend_common/address_domain.hpp"
#include "backend_common/operation_support.hpp"
#include "backend_common/balanced_crt.hpp"
#include "backend_common/bilinear_resources.hpp"
#include "backend_common/rank23_slp_resources.hpp"
#include "backend_common/garner_constants.hpp"
#include "backend_common/native_wmma_queue_plan.hpp"
#include "backend_common/complex_residue_plan.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include "backend_hip_direct/encoded_b_backend.hpp"
#include "backend_hip_direct/finite_verify_backend.hpp"
#include "backend_hip_direct/native_wmma_prepared.hpp"

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
extern "C" int rns8_hip_direct_lift_suffix_device(
    const int8_t*, int8_t*, int, int, int, bool, uint64_t, uint64_t, uint64_t, int*, bool);

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

extern "C" int rns8_hip_direct_pack_u8_grouped_modulus_device(
    const uint8_t* d_src,
    int8_t* const* d_residue_ptrs,
    int task_count,
    int64_t src_task_stride,
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

extern "C" int rns8_hip_direct_ring_gemm_i8_grouped_prefix_region_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int region_row_offset,
    int region_col_offset,
    int region_rows,
    int region_cols,
    int grouped_prefix,
    int safe_k_block);

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

extern "C" int rns8_hip_direct_ring_gemv_n1_i8_grouped_prefix_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemv_small_n_i8_grouped_prefix_device(
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

extern "C" int rns8_hip_direct_ring_gemm_i64_native_a_resident_b_prefix9_colpair_device(
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

extern "C" int rns8_hip_direct_ring_gemm_i64_resident_a_native_b_prefix9_colpair_device(
    const int8_t* d_a,
    const int64_t* d_b,
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

extern "C" int rns8_hip_direct_finite_ring_gemm_i8_region_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int region_row_offset,
    int region_col_offset,
    int region_rows,
    int region_cols,
    int modulus,
    uint32_t modulus_reciprocal,
    int safe_k_block);

extern "C" int rns8_hip_direct_finite_ring_gemm_i8_grouped_device(
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

extern "C" int rns8_hip_direct_ring_gemm_i8_grouped_active_schedule_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    const rns8_plan_tile_schedule_entry* d_active_schedule,
    const uint8_t* d_zero_a_rows,
    const uint8_t* d_zero_b_cols,
    int active_entry_count,
    int max_tile_row_blocks,
    int max_tile_col_blocks,
    int m,
    int k,
    int lda,
    int ldb,
    int ldc,
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

extern "C" int rns8_hip_direct_export_u8_grouped_modulus_device(
    const int8_t* const* d_residue_ptrs,
    uint8_t* d_dst,
    int task_count,
    int rows,
    int cols,
    int modulus);

extern "C" int rns8_hip_direct_export_i64_device(
    const int8_t* d_residues,
    int64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status);

extern "C" int rns8_hip_direct_export_i64_grouped_device(
    const int8_t* const* d_residue_ptrs,
    int64_t* d_dst,
    int task_count,
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

extern "C" int rns8_hip_direct_export_u64_grouped_device(
    const int8_t* const* d_residue_ptrs,
    uint64_t* d_dst,
    int task_count,
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

extern "C" int rns8_hip_direct_export_exact_wide_signed_tree_crt_limbs_device(
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

extern "C" int rns8_hip_direct_export_exact_wide_unsigned_tree_crt_limbs_device(
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
namespace {
std::atomic<hip_deferred_buffer_cleanup*> deferred_probe_buffers{nullptr};
// Caller has already established device-wide completion. Failed frees remain
// queued, including their allocation record; another device's nodes stay live.
rns8_status release_completed_probe_buffers(int device_id,hip_deferred_buffer_cleanup* node,bool completed) {
  rns8_status result=RNS8_SUCCESS;
  while(node) {
    auto* next=node->next;
    if(!completed || node->device_id!=device_id) hip_direct_defer_buffer_cleanup(node);
    else {
      const auto status=hip_direct_free(device_id,node->pointer);
      if(status==RNS8_SUCCESS) delete node;
      else {if(result==RNS8_SUCCESS) result=status;hip_direct_defer_buffer_cleanup(node);}
    }
    node=next;
  }
  return result;
}
}
void hip_direct_defer_buffer_cleanup(hip_deferred_buffer_cleanup* node) noexcept {
  auto* head=deferred_probe_buffers.load(std::memory_order_relaxed);
  do {node->next=head;} while(!deferred_probe_buffers.compare_exchange_weak(
      head,node,std::memory_order_release,std::memory_order_relaxed));
}
rns8_status hip_direct_collect_deferred_buffers(int device_id) {
  if(!deferred_probe_buffers.load(std::memory_order_acquire)) return RNS8_SUCCESS;
  return hip_direct_synchronize(device_id);
}

#include "hip_backend_state_counters.inc"
#include "hip_backend_runtime_launch_helpers.inc"

#include "hip_backend_resources.inc"

#include "hip_backend_pack.inc"

#include "hip_backend_gemm.inc"

#include "hip_backend_export.inc"
#include "hip_backend_lift.inc"
#include "hip_backend_split_k.inc"
#include "hip_backend_epilogue.inc"
#include "hip_backend_complex.inc"
#include "hip_backend_narrow.inc"
#include "hip_backend_int4.inc"
#include "hip_backend_native_wmma.inc"
#include "hip_backend_native_tiled.inc"
#include "hip_backend_wide_gemm.inc"
#include "hip_backend_native_byte.inc"
#include "hip_backend_native_queue.inc"
#include "hip_backend_native_split_k.inc"
#include "hip_backend_bounds.inc"
#include "hip_backend_ragged.inc"
#include "hip_backend_block_sparse.inc"
#include "hip_backend_wrap_radix.inc"
#include "hip_backend_gram.inc"
#include "hip_backend_fixed_conversion.inc"
#include "hip_backend_word_mod.inc"
#include "hip_backend_native_interop.inc"
#include "hip_backend_instruction_roof.inc"
#include "hip_backend_layout.inc"
#include "hip_backend_direct_crt.inc"
#include "hip_backend_balanced_crt.inc"
#include "hip_backend_strassen.inc"
#include "hip_backend_rank23_slp.inc"
#include "hip_backend_adaptive_native.inc"
#include "hip_backend_integrity.inc"
#include "hip_backend_finite_verify.inc"

rns8_status hip_direct_synchronize(int device_id) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  // Capture before the drain. Nodes queued by later concurrent submissions
  // cannot be freed using completion of this earlier device snapshot.
  auto* pending=deferred_probe_buffers.exchange(nullptr,std::memory_order_acquire);
  const hipError_t err = hipDeviceSynchronize();
  const auto cleanup=release_completed_probe_buffers(device_id,pending,err==hipSuccess);
  return err==hipSuccess?cleanup:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

}  // namespace rns8::detail
