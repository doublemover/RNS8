#ifndef RNS8_BACKEND_HIP_DIRECT_HPP
#define RNS8_BACKEND_HIP_DIRECT_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rns8/rns8.h"

namespace rns8::detail {

struct hip_direct_timing_sample {
  std::string label;
  double microseconds = 0.0;
};

struct hip_direct_allocation_counters {
  uint64_t allocate_calls = 0;
  uint64_t free_calls = 0;
  uint64_t allocated_bytes = 0;
};

bool hip_direct_compiled();
void hip_direct_timing_set_enabled(bool enabled);
bool hip_direct_timing_enabled();
void hip_direct_timing_reset();
void hip_direct_timing_flush_pending_events();
void hip_direct_timing_record_sample(const char* label, double microseconds);
void hip_direct_timing_record_pending_event(const char* label, void* start_event, void* stop_event);
void hip_direct_timing_record_pending_event_with_alias(
    const char* label,
    const char* alias,
    void* start_event,
    void* stop_event);
std::vector<hip_direct_timing_sample> hip_direct_timing_snapshot();
void hip_direct_allocation_counters_reset();
hip_direct_allocation_counters hip_direct_allocation_counters_snapshot();
rns8_status hip_direct_probe(int device_id, rns8_device_info& out);
rns8_status hip_direct_allocate(int device_id, std::size_t bytes, void** out);
rns8_status hip_direct_free(int device_id, void* ptr);
rns8_status hip_direct_zero(int device_id, void* ptr, std::size_t bytes);
rns8_status hip_direct_copy_device_to_host(int device_id, void* dst, const void* src, std::size_t bytes);
rns8_status hip_direct_copy_host_to_device(int device_id, void* dst, const void* src, std::size_t bytes);
rns8_status hip_direct_ensure_upload_buffer(int device_id, std::size_t bytes, void** buffer, std::size_t* capacity);
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
rns8_status hip_direct_native_i64_to_rns_device(
    int device_id,
    const void* device_native,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    uint32_t prefix);
rns8_status hip_direct_native_u64_to_rns_device(
    int device_id,
    const void* device_native,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    uint32_t prefix);
rns8_status hip_direct_pack_finite_u8_device(
    int device_id,
    const uint8_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint16_t modulus);
rns8_status hip_direct_ring_gemm_i8_device(
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
    uint16_t modulus,
    uint32_t modulus_index,
    uint32_t selected_prefix);
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
rns8_status hip_direct_gemm_i64_native_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_i64_native_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_i64_uniform_small_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version);
rns8_status hip_direct_gemm_i64_uniform_small_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version);
rns8_status hip_direct_gemm_u64_native_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_u64_native_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_u64_uniform_small_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version);
rns8_status hip_direct_gemm_u64_uniform_small_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version);
rns8_status hip_direct_gemm_uniform_small_i8_ab_resident_b_prefix9_device(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_uniform_small_i8_ab_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    uint64_t source_version);
rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_device(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    uint64_t source_version);
rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_device(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    uint64_t source_version);
rns8_status hip_direct_gemm_rns_tiled_device(
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
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count);
rns8_status hip_direct_gemm_rns_tiled_device_schedule(
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
    const rns8_plan_tile_schedule_entry* host_entries,
    const void* device_entries,
    const void* active_device_entries,
    const uint64_t* active_offsets,
    const uint64_t* active_counts,
    uint32_t active_prefix_count,
    uint64_t entry_count);
rns8_status hip_direct_gemm_finite_u8_resident_device(
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
rns8_status hip_direct_gemm_finite_u8_native_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus);
rns8_status hip_direct_gemm_finite_u8_native_a_resident_b_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus);
rns8_status hip_direct_gemm_finite_u8_native_a_resident_b_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint16_t modulus,
    uint64_t source_version);
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
rns8_status hip_direct_export_i64_tiled_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    const void* device_entries,
    const void* device_bounds,
    uint64_t entry_count,
    uint64_t max_tile_elements,
    bool all_zero_output_tiles,
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
rns8_status hip_direct_export_u64_tiled_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    const void* device_entries,
    const void* device_bounds,
    uint64_t entry_count,
    uint64_t max_tile_elements,
    bool all_zero_output_tiles,
    uint64_t* dst,
    int64_t ld);
rns8_status hip_direct_export_finite_u8_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    int64_t rows,
    int64_t cols,
    uint16_t modulus,
    uint8_t* dst,
    int64_t ld);
rns8_status hip_direct_export_exact_wide_signed_limbs_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count);
rns8_status hip_direct_export_exact_wide_unsigned_limbs_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count);
rns8_status hip_direct_synchronize(int device_id);

}  // namespace rns8::detail

#endif
