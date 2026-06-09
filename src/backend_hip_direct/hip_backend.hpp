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

struct hip_direct_grouped_gemm_task {
  rns8_matrix* a = nullptr;
  rns8_matrix* b = nullptr;
  rns8_matrix* c = nullptr;
  rns8_workspace* workspace = nullptr;
};

struct hip_direct_grouped_gemm_descriptor {
  const rns8_plan* plan = nullptr;
  const hip_direct_grouped_gemm_task* tasks = nullptr;
  uint32_t task_count = 0;
  rns8_semantics semantics = RNS8_BOUNDED_I64;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  uint32_t prefix = 0;
};

struct hip_direct_grouped_gemm_bucket {
  hip_direct_grouped_gemm_descriptor descriptor{};
  uint32_t task_offset = 0;
  uint32_t task_count = 0;
};

struct hip_direct_grouped_gemm_bucket_plan {
  const rns8_plan* plan = nullptr;
  const hip_direct_grouped_gemm_task* tasks = nullptr;
  uint32_t task_count = 0;
  uint32_t bucket_count = 0;
  rns8_semantics semantics = RNS8_BOUNDED_I64;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  uint32_t prefix = 0;
  bool same_shape_required = true;
  std::vector<hip_direct_grouped_gemm_bucket> buckets;
};

struct hip_direct_grouped_gemm_bucket_request {
  const rns8_plan* plan = nullptr;
  const hip_direct_grouped_gemm_task* tasks = nullptr;
  uint32_t task_offset = 0;
  uint32_t task_count = 0;
  rns8_semantics semantics = RNS8_BOUNDED_I64;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  uint32_t prefix = 0;
};

struct hip_direct_grouped_device_buffer {
  int device_id = -1;
  void* ptr = nullptr;
  std::size_t bytes = 0;

  hip_direct_grouped_device_buffer() = default;
  hip_direct_grouped_device_buffer(const hip_direct_grouped_device_buffer&) = delete;
  hip_direct_grouped_device_buffer& operator=(const hip_direct_grouped_device_buffer&) = delete;
  hip_direct_grouped_device_buffer(hip_direct_grouped_device_buffer&& other) noexcept;
  hip_direct_grouped_device_buffer& operator=(hip_direct_grouped_device_buffer&& other) noexcept;
  ~hip_direct_grouped_device_buffer();

  rns8_status allocate(int requested_device_id, std::size_t requested_bytes);
  rns8_status reset() noexcept;

 private:
  void move_from(hip_direct_grouped_device_buffer& other) noexcept;
};

struct hip_direct_grouped_device_resources {
  int device_id = -1;
  hip_direct_grouped_device_buffer a_slab;
  hip_direct_grouped_device_buffer b_slab;
  hip_direct_grouped_device_buffer c_slab;
  hip_direct_grouped_device_buffer status;
  hip_direct_grouped_device_buffer a_residue_ptrs;
  hip_direct_grouped_device_buffer b_residue_ptrs;
  hip_direct_grouped_device_buffer c_residue_ptrs;
  std::vector<rns8_matrix*> a_matrices;
  std::vector<rns8_matrix*> b_matrices;
  std::vector<rns8_matrix*> c_matrices;

  hip_direct_grouped_device_resources() = default;
  hip_direct_grouped_device_resources(const hip_direct_grouped_device_resources&) = delete;
  hip_direct_grouped_device_resources& operator=(const hip_direct_grouped_device_resources&) = delete;
  hip_direct_grouped_device_resources(hip_direct_grouped_device_resources&&) noexcept = default;
  hip_direct_grouped_device_resources& operator=(hip_direct_grouped_device_resources&&) noexcept = default;

  rns8_status reset() noexcept;
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
rns8_status hip_direct_copy_host_to_device_labeled(
    int device_id,
    const char* timing_label,
    void* dst,
    const void* src,
    std::size_t bytes);
rns8_status hip_direct_copy_device_to_device(int device_id, void* dst, const void* src, std::size_t bytes);
rns8_status hip_direct_ensure_upload_buffer(int device_id, std::size_t bytes, void** buffer, std::size_t* capacity);
rns8_status hip_direct_copy_compact_matrix_device_to_host(
    int device_id,
    const char* timing_label,
    void* dst,
    int64_t dst_ld,
    const void* src,
    int64_t rows,
    int64_t cols,
    std::size_t cell_bytes,
    bool default_padded_staging = true);
rns8_status hip_direct_build_same_shape_grouped_bucket_plan(
    const rns8_plan* plan,
    const hip_direct_grouped_gemm_task* tasks,
    uint32_t task_count,
    rns8_semantics semantics,
    int64_t m,
    int64_t n,
    int64_t k,
    uint32_t prefix,
    hip_direct_grouped_gemm_bucket_plan* out);
rns8_status hip_direct_build_bucketed_grouped_gemm_plan(
    const hip_direct_grouped_gemm_bucket_request* requests,
    uint32_t request_count,
    hip_direct_grouped_gemm_bucket_plan* out);
const hip_direct_grouped_gemm_descriptor* hip_direct_single_bucket_descriptor(
    const hip_direct_grouped_gemm_bucket_plan& bucket_plan);
rns8_status hip_direct_validate_grouped_gemm_descriptor_setup(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    int* out_device_id = nullptr);
rns8_status hip_direct_validate_grouped_gemm_descriptor_after_pack(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    uint64_t first_source_version);
rns8_status hip_direct_validate_grouped_gemm_descriptor_after_gemm(
    const hip_direct_grouped_gemm_descriptor& descriptor);
rns8_status hip_direct_allocate_grouped_task_device_resources(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    std::size_t a_slab_bytes,
    std::size_t b_slab_bytes,
    std::size_t c_slab_bytes,
    std::size_t status_bytes,
    hip_direct_grouped_device_resources* out);
rns8_status hip_direct_prepare_grouped_task_residue_pointers(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    int* out_device_id = nullptr);
rns8_status hip_direct_pack_grouped_i64_task_inputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    const int64_t* a_slab,
    const int64_t* b_slab,
    int64_t lda,
    int64_t ldb,
    uint64_t first_source_version);
rns8_status hip_direct_pack_grouped_u64_task_inputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    const uint64_t* a_slab,
    const uint64_t* b_slab,
    int64_t lda,
    int64_t ldb,
    uint64_t first_source_version);
rns8_status hip_direct_pack_grouped_finite_u8_task_inputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    const uint8_t* a_slab,
    const uint8_t* b_slab,
    int64_t lda,
    int64_t ldb,
    uint16_t modulus,
    uint64_t first_source_version);
rns8_status hip_direct_gemm_grouped_rns_task_outputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources);
rns8_status hip_direct_gemm_grouped_finite_u8_task_outputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint16_t modulus);
rns8_status hip_direct_export_grouped_i64_task_outputs_to_host(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint64_t bound,
    int64_t* dst,
    std::size_t dst_elements);
rns8_status hip_direct_export_grouped_u64_task_outputs_to_host(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint64_t bound,
    uint64_t* dst,
    std::size_t dst_elements);
rns8_status hip_direct_export_grouped_finite_u8_task_outputs_to_host(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint16_t modulus,
    uint8_t* dst,
    std::size_t dst_elements);
rns8_status hip_direct_export_grouped_exact_wide_task_outputs_to_host(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint32_t limb_count,
    bool signed_output,
    uint64_t* dst,
    std::size_t dst_limb_elements);
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
rns8_status hip_direct_pack_i64_grouped_matrices_device(
    int device_id,
    const int64_t* src_slab,
    void* device_src_slab,
    std::size_t device_src_slab_bytes,
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_residue_ptrs,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix,
    uint64_t first_source_version);
rns8_status hip_direct_pack_u64_grouped_matrices_device(
    int device_id,
    const uint64_t* src_slab,
    void* device_src_slab,
    std::size_t device_src_slab_bytes,
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_residue_ptrs,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix,
    uint64_t first_source_version);
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
rns8_status hip_direct_pack_finite_u8_grouped_matrices_device(
    int device_id,
    const uint8_t* src_slab,
    void* device_src_slab,
    std::size_t device_src_slab_bytes,
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_residue_ptrs,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint16_t modulus,
    uint64_t first_source_version);
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
rns8_status hip_direct_gemm_rns_region_device(
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
    int64_t region_row_offset,
    int64_t region_col_offset,
    int64_t region_rows,
    int64_t region_cols,
    uint32_t prefix);
rns8_status hip_direct_gemm_rns_grouped_exact_wide_matrices_device(
    int device_id,
    rns8_matrix* const* a_matrices,
    rns8_matrix* const* b_matrices,
    rns8_matrix* const* c_matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_a_residue_ptrs,
    const void* device_b_residue_ptrs,
    const void* device_c_residue_ptrs,
    int64_t m,
    int64_t n,
    int64_t k,
    uint32_t prefix);
rns8_status hip_direct_gemm_rns_grouped_matrices_device(
    int device_id,
    rns8_matrix* const* a_matrices,
    rns8_matrix* const* b_matrices,
    rns8_matrix* const* c_matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_a_residue_ptrs,
    const void* device_b_residue_ptrs,
    const void* device_c_residue_ptrs,
    int64_t m,
    int64_t n,
    int64_t k,
    uint32_t prefix);
rns8_status hip_direct_gemm_rns_matrix_launch_current_device_no_sync(
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    void* stream = nullptr);
rns8_status hip_direct_gemv_n1_rns_prefix9_matrix(
    int device_id,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t k,
    uint64_t source_version);
rns8_status hip_direct_gemv_small_n_rns_prefix9_matrix(
    int device_id,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    uint64_t source_version);
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
rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_colpair_device(
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
rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_colpair_matrix(
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
rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_device(
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
rns8_status hip_direct_gemm_i64_resident_a_native_b_prefix9_colpair_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_native,
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
rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version);
rns8_status hip_direct_gemm_i64_resident_a_native_b_prefix9_colpair_matrix(
    int device_id,
    const rns8_matrix* A,
    const void* device_b_native,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    uint64_t source_version);
rns8_status hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_matrix(
    int device_id,
    const rns8_matrix* A,
    const void* device_b_native,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
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
rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_device(
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
rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix(
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
    const void* zero_a_rows,
    const void* zero_b_cols,
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
rns8_status hip_direct_gemm_finite_u8_region_device(
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
    int64_t region_row_offset,
    int64_t region_col_offset,
    int64_t region_rows,
    int64_t region_cols,
    uint16_t modulus);
rns8_status hip_direct_gemm_finite_u8_grouped_matrices_device(
    int device_id,
    rns8_matrix* const* a_matrices,
    rns8_matrix* const* b_matrices,
    rns8_matrix* const* c_matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_a_residue_ptrs,
    const void* device_b_residue_ptrs,
    const void* device_c_residue_ptrs,
    int64_t m,
    int64_t n,
    int64_t k,
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
rns8_status hip_direct_export_i64_grouped_matrices_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    void* device_status,
    int64_t rows,
    int64_t cols,
    uint64_t bound);
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
    const void* zero_a_rows,
    const void* zero_b_cols,
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
rns8_status hip_direct_export_u64_grouped_matrices_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    void* device_status,
    int64_t rows,
    int64_t cols,
    uint64_t bound);
rns8_status hip_direct_export_finite_u8_grouped_matrices_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint16_t modulus);
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
    const void* zero_a_rows,
    const void* zero_b_cols,
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
rns8_status hip_direct_export_exact_wide_signed_tree_crt_limbs_device(
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
rns8_status hip_direct_export_exact_wide_signed_limbs_to_device(
    int device_id,
    const void* device_residues,
    void* device_dst,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint32_t limb_count);
rns8_status hip_direct_export_exact_wide_signed_matrix_limbs_to_device(
    rns8_matrix* matrix,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint32_t limb_count);
rns8_status hip_direct_prepare_exact_wide_grouped_matrix_residue_pointers(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    void* device_residue_ptrs,
    std::size_t device_residue_ptr_bytes,
    int* out_device_id,
    uint32_t* out_prefix);
rns8_status hip_direct_prepare_grouped_matrix_residue_pointers(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    void* device_residue_ptrs,
    std::size_t device_residue_ptr_bytes,
    int* out_device_id,
    uint32_t* out_prefix);
rns8_status hip_direct_export_exact_wide_signed_grouped_matrix_limbs_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    int64_t rows,
    int64_t cols,
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
rns8_status hip_direct_export_exact_wide_unsigned_tree_crt_limbs_device(
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
rns8_status hip_direct_export_exact_wide_unsigned_limbs_to_device(
    int device_id,
    const void* device_residues,
    void* device_dst,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint32_t limb_count);
rns8_status hip_direct_export_exact_wide_unsigned_matrix_limbs_to_device(
    rns8_matrix* matrix,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint32_t limb_count);
rns8_status hip_direct_export_exact_wide_unsigned_grouped_matrix_limbs_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint32_t limb_count);
rns8_status hip_direct_synchronize(int device_id);

}  // namespace rns8::detail

#endif
