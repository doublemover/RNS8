#ifndef RNS8_CORE_MATRIX_CLEANUP_HPP
#define RNS8_CORE_MATRIX_CLEANUP_HPP
#include "core/internal.hpp"

namespace rns8::detail {
// Commit ownership changes only after the backend acknowledges a release.
// A retry visits remaining allocations, never an already released address.
template<class Release>
rns8_status release_matrix_allocations(rns8_matrix& matrix, Release&& release) {
  rns8_status result = RNS8_SUCCESS;
  const auto slot = [&](void*& pointer, std::size_t& bytes) {
    if (!pointer) { bytes = 0; return; }
    const auto status = release(matrix.hip_device_id, pointer);
    if (status == RNS8_SUCCESS) { pointer = nullptr; bytes = 0; }
    else if (result == RNS8_SUCCESS) result = status;
  };
  slot(matrix.hip_upload_buffer, matrix.hip_upload_bytes);
  slot(matrix.hip_status_buffer, matrix.hip_status_bytes);
  slot(matrix.hip_export_buffer, matrix.hip_export_bytes);
  slot(matrix.hip_export_tile_bounds, matrix.hip_export_tile_bounds_bytes);
  slot(matrix.hip_export_tile_schedule, matrix.hip_export_tile_schedule_bytes);
  slot(matrix.hip_export_zero_a_rows, matrix.hip_export_zero_a_rows_bytes);
  slot(matrix.hip_export_zero_b_cols, matrix.hip_export_zero_b_cols_bytes);
  slot(matrix.hip_residues, matrix.hip_residue_bytes);
  slot(matrix.hip_byte_limbs, matrix.hip_byte_limb_bytes);
  slot(matrix.hip_native_i64, matrix.hip_native_i64_bytes);
  slot(matrix.hip_native_u64, matrix.hip_native_u64_bytes);
  if (!matrix.hip_export_tile_bounds) matrix.hip_export_tile_bounds_count = 0;
  if (!matrix.hip_export_tile_schedule) matrix.hip_export_tile_schedule_count = 0;
  matrix.hip_export_schedule_fingerprint = 0;
  matrix.hip_export_tile_max_elements = 0;
  return result;
}
} // namespace rns8::detail
#endif
