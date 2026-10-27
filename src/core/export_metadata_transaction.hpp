#ifndef RNS8_CORE_EXPORT_METADATA_TRANSACTION_HPP
#define RNS8_CORE_EXPORT_METADATA_TRANSACTION_HPP

#include "core/internal.hpp"

namespace rns8::detail {
// In-place cache replacement is not rollback: invalidate its certificate before
// the first free, resize or upload. Resident value identity is independent.
template<class Upload>
rns8_status replace_export_metadata(rns8_matrix& matrix, uint64_t fingerprint,
                                   uint64_t count, uint64_t max_elements, Upload&& upload) {
  matrix.hip_export_schedule_fingerprint = 0;
  matrix.hip_export_tile_schedule_count = 0;
  matrix.hip_export_tile_bounds_count = 0;
  matrix.hip_export_tile_max_elements = 0;
  const auto status = upload();
  if (status != RNS8_SUCCESS) return status;
  matrix.hip_export_tile_schedule_count = count;
  matrix.hip_export_tile_bounds_count = count;
  matrix.hip_export_tile_max_elements = max_elements;
  matrix.hip_export_schedule_fingerprint = fingerprint;
  return RNS8_SUCCESS;
}
} // namespace rns8::detail
#endif
