#ifndef RNS8_HOST_UPLOAD_HPP
#define RNS8_HOST_UPLOAD_HPP

#include "backend_common/address_domain.hpp"

namespace rns8::detail {

struct HostUploadPlan {
  std::size_t rows = 0;
  std::size_t row_bytes = 0;
  std::size_t source_pitch_bytes = 0;
  std::size_t source_span_bytes = 0;
  std::size_t device_bytes = 0;
  int64_t device_ld = 0;

  bool contiguous() const { return source_pitch_bytes == row_bytes; }
};

// The last row has no accessible trailing padding. Device storage is always
// compact, so consumers must use device_ld rather than the caller's pitch.
inline bool make_host_upload_plan(int64_t rows, int64_t cols, int64_t ld, std::size_t element_bytes,
                                  HostUploadPlan& out) {
  AddressDomain source, compact;
  if (rows <= 0 || cols <= 0 || ld < cols ||
      !checked_host_plane_address(rows, cols, ld, 1, element_bytes, source) ||
      !checked_host_plane_address(rows, cols, cols, 1, element_bytes, compact))
    return false;
  const auto width = static_cast<std::size_t>(cols) * element_bytes;
  // An unused single-row pitch need not even have a representable byte value.
  const auto pitch = rows == 1 ? width : static_cast<std::size_t>(ld) * element_bytes;
  out = {static_cast<std::size_t>(rows),
         width,
         pitch,
         static_cast<std::size_t>(source.byte_extent),
         static_cast<std::size_t>(compact.byte_extent),
         cols};
  return true;
}

} // namespace rns8::detail
#endif
