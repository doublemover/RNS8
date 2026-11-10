#pragma once

#include "rns8/status.h"
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>

namespace rns8::detail {
struct HostUploadRegion {
  size_t offset;
  const void* source;
  size_t bytes;
};

// Preserve explicit device offsets, including zeroed alignment gaps, while
// giving several preparation sources one immutable owned DMA payload.
inline rns8_status make_host_upload_packet(size_t extent,
                                           std::initializer_list<HostUploadRegion> regions,
                                           std::shared_ptr<void>& output) noexcept {
  if (!extent || regions.size() == 0 || regions.size() > 16)
    return RNS8_INVALID_ARGUMENT;
  if (extent > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
    return RNS8_RANGE_ERROR;
  bool populated = false;
  for (auto first = regions.begin(); first != regions.end(); ++first) {
    if (first->offset > extent || first->bytes > extent - first->offset ||
        (first->bytes && !first->source))
      return RNS8_INVALID_ARGUMENT;
    if (!first->bytes)
      continue;
    populated = true;
    for (auto second = regions.begin(); second != first; ++second)
      if (second->bytes && first->offset < second->offset + second->bytes &&
          second->offset < first->offset + first->bytes)
        return RNS8_INVALID_ARGUMENT;
  }
  if (!populated)
    return RNS8_INVALID_ARGUMENT;
  try {
    std::shared_ptr<unsigned char[]> packet(new unsigned char[extent]());
    for (const auto& region : regions)
      if (region.bytes)
        std::memcpy(packet.get() + region.offset, region.source, region.bytes);
    output = {packet, packet.get()};
    return RNS8_SUCCESS;
  } catch (const std::bad_alloc&) {
    return RNS8_INTERNAL_ERROR;
  }
}
} // namespace rns8::detail
