#pragma once
#include "native_tile_bound.hpp"

namespace rns8::detail {
// Matrix instances are never recycled. No caller source-version or pointer-only
// identity can certify a retained native summary.
struct NativeSummaryIdentity {
  uint64_t instance = 0, generation = 0;
  const void* native = nullptr;
  int64_t rows = 0, cols = 0, ld = 0;
  int device = -1;
  NativeBoundMethod method = NativeBoundMethod::tile_max;
  uint32_t panel_k = 0;
  bool unsigned_inputs = false, row_tiles = false;
  bool operator==(const NativeSummaryIdentity& other) const noexcept {
    return instance && generation && native && instance == other.instance &&
        generation == other.generation && native == other.native &&
        rows == other.rows && cols == other.cols && ld == other.ld &&
        device == other.device && method == other.method && panel_k == other.panel_k &&
        unsigned_inputs == other.unsigned_inputs && row_tiles == other.row_tiles;
  }
};
} // namespace rns8::detail
