#pragma once
#include <cstdint>
#include "rns8/status.h"
namespace rns8::detail {
struct NativeTiledResources {
  rns8_status status = RNS8_INVALID_ARGUMENT;
  uint32_t output_rows = 32, output_cols = 32, waves = 4;
  uint64_t raw_lds_bytes = 0;
};
inline NativeTiledResources query_native_tiled_resources(uint32_t depth, uint32_t buffers,
                                                        uint64_t lds_budget) {
  NativeTiledResources out;
  if (buffers > 2 || (depth != 16 && depth != 32) || (!buffers && depth != 16)) return out;
  // A uses a padded depth stride, B a padded 32-column stride. Native i64/u64
  // have identical eight-byte storage; encoded fragments remain in registers.
  out.raw_lds_bytes = uint64_t(buffers) * (32 * (depth + 1) + depth * 33) * 8;
  out.status = out.raw_lds_bytes > lds_budget ? RNS8_WORKSPACE_TOO_SMALL : RNS8_SUCCESS;
  return out;
}
}
