#pragma once
#include <cstdint>
#include "rns8/status.h"
#include "native_conversion_layout.hpp"
namespace rns8::detail {
struct NativeTiledResources {
  rns8_status status = RNS8_INVALID_ARGUMENT;
  uint32_t output_rows = 32, output_cols = 32, waves = 4;
  uint64_t raw_lds_bytes = 0, encoded_lds_bytes = 0, total_lds_bytes = 0;
  uint64_t raw_operand_scalars_per_panel = 0, conversions_per_panel = 0;
  uint32_t accumulator_i32_per_lane = 0, fragment_words_per_lane = 0;
  uint32_t live_native_words_per_producer = 0;
  // Source-level storage/work counts, not compiler VGPR allocation or traffic.
  bool measured_resources = false;
};
inline NativeTiledResources query_native_tiled_resources(uint32_t depth, uint32_t buffers,
                                                        uint64_t lds_budget,
                                                        NativeConversion conversion = NativeConversion::replicated,
                                                        uint32_t group = 1, uint32_t waves = 4) {
  NativeTiledResources out;
  if ((group != 1 && group != 2 && group != 4) || (waves != 1 && waves != 4) ||
      (conversion != NativeConversion::replicated && conversion != NativeConversion::cooperative &&
       conversion != NativeConversion::encoded_lds && conversion != NativeConversion::paired_columns) ||
      (conversion == NativeConversion::paired_columns && (waves!=1 || buffers || depth!=16 || group>2)) ||
      (waves == 1 && (buffers || depth != 16)) ||
      (conversion == NativeConversion::cooperative && buffers) ||
      (conversion == NativeConversion::encoded_lds && (buffers != 1 || waves != 4)) ||
      buffers > 2 || (depth != 16 && depth != 32) || (!buffers && depth != 16)) return out;
  // A uses a padded depth stride, B a padded 32-column stride. Native i64/u64
  // have identical eight-byte storage; encoded fragments remain in registers.
  out.waves = waves; out.output_rows = out.output_cols = waves == 1 ? 16 : 32;
  if(conversion==NativeConversion::paired_columns)out.output_cols=32;
  out.raw_operand_scalars_per_panel = uint64_t(out.output_rows+out.output_cols) * depth;
  out.conversions_per_panel = conversion == NativeConversion::encoded_lds ? out.raw_operand_scalars_per_panel * group :
      uint64_t(waves) * 2 * 16 * depth * group * (conversion == NativeConversion::cooperative ? 1 : 2);
  out.accumulator_i32_per_lane = 8 * group;
  out.fragment_words_per_lane = 8 * group;
  out.live_native_words_per_producer = conversion == NativeConversion::encoded_lds ? 1 : 2;
  if(conversion==NativeConversion::paired_columns) {
    out.conversions_per_panel=uint64_t(32)*3*depth*group;
    out.accumulator_i32_per_lane=16*group;out.fragment_words_per_lane=12*group;out.live_native_words_per_producer=3;
  }
  out.raw_lds_bytes = conversion == NativeConversion::encoded_lds ? 0 : uint64_t(buffers) * (32 * (depth + 1) + depth * 33) * 8;
  out.encoded_lds_bytes = conversion == NativeConversion::encoded_lds ? uint64_t(group) * 2 * 32 * native_encoded_stride(depth) * 4 : 0;
  out.total_lds_bytes = out.raw_lds_bytes + out.encoded_lds_bytes;
  out.status = out.total_lds_bytes > lds_budget ? RNS8_WORKSPACE_TOO_SMALL : RNS8_SUCCESS;
  return out;
}
}
