#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rns8::detail::trace {

struct EtwProfileSample {
  uint64_t instruction_pointer = 0;
  uint32_t thread_id = 0, packed_count_flags_rank = 0;
};

// Native little-endian Windows SampledProfile v2. Never reinterpret a packed
// or unaligned payload as a C++ struct, and preserve the caller on rejection.
inline bool decode_etw_profile_sample(const void* bytes, size_t size, uint32_t version,
                                      uint32_t pointer_bytes, EtwProfileSample& output) noexcept {
  if (!bytes || version != 2 || (pointer_bytes != 4 && pointer_bytes != 8) ||
      size != pointer_bytes + 8)
    return false;
  EtwProfileSample value;
  const auto* raw = static_cast<const unsigned char*>(bytes);
  std::memcpy(&value.instruction_pointer, raw, pointer_bytes);
  std::memcpy(&value.thread_id, raw + pointer_bytes, 4);
  std::memcpy(&value.packed_count_flags_rank, raw + pointer_bytes + 4, 4);
  output = value;
  return true;
}

} // namespace rns8::detail::trace
