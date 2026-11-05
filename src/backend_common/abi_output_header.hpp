#pragma once
#include <rns8/status.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace rns8::detail {
// Read only the mandatory size prefix until the caller admits the full record.
// Reject unsupported extensions and misaligned records before typed access.
template <class T> bool exact_output_header(const T* output) noexcept {
  static_assert(std::is_standard_layout_v<T> && offsetof(T, struct_size) == 0);
  if (!output || reinterpret_cast<uintptr_t>(output) % alignof(T))
    return false;
  decltype(output->struct_size) size{};
  static_assert((sizeof(size) == 4 || sizeof(size) == 8) &&
                sizeof(decltype(output->abi_version)) == sizeof(uint32_t));
  std::memcpy(&size, output, sizeof(size));
  if (size != sizeof(T))
    return false;
  uint32_t version = 0;
  std::memcpy(&version, reinterpret_cast<const unsigned char*>(output) + offsetof(T, abi_version),
              sizeof(version));
  return version == RNS8_ABI_VERSION;
}
} // namespace rns8::detail
