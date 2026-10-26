#pragma once
#include <cstdint>
#include <type_traits>
namespace rns8::detail::distributed {
// Fixed wire record shared by HIP indexing and the host plan lowering. Bounds
// are unsigned magnitudes; signed output interpretation is a separate argument.
struct DeviceTile {
  uint64_t row = 0, col = 0, rows = 0, cols = 0;
  uint64_t bound[3]{};
  uint32_t prefix = 0, reserved = 0;
};
static_assert(std::is_trivially_copyable_v<DeviceTile> && sizeof(DeviceTile) == 64);
} // namespace rns8::detail::distributed
