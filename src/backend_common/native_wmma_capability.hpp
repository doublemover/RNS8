#pragma once
#include <rns8/rns8.h>
#include <array>

namespace rns8::detail {
struct NativeWmmaDeviceCapability;
using NativeWmmaCapabilityProvider = rns8_status (*)(int, NativeWmmaDeviceCapability&);
struct NativeWmmaDeviceCapability {
  int device_id = -1;
  bool supported = false;
  std::array<char, 64> target{};
  // Process-local identity of the loaded provider that ran the compiled probe.
  // Successful unsupported observations may be cached; errors may not.
  NativeWmmaCapabilityProvider provider = nullptr;
};
} // namespace rns8::detail
