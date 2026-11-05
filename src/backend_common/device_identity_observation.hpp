#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "rns8/device_identity.h"

namespace rns8::detail {

struct NativeDeviceIdentityObservation {
  std::array<char, 32> pci_bdf{};
  std::array<uint8_t, 16> uuid{};
  uint32_t pci_status = UINT32_MAX;
  uint32_t uuid_status = UINT32_MAX;
};

inline int identity_hex_digit(char value) noexcept {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

inline bool canonical_pci_bdf(const std::array<char, 32>& source, char (&out)[32]) noexcept {
  if (source[4] != ':' || source[7] != ':' || source[10] != '.' ||
      source[11] < '0' || source[11] > '7' || source[12] != '\0') return false;
  constexpr char digits[] = "0123456789abcdef";
  for (size_t index = 0; index != 12; ++index) {
    if (index == 4 || index == 7 || index == 10) continue;
    const int digit = identity_hex_digit(source[index]);
    if (digit < 0) return false;
  }
  if (identity_hex_digit(source[8]) * 16 + identity_hex_digit(source[9]) > 31) return false;
  for (size_t index = 0; index != 12; ++index) {
    out[index] = index == 4 || index == 7 || index == 10 ?
        source[index] : digits[identity_hex_digit(source[index])];
  }
  out[12] = '\0';
  return true;
}

inline rns8_runtime_device_identity device_identity_observation(
    int ordinal, const NativeDeviceIdentityObservation& observed) noexcept {
  rns8_runtime_device_identity result{};
  result.struct_size = sizeof(result);
  result.abi_version = RNS8_ABI_VERSION;
  result.runtime_ordinal = ordinal;
  result.native_pci_status = observed.pci_status;
  result.native_uuid_status = observed.uuid_status;
  if (observed.pci_status == 0 && canonical_pci_bdf(observed.pci_bdf, result.pci_bdf))
    result.valid_fields |= RNS8_DEVICE_IDENTITY_PCI_BDF;
  bool nonzero = false;
  for (const auto byte : observed.uuid) nonzero = nonzero || byte != 0;
  if (observed.uuid_status == 0 && nonzero) {
    std::memcpy(result.uuid, observed.uuid.data(), sizeof(result.uuid));
    result.valid_fields |= RNS8_DEVICE_IDENTITY_UUID;
  }
  return result;
}

}  // namespace rns8::detail
