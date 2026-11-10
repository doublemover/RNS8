#pragma once

#include <cstring>
#include <ostream>

#include "rns8/device_identity.h"

namespace rns8::tools {

struct DeviceIdentitySnapshot {
  rns8_runtime_device_identity value{};
  rns8_status status = RNS8_UNSUPPORTED_BACKEND;

  DeviceIdentitySnapshot(int ordinal, bool hip_available) {
    value.struct_size = sizeof(value);
    value.abi_version = RNS8_ABI_VERSION;
    value.runtime_ordinal = ordinal;
    value.native_pci_status = value.native_uuid_status = UINT32_MAX;
    if (hip_available)
      status = rns8_query_runtime_device_identity(ordinal, &value);
  }
};

inline bool same_device_identity(const DeviceIdentitySnapshot& before,
                                 const DeviceIdentitySnapshot& after) noexcept {
  constexpr uint32_t known = RNS8_DEVICE_IDENTITY_PCI_BDF | RNS8_DEVICE_IDENTITY_UUID;
  const auto flags = before.value.valid_fields;
  if (before.status != RNS8_SUCCESS || after.status != RNS8_SUCCESS || !flags || (flags & ~known) ||
      flags != after.value.valid_fields ||
      before.value.runtime_ordinal != after.value.runtime_ordinal)
    return false;
  if ((flags & RNS8_DEVICE_IDENTITY_PCI_BDF) &&
      std::memcmp(before.value.pci_bdf, after.value.pci_bdf, sizeof(before.value.pci_bdf)))
    return false;
  return !(flags & RNS8_DEVICE_IDENTITY_UUID) ||
         !std::memcmp(before.value.uuid, after.value.uuid, sizeof(before.value.uuid));
}

inline void print_device_identity(std::ostream& out, const DeviceIdentitySnapshot& snapshot) {
  const auto& identity = snapshot.value;
  out << "{\"schema\":\"rns8_runtime_device_identity_v1\",\"runtime_ordinal\":"
      << identity.runtime_ordinal
      << ",\"query_status_code\":" << static_cast<uint32_t>(snapshot.status)
      << ",\"valid_fields\":" << identity.valid_fields << ",\"pci_bdf\":";
  if (snapshot.status == RNS8_SUCCESS && (identity.valid_fields & RNS8_DEVICE_IDENTITY_PCI_BDF))
    out << '"' << identity.pci_bdf << '"';
  else
    out << "null";
  out << ",\"uuid\":";
  if (snapshot.status == RNS8_SUCCESS && (identity.valid_fields & RNS8_DEVICE_IDENTITY_UUID)) {
    constexpr char digits[] = "0123456789abcdef";
    out << '"';
    for (const auto byte : identity.uuid)
      out << digits[byte >> 4] << digits[byte & 15];
    out << '"';
  } else
    out << "null";
  out << ",\"native_pci_status\":" << identity.native_pci_status
      << ",\"native_uuid_status\":" << identity.native_uuid_status << '}';
}

class DeviceIdentityWitness {
public:
  DeviceIdentityWitness(int ordinal, bool hip_available)
      : ordinal_(ordinal), hip_available_(hip_available), before_(ordinal, hip_available),
        after_(ordinal, false) {}
  void finish() { after_ = DeviceIdentitySnapshot(ordinal_, hip_available_); }
  void print(std::ostream& out) const {
    const char* state = same_device_identity(before_, after_) ? "stable"
                        : before_.status == RNS8_SUCCESS && after_.status == RNS8_SUCCESS &&
                                before_.value.valid_fields && after_.value.valid_fields
                            ? "changed"
                            : "unavailable";
    out << "{\"schema\":\"rns8_device_identity_witness_v1\",\"binding_state\":\"" << state
        << "\",\"before\":";
    print_device_identity(out, before_);
    out << ",\"after\":";
    print_device_identity(out, after_);
    out << '}';
  }

private:
  int ordinal_;
  bool hip_available_;
  DeviceIdentitySnapshot before_;
  DeviceIdentitySnapshot after_;
};

} // namespace rns8::tools
