#include "rns8/device_identity.h"

#include <cstring>
#include "backend_common/device_identity_observation.hpp"

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#include <hip/hip_runtime_api.h>
#endif

rns8_status rns8_query_runtime_device_identity(int ordinal, rns8_runtime_device_identity* out) {
  if (!out || ordinal < 0) return RNS8_INVALID_ARGUMENT;
  uint64_t supplied_size = 0;
  std::memcpy(&supplied_size, out, sizeof(supplied_size));
  if (supplied_size != sizeof(*out) || out->abi_version != RNS8_ABI_VERSION)
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  // Optional identity APIs can report unsupported status. Do not leave that
  // query error for the next kernel's launch check, or consume an earlier
  // pending error without reporting it.
  if (hipPeekAtLastError() != hipSuccess) return RNS8_BACKEND_FAILURE;
  hipDevice_t device{};
  const auto device_status = hipDeviceGet(&device, ordinal);
  if (device_status != hipSuccess) {
    const auto last = hipGetLastError();
    if (last != hipSuccess && last != device_status) return RNS8_BACKEND_FAILURE;
  }
  if (device_status == hipErrorInvalidDevice || device_status == hipErrorInvalidValue)
    return RNS8_INVALID_ARGUMENT;
  if (device_status == hipErrorNoDevice || device_status == hipErrorNotInitialized ||
      device_status == hipErrorInsufficientDriver) return RNS8_UNSUPPORTED_BACKEND;
  if (device_status != hipSuccess) return RNS8_BACKEND_FAILURE;
  rns8::detail::NativeDeviceIdentityObservation observed;
  observed.pci_status = static_cast<uint32_t>(hipDeviceGetPCIBusId(
      observed.pci_bdf.data(), static_cast<int>(observed.pci_bdf.size()), ordinal));
  hipUUID uuid{};
  observed.uuid_status = static_cast<uint32_t>(hipDeviceGetUuid(&uuid, device));
  const auto last = hipGetLastError();
  if (last != hipSuccess && static_cast<uint32_t>(last) != observed.pci_status &&
      static_cast<uint32_t>(last) != observed.uuid_status) return RNS8_BACKEND_FAILURE;
  const auto optional_status = [](uint32_t code) {
    return code == hipSuccess || code == hipErrorNotSupported ||
        code == hipErrorInvalidValue || code == hipErrorInvalidDevice;
  };
  if (!optional_status(observed.pci_status) || !optional_status(observed.uuid_status))
    return RNS8_BACKEND_FAILURE;
  static_assert(sizeof(uuid.bytes) == observed.uuid.size(), "HIP UUID extent");
  std::memcpy(observed.uuid.data(), uuid.bytes, observed.uuid.size());
  const auto result = rns8::detail::device_identity_observation(ordinal, observed);
  *out = result;
  return RNS8_SUCCESS;
#else
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
