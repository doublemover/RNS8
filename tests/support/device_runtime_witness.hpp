#pragma once

// Separate native-Linux device-ASan candidate only. These inquiries do not
// launch kernels; the explicitly selected ordinary tests supply device work.
#include "host_runtime_inventory.hpp"
#include "tool_common/device_identity_json.hpp"
#include <array>
#include <cstring>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hip/hip_runtime_api.h>
#include <iterator>
#include <sstream>

namespace rns8::test {
// Selected HSA headers must reserve the vendor-extension enum range. Older
// narrow declarations cannot safely carry AMD attributes through C++.
static_assert(static_cast<uint32_t>(HSA_AGENT_INFO_LAST) >=
                  static_cast<uint32_t>(HSA_AMD_AGENT_INFO_UUID),
              "HSA enum declaration must cover the queried AMD extension attributes");
inline nlohmann::json device_runtime_snapshot() {
  const auto hip_check = [](hipError_t status) {
    if (status != hipSuccess)
      throw std::runtime_error("device witness HIP inquiry failed");
  };
  int count = 0, current = -1, runtime = 0, driver = 0;
  hip_check(hipGetDeviceCount(&count));
  hip_check(hipGetDevice(&current));
  if (count != 1 || current != 0)
    throw std::runtime_error("device witness requires one visible ordinal-zero GPU");
  hipDeviceProp_t properties{};
  hip_check(hipGetDeviceProperties(&properties, 0));
  hip_check(hipRuntimeGetVersion(&runtime));
  hip_check(hipDriverGetVersion(&driver));
  size_t free_bytes = 0, total_bytes = 0;
  hip_check(hipMemGetInfo(&free_bytes, &total_bytes));
  const rns8::tools::DeviceIdentitySnapshot identity(0, true);
  if (identity.status != RNS8_SUCCESS ||
      !(identity.value.valid_fields & RNS8_DEVICE_IDENTITY_PCI_BDF))
    throw std::runtime_error("device witness requires native HIP PCI identity");
  const auto terminated = [](const auto& text) {
    const auto* data = std::data(text);
    if (!std::memchr(data, '\0', std::size(text)))
      throw std::runtime_error("device witness native name not terminated");
    return std::string(data);
  };
  const auto architecture = terminated(properties.gcnArchName);

  struct HsaOwner {
    bool active = false;
    HsaOwner() {
      if (hsa_init() != HSA_STATUS_SUCCESS)
        throw std::runtime_error("device witness HSA initialization failed");
      active = true;
    }
    HsaOwner(const HsaOwner&) = delete;
    HsaOwner& operator=(const HsaOwner&) = delete;
    void close() {
      active = false; // A failed close is not retried during unwinding.
      if (hsa_shut_down() != HSA_STATUS_SUCCESS)
        throw std::runtime_error("device witness HSA release failed");
    }
    ~HsaOwner() {
      if (active)
        (void)hsa_shut_down(); // Preserve the original failed inquiry.
    }
  } hsa;
  struct Agent {
    unsigned visited = 0, gpus = 0;
    uint32_t bdf = 0, domain = 0;
    std::array<char, 64> name{};
    std::array<char, 21> uuid{};
  } agent;
  const auto visit = [](hsa_agent_t handle, void* opaque) noexcept -> hsa_status_t {
    auto& out = *static_cast<Agent*>(opaque);
    if (++out.visited > 1024)
      return HSA_STATUS_ERROR;
    hsa_device_type_t type{};
    auto status = hsa_agent_get_info(handle, HSA_AGENT_INFO_DEVICE, &type);
    if (status != HSA_STATUS_SUCCESS || type != HSA_DEVICE_TYPE_GPU)
      return status;
    if (++out.gpus != 1)
      return HSA_STATUS_ERROR;
    status = hsa_agent_get_info(handle, HSA_AGENT_INFO_NAME, out.name.data());
    if (status == HSA_STATUS_SUCCESS)
      status = hsa_agent_get_info(handle, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_UUID),
                                  out.uuid.data());
    if (status == HSA_STATUS_SUCCESS)
      status = hsa_agent_get_info(handle, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID),
                                  &out.bdf);
    if (status == HSA_STATUS_SUCCESS)
      status = hsa_agent_get_info(handle, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN),
                                  &out.domain);
    return status;
  };
  if (hsa_iterate_agents(visit, &agent) != HSA_STATUS_SUCCESS || agent.gpus != 1)
    throw std::runtime_error("device witness HSA GPU inventory unavailable or ambiguous");
  const auto agent_name = terminated(agent.name), uuid = terminated(agent.uuid);
  hsa.close();
  std::ostringstream encoded;
  rns8::tools::print_device_identity(encoded, identity);
  return {
      {"schema", "rns8_device_asan_runtime_snapshot_v1"},
      {"visible_hip_devices", count},
      {"current_hip_ordinal", current},
      {"runtime_version", runtime},
      {"driver_version", driver},
      {"architecture", architecture},
      {"runtime_device_identity", nlohmann::json::parse(encoded.str())},
      {"hsa_gpu_count", agent.gpus},
      {"hsa_uuid", uuid},
      {"hsa_name", agent_name},
      {"hsa_bdfid", agent.bdf},
      {"hsa_domain", agent.domain},
      {"free_device_bytes", free_bytes},
      {"total_device_bytes", total_bytes},
      {"hip_symbol_owner", host_symbol_owner(reinterpret_cast<const void*>(&hipGetDeviceCount))},
      {"hsa_symbol_owner", host_symbol_owner(reinterpret_cast<const void*>(&hsa_init))},
      {"scope", "single_visible_hsa_uuid_and_pci_join_to_observed_hip_ordinal_zero"}};
}
} // namespace rns8::test
