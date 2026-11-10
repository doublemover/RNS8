#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <memory>
#include <limits>
#include <sstream>

#include "backend_common/device_identity_observation.hpp"
#include "rns8/rns8.h"
#include "tool_common/device_identity_json.hpp"

TEST_CASE("native device identity observations validate fields before exposing them",
          "[cpu][identity]") {
  rns8::detail::NativeDeviceIdentityObservation observed;
  auto unknown = rns8::detail::device_identity_observation(0, observed);
  CHECK(unknown.valid_fields == 0);
  CHECK(unknown.pci_bdf[0] == 0);
  CHECK(unknown.uuid[0] == 0);
  std::memcpy(observed.pci_bdf.data(), "ABCD:12:1F.7", 13);
  observed.pci_status = 0;
  observed.uuid_status = 0;
  observed.uuid[15] = 0xff;
  const auto known = rns8::detail::device_identity_observation(4, observed);
  CHECK(known.runtime_ordinal == 4);
  CHECK(known.valid_fields == (RNS8_DEVICE_IDENTITY_PCI_BDF | RNS8_DEVICE_IDENTITY_UUID));
  CHECK(std::strcmp(known.pci_bdf, "abcd:12:1f.7") == 0);
  CHECK(known.uuid[15] == 0xff);
  for (const char* invalid : {"0000:00:20.0", "0000:00:00.8", "0000:GG:00.0", "0000:00:00.0x"}) {
    observed.pci_bdf.fill(0);
    std::memcpy(observed.pci_bdf.data(), invalid, std::strlen(invalid));
    const auto result = rns8::detail::device_identity_observation(0, observed);
    CHECK_FALSE(result.valid_fields & RNS8_DEVICE_IDENTITY_PCI_BDF);
    CHECK(result.pci_bdf[0] == 0);
  }
  observed.uuid_status = 1;
  CHECK_FALSE(rns8::detail::device_identity_observation(0, observed).valid_fields &
              RNS8_DEVICE_IDENTITY_UUID);
  observed.uuid_status = 0;
  observed.uuid.fill(0);
  CHECK_FALSE(rns8::detail::device_identity_observation(0, observed).valid_fields &
              RNS8_DEVICE_IDENTITY_UUID);
}

TEST_CASE("device identity preflight rejects without changing caller bytes", "[cpu][abi]") {
  CHECK(rns8_query_runtime_device_identity(0, nullptr) == RNS8_INVALID_ARGUMENT);
  rns8_runtime_device_identity output;
  std::memset(&output, 0xa5, sizeof(output));
  output.struct_size = sizeof(output);
  output.abi_version = RNS8_ABI_VERSION;
  for (unsigned mode = 0; mode != 4; ++mode) {
    auto value = output;
    int ordinal = 0;
    if (mode == 0)
      ordinal = -1;
    else if (mode == 1)
      value.struct_size -= 1;
    else if (mode == 2)
      value.struct_size += 8;
    else
      value.abi_version = UINT32_MAX;
    unsigned char before[sizeof(value)];
    std::memcpy(before, &value, sizeof(value));
    CHECK(rns8_query_runtime_device_identity(ordinal, &value) == RNS8_INVALID_ARGUMENT);
    CHECK(std::memcmp(before, &value, sizeof(value)) == 0);
  }
  struct alignas(rns8_runtime_device_identity) Header {
    uint64_t size;
  } header{sizeof(Header)};
  CHECK(rns8_query_runtime_device_identity(
            0, reinterpret_cast<rns8_runtime_device_identity*>(&header)) == RNS8_INVALID_ARGUMENT);
  CHECK(header.size == sizeof(Header));
}

TEST_CASE("identity serialization and witness comparison retain availability and raw bytes",
          "[cpu][identity]") {
  rns8::detail::NativeDeviceIdentityObservation observed;
  observed.pci_status = observed.uuid_status = 0;
  std::memcpy(observed.pci_bdf.data(), "0000:03:00.0", 13);
  observed.uuid[0] = 0xab;
  observed.uuid[15] = 0xcd;
  rns8::tools::DeviceIdentitySnapshot first(0, false), second(0, false);
  first.status = second.status = RNS8_SUCCESS;
  first.value = second.value = rns8::detail::device_identity_observation(0, observed);
  CHECK(rns8::tools::same_device_identity(first, second));
  std::ostringstream json;
  rns8::tools::print_device_identity(json, first);
  CHECK(json.str().find("\"uuid\":\"ab0000000000000000000000000000cd\"") != std::string::npos);
  CHECK(json.str().find("\"pci_bdf\":\"0000:03:00.0\"") != std::string::npos);
  second.value.uuid[1] = 1;
  CHECK_FALSE(rns8::tools::same_device_identity(first, second));
  second = first;
  second.value.valid_fields = 0;
  CHECK_FALSE(rns8::tools::same_device_identity(first, second));
  second = first;
  second.value.valid_fields |= 8;
  CHECK_FALSE(rns8::tools::same_device_identity(first, second));
  second = first;
  second.value.runtime_ordinal = 1;
  CHECK_FALSE(rns8::tools::same_device_identity(first, second));
}

TEST_CASE("HIP identity inquiry reports observed availability on the selected runtime ordinal",
          "[hip][qualification]") {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
  rns8_context* raw = nullptr;
  const auto status = rns8_create_context(0, &options, &raw);
  if (status == RNS8_UNSUPPORTED_BACKEND || status == RNS8_UNSUPPORTED_ARCH ||
      status == RNS8_UNSUPPORTED_OS)
    SKIP("HIP direct target is unavailable");
  REQUIRE(status == RNS8_SUCCESS);
  std::unique_ptr<rns8_context, decltype(&rns8_destroy_context)> owner(raw, rns8_destroy_context);
  rns8_runtime_device_identity identity{};
  identity.struct_size = sizeof(identity);
  identity.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_query_runtime_device_identity(0, &identity) == RNS8_SUCCESS);
  CHECK(identity.runtime_ordinal == 0);
  CHECK((identity.valid_fields & ~(RNS8_DEVICE_IDENTITY_PCI_BDF | RNS8_DEVICE_IDENTITY_UUID)) == 0);
  CHECK(identity.reserved == 0);
  if (identity.valid_fields & RNS8_DEVICE_IDENTITY_PCI_BDF)
    CHECK(std::strlen(identity.pci_bdf) == 12);
  if (identity.valid_fields & RNS8_DEVICE_IDENTITY_UUID) {
    bool nonzero = false;
    for (const auto byte : identity.uuid)
      nonzero = nonzero || byte != 0;
    CHECK(nonzero);
  }
  unsigned char before[sizeof(identity)];
  std::memcpy(before, &identity, sizeof(identity));
  CHECK(rns8_query_runtime_device_identity(std::numeric_limits<int>::max(), &identity) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(std::memcmp(before, &identity, sizeof(identity)) == 0);
  // A rejected native query must not poison the next query's HIP error check.
  CHECK(rns8_query_runtime_device_identity(0, &identity) == RNS8_SUCCESS);
}
