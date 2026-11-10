#include <cstring>
#include "backend_common/native_device_contract.hpp"
#include "../support/native_device_fixture.hpp"

TEST_CASE("native owner callbacks admit every defined status and reject raw unknown statuses",
          "[api][abi][enum][native-owner][host]") {
  using namespace native_device_fixture;
  alignas(8) uint64_t words[32]{};
  Owner owner;
  const auto desc = descriptor(words, sizeof(words), owner);
  for (auto status : {RNS8_INVALID_ARGUMENT, RNS8_UNSUPPORTED_OS, RNS8_UNSUPPORTED_ARCH,
                      RNS8_UNSUPPORTED_BACKEND, RNS8_RANGE_ERROR, RNS8_ACCUMULATION_OVERFLOW_RISK,
                      RNS8_WORKSPACE_TOO_SMALL, RNS8_BACKEND_FAILURE, RNS8_VERIFICATION_FAILED,
                      RNS8_INTERNAL_ERROR, RNS8_BUSY, RNS8_CANCELED, RNS8_TIMEOUT}) {
    owner.status = status;
    CHECK(rns8::detail::public_status_valid(status));
    CHECK(rns8::detail::acquire_native_owner(desc) == status);
    CHECK(owner.acquired == 0);
    CHECK(owner.released == 0);
  }
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 14u}) {
    std::memcpy(&owner.status, &bits, sizeof(bits));
    CHECK_FALSE(rns8::detail::public_status_valid(owner.status));
    CHECK(rns8::detail::acquire_native_owner(desc) == RNS8_INVALID_ARGUMENT);
    CHECK(owner.acquired == 0);
    CHECK(owner.released == 0);
  }
  owner.status = RNS8_SUCCESS;
  CHECK(rns8::detail::acquire_native_owner(desc) == RNS8_SUCCESS);
  CHECK(owner.acquired == 1);
  release(&owner);
  CHECK(owner.released == 1);
}
TEST_CASE("native device query checks view spans without touching device or owner") {
  using namespace native_device_fixture;
  alignas(8) uint64_t words[32]{};
  Owner owner;
  const auto valid = descriptor(words, sizeof(words), owner);
  rns8_native_device_resources r{};
  r.struct_size = sizeof(r);
  r.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_query_native_device(&valid, &r) == RNS8_SUCCESS);
  REQUIRE(r.view_span_bytes == 19 * 8);
  REQUIRE(r.compact_native_bytes == 15 * 8);
  REQUIRE(r.export_scratch_bytes == 15 * 8 + 4);
  REQUIRE(r.verified_import_scratch_bytes == 8);
  for (unsigned bad = 0; bad < 12; ++bad) {
    auto d = valid;
    switch (bad) {
    case 0:
      d.byte_offset = 1;
      break;
    case 1:
      d.allocation_bytes = 19 * 8;
      break;
    case 2:
      d.rows = INT64_MAX;
      break;
    case 3:
      d.ld = -1;
      break;
    case 4:
      d.producer_complete = 0;
      break;
    case 5:
      d.owner_identity = 0;
      break;
    case 6:
      d.generation = 0;
      break;
    case 7:
      d.acquire = nullptr;
      break;
    case 8:
      d.release = nullptr;
      break;
    case 9:
      d.access = static_cast<rns8_native_device_access>(3);
      break;
    case 10:
      d.element_type = static_cast<rns8_native_device_type>(0);
      break;
    case 11:
      d.allocation_bytes = UINT64_MAX;
      break;
    }
    const auto previous = r;
    REQUIRE(rns8_query_native_device(&d, &r) != RNS8_SUCCESS);
    REQUIRE(r.view_span_bytes == previous.view_span_bytes);
    REQUIRE(r.export_scratch_bytes == previous.export_scratch_bytes);
  }
  REQUIRE(owner.acquired == 0);
  REQUIRE(owner.released == 0);
  auto* ctx = create_cpu_context();
  rns8_native_device_lease* lease = nullptr;
  REQUIRE(rns8_create_native_device_lease(ctx, &valid, &lease) == RNS8_UNSUPPORTED_BACKEND);
  REQUIRE(lease == nullptr);
  REQUIRE(owner.acquired == 0);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("copy-owned native leases require a read snapshot budget before owner acquisition") {
  using namespace native_device_fixture;
  alignas(8) uint64_t words[32]{};
  Owner owner;
  auto d = descriptor(words, sizeof(words), owner);
  rns8_native_device_resources resources{};
  resources.struct_size = sizeof(resources);
  resources.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_query_native_device(&d, &resources) == RNS8_SUCCESS);
  REQUIRE(resources.snapshot_bytes == 0);
  d.policy = RNS8_NATIVE_DEVICE_COPY_OWNED;
  REQUIRE(rns8_query_native_device(&d, &resources) == RNS8_WORKSPACE_TOO_SMALL);
  d.snapshot_budget_bytes = 119;
  REQUIRE(rns8_query_native_device(&d, &resources) == RNS8_WORKSPACE_TOO_SMALL);
  d.snapshot_budget_bytes = 120;
  REQUIRE(rns8_query_native_device(&d, &resources) == RNS8_SUCCESS);
  REQUIRE(resources.snapshot_bytes == 120);
  REQUIRE(resources.view_span_bytes == 152);
  d.access = RNS8_NATIVE_DEVICE_WRITE;
  REQUIRE(rns8_query_native_device(&d, &resources) == RNS8_INVALID_ARGUMENT);
  d.access = RNS8_NATIVE_DEVICE_READ;
  d.policy = RNS8_NATIVE_DEVICE_BORROW;
  REQUIRE(rns8_query_native_device(&d, &resources) == RNS8_INVALID_ARGUMENT);
  REQUIRE(owner.acquired == 0);
  REQUIRE(owner.released == 0);
}

TEST_CASE("native limb descriptors bind integer widths and reject unsupported ownership") {
  using namespace native_device_fixture;
  alignas(8) uint64_t words[2048]{};
  Owner owner;
  for (auto type : {RNS8_NATIVE_DEVICE_SIGNED_LIMBS, RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS})
    for (uint32_t width : {1u, 2u, 3u, 32u}) {
      auto d = descriptor(words, sizeof(words), owner);
      d.element_type = type;
      d.limb_count = width;
      d.access = RNS8_NATIVE_DEVICE_WRITE;
      rns8_native_device_resources r{};
      r.struct_size = sizeof(r);
      r.abi_version = RNS8_ABI_VERSION;
      REQUIRE(rns8_query_native_device(&d, &r) == RNS8_SUCCESS);
      CHECK(r.view_span_bytes == 19 * 8 * width);
      CHECK(r.export_scratch_bytes == 15 * 8 * width + 4);
      CHECK(r.verified_import_scratch_bytes == 0);
      for (unsigned bad = 0; bad < 7; ++bad) {
        auto invalid = d;
        switch (bad) {
        case 0:
          invalid.limb_count = 0;
          break;
        case 1:
          invalid.limb_count = 33;
          break;
        case 2:
          invalid.access = RNS8_NATIVE_DEVICE_READ;
          break;
        case 3:
          invalid.policy = RNS8_NATIVE_DEVICE_COPY_OWNED;
          break;
        case 4:
          invalid.element_type = RNS8_NATIVE_DEVICE_I64;
          break;
        case 5:
          invalid.allocation_bytes = d.byte_offset + r.view_span_bytes - 1;
          break;
        case 6:
          invalid.ld = INT64_MAX;
          break;
        }
        CHECK(rns8_query_native_device(&invalid, &r) != RNS8_SUCCESS);
      }
    }
  CHECK(owner.acquired == 0);
  CHECK(owner.released == 0);
}

TEST_CASE("raw C enum representations are rejected before native owner acquisition") {
  using namespace native_device_fixture;
  uint64_t words[32]{};
  Owner owner;
  const auto valid = descriptor(words, sizeof(words), owner);
  auto* ctx = create_cpu_context();
  const auto references = ctx->prepared_references;
  const auto allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  for (unsigned field = 0; field < 3; ++field)
    for (uint32_t raw : {UINT32_MAX, UINT32_C(0x80000000), field == 0 ? 5u : 3u}) {
      auto d = valid;
      if (field == 0)
        std::memcpy(&d.element_type, &raw, sizeof(raw));
      else if (field == 1)
        std::memcpy(&d.access, &raw, sizeof(raw));
      else
        std::memcpy(&d.policy, &raw, sizeof(raw));
      rns8_native_device_resources out{};
      out.struct_size = sizeof(out);
      out.abi_version = RNS8_ABI_VERSION;
      out.snapshot_bytes = 171;
      const auto before = out;
      CHECK(rns8_query_native_device(&d, &out) == RNS8_INVALID_ARGUMENT);
      CHECK(std::memcmp(&out, &before, sizeof(out)) == 0);
      for (bool poisoned : {false, true}) {
        ctx->hip_failure_poisoned = poisoned;
        auto* const canary = reinterpret_cast<rns8_native_device_lease*>(uintptr_t(1));
        auto* lease = canary;
        CHECK(rns8_create_native_device_lease(ctx, &d, &lease) == RNS8_INVALID_ARGUMENT);
        CHECK(lease == canary);
        CHECK(ctx->prepared_references == references);
        ctx->hip_failure_poisoned = false;
      }
    }
  CHECK(owner.acquired == 0);
  CHECK(owner.released == 0);
  const auto rejected = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(rejected.allocate_calls == allocations.allocate_calls);
  CHECK(rejected.h2d_calls == allocations.h2d_calls);
  CHECK(rejected.d2h_calls == allocations.d2h_calls);
  auto* const canary = reinterpret_cast<rns8_native_device_lease*>(uintptr_t(1));
  auto* lease = canary;
  CHECK(rns8_create_native_device_lease(ctx, &valid, &lease) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(lease == canary);
  CHECK(owner.acquired == 0);
  CHECK(owner.released == 0);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}
