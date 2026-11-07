#include <catch2/catch_test_macros.hpp>
#include "rns8/dlpack.hpp"
#include <array>
#include <cstring>
namespace {
struct Fixture {
  std::array<uint64_t, 40> payload{};
  int64_t shape[2]{3, 5}, strides[2]{8, 1};
  unsigned deleted = 0;
  DLManagedTensorVersioned tensor{};
  rns8::dlpack::Binding binding;
  Fixture() {
    tensor.version = {1, 2};
    tensor.manager_ctx = &deleted;
    tensor.deleter = [](DLManagedTensorVersioned* t) { ++*static_cast<unsigned*>(t->manager_ctx); };
    auto& t = tensor.dl_tensor;
    t.data = payload.data() + 1;
    t.byte_offset = 8;
    t.device = {kDLROCM, 0};
    t.dtype = {kDLInt, 64, 1};
    t.ndim = 2;
    t.shape = shape;
    t.strides = strides;
    binding.allocation_base = payload.data();
    binding.allocation_bytes = sizeof(payload);
    binding.owner_identity = 42;
    binding.generation = 1;
    binding.producer_complete = true;
  }
};
} // namespace
TEST_CASE("DLPack native view preserves interior data offsets and padded row strides") {
  Fixture f;
  rns8_native_device_desc d{};
  REQUIRE(rns8::dlpack::describe(f.tensor, f.binding, d) == RNS8_SUCCESS);
  REQUIRE(d.byte_offset == 16);
  REQUIRE(d.ld == 8);
  REQUIRE(d.rows == 3);
  REQUIRE(d.cols == 5);
  f.tensor.dl_tensor.strides = nullptr;
  REQUIRE(rns8::dlpack::describe(f.tensor, f.binding, d) == RNS8_SUCCESS);
  REQUIRE(d.ld == 5);
  REQUIRE(f.deleted == 0);
}
TEST_CASE("DLPack rejects unsafe layouts flags versions and ownership before consumption") {
  for (unsigned mutation = 0; mutation < 15; ++mutation) {
    Fixture f;
    switch (mutation) {
    case 0:
      f.tensor.version.major = 2;
      break;
    case 1:
      f.tensor.version.minor = 3;
      break;
    case 2:
      f.tensor.flags = 8;
      break;
    case 3:
      f.tensor.dl_tensor.device.device_type = kDLCUDA;
      break;
    case 4:
      f.tensor.dl_tensor.dtype.bits = 32;
      break;
    case 5:
      f.tensor.dl_tensor.dtype.lanes = 2;
      break;
    case 6:
      f.strides[1] = 2;
      break;
    case 7:
      f.strides[0] = -8;
      break;
    case 8:
      f.shape[0] = INT64_MAX;
      break;
    case 9:
      f.tensor.dl_tensor.byte_offset = UINT64_MAX;
      break;
    case 10:
      f.binding.producer_complete = false;
      break;
    case 11:
      f.binding.generation = UINT64_MAX;
      break;
    case 12:
      f.binding.allocation_bytes = 24;
      break;
    case 13:
      f.tensor.flags = DLPACK_FLAG_BITMASK_READ_ONLY;
      f.binding.access = RNS8_NATIVE_DEVICE_WRITE;
      break;
    case 14:
      f.tensor.dl_tensor.ndim = 3;
      break;
    }
    rns8_native_device_desc d{};
    d.owner_identity = 19;
    REQUIRE(rns8::dlpack::describe(f.tensor, f.binding, d) != RNS8_SUCCESS);
    REQUIRE(d.owner_identity == 19);
    REQUIRE(f.deleted == 0);
  }
  Fixture f;
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
  rns8_native_device_lease* lease = nullptr;
  REQUIRE(rns8::dlpack::consume(ctx, &f.tensor, f.binding, &lease) == RNS8_UNSUPPORTED_BACKEND);
  REQUIRE(lease == nullptr);
  REQUIRE(f.deleted == 0);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("DLPack raw foreign device access dtype and flag values preserve caller outputs",
          "[dlpack][abi][enum]") {
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), UINT32_C(19)}) {
    for (bool foreign_device : {false, true}) {
      Fixture f;
      if (foreign_device) {
        static_assert(sizeof(f.tensor.dl_tensor.device.device_type) == sizeof(bits));
        std::memcpy(&f.tensor.dl_tensor.device.device_type, &bits, sizeof(bits));
      } else {
        static_assert(sizeof(f.binding.access) == sizeof(bits));
        std::memcpy(&f.binding.access, &bits, sizeof(bits));
      }
      rns8_native_device_desc output{};
      output.owner_identity = 19;
      output.rows = 23;
      CHECK(rns8::dlpack::describe(f.tensor, f.binding, output) == RNS8_INVALID_ARGUMENT);
      CHECK(output.owner_identity == 19);
      CHECK(output.rows == 23);
      CHECK(output.allocation_base == nullptr);
      CHECK(f.deleted == 0);
    }
  }
  for (uint8_t code : {uint8_t(2), uint8_t(128), uint8_t(255)}) {
    Fixture f;
    f.tensor.dl_tensor.dtype.code = code;
    rns8_native_device_desc output{};
    output.generation = 77;
    CHECK(rns8::dlpack::describe(f.tensor, f.binding, output) == RNS8_INVALID_ARGUMENT);
    CHECK(output.generation == 77);
    CHECK(f.deleted == 0);
  }
  for (uint64_t flags : {UINT64_C(4), UINT64_C(0x8000000000000000), UINT64_MAX}) {
    Fixture f;
    f.tensor.flags = flags;
    rns8_native_device_desc output{};
    output.generation = 77;
    CHECK(rns8::dlpack::describe(f.tensor, f.binding, output) == RNS8_INVALID_ARGUMENT);
    CHECK(output.generation == 77);
    CHECK(f.deleted == 0);
  }
  for (auto access : {RNS8_NATIVE_DEVICE_READ, RNS8_NATIVE_DEVICE_WRITE})
    for (uint8_t code : {uint8_t(kDLInt), uint8_t(kDLUInt)}) {
      Fixture f;
      f.binding.access = access;
      f.tensor.dl_tensor.dtype.code = code;
      rns8_native_device_desc output{};
      REQUIRE(rns8::dlpack::describe(f.tensor, f.binding, output) == RNS8_SUCCESS);
      CHECK(output.access == access);
      CHECK(output.element_type ==
            (code == kDLInt ? RNS8_NATIVE_DEVICE_I64 : RNS8_NATIVE_DEVICE_U64));
      CHECK(f.deleted == 0);
    }
}
