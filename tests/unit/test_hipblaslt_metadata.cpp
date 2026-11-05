#include <algorithm>
#include <atomic>
#include <thread>
#include "backend_common/bounded_algorithm_directory.hpp"
#include "backend_hipblaslt/matmul_metadata_plan.hpp"
#include "backend_hipblaslt/hipblaslt_backend.hpp"
#include <catch2/catch_test_macros.hpp>
using namespace rns8::detail;
TEST_CASE("hipBLASLt algorithm observations retain physical identities and explicit overflow") {
  HipblasLtAlgorithmInventory inventory;
  HipblasLtMetadataKey key{0, 16, 32, 64, 32, 4096};
  inventory.record(key, 7);
  inventory.record(key, 7);
  inventory.record(key, 8);
  REQUIRE(inventory.size == 2);
  CHECK(inventory.entries[0].calls == 2);
  CHECK(inventory.entries[1].algorithm_index == 8);
  for (int index = 9; index < 23; ++index)
    inventory.record(key, index);
  REQUIRE(inventory.size == 16);
  CHECK(inventory.complete);
  inventory.record(key, 23);
  CHECK_FALSE(inventory.complete);
  CHECK(inventory.size == 16);
  inventory = {};
  inventory.record(key, -1);
  CHECK_FALSE(inventory.complete);
  CHECK(inventory.entries[0].algorithm_index == -1);
  inventory = {};
  inventory.record(key, 7);
  inventory.entries[0].calls = UINT64_MAX;
  inventory.record(key, 7);
  CHECK_FALSE(inventory.complete);
  CHECK(inventory.entries[0].calls == UINT64_MAX);
  hipblaslt_algorithm_observations_set_enabled(false);
  CHECK_FALSE(hipblaslt_algorithm_observations_snapshot().complete);
}
TEST_CASE("hipBLASLt operand retention policies are independent and reject unknown bits") {
  const auto original = hipblaslt_prepack_policy();
  struct Restore {
    HipblasLtPrepackPolicy policy;
    ~Restore() { (void)hipblaslt_prepack_set_policy(policy); }
  } restore{original};
  for (const auto policy : {HipblasLtPrepackPolicy::none, HipblasLtPrepackPolicy::a,
                            HipblasLtPrepackPolicy::b, HipblasLtPrepackPolicy::both}) {
    REQUIRE(hipblaslt_prepack_set_policy(policy) == RNS8_SUCCESS);
    CHECK(hipblaslt_prepack_policy() == policy);
    CHECK(hipblaslt_prepack_a_enabled(policy) ==
          (policy == HipblasLtPrepackPolicy::a || policy == HipblasLtPrepackPolicy::both));
    CHECK(hipblaslt_prepack_b_enabled(policy) ==
          (policy == HipblasLtPrepackPolicy::b || policy == HipblasLtPrepackPolicy::both));
  }
  CHECK(hipblaslt_prepack_set_policy(static_cast<HipblasLtPrepackPolicy>(4)) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(hipblaslt_prepack_policy() == HipblasLtPrepackPolicy::both);
}
TEST_CASE("hipBLASLt immutable call directory shares padded geometry across every residue plane") {
  for (unsigned prefix : {1u, 9u, 18u, 20u})
    for (int k : {1, 15, 16, 17, 65535, 65536, 65537, 131072, 131073}) {
      HipblasLtCallMetadataDirectory directory;
      unsigned builds = 0, hits = 0, blocks = 0;
      for (unsigned p = 0; p < prefix; ++p)
        for (int start = 0; start < k; start += 65536) {
          const int slab = std::min(65536, k - start), padded = (slab + 15) / 16 * 16;
          HipblasLtMetadataKey key{0, 16, 32, padded, 32, 1u << 20};
          if (directory.find(key) < 0) {
            const int slot = directory.free_slot();
            REQUIRE(slot >= 0);
            REQUIRE(directory.publish(slot, key));
            ++builds;
          } else
            ++hits;
          ++blocks;
        }
      const unsigned geometries = (k > 65536 && k % 65536 != 0) ? 2 : 1;
      CHECK(builds == geometries);
      CHECK(hits + builds == blocks);
      CHECK(builds <= 2);
    }
}

TEST_CASE(
    "bounded algorithm directory copies selections and isolates library device configurations") {
  using Directory =
      BoundedAlgorithmDirectory<HipblasLtMetadataKey, uint64_t, HipblasLtMetadataKeyHash, 2, 2>;
  Directory directory;
  HipblasLtMetadataKey key{0, 16, 32, 64, 32, 4096};
  key.library_version = 100100;
  key.runtime_version = 100100;
  key.driver_version = 101;
  key.context_configuration = 1;
  key.target_identity = 1100;
  directory.remember(key, 77);
  uint64_t selected = 0;
  REQUIRE(directory.find(key, selected));
  CHECK(selected == 77);
  for (unsigned field = 0; field < 5; ++field) {
    auto other = key;
    switch (field) {
    case 0:
      ++other.library_version;
      break;
    case 1:
      ++other.runtime_version;
      break;
    case 2:
      ++other.driver_version;
      break;
    case 3:
      ++other.context_configuration;
      break;
    case 4:
      ++other.target_identity;
      break;
    }
    CHECK_FALSE(directory.find(other, selected));
  }
  for (unsigned index = 0; index < 1000; ++index) {
    auto other = key;
    other.m = 16 * int(index + 1);
    directory.remember(other, index);
    CHECK(directory.size() <= Directory::capacity);
  }
  CHECK(selected == 77);
  directory.reset();
  CHECK(directory.size() == 0);
  CHECK_FALSE(directory.find(key, selected));
  CHECK(selected == 77);
  directory.remember(key, 88);
  REQUIRE(directory.find(key, selected));
  CHECK(selected == 88);
}

TEST_CASE("bounded algorithm directory supports independent concurrent callers and reset") {
  BoundedAlgorithmDirectory<HipblasLtMetadataKey, uint64_t, HipblasLtMetadataKeyHash> directory;
  std::atomic<bool> valid{true};
  std::array<std::thread, 4> workers;
  for (unsigned worker = 0; worker < workers.size(); ++worker)
    workers[worker] = std::thread([&, worker] {
      for (unsigned index = 0; index < 1000; ++index) {
        HipblasLtMetadataKey key{int(worker), 16, 32, 16 * int(index % 31 + 1), 32, 4096};
        key.context_configuration = worker + 1;
        const uint64_t value = uint64_t(worker) * 1000 + index % 31;
        directory.remember(key, value);
        uint64_t selected = 0;
        if (directory.find(key, selected) && selected != value)
          valid = false;
        if (index % 127 == 0)
          directory.reset();
      }
    });
  for (auto& worker : workers)
    worker.join();
  CHECK(valid);
  CHECK(directory.size() <= 256);
}
TEST_CASE(
    "hipBLASLt metadata identity includes all layout operation device and workspace attributes") {
  HipblasLtMetadataKey key{0, 16, 32, 65536, 32, 4096};
  HipblasLtCallMetadataDirectory d;
  REQUIRE(d.publish(0, key));
  CHECK(d.find(key) == 0);
  CHECK_FALSE(d.publish(0, key));
  for (unsigned field = 0; field < 7; ++field) {
    auto changed = key;
    switch (field) {
    case 0:
      ++changed.device_id;
      break;
    case 1:
      ++changed.m;
      break;
    case 2:
      ++changed.n;
      break;
    case 3:
      ++changed.k;
      break;
    case 4:
      ++changed.scratch_ld;
      break;
    case 5:
      ++changed.workspace_bytes;
      break;
    case 6:
      ++changed.attribute_abi;
      break;
    }
    CHECK_FALSE(changed == key);
    CHECK(d.find(changed) == -1);
  }
  CHECK_FALSE(d.publish(2, key));
  auto bad = key;
  bad.attribute_abi = 2;
  CHECK_FALSE(d.publish(1, bad));
  CHECK(d.free_slot() == 1);
  bad = key;
  bad.device_id = -1;
  CHECK_FALSE(d.publish(1, bad));
  bad = key;
  bad.scratch_ld = 31;
  CHECK_FALSE(d.publish(1, bad));
  auto tail = key;
  tail.k = 16;
  REQUIRE(d.publish(1, tail));
  CHECK(d.free_slot() == -1);
  CHECK(d.find(key) == 0);
  CHECK(d.find(tail) == 1);
}
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
#include "backend_hipblaslt/hipblaslt_headers.hpp"
#include <cmath>
#if __has_include(<hipblaslt/hipblaslt_e5m3.h>)
TEST_CASE("hipBLASLt host E5M3 decodes every positive subnormal exactly", "[hipblaslt][headers]") {
  for (uint8_t bits = 1; bits < 8; ++bits) {
    hipblaslt_e5m3 value;
    value.data = bits;
    CHECK(float(value) == std::ldexp(float(bits), -17));
  }
}
#endif
#if defined(_MSC_VER) && !defined(__clang__)
TEST_CASE("hipBLASLt MSVC leading-zero adaptation covers unsigned boundaries",
          "[hipblaslt][headers]") {
  CHECK(hipblaslt_host_clz(0) == 32);
  for (unsigned bit = 0; bit < 32; ++bit)
    CHECK(hipblaslt_host_clz(1u << bit) == 31 - int(bit));
  CHECK(hipblaslt_host_clz(UINT32_MAX) == 0);
}
#endif
#endif
