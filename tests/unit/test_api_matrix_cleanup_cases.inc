#include "core/matrix_cleanup.hpp"
#include "core/export_metadata_transaction.hpp"
#include <array>

namespace {
struct MockDeviceBufferOperations {
  static inline void* fail_pointer = nullptr;
  static inline bool fail_all = false;
  static inline std::array<rns8::detail::hip_deferred_buffer_cleanup*, 8> deferred{};
  static inline unsigned deferred_count = 0;
  static rns8_status allocate(int, std::size_t bytes, void** out) {
    *out = new unsigned char[bytes]; return RNS8_SUCCESS;
  }
  static rns8_status release(int, void* pointer) {
    if (fail_all || pointer == fail_pointer) return RNS8_BACKEND_FAILURE;
    delete[] static_cast<unsigned char*>(pointer); return RNS8_SUCCESS;
  }
  static void defer(rns8::detail::hip_deferred_buffer_cleanup* node) noexcept {
    deferred[deferred_count++] = node;
  }
};
}

TEST_CASE("device allocation owner retains failed frees and exception cleanup records", "[api][lifetime]") {
  using Buffer = rns8::detail::basic_hip_device_buffer<MockDeviceBufferOperations>;
  using Ops = MockDeviceBufferOperations;
  Ops::fail_pointer = nullptr; Ops::fail_all = false; Ops::deferred_count = 0;
  {
    Buffer a, b; void* pa = nullptr; void* pb = nullptr;
    std::size_t ba = 16, bb = 32;
    REQUIRE(a.allocate_alias(0, ba, pa) == RNS8_SUCCESS);
    REQUIRE(b.allocate_alias(0, bb, pb) == RNS8_SUCCESS);
    Ops::fail_pointer = pa;
    CHECK(a.reset_alias(pa, ba) == RNS8_BACKEND_FAILURE);
    CHECK(pa == a.get()); CHECK(ba == 16);
    CHECK(b.reset_alias(pb, bb) == RNS8_SUCCESS);
    CHECK(pb == nullptr); CHECK(bb == 0);
    Ops::fail_pointer = nullptr;
    CHECK(a.reset_alias(pa, ba) == RNS8_SUCCESS);
    CHECK(pa == nullptr); CHECK(ba == 0);
  }
  try {
    Buffer packed, native;
    REQUIRE(packed.allocate(0, 16) == RNS8_SUCCESS);
    REQUIRE(native.allocate(0, 32) == RNS8_SUCCESS);
    Ops::fail_all = true;
    throw std::bad_alloc(); // Equivalent host staging failure, no HIP involved.
  } catch (const std::bad_alloc&) {}
  REQUIRE(Ops::deferred_count == 2);
  Ops::fail_all = false;
  for (unsigned i = 0; i < Ops::deferred_count; ++i) {
    auto* node = Ops::deferred[i];
    REQUIRE(Ops::release(node->device_id, node->pointer) == RNS8_SUCCESS);
    delete node; Ops::deferred[i] = nullptr;
  }
  Ops::deferred_count = 0;
}

TEST_CASE("tiled export replacement cannot certify partially uploaded metadata", "[api][lifetime]") {
  rns8_matrix matrix;
  for (unsigned fail_after = 0; fail_after < 4; ++fail_after) {
    REQUIRE(rns8::detail::replace_export_metadata(matrix, 1000, 1, 4096, [] { return RNS8_SUCCESS; }) == RNS8_SUCCESS);
    std::array<unsigned, 4> device_metadata{2, 1000, 0, 0};
    const auto failed = rns8::detail::replace_export_metadata(matrix, 10, 1, 4096, [&] {
      for (unsigned stage = 0; stage < device_metadata.size(); ++stage) {
        device_metadata[stage] = stage == 0 ? 1 : 10;
        if (stage == fail_after) return RNS8_BACKEND_FAILURE;
      }
      return RNS8_SUCCESS;
    });
    REQUIRE(failed == RNS8_BACKEND_FAILURE);
    CHECK(matrix.hip_export_schedule_fingerprint == 0);
    CHECK(matrix.hip_export_tile_schedule_count == 0);
    CHECK(matrix.hip_export_tile_bounds_count == 0);
    // Repacking a value changes resident identity, not this independently invalid
    // metadata cache. Equal capacities cannot resurrect the old plan key.
    matrix.source_version = 7;
    CHECK(matrix.hip_export_schedule_fingerprint != 1000);
    for (uint64_t plan_key : {1000u, 10u, 1000u}) {
      REQUIRE(rns8::detail::replace_export_metadata(matrix, plan_key, 1, 4096, [] { return RNS8_SUCCESS; }) == RNS8_SUCCESS);
      CHECK(matrix.hip_export_schedule_fingerprint == plan_key);
      CHECK(matrix.hip_export_tile_schedule_count == 1);
    }
  }
  CHECK(256 % 256 == 0); CHECK(256 % 255 == 1);
}

TEST_CASE("matrix cleanup retains only failed release ownership for retry", "[api][lifetime]") {
  // Pure bookkeeping fixture: these addresses are host markers, never HIP allocations.
  rns8_matrix matrix;
  std::array<int, 3> markers{};
  matrix.hip_upload_buffer = &markers[0]; matrix.hip_upload_bytes = 16;
  matrix.hip_export_tile_schedule = &markers[1]; matrix.hip_export_tile_schedule_bytes = 32;
  matrix.hip_export_tile_schedule_count = 2;
  matrix.hip_residues = &markers[2]; matrix.hip_residue_bytes = 64;
  matrix.hip_export_schedule_fingerprint = 99;
  std::vector<void*> released;
  const auto first = rns8::detail::release_matrix_allocations(matrix, [&](int, void* pointer) {
    if (pointer == &markers[1]) return RNS8_BACKEND_FAILURE;
    released.push_back(pointer); return RNS8_SUCCESS;
  });
  REQUIRE(first == RNS8_BACKEND_FAILURE);
  REQUIRE(released == std::vector<void*>{&markers[0], &markers[2]});
  CHECK(matrix.hip_upload_buffer == nullptr); CHECK(matrix.hip_upload_bytes == 0);
  CHECK(matrix.hip_residues == nullptr); CHECK(matrix.hip_residue_bytes == 0);
  CHECK(matrix.hip_export_tile_schedule == &markers[1]);
  CHECK(matrix.hip_export_tile_schedule_bytes == 32);
  CHECK(matrix.hip_export_tile_schedule_count == 2);
  CHECK(matrix.hip_export_schedule_fingerprint == 0);
  released.clear();
  REQUIRE(rns8::detail::release_matrix_allocations(matrix, [&](int, void* pointer) {
    released.push_back(pointer); return RNS8_SUCCESS;
  }) == RNS8_SUCCESS);
  REQUIRE(released == std::vector<void*>{&markers[1]});
  CHECK(matrix.hip_export_tile_schedule == nullptr);
  CHECK(matrix.hip_export_tile_schedule_bytes == 0);
  CHECK(matrix.hip_export_tile_schedule_count == 0);
  REQUIRE(rns8::detail::release_matrix_allocations(matrix, [](int, void*) {
    FAIL("an empty owner must never release an address"); return RNS8_INTERNAL_ERROR;
  }) == RNS8_SUCCESS);
}

TEST_CASE("matrix destruction rejects active readers and writers without mutation", "[api][lifetime]") {
  auto* matrix = new rns8_matrix;
  matrix->in_flight_readers = 1;
  REQUIRE(rns8_destroy_matrix(matrix) == RNS8_BUSY);
  CHECK(matrix->host_residues_current);
  matrix->in_flight_readers = 0; matrix->in_flight_writer = true;
  REQUIRE(rns8_destroy_matrix(matrix) == RNS8_BUSY);
  CHECK(matrix->host_residues_current);
  matrix->in_flight_writer = false;
  REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
}

TEST_CASE("context destruction drains unpublished failed matrix owners", "[api][lifetime]") {
  auto* context = new rns8_context;
  auto* first = new rns8_matrix;
  auto* second = new rns8_matrix;
  first->failed_cleanup_next = second;
  context->failed_matrix_cleanup = first;
  context->hip_failure_poisoned = true;
  // Empty owners exercise the actual list drain without invoking the HIP runtime.
  context->prepared_references = 1;
  REQUIRE(rns8_destroy_context(context) == RNS8_BUSY);
  REQUIRE(context->failed_matrix_cleanup == first);
  context->prepared_references = 0;
  REQUIRE(rns8_destroy_context(context) == RNS8_SUCCESS);
}
