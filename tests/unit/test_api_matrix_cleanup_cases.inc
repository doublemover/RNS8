#include "core/matrix_cleanup.hpp"
#include <array>

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
