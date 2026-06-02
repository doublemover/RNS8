#include <catch2/catch_test_macros.hpp>

#include "core/autotune_cache.hpp"

namespace {

rns8::detail::AutotuneCacheEntry cache_entry(
    const char* key,
    const char* backend,
    bool performance_validated,
    const char* validation_status,
    uint32_t schema_version = 1) {
  rns8::detail::AutotuneCacheEntry entry{};
  entry.key = key;
  entry.selected_backend = backend;
  entry.selected_kernel = "unit_kernel";
  entry.target_id = "gfx1100";
  entry.hip_sdk_or_library_version = "7.1";
  entry.semantic_contract = "bounded_u64";
  entry.m = 512;
  entry.n = 512;
  entry.k = 512;
  entry.layout = "row_major";
  entry.prefix_schedule_hash = "groups=1";
  entry.k_block_size = 512;
  entry.tile_m = 128;
  entry.tile_n = 128;
  entry.epilogue = "crt_export";
  entry.kernel_family = "unit_kernel";
  entry.performance_validated = performance_validated;
  entry.validation_status = validation_status;
  entry.schema_version = schema_version;
  return entry;
}

}  // namespace

TEST_CASE("autotune cache exposes only reviewed validated entries for selection") {
  rns8::detail::AutotuneCacheSnapshot snapshot{};
  snapshot.loaded = true;
  snapshot.exists = true;
  snapshot.entries.push_back(cache_entry("validated", "ck", true, "reviewed_same_contract_fastest_windows_gfx1100"));
  snapshot.entries.push_back(cache_entry("unvalidated", "ck", false, "schema_v4_capture_emitted_unreviewed"));
  snapshot.entries.push_back(cache_entry("bad-schema", "wmma", true, "reviewed_same_contract_fastest_windows_gfx1100", 99));
  snapshot.entries.push_back(cache_entry("bad-status", "wmma", true, "raw_capture_fastest"));

  CHECK(rns8::detail::find_exact_autotune_entry(snapshot, "validated") != nullptr);
  REQUIRE(rns8::detail::find_validated_autotune_entry(snapshot, "validated") != nullptr);
  CHECK(rns8::detail::find_validated_autotune_entry(snapshot, "validated")->selected_backend == "ck");

  CHECK(rns8::detail::find_exact_autotune_entry(snapshot, "unvalidated") != nullptr);
  CHECK(rns8::detail::find_validated_autotune_entry(snapshot, "unvalidated") == nullptr);
  CHECK(rns8::detail::find_validated_autotune_entry(snapshot, "bad-schema") == nullptr);
  CHECK(rns8::detail::find_validated_autotune_entry(snapshot, "bad-status") == nullptr);

  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, "validated", "hip-direct") ==
      "exact_cache_hit_validated:ck/unit_kernel");
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, "unvalidated", "hip-direct") ==
      "exact_cache_hit_rejected_unvalidated:ck/unit_kernel");
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, "bad-schema", "hip-direct") ==
      "exact_cache_hit_rejected_schema_version:99");
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, "bad-status", "hip-direct") ==
      "exact_cache_hit_rejected_validation_status:raw_capture_fastest");
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, "missing", "hip-direct") ==
      "missing_cache_using_direct_hip_correctness");
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, "missing", "cpu-reference") ==
      "missing_cache_using_cpu_reference");
}
