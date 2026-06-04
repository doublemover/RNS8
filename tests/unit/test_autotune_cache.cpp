#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/autotune_cache.hpp"

namespace {

constexpr const char* kCkBoundedKernel = "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1";
constexpr const char* kCkBoundedEpilogue = "ck_fused_i32_to_centered_residue_then_crt_export";
constexpr const char* kCkFiniteKernel = "ck_wmma_cshuffle_finite_u8_mod251_centered_epilogue_v2";
constexpr const char* kCkFiniteGenericKernel = "ck_wmma_cshuffle_finite_u8_centered_epilogue_v1";
constexpr const char* kCkFiniteEpilogue = "ck_fused_i32_to_centered_residue_then_canonical_u8_export";
constexpr const char* kVectorI64Kernel = "hip_vector_alu_i64_exact_192b_v1";
constexpr const char* kVectorI64GemvKernel = "hip_vector_alu_i64_gemv_n1_exact_192b_v1";
constexpr const char* kVectorU64Kernel = "hip_vector_alu_u64_exact_192b_v1";
constexpr const char* kVectorEpilogue = "direct_int64_export";
constexpr const char* kRocwmmaBoundedKernel = "rocwmma_i8_i32_signed_hot_residue_v1";
constexpr const char* kRocwmmaBoundedEpilogue = "rocwmma_fused_i32_to_centered_residue_then_crt_export";

rns8::detail::AutotuneCacheEntry cache_entry(
    const char* key,
    const char* backend,
    bool performance_validated,
    const char* validation_status,
    uint32_t schema_version = 1) {
  rns8::detail::AutotuneCacheEntry entry{};
  entry.key = key;
  entry.selected_backend = backend;
  entry.selected_kernel = kCkBoundedKernel;
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
  entry.epilogue = kCkBoundedEpilogue;
  entry.kernel_family = kCkBoundedKernel;
  entry.performance_validated = performance_validated;
  entry.validation_status = validation_status;
  entry.schema_version = schema_version;
  return entry;
}

std::string reviewed_key(
    const char* backend = "ck",
    const char* target = "gfx1100",
    const char* version = "7.1",
    const char* semantics = "bounded_u64",
    int64_t m = 512,
    int64_t n = 512,
    int64_t k = 512,
    const char* layout = "row_major",
    int64_t k_block_size = 512,
    uint32_t tile_m = 128,
    uint32_t tile_n = 128,
    const char* kernel = kCkBoundedKernel,
    const char* epilogue = kCkBoundedEpilogue) {
  const std::string backend_text = backend;
  const std::string semantics_text = semantics;
  const bool vector_backend = backend_text == "hip-vector-alu-int64";
  const bool finite_semantics = semantics_text == "finite_ring_u8" || semantics_text == "finite_field_u8";
  const char* accumulator_type = vector_backend ? "software_192bit_limb" : "int32";
  const char* accumulator_signedness =
      vector_backend && semantics_text == "bounded_u64" ? "unsigned_u64x_unsigned_u64"
      : vector_backend                                  ? "signed_i64x_signed_i64"
                                                        : "signed_i8x_signed_i8";
  const char* accumulator_modulus_policy =
      vector_backend ? "native_exact_integer_output" : finite_semantics ? "finite_u8_modulus"
                                                                        : "selected_rns_modulus_ladder";
  const int64_t k_block_cap = vector_backend ? 0 : backend_text == "ck" ? 32768 : 65536;
  return std::string("backend=") + backend +
         ";target_id=" + target +
         ";version=" + version +
         ";semantics=" + semantics +
         ";m=" + std::to_string(m) +
         ";n=" + std::to_string(n) +
         ";k=" + std::to_string(k) +
         ";layout=" + layout +
         ";accumulator_type=" + accumulator_type +
         ";accumulator_signedness=" + accumulator_signedness +
         ";accumulator_modulus_policy=" + accumulator_modulus_policy +
         ";k_block_size=" + std::to_string(k_block_size) +
         ";k_block_cap=" + std::to_string(k_block_cap) +
         ";tile_m=" + std::to_string(tile_m) +
         ";tile_n=" + std::to_string(tile_n) +
         ";kernel=" + kernel +
         ";epilogue=" + epilogue;
}

void set_autotune_cache_path_for_test(const std::filesystem::path& path) {
#if defined(_WIN32)
  _putenv_s("RNS8_AUTOTUNE_CACHE_PATH", path.string().c_str());
#else
  setenv("RNS8_AUTOTUNE_CACHE_PATH", path.string().c_str(), 1);
#endif
}

void clear_autotune_cache_path_for_test() {
#if defined(_WIN32)
  _putenv_s("RNS8_AUTOTUNE_CACHE_PATH", "");
#else
  unsetenv("RNS8_AUTOTUNE_CACHE_PATH");
#endif
}

struct ScopedAutotuneCachePath {
  explicit ScopedAutotuneCachePath(const std::filesystem::path& path) { set_autotune_cache_path_for_test(path); }
  ~ScopedAutotuneCachePath() { clear_autotune_cache_path_for_test(); }
};

std::filesystem::path unique_cache_fixture_path(const char* stem) {
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() / (std::string(stem) + "-" + std::to_string(tick) + ".json");
}

void write_cache_fixture(const std::filesystem::path& path, bool root_schema, bool entry_schema) {
  const std::string key = reviewed_key();
  std::ofstream output(path, std::ios::trunc);
  output << "{";
  if (root_schema) {
    output << "\"schema_version\":1,";
  }
  output << "\"entries\":[{"
         << "\"key\":\"" << key << "\","
         << "\"selected_backend\":\"ck\","
         << "\"selected_kernel\":\"" << kCkBoundedKernel << "\","
         << "\"target_id\":\"gfx1100\","
         << "\"hip_sdk_or_library_version\":\"7.1\","
         << "\"semantic_contract\":\"bounded_u64\","
         << "\"shape\":{\"m\":512,\"n\":512,\"k\":512},"
         << "\"layout\":\"row_major\","
         << "\"prefix_schedule_hash\":\"groups=1\","
         << "\"k_block_size\":512,"
         << "\"tile_m\":128,"
         << "\"tile_n\":128,"
         << "\"epilogue\":\"" << kCkBoundedEpilogue << "\","
         << "\"kernel_family\":\"" << kCkBoundedKernel << "\","
         << "\"workspace_bytes\":0,"
         << "\"measured_medians_us\":{\"pack\":1.0,\"rns_gemm\":2.0,\"crt_export\":3.0,\"end_to_end\":4.0},"
         << "\"performance_validated\":true,"
         << "\"validation_status\":\"reviewed_release_same_contract_fastest_windows_gfx1100\","
         << "\"updated_utc\":\"2026-06-02T00:00:00Z\"";
  if (entry_schema) {
    output << ",\"schema_version\":1";
  }
  output << "}]}";
}

}  // namespace

TEST_CASE("autotune cache exposes only reviewed validated entries for selection") {
  rns8::detail::AutotuneCacheSnapshot snapshot{};
  snapshot.loaded = true;
  snapshot.exists = true;
  const std::string validated_key = reviewed_key().c_str();
  const std::string unvalidated_key = reviewed_key("ck", "gfx1100", "7.1", "bounded_u64", 512, 512, 513);
  const std::string bad_schema_key = reviewed_key("rocwmma", "gfx1100", "7.1", "bounded_u64", 512, 513, 512);
  const std::string bad_status_key =
      reviewed_key(
          "rocwmma", "gfx1100", "7.1", "bounded_u64", 513, 512, 512, "row_major", 512, 128, 128,
          kRocwmmaBoundedKernel, kRocwmmaBoundedEpilogue);
  const std::string old_reviewed_status_key = reviewed_key("ck", "gfx1100", "7.1", "bounded_u64", 514, 512, 512);
  snapshot.entries.push_back(
      cache_entry(validated_key.c_str(), "ck", true, "reviewed_release_same_contract_fastest_windows_gfx1100"));
  snapshot.entries.push_back(
      cache_entry(unvalidated_key.c_str(), "ck", false, "schema_v4_capture_emitted_unreviewed"));
  snapshot.entries.back().k = 513;
  snapshot.entries.push_back(
      cache_entry(bad_schema_key.c_str(), "rocwmma", true, "reviewed_release_same_contract_fastest_windows_gfx1100", 99));
  snapshot.entries.back().n = 513;
  snapshot.entries.push_back(cache_entry(bad_status_key.c_str(), "rocwmma", true, "raw_capture_fastest"));
  snapshot.entries.back().m = 513;
  snapshot.entries.back().selected_kernel = kRocwmmaBoundedKernel;
  snapshot.entries.back().epilogue = kRocwmmaBoundedEpilogue;
  snapshot.entries.back().kernel_family = kRocwmmaBoundedKernel;
  snapshot.entries.push_back(
      cache_entry(old_reviewed_status_key.c_str(), "ck", true, "reviewed_same_contract_fastest_windows_gfx1100"));
  snapshot.entries.back().m = 514;

  CHECK(rns8::detail::find_exact_autotune_entry(snapshot, validated_key) != nullptr);
  REQUIRE(rns8::detail::find_validated_autotune_entry(snapshot, validated_key) != nullptr);
  CHECK(rns8::detail::find_validated_autotune_entry(snapshot, validated_key)->selected_backend == "ck");
  rns8::detail::AutotuneRuntimeIdentity runtime{};
  runtime.target_id = "gfx1100";
  runtime.hip_sdk_or_library_version = "7.1";
  REQUIRE(rns8::detail::find_validated_autotune_entry_for_runtime(snapshot, validated_key, runtime) != nullptr);

  runtime.target_id = "gfx1101";
  CHECK(rns8::detail::find_validated_autotune_entry_for_runtime(snapshot, validated_key, runtime) == nullptr);
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, validated_key, "hip-direct", runtime) ==
      "exact_cache_hit_rejected_identity:runtime_target_id_mismatch:gfx1100!=gfx1101");

  runtime.target_id = "gfx1100";
  runtime.hip_sdk_or_library_version = "7.2";
  CHECK(rns8::detail::find_validated_autotune_entry_for_runtime(snapshot, validated_key, runtime) == nullptr);
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, validated_key, "hip-direct", runtime) ==
      "exact_cache_hit_rejected_identity:runtime_version_mismatch:7.1!=7.2");

  CHECK(rns8::detail::find_exact_autotune_entry(snapshot, unvalidated_key) != nullptr);
  CHECK(rns8::detail::find_validated_autotune_entry(snapshot, unvalidated_key) == nullptr);
  CHECK(rns8::detail::find_validated_autotune_entry(snapshot, bad_schema_key) == nullptr);
  CHECK(rns8::detail::find_validated_autotune_entry(snapshot, bad_status_key) == nullptr);

  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, validated_key, "hip-direct") ==
      std::string("exact_cache_hit_validated:ck/") + kCkBoundedKernel);
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, unvalidated_key, "hip-direct") ==
      std::string("exact_cache_hit_rejected_unvalidated:ck/") + kCkBoundedKernel);
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, bad_schema_key, "hip-direct") ==
      "exact_cache_hit_rejected_schema_version:99");
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, bad_status_key, "hip-direct") ==
      "exact_cache_hit_rejected_validation_status:raw_capture_fastest");
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, old_reviewed_status_key, "hip-direct") ==
      "exact_cache_hit_rejected_validation_status:reviewed_same_contract_fastest_windows_gfx1100");
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, "missing", "hip-direct") ==
      "missing_cache_using_direct_hip_correctness");
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, "missing", "cpu-reference") ==
      "missing_cache_using_cpu_reference");
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, "missing", "wrap64-byte-limb") ==
      "missing_cache_using_wrap64_byte_limb_correctness");

  snapshot.schema_version = 99;
  CHECK(rns8::detail::find_validated_autotune_entry(snapshot, validated_key) == nullptr);
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, validated_key, "hip-direct") ==
      "exact_cache_hit_rejected_cache_schema_version:99");
  CHECK(
      rns8::detail::autotune_selection_rationale(snapshot, "missing", "hip-direct") ==
      "cache_unavailable_schema_version:99");
}

TEST_CASE("autotune cache rejects stale identity fields even with reviewed status") {
  struct Case {
    std::string key;
    rns8::detail::AutotuneCacheEntry entry;
    const char* expected;
  };

  auto entry = [](const std::string& key) {
    return cache_entry(key.c_str(), "ck", true, "reviewed_release_same_contract_fastest_windows_gfx1100");
  };

  std::vector<Case> cases;
  {
    auto item = entry(reviewed_key("ck", "gfx1100", "7.1", "bounded_u64"));
    item.finite_modulus = 251;
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:unexpected_entry_finite_modulus"});
  }
  {
    auto item = entry(reviewed_key("ck", "gfx1100", "7.1", "bounded_u64") + ";finite_modulus=251");
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:unexpected_key_finite_modulus"});
  }
  {
    std::string key = reviewed_key(
        "hip-vector-alu-int64",
        "gfx1100",
        "7.1",
        "bounded_u64",
        512,
        512,
        512,
        "row_major",
        512,
        128,
        128,
        kVectorU64Kernel,
        kVectorEpilogue);
    auto item = entry(key);
    item.selected_backend = "hip-vector-alu-int64";
    item.selected_kernel = kVectorU64Kernel;
    item.epilogue = kVectorEpilogue;
    item.kernel_family = kVectorU64Kernel;
    cases.push_back({item.key, item, ""});
  }
  {
    std::string key = reviewed_key(
        "hip-vector-alu-int64",
        "gfx1100",
        "7.1",
        "bounded_i64",
        256,
        1,
        4096,
        "row_major",
        4096,
        128,
        128,
        kVectorI64GemvKernel,
        kVectorEpilogue);
    auto item = entry(key);
    item.selected_backend = "hip-vector-alu-int64";
    item.selected_kernel = kVectorI64GemvKernel;
    item.semantic_contract = "bounded_i64";
    item.m = 256;
    item.n = 1;
    item.k = 4096;
    item.k_block_size = 4096;
    item.epilogue = kVectorEpilogue;
    item.kernel_family = kVectorI64GemvKernel;
    cases.push_back({item.key, item, ""});
  }
  {
    std::string key = reviewed_key(
        "hip-vector-alu-int64",
        "gfx1100",
        "7.1",
        "bounded_i64",
        256,
        1,
        4096,
        "row_major",
        4096,
        128,
        128,
        kVectorI64Kernel,
        kVectorEpilogue);
    auto item = entry(key);
    item.selected_backend = "hip-vector-alu-int64";
    item.selected_kernel = kVectorI64Kernel;
    item.semantic_contract = "bounded_i64";
    item.m = 256;
    item.n = 1;
    item.k = 4096;
    item.k_block_size = 4096;
    item.epilogue = kVectorEpilogue;
    item.kernel_family = kVectorI64Kernel;
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:unsupported_autotune_kernel_for_contract"});
  }
  {
    std::string key =
        reviewed_key(
            "ck", "gfx1100", "7.1", "finite_ring_u8", 512, 512, 512, "row_major", 512, 128, 128,
            kCkFiniteKernel, kCkFiniteEpilogue) +
        ";finite_modulus=251";
    auto item = entry(key);
    item.selected_kernel = kCkFiniteKernel;
    item.semantic_contract = "finite_ring_u8";
    item.finite_modulus = 251;
    item.epilogue = kCkFiniteEpilogue;
    item.kernel_family = kCkFiniteKernel;
    cases.push_back({item.key, item, ""});
  }
  {
    std::string key =
        reviewed_key(
            "ck", "gfx1100", "7.1", "finite_ring_u8", 512, 512, 512, "row_major", 512, 128, 128,
            kCkFiniteGenericKernel, kCkFiniteEpilogue) +
        ";finite_modulus=251";
    auto item = entry(key);
    item.selected_kernel = kCkFiniteGenericKernel;
    item.semantic_contract = "finite_ring_u8";
    item.finite_modulus = 251;
    item.epilogue = kCkFiniteEpilogue;
    item.kernel_family = kCkFiniteGenericKernel;
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:unsupported_autotune_kernel_for_contract"});
  }
  {
    auto item = entry(
        reviewed_key(
            "ck", "gfx1100", "7.1", "finite_ring_u8", 512, 512, 512, "row_major", 512, 128, 128,
            kCkFiniteKernel, kCkFiniteEpilogue));
    item.selected_kernel = kCkFiniteKernel;
    item.semantic_contract = "finite_ring_u8";
    item.finite_modulus = 251;
    item.epilogue = kCkFiniteEpilogue;
    item.kernel_family = kCkFiniteKernel;
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:missing_or_invalid_key_finite_modulus"});
  }
  {
    std::string key =
        reviewed_key(
            "ck", "gfx1100", "7.1", "finite_ring_u8", 512, 512, 512, "row_major", 512, 128, 128,
            kCkFiniteKernel, kCkFiniteEpilogue) +
        ";finite_modulus=251";
    auto item = entry(key);
    item.selected_kernel = kCkFiniteKernel;
    item.semantic_contract = "finite_ring_u8";
    item.epilogue = kCkFiniteEpilogue;
    item.kernel_family = kCkFiniteKernel;
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:missing_entry_finite_modulus"});
  }
  {
    std::string key =
        reviewed_key(
            "ck", "gfx1100", "7.1", "finite_ring_u8", 512, 512, 512, "row_major", 512, 128, 128,
            kCkFiniteKernel, kCkFiniteEpilogue) +
        ";finite_modulus=251";
    auto item = entry(key);
    item.selected_kernel = kCkFiniteKernel;
    item.semantic_contract = "finite_ring_u8";
    item.finite_modulus = 255;
    item.epilogue = kCkFiniteEpilogue;
    item.kernel_family = kCkFiniteKernel;
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:key_finite_modulus_mismatch"});
  }
  {
    std::string key = reviewed_key(
        "wmma",
        "gfx1100",
        "7.1",
        "bounded_u64",
        512,
        512,
        512,
        "row_major",
        512,
        128,
        128,
        kRocwmmaBoundedKernel,
        kRocwmmaBoundedEpilogue);
    auto item = entry(key);
    item.selected_backend = "wmma";
    item.selected_kernel = kRocwmmaBoundedKernel;
    item.semantic_contract = "bounded_u64";
    item.epilogue = kRocwmmaBoundedEpilogue;
    item.kernel_family = kRocwmmaBoundedKernel;
    cases.push_back(
        {item.key, item, "exact_cache_hit_rejected_identity:unsupported_autotune_backend_semantic_contract"});
  }
  {
    std::string key = reviewed_key(
        "rocwmma",
        "gfx1100",
        "7.1",
        "wrap_u64_mod_2_64",
        64,
        64,
        64,
        "row_major",
        64,
        16,
        16,
        "rocwmma_wrap64_byte_gemm36_candidate_v0",
        "low64_wrap_export");
    auto item = entry(key);
    item.selected_backend = "rocwmma";
    item.selected_kernel = "rocwmma_wrap64_byte_gemm36_candidate_v0";
    item.semantic_contract = "wrap_u64_mod_2_64";
    item.m = 64;
    item.n = 64;
    item.k = 64;
    item.k_block_size = 64;
    item.tile_m = 16;
    item.tile_n = 16;
    item.epilogue = "low64_wrap_export";
    item.kernel_family = "rocwmma_wrap64_byte_gemm36_candidate_v0";
    cases.push_back(
        {item.key, item, "exact_cache_hit_rejected_identity:unsupported_autotune_backend_semantic_contract"});
  }
  {
    std::string key = reviewed_key(
        "hip-direct",
        "gfx1100",
        "HIP runtime",
        "bounded_u64",
        512,
        512,
        512,
        "row_major",
        512,
        128,
        128,
        "direct_hip_tiled_rns_gemm_v1",
        "fused_centered_residue_then_crt_export");
    auto item = entry(key);
    item.selected_backend = "hip-direct";
    item.selected_kernel = "direct_hip_tiled_rns_gemm_v1";
    item.hip_sdk_or_library_version = "HIP runtime";
    item.epilogue = "fused_centered_residue_then_crt_export";
    item.kernel_family = "direct_hip_tiled_rns_gemm_v1";
    cases.push_back(
        {item.key, item, "exact_cache_hit_rejected_identity:unsupported_autotune_backend_semantic_contract"});
  }
  {
    std::string key = reviewed_key(
        "ck",
        "gfx1100",
        "7.1",
        "bounded_u64",
        512,
        512,
        512,
        "row_major",
        512,
        128,
        128,
        "rocwmma_wrap64_byte_gemm36_candidate_v0",
        kCkBoundedEpilogue);
    auto item = entry(key);
    item.selected_kernel = "rocwmma_wrap64_byte_gemm36_candidate_v0";
    item.kernel_family = "rocwmma_wrap64_byte_gemm36_candidate_v0";
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:unsupported_autotune_kernel_for_contract"});
  }
  {
    std::string key = reviewed_key(
        "ck",
        "gfx1100",
        "7.1",
        "bounded_u64",
        512,
        512,
        512,
        "row_major",
        512,
        128,
        128,
        kCkBoundedKernel,
        "low64_wrap_export");
    auto item = entry(key);
    item.epilogue = "low64_wrap_export";
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:unsupported_autotune_epilogue_for_contract"});
  }
  {
    auto item = entry(reviewed_key("ck", "gfx1100", "7.1"));
    item.target_id = "gfx1101";
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:key_target_id_mismatch"});
  }
  {
    auto item = entry(reviewed_key("ck", "gfx1100", "7.1"));
    item.key.replace(
        item.key.find(";target_id=gfx1100;"),
        std::string(";target_id=gfx1100;").size(),
        ";target=gfx1100;");
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:missing_key_target_id"});
  }
  {
    auto item = entry(reviewed_key("ck", "gfx1100", "7.1") + ";target=gfx1101");
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:key_target_mismatch"});
  }
  {
    auto item = entry(reviewed_key("ck", "gfx1100", "7.1"));
    item.hip_sdk_or_library_version = "7.2";
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:key_version_mismatch"});
  }
  {
    auto item = entry(reviewed_key("ck", "gfx1100", "7.1", "bounded_u64"));
    item.semantic_contract = "bounded_i64";
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:key_semantics_mismatch"});
  }
  {
    auto item = entry(reviewed_key("ck", "gfx1100", "7.1", "bounded_u64", 512, 512, 512));
    item.n = 1024;
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:key_n_mismatch"});
  }
  {
    auto item = entry(reviewed_key("ck", "gfx1100", "7.1", "bounded_u64", 512, 512, 512, "row_major"));
    item.layout = "column_major";
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:key_layout_mismatch"});
  }
  {
    auto item = entry(reviewed_key("ck", "gfx1100", "7.1", "bounded_u64", 512, 512, 512));
    item.selected_kernel = "other_kernel";
    cases.push_back({item.key, item, "exact_cache_hit_rejected_identity:key_kernel_mismatch"});
  }

  for (const auto& item : cases) {
    rns8::detail::AutotuneCacheSnapshot snapshot{};
    snapshot.loaded = true;
    snapshot.exists = true;
    snapshot.entries.push_back(item.entry);
    CHECK(rns8::detail::find_exact_autotune_entry(snapshot, item.key) != nullptr);
    if (std::string(item.expected).empty()) {
      CHECK(rns8::detail::find_validated_autotune_entry(snapshot, item.key) != nullptr);
      CHECK(
          rns8::detail::autotune_selection_rationale(snapshot, item.key, "hip-direct") ==
          std::string("exact_cache_hit_validated:") + item.entry.selected_backend + "/" + item.entry.selected_kernel);
    } else {
      CHECK(rns8::detail::find_validated_autotune_entry(snapshot, item.key) == nullptr);
      CHECK(rns8::detail::autotune_selection_rationale(snapshot, item.key, "hip-direct") == item.expected);
    }
  }
}

TEST_CASE("autotune cache reader requires explicit root and entry schema versions") {
  const std::filesystem::path path = unique_cache_fixture_path("rns8-autotune-cache-schema-fixture");
  const std::string key = reviewed_key();
  ScopedAutotuneCachePath scoped_path(path);

  write_cache_fixture(path, false, true);
  {
    const auto snapshot = rns8::detail::read_autotune_cache();
    REQUIRE(snapshot.loaded);
    CHECK(snapshot.schema_version == 0);
    CHECK(rns8::detail::find_exact_autotune_entry(snapshot, key) != nullptr);
    CHECK(rns8::detail::find_validated_autotune_entry(snapshot, key) == nullptr);
    CHECK(
        rns8::detail::autotune_selection_rationale(snapshot, key, "hip-direct") ==
        "exact_cache_hit_rejected_cache_schema_version:0");
  }

  write_cache_fixture(path, true, false);
  {
    const auto snapshot = rns8::detail::read_autotune_cache();
    REQUIRE(snapshot.loaded);
    CHECK(snapshot.schema_version == 1);
    CHECK(rns8::detail::find_exact_autotune_entry(snapshot, key) != nullptr);
    CHECK(rns8::detail::find_validated_autotune_entry(snapshot, key) == nullptr);
    CHECK(
        rns8::detail::autotune_selection_rationale(snapshot, key, "hip-direct") ==
        "exact_cache_hit_rejected_schema_version:0");
  }

  write_cache_fixture(path, true, true);
  {
    const auto snapshot = rns8::detail::read_autotune_cache();
    REQUIRE(snapshot.loaded);
    CHECK(snapshot.schema_version == 1);
    REQUIRE(rns8::detail::find_validated_autotune_entry(snapshot, key) != nullptr);
    CHECK(
        rns8::detail::autotune_selection_rationale(snapshot, key, "hip-direct") ==
        std::string("exact_cache_hit_validated:ck/") + kCkBoundedKernel);
  }

  std::filesystem::remove(path);
}
