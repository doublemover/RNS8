#include "core/autotune_cache.hpp"

#include <nlohmann/json.hpp>

#include "backend_common/finite_u8_reducer.hpp"
#include "rns8/moduli.h"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>
#include <utility>

namespace rns8::detail {
namespace {

using json = nlohmann::ordered_json;

std::string env_value(const char* name) {
#if defined(_WIN32)
  std::size_t required = 0;
  if (getenv_s(&required, nullptr, 0, name) != 0 || required == 0) {
    return {};
  }
  std::string value(required, '\0');
  if (getenv_s(&required, value.data(), value.size(), name) != 0 || required == 0) {
    return {};
  }
  if (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
#else
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string();
#endif
}

std::string current_utc_timestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  std::ostringstream out;
  out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

uint32_t explicit_u32_field_or_zero(const json& object, const char* name) {
  const auto field = object.find(name);
  if (field == object.end() || (!field->is_number_unsigned() && !field->is_number_integer())) {
    return 0;
  }
  try {
    if (field->is_number_unsigned()) {
      const uint64_t value = field->get<uint64_t>();
      if (value > std::numeric_limits<uint32_t>::max()) {
        return 0;
      }
      return static_cast<uint32_t>(value);
    }
    const int64_t value = field->get<int64_t>();
    if (value < 0 || value > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return 0;
    }
    return static_cast<uint32_t>(value);
  } catch (...) {
    return 0;
  }
}

json entry_to_json(AutotuneCacheEntry entry) {
  if (entry.updated_utc.empty()) {
    entry.updated_utc = current_utc_timestamp();
  }
  return json{
      {"key", entry.key},
      {"selected_backend", entry.selected_backend},
      {"selected_kernel", entry.selected_kernel},
      {"target_id", entry.target_id},
      {"hip_sdk_or_library_version", entry.hip_sdk_or_library_version},
      {"semantic_contract", entry.semantic_contract},
      {"finite_modulus", entry.finite_modulus},
      {"shape", {{"m", entry.m}, {"n", entry.n}, {"k", entry.k}}},
      {"layout", entry.layout},
      {"prefix_schedule_hash", entry.prefix_schedule_hash},
      {"k_block_size", entry.k_block_size},
      {"tile_m", entry.tile_m},
      {"tile_n", entry.tile_n},
      {"epilogue", entry.epilogue},
      {"kernel_family", entry.kernel_family},
      {"workspace_bytes", entry.workspace_bytes},
      {"measured_medians_us",
       {
           {"pack", entry.measured_median_pack_us},
           {"pack_a", entry.measured_median_pack_a_us},
           {"pack_b", entry.measured_median_pack_b_us},
           {"rns_gemm", entry.measured_median_gemm_us},
           {"crt_export", entry.measured_median_export_us},
           {"end_to_end", entry.measured_median_end_to_end_us},
       }},
      {"performance_validated", entry.performance_validated},
      {"validation_status", entry.validation_status},
      {"schema_version", entry.schema_version},
      {"updated_utc", entry.updated_utc},
  };
}

AutotuneCacheEntry entry_from_json(const json& item) {
  AutotuneCacheEntry entry{};
  entry.key = item.value("key", "");
  entry.selected_backend = item.value("selected_backend", "");
  entry.selected_kernel = item.value("selected_kernel", "");
  entry.target_id = item.value("target_id", "");
  entry.hip_sdk_or_library_version = item.value("hip_sdk_or_library_version", "");
  entry.semantic_contract = item.value("semantic_contract", "");
  entry.finite_modulus = explicit_u32_field_or_zero(item, "finite_modulus");
  if (const auto shape = item.find("shape"); shape != item.end() && shape->is_object()) {
    entry.m = shape->value("m", int64_t{0});
    entry.n = shape->value("n", int64_t{0});
    entry.k = shape->value("k", int64_t{0});
  }
  entry.layout = item.value("layout", "row_major");
  entry.prefix_schedule_hash = item.value("prefix_schedule_hash", "");
  entry.k_block_size = item.value("k_block_size", int64_t{0});
  entry.tile_m = item.value("tile_m", uint32_t{0});
  entry.tile_n = item.value("tile_n", uint32_t{0});
  entry.epilogue = item.value("epilogue", "");
  entry.kernel_family = item.value("kernel_family", "");
  entry.workspace_bytes = item.value("workspace_bytes", uint64_t{0});
  if (const auto medians = item.find("measured_medians_us"); medians != item.end() && medians->is_object()) {
    entry.measured_median_pack_us = medians->value("pack", 0.0);
    entry.measured_median_pack_a_us = medians->value("pack_a", 0.0);
    entry.measured_median_pack_b_us = medians->value("pack_b", 0.0);
    entry.measured_median_gemm_us = medians->value("rns_gemm", 0.0);
    entry.measured_median_export_us = medians->value("crt_export", 0.0);
    entry.measured_median_end_to_end_us = medians->value("end_to_end", 0.0);
  }
  entry.performance_validated = item.value("performance_validated", false);
  entry.validation_status = item.value("validation_status", "");
  entry.schema_version = explicit_u32_field_or_zero(item, "schema_version");
  entry.updated_utc = item.value("updated_utc", "");
  return entry;
}

std::vector<std::pair<std::string, std::string>> parse_key_fields(const std::string& key) {
  std::vector<std::pair<std::string, std::string>> fields;
  std::size_t begin = 0;
  while (begin <= key.size()) {
    const std::size_t end = key.find(';', begin);
    const std::string part = key.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (!part.empty()) {
      const std::size_t equals = part.find('=');
      if (equals != std::string::npos && equals != 0) {
        fields.emplace_back(part.substr(0, equals), part.substr(equals + 1));
      }
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return fields;
}

std::string key_field(const std::vector<std::pair<std::string, std::string>>& fields, const char* name) {
  for (const auto& field : fields) {
    if (field.first == name) {
      return field.second;
    }
  }
  return {};
}

std::string first_key_field(
    const std::vector<std::pair<std::string, std::string>>& fields,
    const char* first,
    const char* second) {
  std::string value = key_field(fields, first);
  if (!value.empty()) {
    return value;
  }
  return key_field(fields, second);
}

bool parse_i64(const std::string& text, int64_t& value) {
  if (text.empty()) {
    return false;
  }
  std::size_t parsed = 0;
  try {
    const long long result = std::stoll(text, &parsed, 10);
    if (parsed != text.size()) {
      return false;
    }
    value = static_cast<int64_t>(result);
    return true;
  } catch (...) {
    return false;
  }
}

std::string required_key_field(
    const std::vector<std::pair<std::string, std::string>>& fields,
    const char* name) {
  const std::string value = key_field(fields, name);
  return value.empty() ? std::string("missing_key_") + name : std::string();
}

std::string require_key_i64(
    const std::vector<std::pair<std::string, std::string>>& fields,
    const char* name,
    int64_t expected) {
  const std::string value = key_field(fields, name);
  int64_t parsed = 0;
  if (!parse_i64(value, parsed)) {
    return std::string("missing_or_invalid_key_") + name;
  }
  if (parsed != expected) {
    return std::string("key_") + name + "_mismatch";
  }
  return {};
}

std::string require_key_text(
    const std::vector<std::pair<std::string, std::string>>& fields,
    const char* name,
    const char* expected) {
  const std::string value = key_field(fields, name);
  if (value.empty()) {
    return std::string("missing_key_") + name;
  }
  if (value != expected) {
    return std::string("key_") + name + "_mismatch";
  }
  return {};
}

std::string require_optional_key_i64(
    const std::vector<std::pair<std::string, std::string>>& fields,
    const char* name,
    int64_t expected) {
  const std::string value = key_field(fields, name);
  if (value.empty()) {
    return {};
  }
  int64_t parsed = 0;
  if (!parse_i64(value, parsed)) {
    return std::string("invalid_key_") + name;
  }
  if (parsed != expected) {
    return std::string("key_") + name + "_mismatch";
  }
  return {};
}

bool reviewed_autotune_backend_supports_semantic_contract(const AutotuneCacheEntry& entry) {
  if (entry.selected_backend == "hip-vector-alu-int64") {
    return entry.semantic_contract == "bounded_i64" || entry.semantic_contract == "bounded_u64";
  }
  const bool public_accelerator =
      entry.selected_backend == "hipblaslt" || entry.selected_backend == "ck" ||
      entry.selected_backend == "rocwmma" || entry.selected_backend == "amdgpu-builtins";
  const bool hip_resident_rns_semantic =
      entry.semantic_contract == "bounded_i64" || entry.semantic_contract == "bounded_u64" ||
      entry.semantic_contract == "exact_wide_signed" || entry.semantic_contract == "exact_wide_unsigned" ||
      entry.semantic_contract == "finite_ring_u8" || entry.semantic_contract == "finite_field_u8";
  return public_accelerator && hip_resident_rns_semantic;
}

bool is_bounded_rns_semantic(const std::string& semantic_contract) {
  return semantic_contract == "bounded_i64" || semantic_contract == "bounded_u64";
}

bool is_exact_wide_semantic(const std::string& semantic_contract) {
  return semantic_contract == "exact_wide_signed" || semantic_contract == "exact_wide_unsigned";
}

bool is_finite_u8_semantic(const std::string& semantic_contract) {
  return semantic_contract == "finite_ring_u8" || semantic_contract == "finite_field_u8";
}

std::string expected_ck_finite_kernel(uint32_t finite_modulus) {
  switch (finite_modulus) {
    case 251:
      return "ck_wmma_cshuffle_finite_u8_mod251_centered_epilogue_v2";
    case 255:
      return "ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2";
    case 256:
      return "ck_wmma_cshuffle_finite_u8_mod256_centered_epilogue_v2";
    default:
      return "ck_wmma_cshuffle_finite_u8_static_modulus_centered_epilogue_v2";
  }
}

std::string expected_rocwmma_finite_kernel(uint32_t finite_modulus) {
  switch (finite_modulus) {
    case 251:
      return "rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2";
    case 255:
      return "rocwmma_i8_i32_signed_finite_u8_mod255_hot_residue_v2";
    case 256:
      return "rocwmma_i8_i32_signed_finite_u8_mod256_hot_residue_v2";
    default:
      return "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1";
  }
}

std::string expected_vector_alu_kernel(const AutotuneCacheEntry& entry) {
  const bool gemv_n1 = entry.n == 1 && entry.k >= 4096;
  const bool gemv_small_n = entry.n > 1 && entry.n <= 4 && entry.k >= 512;
  if (entry.semantic_contract == "bounded_i64") {
    if (gemv_n1) {
      return "hip_vector_alu_i64_gemv_n1_exact_192b_v1";
    }
    if (gemv_small_n) {
      return "hip_vector_alu_i64_gemv_small_n_exact_192b_v1";
    }
    return "hip_vector_alu_i64_exact_192b_v1";
  }
  if (entry.semantic_contract == "bounded_u64") {
    if (gemv_n1) {
      return "hip_vector_alu_u64_gemv_n1_exact_192b_v1";
    }
    if (gemv_small_n) {
      return "hip_vector_alu_u64_gemv_small_n_exact_192b_v1";
    }
    return "hip_vector_alu_u64_exact_192b_v1";
  }
  return {};
}

bool amdgpu_builtin_cdna3_target_id(const std::string& target_id) {
  return target_id.rfind("gfx942", 0) == 0;
}

bool amdgpu_builtin_rdna3_target_id(const std::string& target_id) {
  return target_id.rfind("gfx110", 0) == 0;
}

bool amdgpu_builtin_rdna4_target_id(const std::string& target_id) {
  return target_id == "gfx1200" || target_id == "gfx1201";
}

bool reviewed_amdgpu_builtin_kernel_supported_for_contract(const AutotuneCacheEntry& entry) {
  if (is_finite_u8_semantic(entry.semantic_contract)) {
    if (!finite_u8::static_byte_modulus_supported(entry.finite_modulus)) {
      return false;
    }
    if (amdgpu_builtin_cdna3_target_id(entry.target_id)) {
      return entry.selected_kernel == "amdgpu_builtin_cdna3_mfma_i32_16x16x32_i8_finite_u8_epilogue_v1" ||
             entry.selected_kernel == "amdgpu_builtin_cdna3_mfma_i32_32x32x16_i8_finite_u8_epilogue_v1";
    }
    if (amdgpu_builtin_rdna3_target_id(entry.target_id)) {
      return entry.selected_kernel == "amdgpu_builtin_rdna3_wmma_i32_16x16x16_iu8_finite_u8_epilogue_v1";
    }
    if (amdgpu_builtin_rdna4_target_id(entry.target_id)) {
      return entry.selected_kernel == "amdgpu_builtin_rdna4_wmma_i32_16x16x16_iu8_finite_u8_epilogue_v1";
    }
    return false;
  }
  if (is_exact_wide_semantic(entry.semantic_contract) || is_bounded_rns_semantic(entry.semantic_contract)) {
    if (amdgpu_builtin_cdna3_target_id(entry.target_id)) {
      return entry.selected_kernel == "amdgpu_builtin_cdna3_mfma_i32_16x16x32_i8_centered_epilogue_v1" ||
             entry.selected_kernel == "amdgpu_builtin_cdna3_mfma_i32_32x32x16_i8_centered_epilogue_v1";
    }
    if (amdgpu_builtin_rdna3_target_id(entry.target_id)) {
      return entry.selected_kernel == "amdgpu_builtin_rdna3_wmma_i32_16x16x16_iu8_centered_epilogue_v1";
    }
    if (amdgpu_builtin_rdna4_target_id(entry.target_id)) {
      return entry.selected_kernel == "amdgpu_builtin_rdna4_wmma_i32_16x16x16_iu8_centered_epilogue_v1";
    }
  }
  return false;
}

bool reviewed_autotune_kernel_supported_for_contract(const AutotuneCacheEntry& entry) {
  if (entry.selected_backend == "hip-vector-alu-int64") {
    const std::string expected = expected_vector_alu_kernel(entry);
    return !expected.empty() && entry.selected_kernel == expected;
  }
  if (entry.selected_backend == "hipblaslt") {
    return entry.selected_kernel == "hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2";
  }
  if (entry.selected_backend == "ck") {
    if (is_finite_u8_semantic(entry.semantic_contract)) {
      if (!finite_u8::static_byte_modulus_supported(entry.finite_modulus)) {
        return false;
      }
      return entry.selected_kernel == expected_ck_finite_kernel(entry.finite_modulus);
    }
    if (is_exact_wide_semantic(entry.semantic_contract)) {
      return entry.selected_kernel == "ck_wmma_cshuffle_i8_i32_default_moduli_static_centered_epilogue_v3";
    }
    if (is_bounded_rns_semantic(entry.semantic_contract)) {
      return entry.selected_kernel == "ck_wmma_cshuffle_i8_i32_default_moduli_static_centered_epilogue_v3" ||
             entry.selected_kernel == "ck_wmma_cshuffle_tiled_i8_i32_default_moduli_static_centered_epilogue_v3";
    }
  }
  if (entry.selected_backend == "rocwmma") {
    if (is_finite_u8_semantic(entry.semantic_contract)) {
      if (!finite_u8::static_byte_modulus_supported(entry.finite_modulus)) {
        return false;
      }
      return entry.selected_kernel == expected_rocwmma_finite_kernel(entry.finite_modulus);
    }
    if (is_exact_wide_semantic(entry.semantic_contract)) {
      return entry.selected_kernel == "rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2";
    }
    if (is_bounded_rns_semantic(entry.semantic_contract)) {
      return entry.selected_kernel == "rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2" ||
             entry.selected_kernel == "rocwmma_i8_i32_signed_tiled_mod251_255_256_hot_residue_v2";
    }
  }
  if (entry.selected_backend == "amdgpu-builtins") {
    return reviewed_amdgpu_builtin_kernel_supported_for_contract(entry);
  }
  return false;
}

bool reviewed_autotune_epilogue_supported_for_contract(const AutotuneCacheEntry& entry) {
  if (entry.selected_backend == "hip-vector-alu-int64") {
    return (entry.semantic_contract == "bounded_i64" || entry.semantic_contract == "bounded_u64") &&
           entry.epilogue == "direct_int64_export";
  }
  if (entry.selected_backend == "hipblaslt") {
    if (is_finite_u8_semantic(entry.semantic_contract)) {
      return entry.epilogue == "separate_i32_scratch_reduce_then_canonical_u8_export";
    }
    if (is_exact_wide_semantic(entry.semantic_contract)) {
      return entry.epilogue == "separate_i32_scratch_reduce_rns_output";
    }
    if (is_bounded_rns_semantic(entry.semantic_contract)) {
      return entry.epilogue == "separate_i32_scratch_reduce_then_crt_export";
    }
  }
  if (entry.selected_backend == "ck") {
    if (is_finite_u8_semantic(entry.semantic_contract)) {
      return entry.epilogue == "ck_fused_i32_to_centered_residue_then_canonical_u8_export";
    }
    if (is_exact_wide_semantic(entry.semantic_contract)) {
      return entry.epilogue == "ck_fused_i32_to_centered_residue_rns_output";
    }
    if (is_bounded_rns_semantic(entry.semantic_contract)) {
      return entry.epilogue == "ck_fused_i32_to_centered_residue_then_crt_export";
    }
  }
  if (entry.selected_backend == "rocwmma") {
    if (is_finite_u8_semantic(entry.semantic_contract)) {
      return entry.epilogue == "rocwmma_fused_i32_to_centered_residue_then_canonical_u8_export";
    }
    if (is_exact_wide_semantic(entry.semantic_contract)) {
      return entry.epilogue == "rocwmma_fused_i32_to_centered_residue_rns_output";
    }
    if (is_bounded_rns_semantic(entry.semantic_contract)) {
      return entry.epilogue == "rocwmma_fused_i32_to_centered_residue_then_crt_export";
    }
  }
  if (entry.selected_backend == "amdgpu-builtins") {
    if (is_finite_u8_semantic(entry.semantic_contract)) {
      return entry.epilogue == "amdgpu_builtin_fused_i32_to_centered_residue_then_canonical_u8_export";
    }
    if (is_exact_wide_semantic(entry.semantic_contract)) {
      return entry.epilogue == "amdgpu_builtin_fused_i32_to_centered_residue_rns_output";
    }
    if (is_bounded_rns_semantic(entry.semantic_contract)) {
      return entry.epilogue == "amdgpu_builtin_fused_i32_to_centered_residue_then_chained_crt_export";
    }
  }
  return false;
}

bool cdna_target_id(const std::string& target_id) {
  return target_id == "gfx90a" || target_id == "gfx942" || target_id == "gfx950";
}

std::string expected_reviewed_validation_status(const std::string& target_id) {
  if (target_id == "gfx1100") {
    return "reviewed_release_same_contract_fastest_windows_gfx1100";
  }
  if (cdna_target_id(target_id)) {
    return "reviewed_release_same_contract_fastest_linux_" + target_id;
  }
  if (target_id.rfind("gfx", 0) == 0) {
    return "reviewed_release_same_contract_fastest_target_" + target_id;
  }
  return {};
}

bool reviewed_validation_status_matches_target(const AutotuneCacheEntry& entry) {
  const std::string expected = expected_reviewed_validation_status(entry.target_id);
  return !expected.empty() && entry.validation_status == expected;
}

const char* expected_accumulator_type_for_entry(const AutotuneCacheEntry& entry) {
  if (entry.selected_backend == "hip-vector-alu-int64") {
    return "software_192bit_limb";
  }
  return "int32";
}

const char* expected_accumulator_signedness_for_entry(const AutotuneCacheEntry& entry) {
  if (entry.selected_backend == "hip-vector-alu-int64") {
    return entry.semantic_contract == "bounded_i64" ? "signed_i64x_signed_i64" : "unsigned_u64x_unsigned_u64";
  }
  return "signed_i8x_signed_i8";
}

const char* expected_accumulator_modulus_policy_for_entry(const AutotuneCacheEntry& entry) {
  if (entry.selected_backend == "hip-vector-alu-int64") {
    return "native_exact_integer_output";
  }
  if (is_finite_u8_semantic(entry.semantic_contract)) {
    return "finite_u8_modulus";
  }
  return "selected_rns_modulus_ladder";
}

int64_t expected_accumulator_k_block_cap_for_entry(const AutotuneCacheEntry& entry) {
  if (entry.selected_backend == "hip-vector-alu-int64") {
    return 0;
  }
  if (entry.selected_backend == "ck") {
    return 32768;
  }
  return static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK);
}

std::string validated_entry_identity_failure(const AutotuneCacheEntry& entry) {
  if (entry.key.empty() || entry.selected_backend.empty() || entry.selected_kernel.empty() ||
      entry.target_id.empty() || entry.hip_sdk_or_library_version.empty() ||
      entry.semantic_contract.empty() || entry.layout.empty() || entry.prefix_schedule_hash.empty() ||
      entry.epilogue.empty() || entry.kernel_family.empty()) {
    return "missing_required_entry_identity_field";
  }
  if (!reviewed_autotune_backend_supports_semantic_contract(entry)) {
    return "unsupported_autotune_backend_semantic_contract";
  }
  if (entry.m <= 0 || entry.n <= 0 || entry.k <= 0 || entry.tile_m == 0 || entry.tile_n == 0 ||
      entry.k_block_size < 0) {
    return "invalid_entry_shape_or_tile";
  }

  const auto fields = parse_key_fields(entry.key);
  if (std::string failure = required_key_field(fields, "backend"); !failure.empty()) return failure;
  if (std::string failure = required_key_field(fields, "semantics"); !failure.empty()) return failure;
  if (std::string failure = required_key_field(fields, "kernel"); !failure.empty()) return failure;
  if (std::string failure = required_key_field(fields, "epilogue"); !failure.empty()) return failure;

  if (key_field(fields, "backend") != entry.selected_backend) {
    return "key_backend_mismatch";
  }
  if (key_field(fields, "semantics") != entry.semantic_contract) {
    return "key_semantics_mismatch";
  }
  if (key_field(fields, "kernel") != entry.selected_kernel || entry.kernel_family != entry.selected_kernel) {
    return "key_kernel_mismatch";
  }
  if (key_field(fields, "epilogue") != entry.epilogue) {
    return "key_epilogue_mismatch";
  }
  if (std::string failure = require_key_i64(fields, "m", entry.m); !failure.empty()) return failure;
  if (std::string failure = require_key_i64(fields, "n", entry.n); !failure.empty()) return failure;
  if (std::string failure = require_key_i64(fields, "k", entry.k); !failure.empty()) return failure;
  if (std::string failure = require_key_i64(fields, "tile_m", static_cast<int64_t>(entry.tile_m)); !failure.empty()) {
    return failure;
  }
  if (std::string failure = require_key_i64(fields, "tile_n", static_cast<int64_t>(entry.tile_n)); !failure.empty()) {
    return failure;
  }
  if (std::string failure = require_key_text(
          fields,
          "accumulator_type",
          expected_accumulator_type_for_entry(entry));
      !failure.empty()) {
    return failure;
  }
  if (std::string failure = require_key_text(
          fields,
          "accumulator_signedness",
          expected_accumulator_signedness_for_entry(entry));
      !failure.empty()) {
    return failure;
  }
  if (std::string failure = require_key_text(
          fields,
          "accumulator_modulus_policy",
          expected_accumulator_modulus_policy_for_entry(entry));
      !failure.empty()) {
    return failure;
  }
  if (std::string failure = require_key_i64(fields, "k_block_size", entry.k_block_size); !failure.empty()) {
    return failure;
  }
  if (std::string failure =
          require_key_i64(fields, "k_block_cap", expected_accumulator_k_block_cap_for_entry(entry));
      !failure.empty()) {
    return failure;
  }
  const int64_t expected_cap = expected_accumulator_k_block_cap_for_entry(entry);
  if (expected_cap > 0 && entry.k_block_size > expected_cap) {
    return "entry_k_block_size_exceeds_accumulator_cap";
  }
  if (entry.selected_backend == "hip-vector-alu-int64" && entry.k_block_size != entry.k) {
    return "vector_alu_entry_k_block_size_must_equal_k";
  }

  const bool finite_contract =
      entry.semantic_contract == "finite_ring_u8" || entry.semantic_contract == "finite_field_u8";
  if (finite_contract) {
    if (entry.finite_modulus == 0) {
      return "missing_entry_finite_modulus";
    }
    if (std::string failure = require_key_i64(fields, "finite_modulus", entry.finite_modulus); !failure.empty()) {
      return failure;
    }
  } else if (entry.finite_modulus != 0) {
    return "unexpected_entry_finite_modulus";
  } else if (!key_field(fields, "finite_modulus").empty()) {
    return "unexpected_key_finite_modulus";
  }

  if (!reviewed_autotune_kernel_supported_for_contract(entry)) {
    return "unsupported_autotune_kernel_for_contract";
  }
  if (!reviewed_autotune_epilogue_supported_for_contract(entry)) {
    return "unsupported_autotune_epilogue_for_contract";
  }

  if (std::string failure = required_key_field(fields, "target_id"); !failure.empty()) {
    return failure;
  }
  if (const std::string target = key_field(fields, "target_id"); target != entry.target_id) {
    return "key_target_id_mismatch";
  }
  if (const std::string legacy_target = key_field(fields, "target"); !legacy_target.empty() &&
      legacy_target != entry.target_id) {
    return "key_target_mismatch";
  }
  if (const std::string version = first_key_field(fields, "hip_sdk_or_library_version", "version");
      !version.empty() && version != entry.hip_sdk_or_library_version) {
    return "key_version_mismatch";
  }
  if (const std::string layout = key_field(fields, "layout"); !layout.empty() && layout != entry.layout) {
    return "key_layout_mismatch";
  }
  return {};
}

std::string validated_runtime_identity_failure(
    const AutotuneCacheEntry& entry,
    const AutotuneRuntimeIdentity& runtime) {
  if (!runtime.target_id.empty() && entry.target_id != runtime.target_id) {
    return "runtime_target_id_mismatch:" + entry.target_id + "!=" + runtime.target_id;
  }
  if (!runtime.hip_sdk_or_library_version.empty() &&
      entry.hip_sdk_or_library_version != runtime.hip_sdk_or_library_version) {
    return "runtime_version_mismatch:" + entry.hip_sdk_or_library_version + "!=" +
           runtime.hip_sdk_or_library_version;
  }
  return {};
}

}  // namespace

std::filesystem::path autotune_cache_path() {
  if (const std::string override_path = env_value("RNS8_AUTOTUNE_CACHE_PATH"); !override_path.empty()) {
    return std::filesystem::path(override_path);
  }
#if defined(_WIN32)
  if (const std::string local_app_data = env_value("LOCALAPPDATA"); !local_app_data.empty()) {
    return std::filesystem::path(local_app_data) / "rns8-gemm" / "autotune.json";
  }
  if (const std::string user_profile = env_value("USERPROFILE"); !user_profile.empty()) {
    return std::filesystem::path(user_profile) / "AppData" / "Local" / "rns8-gemm" / "autotune.json";
  }
#else
  if (const std::string xdg_cache_home = env_value("XDG_CACHE_HOME"); !xdg_cache_home.empty()) {
    return std::filesystem::path(xdg_cache_home) / "rns8-gemm" / "autotune.json";
  }
  if (const std::string home = env_value("HOME"); !home.empty()) {
    return std::filesystem::path(home) / ".cache" / "rns8-gemm" / "autotune.json";
  }
#endif
  return std::filesystem::path("rns8-gemm") / "autotune.json";
}

AutotuneCacheSnapshot read_autotune_cache() {
  AutotuneCacheSnapshot snapshot{};
  snapshot.path = autotune_cache_path();
  std::error_code ec;
  snapshot.exists = std::filesystem::exists(snapshot.path, ec);
  if (ec) {
    snapshot.error = ec.message();
    return snapshot;
  }
  if (!snapshot.exists) {
    snapshot.loaded = true;
    return snapshot;
  }

  try {
    std::ifstream input(snapshot.path);
    if (!input) {
      snapshot.error = "failed to open autotune cache";
      return snapshot;
    }
    json root = json::parse(input);
    if (!root.is_object()) {
      snapshot.error = "autotune cache root must be an object";
      return snapshot;
    }
    snapshot.schema_version = explicit_u32_field_or_zero(root, "schema_version");
    if (const auto entries = root.find("entries"); entries != root.end() && entries->is_array()) {
      for (const auto& item : *entries) {
        if (!item.is_object()) {
          continue;
        }
        AutotuneCacheEntry entry = entry_from_json(item);
        if (!entry.key.empty()) {
          snapshot.entries.push_back(std::move(entry));
        }
      }
    }
    snapshot.loaded = true;
  } catch (const std::exception& error) {
    snapshot.error = error.what();
  }
  return snapshot;
}

const AutotuneCacheEntry* find_exact_autotune_entry(
    const AutotuneCacheSnapshot& snapshot,
    const std::string& key) {
  if (key.empty()) {
    return nullptr;
  }
  for (const auto& entry : snapshot.entries) {
    if (entry.key == key) {
      return &entry;
    }
  }
  return nullptr;
}

const AutotuneCacheEntry* find_validated_autotune_entry(
    const AutotuneCacheSnapshot& snapshot,
    const std::string& key) {
  if (!snapshot.loaded || snapshot.schema_version != 1) {
    return nullptr;
  }
  const AutotuneCacheEntry* hit = find_exact_autotune_entry(snapshot, key);
  if (!hit || !hit->performance_validated || hit->schema_version != 1) {
    return nullptr;
  }
  if (hit->validation_status.rfind("reviewed_release_", 0) != 0) {
    return nullptr;
  }
  if (!validated_entry_identity_failure(*hit).empty()) {
    return nullptr;
  }
  if (!reviewed_validation_status_matches_target(*hit)) {
    return nullptr;
  }
  return hit;
}

const AutotuneCacheEntry* find_validated_autotune_entry_for_runtime(
    const AutotuneCacheSnapshot& snapshot,
    const std::string& key,
    const AutotuneRuntimeIdentity& runtime) {
  const AutotuneCacheEntry* hit = find_validated_autotune_entry(snapshot, key);
  if (!hit || !validated_runtime_identity_failure(*hit, runtime).empty()) {
    return nullptr;
  }
  return hit;
}

bool write_autotune_cache_entry(const AutotuneCacheEntry& entry, std::string& error) {
  if (entry.key.empty()) {
    error = "autotune cache entry key is empty";
    return false;
  }
  AutotuneCacheSnapshot snapshot = read_autotune_cache();
  if (!snapshot.loaded && snapshot.exists) {
    error = snapshot.error.empty() ? "failed to read existing autotune cache" : snapshot.error;
    return false;
  }
  bool replaced = false;
  for (auto& existing : snapshot.entries) {
    if (existing.key == entry.key) {
      existing = entry;
      replaced = true;
      break;
    }
  }
  if (!replaced) {
    snapshot.entries.push_back(entry);
  }

  json root;
  root["schema_version"] = uint32_t{1};
  root["entries"] = json::array();
  for (const auto& cached : snapshot.entries) {
    root["entries"].push_back(entry_to_json(cached));
  }

  const std::filesystem::path path = autotune_cache_path();
  std::error_code ec;
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      error = ec.message();
      return false;
    }
  }
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    error = "failed to open autotune cache for write";
    return false;
  }
  output << std::setw(2) << root << "\n";
  return true;
}

std::string autotune_selection_rationale(
    const AutotuneCacheSnapshot& snapshot,
    const std::string& key,
    const std::string& selected_backend) {
  return autotune_selection_rationale(snapshot, key, selected_backend, AutotuneRuntimeIdentity{});
}

std::string autotune_selection_rationale(
    const AutotuneCacheSnapshot& snapshot,
    const std::string& key,
    const std::string& selected_backend,
    const AutotuneRuntimeIdentity& runtime) {
  if (!snapshot.loaded) {
    return "cache_unavailable:" + snapshot.error;
  }
  if (const AutotuneCacheEntry* hit = find_exact_autotune_entry(snapshot, key)) {
    if (snapshot.schema_version != 1) {
      return "exact_cache_hit_rejected_cache_schema_version:" + std::to_string(snapshot.schema_version);
    }
    if (find_validated_autotune_entry_for_runtime(snapshot, key, runtime)) {
      return "exact_cache_hit_validated:" + hit->selected_backend + "/" + hit->selected_kernel;
    }
    if (!hit->performance_validated) {
      return "exact_cache_hit_rejected_unvalidated:" + hit->selected_backend + "/" + hit->selected_kernel;
    }
    if (hit->schema_version != 1) {
      return "exact_cache_hit_rejected_schema_version:" + std::to_string(hit->schema_version);
    }
    if (const std::string failure = validated_entry_identity_failure(*hit); !failure.empty()) {
      return "exact_cache_hit_rejected_identity:" + failure;
    }
    if (const std::string failure = validated_runtime_identity_failure(*hit, runtime); !failure.empty()) {
      return "exact_cache_hit_rejected_identity:" + failure;
    }
    if (hit->validation_status.rfind("reviewed_release_", 0) == 0 && !reviewed_validation_status_matches_target(*hit)) {
      return "exact_cache_hit_rejected_validation_status_target_mismatch:" + hit->validation_status;
    }
    return "exact_cache_hit_rejected_validation_status:" + hit->validation_status;
  }
  if (snapshot.schema_version != 1) {
    return "cache_unavailable_schema_version:" + std::to_string(snapshot.schema_version);
  }
  if (selected_backend == "hip-direct") {
    return "missing_cache_using_direct_hip_correctness";
  }
  if (selected_backend == "wrap64-byte-limb") {
    return "missing_cache_using_wrap64_byte_limb_correctness";
  }
  if (selected_backend == "cpu-reference") {
    return "missing_cache_using_cpu_reference";
  }
  return "missing_cache_using_capability_gated_backend";
}

}  // namespace rns8::detail
