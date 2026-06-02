#include "core/autotune_cache.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
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
    entry.measured_median_gemm_us = medians->value("rns_gemm", 0.0);
    entry.measured_median_export_us = medians->value("crt_export", 0.0);
    entry.measured_median_end_to_end_us = medians->value("end_to_end", 0.0);
  }
  entry.performance_validated = item.value("performance_validated", false);
  entry.validation_status = item.value("validation_status", "");
  entry.schema_version = item.value("schema_version", uint32_t{1});
  entry.updated_utc = item.value("updated_utc", "");
  return entry;
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
    snapshot.schema_version = root.value("schema_version", uint32_t{1});
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
  const AutotuneCacheEntry* hit = find_exact_autotune_entry(snapshot, key);
  if (!hit || !hit->performance_validated || hit->schema_version != 1) {
    return nullptr;
  }
  if (hit->validation_status.rfind("reviewed_", 0) != 0) {
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
  if (!snapshot.loaded) {
    return "cache_unavailable:" + snapshot.error;
  }
  if (const AutotuneCacheEntry* hit = find_exact_autotune_entry(snapshot, key)) {
    if (find_validated_autotune_entry(snapshot, key)) {
      return "exact_cache_hit_validated:" + hit->selected_backend + "/" + hit->selected_kernel;
    }
    if (!hit->performance_validated) {
      return "exact_cache_hit_rejected_unvalidated:" + hit->selected_backend + "/" + hit->selected_kernel;
    }
    if (hit->schema_version != 1) {
      return "exact_cache_hit_rejected_schema_version:" + std::to_string(hit->schema_version);
    }
    return "exact_cache_hit_rejected_validation_status:" + hit->validation_status;
  }
  if (selected_backend == "hip-direct") {
    return "missing_cache_using_direct_hip_correctness";
  }
  if (selected_backend == "cpu-reference" || selected_backend == "wrap64-byte-limb") {
    return "missing_cache_using_cpu_reference";
  }
  return "missing_cache_using_capability_gated_backend";
}

}  // namespace rns8::detail
