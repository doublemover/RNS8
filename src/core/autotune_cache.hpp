#ifndef RNS8_CORE_AUTOTUNE_CACHE_HPP
#define RNS8_CORE_AUTOTUNE_CACHE_HPP

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace rns8::detail {

struct AutotuneCacheEntry {
  std::string key;
  std::string selected_backend;
  std::string selected_kernel;
  std::string target_id;
  std::string hip_sdk_or_library_version;
  std::string semantic_contract;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  std::string layout = "row_major";
  std::string prefix_schedule_hash;
  int64_t k_block_size = 0;
  uint32_t tile_m = 0;
  uint32_t tile_n = 0;
  std::string epilogue;
  std::string kernel_family;
  uint64_t workspace_bytes = 0;
  double measured_median_pack_us = 0.0;
  double measured_median_gemm_us = 0.0;
  double measured_median_export_us = 0.0;
  double measured_median_end_to_end_us = 0.0;
  bool performance_validated = false;
  std::string validation_status;
  uint32_t schema_version = 1;
  std::string updated_utc;
};

struct AutotuneCacheSnapshot {
  std::filesystem::path path;
  uint32_t schema_version = 1;
  std::vector<AutotuneCacheEntry> entries;
  bool loaded = false;
  bool exists = false;
  std::string error;
};

std::filesystem::path autotune_cache_path();
AutotuneCacheSnapshot read_autotune_cache();
const AutotuneCacheEntry* find_exact_autotune_entry(
    const AutotuneCacheSnapshot& snapshot,
    const std::string& key);
const AutotuneCacheEntry* find_validated_autotune_entry(
    const AutotuneCacheSnapshot& snapshot,
    const std::string& key);
bool write_autotune_cache_entry(const AutotuneCacheEntry& entry, std::string& error);
std::string autotune_selection_rationale(
    const AutotuneCacheSnapshot& snapshot,
    const std::string& key,
    const std::string& selected_backend);

}  // namespace rns8::detail

#endif
