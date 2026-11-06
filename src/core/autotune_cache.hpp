#ifndef RNS8_CORE_AUTOTUNE_CACHE_HPP
#define RNS8_CORE_AUTOTUNE_CACHE_HPP

#include <cstdint>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace rns8::detail {

struct AutotuneCacheEntry {
  std::string key;
  std::string selected_backend;
  std::string selected_kernel;
  std::string target_id;
  std::string hip_sdk_or_library_version;
  std::string semantic_contract;
  uint32_t finite_modulus = 0;
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
  double measured_median_pack_a_us = 0.0;
  double measured_median_pack_b_us = 0.0;
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

struct AutotuneRuntimeIdentity {
  std::string target_id;
  std::string hip_sdk_or_library_version;
};

// Immutable owned data keeps every index/string_view valid for the whole
// selection pass, including a simultaneous replacement of the cache file.
class AutotuneSelectionSnapshot final {
public:
  explicit AutotuneSelectionSnapshot(AutotuneCacheSnapshot snapshot, std::string content_sha256);
  AutotuneSelectionSnapshot(const AutotuneSelectionSnapshot&) = delete;
  AutotuneSelectionSnapshot& operator=(const AutotuneSelectionSnapshot&) = delete;
  const AutotuneCacheSnapshot& cache() const noexcept { return cache_; }
  const std::string& content_sha256() const noexcept { return content_sha256_; }
  bool has_usable_entries() const noexcept { return !cohorts_.empty(); }
  bool plausible(std::string_view backend, std::string_view semantics, uint32_t modulus, int64_t m,
                 int64_t n, int64_t k, std::string_view target) const;
  const AutotuneCacheEntry* exact(const std::string& key) const;
  const AutotuneCacheEntry* validated(const std::string& key) const;
  const AutotuneCacheEntry* runtime(const std::string& key,
                                    const AutotuneRuntimeIdentity& identity) const;

private:
  using Cohort = std::tuple<std::string_view, std::string_view, uint32_t, int64_t, int64_t, int64_t,
                            std::string_view>;
  const AutotuneCacheSnapshot cache_;
  const std::string content_sha256_;
  struct IndexedEntry {
    const AutotuneCacheEntry* entry;
    bool reviewed;
  };
  std::unordered_map<std::string_view, IndexedEntry> exact_;
  std::set<Cohort> cohorts_;
};

// Reopen and hash complete bounded bytes on every selection. Reuse parsing and
// indexes only for the same absolute path/content, never matching timestamps.
std::shared_ptr<const AutotuneSelectionSnapshot>
refresh_autotune_snapshot(const std::shared_ptr<const AutotuneSelectionSnapshot>& previous = {});

std::filesystem::path autotune_cache_path();
AutotuneCacheSnapshot read_autotune_cache();
const AutotuneCacheEntry* find_exact_autotune_entry(const AutotuneCacheSnapshot& snapshot,
                                                    const std::string& key);
const AutotuneCacheEntry* find_validated_autotune_entry(const AutotuneCacheSnapshot& snapshot,
                                                        const std::string& key);
const AutotuneCacheEntry*
find_validated_autotune_entry_for_runtime(const AutotuneCacheSnapshot& snapshot,
                                          const std::string& key,
                                          const AutotuneRuntimeIdentity& runtime);
bool write_autotune_cache_entry(const AutotuneCacheEntry& entry, std::string& error);
std::string autotune_selection_rationale(const AutotuneCacheSnapshot& snapshot,
                                         const std::string& key,
                                         const std::string& selected_backend);
std::string autotune_selection_rationale(const AutotuneCacheSnapshot& snapshot,
                                         const std::string& key,
                                         const std::string& selected_backend,
                                         const AutotuneRuntimeIdentity& runtime);

} // namespace rns8::detail

#endif
