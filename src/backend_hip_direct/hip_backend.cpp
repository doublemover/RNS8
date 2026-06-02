#include "backend_hip_direct/hip_backend.hpp"

#include "core/internal.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <utility>
#include <vector>

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#  include <hip/hip_runtime_api.h>

extern "C" int rns8_hip_direct_pack_i64_device(
    const int64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix);

extern "C" int rns8_hip_direct_pack_u64_device(
    const uint64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix);

extern "C" int rns8_hip_direct_pack_u8_modulus_device(
    const uint8_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int modulus);

extern "C" int rns8_hip_direct_ring_gemm_i8_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    int modulus_index,
    int selected_prefix,
    int safe_k_block);

extern "C" int rns8_hip_direct_finite_ring_gemm_i8_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_i8_scheduled_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    const rns8_plan_tile_schedule_entry* d_schedule,
    int entry_count,
    int max_tile_row_blocks,
    int max_tile_col_blocks,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    int modulus_index,
    int selected_prefix,
    int safe_k_block);

extern "C" int rns8_hip_direct_export_u8_modulus_device(
    const int8_t* d_residues,
    uint8_t* d_dst,
    int rows,
    int cols,
    int ld,
    int modulus);

extern "C" int rns8_hip_direct_export_i64_device(
    const int8_t* d_residues,
    int64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status);

extern "C" int rns8_hip_direct_export_i64_scheduled_device(
    const int8_t* d_residues,
    int64_t* d_dst,
    const rns8_plan_tile_schedule_entry* d_schedule,
    const uint64_t* d_bounds,
    int entry_count,
    int max_tile_elements,
    int rows,
    int cols,
    int* d_status);

extern "C" int rns8_hip_direct_export_u64_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status);

extern "C" int rns8_hip_direct_export_u64_scheduled_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    const rns8_plan_tile_schedule_entry* d_schedule,
    const uint64_t* d_bounds,
    int entry_count,
    int max_tile_elements,
    int rows,
    int cols,
    int* d_status);

extern "C" int rns8_hip_direct_export_exact_wide_signed_limbs_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* d_status);

extern "C" int rns8_hip_direct_export_exact_wide_unsigned_limbs_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* d_status);
#endif

namespace rns8::detail {

namespace {

thread_local bool g_hip_direct_timing_enabled = false;
thread_local std::vector<hip_direct_timing_sample> g_hip_direct_timing_samples;
std::atomic<uint64_t> g_hip_direct_allocate_calls{0};
std::atomic<uint64_t> g_hip_direct_free_calls{0};
std::atomic<uint64_t> g_hip_direct_allocated_bytes{0};

}  // namespace

void hip_direct_timing_set_enabled(bool enabled) {
  g_hip_direct_timing_enabled = enabled;
  if (!enabled) {
    g_hip_direct_timing_samples.clear();
  }
}

bool hip_direct_timing_enabled() {
  return g_hip_direct_timing_enabled;
}

void hip_direct_timing_reset() {
  g_hip_direct_timing_samples.clear();
}

void hip_direct_timing_record_sample(const char* label, double microseconds) {
  if (!g_hip_direct_timing_enabled || !label || microseconds < 0.0) {
    return;
  }
  g_hip_direct_timing_samples.push_back({label, microseconds});
}

std::vector<hip_direct_timing_sample> hip_direct_timing_snapshot() {
  return g_hip_direct_timing_samples;
}

void hip_direct_allocation_counters_reset() {
  g_hip_direct_allocate_calls.store(0, std::memory_order_relaxed);
  g_hip_direct_free_calls.store(0, std::memory_order_relaxed);
  g_hip_direct_allocated_bytes.store(0, std::memory_order_relaxed);
}

hip_direct_allocation_counters hip_direct_allocation_counters_snapshot() {
  hip_direct_allocation_counters counters{};
  counters.allocate_calls = g_hip_direct_allocate_calls.load(std::memory_order_relaxed);
  counters.free_calls = g_hip_direct_free_calls.load(std::memory_order_relaxed);
  counters.allocated_bytes = g_hip_direct_allocated_bytes.load(std::memory_order_relaxed);
  return counters;
}

namespace {

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
template <typename Fn>
hipError_t timed_hip_operation(const char* label, Fn&& fn) {
  if (!g_hip_direct_timing_enabled) {
    return fn();
  }

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  hipError_t event_status = hipEventCreate(&start);
  if (event_status != hipSuccess) {
    return fn();
  }
  event_status = hipEventCreate(&stop);
  if (event_status != hipSuccess) {
    (void)hipEventDestroy(start);
    return fn();
  }

  event_status = hipEventRecord(start, nullptr);
  if (event_status != hipSuccess) {
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    return fn();
  }

  const hipError_t op_status = fn();
  if (op_status == hipSuccess) {
    event_status = hipEventRecord(stop, nullptr);
    if (event_status == hipSuccess) {
      event_status = hipEventSynchronize(stop);
    }
    if (event_status == hipSuccess) {
      float milliseconds = 0.0f;
      event_status = hipEventElapsedTime(&milliseconds, start, stop);
      if (event_status == hipSuccess && milliseconds >= 0.0f) {
        g_hip_direct_timing_samples.push_back({label, static_cast<double>(milliseconds) * 1000.0});
      }
    }
  }

  (void)hipEventDestroy(stop);
  (void)hipEventDestroy(start);
  return op_status;
}

bool checked_i32_shape(int64_t rows, int64_t cols, int64_t ld, uint32_t prefix) {
  if (rows <= 0 || cols <= 0 || ld < cols || prefix == 0 || prefix > RNS8_DEFAULT_MODULUS_COUNT) {
    return false;
  }
  return rows <= std::numeric_limits<int>::max() && cols <= std::numeric_limits<int>::max() &&
         ld <= std::numeric_limits<int>::max() && prefix <= static_cast<uint32_t>(std::numeric_limits<int>::max());
}

rns8_status set_hip_device(int device_id) {
  if (device_id < 0) {
    device_id = 0;
  }
  const hipError_t err = hipSetDevice(device_id);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

bool checked_pack_elements(int64_t rows, int64_t cols, uint32_t prefix) {
  const uint64_t max_pack_elements = static_cast<uint64_t>(std::numeric_limits<int>::max()) * 256u;
  return static_cast<uint64_t>(rows) <= max_pack_elements / static_cast<uint64_t>(cols) / prefix;
}

bool checked_matrix_elements_i32(int64_t rows, int64_t cols) {
  if (rows <= 0 || cols <= 0 || rows > std::numeric_limits<int>::max() || cols > std::numeric_limits<int>::max()) {
    return false;
  }
  return static_cast<uint64_t>(rows) <=
         static_cast<uint64_t>(std::numeric_limits<int>::max()) / static_cast<uint64_t>(cols);
}

bool checked_output_bytes(int64_t rows, int64_t cols, std::size_t element_size) {
  if (rows <= 0 || cols <= 0 || element_size == 0) {
    return false;
  }
  const auto max_size = std::numeric_limits<std::size_t>::max();
  return static_cast<uint64_t>(rows) <=
         static_cast<uint64_t>(max_size / element_size / static_cast<std::size_t>(cols));
}

bool checked_limb_export_pitch(int64_t ld, uint32_t limb_count) {
  if (ld <= 0 || limb_count == 0) {
    return false;
  }
  const auto max_size = std::numeric_limits<std::size_t>::max();
  return static_cast<uint64_t>(ld) <=
         static_cast<uint64_t>(max_size / sizeof(uint64_t) / static_cast<std::size_t>(limb_count));
}

bool checked_tile_entry(const rns8_plan_tile_schedule_entry& entry, int64_t rows, int64_t cols) {
  if (entry.struct_size != sizeof(rns8_plan_tile_schedule_entry) || entry.abi_version != RNS8_ABI_VERSION ||
      entry.flags != 0 || entry.row_offset < 0 || entry.col_offset < 0 || entry.row_extent <= 0 ||
      entry.col_extent <= 0 || entry.required_prefix == 0 || entry.selected_prefix == 0 ||
      entry.required_prefix > entry.selected_prefix || entry.selected_prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return false;
  }
  if (entry.row_offset > rows || entry.col_offset > cols) {
    return false;
  }
  return entry.row_extent <= rows - entry.row_offset && entry.col_extent <= cols - entry.col_offset &&
         entry.row_offset <= std::numeric_limits<int>::max() && entry.col_offset <= std::numeric_limits<int>::max() &&
         entry.row_extent <= std::numeric_limits<int>::max() && entry.col_extent <= std::numeric_limits<int>::max();
}

struct checked_tile_schedule {
  std::vector<uint64_t> row_offsets;
  std::vector<uint64_t> row_extents;
  std::vector<uint64_t> col_offsets;
  std::vector<uint64_t> col_extents;
  std::vector<uint32_t> selected_prefix_groups;
};

bool collect_unique_sorted_u64(std::vector<uint64_t>& values, uint64_t value) {
  if (std::find(values.begin(), values.end(), value) == values.end()) {
    values.push_back(value);
  }
  return values.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

bool collect_unique_sorted_u32(std::vector<uint32_t>& values, uint32_t value) {
  if (std::find(values.begin(), values.end(), value) == values.end()) {
    values.push_back(value);
  }
  return values.size() <= static_cast<std::size_t>(std::numeric_limits<uint32_t>::max());
}

bool checked_tile_axis_coverage(const std::vector<uint64_t>& offsets, const std::vector<uint64_t>& extents, uint64_t total) {
  uint64_t expected_offset = 0;
  for (std::size_t index = 0; index < offsets.size(); ++index) {
    if (expected_offset > total || offsets[index] != expected_offset || extents[index] == 0 ||
        extents[index] > total - expected_offset) {
      return false;
    }
    expected_offset += extents[index];
  }
  return expected_offset == total;
}

bool checked_tile_schedule_contract(
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count,
    int64_t rows,
    int64_t cols,
    checked_tile_schedule* out) {
  if (!entries || entry_count == 0 || entry_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      rows <= 0 || cols <= 0) {
    return false;
  }
  std::vector<uint64_t> row_ids;
  std::vector<uint64_t> col_ids;
  std::vector<uint32_t> selected_prefixes;
  for (uint64_t index = 0; index < entry_count; ++index) {
    const auto& entry = entries[static_cast<std::size_t>(index)];
    if (!checked_tile_entry(entry, rows, cols) || !collect_unique_sorted_u64(row_ids, entry.tile_row) ||
        !collect_unique_sorted_u64(col_ids, entry.tile_col) ||
        !collect_unique_sorted_u32(selected_prefixes, entry.selected_prefix)) {
      return false;
    }
  }
  std::sort(row_ids.begin(), row_ids.end());
  std::sort(col_ids.begin(), col_ids.end());
  std::sort(selected_prefixes.begin(), selected_prefixes.end());
  if (row_ids.empty() || col_ids.empty() ||
      row_ids.size() > static_cast<std::size_t>(std::numeric_limits<uint64_t>::max() / col_ids.size()) ||
      static_cast<uint64_t>(row_ids.size() * col_ids.size()) != entry_count) {
    return false;
  }
  for (std::size_t row = 0; row < row_ids.size(); ++row) {
    if (row_ids[row] != row) {
      return false;
    }
  }
  for (std::size_t col = 0; col < col_ids.size(); ++col) {
    if (col_ids[col] != col) {
      return false;
    }
  }
  checked_tile_schedule schedule{};
  schedule.row_offsets.assign(row_ids.size(), 0);
  schedule.row_extents.assign(row_ids.size(), 0);
  schedule.col_offsets.assign(col_ids.size(), 0);
  schedule.col_extents.assign(col_ids.size(), 0);
  schedule.selected_prefix_groups = selected_prefixes;
  std::vector<uint8_t> seen(static_cast<std::size_t>(entry_count), 0);
  for (uint64_t index = 0; index < entry_count; ++index) {
    const auto& entry = entries[static_cast<std::size_t>(index)];
    const std::size_t tile_row = static_cast<std::size_t>(entry.tile_row);
    const std::size_t tile_col = static_cast<std::size_t>(entry.tile_col);
    if (tile_row != static_cast<std::size_t>(index / static_cast<uint64_t>(col_ids.size())) ||
        tile_col != static_cast<std::size_t>(index % static_cast<uint64_t>(col_ids.size()))) {
      return false;
    }
    const std::size_t linear = tile_row * col_ids.size() + tile_col;
    if (linear >= seen.size() || seen[linear] != 0) {
      return false;
    }
    seen[linear] = 1;
    const uint64_t row_offset = static_cast<uint64_t>(entry.row_offset);
    const uint64_t col_offset = static_cast<uint64_t>(entry.col_offset);
    const uint64_t row_extent = static_cast<uint64_t>(entry.row_extent);
    const uint64_t col_extent = static_cast<uint64_t>(entry.col_extent);
    if (schedule.row_extents[tile_row] == 0) {
      schedule.row_offsets[tile_row] = row_offset;
      schedule.row_extents[tile_row] = row_extent;
    } else if (schedule.row_offsets[tile_row] != row_offset || schedule.row_extents[tile_row] != row_extent) {
      return false;
    }
    if (schedule.col_extents[tile_col] == 0) {
      schedule.col_offsets[tile_col] = col_offset;
      schedule.col_extents[tile_col] = col_extent;
    } else if (schedule.col_offsets[tile_col] != col_offset || schedule.col_extents[tile_col] != col_extent) {
      return false;
    }
    const auto group = std::lower_bound(selected_prefixes.begin(), selected_prefixes.end(), entry.selected_prefix);
    if (group == selected_prefixes.end() || *group != entry.selected_prefix ||
        entry.group_index != static_cast<uint32_t>(group - selected_prefixes.begin())) {
      return false;
    }
  }
  if (std::find(seen.begin(), seen.end(), uint8_t{0}) != seen.end() ||
      !checked_tile_axis_coverage(schedule.row_offsets, schedule.row_extents, static_cast<uint64_t>(rows)) ||
      !checked_tile_axis_coverage(schedule.col_offsets, schedule.col_extents, static_cast<uint64_t>(cols))) {
    return false;
  }
  if (out) {
    *out = std::move(schedule);
  }
  return true;
}

bool scheduled_tile_block_shape(
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count,
    int* max_row_blocks,
    int* max_col_blocks) {
  if (!entries || entry_count == 0 || !max_row_blocks || !max_col_blocks) {
    return false;
  }
  uint64_t row_blocks = 0;
  uint64_t col_blocks = 0;
  for (uint64_t index = 0; index < entry_count; ++index) {
    const auto& entry = entries[static_cast<std::size_t>(index)];
    const uint64_t entry_row_blocks = (static_cast<uint64_t>(entry.row_extent) + 15u) / 16u;
    const uint64_t entry_col_blocks = (static_cast<uint64_t>(entry.col_extent) + 15u) / 16u;
    row_blocks = std::max(row_blocks, entry_row_blocks);
    col_blocks = std::max(col_blocks, entry_col_blocks);
  }
  if (row_blocks == 0 || col_blocks == 0 || row_blocks > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      col_blocks > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  *max_row_blocks = static_cast<int>(row_blocks);
  *max_col_blocks = static_cast<int>(col_blocks);
  return true;
}

struct hip_rns_modulus_launch {
  const int8_t* a = nullptr;
  const int8_t* b = nullptr;
  int8_t* c = nullptr;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t lda = 0;
  int64_t ldb = 0;
  int64_t ldc = 0;
  uint16_t modulus = 0;
  uint32_t modulus_index = 0;
  uint32_t selected_prefix = 0;
};

struct hip_rns_scheduled_modulus_launch {
  const int8_t* a = nullptr;
  const int8_t* b = nullptr;
  int8_t* c = nullptr;
  const rns8_plan_tile_schedule_entry* device_entries = nullptr;
  uint64_t entry_count = 0;
  int max_tile_row_blocks = 0;
  int max_tile_col_blocks = 0;
  int64_t k = 0;
  int64_t lda = 0;
  int64_t ldb = 0;
  int64_t ldc = 0;
  uint16_t modulus = 0;
  uint32_t modulus_index = 0;
  uint32_t selected_prefix = 0;
};

bool checked_rns_modulus_launch(const hip_rns_modulus_launch& launch) {
  if (!launch.a || !launch.b || !launch.c || launch.m <= 0 || launch.n <= 0 || launch.k <= 0 ||
      launch.lda < launch.k || launch.ldb < launch.n || launch.ldc < launch.n || launch.selected_prefix == 0 ||
      launch.selected_prefix > RNS8_MAX_SUPPORTED_PREFIX || launch.modulus_index >= launch.selected_prefix ||
      launch.modulus_index >= RNS8_DEFAULT_MODULUS_COUNT ||
      launch.modulus != kDefaultModuli[launch.modulus_index]) {
    return false;
  }
  return launch.m <= std::numeric_limits<int>::max() && launch.n <= std::numeric_limits<int>::max() &&
         launch.k <= std::numeric_limits<int>::max() && launch.lda <= std::numeric_limits<int>::max() &&
         launch.ldb <= std::numeric_limits<int>::max() && launch.ldc <= std::numeric_limits<int>::max() &&
         RNS8_SAFE_INT32_K_BLOCK <= static_cast<uint32_t>(std::numeric_limits<int>::max());
}

bool checked_scheduled_modulus_launch(const hip_rns_scheduled_modulus_launch& launch) {
  if (!launch.a || !launch.b || !launch.c || !launch.device_entries || launch.entry_count == 0 ||
      launch.max_tile_row_blocks <= 0 || launch.max_tile_col_blocks <= 0 || launch.k <= 0 ||
      launch.lda < launch.k || launch.ldb <= 0 || launch.ldc <= 0 || launch.selected_prefix == 0 ||
      launch.selected_prefix > RNS8_MAX_SUPPORTED_PREFIX || launch.modulus_index >= launch.selected_prefix ||
      launch.modulus_index >= RNS8_DEFAULT_MODULUS_COUNT ||
      launch.modulus != kDefaultModuli[launch.modulus_index]) {
    return false;
  }
  return launch.entry_count <= static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
         launch.max_tile_row_blocks <= std::numeric_limits<int>::max() &&
         launch.max_tile_col_blocks <= std::numeric_limits<int>::max() &&
         launch.k <= std::numeric_limits<int>::max() && launch.lda <= std::numeric_limits<int>::max() &&
         launch.ldb <= std::numeric_limits<int>::max() && launch.ldc <= std::numeric_limits<int>::max() &&
         RNS8_SAFE_INT32_K_BLOCK <= static_cast<uint32_t>(std::numeric_limits<int>::max());
}

hipError_t launch_rns_modulus_gemm(const hip_rns_modulus_launch& launch) {
  if (!checked_rns_modulus_launch(launch)) {
    return hipErrorInvalidValue;
  }
  const int code = rns8_hip_direct_ring_gemm_i8_device(
      launch.a,
      launch.b,
      launch.c,
      static_cast<int>(launch.m),
      static_cast<int>(launch.n),
      static_cast<int>(launch.k),
      static_cast<int>(launch.lda),
      static_cast<int>(launch.ldb),
      static_cast<int>(launch.ldc),
      static_cast<int>(launch.modulus),
      static_cast<int>(launch.modulus_index),
      static_cast<int>(launch.selected_prefix),
      static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
  return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
}

hipError_t launch_rns_scheduled_modulus_gemm(const hip_rns_scheduled_modulus_launch& launch) {
  if (!checked_scheduled_modulus_launch(launch)) {
    return hipErrorInvalidValue;
  }
  const int code = rns8_hip_direct_ring_gemm_i8_scheduled_device(
      launch.a,
      launch.b,
      launch.c,
      launch.device_entries,
      static_cast<int>(launch.entry_count),
      launch.max_tile_row_blocks,
      launch.max_tile_col_blocks,
      static_cast<int>(launch.k),
      static_cast<int>(launch.lda),
      static_cast<int>(launch.ldb),
      static_cast<int>(launch.ldc),
      static_cast<int>(launch.modulus),
      static_cast<int>(launch.modulus_index),
      static_cast<int>(launch.selected_prefix),
      static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
  return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
}
#endif

}  // namespace

bool hip_direct_compiled() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  return true;
#else
  return false;
#endif
}

rns8_status hip_direct_probe(int device_id, rns8_device_info& out) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  if (err != hipSuccess || count <= 0) {
    copy_c_string(out.detail, sizeof(out.detail), hipGetErrorString(err));
    return RNS8_UNSUPPORTED_BACKEND;
  }
  if (device_id < 0) {
    device_id = 0;
  }
  if (device_id >= count) {
    return RNS8_INVALID_ARGUMENT;
  }

  hipDeviceProp_t prop{};
  err = hipGetDeviceProperties(&prop, device_id);
  if (err != hipSuccess) {
    copy_c_string(out.detail, sizeof(out.detail), hipGetErrorString(err));
    return RNS8_BACKEND_FAILURE;
  }

  int runtime_version = 0;
  int driver_version = 0;
  (void)hipRuntimeGetVersion(&runtime_version);
  (void)hipDriverGetVersion(&driver_version);

  out.backend = RNS8_BACKEND_HIP_DIRECT;
  out.device_id = device_id;
  out.hip_available = 1;
  out.hip_runtime_version = static_cast<uint32_t>(runtime_version);
  out.hip_driver_version = static_cast<uint32_t>(driver_version);
  out.global_mem_bytes = static_cast<uint64_t>(prop.totalGlobalMem);
  copy_c_string(out.name, sizeof(out.name), prop.name);
  copy_c_string(out.gcn_arch, sizeof(out.gcn_arch), prop.gcnArchName);
  copy_c_string(out.detail, sizeof(out.detail), "direct HIP runtime detected");
  return RNS8_SUCCESS;
#else
  (void)device_id;
  copy_c_string(out.detail, sizeof(out.detail), "RNS8 was built without the direct HIP backend");
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_allocate(int device_id, std::size_t bytes, void** out) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!out || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  *out = nullptr;
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  void* ptr = nullptr;
  const hipError_t err = hipMalloc(&ptr, bytes);
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  g_hip_direct_allocate_calls.fetch_add(1, std::memory_order_relaxed);
  g_hip_direct_allocated_bytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed);
  *out = ptr;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)bytes;
  (void)out;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_free(int device_id, void* ptr) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!ptr) {
    return RNS8_SUCCESS;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = hipFree(ptr);
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  g_hip_direct_free_calls.fetch_add(1, std::memory_order_relaxed);
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)ptr;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_zero(int device_id, void* ptr, std::size_t bytes) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!ptr || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = hipMemset(ptr, 0, bytes);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)ptr;
  (void)bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_copy_device_to_host(int device_id, void* dst, const void* src, std::size_t bytes) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!dst || !src || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err =
      timed_hip_operation("residue_d2h_sync", [&]() { return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost); });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)dst;
  (void)src;
  (void)bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_copy_host_to_device(int device_id, void* dst, const void* src, std::size_t bytes) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!dst || !src || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err =
      timed_hip_operation("residue_h2d_sync", [&]() { return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice); });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)dst;
  (void)src;
  (void)bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_ensure_upload_buffer(int device_id, std::size_t bytes, void** buffer, std::size_t* capacity) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!buffer || !capacity || bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (*buffer && *capacity >= bytes) {
    return RNS8_SUCCESS;
  }
  if (*buffer) {
    const rns8_status free_status = hip_direct_free(device_id, *buffer);
    if (free_status != RNS8_SUCCESS) {
      return free_status;
    }
    *buffer = nullptr;
    *capacity = 0;
  }
  void* ptr = nullptr;
  const rns8_status alloc_status = hip_direct_allocate(device_id, bytes, &ptr);
  if (alloc_status != RNS8_SUCCESS) {
    return alloc_status;
  }
  *buffer = ptr;
  *capacity = bytes;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)bytes;
  (void)buffer;
  (void)capacity;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_pack_i64_device(
    int device_id,
    const int64_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src || !upload_buffer || !upload_bytes || !device_residues || !checked_i32_shape(rows, cols, ld, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_pack_elements(rows, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t source_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(ld) * sizeof(int64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, source_bytes, upload_buffer, upload_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err =
      timed_hip_operation("pack_h2d", [&]() { return hipMemcpy(*upload_buffer, src, source_bytes, hipMemcpyHostToDevice); });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("pack_kernel", [&]() {
    const int code = rns8_hip_direct_pack_i64_device(
        static_cast<const int64_t*>(*upload_buffer),
        static_cast<int8_t*>(device_residues),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(ld),
        static_cast<int>(prefix));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)src;
  (void)upload_buffer;
  (void)upload_bytes;
  (void)device_residues;
  (void)rows;
  (void)cols;
  (void)ld;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_pack_u64_device(
    int device_id,
    const uint64_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src || !upload_buffer || !upload_bytes || !device_residues || !checked_i32_shape(rows, cols, ld, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_pack_elements(rows, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t source_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(ld) * sizeof(uint64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, source_bytes, upload_buffer, upload_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err =
      timed_hip_operation("pack_h2d", [&]() { return hipMemcpy(*upload_buffer, src, source_bytes, hipMemcpyHostToDevice); });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("pack_kernel", [&]() {
    const int code = rns8_hip_direct_pack_u64_device(
        static_cast<const uint64_t*>(*upload_buffer),
        static_cast<int8_t*>(device_residues),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(ld),
        static_cast<int>(prefix));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)src;
  (void)upload_buffer;
  (void)upload_bytes;
  (void)device_residues;
  (void)rows;
  (void)cols;
  (void)ld;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_ring_gemm_i8_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus,
    uint32_t modulus_index,
    uint32_t selected_prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_rns_modulus_launch launch{};
  launch.a = static_cast<const int8_t*>(device_a_residues);
  launch.b = static_cast<const int8_t*>(device_b_residues);
  launch.c = static_cast<int8_t*>(device_c_residues);
  launch.m = m;
  launch.n = n;
  launch.k = k;
  launch.lda = lda;
  launch.ldb = ldb;
  launch.ldc = ldc;
  launch.modulus = modulus;
  launch.modulus_index = modulus_index;
  launch.selected_prefix = selected_prefix;
  if (!checked_rns_modulus_launch(launch)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("rns_gemm_kernel", [&]() {
    const hipError_t launch_status = launch_rns_modulus_gemm(launch);
    return launch_status == hipSuccess ? hipDeviceSynchronize() : launch_status;
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)modulus;
  (void)modulus_index;
  (void)selected_prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_rns_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_base = static_cast<const int8_t*>(device_a_residues);
  const auto* b_base = static_cast<const int8_t*>(device_b_residues);
  auto* c_base = static_cast<int8_t*>(device_c_residues);
  const hipError_t err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
    for (uint32_t p = 0; p < prefix; ++p) {
      const std::size_t a_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(m) *
                                   static_cast<std::size_t>(lda);
      const std::size_t b_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(k) *
                                   static_cast<std::size_t>(ldb);
      const std::size_t c_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(m) *
                                   static_cast<std::size_t>(ldc);
      const hipError_t launch_status = launch_rns_modulus_gemm({
          a_base + a_offset,
          b_base + b_offset,
          c_base + c_offset,
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          kDefaultModuli[p],
          p,
          prefix});
      if (launch_status != hipSuccess) {
        return launch_status;
      }
    }
    return hipDeviceSynchronize();
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_rns_tiled_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_residues || !device_c_residues || !entries || entry_count == 0 || m <= 0 ||
      n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max() ||
      entry_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return RNS8_INVALID_ARGUMENT;
  }
  checked_tile_schedule schedule{};
  if (!checked_tile_schedule_contract(entries, entry_count, m, n, &schedule)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_base = static_cast<const int8_t*>(device_a_residues);
  const auto* b_base = static_cast<const int8_t*>(device_b_residues);
  auto* c_base = static_cast<int8_t*>(device_c_residues);
  const hipError_t err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
    for (const uint32_t selected_prefix : schedule.selected_prefix_groups) {
      for (uint32_t p = 0; p < selected_prefix; ++p) {
        const std::size_t a_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(m) *
                                     static_cast<std::size_t>(lda);
        const std::size_t b_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(k) *
                                     static_cast<std::size_t>(ldb);
        const std::size_t c_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(m) *
                                     static_cast<std::size_t>(ldc);
        for (uint64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
          const auto& entry = entries[static_cast<std::size_t>(entry_index)];
          if (entry.selected_prefix != selected_prefix) {
            continue;
          }
          const hipError_t launch_status = launch_rns_modulus_gemm({
              a_base + a_offset + static_cast<std::size_t>(entry.row_offset) * static_cast<std::size_t>(lda),
              b_base + b_offset + static_cast<std::size_t>(entry.col_offset),
              c_base + c_offset + static_cast<std::size_t>(entry.row_offset) * static_cast<std::size_t>(ldc) +
                  static_cast<std::size_t>(entry.col_offset),
              entry.row_extent,
              entry.col_extent,
              k,
              lda,
              ldb,
              ldc,
              kDefaultModuli[p],
              p,
              entry.selected_prefix});
          if (launch_status != hipSuccess) {
            return launch_status;
          }
        }
      }
    }
    return hipDeviceSynchronize();
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)entries;
  (void)entry_count;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_rns_tiled_device_schedule(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    const rns8_plan_tile_schedule_entry* host_entries,
    const void* device_entries,
    uint64_t entry_count) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_residues || !device_c_residues || !host_entries || !device_entries ||
      entry_count == 0 || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max() ||
      entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return RNS8_INVALID_ARGUMENT;
  }
  checked_tile_schedule schedule{};
  if (!checked_tile_schedule_contract(host_entries, entry_count, m, n, &schedule)) {
    return RNS8_INVALID_ARGUMENT;
  }
  int max_tile_row_blocks = 0;
  int max_tile_col_blocks = 0;
  if (!scheduled_tile_block_shape(host_entries, entry_count, &max_tile_row_blocks, &max_tile_col_blocks)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_base = static_cast<const int8_t*>(device_a_residues);
  const auto* b_base = static_cast<const int8_t*>(device_b_residues);
  auto* c_base = static_cast<int8_t*>(device_c_residues);
  const auto* schedule_base = static_cast<const rns8_plan_tile_schedule_entry*>(device_entries);
  const uint32_t max_selected_prefix = schedule.selected_prefix_groups.back();
  const hipError_t err = timed_hip_operation("rns_gemm_scheduled_kernel_group", [&]() {
    for (uint32_t p = 0; p < max_selected_prefix; ++p) {
      const std::size_t a_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(m) *
                                   static_cast<std::size_t>(lda);
      const std::size_t b_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(k) *
                                   static_cast<std::size_t>(ldb);
      const std::size_t c_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(m) *
                                   static_cast<std::size_t>(ldc);
      const hipError_t launch_status = launch_rns_scheduled_modulus_gemm({
          a_base + a_offset,
          b_base + b_offset,
          c_base + c_offset,
          schedule_base,
          entry_count,
          max_tile_row_blocks,
          max_tile_col_blocks,
          k,
          lda,
          ldb,
          ldc,
          kDefaultModuli[p],
          p,
          max_selected_prefix});
      if (launch_status != hipSuccess) {
        return launch_status;
      }
    }
    return hipDeviceSynchronize();
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)host_entries;
  (void)device_entries;
  (void)entry_count;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_i64_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t bound,
    int64_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !status_buffer || !status_bytes || !dst ||
      ld < cols || !checked_matrix_elements_i32(rows, cols) || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(int64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, output_bytes, export_buffer, export_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation("crt_export_status_memset", [&]() {
    return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("crt_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_i64_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<int64_t*>(*export_buffer),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        bound,
        static_cast<int*>(*status_buffer));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  int host_status = 0;
  err = timed_hip_operation("crt_export_status_d2h", [&]() {
    return hipMemcpy(&host_status, *status_buffer, sizeof(host_status), hipMemcpyDeviceToHost);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (host_status != static_cast<int>(RNS8_SUCCESS)) {
    return static_cast<rns8_status>(host_status);
  }
  err = timed_hip_operation("crt_export_d2h", [&]() {
    return hipMemcpy2D(
        dst,
        static_cast<std::size_t>(ld) * sizeof(int64_t),
        *export_buffer,
        static_cast<std::size_t>(cols) * sizeof(int64_t),
        static_cast<std::size_t>(cols) * sizeof(int64_t),
        static_cast<std::size_t>(rows),
        hipMemcpyDeviceToHost);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_residues;
  (void)export_buffer;
  (void)export_bytes;
  (void)status_buffer;
  (void)status_bytes;
  (void)rows;
  (void)cols;
  (void)prefix;
  (void)bound;
  (void)dst;
  (void)ld;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_i64_tiled_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    const void* device_entries,
    const void* device_bounds,
    uint64_t entry_count,
    uint64_t max_tile_elements,
    int64_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !status_buffer || !status_bytes || !device_entries ||
      !device_bounds ||
      !dst || ld < cols || !checked_matrix_elements_i32(rows, cols) ||
      !checked_output_bytes(rows, cols, sizeof(int64_t)) || entry_count == 0 ||
      entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) || max_tile_elements == 0 ||
      max_tile_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return RNS8_INVALID_ARGUMENT;
  }
  constexpr uint64_t export_threads = 256;
  const uint64_t blocks_per_tile = (max_tile_elements + export_threads - 1u) / export_threads;
  if (blocks_per_tile == 0 ||
      entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) / blocks_per_tile) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(int64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, output_bytes, export_buffer, export_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation("crt_export_status_memset", [&]() {
    return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("crt_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_i64_scheduled_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<int64_t*>(*export_buffer),
        static_cast<const rns8_plan_tile_schedule_entry*>(device_entries),
        static_cast<const uint64_t*>(device_bounds),
        static_cast<int>(entry_count),
        static_cast<int>(max_tile_elements),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int*>(*status_buffer));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  int host_status = 0;
  err = timed_hip_operation("crt_export_status_d2h", [&]() {
    return hipMemcpy(&host_status, *status_buffer, sizeof(host_status), hipMemcpyDeviceToHost);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (host_status != static_cast<int>(RNS8_SUCCESS)) {
    return static_cast<rns8_status>(host_status);
  }
  err = timed_hip_operation("crt_export_d2h", [&]() {
    return hipMemcpy2D(
        dst,
        static_cast<std::size_t>(ld) * sizeof(int64_t),
        *export_buffer,
        static_cast<std::size_t>(cols) * sizeof(int64_t),
        static_cast<std::size_t>(cols) * sizeof(int64_t),
        static_cast<std::size_t>(rows),
        hipMemcpyDeviceToHost);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_residues;
  (void)export_buffer;
  (void)export_bytes;
  (void)status_buffer;
  (void)status_bytes;
  (void)rows;
  (void)cols;
  (void)device_entries;
  (void)device_bounds;
  (void)entry_count;
  (void)max_tile_elements;
  (void)dst;
  (void)ld;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_u64_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t bound,
    uint64_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !status_buffer || !status_bytes || !dst ||
      ld < cols || !checked_matrix_elements_i32(rows, cols) || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(uint64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, output_bytes, export_buffer, export_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation("crt_export_status_memset", [&]() {
    return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("crt_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_u64_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<uint64_t*>(*export_buffer),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        bound,
        static_cast<int*>(*status_buffer));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  int host_status = 0;
  err = timed_hip_operation("crt_export_status_d2h", [&]() {
    return hipMemcpy(&host_status, *status_buffer, sizeof(host_status), hipMemcpyDeviceToHost);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (host_status != static_cast<int>(RNS8_SUCCESS)) {
    return static_cast<rns8_status>(host_status);
  }
  err = timed_hip_operation("crt_export_d2h", [&]() {
    return hipMemcpy2D(
        dst,
        static_cast<std::size_t>(ld) * sizeof(uint64_t),
        *export_buffer,
        static_cast<std::size_t>(cols) * sizeof(uint64_t),
        static_cast<std::size_t>(cols) * sizeof(uint64_t),
        static_cast<std::size_t>(rows),
        hipMemcpyDeviceToHost);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_residues;
  (void)export_buffer;
  (void)export_bytes;
  (void)status_buffer;
  (void)status_bytes;
  (void)rows;
  (void)cols;
  (void)prefix;
  (void)bound;
  (void)dst;
  (void)ld;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_u64_tiled_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    const void* device_entries,
    const void* device_bounds,
    uint64_t entry_count,
    uint64_t max_tile_elements,
    uint64_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !status_buffer || !status_bytes || !device_entries ||
      !device_bounds ||
      !dst || ld < cols || !checked_matrix_elements_i32(rows, cols) ||
      !checked_output_bytes(rows, cols, sizeof(uint64_t)) || entry_count == 0 ||
      entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) || max_tile_elements == 0 ||
      max_tile_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return RNS8_INVALID_ARGUMENT;
  }
  constexpr uint64_t export_threads = 256;
  const uint64_t blocks_per_tile = (max_tile_elements + export_threads - 1u) / export_threads;
  if (blocks_per_tile == 0 ||
      entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) / blocks_per_tile) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(uint64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, output_bytes, export_buffer, export_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation("crt_export_status_memset", [&]() {
    return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("crt_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_u64_scheduled_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<uint64_t*>(*export_buffer),
        static_cast<const rns8_plan_tile_schedule_entry*>(device_entries),
        static_cast<const uint64_t*>(device_bounds),
        static_cast<int>(entry_count),
        static_cast<int>(max_tile_elements),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int*>(*status_buffer));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  int host_status = 0;
  err = timed_hip_operation("crt_export_status_d2h", [&]() {
    return hipMemcpy(&host_status, *status_buffer, sizeof(host_status), hipMemcpyDeviceToHost);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (host_status != static_cast<int>(RNS8_SUCCESS)) {
    return static_cast<rns8_status>(host_status);
  }
  err = timed_hip_operation("crt_export_d2h", [&]() {
    return hipMemcpy2D(
        dst,
        static_cast<std::size_t>(ld) * sizeof(uint64_t),
        *export_buffer,
        static_cast<std::size_t>(cols) * sizeof(uint64_t),
        static_cast<std::size_t>(cols) * sizeof(uint64_t),
        static_cast<std::size_t>(rows),
        hipMemcpyDeviceToHost);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_residues;
  (void)export_buffer;
  (void)export_bytes;
  (void)status_buffer;
  (void)status_bytes;
  (void)rows;
  (void)cols;
  (void)device_entries;
  (void)device_bounds;
  (void)entry_count;
  (void)max_tile_elements;
  (void)dst;
  (void)ld;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_exact_wide_signed_limbs_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !status_buffer || !status_bytes || !dst ||
      ld < cols || !checked_matrix_elements_i32(rows, cols) || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      limb_count == 0 || limb_count > 32 || !checked_limb_export_pitch(ld, limb_count)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) *
                                   static_cast<std::size_t>(limb_count) * sizeof(uint64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, output_bytes, export_buffer, export_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation(
      "exact_wide_export_status_memset", [&]() { return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr); });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("exact_wide_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_exact_wide_signed_limbs_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<uint64_t*>(*export_buffer),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        static_cast<int>(limb_count),
        static_cast<int*>(*status_buffer));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  int host_status = 0;
  err = timed_hip_operation("exact_wide_export_status_d2h", [&]() {
    return hipMemcpy(&host_status, *status_buffer, sizeof(host_status), hipMemcpyDeviceToHost);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (host_status != static_cast<int>(RNS8_SUCCESS)) {
    return static_cast<rns8_status>(host_status);
  }
  err = timed_hip_operation("exact_wide_export_d2h", [&]() {
    return hipMemcpy2D(
        dst,
        static_cast<std::size_t>(ld) * static_cast<std::size_t>(limb_count) * sizeof(uint64_t),
        *export_buffer,
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(limb_count) * sizeof(uint64_t),
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(limb_count) * sizeof(uint64_t),
        static_cast<std::size_t>(rows),
        hipMemcpyDeviceToHost);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_residues;
  (void)export_buffer;
  (void)export_bytes;
  (void)status_buffer;
  (void)status_bytes;
  (void)rows;
  (void)cols;
  (void)prefix;
  (void)dst;
  (void)ld;
  (void)limb_count;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_exact_wide_unsigned_limbs_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !status_buffer || !status_bytes || !dst ||
      ld < cols || !checked_matrix_elements_i32(rows, cols) || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      limb_count == 0 || limb_count > 32 || !checked_limb_export_pitch(ld, limb_count)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) *
                                   static_cast<std::size_t>(limb_count) * sizeof(uint64_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, output_bytes, export_buffer, export_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation(
      "exact_wide_export_status_memset", [&]() { return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr); });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("exact_wide_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_exact_wide_unsigned_limbs_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<uint64_t*>(*export_buffer),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        static_cast<int>(limb_count),
        static_cast<int*>(*status_buffer));
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  int host_status = 0;
  err = timed_hip_operation("exact_wide_export_status_d2h", [&]() {
    return hipMemcpy(&host_status, *status_buffer, sizeof(host_status), hipMemcpyDeviceToHost);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (host_status != static_cast<int>(RNS8_SUCCESS)) {
    return static_cast<rns8_status>(host_status);
  }
  err = timed_hip_operation("exact_wide_export_d2h", [&]() {
    return hipMemcpy2D(
        dst,
        static_cast<std::size_t>(ld) * static_cast<std::size_t>(limb_count) * sizeof(uint64_t),
        *export_buffer,
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(limb_count) * sizeof(uint64_t),
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(limb_count) * sizeof(uint64_t),
        static_cast<std::size_t>(rows),
        hipMemcpyDeviceToHost);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_residues;
  (void)export_buffer;
  (void)export_bytes;
  (void)status_buffer;
  (void)status_bytes;
  (void)rows;
  (void)cols;
  (void)prefix;
  (void)dst;
  (void)ld;
  (void)limb_count;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_finite_u8_gemm_oneshot_device(
    int device_id,
    const uint8_t* A,
    int64_t lda,
    const uint8_t* B,
    int64_t ldb,
    uint8_t* C,
    int64_t ldc,
    int64_t m,
    int64_t n,
    int64_t k,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!A || !B || !C || modulus < 2 || modulus > 256 || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n ||
      ldc < n || m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_output_bytes(m, lda, sizeof(uint8_t)) || !checked_output_bytes(k, ldb, sizeof(uint8_t)) ||
      !checked_output_bytes(m, ldc, sizeof(uint8_t)) || !checked_output_bytes(m, k, sizeof(int8_t)) ||
      !checked_output_bytes(k, n, sizeof(int8_t)) || !checked_output_bytes(m, n, sizeof(int8_t))) {
    return RNS8_INVALID_ARGUMENT;
  }

  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }

  const std::size_t a_source_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(lda) * sizeof(uint8_t);
  const std::size_t b_source_bytes = static_cast<std::size_t>(k) * static_cast<std::size_t>(ldb) * sizeof(uint8_t);
  const std::size_t c_output_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * sizeof(uint8_t);
  const std::size_t a_residue_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(k) * sizeof(int8_t);
  const std::size_t b_residue_bytes = static_cast<std::size_t>(k) * static_cast<std::size_t>(n) * sizeof(int8_t);
  const std::size_t c_residue_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * sizeof(int8_t);

  void* d_a_src = nullptr;
  void* d_b_src = nullptr;
  void* d_c_dst = nullptr;
  void* d_a_residues = nullptr;
  void* d_b_residues = nullptr;
  void* d_c_residues = nullptr;

  rns8_status status = hip_direct_allocate(device_id, a_source_bytes, &d_a_src);
  if (status == RNS8_SUCCESS) status = hip_direct_allocate(device_id, b_source_bytes, &d_b_src);
  if (status == RNS8_SUCCESS) status = hip_direct_allocate(device_id, c_output_bytes, &d_c_dst);
  if (status == RNS8_SUCCESS) status = hip_direct_allocate(device_id, a_residue_bytes, &d_a_residues);
  if (status == RNS8_SUCCESS) status = hip_direct_allocate(device_id, b_residue_bytes, &d_b_residues);
  if (status == RNS8_SUCCESS) status = hip_direct_allocate(device_id, c_residue_bytes, &d_c_residues);

  if (status == RNS8_SUCCESS) {
    hipError_t err = timed_hip_operation("finite_pack_h2d", [&]() {
      hipError_t copy_status = hipMemcpy(d_a_src, A, a_source_bytes, hipMemcpyHostToDevice);
      if (copy_status != hipSuccess) {
        return copy_status;
      }
      return hipMemcpy(d_b_src, B, b_source_bytes, hipMemcpyHostToDevice);
    });
    if (err != hipSuccess) {
      status = RNS8_BACKEND_FAILURE;
    }
  }

  if (status == RNS8_SUCCESS) {
    hipError_t err = timed_hip_operation("finite_pack_kernel", [&]() {
      int code = rns8_hip_direct_pack_u8_modulus_device(
          static_cast<const uint8_t*>(d_a_src),
          static_cast<int8_t*>(d_a_residues),
          static_cast<int>(m),
          static_cast<int>(k),
          static_cast<int>(lda),
          static_cast<int>(modulus));
      if (code != static_cast<int>(hipSuccess)) {
        return static_cast<hipError_t>(code);
      }
      code = rns8_hip_direct_pack_u8_modulus_device(
          static_cast<const uint8_t*>(d_b_src),
          static_cast<int8_t*>(d_b_residues),
          static_cast<int>(k),
          static_cast<int>(n),
          static_cast<int>(ldb),
          static_cast<int>(modulus));
      if (code != static_cast<int>(hipSuccess)) {
        return static_cast<hipError_t>(code);
      }
      return hipDeviceSynchronize();
    });
    if (err != hipSuccess) {
      status = RNS8_BACKEND_FAILURE;
    }
  }

  if (status == RNS8_SUCCESS) {
    hipError_t err = timed_hip_operation("finite_ring_gemm_kernel", [&]() {
      const int code = rns8_hip_direct_finite_ring_gemm_i8_device(
          static_cast<const int8_t*>(d_a_residues),
          static_cast<const int8_t*>(d_b_residues),
          static_cast<int8_t*>(d_c_residues),
          static_cast<int>(m),
          static_cast<int>(n),
          static_cast<int>(k),
          static_cast<int>(k),
          static_cast<int>(n),
          static_cast<int>(n),
          static_cast<int>(modulus),
          static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
      return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
    });
    if (err != hipSuccess) {
      status = RNS8_BACKEND_FAILURE;
    }
  }

  if (status == RNS8_SUCCESS) {
    hipError_t err = timed_hip_operation("finite_export_kernel", [&]() {
      const int code = rns8_hip_direct_export_u8_modulus_device(
          static_cast<const int8_t*>(d_c_residues),
          static_cast<uint8_t*>(d_c_dst),
          static_cast<int>(m),
          static_cast<int>(n),
          static_cast<int>(n),
          static_cast<int>(modulus));
      return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
    });
    if (err != hipSuccess) {
      status = RNS8_BACKEND_FAILURE;
    }
  }

  if (status == RNS8_SUCCESS) {
    std::vector<uint8_t> compact(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    const hipError_t err = timed_hip_operation("finite_export_d2h", [&]() {
      return hipMemcpy(compact.data(), d_c_dst, c_output_bytes, hipMemcpyDeviceToHost);
    });
    if (err != hipSuccess) {
      status = RNS8_BACKEND_FAILURE;
    } else {
      for (int64_t row = 0; row < m; ++row) {
        const uint8_t* src = compact.data() + static_cast<std::size_t>(row * n);
        uint8_t* dst = C + static_cast<std::size_t>(row * ldc);
        std::copy(src, src + n, dst);
      }
    }
  }

  for (void* ptr : {d_c_residues, d_b_residues, d_a_residues, d_c_dst, d_b_src, d_a_src}) {
    if (ptr) {
      const rns8_status free_status = hip_direct_free(device_id, ptr);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
    }
  }
  return status;
#else
  (void)device_id;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  (void)m;
  (void)n;
  (void)k;
  (void)modulus;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_synchronize(int device_id) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = hipDeviceSynchronize();
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

}  // namespace rns8::detail
