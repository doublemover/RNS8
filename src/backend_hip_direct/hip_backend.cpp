#include "backend_hip_direct/hip_backend.hpp"

#include "core/backend_common.hpp"
#include "core/internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
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
    uint32_t modulus_reciprocal,
    int modulus_index,
    int selected_prefix,
    int safe_k_block);
extern "C" int rns8_hip_direct_ring_gemm_i8_device_on_stream(
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
    uint32_t modulus_reciprocal,
    int modulus_index,
    int selected_prefix,
    int safe_k_block,
    void* stream);

extern "C" int rns8_hip_direct_ring_gemm_i8_grouped_prefix_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block);
extern "C" int rns8_hip_direct_ring_gemm_i8_grouped_prefix_device_on_stream(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block,
    void* stream);

extern "C" int rns8_hip_direct_ring_gemm_i64_native_prefix9_device(
    const int64_t* d_a,
    const int64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_i64_native_prefix9_colpair_device(
    const int64_t* d_a,
    const int64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_i64_native_a_resident_b_prefix9_device(
    const int64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_i64_uniform_small_native_a_resident_b_prefix9_device(
    const int64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_u64_native_prefix9_device(
    const uint64_t* d_a,
    const uint64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_u64_native_prefix9_colpair_device(
    const uint64_t* d_a,
    const uint64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_u64_native_a_resident_b_prefix9_device(
    const uint64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_u64_native_a_resident_b_prefix9_colpair_device(
    const uint64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_u64_resident_a_native_b_prefix9_colpair_device(
    const int8_t* d_a,
    const uint64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_u64_uniform_small_native_a_resident_b_prefix9_device(
    const uint64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_uniform_small_i8_ab_resident_b_prefix9_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_uniform_small_i8_ab_resident_b_prefix9_colpair_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
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
    uint32_t modulus_reciprocal,
    int safe_k_block);

extern "C" int rns8_hip_direct_finite_ring_gemm_u8_native_device(
    const uint8_t* d_a,
    const uint8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int safe_k_block);

extern "C" int rns8_hip_direct_finite_ring_gemm_u8_native_a_i8_b_device(
    const uint8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int safe_k_block);

extern "C" int rns8_hip_direct_ring_gemm_i8_scheduled_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    const rns8_plan_tile_schedule_entry* d_schedule,
    const uint8_t* d_zero_a_rows,
    const uint8_t* d_zero_b_cols,
    int entry_count,
    int max_tile_row_blocks,
    int max_tile_col_blocks,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int modulus_index,
    int selected_prefix,
    int safe_k_block);

extern "C" int rns8_hip_direct_zero_scheduled_residue_tiles_device(
    int8_t* d_c,
    const rns8_plan_tile_schedule_entry* d_schedule,
    int entry_count,
    int max_tile_elements,
    int rows,
    int ldc);

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
    const uint8_t* d_zero_a_rows,
    const uint8_t* d_zero_b_cols,
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
    const uint8_t* d_zero_a_rows,
    const uint8_t* d_zero_b_cols,
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

extern "C" int rns8_hip_direct_export_exact_wide_signed_grouped_limbs_device(
    const int8_t* const* d_residue_ptrs,
    uint64_t* d_dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    int limb_count);

extern "C" int rns8_hip_direct_export_exact_wide_unsigned_limbs_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* d_status);

extern "C" int rns8_hip_direct_export_exact_wide_unsigned_grouped_limbs_device(
    const int8_t* const* d_residue_ptrs,
    uint64_t* d_dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    int limb_count);
#endif

namespace rns8::detail {

namespace {

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
struct hip_direct_pending_timing_sample {
  std::vector<std::string> labels;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

struct pinned_host_staging_buffer {
  int device_id = -1;
  void* ptr = nullptr;
  std::size_t capacity = 0;

  ~pinned_host_staging_buffer() {
    if (ptr) {
      (void)hipHostFree(ptr);
    }
  }
};
#endif

thread_local bool g_hip_direct_timing_enabled = false;
thread_local std::vector<hip_direct_timing_sample> g_hip_direct_timing_samples;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
constexpr std::size_t kMaxPendingTimingEventsBeforeFlush = 16;
constexpr std::size_t kPinnedExportStagingMinBytes = 64u * 1024u;
thread_local std::vector<hip_direct_pending_timing_sample> g_hip_direct_pending_timing_samples;
thread_local pinned_host_staging_buffer g_pinned_export_staging;
#endif
std::atomic<uint64_t> g_hip_direct_allocate_calls{0};
std::atomic<uint64_t> g_hip_direct_free_calls{0};
std::atomic<uint64_t> g_hip_direct_allocated_bytes{0};

constexpr uint32_t kKnownTileScheduleFlags =
    RNS8_TILE_SCHEDULE_ZERO_OUTPUT | RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT;

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
void destroy_pending_event_pair(hipEvent_t start, hipEvent_t stop) {
  if (stop) {
    (void)hipEventDestroy(stop);
  }
  if (start) {
    (void)hipEventDestroy(start);
  }
}

void flush_pending_timing_events() {
  for (auto& pending : g_hip_direct_pending_timing_samples) {
    if (!pending.start || !pending.stop || pending.labels.empty()) {
      destroy_pending_event_pair(pending.start, pending.stop);
      continue;
    }
    hipError_t status = hipEventSynchronize(pending.stop);
    if (status != hipSuccess) {
      (void)hipDeviceSynchronize();
      status = hipEventSynchronize(pending.stop);
    }
    if (status == hipSuccess) {
      float milliseconds = 0.0f;
      status = hipEventElapsedTime(&milliseconds, pending.start, pending.stop);
      if (status != hipSuccess) {
        (void)hipDeviceSynchronize();
        status = hipEventElapsedTime(&milliseconds, pending.start, pending.stop);
      }
      if (status == hipSuccess && milliseconds >= 0.0f) {
        const double microseconds = static_cast<double>(milliseconds) * 1000.0;
        for (const auto& label : pending.labels) {
          if (!label.empty()) {
            g_hip_direct_timing_samples.push_back({label, microseconds});
          }
        }
      }
    }
    destroy_pending_event_pair(pending.start, pending.stop);
  }
  g_hip_direct_pending_timing_samples.clear();
}

void flush_pending_timing_events_if_full() {
  if (g_hip_direct_pending_timing_samples.size() >= kMaxPendingTimingEventsBeforeFlush) {
    flush_pending_timing_events();
  }
}
#endif

}  // namespace

void hip_direct_timing_set_enabled(bool enabled) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!enabled) {
    flush_pending_timing_events();
  }
#endif
  g_hip_direct_timing_enabled = enabled;
  if (!enabled) {
    g_hip_direct_timing_samples.clear();
  }
}

bool hip_direct_timing_enabled() {
  return g_hip_direct_timing_enabled;
}

void hip_direct_timing_reset() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  flush_pending_timing_events();
#endif
  g_hip_direct_timing_samples.clear();
}

void hip_direct_timing_flush_pending_events() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  flush_pending_timing_events();
#endif
}

void hip_direct_timing_record_sample(const char* label, double microseconds) {
  if (!g_hip_direct_timing_enabled || !label || microseconds < 0.0) {
    return;
  }
  g_hip_direct_timing_samples.push_back({label, microseconds});
}

void hip_direct_timing_record_pending_event(const char* label, void* start_event, void* stop_event) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto start = reinterpret_cast<hipEvent_t>(start_event);
  auto stop = reinterpret_cast<hipEvent_t>(stop_event);
  if (!g_hip_direct_timing_enabled || !label || !start || !stop) {
    destroy_pending_event_pair(start, stop);
    return;
  }
  flush_pending_timing_events_if_full();
  g_hip_direct_pending_timing_samples.push_back({{label}, start, stop});
#else
  (void)label;
  (void)start_event;
  (void)stop_event;
#endif
}

void hip_direct_timing_record_pending_event_with_alias(
    const char* label,
    const char* alias,
    void* start_event,
    void* stop_event) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto start = reinterpret_cast<hipEvent_t>(start_event);
  auto stop = reinterpret_cast<hipEvent_t>(stop_event);
  if (!g_hip_direct_timing_enabled || !label || !start || !stop) {
    destroy_pending_event_pair(start, stop);
    return;
  }
  std::vector<std::string> labels{label};
  if (alias && alias[0] != '\0') {
    labels.push_back(alias);
  }
  flush_pending_timing_events_if_full();
  g_hip_direct_pending_timing_samples.push_back({std::move(labels), start, stop});
#else
  (void)label;
  (void)alias;
  (void)start_event;
  (void)stop_event;
#endif
}

std::vector<hip_direct_timing_sample> hip_direct_timing_snapshot() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  flush_pending_timing_events();
#endif
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
      hip_direct_timing_record_pending_event(label, start, stop);
      return op_status;
    }
  }

  (void)hipEventDestroy(stop);
  (void)hipEventDestroy(start);
  return op_status;
}

std::string env_value(const char* name) {
#if defined(_MSC_VER)
  char* buffer = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&buffer, &length, name) != 0 || !buffer) {
    return {};
  }
  std::string value(buffer);
  std::free(buffer);
  return value;
#else
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string{};
#endif
}

bool env_flag_disabled(const char* name) {
  const std::string value = env_value(name);
  return value == "0" || value == "false" || value == "FALSE" || value == "off" ||
         value == "OFF" || value == "no" || value == "NO";
}

bool env_flag_enabled(const char* name) {
  const std::string value = env_value(name);
  return value == "1" || value == "true" || value == "TRUE" || value == "on" ||
         value == "ON" || value == "yes" || value == "YES";
}

bool pinned_export_staging_enabled(std::size_t bytes, bool padded_destination, bool default_padded_staging) {
  if (bytes < kPinnedExportStagingMinBytes || env_flag_disabled("RNS8_HIP_PINNED_EXPORT_STAGING")) {
    return false;
  }
  return (default_padded_staging && padded_destination) || env_flag_enabled("RNS8_HIP_PINNED_EXPORT_STAGING");
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

void release_pinned_export_staging() {
  if (g_pinned_export_staging.ptr) {
    (void)hipHostFree(g_pinned_export_staging.ptr);
  }
  g_pinned_export_staging = {};
}

void* ensure_pinned_export_staging(int device_id, std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  if (g_pinned_export_staging.ptr && g_pinned_export_staging.device_id == device_id &&
      g_pinned_export_staging.capacity >= bytes) {
    return g_pinned_export_staging.ptr;
  }
  release_pinned_export_staging();
  void* ptr = nullptr;
  if (hipHostMalloc(&ptr, bytes, hipHostMallocDefault) != hipSuccess) {
    return nullptr;
  }
  g_pinned_export_staging.device_id = device_id;
  g_pinned_export_staging.ptr = ptr;
  g_pinned_export_staging.capacity = bytes;
  return ptr;
}

bool checked_output_bytes(int64_t rows, int64_t cols, std::size_t element_size) {
  if (rows <= 0 || cols <= 0 || element_size == 0) {
    return false;
  }
  const auto max_size = std::numeric_limits<std::size_t>::max();
  return static_cast<uint64_t>(rows) <=
         static_cast<uint64_t>(max_size / element_size / static_cast<std::size_t>(cols));
}

void scatter_compact_host_matrix(
    void* dst,
    int64_t dst_ld,
    const void* src,
    int64_t rows,
    int64_t cols,
    std::size_t cell_bytes) {
  const std::size_t compact_row_bytes = static_cast<std::size_t>(cols) * cell_bytes;
  if (dst_ld == cols) {
    std::memcpy(dst, src, static_cast<std::size_t>(rows) * compact_row_bytes);
    return;
  }
  auto* dst_bytes = static_cast<std::uint8_t*>(dst);
  const auto* src_bytes = static_cast<const std::uint8_t*>(src);
  const std::size_t dst_row_bytes = static_cast<std::size_t>(dst_ld) * cell_bytes;
  for (int64_t row = 0; row < rows; ++row) {
    std::memcpy(
        dst_bytes + static_cast<std::size_t>(row) * dst_row_bytes,
        src_bytes + static_cast<std::size_t>(row) * compact_row_bytes,
        compact_row_bytes);
  }
}

void record_export_host_staging_copy_sample(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& stop) {
  if (!g_hip_direct_timing_enabled) {
    return;
  }
  const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  if (nanos >= 0) {
    hip_direct_timing_record_sample("export_host_staging_copy", static_cast<double>(nanos) / 1000.0);
  }
}

hipError_t copy_compact_matrix_device_to_host_direct(
    void* dst,
    int64_t dst_ld,
    const void* src,
    int64_t rows,
    int64_t cols,
    std::size_t cell_bytes) {
  const std::size_t compact_row_bytes = static_cast<std::size_t>(cols) * cell_bytes;
  if (dst_ld == cols) {
    return hipMemcpy(
        dst,
        src,
        static_cast<std::size_t>(rows) * compact_row_bytes,
        hipMemcpyDeviceToHost);
  }
  return hipMemcpy2D(
      dst,
      static_cast<std::size_t>(dst_ld) * cell_bytes,
      src,
      compact_row_bytes,
      compact_row_bytes,
      static_cast<std::size_t>(rows),
      hipMemcpyDeviceToHost);
}

hipError_t copy_compact_matrix_device_to_host(
    int device_id,
    const char* label,
    void* dst,
    int64_t dst_ld,
    const void* src,
    int64_t rows,
    int64_t cols,
    std::size_t cell_bytes,
    bool default_padded_staging = true) {
  if (!dst || !src || dst_ld < cols || !checked_output_bytes(rows, cols, cell_bytes) ||
      !checked_output_bytes(rows, dst_ld, cell_bytes)) {
    return hipErrorInvalidValue;
  }
  const std::size_t compact_row_bytes = static_cast<std::size_t>(cols) * cell_bytes;
  const std::size_t compact_bytes = static_cast<std::size_t>(rows) * compact_row_bytes;
  if (pinned_export_staging_enabled(compact_bytes, dst_ld != cols, default_padded_staging)) {
    void* pinned = ensure_pinned_export_staging(device_id, compact_bytes);
    if (pinned) {
      hipError_t err = timed_hip_operation(label, [&]() {
        return hipMemcpy(pinned, src, compact_bytes, hipMemcpyDeviceToHost);
      });
      if (err != hipSuccess) {
        return err;
      }
      const auto start = std::chrono::steady_clock::now();
      scatter_compact_host_matrix(dst, dst_ld, pinned, rows, cols, cell_bytes);
      const auto stop = std::chrono::steady_clock::now();
      record_export_host_staging_copy_sample(start, stop);
      return hipSuccess;
    }
  }
  return timed_hip_operation(label, [&]() {
    return copy_compact_matrix_device_to_host_direct(dst, dst_ld, src, rows, cols, cell_bytes);
  });
}

bool checked_limb_export_pitch(int64_t ld, uint32_t limb_count) {
  if (ld <= 0 || limb_count == 0) {
    return false;
  }
  const auto max_size = std::numeric_limits<std::size_t>::max();
  return static_cast<uint64_t>(ld) <=
         static_cast<uint64_t>(max_size / sizeof(uint64_t) / static_cast<std::size_t>(limb_count));
}

bool exact_wide_signed_export_requires_status(uint32_t limb_count) {
  return limb_count < 3;
}

bool exact_wide_unsigned_export_requires_status(uint32_t limb_count) {
  return limb_count < 3;
}

rns8_status validate_exact_wide_grouped_matrices(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    int64_t rows,
    int64_t cols,
    bool require_current,
    int* out_device_id,
    uint32_t* out_prefix,
    std::vector<const int8_t*>* out_residue_ptrs) {
  if (!matrices || task_count == 0 || !out_device_id || !out_prefix) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (rows != 0 || cols != 0) {
    if (!checked_matrix_elements_i32(rows, cols)) {
      return RNS8_INVALID_ARGUMENT;
    }
  }
  int device_id = -1;
  uint32_t prefix = 0;
  if (out_residue_ptrs) {
    out_residue_ptrs->clear();
    out_residue_ptrs->reserve(task_count);
  }
  for (uint32_t index = 0; index < task_count; ++index) {
    const rns8_matrix* matrix = matrices[index];
    if (!matrix || matrix->backend != RNS8_BACKEND_HIP_DIRECT || matrix->desc.semantics != expected_semantics ||
        !matrix->hip_residues || matrix->hip_device_id < 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (rows != 0 && (matrix->desc.rows != rows || matrix->desc.cols != cols)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (require_current && !matrix->device_residues_current) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (index == 0) {
      device_id = matrix->hip_device_id;
      prefix = matrix->prefix;
    } else if (matrix->hip_device_id != device_id || matrix->prefix != prefix) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (out_residue_ptrs) {
      out_residue_ptrs->push_back(static_cast<const int8_t*>(matrix->hip_residues));
    }
  }
  *out_device_id = device_id;
  *out_prefix = prefix;
  return RNS8_SUCCESS;
}

bool checked_tile_entry(const rns8_plan_tile_schedule_entry& entry, int64_t rows, int64_t cols) {
  if (entry.struct_size != sizeof(rns8_plan_tile_schedule_entry) || entry.abi_version != RNS8_ABI_VERSION ||
      (entry.flags & ~kKnownTileScheduleFlags) != 0 || entry.row_offset < 0 || entry.col_offset < 0 ||
      entry.row_extent <= 0 || entry.col_extent <= 0 || entry.required_prefix == 0 || entry.selected_prefix == 0 ||
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

bool scheduled_tile_max_elements(
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count,
    int* max_elements) {
  if (!entries || entry_count == 0 || !max_elements) {
    return false;
  }
  uint64_t elements = 0;
  for (uint64_t index = 0; index < entry_count; ++index) {
    const auto& entry = entries[static_cast<std::size_t>(index)];
    const uint64_t row_extent = static_cast<uint64_t>(entry.row_extent);
    const uint64_t col_extent = static_cast<uint64_t>(entry.col_extent);
    if (col_extent != 0 && row_extent > static_cast<uint64_t>(std::numeric_limits<int>::max()) / col_extent) {
      return false;
    }
    elements = std::max(elements, row_extent * col_extent);
  }
  if (elements == 0 || elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  *max_elements = static_cast<int>(elements);
  return true;
}

bool schedule_has_zero_output_tiles(const rns8_plan_tile_schedule_entry* entries, uint64_t entry_count) {
  if (!entries) {
    return false;
  }
  for (uint64_t index = 0; index < entry_count; ++index) {
    if ((entries[static_cast<std::size_t>(index)].flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0) {
      return true;
    }
  }
  return false;
}

bool schedule_has_zero_row_col_products(const rns8_plan_tile_schedule_entry* entries, uint64_t entry_count) {
  if (!entries) {
    return false;
  }
  for (uint64_t index = 0; index < entry_count; ++index) {
    if ((entries[static_cast<std::size_t>(index)].flags & RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT) != 0) {
      return true;
    }
  }
  return false;
}

bool schedule_all_zero_output_tiles_uniform_prefix(
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count,
    uint32_t& selected_prefix) {
  if (!entries || entry_count == 0) {
    return false;
  }
  selected_prefix = 0;
  for (uint64_t index = 0; index < entry_count; ++index) {
    const auto& entry = entries[static_cast<std::size_t>(index)];
    if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) == 0) {
      return false;
    }
    if (selected_prefix == 0) {
      selected_prefix = entry.selected_prefix;
    } else if (selected_prefix != entry.selected_prefix) {
      return false;
    }
  }
  return selected_prefix != 0;
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
  void* stream = nullptr;
};

struct hip_rns_scheduled_modulus_launch {
  const int8_t* a = nullptr;
  const int8_t* b = nullptr;
  int8_t* c = nullptr;
  const rns8_plan_tile_schedule_entry* device_entries = nullptr;
  const uint8_t* zero_a_rows = nullptr;
  const uint8_t* zero_b_cols = nullptr;
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

uint32_t modulus_reciprocal_u32(uint16_t modulus) {
  return static_cast<uint32_t>((uint64_t{1} << 32u) / static_cast<uint32_t>(modulus));
}

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
  const int code = rns8_hip_direct_ring_gemm_i8_device_on_stream(
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
      modulus_reciprocal_u32(launch.modulus),
      static_cast<int>(launch.modulus_index),
      static_cast<int>(launch.selected_prefix),
      static_cast<int>(RNS8_SAFE_INT32_K_BLOCK),
      launch.stream);
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
      launch.zero_a_rows,
      launch.zero_b_cols,
      static_cast<int>(launch.entry_count),
      launch.max_tile_row_blocks,
      launch.max_tile_col_blocks,
      static_cast<int>(launch.k),
      static_cast<int>(launch.lda),
      static_cast<int>(launch.ldb),
      static_cast<int>(launch.ldc),
      static_cast<int>(launch.modulus),
      modulus_reciprocal_u32(launch.modulus),
      static_cast<int>(launch.modulus_index),
      static_cast<int>(launch.selected_prefix),
      static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
  return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
}

hipError_t launch_rns_grouped_prefix_gemm(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix,
    void* stream = nullptr) {
  if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max() ||
      RNS8_SAFE_INT32_K_BLOCK > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return hipErrorInvalidValue;
  }
  const int code = rns8_hip_direct_ring_gemm_i8_grouped_prefix_device_on_stream(
      a,
      b,
      c,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      static_cast<int>(lda),
      static_cast<int>(ldb),
      static_cast<int>(ldc),
      static_cast<int>(prefix),
      static_cast<int>(RNS8_SAFE_INT32_K_BLOCK),
      stream);
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

rns8_status hip_direct_copy_compact_matrix_device_to_host(
    int device_id,
    const char* timing_label,
    void* dst,
    int64_t dst_ld,
    const void* src,
    int64_t rows,
    int64_t cols,
    std::size_t cell_bytes,
    bool default_padded_staging) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!timing_label) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err =
      copy_compact_matrix_device_to_host(
          device_id, timing_label, dst, dst_ld, src, rows, cols, cell_bytes, default_padded_staging);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)timing_label;
  (void)dst;
  (void)dst_ld;
  (void)src;
  (void)rows;
  (void)cols;
  (void)cell_bytes;
  (void)default_padded_staging;
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

rns8_status hip_direct_native_i64_to_rns_device(
    int device_id,
    const void* device_native,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_native || !device_residues || !checked_i32_shape(rows, cols, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_pack_elements(rows, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("native_i64_to_rns_kernel", [&]() {
    const int launch_status = rns8_hip_direct_pack_i64_device(
        static_cast<const int64_t*>(device_native),
        static_cast<int8_t*>(device_residues),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(cols),
        static_cast<int>(prefix));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_native;
  (void)device_residues;
  (void)rows;
  (void)cols;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_native_u64_to_rns_device(
    int device_id,
    const void* device_native,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_native || !device_residues || !checked_i32_shape(rows, cols, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_pack_elements(rows, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("native_u64_to_rns_kernel", [&]() {
    const int launch_status = rns8_hip_direct_pack_u64_device(
        static_cast<const uint64_t*>(device_native),
        static_cast<int8_t*>(device_residues),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(cols),
        static_cast<int>(prefix));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_native;
  (void)device_residues;
  (void)rows;
  (void)cols;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_pack_finite_u8_device(
    int device_id,
    const uint8_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src || !upload_buffer || !upload_bytes || !device_residues || modulus < 2 || modulus > 256 ||
      rows <= 0 || cols <= 0 || ld < cols || rows > std::numeric_limits<int>::max() ||
      cols > std::numeric_limits<int>::max() || ld > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_output_bytes(rows, ld, sizeof(uint8_t)) || !checked_output_bytes(rows, cols, sizeof(int8_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t source_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(ld) * sizeof(uint8_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, source_bytes, upload_buffer, upload_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation(
      "finite_pack_h2d", [&]() { return hipMemcpy(*upload_buffer, src, source_bytes, hipMemcpyHostToDevice); });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("finite_pack_kernel", [&]() {
    const int code = rns8_hip_direct_pack_u8_modulus_device(
        static_cast<const uint8_t*>(*upload_buffer),
        static_cast<int8_t*>(device_residues),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(ld),
        static_cast<int>(modulus));
    return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
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
  (void)modulus;
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

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
hipError_t launch_direct_rns_gemm_no_sync(
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix,
    void* stream = nullptr) {
  const auto* a_base = static_cast<const int8_t*>(device_a_residues);
  const auto* b_base = static_cast<const int8_t*>(device_b_residues);
  auto* c_base = static_cast<int8_t*>(device_c_residues);
  if (prefix == RNS8_DEFAULT_BOUNDED_PREFIX || prefix == RNS8_MAX_SUPPORTED_PREFIX) {
    return launch_rns_grouped_prefix_gemm(a_base, b_base, c_base, m, n, k, lda, ldb, ldc, prefix, stream);
  }
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
        prefix,
        stream});
    if (launch_status != hipSuccess) {
      return launch_status;
    }
  }
  return hipSuccess;
}
#endif

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
  if (prefix == RNS8_DEFAULT_BOUNDED_PREFIX || prefix == RNS8_MAX_SUPPORTED_PREFIX) {
    const hipError_t err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
      const hipError_t launch_status =
          launch_direct_rns_gemm_no_sync(a_base, b_base, c_base, m, n, k, lda, ldb, ldc, prefix);
      return launch_status == hipSuccess ? hipDeviceSynchronize() : launch_status;
    });
    return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
  }
  const hipError_t err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
    const hipError_t launch_status =
        launch_direct_rns_gemm_no_sync(a_base, b_base, c_base, m, n, k, lda, ldb, ldc, prefix);
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
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_rns_matrix_launch_current_device_no_sync(
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    void* stream) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!plan || !A || !B || !C || plan->backend != RNS8_BACKEND_HIP_DIRECT || !plan->tile_schedule.empty()) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!A->device_residues_current || !B->device_residues_current || !A->hip_residues || !B->hip_residues ||
      !C->hip_residues) {
    return RNS8_INVALID_ARGUMENT;
  }
  const hipError_t launch_status = launch_direct_rns_gemm_no_sync(
      A->hip_residues,
      B->hip_residues,
      C->hip_residues,
      plan->desc.m,
      plan->desc.n,
      plan->desc.k,
      A->desc.cols,
      B->desc.cols,
      C->desc.cols,
      plan->prefix,
      stream);
  if (launch_status != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  C->device_residues_current = true;
  C->host_residues_current = false;
  C->host_byte_limbs_current = false;
  C->device_byte_limbs_current = false;
  C->host_native_current = false;
  C->device_native_current = false;
  if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
    C->source_version = static_cast<uint64_t>(A->source_version + B->source_version + 1u);
  }
  return RNS8_SUCCESS;
#else
  (void)plan;
  (void)A;
  (void)B;
  (void)C;
  (void)stream;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_i64_native_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_native || !device_c_residues || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("rns_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_i64_native_prefix9_device(
        static_cast<const int64_t*>(device_a_native),
        static_cast<const int64_t*>(device_b_native),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_native;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_i64_native_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_native || !device_c_residues || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("rns_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_i64_native_prefix9_colpair_device(
        static_cast<const int64_t*>(device_a_native),
        static_cast<const int64_t*>(device_b_native),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_native;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_u64_native_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_native || !device_c_residues || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("rns_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_u64_native_prefix9_device(
        static_cast<const uint64_t*>(device_a_native),
        static_cast<const uint64_t*>(device_b_native),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_native;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_u64_native_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_native || !device_c_residues || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("rns_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_u64_native_prefix9_colpair_device(
        static_cast<const uint64_t*>(device_a_native),
        static_cast<const uint64_t*>(device_b_native),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_native;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("bounded_native_a_reuse_b_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_i64_native_a_resident_b_prefix9_device(
        static_cast<const int64_t*>(device_a_native),
        static_cast<const int8_t*>(device_b_residues),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_i64_uniform_small_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code =
      rns8::detail::run_timed_device_code("bounded_uniform_small_native_a_reuse_b_gemm_kernel_group", [&]() {
        const int launch_status = rns8_hip_direct_ring_gemm_i64_uniform_small_native_a_resident_b_prefix9_device(
            static_cast<const int64_t*>(device_a_native),
            static_cast<const int8_t*>(device_b_residues),
            static_cast<int8_t*>(device_c_residues),
            static_cast<int>(m),
            static_cast<int>(n),
            static_cast<int>(k),
            static_cast<int>(lda),
            static_cast<int>(ldb),
            static_cast<int>(ldc),
            static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
        if (launch_status != static_cast<int>(hipSuccess)) {
          return launch_status;
        }
        const hipError_t sync_status = hipDeviceSynchronize();
        return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
      });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version) {
  if (!device_a_native || !B || !C || !B->device_residues_current || !B->hip_residues || !C->hip_residues ||
      B->desc.semantics != RNS8_BOUNDED_I64 || C->desc.semantics != RNS8_BOUNDED_I64 ||
      B->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS || C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS ||
      B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_i64_native_a_resident_b_prefix9_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  C->host_residues_current = false;
  C->device_residues_current = true;
  C->host_byte_limbs_current = false;
  C->device_byte_limbs_current = false;
  C->host_native_current = false;
  C->device_native_current = false;
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_i64_uniform_small_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version) {
  if (!device_a_native || !B || !C || !B->device_residues_current || !B->hip_residues || !C->hip_residues ||
      B->desc.semantics != RNS8_BOUNDED_I64 || C->desc.semantics != RNS8_BOUNDED_I64 ||
      B->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS || C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS ||
      B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_i64_uniform_small_native_a_resident_b_prefix9_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  C->host_residues_current = false;
  C->device_residues_current = true;
  C->host_byte_limbs_current = false;
  C->device_byte_limbs_current = false;
  C->host_native_current = false;
  C->device_native_current = false;
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("bounded_native_a_reuse_b_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_u64_native_a_resident_b_prefix9_device(
        static_cast<const uint64_t*>(device_a_native),
        static_cast<const int8_t*>(device_b_residues),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("bounded_native_a_colpair_reuse_b_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_u64_native_a_resident_b_prefix9_colpair_device(
        static_cast<const uint64_t*>(device_a_native),
        static_cast<const int8_t*>(device_b_residues),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_native || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("bounded_native_b_colpair_reuse_a_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_u64_resident_a_native_b_prefix9_colpair_device(
        static_cast<const int8_t*>(device_a_residues),
        static_cast<const uint64_t*>(device_b_native),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_native;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_u64_uniform_small_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code =
      rns8::detail::run_timed_device_code("bounded_uniform_small_native_a_reuse_b_gemm_kernel_group", [&]() {
        const int launch_status = rns8_hip_direct_ring_gemm_u64_uniform_small_native_a_resident_b_prefix9_device(
            static_cast<const uint64_t*>(device_a_native),
            static_cast<const int8_t*>(device_b_residues),
            static_cast<int8_t*>(device_c_residues),
            static_cast<int>(m),
            static_cast<int>(n),
            static_cast<int>(k),
            static_cast<int>(lda),
            static_cast<int>(ldb),
            static_cast<int>(ldc),
            static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
        if (launch_status != static_cast<int>(hipSuccess)) {
          return launch_status;
        }
        const hipError_t sync_status = hipDeviceSynchronize();
        return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
      });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version) {
  if (!device_a_native || !B || !C || !B->device_residues_current || !B->hip_residues || !C->hip_residues ||
      B->desc.semantics != RNS8_BOUNDED_U64 || C->desc.semantics != RNS8_BOUNDED_U64 ||
      B->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_u64_native_a_resident_b_prefix9_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  C->host_residues_current = false;
  C->device_residues_current = true;
  C->host_byte_limbs_current = false;
  C->device_byte_limbs_current = false;
  C->host_native_current = false;
  C->device_native_current = false;
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version) {
  if (!device_a_native || !B || !C || !B->device_residues_current || !B->hip_residues || !C->hip_residues ||
      B->desc.semantics != RNS8_BOUNDED_U64 || C->desc.semantics != RNS8_BOUNDED_U64 ||
      B->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  C->host_residues_current = false;
  C->device_residues_current = true;
  C->host_byte_limbs_current = false;
  C->device_byte_limbs_current = false;
  C->host_native_current = false;
  C->device_native_current = false;
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_matrix(
    int device_id,
    const rns8_matrix* A,
    const void* device_b_native,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    uint64_t source_version) {
  if (!A || !device_b_native || !C || !A->device_residues_current || !A->hip_residues || !C->hip_residues ||
      A->desc.semantics != RNS8_BOUNDED_U64 || C->desc.semantics != RNS8_BOUNDED_U64 ||
      A->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      A->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      A->desc.rows != m || A->desc.cols != k || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_device(
      device_id,
      A->hip_residues,
      device_b_native,
      C->hip_residues,
      m,
      n,
      k,
      A->desc.logical_ld,
      ldb,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  C->host_residues_current = false;
  C->device_residues_current = true;
  C->host_byte_limbs_current = false;
  C->device_byte_limbs_current = false;
  C->host_native_current = false;
  C->device_native_current = false;
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_u64_uniform_small_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version) {
  if (!device_a_native || !B || !C || !B->device_residues_current || !B->hip_residues || !C->hip_residues ||
      B->desc.semantics != RNS8_BOUNDED_U64 || C->desc.semantics != RNS8_BOUNDED_U64 ||
      B->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_u64_uniform_small_native_a_resident_b_prefix9_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  C->host_residues_current = false;
  C->device_residues_current = true;
  C->host_byte_limbs_current = false;
  C->device_byte_limbs_current = false;
  C->host_native_current = false;
  C->device_native_current = false;
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_uniform_small_i8_ab_resident_b_prefix9_device(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_i8 || !device_b_i8 || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code =
      rns8::detail::run_timed_device_code("bounded_uniform_small_i8_ab_reuse_b_gemm_kernel_group", [&]() {
        const int launch_status = rns8_hip_direct_ring_gemm_uniform_small_i8_ab_resident_b_prefix9_device(
            static_cast<const int8_t*>(device_a_i8),
            static_cast<const int8_t*>(device_b_i8),
            static_cast<int8_t*>(device_c_residues),
            static_cast<int>(m),
            static_cast<int>(n),
            static_cast<int>(k),
            static_cast<int>(lda),
            static_cast<int>(ldb),
            static_cast<int>(ldc),
            static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
        if (launch_status != static_cast<int>(hipSuccess)) {
          return launch_status;
        }
        const hipError_t sync_status = hipDeviceSynchronize();
        return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
      });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_i8;
  (void)device_b_i8;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_uniform_small_i8_ab_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    uint64_t source_version) {
  if (!device_a_i8 || !device_b_i8 || !C || !C->hip_residues ||
      (C->desc.semantics != RNS8_BOUNDED_I64 && C->desc.semantics != RNS8_BOUNDED_U64) ||
      (C->desc.semantics == RNS8_BOUNDED_I64 && C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS) ||
      (C->desc.semantics == RNS8_BOUNDED_U64 && C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED) ||
      C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_uniform_small_i8_ab_resident_b_prefix9_device(
      device_id,
      device_a_i8,
      device_b_i8,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      ldb,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  C->host_residues_current = false;
  C->device_residues_current = true;
  C->host_byte_limbs_current = false;
  C->device_byte_limbs_current = false;
  C->host_native_current = false;
  C->device_native_current = false;
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

namespace {

rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_prefix9_device_with_label(
    const char* timing_label,
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_i8 || !device_b_i8 || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code(timing_label, [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_uniform_small_i8_ab_resident_b_prefix9_colpair_device(
        static_cast<const int8_t*>(device_a_i8),
        static_cast<const int8_t*>(device_b_i8),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)timing_label;
  (void)device_id;
  (void)device_a_i8;
  (void)device_b_i8;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

}  // namespace

rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_device(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  return hip_direct_gemm_uniform_small_i8_ab_colpair_prefix9_device_with_label(
      "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group",
      device_id,
      device_a_i8,
      device_b_i8,
      device_c_residues,
      m,
      n,
      k,
      lda,
      ldb,
      ldc);
}

rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    uint64_t source_version) {
  if (!device_a_i8 || !device_b_i8 || !C || !C->hip_residues ||
      (C->desc.semantics != RNS8_BOUNDED_I64 && C->desc.semantics != RNS8_BOUNDED_U64) ||
      (C->desc.semantics == RNS8_BOUNDED_I64 && C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS) ||
      (C->desc.semantics == RNS8_BOUNDED_U64 && C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED) ||
      C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_device(
      device_id,
      device_a_i8,
      device_b_i8,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      ldb,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  C->host_residues_current = false;
  C->device_residues_current = true;
  C->host_byte_limbs_current = false;
  C->device_byte_limbs_current = false;
  C->host_native_current = false;
  C->device_native_current = false;
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_device(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  return hip_direct_gemm_uniform_small_i8_ab_colpair_prefix9_device_with_label(
      "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group",
      device_id,
      device_a_i8,
      device_b_i8,
      device_c_residues,
      m,
      n,
      k,
      lda,
      ldb,
      ldc);
}

rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    uint64_t source_version) {
  if (!device_a_i8 || !device_b_i8 || !C || !C->hip_residues ||
      (C->desc.semantics != RNS8_BOUNDED_I64 && C->desc.semantics != RNS8_BOUNDED_U64) ||
      (C->desc.semantics == RNS8_BOUNDED_I64 && C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS) ||
      (C->desc.semantics == RNS8_BOUNDED_U64 && C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED) ||
      C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_device(
      device_id,
      device_a_i8,
      device_b_i8,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      ldb,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  C->host_residues_current = false;
  C->device_residues_current = true;
  C->host_byte_limbs_current = false;
  C->device_byte_limbs_current = false;
  C->host_native_current = false;
  C->device_native_current = false;
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_device(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  return hip_direct_gemm_uniform_small_i8_ab_colpair_prefix9_device_with_label(
      "bounded_uniform_small_i8_ab_transient_gemm_kernel_group",
      device_id,
      device_a_i8,
      device_b_i8,
      device_c_residues,
      m,
      n,
      k,
      lda,
      ldb,
      ldc);
}

rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    uint64_t source_version) {
  if (!device_a_i8 || !device_b_i8 || !C || !C->hip_residues ||
      (C->desc.semantics != RNS8_BOUNDED_I64 && C->desc.semantics != RNS8_BOUNDED_U64) ||
      (C->desc.semantics == RNS8_BOUNDED_I64 && C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS) ||
      (C->desc.semantics == RNS8_BOUNDED_U64 && C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED) ||
      C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_device(
      device_id,
      device_a_i8,
      device_b_i8,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      ldb,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  C->host_residues_current = false;
  C->device_residues_current = true;
  C->host_byte_limbs_current = false;
  C->device_byte_limbs_current = false;
  C->host_native_current = false;
  C->device_native_current = false;
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_finite_u8_resident_device(
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
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_residues || !device_c_residues || modulus < 2 || modulus > 256 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("finite_resident_gemm_kernel", [&]() {
    const int code = rns8_hip_direct_finite_ring_gemm_i8_device(
        static_cast<const int8_t*>(device_a_residues),
        static_cast<const int8_t*>(device_b_residues),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
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
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_finite_u8_native_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_native || !device_c_residues || modulus < 2 || modulus > 256 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("finite_native_gemm_kernel", [&]() {
    const int code = rns8_hip_direct_finite_ring_gemm_u8_native_device(
        static_cast<const uint8_t*>(device_a_native),
        static_cast<const uint8_t*>(device_b_native),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_native;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)modulus;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_finite_u8_native_a_resident_b_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || modulus < 2 || modulus > 256 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("finite_native_a_gemm_kernel", [&]() {
    const int code = rns8_hip_direct_finite_ring_gemm_u8_native_a_i8_b_device(
        static_cast<const uint8_t*>(device_a_native),
        static_cast<const int8_t*>(device_b_residues),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)modulus;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_finite_u8_native_a_resident_b_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint16_t modulus,
    uint64_t source_version) {
  if (!B || !C || B->finite_modulus != modulus || !B->device_residues_current ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_finite_u8_native_a_resident_b_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld,
      modulus);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  C->host_residues_current = false;
  C->device_residues_current = true;
  C->host_byte_limbs_current = false;
  C->device_byte_limbs_current = false;
  C->host_native_current = false;
  C->device_native_current = false;
  C->finite_modulus = modulus;
  C->source_version = source_version;
  return RNS8_SUCCESS;
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
          if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0) {
            auto* tile_c =
                c_base + c_offset + static_cast<std::size_t>(entry.row_offset) * static_cast<std::size_t>(ldc) +
                static_cast<std::size_t>(entry.col_offset);
            const hipError_t zero_status = hipMemset2D(
                tile_c,
                static_cast<std::size_t>(ldc) * sizeof(int8_t),
                0,
                static_cast<std::size_t>(entry.col_extent) * sizeof(int8_t),
                static_cast<std::size_t>(entry.row_extent));
            if (zero_status != hipSuccess) {
              return zero_status;
            }
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
    const void* active_device_entries,
    const void* zero_a_rows,
    const void* zero_b_cols,
    const uint64_t* active_offsets,
    const uint64_t* active_counts,
    uint32_t active_prefix_count,
    uint64_t entry_count) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_residues || !device_c_residues || !host_entries ||
      !active_offsets || !active_counts || active_prefix_count == 0 || entry_count == 0 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n) {
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
  int max_tile_elements = 0;
  if (!scheduled_tile_max_elements(host_entries, entry_count, &max_tile_elements)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_base = static_cast<const int8_t*>(device_a_residues);
  const auto* b_base = static_cast<const int8_t*>(device_b_residues);
  auto* c_base = static_cast<int8_t*>(device_c_residues);
  const auto* active_schedule_base = static_cast<const rns8_plan_tile_schedule_entry*>(active_device_entries);
  const auto* zero_a_rows_base = static_cast<const uint8_t*>(zero_a_rows);
  const auto* zero_b_cols_base = static_cast<const uint8_t*>(zero_b_cols);
  const uint32_t max_selected_prefix = schedule.selected_prefix_groups.back();
  if (active_prefix_count != max_selected_prefix) {
    return RNS8_INVALID_ARGUMENT;
  }
  uint64_t running_offset = 0;
  for (uint32_t p = 0; p < max_selected_prefix; ++p) {
    uint64_t expected_count = 0;
    for (uint64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
      const auto& entry = host_entries[static_cast<std::size_t>(entry_index)];
      if (entry.selected_prefix > p && (entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) == 0) {
        ++expected_count;
      }
    }
    if (active_offsets[p] != running_offset || active_counts[p] != expected_count) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (expected_count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return RNS8_INVALID_ARGUMENT;
    }
    running_offset += expected_count;
  }
  if (running_offset != 0 && !active_device_entries) {
    return RNS8_INVALID_ARGUMENT;
  }
  const bool zero_output_tiles = schedule_has_zero_output_tiles(host_entries, entry_count);
  const bool zero_row_col_products = schedule_has_zero_row_col_products(host_entries, entry_count);
  if (zero_row_col_products && (!zero_a_rows || !zero_b_cols)) {
    return RNS8_INVALID_ARGUMENT;
  }
  uint32_t uniform_zero_selected_prefix = 0;
  const bool uniform_all_zero_output_tiles =
      schedule_all_zero_output_tiles_uniform_prefix(host_entries, entry_count, uniform_zero_selected_prefix);
  if (zero_output_tiles && !uniform_all_zero_output_tiles && !device_entries) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (zero_output_tiles) {
    const bool synchronize_zero_fill = running_offset == 0;
    const hipError_t zero_err = timed_hip_operation("direct_hip_zero_output_tile_memset", [&]() {
      if (uniform_all_zero_output_tiles) {
        const auto rows_u = static_cast<uint64_t>(m);
        const auto ldc_u = static_cast<uint64_t>(ldc);
        const uint64_t max_bytes = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max());
        if (ldc_u != 0 && rows_u > max_bytes / ldc_u) {
          return hipErrorInvalidValue;
        }
        const uint64_t plane_stride = rows_u * ldc_u;
        if (plane_stride != 0 && uniform_zero_selected_prefix > max_bytes / plane_stride) {
          return hipErrorInvalidValue;
        }
        const auto zero_bytes = static_cast<std::size_t>(uniform_zero_selected_prefix * plane_stride);
        const hipError_t memset_status = hipMemsetAsync(c_base, 0, zero_bytes, nullptr);
        return memset_status == hipSuccess && synchronize_zero_fill ? hipDeviceSynchronize() : memset_status;
      }
      const int code = rns8_hip_direct_zero_scheduled_residue_tiles_device(
          c_base,
          static_cast<const rns8_plan_tile_schedule_entry*>(device_entries),
          static_cast<int>(entry_count),
          max_tile_elements,
          static_cast<int>(m),
          static_cast<int>(ldc));
      if (code != static_cast<int>(hipSuccess)) {
        return static_cast<hipError_t>(code);
      }
      return synchronize_zero_fill ? hipDeviceSynchronize() : hipSuccess;
    });
    if (zero_err != hipSuccess) {
      return RNS8_BACKEND_FAILURE;
    }
  }
  if (running_offset == 0) {
    return RNS8_SUCCESS;
  }
  const hipError_t err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
    for (uint32_t p = 0; p < max_selected_prefix; ++p) {
      const uint64_t active_count = active_counts[p];
      if (active_count == 0) {
        continue;
      }
      if (active_offsets[p] > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return hipErrorInvalidValue;
      }
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
          active_schedule_base + static_cast<std::size_t>(active_offsets[p]),
          zero_a_rows_base,
          zero_b_cols_base,
          active_count,
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
  (void)active_device_entries;
  (void)zero_a_rows;
  (void)zero_b_cols;
  (void)active_offsets;
  (void)active_counts;
  (void)active_prefix_count;
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
  err = copy_compact_matrix_device_to_host(
      device_id, "crt_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(int64_t));
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
    const void* zero_a_rows,
    const void* zero_b_cols,
    uint64_t entry_count,
    uint64_t max_tile_elements,
    bool all_zero_output_tiles,
    int64_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !dst || ld < cols ||
      !checked_matrix_elements_i32(rows, cols) || !checked_output_bytes(rows, cols, sizeof(int64_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  constexpr uint64_t export_threads = 256;
  if (!all_zero_output_tiles) {
    if (!status_buffer || !status_bytes || !device_entries || !device_bounds || entry_count == 0 ||
        entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) || max_tile_elements == 0 ||
        max_tile_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t blocks_per_tile = (max_tile_elements + export_threads - 1u) / export_threads;
    if (blocks_per_tile == 0 ||
        entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) / blocks_per_tile) {
      return RNS8_INVALID_ARGUMENT;
    }
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
  if (all_zero_output_tiles) {
    const hipError_t zero_err = timed_hip_operation("crt_export_kernel", [&]() {
      const hipError_t memset_status = hipMemsetAsync(*export_buffer, 0, output_bytes, nullptr);
      return memset_status == hipSuccess ? hipDeviceSynchronize() : memset_status;
    });
    if (zero_err != hipSuccess) {
      return RNS8_BACKEND_FAILURE;
    }
    const hipError_t copy_err = copy_compact_matrix_device_to_host(
        device_id, "crt_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(int64_t));
    return copy_err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
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
        static_cast<const uint8_t*>(zero_a_rows),
        static_cast<const uint8_t*>(zero_b_cols),
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
  err = copy_compact_matrix_device_to_host(
      device_id, "crt_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(int64_t));
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
  (void)zero_a_rows;
  (void)zero_b_cols;
  (void)entry_count;
  (void)max_tile_elements;
  (void)all_zero_output_tiles;
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
  err = copy_compact_matrix_device_to_host(
      device_id, "crt_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(uint64_t));
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
    const void* zero_a_rows,
    const void* zero_b_cols,
    uint64_t entry_count,
    uint64_t max_tile_elements,
    bool all_zero_output_tiles,
    uint64_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !dst || ld < cols ||
      !checked_matrix_elements_i32(rows, cols) || !checked_output_bytes(rows, cols, sizeof(uint64_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  constexpr uint64_t export_threads = 256;
  if (!all_zero_output_tiles) {
    if (!status_buffer || !status_bytes || !device_entries || !device_bounds || entry_count == 0 ||
        entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) || max_tile_elements == 0 ||
        max_tile_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t blocks_per_tile = (max_tile_elements + export_threads - 1u) / export_threads;
    if (blocks_per_tile == 0 ||
        entry_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) / blocks_per_tile) {
      return RNS8_INVALID_ARGUMENT;
    }
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
  if (all_zero_output_tiles) {
    const hipError_t zero_err = timed_hip_operation("crt_export_kernel", [&]() {
      const hipError_t memset_status = hipMemsetAsync(*export_buffer, 0, output_bytes, nullptr);
      return memset_status == hipSuccess ? hipDeviceSynchronize() : memset_status;
    });
    if (zero_err != hipSuccess) {
      return RNS8_BACKEND_FAILURE;
    }
    const hipError_t copy_err = copy_compact_matrix_device_to_host(
        device_id, "crt_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(uint64_t));
    return copy_err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
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
        static_cast<const uint8_t*>(zero_a_rows),
        static_cast<const uint8_t*>(zero_b_cols),
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
  err = copy_compact_matrix_device_to_host(
      device_id, "crt_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(uint64_t));
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
  (void)zero_a_rows;
  (void)zero_b_cols;
  (void)entry_count;
  (void)max_tile_elements;
  (void)all_zero_output_tiles;
  (void)dst;
  (void)ld;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_finite_u8_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    int64_t rows,
    int64_t cols,
    uint16_t modulus,
    uint8_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !dst || modulus < 2 || modulus > 256 ||
      ld < cols || !checked_matrix_elements_i32(rows, cols) ||
      !checked_output_bytes(rows, cols, sizeof(uint8_t)) || !checked_output_bytes(rows, ld, sizeof(uint8_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(uint8_t);
  rns8_status status = hip_direct_ensure_upload_buffer(device_id, output_bytes, export_buffer, export_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation("finite_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_u8_modulus_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<uint8_t*>(*export_buffer),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(cols),
        static_cast<int>(modulus));
    return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = copy_compact_matrix_device_to_host(
      device_id, "finite_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(uint8_t));
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_residues;
  (void)export_buffer;
  (void)export_bytes;
  (void)rows;
  (void)cols;
  (void)modulus;
  (void)dst;
  (void)ld;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_exact_wide_signed_limbs_to_device(
    int device_id,
    const void* device_residues,
    void* device_dst,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint32_t limb_count) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !device_dst || !status_buffer || !status_bytes ||
      !checked_matrix_elements_i32(rows, cols) || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      limb_count == 0 || limb_count > 32 ||
      !checked_output_bytes(rows, cols, static_cast<std::size_t>(limb_count) * sizeof(uint64_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const bool requires_status = exact_wide_signed_export_requires_status(limb_count);
  if (requires_status) {
    rns8_status status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    const hipError_t memset_err = timed_hip_operation("exact_wide_export_status_memset", [&]() {
      return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr);
    });
    if (memset_err != hipSuccess) {
      return RNS8_BACKEND_FAILURE;
    }
  }
  hipError_t err = timed_hip_operation("exact_wide_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_exact_wide_signed_limbs_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<uint64_t*>(device_dst),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        static_cast<int>(limb_count),
        requires_status ? static_cast<int*>(*status_buffer) : nullptr);
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipSuccess;
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (requires_status) {
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
  }
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)device_residues;
  (void)device_dst;
  (void)status_buffer;
  (void)status_bytes;
  (void)rows;
  (void)cols;
  (void)prefix;
  (void)limb_count;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_exact_wide_signed_matrix_limbs_to_device(
    rns8_matrix* matrix,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint32_t limb_count) {
  if (!matrix || matrix->backend != RNS8_BACKEND_HIP_DIRECT || !matrix->device_residues_current ||
      !matrix->hip_residues || matrix->hip_device_id < 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  return hip_direct_export_exact_wide_signed_limbs_to_device(
      matrix->hip_device_id,
      matrix->hip_residues,
      device_dst,
      &matrix->hip_status_buffer,
      &matrix->hip_status_bytes,
      rows,
      cols,
      matrix->prefix,
      limb_count);
}

rns8_status hip_direct_prepare_exact_wide_grouped_matrix_residue_pointers(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    void* device_residue_ptrs,
    std::size_t device_residue_ptr_bytes,
    int* out_device_id,
    uint32_t* out_prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residue_ptrs ||
      device_residue_ptr_bytes < static_cast<std::size_t>(task_count) * sizeof(const int8_t*)) {
    return RNS8_INVALID_ARGUMENT;
  }
  std::vector<const int8_t*> host_ptrs;
  int device_id = -1;
  uint32_t prefix = 0;
  rns8_status status = validate_exact_wide_grouped_matrices(
      matrices,
      task_count,
      expected_semantics,
      0,
      0,
      false,
      &device_id,
      &prefix,
      &host_ptrs);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const std::size_t table_bytes = host_ptrs.size() * sizeof(host_ptrs[0]);
  const hipError_t err = timed_hip_operation("exact_wide_grouped_export_pointer_h2d", [&]() {
    return hipMemcpy(device_residue_ptrs, host_ptrs.data(), table_bytes, hipMemcpyHostToDevice);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (out_device_id) {
    *out_device_id = device_id;
  }
  if (out_prefix) {
    *out_prefix = prefix;
  }
  return RNS8_SUCCESS;
#else
  (void)matrices;
  (void)task_count;
  (void)expected_semantics;
  (void)device_residue_ptrs;
  (void)device_residue_ptr_bytes;
  (void)out_device_id;
  (void)out_prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_exact_wide_signed_grouped_matrix_limbs_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint32_t limb_count) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residue_ptrs || !device_dst || task_count == 0 || limb_count == 0 || limb_count > 32 ||
      exact_wide_signed_export_requires_status(limb_count) ||
      !checked_output_bytes(
          rows,
          cols,
          static_cast<std::size_t>(task_count) * static_cast<std::size_t>(limb_count) * sizeof(uint64_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  uint32_t prefix = 0;
  rns8_status status = validate_exact_wide_grouped_matrices(
      matrices,
      task_count,
      RNS8_EXACT_WIDE_SIGNED,
      rows,
      cols,
      true,
      &device_id,
      &prefix,
      nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("exact_wide_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_exact_wide_signed_grouped_limbs_device(
        static_cast<const int8_t* const*>(device_residue_ptrs),
        static_cast<uint64_t*>(device_dst),
        static_cast<int>(task_count),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        static_cast<int>(limb_count));
    return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)matrices;
  (void)task_count;
  (void)device_residue_ptrs;
  (void)device_dst;
  (void)rows;
  (void)cols;
  (void)limb_count;
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
  const bool requires_status = exact_wide_signed_export_requires_status(limb_count);
  if (requires_status) {
    status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    const hipError_t memset_err = timed_hip_operation("exact_wide_export_status_memset", [&]() {
      return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr);
    });
    if (memset_err != hipSuccess) {
      return RNS8_BACKEND_FAILURE;
    }
  }
  hipError_t err = timed_hip_operation("exact_wide_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_exact_wide_signed_limbs_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<uint64_t*>(*export_buffer),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        static_cast<int>(limb_count),
        requires_status ? static_cast<int*>(*status_buffer) : nullptr);
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (requires_status) {
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
  }
  err = copy_compact_matrix_device_to_host(
      device_id,
      "exact_wide_export_d2h",
      dst,
      ld,
      *export_buffer,
      rows,
      cols,
      static_cast<std::size_t>(limb_count) * sizeof(uint64_t),
      false);
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

rns8_status hip_direct_export_exact_wide_unsigned_limbs_to_device(
    int device_id,
    const void* device_residues,
    void* device_dst,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint32_t limb_count) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !device_dst || !status_buffer || !status_bytes ||
      !checked_matrix_elements_i32(rows, cols) || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      limb_count == 0 || limb_count > 32 ||
      !checked_output_bytes(rows, cols, static_cast<std::size_t>(limb_count) * sizeof(uint64_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const bool requires_status = exact_wide_unsigned_export_requires_status(limb_count);
  if (requires_status) {
    rns8_status status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    const hipError_t memset_err = timed_hip_operation("exact_wide_export_status_memset", [&]() {
      return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr);
    });
    if (memset_err != hipSuccess) {
      return RNS8_BACKEND_FAILURE;
    }
  }
  hipError_t err = timed_hip_operation("exact_wide_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_exact_wide_unsigned_limbs_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<uint64_t*>(device_dst),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        static_cast<int>(limb_count),
        requires_status ? static_cast<int*>(*status_buffer) : nullptr);
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipSuccess;
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (requires_status) {
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
  }
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)device_residues;
  (void)device_dst;
  (void)status_buffer;
  (void)status_bytes;
  (void)rows;
  (void)cols;
  (void)prefix;
  (void)limb_count;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_export_exact_wide_unsigned_matrix_limbs_to_device(
    rns8_matrix* matrix,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint32_t limb_count) {
  if (!matrix || matrix->backend != RNS8_BACKEND_HIP_DIRECT || !matrix->device_residues_current ||
      !matrix->hip_residues || matrix->hip_device_id < 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  return hip_direct_export_exact_wide_unsigned_limbs_to_device(
      matrix->hip_device_id,
      matrix->hip_residues,
      device_dst,
      &matrix->hip_status_buffer,
      &matrix->hip_status_bytes,
      rows,
      cols,
      matrix->prefix,
      limb_count);
}

rns8_status hip_direct_export_exact_wide_unsigned_grouped_matrix_limbs_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint32_t limb_count) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residue_ptrs || !device_dst || task_count == 0 || limb_count == 0 || limb_count > 32 ||
      exact_wide_unsigned_export_requires_status(limb_count) ||
      !checked_output_bytes(
          rows,
          cols,
          static_cast<std::size_t>(task_count) * static_cast<std::size_t>(limb_count) * sizeof(uint64_t))) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  uint32_t prefix = 0;
  rns8_status status = validate_exact_wide_grouped_matrices(
      matrices,
      task_count,
      RNS8_EXACT_WIDE_UNSIGNED,
      rows,
      cols,
      true,
      &device_id,
      &prefix,
      nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("exact_wide_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_exact_wide_unsigned_grouped_limbs_device(
        static_cast<const int8_t* const*>(device_residue_ptrs),
        static_cast<uint64_t*>(device_dst),
        static_cast<int>(task_count),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        static_cast<int>(limb_count));
    return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)matrices;
  (void)task_count;
  (void)device_residue_ptrs;
  (void)device_dst;
  (void)rows;
  (void)cols;
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
  const bool requires_status = exact_wide_unsigned_export_requires_status(limb_count);
  if (requires_status) {
    status = hip_direct_ensure_upload_buffer(device_id, sizeof(int), status_buffer, status_bytes);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    const hipError_t memset_err = timed_hip_operation("exact_wide_export_status_memset", [&]() {
      return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr);
    });
    if (memset_err != hipSuccess) {
      return RNS8_BACKEND_FAILURE;
    }
  }
  hipError_t err = timed_hip_operation("exact_wide_export_kernel", [&]() {
    const int code = rns8_hip_direct_export_exact_wide_unsigned_limbs_device(
        static_cast<const int8_t*>(device_residues),
        static_cast<uint64_t*>(*export_buffer),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(prefix),
        static_cast<int>(limb_count),
        requires_status ? static_cast<int*>(*status_buffer) : nullptr);
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  if (requires_status) {
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
  }
  err = copy_compact_matrix_device_to_host(
      device_id,
      "exact_wide_export_d2h",
      dst,
      ld,
      *export_buffer,
      rows,
      cols,
      static_cast<std::size_t>(limb_count) * sizeof(uint64_t));
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
