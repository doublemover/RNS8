#include "backend_hip_direct/hip_backend.hpp"

#include "core/internal.hpp"

#include <limits>

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#  include <hip/hip_runtime_api.h>

extern "C" int rns8_hip_direct_ring_gemm_i8(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus);

extern "C" int rns8_hip_direct_pack_i64(
    const int64_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld,
    int prefix);

extern "C" int rns8_hip_direct_pack_u64(
    const uint64_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld,
    int prefix);

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
    int modulus);

extern "C" int rns8_hip_direct_export_i64_device(
    const int8_t* d_residues,
    int64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status);

extern "C" int rns8_hip_direct_export_i64_tile_device(
    const int8_t* d_residues,
    int64_t* d_dst,
    int rows,
    int cols,
    int row_offset,
    int col_offset,
    int row_extent,
    int col_extent,
    int prefix,
    uint64_t bound,
    int* d_status);

extern "C" int rns8_hip_direct_export_u64_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status);

extern "C" int rns8_hip_direct_export_u64_tile_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int row_offset,
    int col_offset,
    int row_extent,
    int col_extent,
    int prefix,
    uint64_t bound,
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
  if (entry.row_offset < 0 || entry.col_offset < 0 || entry.row_extent <= 0 || entry.col_extent <= 0 ||
      entry.selected_prefix == 0 || entry.selected_prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return false;
  }
  if (entry.row_offset > rows || entry.col_offset > cols) {
    return false;
  }
  return entry.row_extent <= rows - entry.row_offset && entry.col_extent <= cols - entry.col_offset &&
         entry.row_offset <= std::numeric_limits<int>::max() && entry.col_offset <= std::numeric_limits<int>::max() &&
         entry.row_extent <= std::numeric_limits<int>::max() && entry.col_extent <= std::numeric_limits<int>::max();
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
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
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

rns8_status hip_direct_pack_i64(
    int device_id,
    const int64_t* src,
    int8_t* residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src || !residues || !checked_i32_shape(rows, cols, ld, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_pack_elements(rows, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8_hip_direct_pack_i64(
      src,
      residues,
      static_cast<int>(rows),
      static_cast<int>(cols),
      static_cast<int>(ld),
      static_cast<int>(prefix));
  return code == static_cast<int>(hipSuccess) ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)src;
  (void)residues;
  (void)rows;
  (void)cols;
  (void)ld;
  (void)prefix;
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

rns8_status hip_direct_pack_u64(
    int device_id,
    const uint64_t* src,
    int8_t* residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src || !residues || !checked_i32_shape(rows, cols, ld, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_pack_elements(rows, cols, prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8_hip_direct_pack_u64(
      src,
      residues,
      static_cast<int>(rows),
      static_cast<int>(cols),
      static_cast<int>(ld),
      static_cast<int>(prefix));
  return code == static_cast<int>(hipSuccess) ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)src;
  (void)residues;
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

rns8_status hip_direct_ring_gemm_i8(
    int device_id,
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!A || !B || !C || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || modulus < 2 ||
      modulus > 256) {
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
  const int code = rns8_hip_direct_ring_gemm_i8(
      A,
      B,
      C,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      static_cast<int>(lda),
      static_cast<int>(ldb),
      static_cast<int>(ldc),
      static_cast<int>(modulus));
  return code == static_cast<int>(hipSuccess) ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)A;
  (void)B;
  (void)C;
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
      ldb < n || ldc < n || prefix == 0 || prefix > RNS8_DEFAULT_MODULUS_COUNT) {
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
      const int code = rns8_hip_direct_ring_gemm_i8_device(
          a_base + a_offset,
          b_base + b_offset,
          c_base + c_offset,
          static_cast<int>(m),
          static_cast<int>(n),
          static_cast<int>(k),
          static_cast<int>(lda),
          static_cast<int>(ldb),
          static_cast<int>(ldc),
          static_cast<int>(kDefaultModuli[p]));
      if (code != static_cast<int>(hipSuccess)) {
        return static_cast<hipError_t>(code);
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
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_base = static_cast<const int8_t*>(device_a_residues);
  const auto* b_base = static_cast<const int8_t*>(device_b_residues);
  auto* c_base = static_cast<int8_t*>(device_c_residues);
  const hipError_t err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
    for (uint64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
      const auto& entry = entries[static_cast<std::size_t>(entry_index)];
      if (!checked_tile_entry(entry, m, n)) {
        return hipErrorInvalidValue;
      }
      for (uint32_t p = 0; p < entry.selected_prefix; ++p) {
        const std::size_t a_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(m) *
                                     static_cast<std::size_t>(lda);
        const std::size_t b_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(k) *
                                     static_cast<std::size_t>(ldb);
        const std::size_t c_offset = static_cast<std::size_t>(p) * static_cast<std::size_t>(m) *
                                     static_cast<std::size_t>(ldc);
        const int code = rns8_hip_direct_ring_gemm_i8_device(
            a_base + a_offset + static_cast<std::size_t>(entry.row_offset) * static_cast<std::size_t>(lda),
            b_base + b_offset + static_cast<std::size_t>(entry.col_offset),
            c_base + c_offset + static_cast<std::size_t>(entry.row_offset) * static_cast<std::size_t>(ldc) +
                static_cast<std::size_t>(entry.col_offset),
            static_cast<int>(entry.row_extent),
            static_cast<int>(entry.col_extent),
            static_cast<int>(k),
            static_cast<int>(lda),
            static_cast<int>(ldb),
            static_cast<int>(ldc),
            static_cast<int>(kDefaultModuli[p]));
        if (code != static_cast<int>(hipSuccess)) {
          return static_cast<hipError_t>(code);
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
    return prefix > RNS8_MAX_SUPPORTED_PREFIX ? RNS8_UNSUPPORTED_BACKEND : RNS8_INVALID_ARGUMENT;
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
  hipError_t err = timed_hip_operation(
      "crt_export_status_memset", [&]() { return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr); });
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
    const rns8_plan_tile_schedule_entry* entries,
    const uint64_t* bounds,
    uint64_t entry_count,
    int64_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !status_buffer || !status_bytes || !entries || !bounds ||
      !dst || ld < cols || !checked_matrix_elements_i32(rows, cols) ||
      !checked_output_bytes(rows, cols, sizeof(int64_t)) || entry_count == 0 ||
      entry_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
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
  hipError_t err = timed_hip_operation(
      "crt_export_status_memset", [&]() { return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr); });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("crt_export_kernel", [&]() {
    for (uint64_t index = 0; index < entry_count; ++index) {
      const auto& entry = entries[static_cast<std::size_t>(index)];
      if (!checked_tile_entry(entry, rows, cols)) {
        return hipErrorInvalidValue;
      }
      const int code = rns8_hip_direct_export_i64_tile_device(
          static_cast<const int8_t*>(device_residues),
          static_cast<int64_t*>(*export_buffer),
          static_cast<int>(rows),
          static_cast<int>(cols),
          static_cast<int>(entry.row_offset),
          static_cast<int>(entry.col_offset),
          static_cast<int>(entry.row_extent),
          static_cast<int>(entry.col_extent),
          static_cast<int>(entry.selected_prefix),
          bounds[static_cast<std::size_t>(index)],
          static_cast<int*>(*status_buffer));
      if (code != static_cast<int>(hipSuccess)) {
        return static_cast<hipError_t>(code);
      }
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
  (void)entries;
  (void)bounds;
  (void)entry_count;
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
    return prefix > RNS8_MAX_SUPPORTED_PREFIX ? RNS8_UNSUPPORTED_BACKEND : RNS8_INVALID_ARGUMENT;
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
  hipError_t err = timed_hip_operation(
      "crt_export_status_memset", [&]() { return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr); });
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
    const rns8_plan_tile_schedule_entry* entries,
    const uint64_t* bounds,
    uint64_t entry_count,
    uint64_t* dst,
    int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_residues || !export_buffer || !export_bytes || !status_buffer || !status_bytes || !entries || !bounds ||
      !dst || ld < cols || !checked_matrix_elements_i32(rows, cols) ||
      !checked_output_bytes(rows, cols, sizeof(uint64_t)) || entry_count == 0 ||
      entry_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
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
  hipError_t err = timed_hip_operation(
      "crt_export_status_memset", [&]() { return hipMemsetAsync(*status_buffer, 0, sizeof(int), nullptr); });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("crt_export_kernel", [&]() {
    for (uint64_t index = 0; index < entry_count; ++index) {
      const auto& entry = entries[static_cast<std::size_t>(index)];
      if (!checked_tile_entry(entry, rows, cols)) {
        return hipErrorInvalidValue;
      }
      const int code = rns8_hip_direct_export_u64_tile_device(
          static_cast<const int8_t*>(device_residues),
          static_cast<uint64_t*>(*export_buffer),
          static_cast<int>(rows),
          static_cast<int>(cols),
          static_cast<int>(entry.row_offset),
          static_cast<int>(entry.col_offset),
          static_cast<int>(entry.row_extent),
          static_cast<int>(entry.col_extent),
          static_cast<int>(entry.selected_prefix),
          bounds[static_cast<std::size_t>(index)],
          static_cast<int*>(*status_buffer));
      if (code != static_cast<int>(hipSuccess)) {
        return static_cast<hipError_t>(code);
      }
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
  (void)entries;
  (void)bounds;
  (void)entry_count;
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
    return prefix > RNS8_MAX_SUPPORTED_PREFIX ? RNS8_UNSUPPORTED_BACKEND : RNS8_INVALID_ARGUMENT;
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
    return prefix > RNS8_MAX_SUPPORTED_PREFIX ? RNS8_UNSUPPORTED_BACKEND : RNS8_INVALID_ARGUMENT;
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
