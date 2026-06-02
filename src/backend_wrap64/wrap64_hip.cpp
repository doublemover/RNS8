#include "backend_wrap64/wrap64_hip.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "backend_hip_direct/hip_backend.hpp"

#if RNS8_ENABLE_HIP
#  include <hip/hip_runtime_api.h>

extern "C" int rns8_wrap64_hip_pack_u64_device(
    const uint64_t* src,
    uint8_t* byte_limbs,
    int64_t rows,
    int64_t cols,
    int64_t ld);

extern "C" int rns8_wrap64_hip_gemm_byte_limbs_device(
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    uint8_t* c_limbs,
    int64_t m,
    int64_t n,
    int64_t k);

extern "C" int rns8_wrap64_hip_export_u64_device(
    const uint8_t* byte_limbs,
    uint64_t* dst,
    int64_t rows,
    int64_t cols);
#endif

namespace rns8::detail {

namespace {

bool checked_limb_bytes(int64_t rows, int64_t cols, std::size_t* bytes) {
  if (!bytes || rows <= 0 || cols <= 0) {
    return false;
  }
  const auto u_rows = static_cast<uint64_t>(rows);
  const auto u_cols = static_cast<uint64_t>(cols);
  constexpr uint64_t limbs_per_cell = 8;
  const uint64_t max_bytes = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max());
  if (u_cols != 0 && u_rows > max_bytes / u_cols / limbs_per_cell) {
    return false;
  }
  *bytes = static_cast<std::size_t>(u_rows * u_cols * limbs_per_cell);
  return true;
}

bool checked_matrix_elements_i32(int64_t rows, int64_t cols) {
  if (rows <= 0 || cols <= 0 || rows > std::numeric_limits<int>::max() || cols > std::numeric_limits<int>::max()) {
    return false;
  }
  return static_cast<uint64_t>(rows) <=
         static_cast<uint64_t>(std::numeric_limits<int>::max()) / static_cast<uint64_t>(cols);
}

bool checked_u64_row_pitch_bytes(int64_t rows, int64_t ld, std::size_t* bytes) {
  if (!bytes || rows <= 0 || ld <= 0) {
    return false;
  }
  const uint64_t max_items = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(uint64_t));
  if (static_cast<uint64_t>(rows) > max_items / static_cast<uint64_t>(ld)) {
    return false;
  }
  *bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(ld) * sizeof(uint64_t);
  return true;
}

bool checked_u64_compact_bytes(int64_t rows, int64_t cols, std::size_t* bytes) {
  if (!bytes || rows <= 0 || cols <= 0) {
    return false;
  }
  const uint64_t max_items = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(uint64_t));
  if (static_cast<uint64_t>(rows) > max_items / static_cast<uint64_t>(cols)) {
    return false;
  }
  *bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(uint64_t);
  return true;
}

#if RNS8_ENABLE_HIP
rns8_status set_hip_device(int device_id) {
  if (device_id < 0) {
    device_id = 0;
  }
  return hipSetDevice(device_id) == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

template <typename Fn>
hipError_t timed_hip_operation(const char* label, Fn&& fn) {
  if (!hip_direct_timing_enabled()) {
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
        hip_direct_timing_record_sample(label, static_cast<double>(milliseconds) * 1000.0);
      }
    }
  }

  (void)hipEventDestroy(stop);
  (void)hipEventDestroy(start);
  return op_status;
}

rns8_status free_if_allocated(int device_id, void* ptr) {
  return ptr ? hip_direct_free(device_id, ptr) : RNS8_SUCCESS;
}
#endif

}  // namespace

rns8_status wrap64_hip_gemm_byte_limbs(
    int device_id,
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    uint8_t* c_limbs,
    int64_t m,
    int64_t n,
    int64_t k) {
  if (!a_limbs || !b_limbs || !c_limbs) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!checked_matrix_elements_i32(m, n) || !checked_matrix_elements_i32(m, k) ||
      !checked_matrix_elements_i32(k, n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  std::size_t a_bytes = 0;
  std::size_t b_bytes = 0;
  std::size_t c_bytes = 0;
  if (!checked_limb_bytes(m, k, &a_bytes) || !checked_limb_bytes(k, n, &b_bytes) ||
      !checked_limb_bytes(m, n, &c_bytes)) {
    return RNS8_INVALID_ARGUMENT;
  }
#if RNS8_ENABLE_HIP
  void* d_a = nullptr;
  void* d_b = nullptr;
  void* d_c = nullptr;
  rns8_status status = hip_direct_allocate(device_id, a_bytes, &d_a);
  if (status == RNS8_SUCCESS) {
    status = hip_direct_allocate(device_id, b_bytes, &d_b);
  }
  if (status == RNS8_SUCCESS) {
    status = hip_direct_allocate(device_id, c_bytes, &d_c);
  }
  if (status == RNS8_SUCCESS) {
    status = hip_direct_copy_host_to_device(device_id, d_a, a_limbs, a_bytes);
  }
  if (status == RNS8_SUCCESS) {
    status = hip_direct_copy_host_to_device(device_id, d_b, b_limbs, b_bytes);
  }
  if (status == RNS8_SUCCESS) {
    const int code = rns8_wrap64_hip_gemm_byte_limbs_device(
        static_cast<const uint8_t*>(d_a),
        static_cast<const uint8_t*>(d_b),
        static_cast<uint8_t*>(d_c),
        m,
        n,
        k);
    status = code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
  }
  if (status == RNS8_SUCCESS) {
    status = hip_direct_synchronize(device_id);
  }
  if (status == RNS8_SUCCESS) {
    status = hip_direct_copy_device_to_host(device_id, c_limbs, d_c, c_bytes);
  }

  const rns8_status free_c = free_if_allocated(device_id, d_c);
  const rns8_status free_b = free_if_allocated(device_id, d_b);
  const rns8_status free_a = free_if_allocated(device_id, d_a);
  if (status == RNS8_SUCCESS && free_c != RNS8_SUCCESS) status = free_c;
  if (status == RNS8_SUCCESS && free_b != RNS8_SUCCESS) status = free_b;
  if (status == RNS8_SUCCESS && free_a != RNS8_SUCCESS) status = free_a;
  return status;
#else
  (void)device_id;
  (void)a_bytes;
  (void)b_bytes;
  (void)c_bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status wrap64_hip_pack_u64_device(
    int device_id,
    const uint64_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_byte_limbs,
    int64_t rows,
    int64_t cols,
    int64_t ld) {
  if (!src || !upload_buffer || !upload_bytes || !device_byte_limbs || ld < cols) {
    return RNS8_INVALID_ARGUMENT;
  }
  std::size_t source_bytes = 0;
  std::size_t limb_bytes = 0;
  if (!checked_matrix_elements_i32(rows, cols) || !checked_u64_row_pitch_bytes(rows, ld, &source_bytes) ||
      !checked_limb_bytes(rows, cols, &limb_bytes)) {
    return RNS8_INVALID_ARGUMENT;
  }
#if RNS8_ENABLE_HIP
  rns8_status status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = hip_direct_ensure_upload_buffer(device_id, source_bytes, upload_buffer, upload_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation("pack_h2d", [&]() {
    return hipMemcpy(*upload_buffer, src, source_bytes, hipMemcpyHostToDevice);
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("pack_kernel", [&]() {
    const int code = rns8_wrap64_hip_pack_u64_device(
        static_cast<const uint64_t*>(*upload_buffer),
        static_cast<uint8_t*>(device_byte_limbs),
        rows,
        cols,
        ld);
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)upload_buffer;
  (void)upload_bytes;
  (void)device_byte_limbs;
  (void)source_bytes;
  (void)limb_bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status wrap64_hip_gemm_byte_limbs_device_resident(
    int device_id,
    const void* device_a_limbs,
    const void* device_b_limbs,
    void* device_c_limbs,
    int64_t m,
    int64_t n,
    int64_t k) {
  if (!device_a_limbs || !device_b_limbs || !device_c_limbs || !checked_matrix_elements_i32(m, n) ||
      !checked_matrix_elements_i32(m, k) || !checked_matrix_elements_i32(k, n)) {
    return RNS8_INVALID_ARGUMENT;
  }
#if RNS8_ENABLE_HIP
  rns8_status status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const hipError_t err = timed_hip_operation("wrap64_tiled_byte_gemm_kernel", [&]() {
    const int code = rns8_wrap64_hip_gemm_byte_limbs_device(
        static_cast<const uint8_t*>(device_a_limbs),
        static_cast<const uint8_t*>(device_b_limbs),
        static_cast<uint8_t*>(device_c_limbs),
        m,
        n,
        k);
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status wrap64_hip_export_u64_device(
    int device_id,
    const void* device_byte_limbs,
    void** export_buffer,
    std::size_t* export_bytes,
    int64_t rows,
    int64_t cols,
    uint64_t* dst,
    int64_t ld) {
  if (!device_byte_limbs || !export_buffer || !export_bytes || !dst || ld < cols ||
      !checked_matrix_elements_i32(rows, cols)) {
    return RNS8_INVALID_ARGUMENT;
  }
  std::size_t output_bytes = 0;
  if (!checked_u64_compact_bytes(rows, cols, &output_bytes)) {
    return RNS8_INVALID_ARGUMENT;
  }
#if RNS8_ENABLE_HIP
  rns8_status status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = hip_direct_ensure_upload_buffer(device_id, output_bytes, export_buffer, export_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  hipError_t err = timed_hip_operation("wrap64_export_kernel", [&]() {
    const int code = rns8_wrap64_hip_export_u64_device(
        static_cast<const uint8_t*>(device_byte_limbs), static_cast<uint64_t*>(*export_buffer), rows, cols);
    if (code != static_cast<int>(hipSuccess)) {
      return static_cast<hipError_t>(code);
    }
    return hipDeviceSynchronize();
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  err = timed_hip_operation("wrap64_export_d2h", [&]() {
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
  (void)export_buffer;
  (void)export_bytes;
  (void)output_bytes;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

}  // namespace rns8::detail
