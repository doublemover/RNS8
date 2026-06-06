#include "backend_wrap64/wrap64_hip.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "backend_hip_direct/hip_backend.hpp"
#include "core/hip_resources.hpp"

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

struct Wrap64CompactLayout {
  std::size_t byte_limb_bytes = 0;
};

constexpr uint64_t kWrap64ByteLimbsPerCell = 8;
constexpr uint64_t kWrap64LowProductDiagonals = kWrap64ByteLimbsPerCell;
constexpr uint64_t kWrap64LowProductPairCount =
    (kWrap64LowProductDiagonals * (kWrap64LowProductDiagonals + 1ull)) / 2ull;
constexpr uint64_t kWrap64MaxUnsignedByteProduct = 255ull * 255ull;
constexpr uint64_t kWrap64MaxLowDiagonalProductsPerK = kWrap64LowProductDiagonals;
constexpr uint64_t kWrap64MaxLowDiagonalColumnPerK =
    kWrap64MaxLowDiagonalProductsPerK * kWrap64MaxUnsignedByteProduct;
constexpr uint64_t kWrap64MaxCarryInflatedLowDiagonalColumnPerK =
    2ull * kWrap64MaxLowDiagonalColumnPerK;
static_assert(kWrap64LowProductPairCount == 36ull, "low 64-bit wrap64 uses exactly 36 byte-product pairs");

bool checked_diagonal_accumulator_capacity(int64_t k) {
  if (k <= 0) {
    return false;
  }
  return static_cast<uint64_t>(k) <=
         std::numeric_limits<uint64_t>::max() / kWrap64MaxCarryInflatedLowDiagonalColumnPerK;
}

bool checked_limb_bytes(int64_t rows, int64_t cols, std::size_t* bytes) {
  if (!bytes || rows <= 0 || cols <= 0) {
    return false;
  }
  const auto u_rows = static_cast<uint64_t>(rows);
  const auto u_cols = static_cast<uint64_t>(cols);
  const uint64_t max_bytes = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max());
  if (u_cols != 0 && u_rows > max_bytes / u_cols / kWrap64ByteLimbsPerCell) {
    return false;
  }
  *bytes = static_cast<std::size_t>(u_rows * u_cols * kWrap64ByteLimbsPerCell);
  return true;
}

bool checked_compact_byte_limb_layout(int64_t rows, int64_t cols, Wrap64CompactLayout* layout) {
  if (!layout) {
    return false;
  }
  std::size_t bytes = 0;
  if (!checked_limb_bytes(rows, cols, &bytes)) {
    return false;
  }
  layout->byte_limb_bytes = bytes;
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

bool checked_wrap64_gemm_compact_layouts(
    int64_t m,
    int64_t n,
    int64_t k,
    Wrap64CompactLayout* a,
    Wrap64CompactLayout* b,
    Wrap64CompactLayout* c) {
  return checked_matrix_elements_i32(m, n) && checked_matrix_elements_i32(m, k) &&
         checked_diagonal_accumulator_capacity(k) &&
         checked_matrix_elements_i32(k, n) && checked_compact_byte_limb_layout(m, k, a) &&
         checked_compact_byte_limb_layout(k, n, b) && checked_compact_byte_limb_layout(m, n, c);
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

  hip_unique_event_pair events;
  hipError_t event_status = events.create_and_record_start();
  if (event_status != hipSuccess) {
    return fn();
  }

  const hipError_t op_status = fn();
  if (op_status == hipSuccess) {
    event_status = events.record_stop();
    if (event_status == hipSuccess) {
      event_status = hipEventSynchronize(events.stop());
    }
    if (event_status == hipSuccess) {
      float milliseconds = 0.0f;
      event_status = hipEventElapsedTime(&milliseconds, events.start(), events.stop());
      if (event_status == hipSuccess && milliseconds >= 0.0f) {
        hip_direct_timing_record_sample(label, static_cast<double>(milliseconds) * 1000.0);
      }
    }
  }
  return op_status;
}

#endif

}  // namespace

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
  Wrap64CompactLayout compact_layout;
  if (!checked_matrix_elements_i32(rows, cols) || !checked_u64_row_pitch_bytes(rows, ld, &source_bytes) ||
      !checked_compact_byte_limb_layout(rows, cols, &compact_layout)) {
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
  (void)compact_layout;
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
  Wrap64CompactLayout a_layout;
  Wrap64CompactLayout b_layout;
  Wrap64CompactLayout c_layout;
  if (!device_a_limbs || !device_b_limbs || !device_c_limbs ||
      !checked_wrap64_gemm_compact_layouts(m, n, k, &a_layout, &b_layout, &c_layout)) {
    return RNS8_INVALID_ARGUMENT;
  }
#if RNS8_ENABLE_HIP
  rns8_status status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const char* event_label = wrap64_hip_gemm_event_label_for_shape(m, n, k);
  const hipError_t err = timed_hip_operation(event_label, [&]() {
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
  (void)a_layout;
  (void)b_layout;
  (void)c_layout;
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
  std::size_t destination_bytes = 0;
  Wrap64CompactLayout compact_layout;
  if (!checked_u64_compact_bytes(rows, cols, &output_bytes) ||
      !checked_u64_row_pitch_bytes(rows, ld, &destination_bytes) ||
      !checked_compact_byte_limb_layout(rows, cols, &compact_layout)) {
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
  status = hip_direct_copy_compact_matrix_device_to_host(
      device_id, "wrap64_export_d2h", dst, ld, *export_buffer, rows, cols, sizeof(uint64_t), false);
  return status;
#else
  (void)device_id;
  (void)export_buffer;
  (void)export_bytes;
  (void)output_bytes;
  (void)destination_bytes;
  (void)compact_layout;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

}  // namespace rns8::detail
