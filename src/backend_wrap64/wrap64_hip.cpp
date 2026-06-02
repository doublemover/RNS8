#include "backend_wrap64/wrap64_hip.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "backend_hip_direct/hip_backend.hpp"

#if RNS8_ENABLE_HIP
extern "C" int rns8_wrap64_hip_gemm_byte_limbs_device(
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    uint8_t* c_limbs,
    int64_t m,
    int64_t n,
    int64_t k);
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

#if RNS8_ENABLE_HIP
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

}  // namespace rns8::detail
