#include "backend_vector_alu/vector_alu_backend.hpp"

#include "backend_hip_direct/hip_backend.hpp"
#include "core/backend_common.hpp"

#include <limits>

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_vector_alu_i64_gemm_device(
    int device_id,
    const int64_t* a,
    const int64_t* b,
    int64_t* c,
    uint32_t* status,
    int64_t m,
    int64_t n,
    int64_t k);

extern "C" int rns8_vector_alu_u64_gemm_device(
    int device_id,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    uint32_t* status,
    int64_t m,
    int64_t n,
    int64_t k);
#endif

namespace rns8::detail {

bool vector_alu_compiled() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  return true;
#else
  return false;
#endif
}

rns8_status vector_alu_probe(int device_id, rns8_device_info& out) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  return hip_direct_probe(device_id, out);
#else
  static_cast<void>(device_id);
  static_cast<void>(out);
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

namespace {

bool checked_vector_shape(int64_t m, int64_t n, int64_t k) {
  return m > 0 && n > 0 && k > 0 && m <= std::numeric_limits<int>::max() &&
         n <= std::numeric_limits<int>::max() && k <= std::numeric_limits<int>::max();
}

bool vector_gemv_n1_shape(int64_t n, int64_t k) {
  return n == 1 && k >= 4096;
}

bool vector_gemv_small_n_shape(int64_t n, int64_t k) {
  return n > 1 && n <= 8 && k >= 512;
}

const char* vector_i64_event_label(int64_t n, int64_t k) {
  if (vector_gemv_n1_shape(n, k)) {
    return "vector_alu_i64_gemv_n1_kernel";
  }
  if (vector_gemv_small_n_shape(n, k)) {
    return "vector_alu_i64_gemv_small_n_kernel";
  }
  return "vector_alu_i64_kernel";
}

const char* vector_u64_event_label(int64_t n, int64_t k) {
  if (vector_gemv_n1_shape(n, k)) {
    return "vector_alu_u64_gemv_n1_kernel";
  }
  if (vector_gemv_small_n_shape(n, k)) {
    return "vector_alu_u64_gemv_small_n_kernel";
  }
  return "vector_alu_u64_kernel";
}

}  // namespace

rns8_status vector_alu_gemm_i64_device(
    int device_id,
    const void* device_a,
    const void* device_b,
    void* device_c,
    void* device_status,
    int64_t m,
    int64_t n,
    int64_t k) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a || !device_b || !device_c || !device_status || !checked_vector_shape(m, n, k)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const char* event_label = vector_i64_event_label(n, k);
  const int status = run_timed_device_code(event_label, [&]() {
    return rns8_vector_alu_i64_gemm_device(
        device_id,
        static_cast<const int64_t*>(device_a),
        static_cast<const int64_t*>(device_b),
        static_cast<int64_t*>(device_c),
        static_cast<uint32_t*>(device_status),
        m,
        n,
        k);
  });
  return status == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  static_cast<void>(device_id);
  static_cast<void>(device_a);
  static_cast<void>(device_b);
  static_cast<void>(device_c);
  static_cast<void>(device_status);
  static_cast<void>(m);
  static_cast<void>(n);
  static_cast<void>(k);
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status vector_alu_gemm_u64_device(
    int device_id,
    const void* device_a,
    const void* device_b,
    void* device_c,
    void* device_status,
    int64_t m,
    int64_t n,
    int64_t k) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a || !device_b || !device_c || !device_status || !checked_vector_shape(m, n, k)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const char* event_label = vector_u64_event_label(n, k);
  const int status = run_timed_device_code(event_label, [&]() {
    return rns8_vector_alu_u64_gemm_device(
        device_id,
        static_cast<const uint64_t*>(device_a),
        static_cast<const uint64_t*>(device_b),
        static_cast<uint64_t*>(device_c),
        static_cast<uint32_t*>(device_status),
        m,
        n,
        k);
  });
  return status == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  static_cast<void>(device_id);
  static_cast<void>(device_a);
  static_cast<void>(device_b);
  static_cast<void>(device_c);
  static_cast<void>(device_status);
  static_cast<void>(m);
  static_cast<void>(n);
  static_cast<void>(k);
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

}  // namespace rns8::detail
