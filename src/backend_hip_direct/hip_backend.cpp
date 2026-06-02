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
#endif

namespace rns8::detail {

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

rns8_status hip_direct_pack_i64(
    int device_id,
    const int64_t* src,
    int8_t* residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src || !residues || rows <= 0 || cols <= 0 || ld < cols || prefix == 0 ||
      prefix > RNS8_DEFAULT_MODULUS_COUNT) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (rows > std::numeric_limits<int>::max() || cols > std::numeric_limits<int>::max() ||
      ld > std::numeric_limits<int>::max() || prefix > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return RNS8_INVALID_ARGUMENT;
  }
  const uint64_t max_pack_elements = static_cast<uint64_t>(std::numeric_limits<int>::max()) * 256u;
  if (static_cast<uint64_t>(rows) > max_pack_elements / static_cast<uint64_t>(cols) / prefix) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (device_id < 0) {
    device_id = 0;
  }
  hipError_t err = hipSetDevice(device_id);
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
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

rns8_status hip_direct_pack_u64(
    int device_id,
    const uint64_t* src,
    int8_t* residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!src || !residues || rows <= 0 || cols <= 0 || ld < cols || prefix == 0 ||
      prefix > RNS8_DEFAULT_MODULUS_COUNT) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (rows > std::numeric_limits<int>::max() || cols > std::numeric_limits<int>::max() ||
      ld > std::numeric_limits<int>::max() || prefix > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return RNS8_INVALID_ARGUMENT;
  }
  const uint64_t max_pack_elements = static_cast<uint64_t>(std::numeric_limits<int>::max()) * 256u;
  if (static_cast<uint64_t>(rows) > max_pack_elements / static_cast<uint64_t>(cols) / prefix) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (device_id < 0) {
    device_id = 0;
  }
  hipError_t err = hipSetDevice(device_id);
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
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
  if (device_id < 0) {
    device_id = 0;
  }
  hipError_t err = hipSetDevice(device_id);
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
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

}  // namespace rns8::detail
