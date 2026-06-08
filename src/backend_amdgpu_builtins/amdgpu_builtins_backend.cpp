#include "backend_amdgpu_builtins/amdgpu_builtins_backend.hpp"

#include "backend_common/finite_u8_reducer.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include "core/backend_common.hpp"
#include "core/internal.hpp"

#include <cstring>
#include <limits>

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
#  include <hip/hip_runtime_api.h>

extern "C" int rns8_amdgpu_builtin_gemm_rns_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int prefix);

extern "C" int rns8_amdgpu_builtin_gemm_finite_u8_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus);

extern "C" int rns8_amdgpu_builtin_gemm_rns_sparse_a_device(
    int device_id,
    const void* device_a_packed_values,
    const void* device_a_packed_indices,
    const void* device_b_residues,
    void* device_c_residues,
    int m,
    int n,
    int k,
    int ldb,
    int ldc,
    int prefix);

extern "C" int rns8_amdgpu_builtin_gemm_finite_u8_sparse_a_device(
    int device_id,
    const void* device_a_packed_values,
    const void* device_a_packed_indices,
    const void* device_b_residues,
    void* device_c_residues,
    int m,
    int n,
    int k,
    int ldb,
    int ldc,
    int modulus);
#endif

namespace rns8::detail {

namespace {

bool fits_int_argument(int64_t value) {
  return value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

bool valid_dense_shape(
    const void* a,
    const void* b,
    const void* c,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  return a && b && c && m > 0 && n > 0 && k > 0 && lda >= k && ldb >= n && ldc >= n &&
         fits_int_argument(m) && fits_int_argument(n) && fits_int_argument(k) && fits_int_argument(lda) &&
         fits_int_argument(ldb) && fits_int_argument(ldc);
}

bool valid_sparse_shape(
    const void* values,
    const void* indices,
    const void* b,
    const void* c,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    int64_t ldc) {
  return values && indices && b && c && m > 0 && n > 0 && k > 0 && (k % 4) == 0 && ldb >= n && ldc >= n &&
         fits_int_argument(m) && fits_int_argument(n) && fits_int_argument(k) && fits_int_argument(ldb) &&
         fits_int_argument(ldc);
}

}  // namespace

bool amdgpu_builtins_compiled() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  return true;
#else
  return false;
#endif
}

rns8_status amdgpu_builtins_probe(int device_id, rns8_device_info& out) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  const rns8_status status = hip_direct_probe(device_id, out);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const bool supported = std::strncmp(out.gcn_arch, "gfx942", 6) == 0 ||
                         std::strncmp(out.gcn_arch, "gfx110", 6) == 0 ||
                         std::strncmp(out.gcn_arch, "gfx1200", 7) == 0 ||
                         std::strncmp(out.gcn_arch, "gfx1201", 7) == 0;
  if (!supported) {
    copy_c_string(out.detail, sizeof(out.detail), "AMDGPU builtin kernels require gfx942, gfx110x, gfx1200, or gfx1201");
    return RNS8_UNSUPPORTED_BACKEND;
  }
  out.backend = RNS8_BACKEND_AMDGPU_BUILTINS;
  copy_c_string(out.detail, sizeof(out.detail), "AMDGPU builtin matrix-core kernels detected for this target");
  return RNS8_SUCCESS;
#else
  (void)device_id;
  copy_c_string(out.detail, sizeof(out.detail), "RNS8 was built without AMDGPU builtin kernels");
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

bool amdgpu_builtins_workspace_requirements(int64_t max_m, int64_t max_n, int64_t k, std::size_t& total_bytes) {
  total_bytes = 0;
  return max_m > 0 && max_n > 0 && k > 0 && k <= RNS8_SAFE_INT32_K_BLOCK;
}

rns8_status amdgpu_builtins_gemm_rns_device(
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
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  if (!valid_dense_shape(device_a_residues, device_b_residues, device_c_residues, m, n, k, lda, ldb, ldc) ||
      prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (k > RNS8_SAFE_INT32_K_BLOCK) {
    return RNS8_RANGE_ERROR;
  }
  const int code = run_timed_device_code("amdgpu_builtin_rns_matrix_core_kernel", [&]() {
    return rns8_amdgpu_builtin_gemm_rns_device(
        device_id,
        device_a_residues,
        device_b_residues,
        device_c_residues,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(prefix));
  });
  return status_from_device_code(code);
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

rns8_status amdgpu_builtins_gemm_finite_u8_device(
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
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  if (!valid_dense_shape(device_a_residues, device_b_residues, device_c_residues, m, n, k, lda, ldb, ldc) ||
      !finite_u8::static_byte_modulus_supported(modulus)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (k > RNS8_SAFE_INT32_K_BLOCK) {
    return RNS8_RANGE_ERROR;
  }
  const int code = run_timed_device_code("amdgpu_builtin_finite_matrix_core_kernel", [&]() {
    return rns8_amdgpu_builtin_gemm_finite_u8_device(
        device_id,
        device_a_residues,
        device_b_residues,
        device_c_residues,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(modulus));
  });
  return status_from_device_code(code);
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

rns8_status amdgpu_builtins_gemm_rns_sparse_a_device(
    int device_id,
    const void* device_a_packed_values,
    const void* device_a_packed_indices,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  if (!valid_sparse_shape(device_a_packed_values, device_a_packed_indices, device_b_residues, device_c_residues, m, n, k, ldb, ldc) ||
      prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (k > RNS8_SAFE_INT32_K_BLOCK) {
    return RNS8_RANGE_ERROR;
  }
  const int code = run_timed_device_code("amdgpu_builtin_sparse_a_rns_matrix_core_kernel", [&]() {
    return rns8_amdgpu_builtin_gemm_rns_sparse_a_device(
        device_id,
        device_a_packed_values,
        device_a_packed_indices,
        device_b_residues,
        device_c_residues,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(prefix));
  });
  return status_from_device_code(code);
#else
  (void)device_id;
  (void)device_a_packed_values;
  (void)device_a_packed_indices;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)ldb;
  (void)ldc;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status amdgpu_builtins_gemm_finite_u8_sparse_a_device(
    int device_id,
    const void* device_a_packed_values,
    const void* device_a_packed_indices,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  if (!valid_sparse_shape(device_a_packed_values, device_a_packed_indices, device_b_residues, device_c_residues, m, n, k, ldb, ldc) ||
      !finite_u8::static_byte_modulus_supported(modulus)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (k > RNS8_SAFE_INT32_K_BLOCK) {
    return RNS8_RANGE_ERROR;
  }
  const int code = run_timed_device_code("amdgpu_builtin_sparse_a_finite_matrix_core_kernel", [&]() {
    return rns8_amdgpu_builtin_gemm_finite_u8_sparse_a_device(
        device_id,
        device_a_packed_values,
        device_a_packed_indices,
        device_b_residues,
        device_c_residues,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(modulus));
  });
  return status_from_device_code(code);
#else
  (void)device_id;
  (void)device_a_packed_values;
  (void)device_a_packed_indices;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)ldb;
  (void)ldc;
  (void)modulus;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

}  // namespace rns8::detail
