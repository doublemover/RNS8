#include "backend_wmma/wmma_backend.hpp"

#include "backend_hip_direct/hip_backend.hpp"
#include "core/internal.hpp"

#include <limits>

#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
extern "C" int rns8_wmma_gemm_rns_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    unsigned long long workspace_bytes,
    long long m,
    long long n,
    long long k,
    long long lda,
    long long ldb,
    long long ldc,
    unsigned int prefix);

extern "C" int rns8_wmma_gemm_rns_tiled_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    unsigned long long workspace_bytes,
    long long m,
    long long n,
    long long k,
    long long lda,
    long long ldb,
    long long ldc,
    const rns8_plan_tile_schedule_entry* entries,
    unsigned long long entry_count);

extern "C" int rns8_wmma_gemm_finite_u8_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    unsigned long long workspace_bytes,
    long long m,
    long long n,
    long long k,
    long long lda,
    long long ldb,
    long long ldc,
    unsigned int modulus);
#endif

namespace rns8::detail {

namespace {

bool checked_mul_size(uint64_t a, uint64_t b, uint64_t& out) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    return false;
  }
  out = a * b;
  return true;
}

bool round_up_aligned(uint64_t value, uint64_t alignment, uint64_t& out) {
  if (value == 0 || alignment == 0) {
    return false;
  }
  const uint64_t remainder = value % alignment;
  if (remainder == 0) {
    out = value;
    return true;
  }
  const uint64_t delta = alignment - remainder;
  if (value > std::numeric_limits<uint64_t>::max() - delta) {
    return false;
  }
  out = value + delta;
  return true;
}

rns8_status status_from_wmma_code(int code) {
  switch (code) {
    case 0:
      return RNS8_SUCCESS;
    case 1:
      return RNS8_INVALID_ARGUMENT;
    case 2:
      return RNS8_UNSUPPORTED_BACKEND;
    case 3:
      return RNS8_BACKEND_FAILURE;
    case 4:
      return RNS8_RANGE_ERROR;
    default:
      return RNS8_BACKEND_FAILURE;
  }
}

}  // namespace

bool wmma_compiled() {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
  return true;
#else
  return false;
#endif
}

rns8_status wmma_probe(int device_id, rns8_device_info& out) {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
  rns8_status status = hip_direct_probe(device_id, out);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  out.backend = RNS8_BACKEND_WMMA;
  copy_c_string(out.detail, sizeof(out.detail), "rocWMMA accelerator backend detected through HIP runtime");
  return RNS8_SUCCESS;
#else
  (void)device_id;
  copy_c_string(out.detail, sizeof(out.detail), "RNS8 was built without the rocWMMA backend");
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

bool wmma_workspace_requirements(
    int64_t max_m,
    int64_t max_n,
    int64_t k,
    std::size_t& a_pack_bytes,
    std::size_t& b_pack_bytes,
    std::size_t& total_bytes) {
  a_pack_bytes = 0;
  b_pack_bytes = 0;
  total_bytes = 0;
  if (max_m <= 0 || max_n <= 0 || k <= 0) {
    return false;
  }
  constexpr uint64_t kSignedSafeKBlock = 65536;
  uint64_t k_block = static_cast<uint64_t>(k) < kSignedSafeKBlock ? static_cast<uint64_t>(k) : kSignedSafeKBlock;
  if (k_block < 16) {
    k_block = 16;
  }
  if (!round_up_aligned(k_block, 16, k_block)) {
    return false;
  }
  uint64_t padded_m = 0;
  uint64_t padded_n = 0;
  if (!round_up_aligned(static_cast<uint64_t>(max_m), 16, padded_m) ||
      !round_up_aligned(static_cast<uint64_t>(max_n), 16, padded_n)) {
    return false;
  }
  uint64_t a_bytes = 0;
  uint64_t b_bytes = 0;
  if (!checked_mul_size(padded_m, k_block, a_bytes) || !checked_mul_size(k_block, padded_n, b_bytes)) {
    return false;
  }
  if (a_bytes > std::numeric_limits<std::size_t>::max() ||
      b_bytes > std::numeric_limits<std::size_t>::max() ||
      a_bytes > std::numeric_limits<uint64_t>::max() - b_bytes ||
      a_bytes + b_bytes > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  a_pack_bytes = static_cast<std::size_t>(a_bytes);
  b_pack_bytes = static_cast<std::size_t>(b_bytes);
  total_bytes = static_cast<std::size_t>(a_bytes + b_bytes);
  return true;
}

rns8_status wmma_gemm_rns_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
  const int code = rns8_wmma_gemm_rns_device(
      device_id,
      device_a_residues,
      device_b_residues,
      device_c_residues,
      workspace,
      static_cast<unsigned long long>(workspace_bytes),
      static_cast<long long>(m),
      static_cast<long long>(n),
      static_cast<long long>(k),
      static_cast<long long>(lda),
      static_cast<long long>(ldb),
      static_cast<long long>(ldc),
      prefix);
  return status_from_wmma_code(code);
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)workspace;
  (void)workspace_bytes;
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

rns8_status wmma_gemm_rns_tiled_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count) {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
  const int code = rns8_wmma_gemm_rns_tiled_device(
      device_id,
      device_a_residues,
      device_b_residues,
      device_c_residues,
      workspace,
      static_cast<unsigned long long>(workspace_bytes),
      static_cast<long long>(m),
      static_cast<long long>(n),
      static_cast<long long>(k),
      static_cast<long long>(lda),
      static_cast<long long>(ldb),
      static_cast<long long>(ldc),
      entries,
      static_cast<unsigned long long>(entry_count));
  return status_from_wmma_code(code);
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)workspace;
  (void)workspace_bytes;
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

rns8_status wmma_gemm_finite_u8_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
  const int code = rns8_wmma_gemm_finite_u8_device(
      device_id,
      device_a_residues,
      device_b_residues,
      device_c_residues,
      workspace,
      static_cast<unsigned long long>(workspace_bytes),
      static_cast<long long>(m),
      static_cast<long long>(n),
      static_cast<long long>(k),
      static_cast<long long>(lda),
      static_cast<long long>(ldb),
      static_cast<long long>(ldc),
      modulus);
  return status_from_wmma_code(code);
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)workspace;
  (void)workspace_bytes;
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
