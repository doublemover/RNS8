#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr uint64_t kRns8ReciprocalScale = 1ull << 32u;

uint32_t rns8_modulus_reciprocal_host(int modulus) {
  return static_cast<uint32_t>(kRns8ReciprocalScale / static_cast<uint32_t>(modulus));
}

__device__ uint32_t rns8_reduce_u32_small_modulus_device(uint32_t value, uint32_t modulus, uint32_t reciprocal) {
  const uint32_t quotient = static_cast<uint32_t>((static_cast<uint64_t>(value) * reciprocal) >> 32u);
  uint32_t residue = value - quotient * modulus;
  residue -= modulus & (0u - static_cast<uint32_t>(residue >= modulus));
  residue -= modulus & (0u - static_cast<uint32_t>(residue >= modulus));
  return residue;
}

__device__ int8_t rns8_reduce_to_centered_device(int32_t value, int modulus, uint32_t reciprocal) {
  const uint32_t unsigned_modulus = static_cast<uint32_t>(modulus);
  const uint32_t magnitude =
      value < 0 ? static_cast<uint32_t>(-static_cast<int64_t>(value)) : static_cast<uint32_t>(value);
  uint32_t residue = rns8_reduce_u32_small_modulus_device(magnitude, unsigned_modulus, reciprocal);
  const uint32_t negative_nonzero =
      static_cast<uint32_t>(value < 0) & static_cast<uint32_t>(residue != 0);
  const uint32_t negative_mask = 0u - negative_nonzero;
  residue = (residue & ~negative_mask) | ((unsigned_modulus - residue) & negative_mask);
  const uint32_t threshold = (unsigned_modulus + 1u) / 2u;
  residue -= unsigned_modulus & (0u - static_cast<uint32_t>(residue >= threshold));
  return static_cast<int8_t>(static_cast<int32_t>(residue));
}

__global__ void rns8_hipblaslt_pack_transpose_centered_kernel(
    const int8_t* src,
    int8_t* dst,
    int src_rows,
    int src_cols,
    int src_ld,
    int dst_rows,
    int dst_cols,
    int dst_ld) {
  const int64_t elements = static_cast<int64_t>(dst_rows) * static_cast<int64_t>(dst_cols);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  const int dst_col = static_cast<int>(idx / dst_rows);
  const int dst_row = static_cast<int>(idx - static_cast<int64_t>(dst_col) * dst_rows);
  int8_t value = 0;
  if (dst_col < src_rows && dst_row < src_cols) {
    value = src[static_cast<int64_t>(dst_col) * src_ld + dst_row];
  }
  dst[static_cast<int64_t>(dst_col) * dst_ld + dst_row] = value;
}

__global__ void rns8_hipblaslt_reduce_i32_to_centered_kernel(
    const int32_t* scratch,
    int8_t* residues,
    int rows,
    int cols,
    int scratch_ld,
    int ldc,
    int modulus,
    uint32_t reciprocal,
    int accumulate) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  const int row = static_cast<int>(idx / cols);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  int32_t value = scratch[static_cast<int64_t>(row) * scratch_ld + col];
  if (accumulate) {
    value += static_cast<int32_t>(residues[static_cast<int64_t>(row) * ldc + col]);
  }
  residues[static_cast<int64_t>(row) * ldc + col] = rns8_reduce_to_centered_device(value, modulus, reciprocal);
}

}  // namespace

extern "C" int rns8_hipblaslt_reduce_i32_to_centered_device(
    const int32_t* scratch,
    int8_t* residues,
    int rows,
    int cols,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int accumulate) {
  if (!scratch || !residues || rows <= 0 || cols <= 0 || ldc < cols || modulus < 2 || modulus > 256 ||
      modulus_reciprocal != rns8_modulus_reciprocal_host(modulus)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      rns8_hipblaslt_reduce_i32_to_centered_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      nullptr,
      scratch,
      residues,
      rows,
      cols,
      cols,
      ldc,
      modulus,
      modulus_reciprocal,
      accumulate);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hipblaslt_pack_transpose_centered_device(
    const int8_t* src,
    int8_t* dst,
    int src_rows,
    int src_cols,
    int src_ld,
    int dst_rows,
    int dst_cols,
    int dst_ld) {
  if (!src || !dst || src_rows <= 0 || src_cols <= 0 || src_ld < src_cols || dst_rows < src_cols ||
      dst_cols < src_rows || dst_ld < dst_rows) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t elements = static_cast<int64_t>(dst_rows) * static_cast<int64_t>(dst_cols);
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      rns8_hipblaslt_pack_transpose_centered_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      nullptr,
      src,
      dst,
      src_rows,
      src_cols,
      src_ld,
      dst_rows,
      dst_cols,
      dst_ld);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hipblaslt_reduce_i32_to_centered_strided_device(
    const int32_t* scratch,
    int8_t* residues,
    int rows,
    int cols,
    int scratch_ld,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int accumulate) {
  if (!scratch || !residues || rows <= 0 || cols <= 0 || scratch_ld < cols || ldc < cols || modulus < 2 ||
      modulus > 256 || modulus_reciprocal != rns8_modulus_reciprocal_host(modulus)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      rns8_hipblaslt_reduce_i32_to_centered_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      nullptr,
      scratch,
      residues,
      rows,
      cols,
      scratch_ld,
      ldc,
      modulus,
      modulus_reciprocal,
      accumulate);
  return static_cast<int>(hipGetLastError());
}
