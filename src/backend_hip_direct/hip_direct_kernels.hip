#include <hip/hip_runtime.h>

#include <cstdint>

#include "../backend_common/finite_u8_reducer.hpp"

namespace {

struct rns8_plan_tile_schedule_entry {
  uint64_t struct_size;
  uint32_t abi_version;
  uint32_t flags;
  uint64_t tile_row;
  uint64_t tile_col;
  int64_t row_offset;
  int64_t col_offset;
  int64_t row_extent;
  int64_t col_extent;
  uint32_t required_prefix;
  uint32_t selected_prefix;
  uint32_t group_index;
  uint32_t range_bit_length;
};

constexpr int kRns8DefaultModulusCount = 28;

constexpr int rns8_default_moduli_host[kRns8DefaultModulusCount] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

#include "hip_direct_device_common.cuh"

#include "hip_direct_pack_kernels.cuh"

#include "hip_direct_gemm_kernels.cuh"

#include "hip_direct_export_kernels.cuh"

}  // namespace

extern "C" int rns8_hip_direct_pack_i64_device(
    const int64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const int threads = 256;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t total =
      static_cast<int64_t>(prefix) * elements;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const int fixed_blocks = static_cast<int>((elements + threads - 1) / threads);
  switch (prefix) {
#define RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(PrefixValue) \
    case PrefixValue:                                  \
      hipLaunchKernelGGL(                              \
          HIP_KERNEL_NAME(rns8_pack_i64_fixed_prefix_kernel<PrefixValue>), \
          dim3(fixed_blocks),                          \
          dim3(threads),                               \
          0,                                           \
          nullptr,                                     \
          d_src,                                       \
          d_residues,                                  \
          rows,                                        \
          cols,                                        \
          ld);                                         \
      break
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(1);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(2);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(3);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(4);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(5);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(6);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(7);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(8);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(kRns8DefaultBoundedPrefix);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(kRns8MaxSupportedPrefix);
#undef RNS8_LAUNCH_PACK_I64_FIXED_PREFIX
    default:
      hipLaunchKernelGGL(
          rns8_pack_i64_kernel,
          dim3(blocks),
          dim3(threads),
          0,
          nullptr,
          d_src,
          d_residues,
          rows,
          cols,
          ld,
          prefix);
      break;
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_pack_u64_device(
    const uint64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const int threads = 256;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t total =
      static_cast<int64_t>(prefix) * elements;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const int fixed_blocks = static_cast<int>((elements + threads - 1) / threads);
  switch (prefix) {
#define RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(PrefixValue) \
    case PrefixValue:                                  \
      hipLaunchKernelGGL(                              \
          HIP_KERNEL_NAME(rns8_pack_u64_fixed_prefix_kernel<PrefixValue>), \
          dim3(fixed_blocks),                          \
          dim3(threads),                               \
          0,                                           \
          nullptr,                                     \
          d_src,                                       \
          d_residues,                                  \
          rows,                                        \
          cols,                                        \
          ld);                                         \
      break
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(1);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(2);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(3);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(4);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(5);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(6);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(7);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(8);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(kRns8DefaultBoundedPrefix);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(kRns8MaxSupportedPrefix);
#undef RNS8_LAUNCH_PACK_U64_FIXED_PREFIX
    default:
      hipLaunchKernelGGL(
          rns8_pack_u64_kernel,
          dim3(blocks),
          dim3(threads),
          0,
          nullptr,
          d_src,
          d_residues,
          rows,
          cols,
          ld,
          prefix);
      break;
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_pack_i64_grouped_device(
    const int64_t* d_src,
    int8_t* const* d_residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld,
    int prefix) {
  if (!d_src || !d_residue_ptrs || task_count <= 0 || src_task_stride <= 0 ||
      rows <= 0 || cols <= 0 || ld < cols || prefix <= 0 || prefix > kRns8MaxSupportedPrefix) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t per_task_total = static_cast<int64_t>(prefix) * elements;
  const int blocks = static_cast<int>((per_task_total + threads - 1) / threads);
  const int fixed_blocks = static_cast<int>((elements + threads - 1) / threads);
  switch (prefix) {
#define RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(PrefixValue) \
    case PrefixValue:                                          \
      hipLaunchKernelGGL(                                      \
          HIP_KERNEL_NAME(rns8_pack_i64_grouped_fixed_prefix_kernel<PrefixValue>), \
          dim3(fixed_blocks, task_count),                      \
          dim3(threads),                                       \
          0,                                                   \
          nullptr,                                             \
          d_src,                                               \
          d_residue_ptrs,                                      \
          task_count,                                          \
          src_task_stride,                                     \
          rows,                                                \
          cols,                                                \
          ld);                                                 \
      break
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(1);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(2);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(3);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(4);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(5);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(6);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(7);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(8);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(kRns8DefaultBoundedPrefix);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(kRns8MaxSupportedPrefix);
#undef RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX
    default:
      hipLaunchKernelGGL(
          rns8_pack_i64_grouped_kernel,
          dim3(blocks, task_count),
          dim3(threads),
          0,
          nullptr,
          d_src,
          d_residue_ptrs,
          task_count,
          src_task_stride,
          rows,
          cols,
          ld,
          prefix);
      break;
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_pack_u64_grouped_device(
    const uint64_t* d_src,
    int8_t* const* d_residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld,
    int prefix) {
  if (!d_src || !d_residue_ptrs || task_count <= 0 || src_task_stride <= 0 ||
      rows <= 0 || cols <= 0 || ld < cols || prefix <= 0 || prefix > kRns8MaxSupportedPrefix) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t per_task_total = static_cast<int64_t>(prefix) * elements;
  const int blocks = static_cast<int>((per_task_total + threads - 1) / threads);
  const int fixed_blocks = static_cast<int>((elements + threads - 1) / threads);
  switch (prefix) {
#define RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(PrefixValue) \
    case PrefixValue:                                          \
      hipLaunchKernelGGL(                                      \
          HIP_KERNEL_NAME(rns8_pack_u64_grouped_fixed_prefix_kernel<PrefixValue>), \
          dim3(fixed_blocks, task_count),                      \
          dim3(threads),                                       \
          0,                                                   \
          nullptr,                                             \
          d_src,                                               \
          d_residue_ptrs,                                      \
          task_count,                                          \
          src_task_stride,                                     \
          rows,                                                \
          cols,                                                \
          ld);                                                 \
      break
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(1);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(2);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(3);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(4);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(5);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(6);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(7);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(8);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(kRns8DefaultBoundedPrefix);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(kRns8MaxSupportedPrefix);
#undef RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX
    default:
      hipLaunchKernelGGL(
          rns8_pack_u64_grouped_kernel,
          dim3(blocks, task_count),
          dim3(threads),
          0,
          nullptr,
          d_src,
          d_residue_ptrs,
          task_count,
          src_task_stride,
          rows,
          cols,
          ld,
          prefix);
      break;
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_pack_u8_modulus_device(
    const uint8_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int modulus) {
  if (!d_src || !d_residues || rows <= 0 || cols <= 0 || ld < cols || modulus < 2 || modulus > 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t total = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  if (modulus == 256) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_pack_u8_fixed_modulus_kernel<256>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_src,
        d_residues,
        rows,
        cols,
        ld);
    return static_cast<int>(hipGetLastError());
  }
  if (modulus == 255) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_pack_u8_fixed_modulus_kernel<255>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_src,
        d_residues,
        rows,
        cols,
        ld);
    return static_cast<int>(hipGetLastError());
  }
  if (modulus == 251) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_pack_u8_fixed_modulus_kernel<251>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_src,
        d_residues,
        rows,
        cols,
        ld);
    return static_cast<int>(hipGetLastError());
  }
  hipLaunchKernelGGL(
      rns8_pack_u8_modulus_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      nullptr,
      d_src,
      d_residues,
      rows,
      cols,
      ld,
      modulus);
  return static_cast<int>(hipGetLastError());
}

int rns8_hip_direct_ring_gemm_i8_on_stream(
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
    hipStream_t stream) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || modulus <= 0 ||
      modulus_index < 0 || selected_prefix <= 0 || modulus_index >= selected_prefix ||
      selected_prefix > kRns8DefaultModulusCount ||
      modulus != rns8_default_moduli_host[modulus_index] ||
      modulus_reciprocal != rns8_modulus_reciprocal_host(modulus) ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_i8_i32_tiled_kernel,
        grid,
        block,
        0,
        stream,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        modulus,
        modulus_reciprocal,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

template <int Modulus>
hipError_t launch_finite_ring_fixed_modulus_gemm(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block,
    dim3 grid,
    dim3 block) {
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_finite_ring_gemm_i8_i32_fixed_modulus_kernel<Modulus>),
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || modulus < 2 ||
      modulus > 256 || modulus_reciprocal != rns8_modulus_reciprocal_host(modulus) ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  if (modulus == 256) {
    return static_cast<int>(launch_finite_ring_fixed_modulus_gemm<256>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 255) {
    return static_cast<int>(launch_finite_ring_fixed_modulus_gemm<255>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 251) {
    return static_cast<int>(launch_finite_ring_fixed_modulus_gemm<251>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_i8_i32_tiled_kernel,
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        modulus,
        modulus_reciprocal,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

template <int Modulus>
hipError_t launch_finite_ring_native_fixed_modulus_gemm(
    const uint8_t* d_a,
    const uint8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block,
    dim3 grid,
    dim3 block) {
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_finite_ring_gemm_u8_native_i32_fixed_modulus_kernel<Modulus>),
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || modulus < 2 ||
      modulus > 256 || modulus_reciprocal != rns8_modulus_reciprocal_host(modulus) ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  if (modulus == 256) {
    return static_cast<int>(launch_finite_ring_native_fixed_modulus_gemm<256>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 255) {
    return static_cast<int>(launch_finite_ring_native_fixed_modulus_gemm<255>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 251) {
    return static_cast<int>(launch_finite_ring_native_fixed_modulus_gemm<251>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_finite_ring_gemm_u8_native_i32_kernel,
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        modulus,
        modulus_reciprocal,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

template <int Modulus>
hipError_t launch_finite_ring_native_a_fixed_modulus_gemm(
    const uint8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block,
    dim3 grid,
    dim3 block) {
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_finite_ring_gemm_u8_native_a_i8_b_i32_fixed_modulus_kernel<Modulus>),
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || modulus < 2 ||
      modulus > 256 || modulus_reciprocal != rns8_modulus_reciprocal_host(modulus) ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  if (modulus == 256) {
    return static_cast<int>(launch_finite_ring_native_a_fixed_modulus_gemm<256>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 255) {
    return static_cast<int>(launch_finite_ring_native_a_fixed_modulus_gemm<255>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 251) {
    return static_cast<int>(launch_finite_ring_native_a_fixed_modulus_gemm<251>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_finite_ring_gemm_u8_native_a_i8_b_i32_kernel,
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        modulus,
        modulus_reciprocal,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

int rns8_hip_direct_ring_gemm_i8_grouped_prefix_on_stream(
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
    hipStream_t stream) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      grouped_prefix <= 0 || grouped_prefix > kRns8MaxSupportedPrefix ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / grouped_prefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(lda) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb) ||
      static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, grouped_prefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_i8_i32_grouped_prefix_kernel,
        grid,
        block,
        0,
        stream,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        grouped_prefix,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

int rns8_hip_direct_ring_gemm_i8_grouped_task_prefix_on_stream(
    const int8_t* const* d_a_ptrs,
    const int8_t* const* d_b_ptrs,
    int8_t* const* d_c_ptrs,
    int task_count,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block,
    hipStream_t stream) {
  if (!d_a_ptrs || !d_b_ptrs || !d_c_ptrs || task_count <= 0 || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || grouped_prefix <= 0 || grouped_prefix > kRns8MaxSupportedPrefix ||
      task_count > kRns8MaxInt / grouped_prefix || safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / grouped_prefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(lda) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb) ||
      static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid(
      (n + block.x - 1) / block.x,
      (m + block.y - 1) / block.y,
      static_cast<unsigned int>(task_count * grouped_prefix));
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_i8_i32_grouped_task_prefix_kernel,
        grid,
        block,
        0,
        stream,
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        task_count,
        m,
        n,
        k,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        grouped_prefix,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

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
    int safe_k_block) {
  return rns8_hip_direct_ring_gemm_i8_on_stream(
      d_a,
      d_b,
      d_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      modulus,
      modulus_reciprocal,
      modulus_index,
      selected_prefix,
      safe_k_block,
      nullptr);
}

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
    void* stream) {
  return rns8_hip_direct_ring_gemm_i8_on_stream(
      d_a,
      d_b,
      d_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      modulus,
      modulus_reciprocal,
      modulus_index,
      selected_prefix,
      safe_k_block,
      reinterpret_cast<hipStream_t>(stream));
}

template <typename NativeT>
hipError_t launch_native_prefix9_grouped_gemm(
    const NativeT* d_a,
    const NativeT* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_ring_gemm_native_i64_i32_grouped_prefix9_kernel<NativeT>),
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

template <typename NativeT>
hipError_t launch_native_prefix9_colpair_grouped_gemm(
    const NativeT* d_a,
    const NativeT* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN / 2, kRns8HipTileM);
  const dim3 grid(
      (n + kRns8HipTileN - 1) / kRns8HipTileN,
      (m + kRns8HipTileM - 1) / kRns8HipTileM,
      kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_ring_gemm_native_i64_i32_colpair_grouped_prefix9_kernel<NativeT>),
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

template <typename NativeT, bool UniformSmallA>
hipError_t launch_native_a_resident_b_prefix9_grouped_gemm(
    const NativeT* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_ring_gemm_native_a_i8_b_i32_grouped_prefix9_kernel<NativeT, UniformSmallA>),
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

template <typename NativeT, bool UniformSmallA>
hipError_t launch_native_a_resident_b_prefix9_colpair_grouped_gemm(
    const NativeT* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN / 2, kRns8HipTileM);
  const dim3 grid(
      (n + kRns8HipTileN - 1) / kRns8HipTileN,
      (m + kRns8HipTileM - 1) / kRns8HipTileM,
      kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_ring_gemm_native_a_i8_b_i32_colpair_grouped_prefix9_kernel<NativeT, UniformSmallA>),
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

template <typename NativeT, bool UniformSmallB>
hipError_t launch_resident_a_native_b_prefix9_colpair_grouped_gemm(
    const int8_t* d_a,
    const NativeT* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN / 2, kRns8HipTileM);
  const dim3 grid(
      (n + kRns8HipTileN - 1) / kRns8HipTileN,
      (m + kRns8HipTileM - 1) / kRns8HipTileM,
      kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_ring_gemm_i8_a_native_b_i32_colpair_grouped_prefix9_kernel<NativeT, UniformSmallB>),
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

hipError_t launch_uniform_small_i8_ab_resident_b_prefix9_grouped_gemm(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_uniform_small_i8_ab_i32_grouped_prefix9_kernel,
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

hipError_t launch_uniform_small_i8_ab_resident_b_prefix9_colpair_grouped_gemm(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN / 2, kRns8HipTileM);
  const dim3 grid((n + kRns8HipTileN - 1) / kRns8HipTileN, (m + kRns8HipTileM - 1) / kRns8HipTileM, kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_uniform_small_i8_ab_i32_colpair_grouped_prefix9_kernel,
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_prefix9_grouped_gemm(d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_prefix9_colpair_grouped_gemm(d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_a_resident_b_prefix9_grouped_gemm<int64_t, false>(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_a_resident_b_prefix9_grouped_gemm<int64_t, true>(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_prefix9_grouped_gemm(d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_prefix9_colpair_grouped_gemm(d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_a_resident_b_prefix9_grouped_gemm<uint64_t, false>(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_a_resident_b_prefix9_colpair_grouped_gemm<uint64_t, false>(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(lda) ||
      static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_resident_a_native_b_prefix9_colpair_grouped_gemm<uint64_t, false>(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_a_resident_b_prefix9_grouped_gemm<uint64_t, true>(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_uniform_small_i8_ab_resident_b_prefix9_grouped_gemm(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_uniform_small_i8_ab_resident_b_prefix9_colpair_grouped_gemm(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

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
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || !d_schedule || entry_count <= 0 || max_tile_row_blocks <= 0 ||
      max_tile_col_blocks <= 0 || k <= 0 || lda < k || ldb <= 0 || ldc <= 0 || modulus <= 0 ||
      modulus_index < 0 || selected_prefix <= 0 || modulus_index >= selected_prefix ||
      selected_prefix > kRns8DefaultModulusCount ||
      modulus != rns8_default_moduli_host[modulus_index] ||
      modulus_reciprocal != rns8_modulus_reciprocal_host(modulus) ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid(max_tile_col_blocks, max_tile_row_blocks, entry_count);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_i8_i32_scheduled_kernel,
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        d_schedule,
        d_zero_a_rows,
        d_zero_b_cols,
        entry_count,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        modulus,
        modulus_reciprocal,
        modulus_index,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

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
    int safe_k_block) {
  return rns8_hip_direct_ring_gemm_i8_grouped_prefix_on_stream(
      d_a,
      d_b,
      d_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      grouped_prefix,
      safe_k_block,
      nullptr);
}

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
    void* stream) {
  return rns8_hip_direct_ring_gemm_i8_grouped_prefix_on_stream(
      d_a,
      d_b,
      d_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      grouped_prefix,
      safe_k_block,
      reinterpret_cast<hipStream_t>(stream));
}

extern "C" int rns8_hip_direct_ring_gemm_i8_grouped_task_prefix_device(
    const int8_t* const* d_a_ptrs,
    const int8_t* const* d_b_ptrs,
    int8_t* const* d_c_ptrs,
    int task_count,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block) {
  return rns8_hip_direct_ring_gemm_i8_grouped_task_prefix_on_stream(
      d_a_ptrs,
      d_b_ptrs,
      d_c_ptrs,
      task_count,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      grouped_prefix,
      safe_k_block,
      nullptr);
}

__global__ void rns8_zero_scheduled_residue_tiles_kernel(
    int8_t* C,
    const rns8_plan_tile_schedule_entry* schedule,
    int entry_count,
    int max_tile_elements,
    int rows,
    int ldc) {
  const int blocks_per_tile = (max_tile_elements + static_cast<int>(blockDim.x) - 1) / static_cast<int>(blockDim.x);
  if (blocks_per_tile <= 0) {
    return;
  }
  const int entry_index = static_cast<int>(blockIdx.x) / blocks_per_tile;
  if (entry_index >= entry_count) {
    return;
  }
  const int tile_block = static_cast<int>(blockIdx.x) - entry_index * blocks_per_tile;
  const int tile_cell = tile_block * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (tile_cell >= max_tile_elements) {
    return;
  }

  const rns8_plan_tile_schedule_entry entry = schedule[entry_index];
  if ((entry.flags & kRns8TileScheduleZeroOutput) == 0) {
    return;
  }
  const int row_extent = static_cast<int>(entry.row_extent);
  const int col_extent = static_cast<int>(entry.col_extent);
  const int tile_elements = row_extent * col_extent;
  if (tile_cell >= tile_elements) {
    return;
  }
  const int local_row = tile_cell / col_extent;
  const int local_col = tile_cell - local_row * col_extent;
  const int64_t cell_offset =
      static_cast<int64_t>(static_cast<int>(entry.row_offset) + local_row) * ldc +
      static_cast<int64_t>(static_cast<int>(entry.col_offset) + local_col);
  const int64_t plane_stride = static_cast<int64_t>(rows) * ldc;
  for (int prefix = 0; prefix < static_cast<int>(entry.selected_prefix); ++prefix) {
    C[static_cast<int64_t>(prefix) * plane_stride + cell_offset] = 0;
  }
}

extern "C" int rns8_hip_direct_zero_scheduled_residue_tiles_device(
    int8_t* d_c,
    const rns8_plan_tile_schedule_entry* d_schedule,
    int entry_count,
    int max_tile_elements,
    int rows,
    int ldc) {
  if (!d_c || !d_schedule || entry_count <= 0 || max_tile_elements <= 0 || rows <= 0 || ldc <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t block_count =
      static_cast<int64_t>(entry_count) *
      ((static_cast<int64_t>(max_tile_elements) + threads - 1) / threads);
  if (block_count <= 0 || block_count > 0xffffffffLL) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      rns8_zero_scheduled_residue_tiles_kernel,
      dim3(static_cast<unsigned int>(block_count)),
      dim3(threads),
      0,
      nullptr,
      d_c,
      d_schedule,
      entry_count,
      max_tile_elements,
      rows,
      ldc);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_u8_modulus_device(
    const int8_t* d_residues,
    uint8_t* d_dst,
    int rows,
    int cols,
    int ld,
    int modulus) {
  if (!d_residues || !d_dst || rows <= 0 || cols <= 0 || ld < cols || modulus < 2 || modulus > 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (modulus == 256) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_u8_fixed_modulus_kernel<256>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        ld);
    return static_cast<int>(hipGetLastError());
  }
  if (modulus == 255) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_u8_fixed_modulus_kernel<255>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        ld);
    return static_cast<int>(hipGetLastError());
  }
  if (modulus == 251) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_u8_fixed_modulus_kernel<251>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        ld);
    return static_cast<int>(hipGetLastError());
  }
  hipLaunchKernelGGL(
      rns8_export_u8_modulus_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      nullptr,
      d_residues,
      d_dst,
      rows,
      cols,
      ld,
      modulus);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_i64_device(
    const int8_t* d_residues,
    int64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status) {
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (prefix == kRns8DefaultBoundedPrefix) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_i64_fixed_prefix_kernel<kRns8DefaultBoundedPrefix>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        bound,
        d_status);
  } else if (prefix == kRns8MaxSupportedPrefix) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_i64_fixed_prefix_kernel<kRns8MaxSupportedPrefix>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        bound,
        d_status);
  } else {
    hipLaunchKernelGGL(
        rns8_export_i64_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        prefix,
        bound,
        d_status);
  }
  return static_cast<int>(hipGetLastError());
}

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
    int* d_status) {
  if (!d_residues || !d_dst || !d_schedule || !d_bounds || !d_status || entry_count <= 0 ||
      max_tile_elements <= 0 || rows <= 0 || cols <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = (max_tile_elements + threads - 1) / threads;
  if (entry_count > kRns8MaxInt / blocks) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int total_blocks = entry_count * blocks;
  hipLaunchKernelGGL(
      rns8_export_i64_scheduled_kernel,
      dim3(total_blocks),
      dim3(threads),
      0,
      nullptr,
      d_residues,
      d_dst,
      d_schedule,
      d_bounds,
      d_zero_a_rows,
      d_zero_b_cols,
      entry_count,
      max_tile_elements,
      rows,
      cols,
      d_status);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_u64_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* d_status) {
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (prefix == kRns8DefaultBoundedPrefix) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_u64_fixed_prefix_kernel<kRns8DefaultBoundedPrefix>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        bound,
        d_status);
  } else if (prefix == kRns8MaxSupportedPrefix) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_export_u64_fixed_prefix_kernel<kRns8MaxSupportedPrefix>),
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        bound,
        d_status);
  } else {
    hipLaunchKernelGGL(
        rns8_export_u64_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        prefix,
        bound,
        d_status);
  }
  return static_cast<int>(hipGetLastError());
}

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
    int* d_status) {
  if (!d_residues || !d_dst || !d_schedule || !d_bounds || !d_status || entry_count <= 0 ||
      max_tile_elements <= 0 || rows <= 0 || cols <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = (max_tile_elements + threads - 1) / threads;
  if (entry_count > kRns8MaxInt / blocks) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int total_blocks = entry_count * blocks;
  hipLaunchKernelGGL(
      rns8_export_u64_scheduled_kernel,
      dim3(total_blocks),
      dim3(threads),
      0,
      nullptr,
      d_residues,
      d_dst,
      d_schedule,
      d_bounds,
      d_zero_a_rows,
      d_zero_b_cols,
      entry_count,
      max_tile_elements,
      rows,
      cols,
      d_status);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_exact_wide_unsigned_limbs_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* d_status) {
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (prefix == kRns8MaxSupportedPrefix) {
    switch (limb_count) {
      case 1:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 1>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 2:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 2>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 3:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 3>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 4:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 4>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 8:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 8>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 16:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 16>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 32:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 32>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      default:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_limbs_fixed_prefix_kernel<kRns8MaxSupportedPrefix>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            limb_count,
            d_status);
        break;
    }
  } else {
    hipLaunchKernelGGL(
        rns8_export_exact_wide_unsigned_limbs_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        prefix,
        limb_count,
        d_status);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_exact_wide_unsigned_grouped_limbs_device(
    const int8_t* const* d_residue_ptrs,
    uint64_t* d_dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    int limb_count) {
  if (!d_residue_ptrs || !d_dst || task_count <= 0 || rows <= 0 || cols <= 0 ||
      prefix <= 0 || prefix > kRns8MaxSupportedPrefix || limb_count <= 0 || limb_count > 32) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (prefix == kRns8MaxSupportedPrefix) {
    switch (limb_count) {
#define RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(LimbCount) \
      case LimbCount:                                       \
        hipLaunchKernelGGL(                                 \
            HIP_KERNEL_NAME(rns8_export_exact_wide_unsigned_grouped_fixed_prefix_fixed_limbs_kernel< \
                            kRns8MaxSupportedPrefix, LimbCount>), \
            dim3(blocks, task_count),                       \
            dim3(threads),                                  \
            0,                                              \
            nullptr,                                        \
            d_residue_ptrs,                                 \
            d_dst,                                          \
            task_count,                                     \
            rows,                                           \
            cols);                                          \
        break
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(1);
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(2);
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(3);
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(4);
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(8);
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(16);
      RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS(32);
#undef RNS8_LAUNCH_GROUPED_UNSIGNED_FIXED_LIMBS
      default:
        hipLaunchKernelGGL(
            rns8_export_exact_wide_unsigned_grouped_limbs_kernel,
            dim3(blocks, task_count),
            dim3(threads),
            0,
            nullptr,
            d_residue_ptrs,
            d_dst,
            task_count,
            rows,
            cols,
            prefix,
            limb_count);
        break;
    }
  } else {
    hipLaunchKernelGGL(
        rns8_export_exact_wide_unsigned_grouped_limbs_kernel,
        dim3(blocks, task_count),
        dim3(threads),
        0,
        nullptr,
        d_residue_ptrs,
        d_dst,
        task_count,
        rows,
        cols,
        prefix,
        limb_count);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_exact_wide_signed_limbs_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* d_status) {
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (prefix == kRns8MaxSupportedPrefix) {
    switch (limb_count) {
      case 1:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 1>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 2:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 2>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 3:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 3>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 4:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 4>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 8:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 8>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 16:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 16>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      case 32:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_signed_fixed_prefix_fixed_limbs_kernel<kRns8MaxSupportedPrefix, 32>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            d_status);
        break;
      default:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rns8_export_exact_wide_signed_limbs_fixed_prefix_kernel<kRns8MaxSupportedPrefix>),
            dim3(blocks),
            dim3(threads),
            0,
            nullptr,
            d_residues,
            d_dst,
            rows,
            cols,
            limb_count,
            d_status);
        break;
    }
  } else {
    hipLaunchKernelGGL(
        rns8_export_exact_wide_signed_limbs_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        nullptr,
        d_residues,
        d_dst,
        rows,
        cols,
        prefix,
        limb_count,
        d_status);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_export_exact_wide_signed_grouped_limbs_device(
    const int8_t* const* d_residue_ptrs,
    uint64_t* d_dst,
    int task_count,
    int rows,
    int cols,
    int prefix,
    int limb_count) {
  if (!d_residue_ptrs || !d_dst || task_count <= 0 || rows <= 0 || cols <= 0 ||
      prefix <= 0 || prefix > kRns8MaxSupportedPrefix || limb_count <= 0 || limb_count > 32) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int elements = rows * cols;
  const int blocks = (elements + threads - 1) / threads;
  if (prefix == kRns8MaxSupportedPrefix) {
    switch (limb_count) {
#define RNS8_LAUNCH_GROUPED_SIGNED_FIXED_LIMBS(LimbCount) \
      case LimbCount:                                     \
        hipLaunchKernelGGL(                               \
            HIP_KERNEL_NAME(rns8_export_exact_wide_signed_grouped_fixed_prefix_fixed_limbs_kernel< \
                            kRns8MaxSupportedPrefix, LimbCount>), \
            dim3(blocks, task_count),                     \
            dim3(threads),                                \
            0,                                            \
            nullptr,                                      \
            d_residue_ptrs,                               \
            d_dst,                                        \
            task_count,                                   \
            rows,                                         \
            cols);                                        \
        break
      RNS8_LAUNCH_GROUPED_SIGNED_FIXED_LIMBS(1);
      RNS8_LAUNCH_GROUPED_SIGNED_FIXED_LIMBS(2);
      RNS8_LAUNCH_GROUPED_SIGNED_FIXED_LIMBS(3);
      RNS8_LAUNCH_GROUPED_SIGNED_FIXED_LIMBS(4);
      RNS8_LAUNCH_GROUPED_SIGNED_FIXED_LIMBS(8);
      RNS8_LAUNCH_GROUPED_SIGNED_FIXED_LIMBS(16);
      RNS8_LAUNCH_GROUPED_SIGNED_FIXED_LIMBS(32);
#undef RNS8_LAUNCH_GROUPED_SIGNED_FIXED_LIMBS
      default:
        hipLaunchKernelGGL(
            rns8_export_exact_wide_signed_grouped_limbs_kernel,
            dim3(blocks, task_count),
            dim3(threads),
            0,
            nullptr,
            d_residue_ptrs,
            d_dst,
            task_count,
            rows,
            cols,
            prefix,
            limb_count);
        break;
    }
  } else {
    hipLaunchKernelGGL(
        rns8_export_exact_wide_signed_grouped_limbs_kernel,
        dim3(blocks, task_count),
        dim3(threads),
        0,
        nullptr,
        d_residue_ptrs,
        d_dst,
        task_count,
        rows,
        cols,
        prefix,
        limb_count);
  }
  return static_cast<int>(hipGetLastError());
}
