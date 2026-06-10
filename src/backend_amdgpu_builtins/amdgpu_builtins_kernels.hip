#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

#include "../backend_common/finite_u8_reducer.hpp"

namespace {

using i32x2 = int __attribute__((ext_vector_type(2)));
using i32x4 = int __attribute__((ext_vector_type(4)));
using i32x8 = int __attribute__((ext_vector_type(8)));
using i32x16 = int __attribute__((ext_vector_type(16)));

constexpr int kRns8DefaultModulusCount = 28;
constexpr int kRns8DenseTile = 16;
constexpr int kRns8DenseCdnaK = 32;
constexpr int kRns8DenseCdnaWideTile = 32;
constexpr int kRns8DenseCdnaWideK = 16;
constexpr int kRns8DenseRdnaK = 16;
constexpr int kRns8SparseCdnaK = 64;
constexpr int kRns8SparseCdnaWideK = 32;
constexpr int kRns8SparseRdna4K = 32;

__device__ __constant__ int rns8_amdgpu_default_moduli[kRns8DefaultModulusCount] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

enum class rns8_amdgpu_arch_kind {
  unsupported,
  cdna3,
  rdna3,
  rdna4,
};

__device__ __forceinline__ uint32_t pack_byte(uint8_t value, int byte_index) {
  return static_cast<uint32_t>(value) << static_cast<uint32_t>(byte_index * 8);
}

__device__ __forceinline__ int8_t load_centered_i8(
    const int8_t* data,
    int plane,
    int rows,
    int cols,
    int ld,
    int row,
    int col) {
  if (row < 0 || col < 0 || row >= rows || col >= cols) {
    return 0;
  }
  const int plane_stride = rows * ld;
  return data[plane * plane_stride + row * ld + col];
}

__device__ __forceinline__ uint32_t pack4_centered_i8(
    const int8_t* data,
    int plane,
    int rows,
    int cols,
    int ld,
    int row,
    int col_base) {
  uint32_t packed = 0;
  packed |= pack_byte(static_cast<uint8_t>(load_centered_i8(data, plane, rows, cols, ld, row, col_base + 0)), 0);
  packed |= pack_byte(static_cast<uint8_t>(load_centered_i8(data, plane, rows, cols, ld, row, col_base + 1)), 1);
  packed |= pack_byte(static_cast<uint8_t>(load_centered_i8(data, plane, rows, cols, ld, row, col_base + 2)), 2);
  packed |= pack_byte(static_cast<uint8_t>(load_centered_i8(data, plane, rows, cols, ld, row, col_base + 3)), 3);
  return packed;
}

__device__ __forceinline__ uint32_t pack4_centered_i8_column(
    const int8_t* data,
    int plane,
    int rows,
    int cols,
    int ld,
    int row_base,
    int col) {
  uint32_t packed = 0;
  packed |= pack_byte(static_cast<uint8_t>(load_centered_i8(data, plane, rows, cols, ld, row_base + 0, col)), 0);
  packed |= pack_byte(static_cast<uint8_t>(load_centered_i8(data, plane, rows, cols, ld, row_base + 1, col)), 1);
  packed |= pack_byte(static_cast<uint8_t>(load_centered_i8(data, plane, rows, cols, ld, row_base + 2, col)), 2);
  packed |= pack_byte(static_cast<uint8_t>(load_centered_i8(data, plane, rows, cols, ld, row_base + 3, col)), 3);
  return packed;
}

__device__ __forceinline__ int8_t center_canonical_u8(uint8_t value, int modulus) {
  if (modulus == 256) {
    return static_cast<int8_t>(value);
  }
  const uint32_t residue = static_cast<uint32_t>(value % static_cast<uint8_t>(modulus));
  return rns8::detail::finite_u8::center_unsigned_residue(residue, static_cast<uint32_t>(modulus));
}

__device__ __forceinline__ uint32_t sparse_pair_for_group(
    const uint8_t* packed_values,
    int plane,
    int rows,
    int groups_per_row,
    int row,
    int group,
    bool finite,
    int modulus) {
  if (row < 0 || row >= rows || group < 0 || group >= groups_per_row) {
    return 0;
  }
  const int64_t group_offset =
      (static_cast<int64_t>(plane) * rows + row) * static_cast<int64_t>(groups_per_row) + group;
  const uint8_t raw0 = packed_values[group_offset * 2 + 0];
  const uint8_t raw1 = packed_values[group_offset * 2 + 1];
  const uint8_t v0 = static_cast<uint8_t>(finite ? center_canonical_u8(raw0, modulus) : static_cast<int8_t>(raw0));
  const uint8_t v1 = static_cast<uint8_t>(finite ? center_canonical_u8(raw1, modulus) : static_cast<int8_t>(raw1));
  return pack_byte(v0, 0) | pack_byte(v1, 1);
}

__device__ __forceinline__ uint32_t sparse_index_for_group(
    const uint8_t* packed_indices,
    int plane,
    int rows,
    int groups_per_row,
    int row,
    int group) {
  if (row < 0 || row >= rows || group < 0 || group >= groups_per_row) {
    return 0;
  }
  const int64_t group_offset =
      (static_cast<int64_t>(plane) * rows + row) * static_cast<int64_t>(groups_per_row) + group;
  return static_cast<uint32_t>(packed_indices[group_offset] & 0x0f);
}

__device__ __forceinline__ int8_t reduce_centered(int32_t value, int modulus) {
  return rns8::detail::finite_u8::reduce_to_centered_static_byte_modulus_i32(value, static_cast<uint32_t>(modulus));
}

__device__ __forceinline__ void store_rns_accumulator(
    int32_t acc,
    int8_t* c,
    int plane,
    int m,
    int n,
    int ldc,
    int row,
    int col) {
  if (row < m && col < n) {
    const int modulus = rns8_amdgpu_default_moduli[plane];
    c[plane * m * ldc + row * ldc + col] = reduce_centered(acc, modulus);
  }
}

__device__ __forceinline__ void store_finite_accumulator(
    int32_t acc,
    int8_t* c,
    int modulus,
    int m,
    int n,
    int ldc,
    int row,
    int col) {
  if (row < m && col < n) {
    c[row * ldc + col] = reduce_centered(acc, modulus);
  }
}

template <bool Finite>
__device__ __forceinline__ void store_accumulator(
    int32_t acc,
    int8_t* c,
    int plane,
    int modulus,
    int m,
    int n,
    int ldc,
    int row,
    int col) {
  if constexpr (Finite) {
    store_finite_accumulator(acc, c, modulus, m, n, ldc, row, col);
  } else {
    store_rns_accumulator(acc, c, plane, m, n, ldc, row, col);
  }
}

template <bool Finite>
__global__ __launch_bounds__(64) void cdna3_dense_mfma_kernel(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int plane_count_or_modulus) {
#if defined(__gfx942__)
  const int lane = static_cast<int>(threadIdx.x);
  const int lane_row = lane & 15;
  const int lane_col = lane & 15;
  const int k_group = lane >> 4;
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8DenseTile;
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8DenseTile;
  const int plane = Finite ? 0 : static_cast<int>(blockIdx.z);
  const int modulus = Finite ? plane_count_or_modulus : 0;

  i32x4 acc = {0, 0, 0, 0};
  for (int kk = 0; kk < k; kk += kRns8DenseCdnaK) {
    const int a_row = tile_row + lane_row;
    const int b_col = tile_col + lane_col;
    const int k_base = kk + k_group * 8;
    const uint64_t packed_a =
        static_cast<uint64_t>(pack4_centered_i8(a, plane, m, k, lda, a_row, k_base)) |
        (static_cast<uint64_t>(pack4_centered_i8(a, plane, m, k, lda, a_row, k_base + 4)) << 32u);
    const uint64_t packed_b =
        static_cast<uint64_t>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 0, b_col)) |
        (static_cast<uint64_t>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 4, b_col)) << 32u);
    acc = __builtin_amdgcn_mfma_i32_16x16x32_i8(
        static_cast<long long>(packed_a),
        static_cast<long long>(packed_b),
        acc,
        0,
        0,
        0);
  }
  const int row_base = tile_row + (lane >> 4) * 4;
  const int col = tile_col + lane_col;
  store_accumulator<Finite>(acc[0], c, plane, modulus, m, n, ldc, row_base + 0, col);
  store_accumulator<Finite>(acc[1], c, plane, modulus, m, n, ldc, row_base + 1, col);
  store_accumulator<Finite>(acc[2], c, plane, modulus, m, n, ldc, row_base + 2, col);
  store_accumulator<Finite>(acc[3], c, plane, modulus, m, n, ldc, row_base + 3, col);
#else
  (void)a;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)plane_count_or_modulus;
#endif
}

template <bool Finite>
__global__ __launch_bounds__(64) void cdna3_dense_mfma_32x32x16_kernel(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int plane_count_or_modulus) {
#if defined(__gfx942__)
  const int lane = static_cast<int>(threadIdx.x);
  const int lane_row = lane & 31;
  const int lane_col = lane & 31;
  const int lane_half = lane >> 5;
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8DenseCdnaWideTile;
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8DenseCdnaWideTile;
  const int plane = Finite ? 0 : static_cast<int>(blockIdx.z);
  const int modulus = Finite ? plane_count_or_modulus : 0;

  i32x16 acc = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int kk = 0; kk < k; kk += kRns8DenseCdnaWideK) {
    const int a_row = tile_row + lane_row;
    const int b_col = tile_col + lane_col;
    const int k_base = kk + lane_half * 8;
    const uint64_t packed_a =
        static_cast<uint64_t>(pack4_centered_i8(a, plane, m, k, lda, a_row, k_base)) |
        (static_cast<uint64_t>(pack4_centered_i8(a, plane, m, k, lda, a_row, k_base + 4)) << 32u);
    const uint64_t packed_b =
        static_cast<uint64_t>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 0, b_col)) |
        (static_cast<uint64_t>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 4, b_col)) << 32u);
    acc = __builtin_amdgcn_mfma_i32_32x32x16_i8(
        static_cast<long long>(packed_a),
        static_cast<long long>(packed_b),
        acc,
        0,
        0,
        0);
  }
  const int col = tile_col + lane_col;
  for (int reg = 0; reg < 16; ++reg) {
    const int row = tile_row + 8 * (reg >> 2) + 4 * lane_half + (reg & 3);
    store_accumulator<Finite>(acc[reg], c, plane, modulus, m, n, ldc, row, col);
  }
#else
  (void)a;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)plane_count_or_modulus;
#endif
}

template <bool Finite>
__global__ __launch_bounds__(32) void rdna3_dense_wmma_kernel(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int plane_count_or_modulus) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
  const int lane = static_cast<int>(threadIdx.x);
  const int lane_row = lane & 15;
  const int lane_col = lane & 15;
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8DenseTile;
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8DenseTile;
  const int plane = Finite ? 0 : static_cast<int>(blockIdx.z);
  const int modulus = Finite ? plane_count_or_modulus : 0;

  i32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int kk = 0; kk < k; kk += kRns8DenseRdnaK) {
    i32x4 av;
    i32x4 bv;
    const int a_row = tile_row + lane_row;
    const int b_col = tile_col + lane_col;
    av[0] = static_cast<int>(pack4_centered_i8(a, plane, m, k, lda, a_row, kk + 0));
    av[1] = static_cast<int>(pack4_centered_i8(a, plane, m, k, lda, a_row, kk + 4));
    av[2] = static_cast<int>(pack4_centered_i8(a, plane, m, k, lda, a_row, kk + 8));
    av[3] = static_cast<int>(pack4_centered_i8(a, plane, m, k, lda, a_row, kk + 12));
    bv[0] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, kk + 0, b_col));
    bv[1] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, kk + 4, b_col));
    bv[2] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, kk + 8, b_col));
    bv[3] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, kk + 12, b_col));
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, av, true, bv, acc, false);
  }
  const int half = lane >> 4;
  const int col = tile_col + lane_col;
  for (int reg = 0; reg < 8; ++reg) {
    const int row = tile_row + reg * 2 + half;
    store_accumulator<Finite>(acc[reg], c, plane, modulus, m, n, ldc, row, col);
  }
#else
  (void)a;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)plane_count_or_modulus;
#endif
}

template <bool Finite>
__global__ __launch_bounds__(32) void rdna4_dense_wmma_kernel(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int plane_count_or_modulus) {
#if defined(__gfx1200__) || defined(__gfx1201__)
  const int lane = static_cast<int>(threadIdx.x);
  const int half = lane >> 4;
  const int lane_row = lane & 15;
  const int lane_col = lane & 15;
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8DenseTile;
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8DenseTile;
  const int plane = Finite ? 0 : static_cast<int>(blockIdx.z);
  const int modulus = Finite ? plane_count_or_modulus : 0;

  i32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int kk = 0; kk < k; kk += kRns8DenseRdnaK) {
    i32x2 av;
    i32x2 bv;
    const int a_row = tile_row + lane_row;
    const int b_col = tile_col + lane_col;
    const int k_base = kk + half * 8;
    av[0] = static_cast<int>(pack4_centered_i8(a, plane, m, k, lda, a_row, k_base + 0));
    av[1] = static_cast<int>(pack4_centered_i8(a, plane, m, k, lda, a_row, k_base + 4));
    bv[0] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 0, b_col));
    bv[1] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 4, b_col));
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av, true, bv, acc, false);
  }
  const int row_base = tile_row + half * 8;
  const int col = tile_col + lane_col;
  for (int reg = 0; reg < 8; ++reg) {
    store_accumulator<Finite>(acc[reg], c, plane, modulus, m, n, ldc, row_base + reg, col);
  }
#else
  (void)a;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)plane_count_or_modulus;
#endif
}

template <bool Finite>
__global__ __launch_bounds__(64) void cdna3_sparse_smfmac_kernel(
    const uint8_t* a_values,
    const uint8_t* a_indices,
    const int8_t* b,
    int8_t* c,
    int m,
    int n,
    int k,
    int ldb,
    int ldc,
    int plane_count_or_modulus) {
#if defined(__gfx942__)
  const int lane = static_cast<int>(threadIdx.x);
  const int lane_row = lane & 15;
  const int lane_col = lane & 15;
  const int lane_group = lane >> 4;
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8DenseTile;
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8DenseTile;
  const int plane = Finite ? 0 : static_cast<int>(blockIdx.z);
  const int modulus = Finite ? plane_count_or_modulus : 0;
  const int groups_per_row = k / 4;

  i32x4 acc = {0, 0, 0, 0};
  for (int kk = 0; kk < k; kk += kRns8SparseCdnaK) {
    i32x2 av;
    i32x4 bv;
    const int a_row = tile_row + lane_row;
    const int b_col = tile_col + lane_col;
    const int group_base = kk / 4 + lane_group * 4;
    const uint32_t pair0 = sparse_pair_for_group(a_values, plane, m, groups_per_row, a_row, group_base + 0, Finite, modulus);
    const uint32_t pair1 = sparse_pair_for_group(a_values, plane, m, groups_per_row, a_row, group_base + 1, Finite, modulus);
    const uint32_t pair2 = sparse_pair_for_group(a_values, plane, m, groups_per_row, a_row, group_base + 2, Finite, modulus);
    const uint32_t pair3 = sparse_pair_for_group(a_values, plane, m, groups_per_row, a_row, group_base + 3, Finite, modulus);
    av[0] = static_cast<int>(pair0 | (pair1 << 16u));
    av[1] = static_cast<int>(pair2 | (pair3 << 16u));

    uint32_t sparse_idx = 0;
    sparse_idx |= sparse_index_for_group(a_indices, plane, m, groups_per_row, a_row, group_base + 0) << 0u;
    sparse_idx |= sparse_index_for_group(a_indices, plane, m, groups_per_row, a_row, group_base + 1) << 4u;
    sparse_idx |= sparse_index_for_group(a_indices, plane, m, groups_per_row, a_row, group_base + 2) << 8u;
    sparse_idx |= sparse_index_for_group(a_indices, plane, m, groups_per_row, a_row, group_base + 3) << 12u;

    const int k_base = kk + lane_group * 16;
    bv[0] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 0, b_col));
    bv[1] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 4, b_col));
    bv[2] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 8, b_col));
    bv[3] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 12, b_col));
    acc = __builtin_amdgcn_smfmac_i32_16x16x64_i8(av, bv, acc, static_cast<int>(sparse_idx), 0, 0);
  }
  const int row_base = tile_row + (lane >> 4) * 4;
  const int col = tile_col + lane_col;
  store_accumulator<Finite>(acc[0], c, plane, modulus, m, n, ldc, row_base + 0, col);
  store_accumulator<Finite>(acc[1], c, plane, modulus, m, n, ldc, row_base + 1, col);
  store_accumulator<Finite>(acc[2], c, plane, modulus, m, n, ldc, row_base + 2, col);
  store_accumulator<Finite>(acc[3], c, plane, modulus, m, n, ldc, row_base + 3, col);
#else
  (void)a_values;
  (void)a_indices;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)ldb;
  (void)ldc;
  (void)plane_count_or_modulus;
#endif
}

template <bool Finite>
__global__ __launch_bounds__(64) void cdna3_sparse_smfmac_32x32x32_kernel(
    const uint8_t* a_values,
    const uint8_t* a_indices,
    const int8_t* b,
    int8_t* c,
    int m,
    int n,
    int k,
    int ldb,
    int ldc,
    int plane_count_or_modulus) {
#if defined(__gfx942__)
  const int lane = static_cast<int>(threadIdx.x);
  const int lane_row = lane & 31;
  const int lane_col = lane & 31;
  const int lane_half = lane >> 5;
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8DenseCdnaWideTile;
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8DenseCdnaWideTile;
  const int plane = Finite ? 0 : static_cast<int>(blockIdx.z);
  const int modulus = Finite ? plane_count_or_modulus : 0;
  const int groups_per_row = k / 4;

  i32x16 acc = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int kk = 0; kk < k; kk += kRns8SparseCdnaWideK) {
    i32x2 av;
    i32x4 bv;
    const int a_row = tile_row + lane_row;
    const int b_col = tile_col + lane_col;
    const int group_base = kk / 4 + lane_half * 4;
    const uint32_t pair0 = sparse_pair_for_group(a_values, plane, m, groups_per_row, a_row, group_base + 0, Finite, modulus);
    const uint32_t pair1 = sparse_pair_for_group(a_values, plane, m, groups_per_row, a_row, group_base + 1, Finite, modulus);
    const uint32_t pair2 = sparse_pair_for_group(a_values, plane, m, groups_per_row, a_row, group_base + 2, Finite, modulus);
    const uint32_t pair3 = sparse_pair_for_group(a_values, plane, m, groups_per_row, a_row, group_base + 3, Finite, modulus);
    av[0] = static_cast<int>(pair0 | (pair1 << 16u));
    av[1] = static_cast<int>(pair2 | (pair3 << 16u));

    uint32_t sparse_idx = 0;
    sparse_idx |= sparse_index_for_group(a_indices, plane, m, groups_per_row, a_row, group_base + 0) << 0u;
    sparse_idx |= sparse_index_for_group(a_indices, plane, m, groups_per_row, a_row, group_base + 1) << 4u;
    sparse_idx |= sparse_index_for_group(a_indices, plane, m, groups_per_row, a_row, group_base + 2) << 8u;
    sparse_idx |= sparse_index_for_group(a_indices, plane, m, groups_per_row, a_row, group_base + 3) << 12u;

    const int k_base = kk + lane_half * 16;
    bv[0] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 0, b_col));
    bv[1] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 4, b_col));
    bv[2] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 8, b_col));
    bv[3] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 12, b_col));
    acc = __builtin_amdgcn_smfmac_i32_32x32x32_i8(av, bv, acc, static_cast<int>(sparse_idx), 0, 0);
  }
  const int col = tile_col + lane_col;
  for (int reg = 0; reg < 16; ++reg) {
    const int row = tile_row + 8 * (reg >> 2) + 4 * lane_half + (reg & 3);
    store_accumulator<Finite>(acc[reg], c, plane, modulus, m, n, ldc, row, col);
  }
#else
  (void)a_values;
  (void)a_indices;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)ldb;
  (void)ldc;
  (void)plane_count_or_modulus;
#endif
}

template <bool Finite>
__global__ __launch_bounds__(32) void rdna4_sparse_swmmac_kernel(
    const uint8_t* a_values,
    const uint8_t* a_indices,
    const int8_t* b,
    int8_t* c,
    int m,
    int n,
    int k,
    int ldb,
    int ldc,
    int plane_count_or_modulus) {
#if defined(__gfx1200__) || defined(__gfx1201__)
  const int lane = static_cast<int>(threadIdx.x);
  const int half = lane >> 4;
  const int lane_row = lane & 15;
  const int lane_col = lane & 15;
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8DenseTile;
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8DenseTile;
  const int plane = Finite ? 0 : static_cast<int>(blockIdx.z);
  const int modulus = Finite ? plane_count_or_modulus : 0;
  const int groups_per_row = k / 4;

  i32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int kk = 0; kk < k; kk += kRns8SparseRdna4K) {
    i32x2 av;
    i32x4 bv;
    const int a_row = tile_row + lane_row;
    const int b_col = tile_col + lane_col;
    const int group_base = kk / 4 + half * 4;
    const uint32_t pair0 = sparse_pair_for_group(a_values, plane, m, groups_per_row, a_row, group_base + 0, Finite, modulus);
    const uint32_t pair1 = sparse_pair_for_group(a_values, plane, m, groups_per_row, a_row, group_base + 1, Finite, modulus);
    const uint32_t pair2 = sparse_pair_for_group(a_values, plane, m, groups_per_row, a_row, group_base + 2, Finite, modulus);
    const uint32_t pair3 = sparse_pair_for_group(a_values, plane, m, groups_per_row, a_row, group_base + 3, Finite, modulus);
    av[0] = static_cast<int>(pair0 | (pair1 << 16u));
    av[1] = static_cast<int>(pair2 | (pair3 << 16u));

    uint32_t sparse_idx = 0;
    sparse_idx |= sparse_index_for_group(a_indices, plane, m, groups_per_row, a_row, group_base + 0) << 0u;
    sparse_idx |= sparse_index_for_group(a_indices, plane, m, groups_per_row, a_row, group_base + 1) << 4u;
    sparse_idx |= sparse_index_for_group(a_indices, plane, m, groups_per_row, a_row, group_base + 2) << 8u;
    sparse_idx |= sparse_index_for_group(a_indices, plane, m, groups_per_row, a_row, group_base + 3) << 12u;

    const int k_base = kk + half * 16;
    bv[0] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 0, b_col));
    bv[1] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 4, b_col));
    bv[2] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 8, b_col));
    bv[3] = static_cast<int>(pack4_centered_i8_column(b, plane, k, n, ldb, k_base + 12, b_col));
    acc = __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, av, true, bv, acc, static_cast<short>(sparse_idx), false);
  }
  const int row_base = tile_row + half * 8;
  const int col = tile_col + lane_col;
  for (int reg = 0; reg < 8; ++reg) {
    store_accumulator<Finite>(acc[reg], c, plane, modulus, m, n, ldc, row_base + reg, col);
  }
#else
  (void)a_values;
  (void)a_indices;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)ldb;
  (void)ldc;
  (void)plane_count_or_modulus;
#endif
}

rns8_amdgpu_arch_kind detect_arch_kind(const char* arch) {
  if (!arch) {
    return rns8_amdgpu_arch_kind::unsupported;
  }
  if (std::strncmp(arch, "gfx942", 6) == 0) {
    return rns8_amdgpu_arch_kind::cdna3;
  }
  if (std::strncmp(arch, "gfx110", 6) == 0) {
    return rns8_amdgpu_arch_kind::rdna3;
  }
  if (std::strncmp(arch, "gfx1200", 7) == 0 || std::strncmp(arch, "gfx1201", 7) == 0) {
    return rns8_amdgpu_arch_kind::rdna4;
  }
  return rns8_amdgpu_arch_kind::unsupported;
}

int active_device_arch_kind(int device_id, rns8_amdgpu_arch_kind& kind) {
  kind = rns8_amdgpu_arch_kind::unsupported;
  hipError_t err = hipSetDevice(device_id);
  if (err != hipSuccess) {
    return 3;
  }
  hipDeviceProp_t prop{};
  err = hipGetDeviceProperties(&prop, device_id);
  if (err != hipSuccess) {
    return 3;
  }
  kind = detect_arch_kind(prop.gcnArchName);
  return kind == rns8_amdgpu_arch_kind::unsupported ? 2 : 0;
}

bool use_cdna3_mfma_32x32x16(int m, int n, int k) {
  return m >= 128 && n >= 128 && k >= 128;
}

bool use_cdna3_mfma_32x32x16(int m, int n, int k, int dense_kernel_variant) {
  if (dense_kernel_variant == 1) {
    return false;
  }
  if (dense_kernel_variant == 2) {
    return true;
  }
  return use_cdna3_mfma_32x32x16(m, n, k);
}

bool use_cdna3_smfmac_32x32x32(int m, int n, int k) {
  return m >= 128 && n >= 128 && k >= 128;
}

template <bool Finite>
int launch_dense(
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
    int plane_count_or_modulus,
    int dense_kernel_variant) {
  rns8_amdgpu_arch_kind kind{};
  const int arch_status = active_device_arch_kind(device_id, kind);
  if (arch_status != 0) {
    return arch_status;
  }
  hipError_t err = hipSuccess;
  switch (kind) {
    case rns8_amdgpu_arch_kind::cdna3:
      if (use_cdna3_mfma_32x32x16(m, n, k, dense_kernel_variant)) {
        const dim3 grid(
            (static_cast<unsigned>(n) + 31u) / 32u,
            (static_cast<unsigned>(m) + 31u) / 32u,
            Finite ? 1u : static_cast<unsigned>(plane_count_or_modulus));
        hipLaunchKernelGGL(
            (cdna3_dense_mfma_32x32x16_kernel<Finite>),
            grid,
            dim3(64),
            0,
            0,
            static_cast<const int8_t*>(device_a_residues),
            static_cast<const int8_t*>(device_b_residues),
            static_cast<int8_t*>(device_c_residues),
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            plane_count_or_modulus);
      } else {
        const dim3 grid(
            (static_cast<unsigned>(n) + 15u) / 16u,
            (static_cast<unsigned>(m) + 15u) / 16u,
            Finite ? 1u : static_cast<unsigned>(plane_count_or_modulus));
        hipLaunchKernelGGL(
            (cdna3_dense_mfma_kernel<Finite>),
            grid,
            dim3(64),
            0,
            0,
            static_cast<const int8_t*>(device_a_residues),
            static_cast<const int8_t*>(device_b_residues),
            static_cast<int8_t*>(device_c_residues),
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            plane_count_or_modulus);
      }
      break;
    case rns8_amdgpu_arch_kind::rdna3:
    {
      const dim3 grid((static_cast<unsigned>(n) + 15u) / 16u, (static_cast<unsigned>(m) + 15u) / 16u, Finite ? 1u : static_cast<unsigned>(plane_count_or_modulus));
      hipLaunchKernelGGL(
          (rdna3_dense_wmma_kernel<Finite>),
          grid,
          dim3(32),
          0,
          0,
          static_cast<const int8_t*>(device_a_residues),
          static_cast<const int8_t*>(device_b_residues),
          static_cast<int8_t*>(device_c_residues),
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          plane_count_or_modulus);
      break;
    }
    case rns8_amdgpu_arch_kind::rdna4:
    {
      const dim3 grid((static_cast<unsigned>(n) + 15u) / 16u, (static_cast<unsigned>(m) + 15u) / 16u, Finite ? 1u : static_cast<unsigned>(plane_count_or_modulus));
      hipLaunchKernelGGL(
          (rdna4_dense_wmma_kernel<Finite>),
          grid,
          dim3(32),
          0,
          0,
          static_cast<const int8_t*>(device_a_residues),
          static_cast<const int8_t*>(device_b_residues),
          static_cast<int8_t*>(device_c_residues),
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          plane_count_or_modulus);
      break;
    }
    case rns8_amdgpu_arch_kind::unsupported:
      return 2;
  }
  err = hipGetLastError();
  if (err != hipSuccess) {
    return 3;
  }
  err = hipDeviceSynchronize();
  return err == hipSuccess ? 0 : 3;
}

template <bool Finite>
int launch_sparse_a(
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
    int plane_count_or_modulus) {
  rns8_amdgpu_arch_kind kind{};
  const int arch_status = active_device_arch_kind(device_id, kind);
  if (arch_status != 0) {
    return arch_status;
  }
  hipError_t err = hipSuccess;
  switch (kind) {
    case rns8_amdgpu_arch_kind::cdna3:
      if (use_cdna3_smfmac_32x32x32(m, n, k)) {
        const dim3 grid(
            (static_cast<unsigned>(n) + 31u) / 32u,
            (static_cast<unsigned>(m) + 31u) / 32u,
            Finite ? 1u : static_cast<unsigned>(plane_count_or_modulus));
        hipLaunchKernelGGL(
            (cdna3_sparse_smfmac_32x32x32_kernel<Finite>),
            grid,
            dim3(64),
            0,
            0,
            static_cast<const uint8_t*>(device_a_packed_values),
            static_cast<const uint8_t*>(device_a_packed_indices),
            static_cast<const int8_t*>(device_b_residues),
            static_cast<int8_t*>(device_c_residues),
            m,
            n,
            k,
            ldb,
            ldc,
            plane_count_or_modulus);
      } else {
        const dim3 grid(
            (static_cast<unsigned>(n) + 15u) / 16u,
            (static_cast<unsigned>(m) + 15u) / 16u,
            Finite ? 1u : static_cast<unsigned>(plane_count_or_modulus));
        hipLaunchKernelGGL(
            (cdna3_sparse_smfmac_kernel<Finite>),
            grid,
            dim3(64),
            0,
            0,
            static_cast<const uint8_t*>(device_a_packed_values),
            static_cast<const uint8_t*>(device_a_packed_indices),
            static_cast<const int8_t*>(device_b_residues),
            static_cast<int8_t*>(device_c_residues),
            m,
            n,
            k,
            ldb,
            ldc,
            plane_count_or_modulus);
      }
      break;
    case rns8_amdgpu_arch_kind::rdna4:
    {
      const dim3 grid(
          (static_cast<unsigned>(n) + 15u) / 16u,
          (static_cast<unsigned>(m) + 15u) / 16u,
          Finite ? 1u : static_cast<unsigned>(plane_count_or_modulus));
      hipLaunchKernelGGL(
          (rdna4_sparse_swmmac_kernel<Finite>),
          grid,
          dim3(32),
          0,
          0,
          static_cast<const uint8_t*>(device_a_packed_values),
          static_cast<const uint8_t*>(device_a_packed_indices),
          static_cast<const int8_t*>(device_b_residues),
          static_cast<int8_t*>(device_c_residues),
          m,
          n,
          k,
          ldb,
          ldc,
          plane_count_or_modulus);
      break;
    }
    case rns8_amdgpu_arch_kind::rdna3:
    case rns8_amdgpu_arch_kind::unsupported:
      return 2;
  }
  err = hipGetLastError();
  if (err != hipSuccess) {
    return 3;
  }
  err = hipDeviceSynchronize();
  return err == hipSuccess ? 0 : 3;
}

bool common_dense_args_valid(
    const void* a,
    const void* b,
    const void* c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc) {
  return a && b && c && m > 0 && n > 0 && k > 0 && lda >= k && ldb >= n && ldc >= n;
}

bool common_sparse_args_valid(
    const void* values,
    const void* indices,
    const void* b,
    const void* c,
    int m,
    int n,
    int k,
    int ldb,
    int ldc) {
  return values && indices && b && c && m > 0 && n > 0 && k > 0 && (k % 4) == 0 && ldb >= n && ldc >= n;
}

}  // namespace

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
    int prefix,
    int dense_kernel_variant) {
  if (!common_dense_args_valid(device_a_residues, device_b_residues, device_c_residues, m, n, k, lda, ldb, ldc) ||
      prefix <= 0 || prefix > kRns8DefaultModulusCount) {
    return 1;
  }
  return launch_dense<false>(
      device_id,
      device_a_residues,
      device_b_residues,
      device_c_residues,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      prefix,
      dense_kernel_variant);
}

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
    int modulus,
    int dense_kernel_variant) {
  if (!common_dense_args_valid(device_a_residues, device_b_residues, device_c_residues, m, n, k, lda, ldb, ldc) ||
      !rns8::detail::finite_u8::static_byte_modulus_supported(static_cast<uint32_t>(modulus))) {
    return 1;
  }
  return launch_dense<true>(
      device_id,
      device_a_residues,
      device_b_residues,
      device_c_residues,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      modulus,
      dense_kernel_variant);
}

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
    int prefix) {
  if (!common_sparse_args_valid(device_a_packed_values, device_a_packed_indices, device_b_residues, device_c_residues, m, n, k, ldb, ldc) ||
      prefix <= 0 || prefix > kRns8DefaultModulusCount) {
    return 1;
  }
  return launch_sparse_a<false>(
      device_id, device_a_packed_values, device_a_packed_indices, device_b_residues, device_c_residues, m, n, k, ldb, ldc, prefix);
}

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
    int modulus) {
  if (!common_sparse_args_valid(device_a_packed_values, device_a_packed_indices, device_b_residues, device_c_residues, m, n, k, ldb, ldc) ||
      !rns8::detail::finite_u8::static_byte_modulus_supported(static_cast<uint32_t>(modulus))) {
    return 1;
  }
  return launch_sparse_a<true>(
      device_id, device_a_packed_values, device_a_packed_indices, device_b_residues, device_c_residues, m, n, k, ldb, ldc, modulus);
}


// === Phase 3a: WMMA-native pack for AMDGPU builtins ===
// Packs native i64/u64 directly into WMMA-expected K-major centered i8 layout.
// Eliminates the generic Direct HIP pack + transpose step for RDNA3 WMMA paths.

__global__ void rns8_amdgpu_builtin_pack_wmma_i64_kernel(
    const int64_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t total = elements * static_cast<int64_t>(prefix);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int modulus_index = static_cast<int>(idx / elements);
  const int64_t cell = idx - static_cast<int64_t>(modulus_index) * elements;
  const int row = static_cast<int>(cell / cols);
  const int col = static_cast<int>(cell - static_cast<int64_t>(row) * cols);
  // Default modulus ladder: 256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127
  __device__ static constexpr int default_moduli[] = {256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};
  const int modulus = (modulus_index >= 0 && modulus_index < 28) ? default_moduli[modulus_index] : 256;

  // Direct centered residue conversion with contiguous source (ld == cols path)
  // WMMA expects K-major B layout; A remains row-major
  int64_t value = src[static_cast<int64_t>(row) * ld + col];
  int64_t reduced = value % static_cast<int64_t>(modulus);
  if (reduced < 0) reduced += modulus;
  dst[idx] = static_cast<int8_t>(reduced > modulus / 2 ? reduced - modulus : reduced);
}

__global__ void rns8_amdgpu_builtin_pack_wmma_u64_kernel(
    const uint64_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t total = elements * static_cast<int64_t>(prefix);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int modulus_index = static_cast<int>(idx / elements);
  const int64_t cell = idx - static_cast<int64_t>(modulus_index) * elements;
  const int row = static_cast<int>(cell / cols);
  const int col = static_cast<int>(cell - static_cast<int64_t>(row) * cols);
  __device__ static constexpr int default_moduli_u64[] = {256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};
  const int modulus = (modulus_index >= 0 && modulus_index < 28) ? default_moduli_u64[modulus_index] : 256;

  uint64_t value = src[static_cast<int64_t>(row) * ld + col];
  uint64_t reduced = value % static_cast<uint64_t>(modulus);
  dst[idx] = static_cast<int8_t>(reduced > static_cast<uint64_t>(modulus / 2)
      ? static_cast<int>(reduced - modulus) : static_cast<int>(reduced));
}

