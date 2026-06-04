#include <hip/hip_runtime.h>

#include <stdint.h>

#include "wrap64_hip_kernel_config.hpp"

namespace {

constexpr int kWrap64TileM = 16;
constexpr int kWrap64TileN = 16;
constexpr int kWrap64TileK = 16;
constexpr int kWrap64ByteLimbs = 8;
constexpr int kWrap64LowProductDiagonals = kWrap64ByteLimbs;
constexpr int kWrap64LowProductPairCount =
    (kWrap64LowProductDiagonals * (kWrap64LowProductDiagonals + 1)) / 2;
constexpr int kWrap64TileThreads = kWrap64TileM * kWrap64TileN;
static_assert(kWrap64LowProductPairCount == 36, "low 64-bit wrap64 uses exactly 36 byte-product pairs");
static_assert(sizeof(uint64_t) == kWrap64ByteLimbs, "wrap64 compact cell staging requires 64-bit cells");

__device__ int64_t rns8_wrap64_compact_limb_index(int64_t row, int64_t col, int64_t cols, uint32_t limb) {
  return ((row * cols + col) * kWrap64ByteLimbs) + static_cast<int64_t>(limb);
}

__device__ int32_t rns8_wrap64_signed_i8_lane_value(uint8_t value) {
  return value < 128u ? static_cast<int32_t>(value) : static_cast<int32_t>(value) - 256;
}

__device__ int32_t rns8_wrap64_signed_i8_product_correction(uint8_t a, uint8_t b) {
  const int32_t a_high = static_cast<int32_t>(a >> 7u);
  const int32_t b_high = static_cast<int32_t>(b >> 7u);
  return a_high * static_cast<int32_t>(b) * 256 + b_high * static_cast<int32_t>(a) * 256 -
         (a_high & b_high) * 65536;
}

__device__ uint32_t rns8_wrap64_unsigned_byte_product_from_signed_i8(uint8_t a, uint8_t b) {
  const int32_t signed_product = rns8_wrap64_signed_i8_lane_value(a) * rns8_wrap64_signed_i8_lane_value(b);
  const int32_t corrected = signed_product + rns8_wrap64_signed_i8_product_correction(a, b);
  return static_cast<uint32_t>(corrected);
}

__device__ uint32_t rns8_wrap64_unsigned_byte_product(uint8_t a, uint8_t b) {
  return static_cast<uint32_t>(a) * static_cast<uint32_t>(b);
}

__device__ void rns8_store_u64_limbs_vectorized(uint8_t* dst, uint64_t value) {
  *reinterpret_cast<uint64_t*>(dst) = value;
}

__device__ uint64_t rns8_load_u64_limbs_vectorized(const uint8_t* src) {
  return *reinterpret_cast<const uint64_t*>(src);
}

__device__ void rns8_store_u64_limbs_scalar(uint8_t* dst, uint64_t value) {
  for (uint32_t limb = 0; limb < kWrap64ByteLimbs; ++limb) {
    dst[limb] = static_cast<uint8_t>((value >> (8u * limb)) & 0xffu);
  }
}

__device__ uint64_t rns8_load_u64_limbs_scalar(const uint8_t* src) {
  uint64_t value = 0;
#pragma unroll
  for (uint32_t limb = 0; limb < kWrap64ByteLimbs; ++limb) {
    value |= static_cast<uint64_t>(src[limb]) << (8u * limb);
  }
  return value;
}

__device__ uint8_t rns8_wrap64_packed_limb_byte(uint64_t value, uint32_t limb) {
  return static_cast<uint8_t>((value >> (8u * limb)) & 0xffu);
}

template <typename Accumulator>
__device__ uint64_t rns8_wrap64_low64_from_diagonals(const Accumulator* diagonals) {
  uint64_t carry = 0;
  uint64_t value = 0;
#pragma unroll
  for (uint32_t diagonal_index = 0; diagonal_index < kWrap64LowProductDiagonals; ++diagonal_index) {
    const uint64_t column = static_cast<uint64_t>(diagonals[diagonal_index]) + carry;
    value |= (column & 0xffu) << (8u * diagonal_index);
    carry = column >> 8u;
  }
  return value;
}

template <typename Accumulator>
__device__ void rns8_wrap64_accumulate_byte_gemm36_from_packed_cells(
    uint64_t a_value,
    uint64_t b_value,
    Accumulator* diagonals) {
  const uint8_t a0 = rns8_wrap64_packed_limb_byte(a_value, 0);
  const uint8_t a1 = rns8_wrap64_packed_limb_byte(a_value, 1);
  const uint8_t a2 = rns8_wrap64_packed_limb_byte(a_value, 2);
  const uint8_t a3 = rns8_wrap64_packed_limb_byte(a_value, 3);
  const uint8_t a4 = rns8_wrap64_packed_limb_byte(a_value, 4);
  const uint8_t a5 = rns8_wrap64_packed_limb_byte(a_value, 5);
  const uint8_t a6 = rns8_wrap64_packed_limb_byte(a_value, 6);
  const uint8_t a7 = rns8_wrap64_packed_limb_byte(a_value, 7);
  const uint8_t b0 = rns8_wrap64_packed_limb_byte(b_value, 0);
  const uint8_t b1 = rns8_wrap64_packed_limb_byte(b_value, 1);
  const uint8_t b2 = rns8_wrap64_packed_limb_byte(b_value, 2);
  const uint8_t b3 = rns8_wrap64_packed_limb_byte(b_value, 3);
  const uint8_t b4 = rns8_wrap64_packed_limb_byte(b_value, 4);
  const uint8_t b5 = rns8_wrap64_packed_limb_byte(b_value, 5);
  const uint8_t b6 = rns8_wrap64_packed_limb_byte(b_value, 6);
  const uint8_t b7 = rns8_wrap64_packed_limb_byte(b_value, 7);

  diagonals[0] += static_cast<Accumulator>(rns8_wrap64_unsigned_byte_product(a0, b0));
  diagonals[1] += static_cast<Accumulator>(rns8_wrap64_unsigned_byte_product(a0, b1)) +
                  rns8_wrap64_unsigned_byte_product(a1, b0);
  diagonals[2] += static_cast<Accumulator>(rns8_wrap64_unsigned_byte_product(a0, b2)) +
                  rns8_wrap64_unsigned_byte_product(a1, b1) +
                  rns8_wrap64_unsigned_byte_product(a2, b0);
  diagonals[3] += static_cast<Accumulator>(rns8_wrap64_unsigned_byte_product(a0, b3)) +
                  rns8_wrap64_unsigned_byte_product(a1, b2) +
                  rns8_wrap64_unsigned_byte_product(a2, b1) +
                  rns8_wrap64_unsigned_byte_product(a3, b0);
  diagonals[4] += static_cast<Accumulator>(rns8_wrap64_unsigned_byte_product(a0, b4)) +
                  rns8_wrap64_unsigned_byte_product(a1, b3) +
                  rns8_wrap64_unsigned_byte_product(a2, b2) +
                  rns8_wrap64_unsigned_byte_product(a3, b1) +
                  rns8_wrap64_unsigned_byte_product(a4, b0);
  diagonals[5] += static_cast<Accumulator>(rns8_wrap64_unsigned_byte_product(a0, b5)) +
                  rns8_wrap64_unsigned_byte_product(a1, b4) +
                  rns8_wrap64_unsigned_byte_product(a2, b3) +
                  rns8_wrap64_unsigned_byte_product(a3, b2) +
                  rns8_wrap64_unsigned_byte_product(a4, b1) +
                  rns8_wrap64_unsigned_byte_product(a5, b0);
  diagonals[6] += static_cast<Accumulator>(rns8_wrap64_unsigned_byte_product(a0, b6)) +
                  rns8_wrap64_unsigned_byte_product(a1, b5) +
                  rns8_wrap64_unsigned_byte_product(a2, b4) +
                  rns8_wrap64_unsigned_byte_product(a3, b3) +
                  rns8_wrap64_unsigned_byte_product(a4, b2) +
                  rns8_wrap64_unsigned_byte_product(a5, b1) +
                  rns8_wrap64_unsigned_byte_product(a6, b0);
  diagonals[7] += static_cast<Accumulator>(rns8_wrap64_unsigned_byte_product(a0, b7)) +
                  rns8_wrap64_unsigned_byte_product(a1, b6) +
                  rns8_wrap64_unsigned_byte_product(a2, b5) +
                  rns8_wrap64_unsigned_byte_product(a3, b4) +
                  rns8_wrap64_unsigned_byte_product(a4, b3) +
                  rns8_wrap64_unsigned_byte_product(a5, b2) +
                  rns8_wrap64_unsigned_byte_product(a6, b1) +
                  rns8_wrap64_unsigned_byte_product(a7, b0);
}

__global__ void rns8_wrap64_pack_u64_kernel(
    const uint64_t* src,
    uint8_t* byte_limbs,
    int64_t rows,
    int64_t cols,
    int64_t ld) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  rns8_store_u64_limbs_vectorized(
      byte_limbs + rns8_wrap64_compact_limb_index(row, col, cols, 0), src[row * ld + col]);
}

__global__ void rns8_wrap64_pack_u64_scalar_kernel(
    const uint64_t* src,
    uint8_t* byte_limbs,
    int64_t rows,
    int64_t cols,
    int64_t ld) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  rns8_store_u64_limbs_scalar(
      byte_limbs + rns8_wrap64_compact_limb_index(row, col, cols, 0), src[row * ld + col]);
}

template <typename Accumulator>
__device__ void rns8_wrap64_byte_gemm36_tiled_kernel_impl(
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    uint8_t* c_limbs,
    int64_t m,
    int64_t n,
    int64_t k) {
  // Stage compact cells once; all unsigned byte lanes are still extracted explicitly below.
  __shared__ uint64_t a_tile[kWrap64TileM * kWrap64TileK];
  __shared__ uint64_t b_tile[kWrap64TileK * kWrap64TileN];

  const int64_t tile_col = static_cast<int64_t>(blockIdx.x);
  const int64_t tile_row = static_cast<int64_t>(blockIdx.y);
  const int64_t row = tile_row * kWrap64TileM + threadIdx.y;
  const int64_t col = tile_col * kWrap64TileN + threadIdx.x;
  const int thread_index = static_cast<int>(threadIdx.y) * kWrap64TileN + static_cast<int>(threadIdx.x);
  const bool stores_output = row < m && col < n;

  // Strict mod 2^64 needs only the low eight product diagonals; higher diagonals are multiples of 2^64.
  Accumulator diagonals[kWrap64LowProductDiagonals] = {};
  for (int64_t k_base = 0; k_base < k; k_base += kWrap64TileK) {
    const int64_t k_extent = k - k_base < kWrap64TileK ? k - k_base : kWrap64TileK;

    for (int cell = thread_index; cell < kWrap64TileM * kWrap64TileK; cell += kWrap64TileThreads) {
      const int local_k = cell % kWrap64TileK;
      const int local_row = cell / kWrap64TileK;
      const int64_t global_row = tile_row * kWrap64TileM + local_row;
      const int64_t global_k = k_base + local_k;
      uint64_t value = 0;
      if (global_row < m && local_k < k_extent) {
        value = rns8_load_u64_limbs_vectorized(
            a_limbs + rns8_wrap64_compact_limb_index(global_row, global_k, k, 0));
      }
      a_tile[cell] = value;
    }

    for (int cell = thread_index; cell < kWrap64TileK * kWrap64TileN; cell += kWrap64TileThreads) {
      const int local_col = cell % kWrap64TileN;
      const int local_k = cell / kWrap64TileN;
      const int64_t global_col = tile_col * kWrap64TileN + local_col;
      const int64_t global_k = k_base + local_k;
      uint64_t value = 0;
      if (local_k < k_extent && global_col < n) {
        value = rns8_load_u64_limbs_vectorized(
            b_limbs + rns8_wrap64_compact_limb_index(global_k, global_col, n, 0));
      }
      b_tile[cell] = value;
    }

    __syncthreads();

    if (stores_output) {
      for (int local_k = 0; local_k < k_extent; ++local_k) {
        const uint64_t a = a_tile[static_cast<int>(threadIdx.y) * kWrap64TileK + local_k];
        const uint64_t b = b_tile[local_k * kWrap64TileN + static_cast<int>(threadIdx.x)];
        rns8_wrap64_accumulate_byte_gemm36_from_packed_cells(a, b, diagonals);
      }
    }

    __syncthreads();
  }

  if (!stores_output) {
    return;
  }

  uint8_t* dst = c_limbs + rns8_wrap64_compact_limb_index(row, col, n, 0);
  rns8_store_u64_limbs_vectorized(dst, rns8_wrap64_low64_from_diagonals(diagonals));
}

__global__ void rns8_wrap64_byte_gemm36_tiled_u32acc_kernel(
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    uint8_t* c_limbs,
    int64_t m,
    int64_t n,
    int64_t k) {
  rns8_wrap64_byte_gemm36_tiled_kernel_impl<uint32_t>(a_limbs, b_limbs, c_limbs, m, n, k);
}

__global__ void rns8_wrap64_byte_gemm36_tiled_u64acc_kernel(
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    uint8_t* c_limbs,
    int64_t m,
    int64_t n,
    int64_t k) {
  rns8_wrap64_byte_gemm36_tiled_kernel_impl<uint64_t>(a_limbs, b_limbs, c_limbs, m, n, k);
}

__global__ void rns8_wrap64_export_u64_kernel(
    const uint8_t* byte_limbs,
    uint64_t* dst,
    int64_t rows,
    int64_t cols) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  dst[row * cols + col] =
      rns8_load_u64_limbs_vectorized(byte_limbs + rns8_wrap64_compact_limb_index(row, col, cols, 0));
}

__global__ void rns8_wrap64_export_u64_scalar_kernel(
    const uint8_t* byte_limbs,
    uint64_t* dst,
    int64_t rows,
    int64_t cols) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  dst[row * cols + col] =
      rns8_load_u64_limbs_scalar(byte_limbs + rns8_wrap64_compact_limb_index(row, col, cols, 0));
}

}  // namespace

extern "C" int rns8_wrap64_hip_pack_u64_device(
    const uint64_t* src,
    uint8_t* byte_limbs,
    int64_t rows,
    int64_t cols,
    int64_t ld) {
  constexpr int threads_x = 16;
  constexpr int threads_y = 16;
  const dim3 threads(threads_x, threads_y);
  const dim3 blocks(
      static_cast<unsigned int>((cols + threads_x - 1) / threads_x),
      static_cast<unsigned int>((rows + threads_y - 1) / threads_y));
  if (rns8::detail::wrap64_hip_uses_vectorized_pack_export(rows, cols)) {
    hipLaunchKernelGGL(
        rns8_wrap64_pack_u64_kernel,
        blocks,
        threads,
        0,
        nullptr,
        src,
        byte_limbs,
        rows,
        cols,
        ld);
  } else {
    hipLaunchKernelGGL(
        rns8_wrap64_pack_u64_scalar_kernel,
        blocks,
        threads,
        0,
        nullptr,
        src,
        byte_limbs,
        rows,
        cols,
        ld);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_wrap64_hip_gemm_byte_limbs_device(
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    uint8_t* c_limbs,
    int64_t m,
    int64_t n,
    int64_t k) {
  const dim3 threads(kWrap64TileN, kWrap64TileM);
  const int64_t tile_rows = (m + kWrap64TileM - 1) / kWrap64TileM;
  const int64_t tile_cols = (n + kWrap64TileN - 1) / kWrap64TileN;
  const dim3 blocks(static_cast<unsigned int>(tile_cols), static_cast<unsigned int>(tile_rows));
  if (rns8::detail::wrap64_hip_uses_u32_accumulator(k)) {
    hipLaunchKernelGGL(
        rns8_wrap64_byte_gemm36_tiled_u32acc_kernel,
        blocks,
        threads,
        0,
        nullptr,
        a_limbs,
        b_limbs,
        c_limbs,
        m,
        n,
        k);
  } else {
    hipLaunchKernelGGL(
        rns8_wrap64_byte_gemm36_tiled_u64acc_kernel,
        blocks,
        threads,
        0,
        nullptr,
        a_limbs,
        b_limbs,
        c_limbs,
        m,
        n,
        k);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_wrap64_hip_export_u64_device(
    const uint8_t* byte_limbs,
    uint64_t* dst,
    int64_t rows,
    int64_t cols) {
  constexpr int threads_x = 16;
  constexpr int threads_y = 16;
  const dim3 threads(threads_x, threads_y);
  const dim3 blocks(
      static_cast<unsigned int>((cols + threads_x - 1) / threads_x),
      static_cast<unsigned int>((rows + threads_y - 1) / threads_y));
  if (rns8::detail::wrap64_hip_uses_vectorized_pack_export(rows, cols)) {
    hipLaunchKernelGGL(
        rns8_wrap64_export_u64_kernel,
        blocks,
        threads,
        0,
        nullptr,
        byte_limbs,
        dst,
        rows,
        cols);
  } else {
    hipLaunchKernelGGL(
        rns8_wrap64_export_u64_scalar_kernel,
        blocks,
        threads,
        0,
        nullptr,
        byte_limbs,
        dst,
        rows,
        cols);
  }
  return static_cast<int>(hipGetLastError());
}
