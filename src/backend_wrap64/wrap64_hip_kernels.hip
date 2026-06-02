#include <hip/hip_runtime.h>

#include <stdint.h>

namespace {

constexpr int kWrap64TileM = 16;
constexpr int kWrap64TileN = 16;
constexpr int kWrap64TileK = 16;
constexpr int kWrap64ByteLimbs = 8;
constexpr int kWrap64TileThreads = kWrap64TileM * kWrap64TileN;

__device__ int64_t rns8_wrap64_compact_limb_index(int64_t row, int64_t col, int64_t cols, uint32_t limb) {
  return ((row * cols + col) * kWrap64ByteLimbs) + static_cast<int64_t>(limb);
}

__device__ int32_t rns8_wrap64_signed_i8_lane_value(uint8_t value) {
  return value < 128u ? static_cast<int32_t>(value) : static_cast<int32_t>(value) - 256;
}

__device__ int32_t rns8_wrap64_signed_i8_product_correction(uint8_t a, uint8_t b) {
  const int32_t signed_product = rns8_wrap64_signed_i8_lane_value(a) * rns8_wrap64_signed_i8_lane_value(b);
  const int32_t unsigned_product = static_cast<int32_t>(a) * static_cast<int32_t>(b);
  return unsigned_product - signed_product;
}

__device__ uint32_t rns8_wrap64_unsigned_byte_product_from_signed_i8(uint8_t a, uint8_t b) {
  const int32_t signed_product = rns8_wrap64_signed_i8_lane_value(a) * rns8_wrap64_signed_i8_lane_value(b);
  const int32_t corrected = signed_product + rns8_wrap64_signed_i8_product_correction(a, b);
  return static_cast<uint32_t>(corrected);
}

__device__ void rns8_store_u64_limbs(uint8_t* dst, uint64_t value) {
  for (uint32_t limb = 0; limb < kWrap64ByteLimbs; ++limb) {
    dst[limb] = static_cast<uint8_t>((value >> (8u * limb)) & 0xffu);
  }
}

__device__ uint64_t rns8_load_u64_limbs(const uint8_t* src) {
  uint64_t value = 0;
  for (uint32_t limb = 0; limb < kWrap64ByteLimbs; ++limb) {
    value |= static_cast<uint64_t>(src[limb]) << (8u * limb);
  }
  return value;
}

__device__ void rns8_wrap64_store_low64_from_diagonals(uint8_t* dst, const uint64_t* diagonals) {
  uint64_t carry = 0;
  for (uint32_t diagonal_index = 0; diagonal_index < kWrap64ByteLimbs; ++diagonal_index) {
    const uint64_t column = diagonals[diagonal_index] + carry;
    dst[diagonal_index] = static_cast<uint8_t>(column & 0xffu);
    carry = column >> 8u;
  }
}

__global__ void rns8_wrap64_pack_u64_kernel(
    const uint64_t* src,
    uint8_t* byte_limbs,
    int64_t rows,
    int64_t cols,
    int64_t ld) {
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * cols;
  if (cell >= total) {
    return;
  }
  const int64_t row = cell / cols;
  const int64_t col = cell - row * cols;
  rns8_store_u64_limbs(byte_limbs + rns8_wrap64_compact_limb_index(row, col, cols, 0), src[row * ld + col]);
}

__global__ void rns8_wrap64_tiled_byte_gemm_kernel(
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    uint8_t* c_limbs,
    int64_t m,
    int64_t n,
    int64_t k) {
  __shared__ uint8_t a_tile[kWrap64TileM * kWrap64TileK * kWrap64ByteLimbs];
  __shared__ uint8_t b_tile[kWrap64TileK * kWrap64TileN * kWrap64ByteLimbs];

  const int64_t tile_cols = (n + kWrap64TileN - 1) / kWrap64TileN;
  const int64_t tile_index = static_cast<int64_t>(blockIdx.x);
  const int64_t tile_row = tile_index / tile_cols;
  const int64_t tile_col = tile_index - tile_row * tile_cols;
  const int64_t row = tile_row * kWrap64TileM + threadIdx.y;
  const int64_t col = tile_col * kWrap64TileN + threadIdx.x;
  const int thread_index = static_cast<int>(threadIdx.y) * kWrap64TileN + static_cast<int>(threadIdx.x);
  const bool stores_output = row < m && col < n;

  // Host guards keep k <= INT_MAX; each diagonal has at most eight 255*255 terms per k.
  uint64_t diagonals[kWrap64ByteLimbs] = {};
  for (int64_t k_base = 0; k_base < k; k_base += kWrap64TileK) {
    const int64_t k_extent = k - k_base < kWrap64TileK ? k - k_base : kWrap64TileK;

    for (int index = thread_index; index < kWrap64TileM * kWrap64TileK * kWrap64ByteLimbs;
         index += kWrap64TileThreads) {
      const int limb = index % kWrap64ByteLimbs;
      const int cell = index / kWrap64ByteLimbs;
      const int local_k = cell % kWrap64TileK;
      const int local_row = cell / kWrap64TileK;
      const int64_t global_row = tile_row * kWrap64TileM + local_row;
      const int64_t global_k = k_base + local_k;
      uint8_t value = 0;
      if (global_row < m && local_k < k_extent) {
        value = a_limbs[rns8_wrap64_compact_limb_index(global_row, global_k, k, static_cast<uint32_t>(limb))];
      }
      a_tile[index] = value;
    }

    for (int index = thread_index; index < kWrap64TileK * kWrap64TileN * kWrap64ByteLimbs;
         index += kWrap64TileThreads) {
      const int limb = index % kWrap64ByteLimbs;
      const int cell = index / kWrap64ByteLimbs;
      const int local_col = cell % kWrap64TileN;
      const int local_k = cell / kWrap64TileN;
      const int64_t global_col = tile_col * kWrap64TileN + local_col;
      const int64_t global_k = k_base + local_k;
      uint8_t value = 0;
      if (local_k < k_extent && global_col < n) {
        value = b_limbs[rns8_wrap64_compact_limb_index(global_k, global_col, n, static_cast<uint32_t>(limb))];
      }
      b_tile[index] = value;
    }

    __syncthreads();

    if (stores_output) {
      for (int local_k = 0; local_k < k_extent; ++local_k) {
        const uint8_t* a = a_tile + ((static_cast<int>(threadIdx.y) * kWrap64TileK + local_k) * kWrap64ByteLimbs);
        const uint8_t* b = b_tile + ((local_k * kWrap64TileN + static_cast<int>(threadIdx.x)) * kWrap64ByteLimbs);
        for (uint32_t a_limb = 0; a_limb < kWrap64ByteLimbs; ++a_limb) {
          for (uint32_t b_limb = 0; b_limb + a_limb < kWrap64ByteLimbs; ++b_limb) {
            diagonals[a_limb + b_limb] +=
                static_cast<uint64_t>(rns8_wrap64_unsigned_byte_product_from_signed_i8(a[a_limb], b[b_limb]));
          }
        }
      }
    }

    __syncthreads();
  }

  if (!stores_output) {
    return;
  }

  uint8_t* dst = c_limbs + rns8_wrap64_compact_limb_index(row, col, n, 0);
  rns8_wrap64_store_low64_from_diagonals(dst, diagonals);
}

__global__ void rns8_wrap64_export_u64_kernel(
    const uint8_t* byte_limbs,
    uint64_t* dst,
    int64_t rows,
    int64_t cols) {
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * cols;
  if (cell >= total) {
    return;
  }
  const int64_t row = cell / cols;
  const int64_t col = cell - row * cols;
  dst[cell] = rns8_load_u64_limbs(byte_limbs + rns8_wrap64_compact_limb_index(row, col, cols, 0));
}

}  // namespace

extern "C" int rns8_wrap64_hip_pack_u64_device(
    const uint64_t* src,
    uint8_t* byte_limbs,
    int64_t rows,
    int64_t cols,
    int64_t ld) {
  constexpr int threads = 256;
  const int64_t total = rows * cols;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(
      rns8_wrap64_pack_u64_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      nullptr,
      src,
      byte_limbs,
      rows,
      cols,
      ld);
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
  const dim3 blocks(static_cast<unsigned int>(tile_rows * tile_cols));
  hipLaunchKernelGGL(
      rns8_wrap64_tiled_byte_gemm_kernel,
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
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_wrap64_hip_export_u64_device(
    const uint8_t* byte_limbs,
    uint64_t* dst,
    int64_t rows,
    int64_t cols) {
  constexpr int threads = 256;
  const int64_t total = rows * cols;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(
      rns8_wrap64_export_u64_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      nullptr,
      byte_limbs,
      dst,
      rows,
      cols);
  return static_cast<int>(hipGetLastError());
}
