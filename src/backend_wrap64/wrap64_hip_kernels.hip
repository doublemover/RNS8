#include <hip/hip_runtime.h>

#include <stdint.h>

namespace {

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
  for (uint32_t limb = 0; limb < 8; ++limb) {
    dst[limb] = static_cast<uint8_t>((value >> (8u * limb)) & 0xffu);
  }
}

__device__ uint64_t rns8_load_u64_limbs(const uint8_t* src) {
  uint64_t value = 0;
  for (uint32_t limb = 0; limb < 8; ++limb) {
    value |= static_cast<uint64_t>(src[limb]) << (8u * limb);
  }
  return value;
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
  rns8_store_u64_limbs(byte_limbs + cell * 8, src[row * ld + col]);
}

__global__ void rns8_wrap64_byte_gemm36_kernel(
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    uint8_t* c_limbs,
    int64_t m,
    int64_t n,
    int64_t k) {
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = m * n;
  if (cell >= total) {
    return;
  }
  const int64_t row = cell / n;
  const int64_t col = cell - row * n;
  // Host guards keep k <= INT_MAX; each diagonal has at most eight 255*255 terms per k.
  uint64_t diagonals[8] = {};
  for (int64_t kk = 0; kk < k; ++kk) {
    const uint8_t* a = a_limbs + ((row * k + kk) * 8);
    const uint8_t* b = b_limbs + ((kk * n + col) * 8);
    for (uint32_t a_limb = 0; a_limb < 8; ++a_limb) {
      for (uint32_t b_limb = 0; b_limb + a_limb < 8; ++b_limb) {
        diagonals[a_limb + b_limb] +=
            static_cast<uint64_t>(rns8_wrap64_unsigned_byte_product_from_signed_i8(a[a_limb], b[b_limb]));
      }
    }
  }
  uint8_t* dst = c_limbs + cell * 8;
  uint64_t carry = 0;
  for (uint32_t diagonal_index = 0; diagonal_index < 8; ++diagonal_index) {
    const uint64_t column = diagonals[diagonal_index] + carry;
    dst[diagonal_index] = static_cast<uint8_t>(column & 0xffu);
    carry = column >> 8u;
  }
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
  dst[cell] = rns8_load_u64_limbs(byte_limbs + cell * 8);
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
  constexpr int threads = 256;
  const int64_t total = m * n;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(
      rns8_wrap64_byte_gemm36_kernel,
      dim3(blocks),
      dim3(threads),
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
