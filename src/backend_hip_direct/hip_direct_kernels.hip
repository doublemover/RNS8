#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

__device__ int8_t rns8_reduce_to_centered_device(int value, int modulus) {
  int residue = value % modulus;
  if (residue < 0) {
    residue += modulus;
  }
  const int threshold = (modulus + 1) / 2;
  if (residue >= threshold) {
    residue -= modulus;
  }
  return static_cast<int8_t>(residue);
}

__global__ void rns8_ring_gemm_kernel(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  int acc = 0;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int>(A[row * lda + kk]) * static_cast<int>(B[kk * ldb + col]);
  }
  C[row * ldc + col] = rns8_reduce_to_centered_device(acc, modulus);
}

}  // namespace

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
    int modulus) {
  const size_t bytes_a = static_cast<size_t>(m) * static_cast<size_t>(lda) * sizeof(int8_t);
  const size_t bytes_b = static_cast<size_t>(k) * static_cast<size_t>(ldb) * sizeof(int8_t);
  const size_t bytes_c = static_cast<size_t>(m) * static_cast<size_t>(ldc) * sizeof(int8_t);

  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  int8_t* d_c = nullptr;

  hipError_t err = hipMalloc(&d_a, bytes_a);
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }
  err = hipMalloc(&d_b, bytes_b);
  if (err != hipSuccess) {
    (void)hipFree(d_a);
    return static_cast<int>(err);
  }
  err = hipMalloc(&d_c, bytes_c);
  if (err != hipSuccess) {
    (void)hipFree(d_b);
    (void)hipFree(d_a);
    return static_cast<int>(err);
  }

  err = hipMemcpy(d_a, A, bytes_a, hipMemcpyHostToDevice);
  if (err == hipSuccess) {
    err = hipMemcpy(d_b, B, bytes_b, hipMemcpyHostToDevice);
  }
  if (err == hipSuccess) {
    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    hipLaunchKernelGGL(rns8_ring_gemm_kernel, grid, block, 0, nullptr, d_a, d_b, d_c, m, n, k, lda, ldb, ldc, modulus);
    err = hipGetLastError();
  }
  if (err == hipSuccess) {
    err = hipDeviceSynchronize();
  }
  if (err == hipSuccess) {
    err = hipMemcpy(C, d_c, bytes_c, hipMemcpyDeviceToHost);
  }

  (void)hipFree(d_c);
  (void)hipFree(d_b);
  (void)hipFree(d_a);
  return static_cast<int>(err);
}
