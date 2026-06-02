#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

__device__ __constant__ int rns8_default_moduli_device[28] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

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

__device__ int8_t rns8_center_i64_device(int64_t value, int modulus) {
  int64_t residue = value % static_cast<int64_t>(modulus);
  if (residue < 0) {
    residue += modulus;
  }
  const int64_t threshold = (static_cast<int64_t>(modulus) + 1) / 2;
  if (residue >= threshold) {
    residue -= modulus;
  }
  return static_cast<int8_t>(residue);
}

__device__ int8_t rns8_center_u64_device(uint64_t value, int modulus) {
  uint64_t residue = value % static_cast<uint64_t>(modulus);
  const uint64_t threshold = (static_cast<uint64_t>(modulus) + 1) / 2;
  int64_t centered = static_cast<int64_t>(residue);
  if (residue >= threshold) {
    centered -= modulus;
  }
  return static_cast<int8_t>(centered);
}

__global__ void rns8_pack_i64_kernel(
    const int64_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t total = elements * static_cast<int64_t>(prefix);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int modulus_index = static_cast<int>(idx / elements);
  const int64_t element = idx - static_cast<int64_t>(modulus_index) * elements;
  const int row = static_cast<int>(element / cols);
  const int col = static_cast<int>(element - static_cast<int64_t>(row) * cols);
  const int modulus = rns8_default_moduli_device[modulus_index];
  residues[idx] = rns8_center_i64_device(src[static_cast<int64_t>(row) * ld + col], modulus);
}

__global__ void rns8_pack_u64_kernel(
    const uint64_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t total = elements * static_cast<int64_t>(prefix);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int modulus_index = static_cast<int>(idx / elements);
  const int64_t element = idx - static_cast<int64_t>(modulus_index) * elements;
  const int row = static_cast<int>(element / cols);
  const int col = static_cast<int>(element - static_cast<int64_t>(row) * cols);
  const int modulus = rns8_default_moduli_device[modulus_index];
  residues[idx] = rns8_center_u64_device(src[static_cast<int64_t>(row) * ld + col], modulus);
}

__global__ void rns8_ring_gemm_kernel(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int m,
    int n,
    int k_offset,
    int k_block,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    int accumulate) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  int acc = 0;
  for (int kk = 0; kk < k_block; ++kk) {
    const int source_k = k_offset + kk;
    acc += static_cast<int>(A[row * lda + source_k]) * static_cast<int>(B[source_k * ldb + col]);
  }
  if (accumulate) {
    acc += static_cast<int>(C[row * ldc + col]);
  }
  C[row * ldc + col] = rns8_reduce_to_centered_device(acc, modulus);
}

}  // namespace

extern "C" int rns8_hip_direct_pack_i64(
    const int64_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const size_t bytes_src = static_cast<size_t>(rows) * static_cast<size_t>(ld) * sizeof(int64_t);
  const size_t bytes_residues =
      static_cast<size_t>(prefix) * static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(int8_t);

  int64_t* d_src = nullptr;
  int8_t* d_residues = nullptr;

  hipError_t err = hipMalloc(&d_src, bytes_src);
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }
  err = hipMalloc(&d_residues, bytes_residues);
  if (err != hipSuccess) {
    (void)hipFree(d_src);
    return static_cast<int>(err);
  }

  err = hipMemcpy(d_src, src, bytes_src, hipMemcpyHostToDevice);
  if (err == hipSuccess) {
    const int threads = 256;
    const int64_t total =
        static_cast<int64_t>(prefix) * static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    const int blocks = static_cast<int>((total + threads - 1) / threads);
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
    err = hipGetLastError();
  }
  if (err == hipSuccess) {
    err = hipDeviceSynchronize();
  }
  if (err == hipSuccess) {
    err = hipMemcpy(residues, d_residues, bytes_residues, hipMemcpyDeviceToHost);
  }

  (void)hipFree(d_residues);
  (void)hipFree(d_src);
  return static_cast<int>(err);
}

extern "C" int rns8_hip_direct_pack_u64(
    const uint64_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const size_t bytes_src = static_cast<size_t>(rows) * static_cast<size_t>(ld) * sizeof(uint64_t);
  const size_t bytes_residues =
      static_cast<size_t>(prefix) * static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(int8_t);

  uint64_t* d_src = nullptr;
  int8_t* d_residues = nullptr;

  hipError_t err = hipMalloc(&d_src, bytes_src);
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }
  err = hipMalloc(&d_residues, bytes_residues);
  if (err != hipSuccess) {
    (void)hipFree(d_src);
    return static_cast<int>(err);
  }

  err = hipMemcpy(d_src, src, bytes_src, hipMemcpyHostToDevice);
  if (err == hipSuccess) {
    const int threads = 256;
    const int64_t total =
        static_cast<int64_t>(prefix) * static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    const int blocks = static_cast<int>((total + threads - 1) / threads);
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
    err = hipGetLastError();
  }
  if (err == hipSuccess) {
    err = hipDeviceSynchronize();
  }
  if (err == hipSuccess) {
    err = hipMemcpy(residues, d_residues, bytes_residues, hipMemcpyDeviceToHost);
  }

  (void)hipFree(d_residues);
  (void)hipFree(d_src);
  return static_cast<int>(err);
}

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
    int k_offset = 0;
    int accumulate = 0;
    while (k_offset < k && err == hipSuccess) {
      const int k_remaining = k - k_offset;
      const int k_block = k_remaining > 65536 ? 65536 : k_remaining;
      hipLaunchKernelGGL(
          rns8_ring_gemm_kernel,
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
          accumulate);
      err = hipGetLastError();
      k_offset += k_block;
      accumulate = 1;
    }
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
