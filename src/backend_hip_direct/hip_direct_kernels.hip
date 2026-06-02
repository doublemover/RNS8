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

__device__ uint32_t rns8_canonical_from_centered_device(int8_t residue, int modulus) {
  const int value = static_cast<int>(residue);
  return static_cast<uint32_t>(value < 0 ? value + modulus : value);
}

__device__ uint32_t rns8_mod_inverse_device(uint32_t a, uint32_t modulus) {
  int64_t t = 0;
  int64_t next_t = 1;
  int64_t r = modulus;
  int64_t next_r = a % modulus;

  while (next_r != 0) {
    const int64_t quotient = r / next_r;
    const int64_t new_t = t - quotient * next_t;
    t = next_t;
    next_t = new_t;
    const int64_t new_r = r - quotient * next_r;
    r = next_r;
    next_r = new_r;
  }

  if (r != 1) {
    return 0;
  }
  if (t < 0) {
    t += modulus;
  }
  return static_cast<uint32_t>(t);
}

__device__ unsigned __int128 rns8_reconstruct_canonical_device(
    const int8_t* residues,
    int cell,
    int elements,
    int prefix) {
  unsigned __int128 x = 0;
  unsigned __int128 product = 1;
  for (int i = 0; i < prefix; ++i) {
    const int modulus = rns8_default_moduli_device[i];
    const uint32_t target =
        rns8_canonical_from_centered_device(residues[static_cast<int64_t>(i) * elements + cell], modulus);
    const uint32_t x_mod = static_cast<uint32_t>(x % static_cast<uint32_t>(modulus));
    const uint32_t product_mod = static_cast<uint32_t>(product % static_cast<uint32_t>(modulus));
    const uint32_t inverse = rns8_mod_inverse_device(product_mod, static_cast<uint32_t>(modulus));
    int64_t delta_mod = static_cast<int64_t>(target) - static_cast<int64_t>(x_mod);
    delta_mod %= modulus;
    if (delta_mod < 0) {
      delta_mod += modulus;
    }
    const uint32_t coefficient = static_cast<uint32_t>(
        (static_cast<uint64_t>(delta_mod) * static_cast<uint64_t>(inverse)) % static_cast<uint32_t>(modulus));
    x += product * static_cast<unsigned __int128>(coefficient);
    product *= static_cast<unsigned __int128>(modulus);
  }
  return x;
}

__device__ unsigned __int128 rns8_modulus_product_device(int prefix) {
  unsigned __int128 product = 1;
  for (int i = 0; i < prefix; ++i) {
    product *= static_cast<unsigned __int128>(rns8_default_moduli_device[i]);
  }
  return product;
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

__global__ void rns8_export_i64_kernel(
    const int8_t* residues,
    int64_t* dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  const unsigned __int128 x = rns8_reconstruct_canonical_device(residues, cell, elements, prefix);
  const unsigned __int128 product = rns8_modulus_product_device(prefix);
  const unsigned __int128 bound128 = static_cast<unsigned __int128>(bound);
  const unsigned __int128 int64_max128 = static_cast<unsigned __int128>(0x7fffffffffffffffULL);

  if (x > product / 2) {
    const unsigned __int128 magnitude = product - x;
    if (magnitude > bound128 || magnitude > int64_max128) {
      atomicCAS(status, 0, 5);
      return;
    }
    dst[cell] = -static_cast<int64_t>(static_cast<uint64_t>(magnitude));
    return;
  }

  if (x > bound128 || x > int64_max128) {
    atomicCAS(status, 0, 5);
    return;
  }
  dst[cell] = static_cast<int64_t>(static_cast<uint64_t>(x));
}

__global__ void rns8_export_u64_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int prefix,
    uint64_t bound,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  const unsigned __int128 x = rns8_reconstruct_canonical_device(residues, cell, elements, prefix);
  const unsigned __int128 bound128 = static_cast<unsigned __int128>(bound);
  const unsigned __int128 uint64_max128 = static_cast<unsigned __int128>(0xffffffffffffffffULL);
  if (x > bound128 || x > uint64_max128) {
    atomicCAS(status, 0, 5);
    return;
  }
  dst[cell] = static_cast<uint64_t>(x);
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

extern "C" int rns8_hip_direct_pack_i64_device(
    const int64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
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
  return static_cast<int>(hipGetLastError());
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

extern "C" int rns8_hip_direct_pack_u64_device(
    const uint64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
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
  return static_cast<int>(hipGetLastError());
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
    int modulus) {
  const dim3 block(16, 16);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  hipError_t err = hipSuccess;
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
  return static_cast<int>(err);
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
  return static_cast<int>(hipGetLastError());
}
