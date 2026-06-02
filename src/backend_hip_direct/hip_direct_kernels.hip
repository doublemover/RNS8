#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int kRns8DefaultModulusCount = 28;

__device__ __constant__ int rns8_default_moduli_device[kRns8DefaultModulusCount] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

constexpr int kRns8HipTileM = 16;
constexpr int kRns8HipTileN = 16;
constexpr int kRns8HipTileK = 64;
constexpr int kRns8SafeInt32KBlock = 65536;

__device__ int8_t rns8_reduce_to_centered_device(int value, int modulus) {
  int residue = value % modulus;
  residue += modulus & -static_cast<int>(residue < 0);
  const int threshold = (modulus + 1) / 2;
  residue -= modulus & -static_cast<int>(residue >= threshold);
  return static_cast<int8_t>(residue);
}

__device__ uint32_t rns8_canonical_from_centered_device(int8_t residue, int modulus) {
  const int value = static_cast<int>(residue);
  return static_cast<uint32_t>(value + (modulus & -static_cast<int>(value < 0)));
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

struct rns8_u192_device {
  uint64_t limb0;
  uint64_t limb1;
  uint64_t limb2;
};

__device__ rns8_u192_device rns8_u192_from_u64_device(uint64_t value) {
  return {value, 0, 0};
}

__device__ int rns8_u192_compare_device(rns8_u192_device a, rns8_u192_device b) {
  if (a.limb2 != b.limb2) {
    return a.limb2 > b.limb2 ? 1 : -1;
  }
  if (a.limb1 != b.limb1) {
    return a.limb1 > b.limb1 ? 1 : -1;
  }
  if (a.limb0 != b.limb0) {
    return a.limb0 > b.limb0 ? 1 : -1;
  }
  return 0;
}

__device__ rns8_u192_device rns8_u192_add_device(rns8_u192_device a, rns8_u192_device b) {
  rns8_u192_device out{};
  out.limb0 = a.limb0 + b.limb0;
  const uint64_t carry0 = out.limb0 < a.limb0 ? 1 : 0;
  out.limb1 = a.limb1 + b.limb1 + carry0;
  const uint64_t carry1 = out.limb1 < a.limb1 || (carry0 != 0 && out.limb1 == a.limb1) ? 1 : 0;
  out.limb2 = a.limb2 + b.limb2 + carry1;
  return out;
}

__device__ rns8_u192_device rns8_u192_sub_device(rns8_u192_device a, rns8_u192_device b) {
  rns8_u192_device out{};
  out.limb0 = a.limb0 - b.limb0;
  const uint64_t borrow0 = a.limb0 < b.limb0 ? 1 : 0;
  out.limb1 = a.limb1 - b.limb1 - borrow0;
  const uint64_t borrow1 = a.limb1 < b.limb1 || (borrow0 != 0 && a.limb1 == b.limb1) ? 1 : 0;
  out.limb2 = a.limb2 - b.limb2 - borrow1;
  return out;
}

__device__ rns8_u192_device rns8_u192_mul_u32_device(rns8_u192_device value, uint32_t multiplier) {
  rns8_u192_device out{};
  unsigned __int128 acc = static_cast<unsigned __int128>(value.limb0) * multiplier;
  out.limb0 = static_cast<uint64_t>(acc);
  uint64_t carry = static_cast<uint64_t>(acc >> 64u);
  acc = static_cast<unsigned __int128>(value.limb1) * multiplier + carry;
  out.limb1 = static_cast<uint64_t>(acc);
  carry = static_cast<uint64_t>(acc >> 64u);
  acc = static_cast<unsigned __int128>(value.limb2) * multiplier + carry;
  out.limb2 = static_cast<uint64_t>(acc);
  return out;
}

__device__ uint32_t rns8_u192_mod_u32_device(rns8_u192_device value, uint32_t modulus) {
  unsigned __int128 rem = 0;
  rem = ((rem << 64u) | value.limb2) % modulus;
  rem = ((rem << 64u) | value.limb1) % modulus;
  rem = ((rem << 64u) | value.limb0) % modulus;
  return static_cast<uint32_t>(rem);
}

__device__ rns8_u192_device rns8_u192_shr1_device(rns8_u192_device value) {
  return {
      (value.limb0 >> 1u) | (value.limb1 << 63u),
      (value.limb1 >> 1u) | (value.limb2 << 63u),
      value.limb2 >> 1u};
}

__device__ bool rns8_u192_gt_u64_device(rns8_u192_device value, uint64_t limit) {
  return value.limb2 != 0 || value.limb1 != 0 || value.limb0 > limit;
}

__device__ uint64_t rns8_u192_limb_device(rns8_u192_device value, int limb) {
  if (limb == 0) {
    return value.limb0;
  }
  if (limb == 1) {
    return value.limb1;
  }
  if (limb == 2) {
    return value.limb2;
  }
  return 0;
}

__device__ bool rns8_u192_unsigned_fits_limbs_device(rns8_u192_device value, int limb_count) {
  if (limb_count >= 3) {
    return true;
  }
  if (limb_count == 2) {
    return value.limb2 == 0;
  }
  return value.limb2 == 0 && value.limb1 == 0;
}

__device__ bool rns8_u192_signed_positive_fits_limbs_device(rns8_u192_device value, int limb_count) {
  constexpr uint64_t sign_bit_limit = 0x7fffffffffffffffULL;
  if (limb_count >= 4) {
    return true;
  }
  if (limb_count == 3) {
    return value.limb2 <= sign_bit_limit;
  }
  if (limb_count == 2) {
    return value.limb2 == 0 && value.limb1 <= sign_bit_limit;
  }
  return value.limb2 == 0 && value.limb1 == 0 && value.limb0 <= sign_bit_limit;
}

__device__ bool rns8_u192_signed_negative_magnitude_fits_limbs_device(rns8_u192_device magnitude, int limb_count) {
  constexpr uint64_t sign_bit = 0x8000000000000000ULL;
  if (limb_count >= 4) {
    return true;
  }
  const int sign_limb = limb_count - 1;
  for (int limb = limb_count; limb < 3; ++limb) {
    if (rns8_u192_limb_device(magnitude, limb) != 0) {
      return false;
    }
  }
  const uint64_t high = rns8_u192_limb_device(magnitude, sign_limb);
  if (high < sign_bit) {
    return true;
  }
  if (high > sign_bit) {
    return false;
  }
  for (int limb = 0; limb < sign_limb; ++limb) {
    if (rns8_u192_limb_device(magnitude, limb) != 0) {
      return false;
    }
  }
  return true;
}

__device__ void rns8_store_u192_unsigned_limbs_device(uint64_t* dst, rns8_u192_device value, int limb_count) {
  for (int limb = 0; limb < limb_count; ++limb) {
    dst[limb] = rns8_u192_limb_device(value, limb);
  }
}

__device__ void rns8_store_u192_negative_twos_complement_limbs_device(
    uint64_t* dst,
    rns8_u192_device magnitude,
    int limb_count) {
  uint64_t carry = 1;
  for (int limb = 0; limb < limb_count; ++limb) {
    const uint64_t inverted = ~rns8_u192_limb_device(magnitude, limb);
    const uint64_t value = inverted + carry;
    carry = value < inverted ? 1 : 0;
    dst[limb] = value;
  }
}

__device__ void rns8_reconstruct_canonical_wide_device(
    const int8_t* residues,
    int cell,
    int elements,
    int prefix,
    rns8_u192_device* out_x,
    rns8_u192_device* out_product) {
  rns8_u192_device x = rns8_u192_from_u64_device(0);
  rns8_u192_device product = rns8_u192_from_u64_device(1);
  for (int i = 0; i < prefix; ++i) {
    const int modulus = rns8_default_moduli_device[i];
    const uint32_t target =
        rns8_canonical_from_centered_device(residues[static_cast<int64_t>(i) * elements + cell], modulus);
    const uint32_t x_mod = rns8_u192_mod_u32_device(x, static_cast<uint32_t>(modulus));
    const uint32_t product_mod = rns8_u192_mod_u32_device(product, static_cast<uint32_t>(modulus));
    const uint32_t inverse = rns8_mod_inverse_device(product_mod, static_cast<uint32_t>(modulus));
    int64_t delta_mod = static_cast<int64_t>(target) - static_cast<int64_t>(x_mod);
    delta_mod %= modulus;
    delta_mod += static_cast<int64_t>(modulus) & -static_cast<int64_t>(delta_mod < 0);
    const uint32_t coefficient = static_cast<uint32_t>(
        (static_cast<uint64_t>(delta_mod) * static_cast<uint64_t>(inverse)) % static_cast<uint32_t>(modulus));
    x = rns8_u192_add_device(x, rns8_u192_mul_u32_device(product, coefficient));
    product = rns8_u192_mul_u32_device(product, static_cast<uint32_t>(modulus));
  }
  *out_x = x;
  *out_product = product;
}

__device__ int8_t rns8_center_i64_device(int64_t value, int modulus) {
  int64_t residue = value % static_cast<int64_t>(modulus);
  residue += static_cast<int64_t>(modulus) & -static_cast<int64_t>(residue < 0);
  const int64_t threshold = (static_cast<int64_t>(modulus) + 1) / 2;
  residue -= static_cast<int64_t>(modulus) & -static_cast<int64_t>(residue >= threshold);
  return static_cast<int8_t>(residue);
}

__device__ int8_t rns8_center_u64_device(uint64_t value, int modulus) {
  uint64_t residue = value % static_cast<uint64_t>(modulus);
  const uint64_t threshold = (static_cast<uint64_t>(modulus) + 1) / 2;
  residue -= static_cast<uint64_t>(modulus) & (0u - static_cast<uint64_t>(residue >= threshold));
  return static_cast<int8_t>(static_cast<int64_t>(residue));
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
  __shared__ int8_t a_tile[kRns8HipTileM][kRns8HipTileK];
  __shared__ int8_t b_tile[kRns8HipTileK][kRns8HipTileN];

  const int thread_row = static_cast<int>(threadIdx.y);
  const int thread_col = static_cast<int>(threadIdx.x);
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8HipTileM;
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8HipTileN;
  const int row = tile_row + thread_row;
  const int col = tile_col + thread_col;
  const int lane = thread_row * static_cast<int>(blockDim.x) + thread_col;
  const int block_threads = static_cast<int>(blockDim.x * blockDim.y);
  const bool output_active = row < m && col < n;
  int acc = 0;

  for (int tile_k = 0; tile_k < k_block; tile_k += kRns8HipTileK) {
    const int tile_extent =
        k_block - tile_k < kRns8HipTileK ? k_block - tile_k : kRns8HipTileK;

    for (int index = lane; index < kRns8HipTileM * kRns8HipTileK; index += block_threads) {
      const int local_row = index / kRns8HipTileK;
      const int local_k = index - local_row * kRns8HipTileK;
      const int global_row = tile_row + local_row;
      const int source_k = k_offset + tile_k + local_k;
      a_tile[local_row][local_k] =
          global_row < m && local_k < tile_extent ? A[global_row * lda + source_k] : 0;
    }

    for (int index = lane; index < kRns8HipTileK * kRns8HipTileN; index += block_threads) {
      const int local_k = index / kRns8HipTileN;
      const int local_col = index - local_k * kRns8HipTileN;
      const int global_col = tile_col + local_col;
      const int source_k = k_offset + tile_k + local_k;
      b_tile[local_k][local_col] =
          local_k < tile_extent && global_col < n ? B[source_k * ldb + global_col] : 0;
    }

    __syncthreads();

    if (output_active) {
      for (int kk = 0; kk < tile_extent; ++kk) {
        acc += static_cast<int>(a_tile[thread_row][kk]) * static_cast<int>(b_tile[kk][thread_col]);
      }
    }

    __syncthreads();
  }

  if (output_active && accumulate) {
    acc += static_cast<int>(C[row * ldc + col]);
  }
  if (output_active) {
    C[row * ldc + col] = rns8_reduce_to_centered_device(acc, modulus);
  }
}

__device__ void rns8_export_i64_cell_device(
    const int8_t* residues,
    int64_t* dst,
    int cell,
    int elements,
    int prefix,
    uint64_t bound,
    int* status) {
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_device(residues, cell, elements, prefix, &x, &product);
  const rns8_u192_device half_product = rns8_u192_shr1_device(product);

  if (rns8_u192_compare_device(x, half_product) > 0) {
    const rns8_u192_device magnitude = rns8_u192_sub_device(product, x);
    constexpr uint64_t int64_min_magnitude = 0x8000000000000000ULL;
    if (rns8_u192_gt_u64_device(magnitude, bound) || rns8_u192_gt_u64_device(magnitude, int64_min_magnitude)) {
      atomicCAS(status, 0, 5);
      return;
    }
    if (magnitude.limb0 == int64_min_magnitude) {
      dst[cell] = (-9223372036854775807LL - 1LL);
      return;
    }
    dst[cell] = -static_cast<int64_t>(magnitude.limb0);
    return;
  }

  if (rns8_u192_gt_u64_device(x, bound) || rns8_u192_gt_u64_device(x, 0x7fffffffffffffffULL)) {
    atomicCAS(status, 0, 5);
    return;
  }
  dst[cell] = static_cast<int64_t>(x.limb0);
}

__device__ void rns8_export_u64_cell_device(
    const int8_t* residues,
    uint64_t* dst,
    int cell,
    int elements,
    int prefix,
    uint64_t bound,
    int* status) {
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_device(residues, cell, elements, prefix, &x, &product);
  if (rns8_u192_gt_u64_device(x, bound)) {
    atomicCAS(status, 0, 5);
    return;
  }
  dst[cell] = x.limb0;
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
  rns8_export_i64_cell_device(residues, dst, cell, elements, prefix, bound, status);
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
  rns8_export_u64_cell_device(residues, dst, cell, elements, prefix, bound, status);
}

__global__ void rns8_export_i64_tile_kernel(
    const int8_t* residues,
    int64_t* dst,
    int rows,
    int cols,
    int row_offset,
    int col_offset,
    int row_extent,
    int col_extent,
    int prefix,
    uint64_t bound,
    int* status) {
  const int tile_cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int tile_elements = row_extent * col_extent;
  if (tile_cell >= tile_elements) {
    return;
  }
  const int local_row = tile_cell / col_extent;
  const int local_col = tile_cell - local_row * col_extent;
  const int global_cell = (row_offset + local_row) * cols + (col_offset + local_col);
  rns8_export_i64_cell_device(residues, dst, global_cell, rows * cols, prefix, bound, status);
}

__global__ void rns8_export_u64_tile_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int row_offset,
    int col_offset,
    int row_extent,
    int col_extent,
    int prefix,
    uint64_t bound,
    int* status) {
  const int tile_cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int tile_elements = row_extent * col_extent;
  if (tile_cell >= tile_elements) {
    return;
  }
  const int local_row = tile_cell / col_extent;
  const int local_col = tile_cell - local_row * col_extent;
  const int global_cell = (row_offset + local_row) * cols + (col_offset + local_col);
  rns8_export_u64_cell_device(residues, dst, global_cell, rows * cols, prefix, bound, status);
}

__global__ void rns8_export_exact_wide_unsigned_limbs_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_device(residues, cell, elements, prefix, &x, &product);
  if (!rns8_u192_unsigned_fits_limbs_device(x, limb_count)) {
    atomicCAS(status, 0, 5);
    return;
  }
  rns8_store_u192_unsigned_limbs_device(dst + static_cast<int64_t>(cell) * limb_count, x, limb_count);
}

__global__ void rns8_export_exact_wide_signed_limbs_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int prefix,
    int limb_count,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_device(residues, cell, elements, prefix, &x, &product);
  const rns8_u192_device half_product = rns8_u192_shr1_device(product);
  uint64_t* cell_dst = dst + static_cast<int64_t>(cell) * limb_count;
  if (rns8_u192_compare_device(x, half_product) > 0) {
    const rns8_u192_device magnitude = rns8_u192_sub_device(product, x);
    if (!rns8_u192_signed_negative_magnitude_fits_limbs_device(magnitude, limb_count)) {
      atomicCAS(status, 0, 5);
      return;
    }
    rns8_store_u192_negative_twos_complement_limbs_device(cell_dst, magnitude, limb_count);
    return;
  }

  if (!rns8_u192_signed_positive_fits_limbs_device(x, limb_count)) {
    atomicCAS(status, 0, 5);
    return;
  }
  rns8_store_u192_unsigned_limbs_device(cell_dst, x, limb_count);
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
    const dim3 block(kRns8HipTileN, kRns8HipTileM);
    const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    int k_offset = 0;
    int accumulate = 0;
    while (k_offset < k && err == hipSuccess) {
      const int k_remaining = k - k_offset;
      const int k_block = k_remaining > kRns8SafeInt32KBlock ? kRns8SafeInt32KBlock : k_remaining;
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
    int modulus,
    int modulus_index,
    int selected_prefix,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || modulus <= 0 ||
      modulus_index < 0 || selected_prefix <= 0 || modulus_index >= selected_prefix ||
      selected_prefix > kRns8DefaultModulusCount || safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
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

extern "C" int rns8_hip_direct_export_i64_tile_device(
    const int8_t* d_residues,
    int64_t* d_dst,
    int rows,
    int cols,
    int row_offset,
    int col_offset,
    int row_extent,
    int col_extent,
    int prefix,
    uint64_t bound,
    int* d_status) {
  const int threads = 256;
  const int elements = row_extent * col_extent;
  const int blocks = (elements + threads - 1) / threads;
  hipLaunchKernelGGL(
      rns8_export_i64_tile_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      nullptr,
      d_residues,
      d_dst,
      rows,
      cols,
      row_offset,
      col_offset,
      row_extent,
      col_extent,
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

extern "C" int rns8_hip_direct_export_u64_tile_device(
    const int8_t* d_residues,
    uint64_t* d_dst,
    int rows,
    int cols,
    int row_offset,
    int col_offset,
    int row_extent,
    int col_extent,
    int prefix,
    uint64_t bound,
    int* d_status) {
  const int threads = 256;
  const int elements = row_extent * col_extent;
  const int blocks = (elements + threads - 1) / threads;
  hipLaunchKernelGGL(
      rns8_export_u64_tile_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      nullptr,
      d_residues,
      d_dst,
      rows,
      cols,
      row_offset,
      col_offset,
      row_extent,
      col_extent,
      prefix,
      bound,
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
  return static_cast<int>(hipGetLastError());
}
