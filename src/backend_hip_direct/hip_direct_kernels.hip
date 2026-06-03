#include <hip/hip_runtime.h>

#include <cstdint>

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

__device__ __constant__ int rns8_default_moduli_device[kRns8DefaultModulusCount] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

constexpr int kRns8HipTileM = 16;
constexpr int kRns8HipTileN = 16;
constexpr int kRns8HipTileK = 64;
constexpr int kRns8HipTileNPadded = kRns8HipTileN + 1;
constexpr int kRns8SafeInt32KBlock = 65536;
constexpr int kRns8MaxInt = 2147483647;
constexpr int kRns8DefaultBoundedPrefix = 9;
constexpr int kRns8MaxSupportedPrefix = 20;

static_assert(
    kRns8HipTileM * kRns8HipTileN == 256,
    "bounded direct HIP GEMM uses one 16x16 output tile per block");

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

__device__ uint32_t rns8_reduce_u32_mod255_device(uint32_t value) {
  uint32_t residue =
      (value & 0xffu) + ((value >> 8u) & 0xffu) + ((value >> 16u) & 0xffu) + ((value >> 24u) & 0xffu);
  residue = (residue & 0xffu) + (residue >> 8u);
  residue -= 255u & (0u - static_cast<uint32_t>(residue >= 255u));
  return residue;
}

__device__ uint32_t rns8_reduce_u32_mod251_device(uint32_t value) {
  uint32_t residue = (value & 0xffu) + 5u * ((value >> 8u) & 0xffu) +
                     25u * ((value >> 16u) & 0xffu) + 125u * ((value >> 24u) & 0xffu);
  residue = (residue & 0xffu) + 5u * (residue >> 8u);
  residue -= 251u & (0u - static_cast<uint32_t>(residue >= 251u));
  residue -= 251u & (0u - static_cast<uint32_t>(residue >= 251u));
  residue -= 251u & (0u - static_cast<uint32_t>(residue >= 251u));
  residue -= 251u & (0u - static_cast<uint32_t>(residue >= 251u));
  return residue;
}

__device__ int8_t rns8_reduce_to_centered_mod256_device(int value) {
  return static_cast<int8_t>(value & 0xff);
}

__device__ int8_t rns8_reduce_to_centered_mod255_device(int value) {
  const uint32_t magnitude =
      value < 0 ? static_cast<uint32_t>(-static_cast<int64_t>(value)) : static_cast<uint32_t>(value);
  uint32_t residue = rns8_reduce_u32_mod255_device(magnitude);
  const uint32_t negative_nonzero =
      static_cast<uint32_t>(value < 0) & static_cast<uint32_t>(residue != 0);
  const uint32_t negative_mask = 0u - negative_nonzero;
  residue = (residue & ~negative_mask) | ((255u - residue) & negative_mask);
  residue -= 255u & (0u - static_cast<uint32_t>(residue >= 128u));
  return static_cast<int8_t>(static_cast<int32_t>(residue));
}

__device__ int8_t rns8_reduce_to_centered_mod251_device(int value) {
  const uint32_t magnitude =
      value < 0 ? static_cast<uint32_t>(-static_cast<int64_t>(value)) : static_cast<uint32_t>(value);
  uint32_t residue = rns8_reduce_u32_mod251_device(magnitude);
  const uint32_t negative_nonzero =
      static_cast<uint32_t>(value < 0) & static_cast<uint32_t>(residue != 0);
  const uint32_t negative_mask = 0u - negative_nonzero;
  residue = (residue & ~negative_mask) | ((251u - residue) & negative_mask);
  residue -= 251u & (0u - static_cast<uint32_t>(residue >= 126u));
  return static_cast<int8_t>(static_cast<int32_t>(residue));
}

__device__ int8_t rns8_reduce_to_centered_device(int value, int modulus, uint32_t reciprocal) {
  if (modulus == 256) {
    return rns8_reduce_to_centered_mod256_device(value);
  }
  if (modulus == 255) {
    return rns8_reduce_to_centered_mod255_device(value);
  }
  if (modulus == 251) {
    return rns8_reduce_to_centered_mod251_device(value);
  }
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

__device__ bool rns8_u192_centered_is_negative_device(rns8_u192_device value, rns8_u192_device product) {
  rns8_u192_device threshold = rns8_u192_shr1_device(product);
  if ((product.limb0 & 1u) != 0) {
    threshold = rns8_u192_add_device(threshold, rns8_u192_from_u64_device(1));
  }
  return rns8_u192_compare_device(value, threshold) >= 0;
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

template <int Prefix>
__device__ void rns8_reconstruct_canonical_wide_fixed_prefix_device(
    const int8_t* residues,
    int cell,
    int elements,
    rns8_u192_device* out_x,
    rns8_u192_device* out_product) {
  rns8_u192_device x = rns8_u192_from_u64_device(0);
  rns8_u192_device product = rns8_u192_from_u64_device(1);
#pragma unroll
  for (int i = 0; i < Prefix; ++i) {
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

__device__ int8_t rns8_center_u8_device(uint8_t value, int modulus) {
  if (modulus == 256) {
    return static_cast<int8_t>(value);
  }
  if (modulus == 255) {
    return rns8_reduce_to_centered_mod255_device(static_cast<int>(value));
  }
  if (modulus == 251) {
    return rns8_reduce_to_centered_mod251_device(static_cast<int>(value));
  }
  int residue = static_cast<int>(value) % modulus;
  const int threshold = (modulus + 1) / 2;
  residue -= modulus & -static_cast<int>(residue >= threshold);
  return static_cast<int8_t>(residue);
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

__global__ void rns8_pack_u8_modulus_kernel(
    const uint8_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld,
    int modulus) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  const int row = static_cast<int>(idx / cols);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  residues[idx] = rns8_center_u8_device(src[static_cast<int64_t>(row) * ld + col], modulus);
}

__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN) rns8_ring_gemm_i8_i32_tiled_kernel(
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
    uint32_t modulus_reciprocal,
    int accumulate) {
  __shared__ int8_t a_tile[kRns8HipTileM][kRns8HipTileK];
  __shared__ int8_t b_tile[kRns8HipTileK][kRns8HipTileNPadded];

  const int thread_row = static_cast<int>(threadIdx.y);
  const int thread_col = static_cast<int>(threadIdx.x);
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8HipTileM;
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8HipTileN;
  const int row = tile_row + thread_row;
  const int col = tile_col + thread_col;
  const int lane = thread_row * static_cast<int>(blockDim.x) + thread_col;
  const int block_threads = static_cast<int>(blockDim.x * blockDim.y);
  const bool output_active = row < m && col < n;
  int32_t acc = 0;

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
        acc += static_cast<int32_t>(a_tile[thread_row][kk]) * static_cast<int32_t>(b_tile[kk][thread_col]);
      }
    }

    __syncthreads();
  }

  if (output_active && accumulate) {
    acc += static_cast<int32_t>(C[row * ldc + col]);
  }
  if (output_active) {
    C[row * ldc + col] = rns8_reduce_to_centered_device(acc, modulus, modulus_reciprocal);
  }
}

__global__ void __launch_bounds__(kRns8HipTileM * kRns8HipTileN) rns8_ring_gemm_i8_i32_scheduled_kernel(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    const rns8_plan_tile_schedule_entry* schedule,
    int entry_count,
    int k_offset,
    int k_block,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int modulus_index,
    int accumulate) {
  const int entry_index = static_cast<int>(blockIdx.z);
  if (entry_index >= entry_count) {
    return;
  }
  const rns8_plan_tile_schedule_entry entry = schedule[entry_index];
  if (modulus_index >= static_cast<int>(entry.selected_prefix)) {
    return;
  }

  __shared__ int8_t a_tile[kRns8HipTileM][kRns8HipTileK];
  __shared__ int8_t b_tile[kRns8HipTileK][kRns8HipTileNPadded];

  const int thread_row = static_cast<int>(threadIdx.y);
  const int thread_col = static_cast<int>(threadIdx.x);
  const int tile_local_row = static_cast<int>(blockIdx.y) * kRns8HipTileM;
  const int tile_local_col = static_cast<int>(blockIdx.x) * kRns8HipTileN;
  const int row = static_cast<int>(entry.row_offset) + tile_local_row + thread_row;
  const int col = static_cast<int>(entry.col_offset) + tile_local_col + thread_col;
  const int lane = thread_row * static_cast<int>(blockDim.x) + thread_col;
  const int block_threads = static_cast<int>(blockDim.x * blockDim.y);
  const bool output_active =
      tile_local_row + thread_row < static_cast<int>(entry.row_extent) &&
      tile_local_col + thread_col < static_cast<int>(entry.col_extent);
  int32_t acc = 0;

  for (int tile_k = 0; tile_k < k_block; tile_k += kRns8HipTileK) {
    const int tile_extent =
        k_block - tile_k < kRns8HipTileK ? k_block - tile_k : kRns8HipTileK;

    for (int index = lane; index < kRns8HipTileM * kRns8HipTileK; index += block_threads) {
      const int local_row = index / kRns8HipTileK;
      const int local_k = index - local_row * kRns8HipTileK;
      const int local_output_row = tile_local_row + local_row;
      const int global_row = static_cast<int>(entry.row_offset) + local_output_row;
      const int source_k = k_offset + tile_k + local_k;
      a_tile[local_row][local_k] =
          local_output_row < static_cast<int>(entry.row_extent) && local_k < tile_extent
              ? A[global_row * lda + source_k]
              : 0;
    }

    for (int index = lane; index < kRns8HipTileK * kRns8HipTileN; index += block_threads) {
      const int local_k = index / kRns8HipTileN;
      const int local_col = index - local_k * kRns8HipTileN;
      const int local_output_col = tile_local_col + local_col;
      const int global_col = static_cast<int>(entry.col_offset) + local_output_col;
      const int source_k = k_offset + tile_k + local_k;
      b_tile[local_k][local_col] =
          local_k < tile_extent && local_output_col < static_cast<int>(entry.col_extent)
              ? B[source_k * ldb + global_col]
              : 0;
    }

    __syncthreads();

    if (output_active) {
      for (int kk = 0; kk < tile_extent; ++kk) {
        acc += static_cast<int32_t>(a_tile[thread_row][kk]) * static_cast<int32_t>(b_tile[kk][thread_col]);
      }
    }

    __syncthreads();
  }

  if (output_active && accumulate) {
    acc += static_cast<int32_t>(C[row * ldc + col]);
  }
  if (output_active) {
    C[row * ldc + col] = rns8_reduce_to_centered_device(acc, modulus, modulus_reciprocal);
  }
}

__global__ void rns8_export_u8_modulus_kernel(
    const int8_t* residues,
    uint8_t* dst,
    int rows,
    int cols,
    int ld,
    int modulus) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  const int row = static_cast<int>(idx / cols);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  dst[static_cast<int64_t>(row) * ld + col] =
      static_cast<uint8_t>(rns8_canonical_from_centered_device(residues[idx], modulus));
}

__device__ void rns8_export_i64_cell_device(
    int64_t* dst,
    int cell,
    rns8_u192_device x,
    rns8_u192_device product,
    uint64_t bound,
    int* status) {
  if (rns8_u192_centered_is_negative_device(x, product)) {
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
    uint64_t* dst,
    int cell,
    rns8_u192_device x,
    uint64_t bound,
    int* status) {
  if (rns8_u192_gt_u64_device(x, bound)) {
    atomicCAS(status, 0, 5);
    return;
  }
  dst[cell] = x.limb0;
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
  rns8_export_i64_cell_device(dst, cell, x, product, bound, status);
}

template <int Prefix>
__device__ void rns8_export_i64_fixed_prefix_cell_device(
    const int8_t* residues,
    int64_t* dst,
    int cell,
    int elements,
    uint64_t bound,
    int* status) {
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_i64_cell_device(dst, cell, x, product, bound, status);
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
  rns8_export_u64_cell_device(dst, cell, x, bound, status);
}

template <int Prefix>
__device__ void rns8_export_u64_fixed_prefix_cell_device(
    const int8_t* residues,
    uint64_t* dst,
    int cell,
    int elements,
    uint64_t bound,
    int* status) {
  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_u64_cell_device(dst, cell, x, bound, status);
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

template <int Prefix>
__global__ void rns8_export_i64_fixed_prefix_kernel(
    const int8_t* residues,
    int64_t* dst,
    int rows,
    int cols,
    uint64_t bound,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }
  rns8_export_i64_fixed_prefix_cell_device<Prefix>(residues, dst, cell, elements, bound, status);
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

template <int Prefix>
__global__ void rns8_export_u64_fixed_prefix_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    uint64_t bound,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }
  rns8_export_u64_fixed_prefix_cell_device<Prefix>(residues, dst, cell, elements, bound, status);
}

__global__ void rns8_export_i64_scheduled_kernel(
    const int8_t* residues,
    int64_t* dst,
    const rns8_plan_tile_schedule_entry* schedule,
    const uint64_t* bounds,
    int entry_count,
    int max_tile_elements,
    int rows,
    int cols,
    int* status) {
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
  const int row_extent = static_cast<int>(entry.row_extent);
  const int col_extent = static_cast<int>(entry.col_extent);
  const int tile_elements = row_extent * col_extent;
  if (tile_cell >= tile_elements) {
    return;
  }
  const int local_row = tile_cell / col_extent;
  const int local_col = tile_cell - local_row * col_extent;
  const int global_cell =
      (static_cast<int>(entry.row_offset) + local_row) * cols + (static_cast<int>(entry.col_offset) + local_col);
  rns8_export_i64_cell_device(
      residues, dst, global_cell, rows * cols, static_cast<int>(entry.selected_prefix), bounds[entry_index], status);
}

__global__ void rns8_export_u64_scheduled_kernel(
    const int8_t* residues,
    uint64_t* dst,
    const rns8_plan_tile_schedule_entry* schedule,
    const uint64_t* bounds,
    int entry_count,
    int max_tile_elements,
    int rows,
    int cols,
    int* status) {
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
  const int row_extent = static_cast<int>(entry.row_extent);
  const int col_extent = static_cast<int>(entry.col_extent);
  const int tile_elements = row_extent * col_extent;
  if (tile_cell >= tile_elements) {
    return;
  }
  const int local_row = tile_cell / col_extent;
  const int local_col = tile_cell - local_row * col_extent;
  const int global_cell =
      (static_cast<int>(entry.row_offset) + local_row) * cols + (static_cast<int>(entry.col_offset) + local_col);
  rns8_export_u64_cell_device(
      residues, dst, global_cell, rows * cols, static_cast<int>(entry.selected_prefix), bounds[entry_index], status);
}

__device__ void rns8_export_exact_wide_unsigned_limbs_device(
    uint64_t* dst,
    int cell,
    rns8_u192_device x,
    int limb_count,
    int* status) {
  if (!rns8_u192_unsigned_fits_limbs_device(x, limb_count)) {
    atomicCAS(status, 0, 5);
    return;
  }
  rns8_store_u192_unsigned_limbs_device(dst + static_cast<int64_t>(cell) * limb_count, x, limb_count);
}

__device__ void rns8_export_exact_wide_signed_limbs_device(
    uint64_t* dst,
    int cell,
    rns8_u192_device x,
    rns8_u192_device product,
    int limb_count,
    int* status);

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
  rns8_export_exact_wide_signed_limbs_device(dst, cell, x, product, limb_count, status);
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
  rns8_export_exact_wide_unsigned_limbs_device(dst, cell, x, limb_count, status);
}

template <int Prefix>
__global__ void rns8_export_exact_wide_unsigned_limbs_fixed_prefix_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int limb_count,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_unsigned_limbs_device(dst, cell, x, limb_count, status);
}

__device__ void rns8_export_exact_wide_signed_limbs_device(
    uint64_t* dst,
    int cell,
    rns8_u192_device x,
    rns8_u192_device product,
    int limb_count,
    int* status) {
  uint64_t* cell_dst = dst + static_cast<int64_t>(cell) * limb_count;
  if (rns8_u192_centered_is_negative_device(x, product)) {
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

template <int Prefix>
__global__ void rns8_export_exact_wide_signed_limbs_fixed_prefix_kernel(
    const int8_t* residues,
    uint64_t* dst,
    int rows,
    int cols,
    int limb_count,
    int* status) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = rows * cols;
  if (cell >= elements) {
    return;
  }

  rns8_u192_device x{};
  rns8_u192_device product{};
  rns8_reconstruct_canonical_wide_fixed_prefix_device<Prefix>(residues, cell, elements, &x, &product);
  rns8_export_exact_wide_signed_limbs_device(dst, cell, x, product, limb_count, status);
}

}  // namespace

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

extern "C" int rns8_hip_direct_ring_gemm_i8_scheduled_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    const rns8_plan_tile_schedule_entry* d_schedule,
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
