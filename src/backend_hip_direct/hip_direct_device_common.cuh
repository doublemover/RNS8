__device__ __constant__ int rns8_default_moduli_device[kRns8DefaultModulusCount] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

constexpr int kRns8HipTileM = 16;
constexpr int kRns8HipTileN = 16;
constexpr int kRns8HipTileK = 64;
constexpr int kRns8HipTileNPadded = kRns8HipTileN + 1;
constexpr int kRns8SafeInt32KBlock = 65536;
constexpr int kRns8MaxInt = 2147483647;
constexpr int64_t kRns8MaxInt64 = 0x7fffffffffffffffLL;
constexpr int kRns8DefaultBoundedPrefix = 9;
constexpr int kRns8MaxSupportedPrefix = 20;
constexpr uint32_t kRns8TileScheduleZeroOutput = 0x00000001u;
constexpr uint32_t kRns8TileScheduleZeroRowColProduct = 0x00000002u;

static_assert(
    kRns8HipTileM * kRns8HipTileN == 256,
    "bounded direct HIP GEMM uses one 16x16 output tile per block");

constexpr uint64_t kRns8ReciprocalScale = 1ull << 32u;

uint32_t rns8_modulus_reciprocal_host(int modulus) {
  return static_cast<uint32_t>(kRns8ReciprocalScale / static_cast<uint32_t>(modulus));
}

__device__ uint32_t rns8_reduce_u32_small_modulus_device(uint32_t value, uint32_t modulus, uint32_t reciprocal) {
  return rns8::detail::finite_u8::reduce_u32_small_modulus(value, modulus, reciprocal);
}

__device__ uint32_t rns8_reduce_u32_mod255_device(uint32_t value) {
  return rns8::detail::finite_u8::reduce_u32_mod255(value);
}

__device__ uint32_t rns8_reduce_u32_mod251_device(uint32_t value) {
  return rns8::detail::finite_u8::reduce_u32_mod251(value);
}

__device__ int8_t rns8_reduce_to_centered_mod256_device(int value) {
  return rns8::detail::finite_u8::reduce_to_centered_mod256_i32(value);
}

__device__ int8_t rns8_reduce_to_centered_mod255_device(int value) {
  return rns8::detail::finite_u8::reduce_to_centered_mod255_i32(value);
}

__device__ int8_t rns8_reduce_to_centered_mod251_device(int value) {
  return rns8::detail::finite_u8::reduce_to_centered_mod251_i32(value);
}

__device__ int8_t rns8_reduce_to_centered_device(int value, int modulus, uint32_t reciprocal) {
  return rns8::detail::finite_u8::reduce_to_centered_i32(value, static_cast<uint32_t>(modulus), reciprocal);
}

template <int Modulus>
__device__ int8_t rns8_reduce_to_centered_fixed_modulus_device(int value) {
  return rns8::detail::finite_u8::reduce_to_centered_fixed_i32<static_cast<uint32_t>(Modulus)>(value);
}

__device__ int8_t rns8_reduce_to_centered_default_modulus_fixed_device(
    int value,
    int modulus_index,
    int fallback_modulus,
    uint32_t fallback_reciprocal) {
  switch (modulus_index) {
    case 0:
      return rns8_reduce_to_centered_fixed_modulus_device<256>(value);
    case 1:
      return rns8_reduce_to_centered_fixed_modulus_device<255>(value);
    case 2:
      return rns8_reduce_to_centered_fixed_modulus_device<253>(value);
    case 3:
      return rns8_reduce_to_centered_fixed_modulus_device<251>(value);
    case 4:
      return rns8_reduce_to_centered_fixed_modulus_device<247>(value);
    case 5:
      return rns8_reduce_to_centered_fixed_modulus_device<239>(value);
    case 6:
      return rns8_reduce_to_centered_fixed_modulus_device<233>(value);
    case 7:
      return rns8_reduce_to_centered_fixed_modulus_device<229>(value);
    case 8:
      return rns8_reduce_to_centered_fixed_modulus_device<227>(value);
    case 9:
      return rns8_reduce_to_centered_fixed_modulus_device<223>(value);
    case 10:
      return rns8_reduce_to_centered_fixed_modulus_device<217>(value);
    case 11:
      return rns8_reduce_to_centered_fixed_modulus_device<211>(value);
    case 12:
      return rns8_reduce_to_centered_fixed_modulus_device<199>(value);
    case 13:
      return rns8_reduce_to_centered_fixed_modulus_device<197>(value);
    case 14:
      return rns8_reduce_to_centered_fixed_modulus_device<193>(value);
    case 15:
      return rns8_reduce_to_centered_fixed_modulus_device<191>(value);
    case 16:
      return rns8_reduce_to_centered_fixed_modulus_device<181>(value);
    case 17:
      return rns8_reduce_to_centered_fixed_modulus_device<179>(value);
    case 18:
      return rns8_reduce_to_centered_fixed_modulus_device<173>(value);
    case 19:
      return rns8_reduce_to_centered_fixed_modulus_device<167>(value);
    default:
      return rns8_reduce_to_centered_device(value, fallback_modulus, fallback_reciprocal);
  }
}

__device__ uint32_t rns8_canonical_from_centered_device(int8_t residue, int modulus) {
  return rns8::detail::finite_u8::canonical_from_centered(residue, static_cast<uint32_t>(modulus));
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

template <int LimbCount>
__device__ bool rns8_u192_unsigned_fits_fixed_limbs_device(rns8_u192_device value) {
  if (LimbCount >= 3) {
    return true;
  }
  if (LimbCount == 2) {
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

template <int LimbCount>
__device__ bool rns8_u192_signed_positive_fits_fixed_limbs_device(rns8_u192_device value) {
  constexpr uint64_t sign_bit_limit = 0x7fffffffffffffffULL;
  if (LimbCount >= 4) {
    return true;
  }
  if (LimbCount == 3) {
    return value.limb2 <= sign_bit_limit;
  }
  if (LimbCount == 2) {
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

template <int LimbCount>
__device__ bool rns8_u192_signed_negative_magnitude_fits_fixed_limbs_device(rns8_u192_device magnitude) {
  constexpr uint64_t sign_bit = 0x8000000000000000ULL;
  if (LimbCount >= 4) {
    return true;
  }
  constexpr int sign_limb = LimbCount - 1;
#pragma unroll
  for (int limb = LimbCount; limb < 3; ++limb) {
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
#pragma unroll
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

template <int LimbCount>
__device__ void rns8_store_u192_unsigned_fixed_limbs_device(uint64_t* dst, rns8_u192_device value) {
#pragma unroll
  for (int limb = 0; limb < LimbCount; ++limb) {
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

template <int LimbCount>
__device__ void rns8_store_u192_negative_twos_complement_fixed_limbs_device(
    uint64_t* dst,
    rns8_u192_device magnitude) {
  uint64_t carry = 1;
#pragma unroll
  for (int limb = 0; limb < LimbCount; ++limb) {
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

template <int Prefix>
__device__ void rns8_reconstruct_canonical_wide_tree_pairs_fixed_prefix_device(
    const int8_t* residues,
    int cell,
    int elements,
    rns8_u192_device* out_x,
    rns8_u192_device* out_product) {
  static_assert(Prefix == 18 || Prefix == 20, "tree-pair CRT is specialized for exact-wide prefix 18 or 20");
  constexpr int kPairCount = Prefix / 2;
  uint32_t pair_values[kPairCount]{};
  uint32_t pair_moduli[kPairCount]{};

#pragma unroll
  for (int pair = 0; pair < kPairCount; ++pair) {
    const int lhs_index = pair * 2;
    const uint32_t lhs_modulus = static_cast<uint32_t>(rns8_default_moduli_device[lhs_index]);
    const uint32_t rhs_modulus = static_cast<uint32_t>(rns8_default_moduli_device[lhs_index + 1]);
    const uint32_t lhs_target = rns8_canonical_from_centered_device(
        residues[static_cast<int64_t>(lhs_index) * elements + cell],
        static_cast<int>(lhs_modulus));
    const uint32_t rhs_target = rns8_canonical_from_centered_device(
        residues[static_cast<int64_t>(lhs_index + 1) * elements + cell],
        static_cast<int>(rhs_modulus));
    const uint32_t inverse = rns8_mod_inverse_device(lhs_modulus % rhs_modulus, rhs_modulus);
    int64_t delta_mod = static_cast<int64_t>(rhs_target) - static_cast<int64_t>(lhs_target % rhs_modulus);
    delta_mod %= static_cast<int64_t>(rhs_modulus);
    delta_mod += static_cast<int64_t>(rhs_modulus) & -static_cast<int64_t>(delta_mod < 0);
    const uint32_t coefficient = static_cast<uint32_t>(
        (static_cast<uint64_t>(delta_mod) * static_cast<uint64_t>(inverse)) % rhs_modulus);
    pair_values[pair] = lhs_target + lhs_modulus * coefficient;
    pair_moduli[pair] = lhs_modulus * rhs_modulus;
  }

  rns8_u192_device x = rns8_u192_from_u64_device(0);
  rns8_u192_device product = rns8_u192_from_u64_device(1);
#pragma unroll
  for (int pair = 0; pair < kPairCount; ++pair) {
    const uint32_t modulus = pair_moduli[pair];
    const uint32_t target = pair_values[pair];
    const uint32_t x_mod = rns8_u192_mod_u32_device(x, modulus);
    const uint32_t product_mod = rns8_u192_mod_u32_device(product, modulus);
    const uint32_t inverse = rns8_mod_inverse_device(product_mod, modulus);
    int64_t delta_mod = static_cast<int64_t>(target) - static_cast<int64_t>(x_mod);
    delta_mod %= static_cast<int64_t>(modulus);
    delta_mod += static_cast<int64_t>(modulus) & -static_cast<int64_t>(delta_mod < 0);
    const uint32_t coefficient = static_cast<uint32_t>(
        (static_cast<uint64_t>(delta_mod) * static_cast<uint64_t>(inverse)) % modulus);
    x = rns8_u192_add_device(x, rns8_u192_mul_u32_device(product, coefficient));
    product = rns8_u192_mul_u32_device(product, modulus);
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

template <int Modulus>
__device__ int8_t rns8_center_i64_fixed_modulus_device(int64_t value) {
  int64_t residue = value % static_cast<int64_t>(Modulus);
  residue += static_cast<int64_t>(Modulus) & -static_cast<int64_t>(residue < 0);
  const int64_t threshold = (static_cast<int64_t>(Modulus) + 1) / 2;
  residue -= static_cast<int64_t>(Modulus) & -static_cast<int64_t>(residue >= threshold);
  return static_cast<int8_t>(residue);
}

template <int Modulus>
__device__ int8_t rns8_center_u64_fixed_modulus_device(uint64_t value) {
  uint64_t residue = value % static_cast<uint64_t>(Modulus);
  const uint64_t threshold = (static_cast<uint64_t>(Modulus) + 1) / 2;
  residue -= static_cast<uint64_t>(Modulus) & (0u - static_cast<uint64_t>(residue >= threshold));
  return static_cast<int8_t>(static_cast<int64_t>(residue));
}

__device__ int8_t rns8_center_i64_default_modulus_fixed_device(
    int64_t value,
    int modulus_index,
    int fallback_modulus) {
  switch (modulus_index) {
    case 0:
      return rns8_center_i64_fixed_modulus_device<256>(value);
    case 1:
      return rns8_center_i64_fixed_modulus_device<255>(value);
    case 2:
      return rns8_center_i64_fixed_modulus_device<253>(value);
    case 3:
      return rns8_center_i64_fixed_modulus_device<251>(value);
    case 4:
      return rns8_center_i64_fixed_modulus_device<247>(value);
    case 5:
      return rns8_center_i64_fixed_modulus_device<239>(value);
    case 6:
      return rns8_center_i64_fixed_modulus_device<233>(value);
    case 7:
      return rns8_center_i64_fixed_modulus_device<229>(value);
    case 8:
      return rns8_center_i64_fixed_modulus_device<227>(value);
    case 9:
      return rns8_center_i64_fixed_modulus_device<223>(value);
    case 10:
      return rns8_center_i64_fixed_modulus_device<217>(value);
    case 11:
      return rns8_center_i64_fixed_modulus_device<211>(value);
    case 12:
      return rns8_center_i64_fixed_modulus_device<199>(value);
    case 13:
      return rns8_center_i64_fixed_modulus_device<197>(value);
    case 14:
      return rns8_center_i64_fixed_modulus_device<193>(value);
    case 15:
      return rns8_center_i64_fixed_modulus_device<191>(value);
    case 16:
      return rns8_center_i64_fixed_modulus_device<181>(value);
    case 17:
      return rns8_center_i64_fixed_modulus_device<179>(value);
    case 18:
      return rns8_center_i64_fixed_modulus_device<173>(value);
    case 19:
      return rns8_center_i64_fixed_modulus_device<167>(value);
    default:
      return rns8_center_i64_device(value, fallback_modulus);
  }
}

__device__ int8_t rns8_center_u64_default_modulus_fixed_device(
    uint64_t value,
    int modulus_index,
    int fallback_modulus) {
  switch (modulus_index) {
    case 0:
      return rns8_center_u64_fixed_modulus_device<256>(value);
    case 1:
      return rns8_center_u64_fixed_modulus_device<255>(value);
    case 2:
      return rns8_center_u64_fixed_modulus_device<253>(value);
    case 3:
      return rns8_center_u64_fixed_modulus_device<251>(value);
    case 4:
      return rns8_center_u64_fixed_modulus_device<247>(value);
    case 5:
      return rns8_center_u64_fixed_modulus_device<239>(value);
    case 6:
      return rns8_center_u64_fixed_modulus_device<233>(value);
    case 7:
      return rns8_center_u64_fixed_modulus_device<229>(value);
    case 8:
      return rns8_center_u64_fixed_modulus_device<227>(value);
    case 9:
      return rns8_center_u64_fixed_modulus_device<223>(value);
    case 10:
      return rns8_center_u64_fixed_modulus_device<217>(value);
    case 11:
      return rns8_center_u64_fixed_modulus_device<211>(value);
    case 12:
      return rns8_center_u64_fixed_modulus_device<199>(value);
    case 13:
      return rns8_center_u64_fixed_modulus_device<197>(value);
    case 14:
      return rns8_center_u64_fixed_modulus_device<193>(value);
    case 15:
      return rns8_center_u64_fixed_modulus_device<191>(value);
    case 16:
      return rns8_center_u64_fixed_modulus_device<181>(value);
    case 17:
      return rns8_center_u64_fixed_modulus_device<179>(value);
    case 18:
      return rns8_center_u64_fixed_modulus_device<173>(value);
    case 19:
      return rns8_center_u64_fixed_modulus_device<167>(value);
    default:
      return rns8_center_u64_device(value, fallback_modulus);
  }
}

__device__ int8_t rns8_center_u8_device(uint8_t value, int modulus) {
  return rns8::detail::finite_u8::center_u8(value, static_cast<uint32_t>(modulus));
}

template <int Modulus>
__device__ int8_t rns8_center_u8_fixed_modulus_device(uint8_t value) {
  return rns8::detail::finite_u8::center_u8_fixed<static_cast<uint32_t>(Modulus)>(value);
}

template <int Modulus>
__device__ uint32_t rns8_canonical_from_centered_fixed_modulus_device(int8_t residue) {
  return rns8::detail::finite_u8::canonical_from_centered_fixed<static_cast<uint32_t>(Modulus)>(residue);
}

