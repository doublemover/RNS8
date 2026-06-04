#pragma once

#include <cstdint>

#define RNS8_FINITE_U8_INLINE __host__ __device__ __forceinline__

namespace rns8::detail::finite_u8 {

constexpr uint64_t kReciprocalScale = 1ull << 32u;

RNS8_FINITE_U8_INLINE uint32_t modulus_reciprocal_u32(uint32_t modulus) {
  return static_cast<uint32_t>(kReciprocalScale / modulus);
}

RNS8_FINITE_U8_INLINE uint32_t abs_i32_to_u32(int32_t value) {
  return value < 0 ? static_cast<uint32_t>(-static_cast<int64_t>(value)) : static_cast<uint32_t>(value);
}

RNS8_FINITE_U8_INLINE uint32_t reduce_u32_small_modulus(
    uint32_t value,
    uint32_t modulus,
    uint32_t reciprocal) {
  const uint32_t quotient = static_cast<uint32_t>((static_cast<uint64_t>(value) * reciprocal) >> 32u);
  uint32_t residue = value - quotient * modulus;
  residue -= modulus & (0u - static_cast<uint32_t>(residue >= modulus));
  residue -= modulus & (0u - static_cast<uint32_t>(residue >= modulus));
  return residue;
}

RNS8_FINITE_U8_INLINE uint32_t reduce_u32_mod255(uint32_t value) {
  uint32_t residue =
      (value & 0xffu) + ((value >> 8u) & 0xffu) + ((value >> 16u) & 0xffu) + ((value >> 24u) & 0xffu);
  residue = (residue & 0xffu) + (residue >> 8u);
  residue -= 255u & (0u - static_cast<uint32_t>(residue >= 255u));
  return residue;
}

RNS8_FINITE_U8_INLINE uint32_t reduce_u32_mod251(uint32_t value) {
  uint32_t residue = (value & 0xffu) + 5u * ((value >> 8u) & 0xffu) +
                     25u * ((value >> 16u) & 0xffu) + 125u * ((value >> 24u) & 0xffu);
  residue = (residue & 0xffu) + 5u * (residue >> 8u);
  residue -= 251u & (0u - static_cast<uint32_t>(residue >= 251u));
  residue -= 251u & (0u - static_cast<uint32_t>(residue >= 251u));
  residue -= 251u & (0u - static_cast<uint32_t>(residue >= 251u));
  residue -= 251u & (0u - static_cast<uint32_t>(residue >= 251u));
  return residue;
}

RNS8_FINITE_U8_INLINE uint32_t reduce_u32(uint32_t value, uint32_t modulus, uint32_t reciprocal) {
  if (modulus == 256u) {
    return value & 0xffu;
  }
  if (modulus == 255u) {
    return reduce_u32_mod255(value);
  }
  if (modulus == 251u) {
    return reduce_u32_mod251(value);
  }
  return reduce_u32_small_modulus(value, modulus, reciprocal);
}

template <uint32_t Modulus>
RNS8_FINITE_U8_INLINE uint32_t reduce_u32_fixed(uint32_t value) {
  if constexpr (Modulus == 256u) {
    return value & 0xffu;
  } else if constexpr (Modulus == 255u) {
    return reduce_u32_mod255(value);
  } else if constexpr (Modulus == 251u) {
    return reduce_u32_mod251(value);
  } else {
    return reduce_u32_small_modulus(value, Modulus, modulus_reciprocal_u32(Modulus));
  }
}

RNS8_FINITE_U8_INLINE int8_t center_unsigned_residue(uint32_t residue, uint32_t modulus) {
  const uint32_t threshold = (modulus + 1u) / 2u;
  residue -= modulus & (0u - static_cast<uint32_t>(residue >= threshold));
  return static_cast<int8_t>(static_cast<int32_t>(residue));
}

RNS8_FINITE_U8_INLINE int8_t reduce_to_centered_mod256_i32(int32_t value) {
  return static_cast<int8_t>(value & 0xff);
}

RNS8_FINITE_U8_INLINE int8_t reduce_to_centered_mod255_i32(int32_t value) {
  uint32_t residue = reduce_u32_mod255(abs_i32_to_u32(value));
  const uint32_t negative_nonzero =
      static_cast<uint32_t>(value < 0) & static_cast<uint32_t>(residue != 0);
  const uint32_t negative_mask = 0u - negative_nonzero;
  residue = (residue & ~negative_mask) | ((255u - residue) & negative_mask);
  return center_unsigned_residue(residue, 255u);
}

RNS8_FINITE_U8_INLINE int8_t reduce_to_centered_mod251_i32(int32_t value) {
  uint32_t residue = reduce_u32_mod251(abs_i32_to_u32(value));
  const uint32_t negative_nonzero =
      static_cast<uint32_t>(value < 0) & static_cast<uint32_t>(residue != 0);
  const uint32_t negative_mask = 0u - negative_nonzero;
  residue = (residue & ~negative_mask) | ((251u - residue) & negative_mask);
  return center_unsigned_residue(residue, 251u);
}

RNS8_FINITE_U8_INLINE int8_t reduce_to_centered_i32(
    int32_t value,
    uint32_t modulus,
    uint32_t reciprocal) {
  if (modulus == 256u) {
    return reduce_to_centered_mod256_i32(value);
  }
  if (modulus == 255u) {
    return reduce_to_centered_mod255_i32(value);
  }
  if (modulus == 251u) {
    return reduce_to_centered_mod251_i32(value);
  }
  const uint32_t magnitude = abs_i32_to_u32(value);
  uint32_t residue = reduce_u32_small_modulus(magnitude, modulus, reciprocal);
  const uint32_t negative_nonzero =
      static_cast<uint32_t>(value < 0) & static_cast<uint32_t>(residue != 0);
  const uint32_t negative_mask = 0u - negative_nonzero;
  residue = (residue & ~negative_mask) | ((modulus - residue) & negative_mask);
  return center_unsigned_residue(residue, modulus);
}

RNS8_FINITE_U8_INLINE int8_t reduce_to_centered_accelerator_i32(
    int32_t value,
    uint32_t modulus,
    uint32_t reciprocal) {
  if (modulus == 256u) {
    return reduce_to_centered_mod256_i32(value);
  }
  const uint32_t magnitude = abs_i32_to_u32(value);
  uint32_t residue = reduce_u32_small_modulus(magnitude, modulus, reciprocal);
  const uint32_t negative_nonzero =
      static_cast<uint32_t>(value < 0) & static_cast<uint32_t>(residue != 0);
  const uint32_t negative_mask = 0u - negative_nonzero;
  residue = (residue & ~negative_mask) | ((modulus - residue) & negative_mask);
  return center_unsigned_residue(residue, modulus);
}

RNS8_FINITE_U8_INLINE int8_t reduce_to_centered_ck_i32(
    int32_t value,
    uint32_t modulus,
    uint32_t reciprocal) {
  if (modulus == 256u) {
    return reduce_to_centered_mod256_i32(value);
  }
  const bool negative = value < 0;
  uint32_t magnitude = abs_i32_to_u32(value);
  uint32_t quotient = static_cast<uint32_t>((static_cast<uint64_t>(magnitude) * reciprocal) >> 32u);
  uint32_t residue = magnitude - quotient * modulus;
  while (residue >= modulus) {
    residue -= modulus;
  }
  int32_t centered = static_cast<int32_t>(residue);
  if (negative && centered != 0) {
    centered = static_cast<int32_t>(modulus) - centered;
  }
  const int32_t threshold = (static_cast<int32_t>(modulus) + 1) / 2;
  if (centered >= threshold) {
    centered -= static_cast<int32_t>(modulus);
  }
  return static_cast<int8_t>(centered);
}

template <uint32_t Modulus>
RNS8_FINITE_U8_INLINE int8_t reduce_to_centered_fixed_i32(int32_t value) {
  if constexpr (Modulus == 256u) {
    return reduce_to_centered_mod256_i32(value);
  } else if constexpr (Modulus == 255u) {
    return reduce_to_centered_mod255_i32(value);
  } else if constexpr (Modulus == 251u) {
    return reduce_to_centered_mod251_i32(value);
  } else {
    uint32_t residue = reduce_u32_fixed<Modulus>(abs_i32_to_u32(value));
    constexpr uint32_t modulus = Modulus;
    const uint32_t negative_nonzero =
        static_cast<uint32_t>(value < 0) & static_cast<uint32_t>(residue != 0);
    const uint32_t negative_mask = 0u - negative_nonzero;
    residue = (residue & ~negative_mask) | ((modulus - residue) & negative_mask);
    return center_unsigned_residue(residue, modulus);
  }
}

RNS8_FINITE_U8_INLINE int8_t center_u8(uint8_t value, uint32_t modulus) {
  if (modulus == 256u) {
    return static_cast<int8_t>(value);
  }
  if (modulus == 255u) {
    return reduce_to_centered_mod255_i32(static_cast<int32_t>(value));
  }
  if (modulus == 251u) {
    return reduce_to_centered_mod251_i32(static_cast<int32_t>(value));
  }
  return center_unsigned_residue(static_cast<uint32_t>(value) % modulus, modulus);
}

template <uint32_t Modulus>
RNS8_FINITE_U8_INLINE int8_t center_u8_fixed(uint8_t value) {
  if constexpr (Modulus == 256u) {
    return static_cast<int8_t>(value);
  } else if constexpr (Modulus == 255u) {
    return reduce_to_centered_mod255_i32(static_cast<int32_t>(value));
  } else if constexpr (Modulus == 251u) {
    return reduce_to_centered_mod251_i32(static_cast<int32_t>(value));
  } else {
    return center_unsigned_residue(static_cast<uint32_t>(value) % Modulus, Modulus);
  }
}

RNS8_FINITE_U8_INLINE uint32_t canonical_from_centered(int8_t residue, uint32_t modulus) {
  const int32_t value = static_cast<int32_t>(residue);
  return static_cast<uint32_t>(value + (static_cast<int32_t>(modulus) & -static_cast<int32_t>(value < 0)));
}

template <uint32_t Modulus>
RNS8_FINITE_U8_INLINE uint32_t canonical_from_centered_fixed(int8_t residue) {
  const int32_t value = static_cast<int32_t>(residue);
  return static_cast<uint32_t>(value + (static_cast<int32_t>(Modulus) & -static_cast<int32_t>(value < 0)));
}

}  // namespace rns8::detail::finite_u8

#undef RNS8_FINITE_U8_INLINE
