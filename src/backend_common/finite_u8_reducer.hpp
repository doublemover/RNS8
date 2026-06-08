#pragma once

#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define RNS8_FINITE_U8_INLINE __host__ __device__ __forceinline__
#else
#define RNS8_FINITE_U8_INLINE inline
#endif

namespace rns8::detail::finite_u8 {

constexpr uint64_t kReciprocalScale = 1ull << 32u;

RNS8_FINITE_U8_INLINE uint32_t modulus_reciprocal_u32(uint32_t modulus) {
  return static_cast<uint32_t>(kReciprocalScale / modulus);
}

RNS8_FINITE_U8_INLINE bool static_byte_modulus_supported(uint32_t modulus) {
  switch (modulus) {
    case 256:
    case 255:
    case 253:
    case 251:
    case 247:
    case 243:
    case 241:
    case 239:
    case 233:
    case 229:
    case 227:
    case 223:
    case 217:
    case 211:
    case 199:
    case 197:
    case 193:
    case 191:
    case 181:
    case 179:
    case 173:
    case 167:
    case 163:
    case 157:
    case 151:
    case 149:
    case 139:
    case 137:
    case 131:
    case 127:
      return true;
    default:
      return false;
  }
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

RNS8_FINITE_U8_INLINE int8_t reduce_to_centered_ck_i32(
    int32_t value,
    uint32_t modulus,
    uint32_t reciprocal) {
  return reduce_to_centered_accelerator_i32(value, modulus, reciprocal);
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

RNS8_FINITE_U8_INLINE int8_t reduce_to_centered_static_byte_modulus_i32(int32_t value, uint32_t modulus) {
  switch (modulus) {
    case 256:
      return reduce_to_centered_fixed_i32<256>(value);
    case 255:
      return reduce_to_centered_fixed_i32<255>(value);
    case 253:
      return reduce_to_centered_fixed_i32<253>(value);
    case 251:
      return reduce_to_centered_fixed_i32<251>(value);
    case 247:
      return reduce_to_centered_fixed_i32<247>(value);
    case 243:
      return reduce_to_centered_fixed_i32<243>(value);
    case 241:
      return reduce_to_centered_fixed_i32<241>(value);
    case 239:
      return reduce_to_centered_fixed_i32<239>(value);
    case 233:
      return reduce_to_centered_fixed_i32<233>(value);
    case 229:
      return reduce_to_centered_fixed_i32<229>(value);
    case 227:
      return reduce_to_centered_fixed_i32<227>(value);
    case 223:
      return reduce_to_centered_fixed_i32<223>(value);
    case 217:
      return reduce_to_centered_fixed_i32<217>(value);
    case 211:
      return reduce_to_centered_fixed_i32<211>(value);
    case 199:
      return reduce_to_centered_fixed_i32<199>(value);
    case 197:
      return reduce_to_centered_fixed_i32<197>(value);
    case 193:
      return reduce_to_centered_fixed_i32<193>(value);
    case 191:
      return reduce_to_centered_fixed_i32<191>(value);
    case 181:
      return reduce_to_centered_fixed_i32<181>(value);
    case 179:
      return reduce_to_centered_fixed_i32<179>(value);
    case 173:
      return reduce_to_centered_fixed_i32<173>(value);
    case 167:
      return reduce_to_centered_fixed_i32<167>(value);
    case 163:
      return reduce_to_centered_fixed_i32<163>(value);
    case 157:
      return reduce_to_centered_fixed_i32<157>(value);
    case 151:
      return reduce_to_centered_fixed_i32<151>(value);
    case 149:
      return reduce_to_centered_fixed_i32<149>(value);
    case 139:
      return reduce_to_centered_fixed_i32<139>(value);
    case 137:
      return reduce_to_centered_fixed_i32<137>(value);
    case 131:
      return reduce_to_centered_fixed_i32<131>(value);
    case 127:
      return reduce_to_centered_fixed_i32<127>(value);
    default:
      return 0;
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
