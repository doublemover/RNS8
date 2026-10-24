#pragma once
#include <cstdint>
#include "finite_u8_reducer.hpp"

#if defined(__HIPCC__)
#define RNS8_EPILOGUE_INLINE __host__ __device__ __forceinline__
#else
#define RNS8_EPILOGUE_INLINE inline
#endif

namespace rns8::detail {
// Coefficients have explicit integer semantics, including INT64_MIN. Reduce
// before multiplication; each term is bounded by 255*128 and their sum by 65280.
RNS8_EPILOGUE_INLINE int8_t centered_residue_axpby(
    int8_t product, int8_t addend, int64_t alpha, int64_t beta, uint32_t modulus) {
  int32_t a = static_cast<int32_t>(alpha % static_cast<int64_t>(modulus));
  int32_t b = static_cast<int32_t>(beta % static_cast<int64_t>(modulus));
  if (a < 0) a += static_cast<int32_t>(modulus);
  if (b < 0) b += static_cast<int32_t>(modulus);
  return finite_u8::reduce_to_centered_i32(
      a * static_cast<int32_t>(product) + b * static_cast<int32_t>(addend), modulus,
      finite_u8::modulus_reciprocal_u32(modulus));
}
}
#undef RNS8_EPILOGUE_INLINE
