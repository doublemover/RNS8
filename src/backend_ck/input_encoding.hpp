#pragma once
#include "backend_common/finite_u8_reducer.hpp"

#if defined(__HIPCC__)
#define RNS8_CK_INPUT_INLINE __host__ __device__ __forceinline__
#else
#define RNS8_CK_INPUT_INLINE inline
#endif

namespace rns8::detail {
RNS8_CK_INPUT_INLINE int8_t ck_encode_matrix_input(int8_t residue, uint32_t modulus, bool signed_operands) {
  return signed_operands ? residue : static_cast<int8_t>(finite_u8::canonical_from_centered(residue, modulus));
}
}
#undef RNS8_CK_INPUT_INLINE
