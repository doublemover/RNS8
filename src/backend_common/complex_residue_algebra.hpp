#pragma once
#include "finite_u8_reducer.hpp"
#if defined(__HIPCC__)
#define RNS8_COMPLEX_INLINE __host__ __device__ __forceinline__
#else
#define RNS8_COMPLEX_INLINE inline
#endif
namespace rns8::detail {
struct ComplexResidueCell {int8_t real,imag;};
// Inputs are values: every cell's product loads must finish before either
// internal product slot is overwritten. There are no cross-cell reads.
RNS8_COMPLEX_INLINE ComplexResidueCell complex_recombine_cell(
    int32_t p0,int32_t p1,int32_t p2,int32_t p3,bool three,uint32_t modulus,uint32_t reciprocal) {
  return {finite_u8::reduce_to_centered_i32(p0-p1,modulus,reciprocal),
          finite_u8::reduce_to_centered_i32(three?p2-p0-p1:p2+p3,modulus,reciprocal)};
}
}
#undef RNS8_COMPLEX_INLINE
