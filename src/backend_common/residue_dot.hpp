#pragma once

#include <cstdint>
#include <limits>

#include "finite_u8_reducer.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define RNS8_RESIDUE_DOT_INLINE __host__ __device__ __forceinline__
#else
#define RNS8_RESIDUE_DOT_INLINE inline
#endif

namespace rns8::detail {

// The same arithmetic is exercised on the CPU and in the persistent HIP kernel.
// Callers validate pointers, strides, K and modulus before dispatch.
constexpr int64_t kResidueDotKBlock = 65536;
static_assert(kResidueDotKBlock * 128 * 128 + 128 <= std::numeric_limits<int32_t>::max());

RNS8_RESIDUE_DOT_INLINE int8_t centered_residue_dot(
    const int8_t* a, const int8_t* b, int64_t k, int64_t b_stride, uint32_t modulus) {
  const uint32_t reciprocal = finite_u8::modulus_reciprocal_u32(modulus);
  int8_t reduced = 0;
  int64_t offset = 0;
  while (offset < k) {
    const int64_t remaining = k - offset;
    const int64_t block = remaining < kResidueDotKBlock ? remaining : kResidueDotKBlock;
    int32_t accumulator = 0;
    for (int64_t i = offset; i < offset + block; ++i) {
      accumulator += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i * b_stride]);
    }
    reduced = finite_u8::reduce_to_centered_i32(
        accumulator + static_cast<int32_t>(reduced), modulus, reciprocal);
    offset += block;
  }
  return reduced;
}

}  // namespace rns8::detail

#undef RNS8_RESIDUE_DOT_INLINE
