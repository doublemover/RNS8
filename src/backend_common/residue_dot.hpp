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
#if defined(RNS8_EXPERIMENTAL_RESIDUE_SCALAR_CONTROL) && RNS8_EXPERIMENTAL_RESIDUE_SCALAR_CONTROL
constexpr bool kResidueScalarControl = true;
#else
constexpr bool kResidueScalarControl = false;
#endif
#if defined(RNS8_EXPERIMENTAL_RESIDUE_SCALAR_CONTROL) && RNS8_EXPERIMENTAL_RESIDUE_SCALAR_CONTROL && defined(RNS8_EXPERIMENTAL_RESIDUE_DOT4) && RNS8_EXPERIMENTAL_RESIDUE_DOT4
#error "Scalar residue control and packed-dot candidate are mutually exclusive"
#endif

// Explicit research opt-in. OFF preserves the existing scheduler/control;
// ON changes only its byte-dot inner loop, not launch topology or reduction.
#if defined(RNS8_EXPERIMENTAL_RESIDUE_DOT4) && RNS8_EXPERIMENTAL_RESIDUE_DOT4
constexpr bool kResidueDot4Enabled = true;
#else
constexpr bool kResidueDot4Enabled = false;
#endif

RNS8_RESIDUE_DOT_INLINE int32_t signed_byte_dot4(uint32_t a, uint32_t b, int32_t accumulator) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
  // ROCm 7.1's signed alias can omit the sign-extension bits. Spell the
  // verified gfx11 encoding explicitly; never emit it for CDNA/unknown ISAs.
  asm volatile("v_dot4_i32_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]"
               : "+v"(accumulator) : "v"(a), "v"(b));
  return accumulator;
#elif defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || \
      defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
#if __has_builtin(__builtin_amdgcn_sdot4)
  return __builtin_amdgcn_sdot4(static_cast<int32_t>(a), static_cast<int32_t>(b), accumulator, false);
#endif
#elif defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
      defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
#if __has_builtin(__builtin_amdgcn_sudot4)
  return __builtin_amdgcn_sudot4(true, static_cast<int32_t>(a), true, static_cast<int32_t>(b), accumulator, false);
#endif
#endif
#endif
  // Host oracle and unsupported-target fallback: no reinterpret/unaligned
  // loads, signed shift, or assumption about implementation-defined narrowing.
  for (unsigned lane = 0; lane < 4; ++lane) {
    const int av = static_cast<int>((a >> (8 * lane)) & 255u);
    const int bv = static_cast<int>((b >> (8 * lane)) & 255u);
    accumulator += (av - 256 * (av >= 128)) * (bv - 256 * (bv >= 128));
  }
  return accumulator;
}

RNS8_RESIDUE_DOT_INLINE int32_t accumulate_residue_tile(
    const int8_t* a, const int8_t* b, int64_t b_stride, int64_t extent,
    int32_t accumulator, bool packed = kResidueDot4Enabled) {
  if (kResidueScalarControl || !packed) {
    for (int64_t i = 0; i < extent; ++i)
      accumulator += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i * b_stride]);
    return accumulator;
  }
  for (int64_t i = 0; i < extent; i += 4) {
    uint32_t ap = 0, bp = 0;
    for (unsigned lane = 0; lane < 4 && i + lane < extent; ++lane) {
      ap |= uint32_t{static_cast<uint8_t>(a[i + lane])} << (8 * lane);
      bp |= uint32_t{static_cast<uint8_t>(b[(i + lane) * b_stride])} << (8 * lane);
    }
    accumulator = signed_byte_dot4(ap, bp, accumulator);
  }
  return accumulator;
}

// The same arithmetic is exercised on the CPU and in the persistent HIP kernel.
// Callers validate pointers, strides, K and modulus before dispatch.
constexpr int64_t kResidueDotKBlock = 65536;
static_assert(kResidueDotKBlock * 128 * 128 + 128 <= std::numeric_limits<int32_t>::max());

RNS8_RESIDUE_DOT_INLINE int8_t centered_residue_dot(
    const int8_t* a, const int8_t* b, int64_t k, int64_t b_stride, uint32_t modulus, bool packed = kResidueDot4Enabled) {
  const uint32_t reciprocal = finite_u8::modulus_reciprocal_u32(modulus);
  int8_t reduced = 0;
  int64_t offset = 0;
  while (offset < k) {
    const int64_t remaining = k - offset;
    const int64_t block = remaining < kResidueDotKBlock ? remaining : kResidueDotKBlock;
    const int32_t accumulator = accumulate_residue_tile(
        a + offset, b + offset * b_stride, b_stride, block, 0, packed);
    reduced = finite_u8::reduce_to_centered_i32(
        accumulator + static_cast<int32_t>(reduced), modulus, reciprocal);
    offset += block;
  }
  return reduced;
}

}  // namespace rns8::detail

#undef RNS8_RESIDUE_DOT_INLINE
