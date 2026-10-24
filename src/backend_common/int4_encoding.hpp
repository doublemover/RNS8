#ifndef RNS8_INT4_ENCODING_HPP
#define RNS8_INT4_ENCODING_HPP
#include <cstdint>
#if defined(__HIPCC__)
#define RNS8_INT4_HD __host__ __device__
#else
#define RNS8_INT4_HD
#endif
namespace rns8::detail {
// Finite-ring interpretation is explicit. This is not a truncation of a
// default-ladder RNS plane: only moduli 2..16 have this centered alphabet.
RNS8_INT4_HD constexpr int centered_int4_residue(int8_t value, int modulus) {
  int r = static_cast<int>(value) % modulus;
  if (r < 0) r += modulus;
  return r >= (modulus + 1) / 2 ? r - modulus : r;
}
RNS8_INT4_HD constexpr uint32_t finite_int4_nibble(int8_t value, int modulus) {
  return static_cast<uint32_t>(centered_int4_residue(value, modulus)) & 15u;
}
} // namespace rns8::detail
#undef RNS8_INT4_HD
#endif
