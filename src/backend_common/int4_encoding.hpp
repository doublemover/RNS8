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
RNS8_INT4_HD constexpr int signed_nibble(uint32_t word,uint32_t nibble) {
  const int value=int((word>>(4*nibble))&15u);return value>=8?value-16:value;
}
// v1 B is [ceil(K/16)][N][2] uint32 words; low nibble is smaller K.
// Wave halves read the same column fragment, with numerical-zero tail nibbles.
RNS8_INT4_HD constexpr uint64_t int4_b_word_index(uint64_t block,uint64_t col,uint64_t n,uint32_t word) {
  return (block*n+col)*2+word;
}
} // namespace rns8::detail
#undef RNS8_INT4_HD
#endif
