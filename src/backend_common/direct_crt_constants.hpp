#pragma once
#include <cstdint>
#include "garner_constants.hpp"

namespace rns8::detail::direct_crt {
struct Constants {
  uint64_t products[21][3]{};
  uint64_t weights[21][20][3]{};
  uint32_t inverses[21][20]{};
};

constexpr void divide_small(const uint64_t* input, uint32_t divisor, uint64_t* quotient) {
  uint64_t remainder = 0;
  for (int digit = 5; digit >= 0; --digit) {
    const int limb = digit / 2, shift = (digit % 2) * 32;
    const uint64_t current = (remainder << 32) | ((input[limb] >> shift) & 0xffffffffULL);
    quotient[limb] |= (current / divisor) << shift;
    remainder = current % divisor;
  }
}

constexpr uint32_t modulo_small(const uint64_t* input, uint32_t modulus) {
  uint64_t remainder = 0;
  for (int digit = 5; digit >= 0; --digit)
    remainder = ((remainder << 32) | ((input[digit / 2] >> ((digit % 2) * 32)) & 0xffffffffULL)) % modulus;
  return static_cast<uint32_t>(remainder);
}

constexpr Constants make_constants() {
  Constants out{};
  for (uint32_t prefix = 1; prefix <= 20; ++prefix) {
    for (uint32_t limb = 0; limb < 3; ++limb) out.products[prefix][limb] = garner::kConstants.products[prefix][limb];
    for (uint32_t p = 0; p < prefix; ++p) {
      const uint32_t modulus = garner::kConstants.moduli[p];
      divide_small(out.products[prefix], modulus, out.weights[prefix][p]);
      out.inverses[prefix][p] = garner::inverse(modulo_small(out.weights[prefix][p], modulus), modulus);
    }
  }
  return out;
}
inline constexpr Constants kConstants = make_constants();
// Every weighted term is <P. Sum <20*P <20*2^160 <2^165,
// so unreduced direct CRT and every ordered intermediate fit 192 bits.
static_assert((kConstants.products[20][2] >> 32) == 0);
}
