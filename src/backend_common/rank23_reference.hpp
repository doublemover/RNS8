#pragma once
#include "rank23_program.hpp"
namespace rns8::detail::rank23 {
// Scalar modular prototype, independent of HIP transform/recombination code.
// Inputs are logical 3x3 row-major byte matrices. Stage the result to allow
// aliasing and leave it untouched on malformed modulus/pointer arguments.
inline bool multiply3x3_modular(const int8_t *a, const int8_t *b,
                                int8_t *output, uint32_t modulus) {
  if (!a || !b || !output || modulus < 2 || modulus > 256)
    return false;
  const auto center = [modulus](int value) {
    value %= static_cast<int>(modulus);
    if (value < 0)
      value += modulus;
    if (value >= static_cast<int>((modulus + 1) / 2))
      value -= modulus;
    return value;
  };
  int8_t staged[9]{};
  for (const auto &term : program().products) {
    int left = 0, right = 0;
    for (unsigned i = 0; i < 9; ++i) {
      left += term.a[i] * a[i];
      right += term.b[i] * b[i];
    }
    const int product = center(center(left) * center(right));
    for (unsigned i = 0; i < 9; ++i)
      staged[i] = static_cast<int8_t>(center(staged[i] + term.c[i] * product));
  }
  for (unsigned i = 0; i < 9; ++i)
    output[i] = staged[i];
  return true;
}
} // namespace rns8::detail::rank23
