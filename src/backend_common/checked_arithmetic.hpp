#pragma once
#include <cstdint>

namespace rns8::detail::checked {
// Out parameters remain unchanged on overflow, including when they alias
// inputs.
constexpr bool add(uint64_t a, uint64_t b, uint64_t& output) noexcept {
  if (a > UINT64_MAX - b) return false;
  output = a + b;
  return true;
}
constexpr bool multiply(uint64_t a, uint64_t b, uint64_t& output) noexcept {
  if (b && a > UINT64_MAX / b) return false;
  output = a * b;
  return true;
}
}  // namespace rns8::detail::checked
