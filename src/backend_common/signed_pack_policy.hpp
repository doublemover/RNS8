#pragma once
#include <cstdint>
namespace rns8::detail {
enum class SignedPackVariant : uint32_t { public_selected = 0, fixed_prefix = 1, generic = 2 };
enum class SignedPackKernel : uint32_t { small, four_wide, eight_wide, fixed_prefix, generic };
// Caller has admitted positive, nonoverflowing dimensions. Public policy is
// deliberately unchanged until independent paired crossover evidence exists.
inline SignedPackKernel signed_pack_kernel(int64_t rows, int64_t cols, int64_t ld,
                                          SignedPackVariant variant, uint32_t prefix = 9) {
  if (variant == SignedPackVariant::generic) return SignedPackKernel::generic;
  if (variant == SignedPackVariant::fixed_prefix) return SignedPackKernel::fixed_prefix;
  const int64_t cells = rows * cols;
  if (ld == cols && cells >= 256 && cells <= 4096) return SignedPackKernel::four_wide;
  if (ld == cols && cells > 4096) return SignedPackKernel::eight_wide;
  if (cells <= 4096) return SignedPackKernel::small;
  return prefix <= 9 || prefix == 18 || prefix == 20 ? SignedPackKernel::fixed_prefix : SignedPackKernel::generic;
}
inline const char* signed_pack_kernel_name(SignedPackKernel kernel) {
  switch (kernel) {
    case SignedPackKernel::small: return "small_per_residue";
    case SignedPackKernel::four_wide: return "four_wide_per_plane";
    case SignedPackKernel::eight_wide: return "eight_wide_per_plane";
    case SignedPackKernel::fixed_prefix: return "fixed_prefix_one_load";
    case SignedPackKernel::generic: return "generic_per_residue";
  }
  return "invalid";
}
}
