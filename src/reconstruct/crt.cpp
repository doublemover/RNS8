#include "core/internal.hpp"

#include <limits>

namespace rns8::detail {

namespace {

uint32_t mod_inverse(uint32_t a, uint32_t modulus) {
  int64_t t = 0;
  int64_t next_t = 1;
  int64_t r = modulus;
  int64_t next_r = a % modulus;

  while (next_r != 0) {
    const int64_t quotient = r / next_r;
    const int64_t new_t = t - quotient * next_t;
    t = next_t;
    next_t = new_t;
    const int64_t new_r = r - quotient * next_r;
    r = next_r;
    next_r = new_r;
  }

  if (r != 1) {
    return 0;
  }
  if (t < 0) {
    t += modulus;
  }
  return static_cast<uint32_t>(t);
}

cpp_int reconstruct_canonical(const std::vector<int8_t>& residues, uint32_t prefix) {
  cpp_int x = 0;
  cpp_int product = 1;
  for (uint32_t i = 0; i < prefix; ++i) {
    const uint32_t modulus = kDefaultModuli[i];
    const uint32_t target = canonical_from_centered(residues[i], static_cast<uint16_t>(modulus));
    cpp_int x_mod = x % modulus;
    if (x_mod < 0) {
      x_mod += modulus;
    }
    const uint32_t product_mod = static_cast<uint32_t>(product % modulus);
    const uint32_t inverse = mod_inverse(product_mod, modulus);
    const int64_t delta = static_cast<int64_t>(target) - static_cast<int64_t>(x_mod);
    int64_t delta_mod = delta % static_cast<int64_t>(modulus);
    if (delta_mod < 0) {
      delta_mod += modulus;
    }
    const uint32_t coefficient = static_cast<uint32_t>((delta_mod * inverse) % modulus);
    x += product * coefficient;
    product *= modulus;
  }
  return x;
}

bool fits_uint64(const cpp_int& value) {
  return value >= 0 && value <= cpp_int(std::numeric_limits<uint64_t>::max());
}

bool fits_int64(const cpp_int& value) {
  return value >= cpp_int(std::numeric_limits<int64_t>::min()) &&
         value <= cpp_int(std::numeric_limits<int64_t>::max());
}

cpp_int abs_cpp(const cpp_int& value) {
  return value < 0 ? -value : value;
}

}  // namespace

rns8_status reconstruct_unsigned(
    const std::vector<int8_t>& residues,
    uint32_t prefix,
    uint64_t bound,
    uint64_t& out) {
  const rns8_status range_status =
      validate_bound_contract(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, bound, prefix);
  if (range_status != RNS8_SUCCESS) {
    return range_status;
  }
  if (residues.size() < prefix) {
    return RNS8_INVALID_ARGUMENT;
  }

  const cpp_int value = reconstruct_canonical(residues, prefix);
  if (!fits_uint64(value) || value > cpp_int(bound)) {
    return RNS8_RANGE_ERROR;
  }
  out = static_cast<uint64_t>(value);
  return RNS8_SUCCESS;
}

rns8_status reconstruct_signed(
    const std::vector<int8_t>& residues,
    uint32_t prefix,
    uint64_t bound,
    int64_t& out) {
  const rns8_status range_status =
      validate_bound_contract(RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS, bound, prefix);
  if (range_status != RNS8_SUCCESS) {
    return range_status;
  }
  if (residues.size() < prefix) {
    return RNS8_INVALID_ARGUMENT;
  }

  const cpp_int product = modulus_product(prefix);
  cpp_int value = reconstruct_canonical(residues, prefix);
  if (value > product / 2) {
    value -= product;
  }
  if (!fits_int64(value) || abs_cpp(value) > cpp_int(bound)) {
    return RNS8_RANGE_ERROR;
  }
  out = static_cast<int64_t>(value);
  return RNS8_SUCCESS;
}

}  // namespace rns8::detail

