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

cpp_int centered_from_canonical(const cpp_int& value, const cpp_int& product) {
  const cpp_int threshold = (product + 1) / 2;
  return value >= threshold ? value - product : value;
}

}  // namespace

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
  const cpp_int value = centered_from_canonical(reconstruct_canonical(residues, prefix), product);
  if (!fits_int64(value) || abs_cpp(value) > cpp_int(bound)) {
    return RNS8_RANGE_ERROR;
  }
  out = static_cast<int64_t>(value);
  return RNS8_SUCCESS;
}

rns8_status export_exact_wide_unsigned_limbs(
    const std::vector<int8_t>& residues,
    uint32_t prefix,
    uint64_t* out,
    uint32_t limb_count) {
  if (!out || residues.size() < prefix || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX || limb_count == 0 ||
      limb_count > 32) {
    return RNS8_INVALID_ARGUMENT;
  }
  cpp_int value = reconstruct_canonical(residues, prefix);
  const cpp_int limit = cpp_int(1) << (64u * limb_count);
  if (value < 0 || value >= limit) {
    return RNS8_RANGE_ERROR;
  }
  for (uint32_t limb = 0; limb < limb_count; ++limb) {
    out[limb] = static_cast<uint64_t>(value & cpp_int(std::numeric_limits<uint64_t>::max()));
    value >>= 64u;
  }
  return RNS8_SUCCESS;
}

rns8_status export_exact_wide_signed_limbs(
    const std::vector<int8_t>& residues,
    uint32_t prefix,
    uint64_t* out,
    uint32_t limb_count) {
  if (!out || residues.size() < prefix || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX || limb_count == 0 ||
      limb_count > 32) {
    return RNS8_INVALID_ARGUMENT;
  }
  const cpp_int product = modulus_product(prefix);
  cpp_int value = centered_from_canonical(reconstruct_canonical(residues, prefix), product);

  const uint32_t bits = 64u * limb_count;
  const cpp_int min_value = -(cpp_int(1) << (bits - 1u));
  const cpp_int max_value = (cpp_int(1) << (bits - 1u)) - 1;
  if (value < min_value || value > max_value) {
    return RNS8_RANGE_ERROR;
  }
  if (value < 0) {
    value += cpp_int(1) << bits;
  }
  for (uint32_t limb = 0; limb < limb_count; ++limb) {
    out[limb] = static_cast<uint64_t>(value & cpp_int(std::numeric_limits<uint64_t>::max()));
    value >>= 64u;
  }
  return RNS8_SUCCESS;
}

}  // namespace rns8::detail
