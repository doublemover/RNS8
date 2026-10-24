#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>

namespace rns8::bench {

struct ExactWideMagnitudeStats {
  bool available = false;
  bool oracle_complete = false;
  uint32_t a_max_magnitude_bits = 0, b_max_magnitude_bits = 0;
  uint32_t output_min_magnitude_bits = UINT32_MAX, output_max_magnitude_bits = 0;
  uint64_t checked_outputs = 0, checked_limbs = 0, negative_outputs = 0;
  std::string a_max_abs = "0", b_max_abs = "0", admitted_abs_bound = "0", basis_product = "0";
  double admission_us = 0, oracle_us = 0;
};

inline uint32_t magnitude_bits(const boost::multiprecision::cpp_int& value) {
  const boost::multiprecision::cpp_int absolute = value < 0 ? -value : value;
  return absolute == 0 ? 0 : static_cast<uint32_t>(boost::multiprecision::msb(absolute) + 1);
}

// exponent=0 preserves the historical seeded small-input distribution exactly.
// Wide mixed fixtures have bounded nonconstant magnitudes and independently
// signed operands; constants are separately retained as guaranteed wide outputs.
template<class T>
void fill_exact_wide_magnitude(std::vector<T>& values, std::mt19937_64& rng,
                               unsigned exponent, bool mixed) {
  static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>);
  if (exponent != 0 && exponent != 40 && exponent != 62)
    throw std::invalid_argument("unknown exact-wide magnitude exponent");
  if (exponent == 0) {
    std::uniform_int_distribution<T> distribution(std::is_signed_v<T> ? T(-16) : T(0), T(16));
    for (auto& value : values) value = distribution(rng);
    return;
  }
  for (auto& value : values) {
    const uint64_t random = mixed ? rng() : 0;
    const uint64_t magnitude = (UINT64_C(1) << exponent) + (mixed ? random & 1023u : 0);
    value = static_cast<T>(magnitude);
    if constexpr (std::is_signed_v<T>) if (mixed && (random >> 63)) value = -value;
  }
}

template<class T>
void admit_exact_wide_fixture(const std::vector<T>& a, const std::vector<T>& b,
                             uint64_t k, const std::vector<uint16_t>& moduli,
                             uint32_t limbs, ExactWideMagnitudeStats& stats) {
  using boost::multiprecision::cpp_int;
  if (k == 0 || a.empty() || b.empty() || moduli.empty() || limbs == 0 || limbs > 32)
    throw std::invalid_argument("invalid exact-wide fixture geometry");
  cpp_int a_max = 0, b_max = 0, product = 1;
  for (T value : a) { cpp_int x = value; if (x < 0) x = -x; if (x > a_max) a_max = x; }
  for (T value : b) { cpp_int x = value; if (x < 0) x = -x; if (x > b_max) b_max = x; }
  for (auto modulus : moduli) product *= modulus;
  const cpp_int bound = cpp_int(k) * a_max * b_max;
  const cpp_int threshold = std::is_signed_v<T> ? cpp_int(2 * bound) : bound;
  const cpp_int storage_max = (cpp_int(1) << (64u * limbs - (std::is_signed_v<T> ? 1u : 0u))) - 1;
  if (product <= threshold) throw std::range_error("fixture bound is not strictly covered by selected CRT prefix");
  if (bound > storage_max) throw std::range_error("fixture bound exceeds requested signed/unsigned limb storage");
  stats.available = true;
  stats.a_max_magnitude_bits = std::max(stats.a_max_magnitude_bits, magnitude_bits(a_max));
  stats.b_max_magnitude_bits = std::max(stats.b_max_magnitude_bits, magnitude_bits(b_max));
  if (a_max > cpp_int(stats.a_max_abs)) stats.a_max_abs = a_max.str();
  if (b_max > cpp_int(stats.b_max_abs)) stats.b_max_abs = b_max.str();
  if (bound > cpp_int(stats.admitted_abs_bound)) stats.admitted_abs_bound = bound.str();
  stats.basis_product = product.str();
}

// Independent mathematical integer GEMM, not CRT or the library's reference
// implementation. Every logical limb and every padded output cell is checked.
// This pre/postflight is never included in timed kernel or public-call samples.
template<class T>
void verify_exact_wide_fixture(const std::vector<T>& a, const std::vector<T>& b,
                              const std::vector<uint64_t>& c, uint64_t m, uint64_t n,
                              uint64_t k, uint64_t ldc, uint32_t limbs,
                              ExactWideMagnitudeStats& stats) {
  using boost::multiprecision::cpp_int;
  if (m == 0 || n == 0 || k == 0 || ldc < n || limbs == 0 || limbs > 32 || a.size() != m * k || b.size() != k * n ||
      c.size() < ((m - 1) * ldc + n) * limbs)
    throw std::invalid_argument("exact-wide oracle shape mismatch");
  const cpp_int word_mask = (cpp_int(1) << 64) - 1;
  const cpp_int storage_modulus = cpp_int(1) << (64u * limbs);
  for (uint64_t row = 0; row < m; ++row) {
    for (uint64_t col = 0; col < n; ++col) {
      cpp_int expected = 0;
      for (uint64_t inner = 0; inner < k; ++inner)
        expected += cpp_int(a[row * k + inner]) * cpp_int(b[inner * n + col]);
      const uint32_t bits = magnitude_bits(expected);
      stats.output_min_magnitude_bits = std::min(stats.output_min_magnitude_bits, bits);
      stats.output_max_magnitude_bits = std::max(stats.output_max_magnitude_bits, bits);
      stats.negative_outputs += expected < 0;
      if (expected < 0) expected += storage_modulus;
      if (expected < 0 || expected >= storage_modulus) throw std::range_error("oracle exceeds limb width");
      for (uint32_t limb = 0; limb < limbs; ++limb) {
        if (c[(row * ldc + col) * limbs + limb] != static_cast<uint64_t>(expected & word_mask))
          throw std::runtime_error("exact-wide public output disagrees with full integer oracle");
        expected >>= 64;
        ++stats.checked_limbs;
      }
      ++stats.checked_outputs;
    }
    for (uint64_t col = n; col < ldc && (row * ldc + col + 1) * limbs <= c.size(); ++col)
      for (uint32_t limb = 0; limb < limbs; ++limb)
        if (c[(row * ldc + col) * limbs + limb] != UINT64_C(0x5a5a5a5a5a5a5a5a))
          throw std::runtime_error("exact-wide export changed output padding");
  }
  stats.oracle_complete = true;
}

}  // namespace rns8::bench
