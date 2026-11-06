#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include "rns8/rns8.h"

namespace rns8_fuzz {
using boost::multiprecision::cpp_int;
constexpr std::size_t header_bytes = 24, max_values = 262144;
constexpr std::size_t max_case_bytes = header_bytes + 8 * max_values;
// Literal reference data intentionally does not call the production ladder.
inline constexpr std::array<uint16_t, 20> oracle_moduli = {256, 255, 253, 251, 247, 239, 233,
                                                           229, 227, 223, 217, 211, 199, 197,
                                                           193, 191, 181, 179, 173, 167};

struct ArithmeticCase {
  // Wire codes are contracts, independent of public enum representations.
  uint8_t semantic = 0, operations = 0, padding_a = 0, padding_b = 0, padding_c = 0;
  uint32_t m = 0, n = 0, k = 0, prefix = 0;
  uint16_t modulus = 0;
  std::vector<uint64_t> a, b;
  bool signed_input() const { return semantic == 0 || semantic == 4; }
  bool finite() const { return semantic == 2; }
  bool wrap() const { return semantic == 3; }
  bool wide() const { return semantic >= 4; }
  rns8_semantics semantics() const {
    constexpr rns8_semantics values[] = {RNS8_BOUNDED_I64,       RNS8_BOUNDED_U64,
                                         RNS8_FINITE_RING_U8,    RNS8_WRAP_U64_MOD_2_64,
                                         RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED};
    return values[semantic];
  }
};

inline uint64_t little_integer(const uint8_t* data, unsigned bytes) {
  uint64_t value = 0;
  for (unsigned index = 0; index < bytes; ++index)
    value |= uint64_t(data[index]) << (8 * index);
  return value;
}
inline int64_t signed_bits(uint64_t value) {
  // Avoid out-of-range unsigned-to-signed conversion and abs(INT64_MIN).
  return value <= uint64_t(INT64_MAX) ? int64_t(value) : -1 - int64_t(UINT64_MAX - value);
}
inline bool decode(const uint8_t* data, std::size_t size, ArithmeticCase& out) {
  if (size < header_bytes || size > max_case_bytes || data[0] != 'R' || data[1] != '8' ||
      data[2] != 'F' || data[3] != '1')
    return false;
  ArithmeticCase value;
  value.semantic = data[4];
  value.operations = data[5];
  value.m = data[6];
  value.n = data[7];
  value.k = uint32_t(little_integer(data + 8, 4));
  value.padding_a = data[12];
  value.padding_b = data[13];
  value.padding_c = data[14];
  value.prefix = data[15];
  value.modulus = uint16_t(little_integer(data + 16, 2));
  if (value.semantic > 5 || value.operations > 7 || !value.m || value.m > 8 || !value.n ||
      value.n > 8 || !value.k || value.k > 65537 || value.padding_a > 7 || value.padding_b > 7 ||
      value.padding_c > 7 || little_integer(data + 18, 6) != 0)
    return false;
  if (uint64_t(value.m) * value.n * value.k > 262144)
    return false;
  const uint64_t count = uint64_t(value.m + value.n) * value.k;
  if (count > max_values || size != header_bytes + 8 * count)
    return false;
  if (value.finite() || value.wrap()) {
    if (value.prefix || (value.operations & 4))
      return false;
  } else if (!value.prefix || value.prefix > 20)
    return false;
  if (value.finite() ? value.modulus < 2 || value.modulus > 256 : value.modulus != 0)
    return false;
  // Admission and allocation budgets precede pointer advancement/allocation.
  value.a.resize(std::size_t(value.m) * value.k);
  value.b.resize(std::size_t(value.k) * value.n);
  std::size_t offset = header_bytes;
  for (auto* values : {&value.a, &value.b})
    for (auto& item : *values) {
      item = little_integer(data + offset, 8);
      offset += 8;
      if (value.finite() && item > 255)
        return false;
    }
  out = std::move(value);
  return true;
}
inline cpp_int integer(uint64_t value, bool is_signed) {
  return is_signed ? cpp_int(signed_bits(value)) : cpp_int(value);
}
inline cpp_int magnitude(cpp_int value) {
  return value < 0 ? -value : value;
}
struct Expected {
  std::vector<cpp_int> cells;
  uint64_t bound = 0;
  bool admitted = false;
};

inline Expected oracle(const ArithmeticCase& value) {
  Expected result;
  result.cells.resize(std::size_t(value.m) * value.n);
  cpp_int bound = 0, max_a = 0, max_b = 0;
  for (auto raw : value.a)
    max_a = (std::max)(max_a, magnitude(integer(raw, value.signed_input())));
  for (auto raw : value.b)
    max_b = (std::max)(max_b, magnitude(integer(raw, value.signed_input())));
  for (uint32_t row = 0; row < value.m; ++row)
    for (uint32_t col = 0; col < value.n; ++col) {
      cpp_int sum = 0;
      for (uint32_t x = 0; x < value.k; ++x)
        sum += integer(value.a[std::size_t(row) * value.k + x], value.signed_input()) *
               integer(value.b[std::size_t(x) * value.n + col], value.signed_input());
      bound = (std::max)(bound, magnitude(sum));
      if (!value.wide() && !value.wrap() && !value.finite() &&
          (sum < (value.signed_input() ? cpp_int(INT64_MIN) : cpp_int(0)) ||
           sum > (value.signed_input() ? cpp_int(INT64_MAX) : cpp_int(UINT64_MAX))))
        return result;
      if (value.finite() || value.wrap()) {
        const cpp_int modulus = value.finite() ? cpp_int(value.modulus) : cpp_int(1) << 64;
        sum %= modulus;
        if (sum < 0)
          sum += modulus;
      }
      result.cells[std::size_t(row) * value.n + col] = sum;
    }
  if (!value.finite() && !value.wrap()) {
    // Deliberately independent literal ladder; no RNS8 reducers/CRT/selectors.
    cpp_int product = 1;
    for (uint32_t index = 0; index < value.prefix; ++index)
      product *= oracle_moduli[index];
    // A zero/cancelled product does not certify representability of its inputs.
    // Exact-wide packing admits each operand independently of the GEMM bound.
    if (value.wide() && (value.signed_input() ? cpp_int(2) * (std::max)(max_a, max_b)
                                              : (std::max)(max_a, max_b)) >= product)
      return result;
    const cpp_int proof = value.wide() ? cpp_int(value.k) * max_a * max_b : bound;
    if ((value.signed_input() ? cpp_int(2) * proof : proof) >= product)
      return result;
    if (!value.wide())
      result.bound = bound.convert_to<uint64_t>();
  }
  result.admitted = true;
  return result;
}
} // namespace rns8_fuzz
