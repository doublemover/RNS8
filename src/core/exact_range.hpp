#ifndef RNS8_CORE_EXACT_RANGE_HPP
#define RNS8_CORE_EXACT_RANGE_HPP

#include "core/internal.hpp"

#include <algorithm>
#include <limits>
#include <type_traits>

namespace rns8::detail {

inline bool exact_wide_semantics(rns8_semantics semantics) {
  return semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED;
}

inline bool exact_range_fits(
    rns8_semantics semantics, const boost::multiprecision::cpp_int& bound,
    const boost::multiprecision::cpp_int& product) {
  return bound >= 0 && (semantics == RNS8_EXACT_WIDE_SIGNED ? bound * 2 < product : bound < product);
}

// Pack inputs are host-native integers. Inspect logical cells only; computing
// the magnitude in unsigned arithmetic handles INT64_MIN without signed UB.
template <typename T>
uint64_t native_max_magnitude(const T* src, int64_t rows, int64_t cols, int64_t ld) {
  uint64_t maximum = 0;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      const T value = src[row * ld + col];
      uint64_t magnitude = static_cast<uint64_t>(value);
      if constexpr (std::is_signed_v<T>) {
        if (value < 0) magnitude = uint64_t{0} - magnitude;
      }
      maximum = std::max(maximum, magnitude);
    }
  }
  return maximum;
}

inline rns8_status exact_gemm_range(
    const rns8_plan& plan, const rns8_matrix& A, const rns8_matrix& B,
    boost::multiprecision::cpp_int& output_bound) {
  if (!exact_wide_semantics(plan.desc.semantics)) return RNS8_SUCCESS;
  if (A.exact_range_prefix < plan.prefix || B.exact_range_prefix < plan.prefix) {
    return RNS8_RANGE_ERROR;
  }
  output_bound = boost::multiprecision::cpp_int(plan.desc.k) * A.exact_max_magnitude * B.exact_max_magnitude;
  return exact_range_fits(plan.desc.semantics, output_bound, plan.modulus_product)
             ? RNS8_SUCCESS : RNS8_RANGE_ERROR;
}

inline void commit_exact_range(
    rns8_matrix& matrix, const rns8_plan& plan,
    boost::multiprecision::cpp_int& bound) {
  if (exact_wide_semantics(plan.desc.semantics)) {
    matrix.exact_max_magnitude.swap(bound);
    matrix.exact_range_prefix = plan.prefix;
  }
}

}  // namespace rns8::detail
#endif
