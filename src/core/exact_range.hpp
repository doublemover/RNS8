#ifndef RNS8_CORE_EXACT_RANGE_HPP
#define RNS8_CORE_EXACT_RANGE_HPP

#include <algorithm>

#include "core/internal.hpp"

namespace rns8::detail {

using boost::multiprecision::cpp_int;

inline bool exact_wide_semantics(rns8_semantics semantics) {
  return semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED;
}

inline bool exact_range_fits(rns8_semantics semantics, const cpp_int& bound, const cpp_int& product) {
  return bound >= 0 && (semantics == RNS8_EXACT_WIDE_SIGNED ? bound * 2 < product : bound < product);
}

inline void tighten_exact_bound(cpp_int& bound, cpp_int candidate) {
  if (candidate < bound) bound.swap(candidate);
}

inline cpp_int exact_axis_center(const rns8_exact_axis_range& axis) {
  return (axis.minimum + axis.maximum) / 2;
}

inline cpp_int exact_axis_deviation_max(const rns8_exact_axis_range& axis, const cpp_int& center) {
  cpp_int low = boost::multiprecision::abs(axis.minimum - center);
  cpp_int high = boost::multiprecision::abs(axis.maximum - center);
  return std::max(low, high);
}

inline void add_exact_native_value(rns8_exact_axis_range& axis, const cpp_int& value, bool first) {
  const cpp_int magnitude = boost::multiprecision::abs(value);
  axis.max_magnitude = std::max(axis.max_magnitude, magnitude);
  axis.absolute_sum += magnitude;
  axis.sum += value;
  axis.squared_sum += value * value;
  axis.squared_sum_known = true;
  if (first) {
    axis.minimum = value;
    axis.maximum = value;
  } else {
    axis.minimum = std::min(axis.minimum, value);
    axis.maximum = std::max(axis.maximum, value);
  }
  axis.sum_known = true;
}

// All arithmetic is multiprecision, including sums exceeding UINT64_MAX.
// Stage before packing; logical padding never contributes to the proof.
template <typename T>
rns8_exact_matrix_ranges native_exact_ranges(const T* src, int64_t rows, int64_t cols, int64_t ld,
                                             cpp_int& maximum) {
  rns8_exact_matrix_ranges ranges;
  ranges.rows.resize(static_cast<std::size_t>(rows));
  ranges.cols.resize(static_cast<std::size_t>(cols));
  maximum = 0;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      const cpp_int value = src[row * ld + col];
      add_exact_native_value(ranges.rows[row], value, col == 0);
      add_exact_native_value(ranges.cols[col], value, row == 0);
    }
    maximum = std::max(maximum, ranges.rows[row].max_magnitude);
  }
  // Axis extrema are final after the first scan. Compute each column midpoint
  // once rather than repeating multiprecision addition/division for every row.
  std::vector<cpp_int> column_centers;
  column_centers.reserve(static_cast<std::size_t>(cols));
  for (const auto& column : ranges.cols) column_centers.push_back(exact_axis_center(column));
  // A second linear scan supplies the L1 distance from each axis midpoint.
  for (int64_t row = 0; row < rows; ++row) {
    const cpp_int row_center = exact_axis_center(ranges.rows[row]);
    for (int64_t col = 0; col < cols; ++col) {
      const cpp_int value = src[row * ld + col];
      ranges.rows[row].deviation_sum += boost::multiprecision::abs(value - row_center);
      ranges.cols[col].deviation_sum +=
          boost::multiprecision::abs(value - column_centers[col]);
    }
  }
  return ranges;
}

inline cpp_int exact_axis_product_bound(const rns8_exact_axis_range& a, const rns8_exact_axis_range& b,
                                        int64_t k) {
  cpp_int bound = cpp_int(k) * a.max_magnitude * b.max_magnitude;
  tighten_exact_bound(bound, a.absolute_sum * b.max_magnitude);
  tighten_exact_bound(bound, a.max_magnitude * b.absolute_sum);
  if (a.squared_sum_known && b.squared_sum_known) {
    // Cauchy-Schwarz with integer rounding: an integer |dot| below a real
    // square root is also below its floor. No floating-point approximation.
    const cpp_int squares = a.squared_sum * b.squared_sum;
    if (squares < bound * bound)
      tighten_exact_bound(bound, boost::multiprecision::sqrt(squares));
    if (a.sum_known && b.sum_known) {
      // K*dot - sum(a)*sum(b) is the centered covariance numerator.
      // The variance numerators are nonnegative even when squared sums are
      // upper bounds. Exact signed sums, however, are mandatory here.
      const cpp_int va = cpp_int(k) * a.squared_sum - a.sum * a.sum;
      const cpp_int vb = cpp_int(k) * b.squared_sum - b.sum * b.sum;
      const cpp_int variance_product = va * vb;
      const cpp_int center = boost::multiprecision::abs(a.sum * b.sum);
      const cpp_int gap = cpp_int(k) * bound - center;
      // Avoid integer roots that cannot improve the already established bound.
      if (gap > 0 && variance_product < gap * gap) {
        const cpp_int remainder = boost::multiprecision::sqrt(variance_product);
        tighten_exact_bound(bound, (center + remainder) / k);
      }
    }
  }
  if (a.sum_known && b.sum_known) {
    const cpp_int ca = exact_axis_center(a);
    const cpp_int cb = exact_axis_center(b);
    const cpp_int da = exact_axis_deviation_max(a, ca);
    const cpp_int db = exact_axis_deviation_max(b, cb);
    cpp_int remainder = cpp_int(k) * da * db;
    tighten_exact_bound(remainder, a.deviation_sum * db);
    tighten_exact_bound(remainder, da * b.deviation_sum);
    // dot(a,b) = ca*sum(b) + cb*sum(a) - K*ca*cb + dot(a-ca,b-cb).
    // Exact sums are mandatory: output magnitude bounds are not signed sums.
    const cpp_int center_term = ca * b.sum + cb * a.sum - cpp_int(k) * ca * cb;
    tighten_exact_bound(bound, boost::multiprecision::abs(center_term) + remainder);
  }
  return bound;
}

inline rns8_exact_axis_range uniform_exact_axis(const cpp_int& maximum, int64_t length) {
  rns8_exact_axis_range axis;
  axis.max_magnitude = maximum;
  axis.absolute_sum = cpp_int(length) * maximum;
  return axis;
}

// Pure bound construction shared by ordinary admission and the explicit CPU
// selection route. It does not authorize using any unwritten residue plane.
inline void exact_gemm_bound(const rns8_plan& plan, const rns8_matrix& A, const rns8_matrix& B,
                             cpp_int& output_bound, rns8_exact_matrix_ranges* output_ranges = nullptr) {
  const bool a_rows = A.exact_axis_ranges.rows.size() == static_cast<std::size_t>(plan.desc.m);
  const bool b_cols = B.exact_axis_ranges.cols.size() == static_cast<std::size_t>(plan.desc.n);
  output_bound = cpp_int(plan.desc.k) * A.exact_max_magnitude * B.exact_max_magnitude;
  if (output_ranges) {
    output_ranges->rows.clear();
    output_ranges->cols.clear();
  }
  if (a_rows || b_cols) {
    const auto fallback_a = uniform_exact_axis(A.exact_max_magnitude, plan.desc.k);
    const auto fallback_b = uniform_exact_axis(B.exact_max_magnitude, plan.desc.k);
    if (output_ranges) {
      output_ranges->rows.resize(static_cast<std::size_t>(plan.desc.m));
      output_ranges->cols.resize(static_cast<std::size_t>(plan.desc.n));
    }
    cpp_int maximum = 0;
    for (int64_t row = 0; row < plan.desc.m; ++row) {
      const auto& a = a_rows ? A.exact_axis_ranges.rows[row] : fallback_a;
      for (int64_t col = 0; col < plan.desc.n; ++col) {
        const auto& b = b_cols ? B.exact_axis_ranges.cols[col] : fallback_b;
        const cpp_int bound = exact_axis_product_bound(a, b, plan.desc.k);
        maximum = std::max(maximum, bound);
        if (output_ranges) {
          auto& r = output_ranges->rows[row];
          auto& c = output_ranges->cols[col];
          r.max_magnitude = std::max(r.max_magnitude, bound);
          c.max_magnitude = std::max(c.max_magnitude, bound);
          r.absolute_sum += bound;
          c.absolute_sum += bound;
          const cpp_int squared = bound * bound;
          r.squared_sum += squared;
          c.squared_sum += squared;
          r.squared_sum_known = true;
          c.squared_sum_known = true;
        }
      }
    }
    tighten_exact_bound(output_bound, maximum);
  }
}

inline rns8_status exact_gemm_range(const rns8_plan& plan, const rns8_matrix& A, const rns8_matrix& B,
                                    cpp_int& output_bound,
                                    rns8_exact_matrix_ranges* output_ranges = nullptr) {
  if (!exact_wide_semantics(plan.desc.semantics)) return RNS8_SUCCESS;
  if (A.exact_range_prefix < plan.prefix || B.exact_range_prefix < plan.prefix) return RNS8_RANGE_ERROR;
  if ((plan.desc.flags & RNS8_PLAN_EXACT_SCALAR_FIRST) != 0) {
    output_bound = cpp_int(plan.desc.k) * A.exact_max_magnitude * B.exact_max_magnitude;
    if (exact_range_fits(plan.desc.semantics, output_bound, plan.modulus_product)) {
      if (output_ranges) { output_ranges->rows.clear(); output_ranges->cols.clear(); }
      return RNS8_SUCCESS;
    }
  }
  exact_gemm_bound(plan, A, B, output_bound, output_ranges);
  return exact_range_fits(plan.desc.semantics, output_bound, plan.modulus_product) ? RNS8_SUCCESS
                                                                                   : RNS8_RANGE_ERROR;
}

// Admit the complete expression, without requiring the unpublished AB to fit.
inline rns8_status exact_axpby_range(const rns8_plan& plan, const rns8_matrix& A,
                                    const rns8_matrix& B, const rns8_matrix& D,
                                    int64_t alpha, int64_t beta, cpp_int& bound) {
  if (!exact_wide_semantics(plan.desc.semantics) ||
      (plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED && (alpha < 0 || beta < 0)))
    return RNS8_INVALID_ARGUMENT;
  if (A.exact_range_prefix < plan.prefix || B.exact_range_prefix < plan.prefix ||
      D.exact_range_prefix < plan.prefix) return RNS8_RANGE_ERROR;
  // Only an original integer zero removes the AB bound. A coefficient that
  // vanishes in one residue channel still contributes its full integer magnitude.
  bound=0;
  if(alpha!=0) {
    cpp_int product_bound;
    exact_gemm_bound(plan,A,B,product_bound);
    bound=boost::multiprecision::abs(cpp_int(alpha))*product_bound;
  }
  if(beta!=0)bound+=boost::multiprecision::abs(cpp_int(beta))*D.exact_max_magnitude;
  return exact_range_fits(plan.desc.semantics, bound, plan.modulus_product)
             ? RNS8_SUCCESS : RNS8_RANGE_ERROR;
}

inline void commit_exact_range(rns8_matrix& matrix, const rns8_plan& plan, cpp_int& bound,
                               rns8_exact_matrix_ranges* ranges = nullptr) {
  if (exact_wide_semantics(plan.desc.semantics)) {
    matrix.exact_max_magnitude.swap(bound);
    matrix.exact_axis_ranges.rows.clear();
    matrix.exact_axis_ranges.cols.clear();
    if (ranges) {
      matrix.exact_axis_ranges.rows.swap(ranges->rows);
      matrix.exact_axis_ranges.cols.swap(ranges->cols);
    }
    matrix.exact_range_prefix = plan.prefix;
  }
}

}  // namespace rns8::detail
#endif
