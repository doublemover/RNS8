#ifndef RNS8_CORE_EXACT_RANGE_HPP
#define RNS8_CORE_EXACT_RANGE_HPP

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>

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

// An axis contains at most INT64_MAX native values. Its absolute/signed/deviation
// sums are below 2^127, and its squared sum is below 2^191, even for UINT64_MAX.
// Fixed accumulators therefore lose no information and allocate no limb storage
// per input value. Publish the same arbitrary-precision facts once per axis.
template <typename T>
struct NativeAxisAccumulator {
  using u128 = boost::multiprecision::uint128_t;
  using i128 = boost::multiprecision::int128_t;
  using u256 = boost::multiprecision::uint256_t;
  static_assert(std::is_integral_v<T> && sizeof(T)<=sizeof(uint64_t),"native integer width");
  T minimum=std::numeric_limits<T>::max(),maximum=std::numeric_limits<T>::lowest(),center=0;
  uint64_t max_magnitude=0;
  u128 absolute_sum=0,deviation_sum=0;
  i128 sum=0;
  u256 squared_sum=0;
  bool nonempty=false;

  void add(T value) {
    const auto bits=static_cast<uint64_t>(value);
    uint64_t magnitude=bits;
    if constexpr(std::is_signed_v<T>)if(value<0)magnitude=uint64_t(0)-bits;
    max_magnitude=std::max(max_magnitude,magnitude);
    minimum=std::min(minimum,value);maximum=std::max(maximum,value);nonempty=true;
    absolute_sum+=magnitude;sum+=value;
    const u128 square=u128(magnitude)*magnitude;
    squared_sum+=square;
  }
  void set_center() {
    if(nonempty) {
      const i128 midpoint=(i128(minimum)+i128(maximum))/2;
      center=midpoint.template convert_to<T>();
    }
  }
  void add_deviation(T value) {
    // Unsigned subtraction gives the exact nonnegative distance, including the
    // complete signed INT64_MIN..INT64_MAX span, without signed overflow.
    const auto a=static_cast<uint64_t>(value),b=static_cast<uint64_t>(center);
    deviation_sum+=value>=center?a-b:b-a;
  }
  rns8_exact_axis_range finish() const {
    rns8_exact_axis_range out;
    if(nonempty) {
      out.max_magnitude=max_magnitude;out.absolute_sum=absolute_sum;out.sum=sum;
      out.minimum=minimum;out.maximum=maximum;out.deviation_sum=deviation_sum;
      out.squared_sum=squared_sum;out.squared_sum_known=out.sum_known=true;
    }
    return out;
  }
};

// Stage before packing; logical padding never contributes to the proof. Rows
// and bounded column blocks each scan twice. This avoids an O(rows+cols) native
// scratch mirror and keeps the retained proof representation unchanged.
template <typename T>
rns8_exact_matrix_ranges native_exact_ranges(const T* src, int64_t rows, int64_t cols, int64_t ld,
                                             cpp_int& maximum) {
  rns8_exact_matrix_ranges ranges;
  ranges.rows.resize(static_cast<std::size_t>(rows));
  ranges.cols.resize(static_cast<std::size_t>(cols));
  if(rows==0 || cols==0){maximum=0;return ranges;}
  uint64_t maximum_native=0;
  for (int64_t row = 0; row < rows; ++row) {
    NativeAxisAccumulator<T> axis;
    const T* input=src+row*ld;
    for(int64_t col=0;col<cols;++col)axis.add(input[col]);
    axis.set_center();
    for(int64_t col=0;col<cols;++col)axis.add_deviation(input[col]);
    maximum_native=std::max(maximum_native,axis.max_magnitude);
    ranges.rows[static_cast<size_t>(row)]=axis.finish();
  }
  constexpr int64_t block_columns=16;
  for(int64_t first=0;first<cols;) {
    const auto count=std::min(block_columns,cols-first);
    std::array<NativeAxisAccumulator<T>,block_columns> axes;
    for(int64_t row=0;row<rows;++row) {
      const T* input=src+row*ld+first;
      for(int64_t lane=0;lane<count;++lane)axes[static_cast<size_t>(lane)].add(input[lane]);
    }
    for(int64_t lane=0;lane<count;++lane)axes[static_cast<size_t>(lane)].set_center();
    for(int64_t row=0;row<rows;++row) {
      const T* input=src+row*ld+first;
      for(int64_t lane=0;lane<count;++lane)axes[static_cast<size_t>(lane)].add_deviation(input[lane]);
    }
    for(int64_t lane=0;lane<count;++lane)ranges.cols[static_cast<size_t>(first+lane)]=axes[static_cast<size_t>(lane)].finish();
    first+=count;
  }
  maximum=maximum_native;
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
