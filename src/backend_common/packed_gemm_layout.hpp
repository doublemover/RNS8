#pragma once

#include "checked_arithmetic.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rns8::detail {

struct PackedGemmLayout {
  uint64_t rows = 0, cols = 0, k = 0, a_bytes = 0, b_bytes = 0, c_bytes = 0, total_bytes = 0;
};

// These adapters still pass signed32 dimensions into the matrix library and
// source-owned pack/copy kernels. Qualify the generated domains, not only the
// original matrix. The rounding margin also admits inactive launch lanes.
inline bool packed_gemm_layout(uint64_t rows, uint64_t cols, uint64_t k, uint64_t shape_alignment,
                               uint64_t minimum_k, uint64_t k_alignment, uint64_t safe_k,
                               bool temporary_output, PackedGemmLayout& output) noexcept {
  constexpr uint64_t limit = uint64_t(INT32_MAX) - 255;
  if (!rows || !cols || !k || !shape_alignment || !minimum_k || !k_alignment || !safe_k)
    return false;
  const auto round = [](uint64_t value, uint64_t alignment, uint64_t& result) {
    const uint64_t remainder = value % alignment;
    return checked::add(value, remainder ? alignment - remainder : 0, result);
  };
  PackedGemmLayout plan;
  if (!round(rows, shape_alignment, plan.rows) || !round(cols, shape_alignment, plan.cols) ||
      !round(std::max(minimum_k, std::min(k, safe_k)), k_alignment, plan.k) || plan.rows > limit ||
      plan.cols > limit || plan.k > limit || !checked::multiply(plan.rows, plan.k, plan.a_bytes) ||
      !checked::multiply(plan.cols, plan.k, plan.b_bytes) ||
      (temporary_output && !checked::multiply(plan.rows, plan.cols, plan.c_bytes)) ||
      plan.a_bytes > limit || plan.b_bytes > limit || plan.c_bytes > limit ||
      !checked::add(plan.a_bytes, plan.b_bytes, plan.total_bytes) ||
      !checked::add(plan.total_bytes, plan.c_bytes, plan.total_bytes) ||
      plan.total_bytes > std::numeric_limits<std::size_t>::max())
    return false;
  output = plan;
  return true;
}

inline bool ck_packed_gemm_layout(uint64_t m, uint64_t n, uint64_t k, uint64_t alignment,
                                  PackedGemmLayout& out) noexcept {
  return (alignment == 64 || alignment == 128) &&
         packed_gemm_layout(m, n, k, alignment, 64, 1, 32768, true, out);
}

inline bool rocwmma_packed_gemm_layout(uint64_t m, uint64_t n, uint64_t k,
                                       PackedGemmLayout& out) noexcept {
  return packed_gemm_layout(m, n, k, 16, 16, 16, 65536, false, out);
}

} // namespace rns8::detail
