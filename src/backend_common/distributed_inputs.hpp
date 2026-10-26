#pragma once
#include "distributed_plan.hpp"

namespace rns8::detail::distributed {
// Native snapshots are immutable, generation-bound values shared by the mock
// and real runtime. Signedness is explicit; uint64_t carries bits in either case.
class Inputs {
 public:
  Inputs(ValueIdentity a, ValueIdentity b, std::vector<uint64_t> lhs, std::vector<uint64_t> rhs)
      : a_(a), b_(b), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}
  const ValueIdentity& a() const { return a_; }
  const ValueIdentity& b() const { return b_; }
  const std::vector<uint64_t>& lhs() const { return lhs_; }
  const std::vector<uint64_t>& rhs() const { return rhs_; }
 private:
  const ValueIdentity a_, b_;
  const std::vector<uint64_t> lhs_, rhs_;
};
inline Integer native_integer(uint64_t bits, bool signed_value) {
  Integer value = bits;
  if (signed_value && (bits >> 63)) value -= Integer(1) << 64;
  return value;
}
inline uint32_t native_residue(uint64_t bits, bool signed_value, uint32_t modulus) {
  uint32_t result = static_cast<uint32_t>(bits % modulus);
  if (signed_value && (bits >> 63)) {
    const auto radix = static_cast<uint32_t>(((UINT64_MAX % modulus) + 1) % modulus);
    result = (result + modulus - radix) % modulus;
  }
  return result;
}
// Prove uniqueness in every actual tile basis from immutable inputs before
// device work. The checked GPU export separately enforces the claimed output
// bound. This admits large cancellation with a sufficient basis and prevents a
// false small bound from disguising an integer as a plausible CRT alias.
inline Admission validate_inputs(const Plan& plan, const Inputs& inputs) {
  const auto& r = plan.request();
  AddressDomain a, b;
  if (!(inputs.a() == r.a) || !(inputs.b() == r.b) ||
      !checked_host_plane_address(r.m, r.k, r.lda, 1, 8, a) ||
      !checked_host_plane_address(r.k, r.n, r.ldb, 1, 8, b) ||
      inputs.lhs().size() != a.byte_extent / 8 || inputs.rhs().size() != b.byte_extent / 8)
    return reject(RNS8_INVALID_ARGUMENT, "input_identity_or_extent");
  try {
    std::vector<Integer> row_max(r.m), row_sum(r.m), col_max(r.n), col_sum(r.n);
    const auto magnitude = [&](uint64_t bits) -> Integer {
      Integer value = native_integer(bits, plan.signed_value()); return value < 0 ? -value : value;
    };
    for (uint64_t row = 0; row < r.m; ++row) for (uint64_t k = 0; k < r.k; ++k) {
      const auto value = magnitude(inputs.lhs()[row * r.lda + k]);
      row_max[row] = std::max(row_max[row], value); row_sum[row] += value;
    }
    for (uint64_t k = 0; k < r.k; ++k) for (uint64_t col = 0; col < r.n; ++col) {
      const auto value = magnitude(inputs.rhs()[k * r.ldb + col]);
      col_max[col] = std::max(col_max[col], value); col_sum[col] += value;
    }
    for (const auto& tile : plan.tiles()) {
      Integer am = 0, as = 0, bm = 0, bs = 0;
      for (uint64_t row = tile.row; row < tile.row + tile.rows; ++row) {
        am = std::max(am, row_max[row]); as = std::max(as, row_sum[row]);
      }
      for (uint64_t col = tile.col; col < tile.col + tile.cols; ++col) {
        bm = std::max(bm, col_max[col]); bs = std::max(bs, col_sum[col]);
      }
      const Integer left = am * bs, right = as * bm;
      Integer product = 1;
      for (uint32_t plane = 0; plane < tile.prefix; ++plane) product *= garner::kConstants.moduli[plane];
      if (product <= std::min(left, right) * (plan.signed_value() ? 2 : 1))
        return reject(RNS8_RANGE_ERROR, "native_uniqueness_basis_insufficient");
    }
    return {};
  } catch (const std::bad_alloc&) { return reject(RNS8_INTERNAL_ERROR, "input_proof_allocation"); }
}
} // namespace rns8::detail::distributed
