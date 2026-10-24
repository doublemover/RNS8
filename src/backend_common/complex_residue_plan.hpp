#pragma once
#include "address_domain.hpp"
#include "rns8/rns8.h"
#include <array>
#include <boost/multiprecision/cpp_int.hpp>

namespace rns8::detail {
enum class ComplexResidueDomain { bounded_signed, wide_signed, finite_ring };
enum class ComplexProductMethod { three_real, four_real };
struct ComplexResidueBasis {
  ComplexResidueDomain domain = ComplexResidueDomain::wide_signed;
  uint32_t version = 1, prefix = 0, modulus = 0;
  bool operator==(const ComplexResidueBasis &b) const {
    return domain == b.domain && version == b.version && prefix == b.prefix && modulus == b.modulus;
  }
};
template <class Byte> struct ComplexResidueView {
  Byte *data = nullptr;
  int64_t rows = 0, cols = 0, ld = 0;
  ComplexResidueBasis basis;
};
template <class Byte> struct ComplexResiduePair {
  ComplexResidueView<Byte> real, imag;
};
// Original-component magnitude certificates, not bounds on transformed panels.
// The caller binds these to immutable input values; no proof is inferred from a
// pointer or from the signed storage byte. Finite rings need no integer bound.
struct ComplexMagnitudeBounds {
  boost::multiprecision::cpp_int ar, ai, br, bi;
};
struct ComplexResidueScratch {
  uint64_t sum_a_bytes = 0, sum_b_bytes = 0, product_bytes = 0, scratch_bytes = 0;
  uint32_t products = 0;
};
struct ComplexResiduePlan : ComplexResidueScratch {
  rns8_status status = RNS8_INVALID_ARGUMENT;
  boost::multiprecision::cpp_int real_bound, imag_bound;
};
inline bool valid_complex_basis(const ComplexResidueBasis &basis) {
  if (basis.version != 1)
    return false;
  if (basis.domain == ComplexResidueDomain::finite_ring)
    return basis.prefix == 1 && basis.modulus >= 2 && basis.modulus <= 256;
  return (basis.domain == ComplexResidueDomain::bounded_signed ||
          basis.domain == ComplexResidueDomain::wide_signed) &&
         basis.prefix >= 1 && basis.prefix <= 20 && basis.modulus == 0;
}
inline ComplexResiduePlan plan_complex_residue_product(const ComplexResiduePair<const int8_t> &a,
                                                       const ComplexResiduePair<const int8_t> &b,
                                                       const ComplexResiduePair<int8_t> &c,
                                                       const ComplexMagnitudeBounds &bounds,
                                                       ComplexProductMethod method,
                                                       uint64_t budget) {
  using boost::multiprecision::cpp_int;
  ComplexResiduePlan plan;
  if ((method != ComplexProductMethod::three_real && method != ComplexProductMethod::four_real) ||
      !valid_complex_basis(a.real.basis))
    return plan;
  const auto &basis = a.real.basis;
  const auto same_pair = [&](const auto &pair) {
    return pair.real.rows == pair.imag.rows && pair.real.cols == pair.imag.cols &&
           pair.real.basis == basis && pair.imag.basis == basis;
  };
  if (!same_pair(a) || !same_pair(b) || !same_pair(c) || a.real.cols != b.real.rows ||
      c.real.rows != a.real.rows || c.real.cols != b.real.cols)
    return plan;
  struct Span {
    const void *data;
    uint64_t bytes;
  };
  std::array<Span, 6> spans{};
  std::size_t index = 0;
  const auto admit = [&](const auto &view) {
    AddressDomain domain{};
    if (!view.data || view.rows <= 0 || view.cols <= 0 || view.ld < view.cols)
      return false;
    if (!qualified_i32_matrix_address(view.rows, view.cols, view.ld, basis.prefix) ||
        !checked_host_plane_address(view.rows, view.cols, view.ld, basis.prefix, 1, domain))
      return false;
    spans[index++] = {view.data, domain.byte_extent};
    return true;
  };
  if (!admit(a.real) || !admit(a.imag) || !admit(b.real) || !admit(b.imag) || !admit(c.real) ||
      !admit(c.imag))
    return plan;
  for (std::size_t output = 4; output < 6; ++output)
    for (std::size_t other = 0; other < output; ++other)
      if (!nonoverlapping_byte_ranges(spans[output].data, spans[output].bytes, spans[other].data,
                                      spans[other].bytes))
        return plan;
  if (basis.domain != ComplexResidueDomain::finite_ring) {
    if (bounds.ar < 0 || bounds.ai < 0 || bounds.br < 0 || bounds.bi < 0)
      return plan;
    plan.real_bound = cpp_int(a.real.cols) * (bounds.ar * bounds.br + bounds.ai * bounds.bi);
    plan.imag_bound = cpp_int(a.real.cols) * (bounds.ar * bounds.bi + bounds.ai * bounds.br);
    cpp_int product = 1;
    for (uint32_t p = 0; p < basis.prefix; ++p)
      product *= rns8_default_modulus(p);
    if (product <= 2 * plan.real_bound || product <= 2 * plan.imag_bound ||
        (basis.domain == ComplexResidueDomain::bounded_signed &&
         (plan.real_bound > INT64_MAX || plan.imag_bound > INT64_MAX))) {
      plan.status = RNS8_RANGE_ERROR;
      return plan;
    }
  }
  plan.products = method == ComplexProductMethod::three_real ? 3 : 4;
  plan.product_bytes = uint64_t(c.real.rows) * c.real.cols * basis.prefix;
  if (method == ComplexProductMethod::three_real) {
    plan.sum_a_bytes = uint64_t(a.real.rows) * a.real.cols * basis.prefix;
    plan.sum_b_bytes = uint64_t(b.real.rows) * b.real.cols * basis.prefix;
  }
  // All per-matrix extents are i32-qualified. Still prove host-size accumulation.
  uint64_t total = 0;
  for (uint64_t bytes :
       {plan.sum_a_bytes, plan.sum_b_bytes, plan.product_bytes * (plan.products + 2)}) {
    if (bytes > SIZE_MAX - total) {
      plan.status = RNS8_RANGE_ERROR;
      return plan;
    }
    total += bytes;
  }
  plan.scratch_bytes = total;
  plan.status = total > budget ? RNS8_WORKSPACE_TOO_SMALL : RNS8_SUCCESS;
  return plan;
}
} // namespace rns8::detail
