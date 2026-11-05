#pragma once
#include "implementation_contract.hpp"
#include "backend_common/arithmetic_certificate.hpp"

namespace rns8::detail {
// Use the common checker: reductions prove the modular equivalence between
// logical centered residues and the unsigned physical CK operand alphabet.
inline bool append_ck_arithmetic_contract(arithmetic::Certificate& c, const CkImplementation& d,
                                          const std::vector<uint32_t>& moduli, uint64_t slab_k) {
  using namespace arithmetic;
  if (!d.valid() || !slab_k || moduli.empty())
    return false;
  auto& i = c.instruction;
  i.operand_bits = d.operand_bits;
  i.accumulator_bits = d.accumulator_bits;
  i.signed_operands = d.signed_operands;
  i.signed_accumulator = d.signed_accumulator;
  i.operand_encoding =
      d.signed_operands ? OperandEncoding::centered_signed : OperandEncoding::canonical_unsigned;
  i.safe_k = slab_k;
  i.prior_abs = 0;
  i.lhs_abs = i.rhs_abs = 0;
  i.lowering_identity += ';' + d.lowering_identity();
  const auto append = [&](Node n) {
    c.dag.push_back(std::move(n));
    return static_cast<uint32_t>(c.dag.size() - 1);
  };
  for (auto q : moduli) {
    if (q < 2 || q > 256)
      return false;
    i.lhs_abs = std::max(i.lhs_abs, d.max_input(q));
    i.rhs_abs = i.lhs_abs;
    Node logical;
    logical.input = {-Integer(q / 2), Integer((q - 1) / 2), 1, q};
    logical.bits = 8;
    logical.equality = Equality::modular_value;
    auto a = append(logical), b = append(logical);
    if (!d.signed_operands) {
      Node physical;
      physical.operation = Operation::canonical_reduce;
      physical.equality = Equality::modular_value;
      physical.modulus = q;
      physical.bits = 8;
      physical.is_signed = false;
      physical.left = a;
      a = append(physical);
      physical.left = b;
      b = append(physical);
    }
    Node dot;
    dot.operation = Operation::dot;
    dot.equality = Equality::modular_value;
    dot.left = a;
    dot.right = b;
    dot.count = slab_k;
    Node reduced;
    reduced.operation = Operation::centered_reduce;
    reduced.equality = Equality::modular_value;
    reduced.left = append(dot);
    reduced.modulus = q;
    reduced.bits = 8;
    const auto current = append(reduced), prior = append(logical);
    Node merge;
    merge.operation = Operation::add;
    merge.equality = Equality::modular_value;
    merge.left = current;
    merge.right = prior;
    reduced.left = append(merge);
    append(reduced);
  }
  return true;
}
} // namespace rns8::detail
