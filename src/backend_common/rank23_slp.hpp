#pragma once
#include "rank23_program.hpp"
#include <cstdint>

// Mathematical SLP data: Stapleton/Perminov, arXiv:2609.32388; repository
// Joshua-Stapleton/51-addition-alt-basis-3x3-rank23, commit
// 473661e52ea9d6db7988e038212b4de9f34aa6e1, certificate Git blob
// 7e5d2249242ddeff8dd88ff7da80b4d4e0689d06. No upstream code is executed.
namespace rns8::detail::rank23::slp {
constexpr int capacity = 64;
struct Assignment { int target, left, right; bool subtract; }; // right=-1: alias
struct RawStage {
  int inputs = 0, outputs = 0, count = 0;
  Assignment assignments[capacity]{};
  int results[23]{};
};
struct Node { int left = 0, right = 0; bool subtract = false; };
struct Step { int target = 0, left = 0, right = 0; bool subtract = false; };
struct Plan {
  bool valid = false;
  int inputs = 0, outputs = 0, count = 0, slots = 0, source_additions = 0;
  Step steps[capacity]{};
  int results[23]{};
};
#include "rank23_slp_data.hpp"

// Mutable certificate names become immutable node IDs. CSE only identifies
// identical binary expressions (including commutative addition); no division,
// probabilistic equality or unverified reassociation is used.
constexpr Plan compile(const RawStage &raw) {
  Plan plan{};
  if (raw.inputs < 1 || raw.inputs > 23 || raw.outputs < 1 ||
      raw.outputs > 23 || raw.count < 0 || raw.count > capacity) return plan;
  int names[capacity]{}, last[capacity * 2]{}, locations[capacity * 2]{};
  Node nodes[capacity]{};
  for (int &name : names) name = -1;
  for (int &value : last) value = -1;
  for (int i = 0; i < raw.inputs; ++i) names[i] = i;
  plan.inputs = raw.inputs; plan.outputs = raw.outputs;
  for (int i = 0; i < raw.count; ++i) {
    const auto instruction = raw.assignments[i];
    if (instruction.target < 0 || instruction.target >= capacity ||
        instruction.left < 0 || instruction.left >= capacity ||
        instruction.right < -1 || instruction.right >= capacity ||
        names[instruction.left] < 0) return Plan{};
    int left = names[instruction.left];
    if (instruction.right == -1) { names[instruction.target] = left; continue; }
    if (names[instruction.right] < 0) return Plan{};
    ++plan.source_additions;
    int right = names[instruction.right];
    if (!instruction.subtract && left > right) { const int t = left; left = right; right = t; }
    int found = -1;
    for (int j = 0; j < plan.count; ++j)
      if (nodes[j].left == left && nodes[j].right == right &&
          nodes[j].subtract == instruction.subtract) { found = j; break; }
    if (found < 0) {
      if (plan.count == capacity) return Plan{};
      found = plan.count++;
      nodes[found] = {left, right, instruction.subtract};
    }
    names[instruction.target] = raw.inputs + found;
  }
  int outputs[23]{};
  for (int i = 0; i < raw.outputs; ++i) {
    if (raw.results[i] < 0 || raw.results[i] >= capacity || names[raw.results[i]] < 0) return Plan{};
    outputs[i] = names[raw.results[i]];
  }
  for (int i = 0; i < plan.count; ++i)
    last[nodes[i].left] = last[nodes[i].right] = i;
  for (int i = 0; i < raw.outputs; ++i) last[outputs[i]] = plan.count;
  bool occupied[capacity]{};
  for (int i = 0; i < raw.inputs; ++i) {
    locations[i] = i;
    occupied[i] = last[i] >= 0;
  }
  plan.slots = raw.inputs;
  for (int i = 0; i < plan.count; ++i) {
    const auto node = nodes[i];
    const int left = locations[node.left], right = locations[node.right];
    if (last[node.left] == i) occupied[left] = false;
    if (last[node.right] == i) occupied[right] = false;
    int destination = 0;
    while (destination < capacity && occupied[destination]) ++destination;
    if (destination == capacity) return Plan{};
    locations[raw.inputs + i] = destination;
    occupied[destination] = last[raw.inputs + i] >= i;
    if (destination >= plan.slots) plan.slots = destination + 1;
    plan.steps[i] = {destination, left, right, node.subtract};
  }
  for (int i = 0; i < raw.outputs; ++i) plan.results[i] = locations[outputs[i]];
  plan.valid = true;
  return plan;
}

inline constexpr Plan u = compile(raw_u()), v = compile(raw_v()), w = compile(raw_w());
static_assert(u.valid && v.valid && w.valid, "rank23 SLP must have a finite valid schedule");
static_assert(u.source_additions == 14 && v.source_additions == 15 && w.source_additions == 27,
              "51 kernel additions plus five basis additions");

// Exact linear expansion of the allocated execution plan independently checks
// every U/V/W coefficient against the already tensor-verified original program.
constexpr bool matches_coefficients() {
  const Plan plans[]{u,v,w};
  const auto original = program();
  for (int stage = 0; stage < 3; ++stage) {
    const auto &plan = plans[stage];
    for (int input = 0; input < plan.inputs; ++input) {
      int slots[capacity]{};
      slots[input] = 1;
      for (int i = 0; i < plan.count; ++i) {
        const auto op = plan.steps[i];
        const int left = slots[op.left], right = slots[op.right];
        slots[op.target] = left + (op.subtract ? -right : right);
      }
      for (int output = 0; output < plan.outputs; ++output) {
        const int expected = stage == 0 ? original.products[output].a[input] :
                             stage == 1 ? original.products[output].b[input] :
                                          original.products[input].c[output];
        if (slots[plan.results[output]] != expected) return false;
      }
    }
  }
  return true;
}
static_assert(matches_coefficients(), "allocated rank23 SLP expands to the exact tensor program");

inline int center(int value, int modulus) {
  value %= modulus;
  if (value < 0) value += modulus;
  return value >= (modulus + 1) / 2 ? value - modulus : value;
}
inline void evaluate(const Plan &plan, const int8_t *input, int8_t *output, int modulus) {
  int slots[capacity]{};
  for (int i = 0; i < plan.inputs; ++i) slots[i] = center(input[i], modulus);
  for (int i = 0; i < plan.count; ++i) {
    const auto op = plan.steps[i];
    const int left = slots[op.left], right = slots[op.right];
    slots[op.target] = center(left + (op.subtract ? -right : right), modulus);
  }
  for (int i = 0; i < plan.outputs; ++i) output[i] = static_cast<int8_t>(slots[plan.results[i]]);
}
inline bool multiply3x3(const int8_t *a, const int8_t *b, int8_t *output, uint32_t modulus) {
  if (!a || !b || !output || modulus < 2 || modulus > 256) return false;
  int8_t left[23]{}, right[23]{}, product[23]{}, staged[9]{};
  evaluate(u, a, left, modulus); evaluate(v, b, right, modulus);
  for (int i = 0; i < 23; ++i) product[i] = static_cast<int8_t>(center(int(left[i])*right[i], modulus));
  evaluate(w, product, staged, modulus);
  for (int i = 0; i < 9; ++i) output[i] = staged[i];
  return true;
}
} // namespace rns8::detail::rank23::slp
