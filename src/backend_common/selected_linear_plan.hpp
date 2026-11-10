#pragma once
#include <cstdint>

namespace rns8::detail {
// Sources name immutable definitions, never reusable physical slots.
struct LinearDefinition {
  int left = -1, right = -1;
  int8_t left_sign = 1, right_sign = 1;
  bool halve = false;
};
struct LinearDag {
  unsigned inputs = 0, count = 0, outputs = 0;
  LinearDefinition nodes[90]{};
  unsigned results[48]{};
};
struct SelectedLinearPlan {
  bool valid = false;
  unsigned input_count = 0, count = 0, slots = 0, result = 0;
  uint8_t input_ids[48]{};
  struct Step {
    uint8_t target = 0, left = 0, right = 0;
    int8_t left_sign = 1, right_sign = 1;
    bool halve = false;
  };
  Step steps[90]{};
};
constexpr SelectedLinearPlan select_linear_output(const LinearDag& dag, unsigned output) {
  SelectedLinearPlan plan;
  if (!dag.inputs || dag.inputs > 48 || dag.count > 90 || output >= dag.outputs)
    return plan;
  constexpr unsigned capacity = 138;
  const auto root = dag.results[output];
  if (root >= dag.inputs + dag.count)
    return plan;
  bool needed[capacity]{};
  needed[root] = true;
  // Validate every definition before using it as an index, including dead code.
  for (unsigned index = 0; index < dag.count; ++index) {
    const auto node = dag.nodes[index];
    const int id = int(dag.inputs + index);
    if (node.left < 0 || node.left >= id || node.right >= id || node.right < -1 ||
        (node.left_sign != 1 && node.left_sign != -1) ||
        (node.right_sign != 1 && node.right_sign != -1) || (node.halve && node.right >= 0))
      return plan;
  }
  for (unsigned index = dag.count; index > 0; --index)
    if (needed[dag.inputs + index - 1]) {
      const auto node = dag.nodes[index - 1];
      needed[node.left] = true;
      if (node.right >= 0)
        needed[node.right] = true;
    }
  int location[capacity]{}, last[capacity]{};
  for (unsigned id = 0; id < capacity; ++id) {
    location[id] = -1;
    last[id] = -1;
  }
  for (unsigned index = 0; index < dag.count; ++index)
    if (needed[dag.inputs + index]) {
      const auto node = dag.nodes[index];
      last[node.left] = int(plan.count);
      if (node.right >= 0)
        last[node.right] = int(plan.count);
      ++plan.count;
    }
  last[root] = int(plan.count);
  bool occupied[48]{};
  for (unsigned input = 0; input < dag.inputs; ++input)
    if (needed[input]) {
      location[input] = int(plan.input_count);
      occupied[plan.input_count] = true;
      plan.input_ids[plan.input_count++] = uint8_t(input);
    }
  plan.slots = plan.input_count;
  unsigned emitted = 0;
  for (unsigned index = 0; index < dag.count; ++index)
    if (needed[dag.inputs + index]) {
      const auto node = dag.nodes[index];
      const auto a = location[node.left];
      const auto b = node.right >= 0 ? location[node.right] : -1;
      if (a < 0 || (node.right >= 0 && b < 0))
        return {};
      if (last[node.left] == int(emitted))
        occupied[a] = false;
      if (node.right >= 0 && last[node.right] == int(emitted))
        occupied[b] = false;
      unsigned target = 0;
      while (target < 48 && occupied[target])
        ++target;
      if (target == 48)
        return {};
      occupied[target] = true;
      location[dag.inputs + index] = int(target);
      plan.steps[emitted++] = {uint8_t(target),
                               uint8_t(a),
                               uint8_t(b < 0 ? a : b),
                               node.left_sign,
                               int8_t(b < 0 ? 0 : node.right_sign),
                               node.halve};
      if (plan.slots < target + 1)
        plan.slots = target + 1;
    }
  if (location[root] < 0)
    return {};
  plan.result = unsigned(location[root]);
  plan.valid = true;
  return plan;
}
#if defined(__HIPCC__)
#define RNS8_SELECTED_HD __host__ __device__
#else
#define RNS8_SELECTED_HD
#endif
RNS8_SELECTED_HD inline int selected_linear_center(int value, int q) {
  value %= q;
  if (value < 0)
    value += q;
  return value >= (q + 1) / 2 ? value - q : value;
}
RNS8_SELECTED_HD inline void evaluate_selected_linear(const SelectedLinearPlan& plan, int8_t* slots,
                                                      int q) {
  for (unsigned index = 0; index < plan.count; ++index) {
    const auto step = plan.steps[index];
    int value = int(slots[step.left]) * step.left_sign + int(slots[step.right]) * step.right_sign;
    if (step.halve)
      value *= ((q + 1) / 2);
    slots[step.target] = int8_t(selected_linear_center(value, q));
  }
}
#undef RNS8_SELECTED_HD
} // namespace rns8::detail
