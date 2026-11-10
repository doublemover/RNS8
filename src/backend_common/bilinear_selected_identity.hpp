#pragma once
#include "bilinear_selected_transforms.hpp"
#include "sha256.hpp"
#include <string>
namespace rns8::detail {
// Canonical little-endian u32 encoding is shared with the independent collector.
// Serialize meaningful fields only; padding and unused array capacity have no identity.
inline std::string selected_bilinear_schedule_sha256(bool rank48) {
  std::string bytes = "rns8_selected_bilinear_v1";
  bytes.push_back('\0');
  const auto append = [&](uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      bytes.push_back(char((value >> (8 * i)) & 255));
  };
  append(rank48 ? 48 : 23);
  for (bool lhs : {true, false})
    for (unsigned output = 0; output < (rank48 ? 48u : 23u); ++output) {
      const auto plan =
          select_linear_output(rank48 ? rank48_input_dag(lhs) : october55_input_dag(lhs), output);
      if (!plan.valid)
        return {};
      append(plan.input_count);
      append(plan.count);
      append(plan.slots);
      append(plan.result);
      for (unsigned input = 0; input < plan.input_count; ++input)
        append(plan.input_ids[input]);
      for (unsigned index = 0; index < plan.count; ++index) {
        const auto step = plan.steps[index];
        append(step.target);
        append(step.left);
        append(step.right);
        append(uint32_t(int32_t(step.left_sign)));
        append(uint32_t(int32_t(step.right_sign)));
        append(step.halve ? 1 : 0);
      }
    }
  return sha256(bytes);
}
struct SelectedBilinearWork {
  uint64_t binary[2]{}, halving[2]{}, input_loads[2]{};
};
inline SelectedBilinearWork selected_bilinear_work(bool rank48) {
  SelectedBilinearWork work;
  for (unsigned side = 0; side < 2; ++side)
    for (unsigned output = 0; output < (rank48 ? 48u : 23u); ++output) {
      const auto plan = select_linear_output(
          rank48 ? rank48_input_dag(side == 0) : october55_input_dag(side == 0), output);
      work.input_loads[side] += plan.input_count;
      for (unsigned index = 0; index < plan.count; ++index) {
        const auto step = plan.steps[index];
        work.binary[side] += step.right_sign != 0;
        work.halving[side] += step.halve;
      }
    }
  return work;
}
} // namespace rns8::detail
