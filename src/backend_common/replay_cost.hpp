#pragma once
#include <cstdint>
#include <limits>

namespace rns8::detail {
// UINT64_MAX is unknown/overflow throughout the cost domain. No arithmetic
// estimate can change range admission or imply measured qualification.
inline uint64_t replay_cost_add(uint64_t a, uint64_t b) {
  return a == UINT64_MAX || b == UINT64_MAX || a >= UINT64_MAX - b ? UINT64_MAX : a + b;
}
inline uint64_t replay_cost_mul(uint64_t value, uint64_t count) {
  if (!count) return 0;
  return value == UINT64_MAX || value > (UINT64_MAX - 1) / count ? UINT64_MAX : value * count;
}
struct ReplayCostPolicy {
  uint64_t replays = 1, preparation_ns = 0, calibration_ns = 0;
  uint64_t orchestration_ns = 0, publication_ns = 0;
  uint64_t baseline_replay_ns = UINT64_MAX, baseline_preparation_ns = 0;
  bool retain_external_b = false, require_amortization = false;
};
struct ReplayCostEstimate {
  uint64_t first_ns = UINT64_MAX, steady_ns = UINT64_MAX, once_ns = UINT64_MAX;
  uint64_t lifetime_ns = UINT64_MAX, amortized_ns = UINT64_MAX, break_even_replays = 0;
  // 0 unknown baseline, 1 projected profitable, 2 projected nonprofitable.
  uint32_t amortization_status = 0;
};
inline ReplayCostEstimate estimate_replay_cost(const ReplayCostPolicy& policy,
                                               uint64_t first, uint64_t steady) {
  ReplayCostEstimate result;
  if (!policy.replays || policy.replays == UINT64_MAX) return result;
  const auto common = replay_cost_add(policy.orchestration_ns, policy.publication_ns);
  result.first_ns = replay_cost_add(first, common);
  result.steady_ns = replay_cost_add(steady, common);
  result.once_ns = replay_cost_add(policy.preparation_ns, policy.calibration_ns);
  result.lifetime_ns = replay_cost_add(result.once_ns,
      replay_cost_add(result.first_ns, replay_cost_mul(result.steady_ns, policy.replays - 1)));
  if (result.lifetime_ns == UINT64_MAX) return result;
  result.amortized_ns = result.lifetime_ns / policy.replays + (result.lifetime_ns % policy.replays != 0);
  if (policy.baseline_replay_ns == UINT64_MAX || policy.baseline_preparation_ns == UINT64_MAX) return result;
  const auto baseline = replay_cost_add(policy.baseline_preparation_ns,
      replay_cost_mul(policy.baseline_replay_ns, policy.replays));
  if (baseline == UINT64_MAX) return result;
  result.amortization_status = result.lifetime_ns < baseline ? 1 : 2;
  if (result.steady_ns >= policy.baseline_replay_ns) {
    if (replay_cost_add(result.once_ns, result.first_ns) <
        replay_cost_add(policy.baseline_preparation_ns, policy.baseline_replay_ns)) result.break_even_replays = 1;
    return result;
  }
  // first >= steady for retained conversion. Reject an impossible negative
  // cold premium rather than laundering it into an attractive estimate.
  if (result.first_ns < result.steady_ns) return result;
  const auto premium = replay_cost_add(result.once_ns, result.first_ns - result.steady_ns);
  if (premium == UINT64_MAX) return result;
  if (premium < policy.baseline_preparation_ns) result.break_even_replays = 1;
  else {
    const auto saving = policy.baseline_replay_ns - result.steady_ns;
    const auto calls = (premium - policy.baseline_preparation_ns) / saving;
    if (calls < UINT64_MAX - 1) result.break_even_replays = calls + 1; // strict improvement
  }
  return result;
}
} // namespace rns8::detail
