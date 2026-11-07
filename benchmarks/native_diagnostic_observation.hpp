#pragma once
#include <rns8/rns8.h>
#include "backend_hip_direct/hip_backend.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace rns8_benchmark {
inline void observation_require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// A separate batch only: callers must keep this outside production intervals
// and independently verify its full output after event collection completes.
template <class Run> nlohmann::json observe_native_batch(rns8_context* ctx, Run&& run) {
  using namespace rns8::detail;
  struct Timing {
    Timing() {
      observation_require(!hip_direct_timing_enabled(), "native timing was already enabled");
      hip_direct_timing_reset();
      hip_direct_timing_set_enabled(true);
    }
    ~Timing() {
      hip_direct_timing_set_enabled(false);
      hip_direct_timing_reset();
    }
  } timing;
  const auto allocation = [&] {
    rns8_allocation_observation value{};
    value.struct_size = sizeof(value);
    value.abi_version = RNS8_ABI_VERSION;
    observation_require(rns8_get_allocation_observation(ctx, &value) == RNS8_SUCCESS,
                        "native allocation observation unavailable");
    return value;
  };
  const auto before = allocation();
  const auto copies_before = hip_direct_allocation_counters_snapshot();
  run();
  hip_direct_timing_flush_pending_events();
  const auto copies_after = hip_direct_allocation_counters_snapshot();
  const auto after = allocation();
  observation_require(before.epoch == after.epoch, "allocation observation epoch changed");
  const auto failures = hip_direct_timing_failure_snapshot();
  if (!hip_direct_timing_failures_complete() || !failures.empty()) {
    std::string message = "native diagnostic events are incomplete";
    for (const auto& failure : failures)
      message +=
          "; " + failure.label + ":" + failure.phase + "=" + std::to_string(failure.backend_status);
    throw std::runtime_error(message);
  }
  const auto delta = [](uint64_t last, uint64_t first) {
    observation_require(last >= first, "native diagnostic counters went backwards");
    return last - first;
  };
  using json = nlohmann::json;
  json labels = json::object(), durations = json::object();
  const auto events = hip_direct_timing_snapshot();
  observation_require(!events.empty() && events.size() <= 65536,
                      "native diagnostic event count is outside bounds");
  for (const auto& event : events) {
    observation_require(!event.label.empty() && event.label.size() <= 256 &&
                            std::isfinite(event.microseconds) && event.microseconds >= 0,
                        "invalid native diagnostic event");
    labels[event.label] = labels.value(event.label, uint64_t{0}) + 1;
    const auto total = durations.value(event.label, 0.0) + event.microseconds;
    observation_require(std::isfinite(total), "native diagnostic duration overflow");
    durations[event.label] = total;
  }
  observation_require(labels.size() <= 64, "native diagnostic label count is outside bounds");
  return {{"allocation_epoch", after.epoch},
          {"device_allocation_calls",
           delta(after.device.allocation_calls, before.device.allocation_calls)},
          {"device_free_calls", delta(after.device.free_calls, before.device.free_calls)},
          {"h2d_calls", delta(copies_after.h2d_calls, copies_before.h2d_calls)},
          {"h2d_bytes", delta(copies_after.h2d_bytes, copies_before.h2d_bytes)},
          {"d2h_calls", delta(copies_after.d2h_calls, copies_before.d2h_calls)},
          {"d2h_bytes", delta(copies_after.d2h_bytes, copies_before.d2h_bytes)},
          {"copy_scope", "tracked_completed_synchronous_HIP_copy_wrappers"},
          {"native_event_receipts_complete", true},
          {"event_labels", labels},
          {"event_microseconds", durations}};
}
} // namespace rns8_benchmark
