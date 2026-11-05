#pragma once

#include "backend_common/completed_observation.hpp"
#include "core/hip_runtime_handles.hpp"
#include <chrono>
#include <cmath>
#include <string>
#include <utility>

#if (defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP) || defined(__HIPCC__)
namespace rns8::detail {
bool hip_direct_timing_enabled();
void hip_direct_timing_record_sample(const char*, double);
void hip_direct_timing_record_failure(const char*, const char*, int) noexcept;

inline hipError_t record_synchronized_host_timing(const char* label, const char* alias,
                                                  std::chrono::steady_clock::time_point start) {
  const auto status = publish_after_completion(
      [] { return hipDeviceSynchronize(); },
      [&] {
        const double microseconds =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
                .count();
        hip_direct_timing_record_failure(label, "synchronized_host_fallback", -1);
        hip_direct_timing_record_sample(
            (std::string("host_synchronized_fallback:") + label).c_str(), microseconds);
        if (alias && alias[0]) {
          hip_direct_timing_record_failure(alias, "synchronized_host_fallback", -1);
          hip_direct_timing_record_sample(
              (std::string("host_synchronized_fallback:") + alias).c_str(), microseconds);
        }
      });
  if (status != hipSuccess) {
    hip_direct_timing_record_failure(label, "synchronized_host_fallback_drain", int(status));
    if (alias && alias[0])
      hip_direct_timing_record_failure(alias, "synchronized_host_fallback_drain", int(status));
  }
  return status;
}

// CK's integer status and HIP's runtime status share the event/fallback policy,
// while the caller supplies their distinct success and runtime-error mapping.
// Nondefault streams bypass instrumentation to preserve the enqueue contract.
template <class Fn, class Status, class RuntimeFailure>
Status timed_hip_backend_operation(const char* label, const char* alias, Fn&& fn, Status success,
                                   RuntimeFailure&& runtime_failure, hipStream_t stream = nullptr,
                                   bool synchronize_before_stop = false) {
  if (stream || !hip_direct_timing_enabled())
    return fn();
  const auto host_start = std::chrono::steady_clock::now();
  const auto failed_completion = [&](hipError_t error, const char* phase) {
    hip_direct_timing_record_failure(label, phase, int(error));
    if (alias && alias[0])
      hip_direct_timing_record_failure(alias, phase, int(error));
    return runtime_failure(error);
  };
  const auto fallback = [&]() {
    const auto drained = record_synchronized_host_timing(label, alias, host_start);
    return drained == hipSuccess ? success : runtime_failure(drained);
  };
  hip_unique_event_pair events;
  if (!hip_create_timing_events_with_retry(events)) {
    const auto status = fn();
    return status == success ? fallback() : status;
  }
  const auto status = fn();
  if (status != success)
    return status;
  // hipBLASLt's small Windows captures retain their explicitly synchronized
  // diagnostic boundary; ordinary runtime and enqueue calls never take it.
  if (synchronize_before_stop) {
    const auto drained = hipDeviceSynchronize();
    if (drained != hipSuccess)
      return failed_completion(drained, "library_before_stop_drain");
  }
  auto stop = hipEventRecord(events.stop(), nullptr);
  if (stop != hipSuccess) {
    hip_direct_timing_flush_pending_events();
    const auto drained = hipDeviceSynchronize();
    if (drained != hipSuccess)
      return failed_completion(drained, "stop_record_drain");
    stop = hipEventRecord(events.stop(), nullptr);
  }
  if (stop == hipSuccess) {
    auto observed = hipEventSynchronize(events.stop());
    if (observed != hipSuccess) {
      const auto drained = hipDeviceSynchronize();
      if (drained != hipSuccess)
        return failed_completion(drained, "stop_observation_drain");
      observed = hipEventSynchronize(events.stop());
    }
    if (observed == hipSuccess) {
      float milliseconds = 0;
      auto elapsed = hipEventElapsedTime(&milliseconds, events.start(), events.stop());
      if (elapsed != hipSuccess) {
        const auto drained = hipDeviceSynchronize();
        if (drained != hipSuccess)
          return failed_completion(drained, "elapsed_retry_drain");
        elapsed = hipEventElapsedTime(&milliseconds, events.start(), events.stop());
      }
      if (elapsed == hipSuccess && std::isfinite(milliseconds) && milliseconds >= 0) {
        const double microseconds = double(milliseconds) * 1000;
        hip_direct_timing_record_sample(label, microseconds);
        if (alias && alias[0])
          hip_direct_timing_record_sample(alias, microseconds);
        return status;
      }
    }
  }
  return fallback();
}

template <class Fn>
hipError_t timed_hip_backend_launch(const char* label, const char* alias, Fn&& fn,
                                    hipStream_t stream = nullptr) {
  return timed_hip_backend_operation(
      label, alias, std::forward<Fn>(fn), hipSuccess, [](hipError_t error) { return error; },
      stream);
}
} // namespace rns8::detail
#endif
