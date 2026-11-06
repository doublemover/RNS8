#ifndef RNS8_HOST_CONTROL_OBSERVATIONS_HPP
#define RNS8_HOST_CONTROL_OBSERVATIONS_HPP
#include "rns8/status.h"
#include <cstdint>
#if defined(RNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS) && RNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS
#include <chrono>
#endif

namespace rns8::detail {
// Attempt counts at the named host call sites, not device activity or a profiler.
// One borrowed, thread-local observer; disabled production calls allocate nothing.
struct HostControlObservations {
  uint64_t device_gets = 0, device_sets = 0, stream_device_queries = 0, stream_capture_queries = 0;
  uint64_t event_creates = 0, event_destroys = 0, event_records = 0, event_queries = 0,
           event_waits = 0;
  uint64_t submission_proofs = 0, proof_builds = 0, proof_reuses = 0, graph_launches = 0;
  uint64_t range_proof_calls = 0, range_proof_host_ns = 0, proof_identity_host_ns = 0,
           proof_metadata_host_ns = 0, graph_preflight_host_ns = 0;
};
#if defined(__HIPCC__)
// HIP objects are shared by static/shared libraries; their internal reference
// must not become a Windows import-thunk reference into a separate DLL.
#define RNS8_HOST_CONTROL_API
#else
#define RNS8_HOST_CONTROL_API RNS8_API
#endif
RNS8_HOST_CONTROL_API bool host_control_diagnostics_compiled() noexcept;
RNS8_HOST_CONTROL_API HostControlObservations* current_host_control_observer() noexcept;
RNS8_HOST_CONTROL_API HostControlObservations*
exchange_host_control_observer(HostControlObservations*) noexcept;
#undef RNS8_HOST_CONTROL_API
inline void add_host_control(HostControlObservations& observer,
                             uint64_t HostControlObservations::* field, uint64_t amount) noexcept {
  auto& value = observer.*field;
  value += amount > UINT64_MAX - value ? UINT64_MAX - value : amount;
}
inline void record_host_control(uint64_t HostControlObservations::* field) noexcept {
#if defined(RNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS) && RNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS
  if (auto* observer = current_host_control_observer()) {
    add_host_control(*observer, field, 1);
  }
#else
  (void)field;
#endif
}
// Diagnostic calling-thread elapsed time, including exceptional exits. Parent
// preflight scopes include nested proof work; these observations are not additive.
class HostControlPhaseScope {
public:
  explicit HostControlPhaseScope(uint64_t HostControlObservations::* field) noexcept
#if defined(RNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS) && RNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS
      : observer_(current_host_control_observer()), field_(field) {
    if (observer_)
      start_ = std::chrono::steady_clock::now();
  }
  ~HostControlPhaseScope() {
    if (observer_) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start_)
                               .count();
      if (elapsed > 0)
        add_host_control(*observer_, field_, static_cast<uint64_t>(elapsed));
    }
  }
#else
  {
    (void)field;
  }
  ~HostControlPhaseScope() = default;
#endif
  HostControlPhaseScope(const HostControlPhaseScope&) = delete;
  HostControlPhaseScope& operator=(const HostControlPhaseScope&) = delete;

private:
#if defined(RNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS) && RNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS
  HostControlObservations* observer_;
  uint64_t HostControlObservations::* field_;
  std::chrono::steady_clock::time_point start_{};
#endif
};
class HostControlObservationScope {
public:
  explicit HostControlObservationScope(HostControlObservations& out) noexcept
      : previous_(exchange_host_control_observer(&out)) {}
  ~HostControlObservationScope() { exchange_host_control_observer(previous_); }
  HostControlObservationScope(const HostControlObservationScope&) = delete;
  HostControlObservationScope& operator=(const HostControlObservationScope&) = delete;

private:
  HostControlObservations* previous_;
};
} // namespace rns8::detail
#endif
