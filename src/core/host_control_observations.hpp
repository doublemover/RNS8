#ifndef RNS8_HOST_CONTROL_OBSERVATIONS_HPP
#define RNS8_HOST_CONTROL_OBSERVATIONS_HPP
#include "rns8/status.h"
#include <cstdint>

namespace rns8::detail {
// Attempt counts at the named host call sites, not device activity or a profiler.
// One borrowed, thread-local observer; disabled production calls allocate nothing.
struct HostControlObservations {
  uint64_t device_gets=0,device_sets=0,stream_device_queries=0,stream_capture_queries=0;
  uint64_t event_creates=0,event_destroys=0,event_records=0,event_queries=0,event_waits=0;
  uint64_t submission_proofs=0,proof_builds=0,proof_reuses=0,graph_launches=0;
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
RNS8_HOST_CONTROL_API HostControlObservations* exchange_host_control_observer(HostControlObservations*) noexcept;
#undef RNS8_HOST_CONTROL_API
inline void record_host_control(uint64_t HostControlObservations::* field) noexcept {
#if defined(RNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS) && RNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS
  if(auto* observer=current_host_control_observer()) {
    auto& value=observer->*field;
    if(value!=UINT64_MAX)++value;
  }
#else
  (void)field;
#endif
}
class HostControlObservationScope {
 public:
  explicit HostControlObservationScope(HostControlObservations& out) noexcept
      : previous_(exchange_host_control_observer(&out)) {}
  ~HostControlObservationScope(){exchange_host_control_observer(previous_);}
  HostControlObservationScope(const HostControlObservationScope&)=delete;
  HostControlObservationScope& operator=(const HostControlObservationScope&)=delete;
 private:
  HostControlObservations* previous_;
};
}
#endif
