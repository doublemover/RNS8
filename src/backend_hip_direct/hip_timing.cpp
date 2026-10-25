#include "backend_hip_direct/hip_backend.hpp"

#include "core/hip_resources.hpp"

#include <string>
#include <utility>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#  include <hip/hip_runtime_api.h>
#endif

namespace rns8::detail {

namespace {

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
struct hip_direct_pending_timing_sample {
  std::vector<std::string> labels;
  hip_unique_event start;
  hip_unique_event stop;
};

constexpr std::size_t kMaxPendingTimingEventsBeforeFlush = 16;
thread_local std::vector<hip_direct_pending_timing_sample> g_hip_direct_pending_timing_samples;
#endif

thread_local bool g_hip_direct_timing_enabled = false;
thread_local std::vector<hip_direct_timing_sample> g_hip_direct_timing_samples;
struct StoredTimingFailure {char label[128]{},phase[48]{};int status=0;};
thread_local std::array<StoredTimingFailure,64> g_timing_failures;
thread_local size_t g_timing_failure_count=0;
thread_local bool g_timing_failures_complete=true;

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
void flush_pending_timing_events() {
  for (auto& pending : g_hip_direct_pending_timing_samples) {
    if (!pending.start || !pending.stop || pending.labels.empty()) {
      for(const auto& label:pending.labels)hip_direct_timing_record_failure(label.c_str(),"invalid_event_pair",static_cast<int>(hipErrorInvalidValue));
      continue;
    }
    hipError_t status = hipEventSynchronize(pending.stop.get());
    if (status != hipSuccess) {
      (void)hipDeviceSynchronize();
      status = hipEventSynchronize(pending.stop.get());
    }
    if (status == hipSuccess) {
      float milliseconds = 0.0f;
      status = hipEventElapsedTime(&milliseconds, pending.start.get(), pending.stop.get());
      if (status != hipSuccess) {
        (void)hipDeviceSynchronize();
        status = hipEventElapsedTime(&milliseconds, pending.start.get(), pending.stop.get());
      }
      if (status == hipSuccess && std::isfinite(milliseconds) && milliseconds >= 0.0f) {
        const double microseconds = static_cast<double>(milliseconds) * 1000.0;
        for (const auto& label : pending.labels) {
          if (!label.empty()) {
            g_hip_direct_timing_samples.push_back({label, microseconds});
          }
        }
      } else {
        for(const auto& label:pending.labels)hip_direct_timing_record_failure(label.c_str(),
            status==hipSuccess?"invalid_elapsed_time":"elapsed_time",static_cast<int>(status));
      }
    } else {
      for(const auto& label:pending.labels)hip_direct_timing_record_failure(label.c_str(),"stop_synchronize",static_cast<int>(status));
    }
  }
  g_hip_direct_pending_timing_samples.clear();
}

void flush_pending_timing_events_if_full() {
  if (g_hip_direct_pending_timing_samples.size() >= kMaxPendingTimingEventsBeforeFlush) {
    flush_pending_timing_events();
  }
}
#endif

}  // namespace

void hip_direct_timing_set_enabled(bool enabled) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!enabled) {
    flush_pending_timing_events();
  }
#endif
  g_hip_direct_timing_enabled = enabled;
  if (!enabled) {
    g_hip_direct_timing_samples.clear();
    g_timing_failure_count=0;g_timing_failures_complete=true;
  }
}

bool hip_direct_timing_enabled() {
  return g_hip_direct_timing_enabled;
}

void hip_direct_timing_reset() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  flush_pending_timing_events();
#endif
  g_hip_direct_timing_samples.clear();
  g_timing_failure_count=0;g_timing_failures_complete=true;
}

void hip_direct_timing_flush_pending_events() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  flush_pending_timing_events();
#endif
}

void hip_direct_timing_record_sample(const char* label, double microseconds) {
  if (!g_hip_direct_timing_enabled || !label) {
    return;
  }
  if(!std::isfinite(microseconds) || microseconds<0) {hip_direct_timing_record_failure(label,"invalid_host_sample",0);return;}
  g_hip_direct_timing_samples.push_back({label, microseconds});
}

void hip_direct_timing_record_pending_event(const char* label, void* start_event, void* stop_event) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_unique_event start(reinterpret_cast<hipEvent_t>(start_event));
  hip_unique_event stop(reinterpret_cast<hipEvent_t>(stop_event));
  if (!g_hip_direct_timing_enabled || !label || !start || !stop) {
    if(g_hip_direct_timing_enabled && label)hip_direct_timing_record_failure(label,"invalid_event_pair",static_cast<int>(hipErrorInvalidValue));
    return;
  }
  flush_pending_timing_events_if_full();
  hip_direct_pending_timing_sample sample;
  sample.labels.push_back(label);
  sample.start = std::move(start);
  sample.stop = std::move(stop);
  g_hip_direct_pending_timing_samples.push_back(std::move(sample));
#else
  (void)label;
  (void)start_event;
  (void)stop_event;
#endif
}

void hip_direct_timing_record_pending_event_with_alias(
    const char* label,
    const char* alias,
    void* start_event,
    void* stop_event) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_unique_event start(reinterpret_cast<hipEvent_t>(start_event));
  hip_unique_event stop(reinterpret_cast<hipEvent_t>(stop_event));
  if (!g_hip_direct_timing_enabled || !label || !start || !stop) {
    if(g_hip_direct_timing_enabled && label)hip_direct_timing_record_failure(label,"invalid_event_pair",static_cast<int>(hipErrorInvalidValue));
    return;
  }
  std::vector<std::string> labels{label};
  if (alias && alias[0] != '\0') {
    labels.push_back(alias);
  }
  flush_pending_timing_events_if_full();
  hip_direct_pending_timing_sample sample;
  sample.labels = std::move(labels);
  sample.start = std::move(start);
  sample.stop = std::move(stop);
  g_hip_direct_pending_timing_samples.push_back(std::move(sample));
#else
  (void)label;
  (void)alias;
  (void)start_event;
  (void)stop_event;
#endif
}

std::vector<hip_direct_timing_sample> hip_direct_timing_snapshot() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  flush_pending_timing_events();
#endif
  return g_hip_direct_timing_samples;
}
void hip_direct_timing_record_failure(const char* label,const char* phase,int status) noexcept {
  if(!g_hip_direct_timing_enabled || !label || !phase)return;
  if(g_timing_failure_count==g_timing_failures.size()){g_timing_failures_complete=false;return;}
  auto& failure=g_timing_failures[g_timing_failure_count++];
  const auto copy=[](auto& destination,const char* source){const auto count=std::strlen(source);
    const auto bytes=std::min(count,sizeof(destination)-1);std::memcpy(destination,source,bytes);destination[bytes]=0;return count<sizeof(destination);};
  const bool label_complete=copy(failure.label,label),phase_complete=copy(failure.phase,phase);
  if(!label_complete || !phase_complete)g_timing_failures_complete=false;
  failure.status=status;
}
bool hip_direct_timing_failures_complete() noexcept {return g_timing_failures_complete;}
std::vector<hip_direct_timing_failure> hip_direct_timing_failure_snapshot() {
  hip_direct_timing_flush_pending_events();std::vector<hip_direct_timing_failure> result;result.reserve(g_timing_failure_count);
  for(size_t i=0;i<g_timing_failure_count;++i){const auto& f=g_timing_failures[i];result.push_back({f.label,f.phase,f.status});}
  return result;
}

}  // namespace rns8::detail
