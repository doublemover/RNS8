#include "backend_hip_direct/hip_backend.hpp"

#include "core/hip_resources.hpp"

#include <string>
#include <utility>
#include <vector>

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

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
void flush_pending_timing_events() {
  for (auto& pending : g_hip_direct_pending_timing_samples) {
    if (!pending.start || !pending.stop || pending.labels.empty()) {
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
      if (status == hipSuccess && milliseconds >= 0.0f) {
        const double microseconds = static_cast<double>(milliseconds) * 1000.0;
        for (const auto& label : pending.labels) {
          if (!label.empty()) {
            g_hip_direct_timing_samples.push_back({label, microseconds});
          }
        }
      }
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
}

void hip_direct_timing_flush_pending_events() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  flush_pending_timing_events();
#endif
}

void hip_direct_timing_record_sample(const char* label, double microseconds) {
  if (!g_hip_direct_timing_enabled || !label || microseconds < 0.0) {
    return;
  }
  g_hip_direct_timing_samples.push_back({label, microseconds});
}

void hip_direct_timing_record_pending_event(const char* label, void* start_event, void* stop_event) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_unique_event start(reinterpret_cast<hipEvent_t>(start_event));
  hip_unique_event stop(reinterpret_cast<hipEvent_t>(stop_event));
  if (!g_hip_direct_timing_enabled || !label || !start || !stop) {
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

}  // namespace rns8::detail
