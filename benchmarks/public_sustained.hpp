#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rns8::bench {
struct SustainedObservation {
  uint32_t calls = 0;
  uint64_t tasks = 0;
  double duration_us = 0;
  bool duration_reached = false;
};

// Every callback completes the same production output boundary as the latency
// lane. At least one call always completes; the hard cap is checked between
// calls, so a single slow call is never interrupted or treated as successful.
template<class Call, class Clock = std::chrono::steady_clock>
SustainedObservation measure_sustained(Call&& call, uint32_t min_calls, uint32_t max_calls,
                                      double min_duration_us, uint64_t tasks_per_call,
                                      uint64_t& source_version) {
  if (!min_calls || min_calls > max_calls || max_calls > 65536 || !std::isfinite(min_duration_us) || min_duration_us < 0 ||
      !tasks_per_call || tasks_per_call > UINT64_MAX / max_calls || source_version > UINT64_MAX-max_calls)
    throw std::invalid_argument("invalid sustained production interval bounds");
  SustainedObservation result;
  const auto start = Clock::now();
  do {
    call(source_version++);
    ++result.calls;
    result.duration_us = std::chrono::duration<double,std::micro>(Clock::now()-start).count();
    result.duration_reached = result.duration_us >= min_duration_us && result.calls >= min_calls;
  } while (!result.duration_reached && result.calls < max_calls);
  result.tasks = static_cast<uint64_t>(result.calls) * tasks_per_call;
  return result;
}
} // namespace rns8::bench
