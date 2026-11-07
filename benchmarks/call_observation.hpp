#pragma once
#include "public_sustained.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <nlohmann/json.hpp>

namespace rns8::bench {
// Sustained callbacks keep their operation/completion boundary but skip phase
// clocks. The outer interval alone measures them; samples enable phase clocks.
template <class Clock = std::chrono::steady_clock> class CallClock {
public:
  explicit CallClock(bool enabled) : enabled_(enabled) {}
  typename Clock::time_point now() const {
    return enabled_ ? Clock::now() : typename Clock::time_point{};
  }

private:
  bool enabled_;
};
template <std::size_t N, class Rep = double> struct PhaseObservation {
  std::array<Rep, N> phases{};
  Rep total{};
};
// Call only after the completed-call/outer sustained timer has stopped. The
// observation itself has fixed storage and no report-container ownership.
template <std::size_t N, class Rep>
nlohmann::json phase_observation_json(const PhaseObservation<N, Rep>& value,
                                      const std::array<const char*, N>& keys,
                                      const char* total_key = "total_us") {
  nlohmann::json result = nlohmann::json::object();
  for (std::size_t i = 0; i < N; ++i)
    result[keys[i]] = value.phases[i];
  result[total_key] = value.total;
  return result;
}
} // namespace rns8::bench
