#pragma once
#include <array>
#include <chrono>
#include "rns8/rns8.h"

namespace rns8::bench {
struct GraphCallerObservation {
  rns8_status status = RNS8_SUCCESS;
  std::array<double, 3> phases_us{};
  double total_us = 0;
};

// Production keeps only the outer clock pair. Diagnostic phase clocks describe
// staging/replay/delivery separately and never become production wall latency.
template<class Clock = std::chrono::steady_clock, class Stage, class Launch, class Deliver>
GraphCallerObservation completed_graph_caller_transaction(
    bool phase_clocks, Stage&& stage, Launch&& launch, Deliver&& deliver) {
  const auto start = Clock::now();
  stage();
  const auto staged = phase_clocks ? Clock::now() : start;
  GraphCallerObservation result;
  result.status = launch();
  const auto replayed = phase_clocks ? Clock::now() : start;
  if (result.status == RNS8_SUCCESS) result.status = deliver();
  const auto end = Clock::now();
  result.total_us = std::chrono::duration<double, std::micro>(end - start).count();
  if (phase_clocks) {
    result.phases_us = {
        std::chrono::duration<double, std::micro>(staged - start).count(),
        std::chrono::duration<double, std::micro>(replayed - staged).count(),
        std::chrono::duration<double, std::micro>(end - replayed).count()};
  }
  return result;
}

// Declare before the owned buffers: their destructors run before this receipt's
// terminal clock read. Common matrix allocation can be a separately timed child.
class GraphCallerOwnerAccounting {
 public:
  GraphCallerOwnerAccounting(double& preparation, double& teardown)
      : preparation_(preparation), teardown_(teardown), start_(Clock::now()) {}
  GraphCallerOwnerAccounting(const GraphCallerOwnerAccounting&) = delete;
  GraphCallerOwnerAccounting& operator=(const GraphCallerOwnerAccounting&) = delete;
  void finish_preparation(double separately_recorded_us = 0) {
    const double elapsed = std::chrono::duration<double, std::micro>(Clock::now() - start_).count();
    preparation_ = elapsed > separately_recorded_us ? elapsed - separately_recorded_us : 0;
  }
  void begin_teardown() { teardown_start_ = Clock::now(); teardown_started_ = true; }
  ~GraphCallerOwnerAccounting() {
    if (teardown_started_) teardown_ = std::chrono::duration<double, std::micro>(Clock::now() - teardown_start_).count();
  }
 private:
  using Clock = std::chrono::steady_clock;
  double& preparation_;
  double& teardown_;
  Clock::time_point start_, teardown_start_{};
  bool teardown_started_ = false;
};
} // namespace rns8::bench
