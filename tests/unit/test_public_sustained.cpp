#include "../../benchmarks/public_sustained.hpp"
#include "../../benchmarks/call_observation.hpp"
#include "../../benchmarks/graph_caller_transaction.hpp"
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <type_traits>
namespace {
struct FixtureClock {
  using duration = std::chrono::nanoseconds;
  using time_point = std::chrono::time_point<FixtureClock>;
  static inline uint64_t ticks = 0;
  static time_point now() { return time_point(duration(ticks)); }
};
} // namespace
TEST_CASE("sustained public intervals count completed boundaries and preserve source generations") {
  using namespace rns8::bench;
  uint64_t version = 20;
  std::vector<uint64_t> generations;
  auto call = [&](uint64_t value) {
    generations.push_back(value);
    FixtureClock::ticks += 3000;
  };
  FixtureClock::ticks = 0;
  auto first = measure_sustained<decltype(call)&, FixtureClock>(call, 4, 10, 15, 3, version);
  REQUIRE(first.calls == 5);
  REQUIRE(first.tasks == 15);
  REQUIRE(first.duration_us == 15);
  REQUIRE(first.duration_reached);
  REQUIRE(version == 25);
  REQUIRE(first.duration_ns == 15000);
  REQUIRE(generations == std::vector<uint64_t>{20, 21, 22, 23, 24});
  auto capped = measure_sustained<decltype(call)&, FixtureClock>(call, 4, 6, 100, 3, version);
  REQUIRE(capped.calls == 6);
  REQUIRE_FALSE(capped.duration_reached);
  REQUIRE(version == 31);
  auto zero_duration = measure_sustained<decltype(call)&, FixtureClock>(call, 2, 6, 0, 1, version);
  REQUIRE(zero_duration.calls == 2);
  REQUIRE(zero_duration.duration_reached);
  REQUIRE_THROWS((measure_sustained<decltype(call)&, FixtureClock>(call, 7, 6, 1, 1, version)));
  version = UINT64_MAX - 1;
  REQUIRE_THROWS((measure_sustained<decltype(call)&, FixtureClock>(call, 1, 6, 1, 1, version)));
}
TEST_CASE("completed-call observations skip disabled clocks and serialize only fixed phase data") {
  using namespace rns8::bench;
  struct CountingClock {
    using time_point = std::chrono::steady_clock::time_point;
    static time_point now() { throw std::runtime_error("disabled phase clock was sampled"); }
  };
  CallClock<CountingClock> disabled(false);
  CHECK(disabled.now() == CountingClock::time_point{});
  FixtureClock::ticks = 123;
  CallClock<FixtureClock> enabled(true);
  CHECK(enabled.now().time_since_epoch().count() == 123);
  static_assert(std::is_trivially_copyable_v<PhaseObservation<4>>);
  const PhaseObservation<2> sample{{1.25, 2.75}, 4.0};
  const auto result = phase_observation_json(sample, {"enqueue_us", "wait_us"});
  CHECK(result == nlohmann::json{{"enqueue_us", 1.25}, {"wait_us", 2.75}, {"total_us", 4.0}});
  const PhaseObservation<2, uint64_t> nanoseconds{{0, 17}, 17};
  const auto integer = phase_observation_json(nanoseconds, {"enqueue_ns", "wait_ns"}, "total_ns");
  CHECK(integer.at("total_ns").is_number_unsigned());
  CHECK(integer.at("total_ns") == 17);
}
TEST_CASE("completed callback adapter counts completed calls and propagates operation failures") {
  using namespace rns8::bench;
  unsigned completed = 0;
  FixtureClock::ticks = 0;
  auto call = [&]() {
    ++completed;
    FixtureClock::ticks += 1000;
    return PhaseObservation<2>{{0, 0}, 0};
  };
  const auto result = measure_completed_calls<decltype(call)&, FixtureClock>(call, 2, 8, 3);
  CHECK(completed == 3);
  CHECK(result.calls == 3);
  CHECK(result.duration_us == 3);
  CHECK(result.duration_ns == 3000);
  unsigned attempted = 0;
  auto fail = [&]() {
    if (++attempted == 2)
      throw std::runtime_error("operation failed before completion");
    FixtureClock::ticks += 1000;
  };
  REQUIRE_THROWS((measure_completed_calls<decltype(fail)&, FixtureClock>(fail, 3, 8, 3)));
  CHECK(attempted == 2);
}
TEST_CASE("failed sustained calls propagate without a successful observation") {
  uint64_t version = 0;
  auto failure = [](uint64_t) { throw std::runtime_error("production completion failed"); };
  REQUIRE_THROWS(rns8::bench::measure_sustained(failure, 1, 8, 0, 1, version));
}

namespace {
struct GraphFixtureClock {
  using duration = std::chrono::nanoseconds;
  using time_point = std::chrono::time_point<GraphFixtureClock>;
  static inline uint64_t ticks = 0, reads = 0;
  static time_point now() {
    ++reads;
    return time_point(duration(ticks));
  }
};
} // namespace

TEST_CASE("graph caller transactions include staging and every output delivery") {
  using rns8::bench::completed_graph_caller_transaction;
  for (const bool detailed : {false, true}) {
    for (const unsigned warmups : {0u, 2u}) {
      GraphFixtureClock::ticks = GraphFixtureClock::reads = 0;
      uint64_t source = 0, staged = 0, device_output = 0, caller_output = 0;
      unsigned stage_count = 0, launch_count = 0, delivery_count = 0;
      for (unsigned call = 0; call < warmups + 3; ++call) {
        source = 7 + call;
        const auto observation = completed_graph_caller_transaction<GraphFixtureClock>(
            detailed,
            [&] {
              staged = source;
              ++stage_count;
              GraphFixtureClock::ticks += 1000;
            },
            [&] {
              device_output = staged * 3;
              ++launch_count;
              GraphFixtureClock::ticks += 2000;
              return RNS8_SUCCESS;
            },
            [&] {
              caller_output = device_output;
              ++delivery_count;
              GraphFixtureClock::ticks += 3000;
              return RNS8_SUCCESS;
            });
        REQUIRE(observation.status == RNS8_SUCCESS);
        REQUIRE(caller_output == source * 3);
        REQUIRE(observation.total_us == 6);
        if (detailed) {
          REQUIRE(observation.phases_us[0] == 1);
          REQUIRE(observation.phases_us[1] == 2);
          REQUIRE(observation.phases_us[2] == 3);
        } else {
          REQUIRE((observation.phases_us == std::array<double, 3>{}));
        }
      }
      REQUIRE(stage_count == warmups + 3);
      REQUIRE(launch_count == stage_count);
      REQUIRE(delivery_count == stage_count);
      REQUIRE(GraphFixtureClock::reads == stage_count * (detailed ? 4 : 2));
    }
  }
}

TEST_CASE("graph caller transactions do not deliver after a failed replay or status") {
  using rns8::bench::completed_graph_caller_transaction;
  for (const auto failure : {RNS8_BACKEND_FAILURE, RNS8_RANGE_ERROR}) {
    bool delivered = false;
    const auto observation = completed_graph_caller_transaction<GraphFixtureClock>(
        false, [] {}, [&] { return failure; },
        [&] {
          delivered = true;
          return RNS8_SUCCESS;
        });
    REQUIRE(observation.status == failure);
    REQUIRE_FALSE(delivered);
  }
  bool delivered = false;
  const auto observation = completed_graph_caller_transaction<GraphFixtureClock>(
      true, [] {}, [] { return RNS8_SUCCESS; },
      [&] {
        delivered = true;
        return RNS8_RANGE_ERROR;
      });
  REQUIRE(delivered);
  REQUIRE(observation.status == RNS8_RANGE_ERROR);
}
