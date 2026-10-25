#include "../../benchmarks/public_sustained.hpp"
#include <catch2/catch_test_macros.hpp>
#include <limits>
namespace {
struct FixtureClock {
  using duration=std::chrono::nanoseconds;
  using time_point=std::chrono::time_point<FixtureClock>;
  static inline uint64_t ticks=0;
  static time_point now(){return time_point(duration(ticks));}
};
}
TEST_CASE("sustained public intervals count completed boundaries and preserve source generations") {
  using namespace rns8::bench;
  uint64_t version=20;
  std::vector<uint64_t> generations;
  auto call=[&](uint64_t value){generations.push_back(value);FixtureClock::ticks+=3000;};
  FixtureClock::ticks=0;
  auto first=measure_sustained<decltype(call)&,FixtureClock>(call,4,10,15,3,version);
  REQUIRE(first.calls==5);REQUIRE(first.tasks==15);REQUIRE(first.duration_us==15);
  REQUIRE(first.duration_reached);REQUIRE(version==25);
  REQUIRE(generations==std::vector<uint64_t>{20,21,22,23,24});
  auto capped=measure_sustained<decltype(call)&,FixtureClock>(call,4,6,100,3,version);
  REQUIRE(capped.calls==6);REQUIRE_FALSE(capped.duration_reached);REQUIRE(version==31);
  auto zero_duration=measure_sustained<decltype(call)&,FixtureClock>(call,2,6,0,1,version);
  REQUIRE(zero_duration.calls==2);REQUIRE(zero_duration.duration_reached);
  REQUIRE_THROWS((measure_sustained<decltype(call)&,FixtureClock>(call,7,6,1,1,version)));
  version=UINT64_MAX-1;
  REQUIRE_THROWS((measure_sustained<decltype(call)&,FixtureClock>(call,1,6,1,1,version)));
}
TEST_CASE("failed sustained calls propagate without a successful observation") {
  uint64_t version=0;
  auto failure=[](uint64_t){throw std::runtime_error("production completion failed");};
  REQUIRE_THROWS(rns8::bench::measure_sustained(failure,1,8,0,1,version));
}
