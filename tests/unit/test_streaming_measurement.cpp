#include "../../benchmarks/streaming_measurement.hpp"
#include <catch2/catch_test_macros.hpp>
#include <set>
using namespace rns8::bench;
TEST_CASE("streaming warmups visit both slots and observations retain complete batches") {
  for(uint32_t warmups:{0u,1u,3u,9u})for(uint32_t calls:{1u,2u,9u,32u}) {
    StreamingMeasurementPlan plan;
    REQUIRE(streaming_measurement_plan(warmups,calls,9,plan));
    CHECK(plan.warmup_calls>=2);CHECK(plan.measured_calls==calls);CHECK(plan.batches==9);
    uint64_t version=1;
    for(uint32_t warm=0;warm<warmups;++warm){std::set<unsigned> slots;for(uint32_t i=0;i<plan.warmup_calls;++i){slots.insert(i&1);++version;}CHECK(slots.size()==2);}
    std::set<uint64_t> observed_versions;
    for(uint32_t batch=0;batch<plan.batches;++batch)for(uint32_t i=0;i<plan.measured_calls;++i)REQUIRE(observed_versions.insert(version++).second);
    CHECK(version==plan.total_calls+1);CHECK(observed_versions.size()==uint64_t(calls)*9);
  }
}
TEST_CASE("streaming bounded plans reject unsupported counts transactionally") {
  StreamingMeasurementPlan plan{2,9,9,99};
  for(auto bad: {StreamingMeasurementPlan{1025,9,9,0},StreamingMeasurementPlan{1,0,9,0},
       StreamingMeasurementPlan{1,65537,9,0},StreamingMeasurementPlan{1,9,2,0},
       StreamingMeasurementPlan{1,9,65,0},StreamingMeasurementPlan{1,65536,64,0}}) {
    CHECK_FALSE(streaming_measurement_plan(bad.warmup_calls,bad.measured_calls,bad.batches,plan));
    CHECK(plan.total_calls==99);
  }
}
