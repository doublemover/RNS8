#include <catch2/catch_test_macros.hpp>
#include "backend_common/system_roof_plan.hpp"
namespace s=rns8::detail::system_roof;
TEST_CASE("system roofs price actual LDS accesses and bound every live payload", "[calibration][cpu]") {
  s::Problem p;p.mode=s::Mode::lds_read_write;p.words=384;p.blocks=3;p.iterations=17;
  for(uint32_t stride:{1u,2u,4u,8u,16u,32u}) {
    p.stride=stride;auto q=s::query(p);REQUIRE(q.admitted);
    CHECK(q.lds_bytes_per_block==128*stride*4);
    CHECK(q.issued_lds_read_bytes==384*4*17);CHECK(q.issued_lds_write_bytes==384*4*18);
    auto reference=s::reference_block(p,2);REQUIRE(reference.size()==128);
    // First two transitions independently expanded with finite uint32 arithmetic.
    p.iterations=2;reference=s::reference_block(p,2);
    for(uint32_t lane=0;lane<128;++lane) {
      const uint32_t neighbor=(lane+1)%128;
      CHECK(reference[lane]==uint32_t(s::seed(256+lane)+s::seed(256+neighbor)+neighbor));
    }p.iterations=17;
    p.budget_bytes=q.total_bytes-1;CHECK_FALSE(s::query(p).admitted);p.budget_bytes=q.total_bytes;
    CHECK(s::query(p).admitted);p.budget_bytes=uint64_t{1}<<30;
  }
  p.stride=3;CHECK_FALSE(s::query(p).admitted);
  p.stride=1;p.words=383;CHECK_FALSE(s::query(p).admitted);
  p.words=384;p.iterations=0;CHECK_FALSE(s::query(p).admitted);
}
TEST_CASE("system event and graph probes cannot masquerade as global bandwidth", "[calibration][cpu]") {
  s::Problem p;p.mode=s::Mode::event;p.words=1;
  auto q=s::query(p);REQUIRE(q.admitted);CHECK(q.unique_source_bytes==0);CHECK(q.issued_global_write_bytes==0);
  p.words=2;CHECK_FALSE(s::query(p).admitted);
  p.mode=s::Mode::global_copy;p.words=(uint64_t{1}<<29)+1;CHECK_FALSE(s::query(p).admitted);
  p.words=1;p.displacement_words=100000;q=s::query(p);REQUIRE(q.admitted);
  CHECK(q.displacement_bytes==800000);CHECK(q.pinned_bytes==(65536+1)*4);
  s::Mode parsed;CHECK(s::parse("graph",parsed));CHECK(parsed==s::Mode::graph);CHECK_FALSE(s::parse("hbm-observed",parsed));
}
