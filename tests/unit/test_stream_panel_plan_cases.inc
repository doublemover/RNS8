#include "backend_common/stream_panel_plan.hpp"
TEST_CASE("out-of-core panel plan accounts bounded storage and exact K coverage") {
  using namespace rns8::detail;
  for(uint32_t pm:{1,3,16}) for(uint32_t pn:{1,4,16}) for(uint32_t pk:{1,5,16}) {
    StreamPanelProblem p;
    p.m=7;p.n=9;p.k=19;p.lda=23;p.ldb=11;p.ldc=13;
    p.panel_m=pm;p.panel_n=pn;p.panel_k=pk;p.prefix=9;
    p.device_budget=p.host_budget=UINT64_MAX;
    const auto r=plan_stream_panels(p);REQUIRE(r.status==RNS8_SUCCESS);
    uint64_t allocated=0;
    for(size_t i=0;i<stream_slot_count;++i) {
      CHECK(r.offsets[i]%256==0);
      CHECK(r.offsets[i]>=allocated);
      allocated=r.offsets[i]+r.bytes[i];
    }
    CHECK(allocated<=r.device_bytes);
    CHECK(r.minimum_device_bytes<=r.device_bytes);
    CHECK(r.minimum_host_bytes<=r.host_bytes);
    StreamPanelCoverage coverage(p,r,7,11,1,12,3);
    std::vector<unsigned> owners(7*9*19*9,0);
    StreamPanelStep step;
    while(coverage.next(step)) {
      CHECK_FALSE(coverage.acknowledge(step.index+1,7,11,1,12,3));
      CHECK_FALSE(coverage.acknowledge(step.index,6,11,1,12,3));
      CHECK_FALSE(coverage.acknowledge(step.index,7,11,2,12,3));
      for(uint64_t row=step.row;row<step.row+step.rows;++row)
        for(uint64_t col=step.col;col<step.col+step.cols;++col)
          for(uint64_t k=step.k_begin;k<step.k_begin+step.depth;++k)
            for(unsigned plane=0;plane<9;++plane) ++owners[((plane*7+row)*9+col)*19+k];
      CHECK(step.first==(step.k_begin==0));
      CHECK(step.last==(step.k_begin+step.depth==19));
      REQUIRE(coverage.acknowledge(step.index,7,11,1,12,3));
      CHECK_FALSE(coverage.acknowledge(step.index,7,11,1,12,3));
    }
    REQUIRE(coverage.complete());for(auto count:owners) REQUIRE(count==1);
    p.device_budget=r.device_bytes-1;
    auto rejected=plan_stream_panels(p);
    CHECK(rejected.status==RNS8_WORKSPACE_TOO_SMALL);
    CHECK(rejected.device_bytes==r.device_bytes);
    CHECK(rejected.minimum_device_bytes==r.minimum_device_bytes);
    p.device_budget=r.device_bytes;p.host_budget=r.host_bytes-1;
    CHECK(plan_stream_panels(p).status==RNS8_WORKSPACE_TOO_SMALL);
  }
}
TEST_CASE("out-of-core panels do not hide overflowing global spans") {
  using namespace rns8::detail;
  StreamPanelProblem p;
  p.m=INT64_MAX;p.n=1;p.k=1;p.lda=p.ldb=p.ldc=1;
  p.panel_m=p.panel_n=p.panel_k=p.prefix=1;p.device_budget=p.host_budget=UINT64_MAX;
  CHECK(plan_stream_panels(p).status==RNS8_RANGE_ERROR);
  p.m=1;p.semantics=RNS8_WRAP_U64_MOD_2_64;
  CHECK(plan_stream_panels(p).status==RNS8_UNSUPPORTED_BACKEND);
}
