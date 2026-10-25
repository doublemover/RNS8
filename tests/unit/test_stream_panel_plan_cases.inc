#include "backend_common/stream_panel_plan.hpp"
#include "backend_common/stream_panel_cost.hpp"
TEST_CASE("stream panel shortlist uses runtime admission and identity-bound complete costs") {
  rns8_context ctx{};ctx.backend=RNS8_BACKEND_HIP_DIRECT;std::strcpy(ctx.device_info.gcn_arch,"gfx1100");
  rns8_stream_gemm_desc base{};base.struct_size=sizeof(base);base.abi_version=RNS8_ABI_VERSION;
  base.semantics=RNS8_BOUNDED_I64;base.m=7;base.n=9;base.k=19;base.lda=23;base.ldb=11;base.ldc=13;
  base.prefix=9;base.max_abs_a=base.max_abs_b=3;base.device_budget_bytes=base.host_budget_bytes=65536;
  const rns8_stream_gemm_panel_choice choices[]{{3,4,5,RNS8_STREAM_GEMM_SERIAL,65536},
      {16,16,32,RNS8_STREAM_GEMM_OVERLAPPED,65536},{0,4,5,RNS8_STREAM_GEMM_SERIAL,0}};
  rns8_stream_gemm_cost_profile costs{};costs.struct_size=sizeof(costs);costs.abi_version=RNS8_ABI_VERSION;
  costs.max_modeled_panels=4096;costs.reuse_count=8;costs.semantics=base.semantics;costs.prefix=9;costs.schedule_mask=3;
  auto& rates=costs.stages;rates.struct_size=sizeof(rates);rates.abi_version=RNS8_ABI_VERSION;rates.backend=RNS8_BACKEND_HIP_DIRECT;
  std::strcpy(rates.target_id,"gfx1100");std::strcpy(rates.calibration_identity,"synthetic-model-fixture-not-evidence");
  REQUIRE(rns8_get_pipeline_build_identity(rates.build_identity_sha256,sizeof(rates.build_identity_sha256))==RNS8_SUCCESS);
  rates.known_mask=47;rates.lower_launch_ns=5;rates.upper_launch_ns=7;
  for(unsigned i=0;i<6;++i){rates.lower_ns_per_unit[i]=0.1;rates.upper_ns_per_unit[i]=0.2;}
  costs.lower_host_copy_ns_per_byte=0.1;costs.upper_host_copy_ns_per_byte=0.2;
  costs.lower_input_scan_ns_per_cell=1;costs.upper_input_scan_ns_per_cell=2;
  costs.lower_boundary_ns=2;costs.upper_boundary_ns=3;costs.lower_duplex_multiplier=1.2;costs.upper_duplex_multiplier=1.5;
  std::array<rns8_stream_gemm_candidate,3> out{};uint32_t required=0;
  CHECK(rns8_plan_stream_gemm(&ctx,&base,choices,3,&costs,nullptr,0,&required)==RNS8_WORKSPACE_TOO_SMALL);CHECK(required==3);
  REQUIRE(rns8_plan_stream_gemm(&ctx,&base,choices,3,&costs,out.data(),3,&required)==RNS8_SUCCESS);
  CHECK(out[0].cost_known==1);CHECK(out[1].cost_known==1);CHECK(out[0].upper_amortized_ns<=out[1].upper_amortized_ns);
  CHECK(out[2].status==RNS8_INVALID_ARGUMENT);CHECK(out[2].choice_index==2);
  for(unsigned i=0;i<2;++i) {
    CHECK(out[i].lower_amortized_ns>0);CHECK(out[i].upper_amortized_ns>=out[i].lower_amortized_ns);
    rns8_stream_gemm_resources r{};r.struct_size=sizeof(r);r.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_query_stream_gemm(&out[i].executable_problem,&r)==RNS8_SUCCESS);
    CHECK(r.device_bytes==out[i].resources.device_bytes);CHECK(r.host_bytes==out[i].resources.host_bytes);
  }
  rns8_stream_gemm_candidate cold{},warm{};costs.reuse_count=1;
  REQUIRE(rns8_plan_stream_gemm(&ctx,&base,choices,1,&costs,&cold,1,&required)==RNS8_SUCCESS);
  costs.reuse_count=8;
  REQUIRE(rns8_plan_stream_gemm(&ctx,&base,choices,1,&costs,&warm,1,&required)==RNS8_SUCCESS);
  CHECK(warm.upper_amortized_ns<cold.upper_amortized_ns);
  for(unsigned mutation=0;mutation<5;++mutation) {
    auto stale=costs;
    switch(mutation){case 0:stale.stages.build_identity_sha256[0]='!';break;case 1:stale.stages.target_id[0]='!';break;
      case 2:stale.stages.known_mask&=~2u;break;case 3:stale.max_modeled_panels=1;break;case 4:stale.semantics=RNS8_BOUNDED_U64;break;}
    REQUIRE(rns8_plan_stream_gemm(&ctx,&base,choices,1,&stale,&warm,1,&required)==RNS8_SUCCESS);
    CHECK(warm.status==RNS8_SUCCESS);CHECK(warm.cost_known==0);CHECK(warm.upper_amortized_ns==0);
  }
  costs.lower_boundary_ns=-1;
  CHECK(rns8_plan_stream_gemm(&ctx,&base,choices,1,&costs,&warm,1,&required)==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_plan_stream_gemm(&ctx,&base,choices,1,nullptr,&warm,1,&required)==RNS8_SUCCESS);
  CHECK(warm.cost_known==0);CHECK(warm.status==RNS8_SUCCESS);
}
TEST_CASE("overlapped panel storage reserves both rings and hard-budgeted pinned controls") {
  using namespace rns8::detail;
  StreamPanelProblem p;p.m=17;p.n=19;p.k=33;p.lda=35;p.ldb=21;p.ldc=23;
  p.panel_m=3;p.panel_n=4;p.panel_k=5;p.prefix=9;p.device_budget=p.host_budget=UINT64_MAX;
  const auto serial=plan_stream_panels(p);REQUIRE(serial.status==RNS8_SUCCESS);
  p.schedule=RNS8_STREAM_GEMM_OVERLAPPED;
  const auto overlap=plan_stream_panels(p);REQUIRE(overlap.status==RNS8_SUCCESS);
  CHECK(overlap.device_bytes>serial.device_bytes);CHECK(overlap.pinned_bytes>0);
  CHECK(overlap.host_bytes==overlap.pinned_bytes+overlap.control_bytes);CHECK(overlap.ordinary_bytes==overlap.control_bytes);
  CHECK(overlap.streams==3);CHECK(overlap.events==8);CHECK(overlap.minimum_device_bytes==3584);CHECK(overlap.minimum_host_bytes==2304);
  CHECK(overlap.total_steps==serial.total_steps);CHECK(overlap.tile_count==serial.tile_count);
  uint64_t end=0;
  for(size_t i=0;i<stream_slot_count;++i) {
    CHECK(overlap.offsets[i]>=end);CHECK(overlap.offsets[i]%256==0);end=overlap.offsets[i]+overlap.bytes[i];
  }
  CHECK(end<=overlap.device_bytes);
  end=0;
  for(auto slot:{stream_native_a,stream_native_b,stream_export,stream_status,stream_native_a1,stream_native_b1,stream_export1,stream_status1}) {
    CHECK(overlap.host_offsets[slot]>=end);CHECK(overlap.host_offsets[slot]%256==0);
    end=overlap.host_offsets[slot]+overlap.bytes[slot];
  }
  CHECK(end<=overlap.pinned_bytes);
  p.device_budget=overlap.device_bytes-1;CHECK(plan_stream_panels(p).status==RNS8_WORKSPACE_TOO_SMALL);
  p.device_budget=overlap.device_bytes;p.host_budget=overlap.host_bytes-1;CHECK(plan_stream_panels(p).status==RNS8_WORKSPACE_TOO_SMALL);
  p.host_budget=overlap.host_bytes;CHECK(plan_stream_panels(p).status==RNS8_SUCCESS);
  p.schedule=static_cast<rns8_stream_gemm_schedule>(2);CHECK(plan_stream_panels(p).status==RNS8_INVALID_ARGUMENT);
}
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

TEST_CASE("streaming B cache reload model follows bounded physical slots") {
  using namespace rns8::detail;
  for(uint32_t n:{1,5,9}) for(uint32_t k:{1,7,19}) for(uint64_t slots=0;slots<=16;++slots) {
    StreamPanelProblem p;
    p.m=7;p.n=n;p.k=k;p.lda=k;p.ldb=n;p.ldc=n;
    p.panel_m=3;p.panel_n=4;p.panel_k=5;p.prefix=9;
    p.device_budget=p.host_budget=UINT64_MAX;
    const auto baseline=plan_stream_panels(p);REQUIRE(baseline.status==RNS8_SUCCESS);
    p.retained_b_budget=slots*baseline.b_cache_stride;
    const auto r=plan_stream_panels(p);REQUIRE(r.status==RNS8_SUCCESS);
    const auto available=std::min(slots,r.b_panels);
    CHECK(r.b_cache_entries==(available>r.b_panels/2?available:0));
    CHECK(r.device_bytes==baseline.device_bytes+r.b_cache_bytes);
    CHECK(r.host_bytes==baseline.host_bytes+r.b_cache_entries*sizeof(StreamPanelCacheEntry));
    std::vector<StreamPanelCacheEntry> tags(r.b_cache_entries);
    const int source=0;
    unsigned pass=0;
    for(uint64_t generation:{1,1,2}) {
      uint64_t loads=0,hits=0;
      StreamPanelCoverage coverage(p,r,1,11,1,12,generation);StreamPanelStep step;
      while(coverage.next(step)) {
        const auto key=step.index%r.b_panels;
        auto* tag=tags.empty()?nullptr:&tags[key%tags.size()];
        if(tag && tag->matches(&source,12,generation,key)) ++hits;
        else {++loads;if(tag) *tag={&source,12,generation,key};}
        REQUIRE(coverage.acknowledge(step.index,1,11,1,12,generation));
      }
      CHECK(loads+hits==r.total_steps);
      CHECK(loads==(pass==1?r.b_warm_loads:r.b_cold_loads));
      ++pass;
    }
    p.device_budget=baseline.device_bytes;p.host_budget=baseline.host_bytes;
    const auto no_space=plan_stream_panels(p);REQUIRE(no_space.status==RNS8_SUCCESS);
    CHECK(no_space.b_cache_entries==0);CHECK(no_space.b_cold_loads==no_space.total_steps);
    if(r.b_cache_entries) {
      const auto useful=r.b_panels/2+1;
      p.device_budget=baseline.device_bytes+useful*r.b_cache_stride;
      p.host_budget=baseline.host_bytes+useful*sizeof(StreamPanelCacheEntry)-1;
      CHECK(plan_stream_panels(p).b_cache_entries==0);
      ++p.host_budget;CHECK(plan_stream_panels(p).b_cache_entries==useful);
    }
  }
  int first=0,second=0;
  const StreamPanelCacheEntry tag{&first,12,4,7};
  CHECK(tag.matches(&first,12,4,7));
  CHECK_FALSE(tag.matches(&second,12,4,7));
  CHECK_FALSE(tag.matches(&first,13,4,7));
  CHECK_FALSE(tag.matches(&first,12,5,7));
  CHECK_FALSE(tag.matches(&first,12,4,8));
}

TEST_CASE("public streaming resource query proves native output envelope") {
  rns8_stream_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
  d.semantics=RNS8_BOUNDED_I64;d.m=17;d.n=19;d.k=33;d.lda=35;d.ldb=21;d.ldc=23;
  d.panel_m=3;d.panel_n=4;d.panel_k=5;d.prefix=9;d.max_abs_a=5;d.max_abs_b=3;
  rns8_stream_gemm_resources r{};r.struct_size=sizeof(r);r.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_query_stream_gemm(&d,&r)==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(r.device_bytes>0);CHECK(r.host_bytes>0);CHECK(r.minimum_host_bytes==24);
  d.device_budget_bytes=r.device_bytes;d.host_budget_bytes=r.host_bytes;
  REQUIRE(rns8_query_stream_gemm(&d,&r)==RNS8_SUCCESS);
  d.max_abs_a=INT64_MAX;d.max_abs_b=2;
  CHECK(rns8_query_stream_gemm(&d,&r)==RNS8_RANGE_ERROR);
  d.semantics=RNS8_WRAP_U64_MOD_2_64;
  CHECK(rns8_query_stream_gemm(&d,&r)==RNS8_UNSUPPORTED_BACKEND);
  CHECK(rns8_destroy_stream_gemm(nullptr)==RNS8_SUCCESS);
}
