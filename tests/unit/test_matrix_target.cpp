#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include "backend_common/matrix_target.hpp"
#include "backend_common/locality_schedule.hpp"
#include "backend_common/stream_panel_plan.hpp"

TEST_CASE("builtin target admission matches exact implemented body families") {
  using namespace rns8::detail;
  for (const auto* target : {"gfx1100", "gfx1101:xnack-", "gfx1102:sramecc+:xnack-"})
    CHECK(builtin_arch(target) == BuiltinArch::rdna3);
  CHECK(builtin_arch("gfx942:xnack+:sramecc-") == BuiltinArch::cdna3);
  CHECK(builtin_arch("gfx1200") == BuiltinArch::rdna4);
  CHECK(builtin_arch("gfx1201:xnack-") == BuiltinArch::rdna4);
  for (const auto* target : {"", "gfx1103", "gfx1150", "gfx9420", "gfx11000", "gfx1201junk",
      "gfx1100:", "gfx1100:xnack", "gfx1100:xnack+:xnack-", "gfx942:unknown+"}) {
    CHECK(builtin_arch(target) == BuiltinArch::unsupported);
    CHECK_FALSE(builtin_sparse_target(target));
  }
  CHECK_FALSE(builtin_sparse_target("gfx1100"));
  CHECK(builtin_sparse_target("gfx942:xnack-"));
  CHECK(builtin_sparse_target("gfx1201"));
}

TEST_CASE("fixed public enums admit raw bytes but reject invalid scheduling modes") {
  using namespace rns8::detail;
  for (uint32_t raw : {UINT32_MAX, UINT32_C(0x80000000), 2u}) {
    StreamPanelProblem p;
    p.m=p.n=p.k=p.lda=p.ldb=p.ldc=1;p.panel_m=p.panel_n=p.panel_k=p.prefix=1;
    p.device_budget=p.host_budget=65536;
    std::memcpy(&p.schedule,&raw,sizeof(raw));
    CHECK(plan_stream_panels(p).status==RNS8_INVALID_ARGUMENT);
  }
  rns8_locality_desc d{};
  d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.topology_version=1;
  d.cache_domains=d.concurrent_groups_per_domain=1;
  CHECK(locality_descriptor_valid(d));
  for(uint32_t raw:{UINT32_MAX,UINT32_C(0x80000000),4u}) {
    std::memcpy(&d.order,&raw,sizeof(raw));
    CHECK_FALSE(locality_descriptor_valid(d));
  }
}
