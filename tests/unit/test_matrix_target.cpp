#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include "backend_common/matrix_target.hpp"
#include "backend_common/locality_schedule.hpp"
#include "backend_common/stream_panel_plan.hpp"
#include "core/api_internal.hpp"

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

TEST_CASE("operation provider rejects missing builtin target bodies before resource admission") {
  rns8_context ctx{};ctx.backend=RNS8_BACKEND_HIP_DIRECT;ctx.device_info.hip_available=1;
  rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
  q.operation=RNS8_OP_GEMM;q.execution_modes=RNS8_EXEC_SYNC;q.device_count=1;
  auto& d=q.problem;d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
  d.requested_backend=RNS8_BACKEND_AMDGPU_BUILTINS;d.semantics=RNS8_FINITE_RING_U8;
  d.bound_kind=RNS8_BOUND_NONE;d.m=d.n=d.k=1;d.max_prefix=1;d.finite_modulus=251;
  for(const char* target:{"gfx1103","gfx1150","gfx1100junk","gfx942:xnack","gfx1201:"}) {
    std::strcpy(ctx.device_info.gcn_arch,target);
    const auto result=rns8::detail::api::operation_support(ctx,q,RNS8_BACKEND_AMDGPU_BUILTINS);
    CHECK(result.status==RNS8_UNSUPPORTED_BACKEND);CHECK(result.reason_code==RNS8_SUPPORT_TARGET);
  }
}
