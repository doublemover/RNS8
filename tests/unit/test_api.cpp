#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "core/api_internal.hpp"
#include "backend_common/complex_residue_plan.hpp"
#include "core/plan_lowering.hpp"
#include "rns8/rns8.h"
#include "rns8/rns8.hpp"

namespace {
#include "test_api_support.inc"

}  // namespace

#include "test_api_validation_cases.inc"
#include "test_api_exact_wide_cases.inc"
#include "test_api_host_address_cases.inc"
#include "test_api_backend_info_cases.inc"
#include "test_api_storage_prepack_cases.inc"
#include "test_api_sparse_cases.inc"
#include "test_api_plan_packing_cases.inc"
#include "test_api_export_plan_cases.inc"
#include "test_api_result_cache_cases.inc"
#include "test_api_auto_selector_cases.inc"
#include "test_api_status_cases.inc"

#include "test_api_value_lifecycle_cases.inc"
#include "test_api_prepared_cases.inc"
#include "test_api_prepared_group_cases.inc"
#include "test_api_resident_graph_cases.inc"
#include "test_api_exact_epilogue_cases.inc"
#include "test_api_complex_residue_cases.inc"
#include "test_api_prepared_hip_cases.inc"
#include "test_api_plan_validation_cache_cases.inc"

#include "test_runtime_product_recipe_cases.inc"

#include "test_api_finite_verify_cases.inc"

#include "../support/ragged_owner_fixture.hpp"
TEST_CASE("public ragged owner preserves exact values across differently tiled consumers on CPU") {
  auto* ctx=create_cpu_context();
  for(bool u:{false,true}) {ragged_owner_fixture::roundtrip(ctx,u);ragged_owner_fixture::wide_roundtrip(ctx,u);}
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

#include "../support/gram_fixture.hpp"
#include "test_api_gram_cases.inc"

#include "test_stream_panel_plan_cases.inc"

#include "test_fixed_conversion_cases.inc"

#include "test_word_mod_cases.inc"

#include "test_native_device_contract_cases.inc"

#include "../support/prepared_multi_dependency_fixture.hpp"
TEST_CASE("three prepared producer epochs feed one exact epilogue on CPU") {
  check_prepared_multi_dependencies(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("three HIP producer publications feed an exact epilogue in reverse observation order") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  check_prepared_multi_dependencies(RNS8_BACKEND_HIP_DIRECT);
#else
  SKIP("HIP dependency execution unavailable in this build");
#endif
}

#include "../support/product_recipe_rebind_fixture.hpp"
TEST_CASE("public CPU product recipe rebinds generations within its compiled proof envelope") {
  check_product_recipe_rebind(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("public HIP product recipe rebinds owners without reallocating its arena") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  check_product_recipe_rebind(RNS8_BACKEND_HIP_DIRECT);
#else
  SKIP("HIP recipe execution unavailable in this build");
#endif
}

#include "test_api_split_k_cases.inc"

#include "test_instruction_roof_cases.inc"

#include "test_api_prepared_finite_hip_cases.inc"
#include "test_api_prepared_bounded_hip_cases.inc"

#include "test_api_resident_finite_cases.inc"

#include "test_api_matrix_cleanup_cases.inc"

#include "test_api_resident_bounded_cases.inc"

TEST_CASE("HIP event-dependent joins submit before producer observation", "[api][async][pipeline]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  check_prepared_multi_dependencies(RNS8_BACKEND_HIP_DIRECT,true,false);
  check_prepared_multi_dependencies(RNS8_BACKEND_HIP_DIRECT,true,true);
#else
  SKIP("HIP event-dependent submission is unavailable in this build");
#endif
}

#include "test_api_pipeline_bounded_cases.inc"

#include "test_api_materialized_prefix_cases.inc"

#include "test_arithmetic_certificate_cases.inc"

#include "test_quarter_crt_cases.inc"

#include "test_api_operation_provider_cases.inc"

#include "test_api_int4_cases.inc"

#include "backend_common/direct_rns_route.hpp"
#include "backend_common/vector_small_n.hpp"
TEST_CASE("public uniform plan metadata and dispatch share one qualified route", "[api][skinny]") {
  using namespace rns8::detail;
  for(auto semantics:{RNS8_BOUNDED_I64,RNS8_BOUNDED_U64,RNS8_EXACT_WIDE_SIGNED})for(int64_t m:{1,16,64,65,128})for(int64_t n:{1,2,4,8,9})for(uint32_t prefix:{1u,3u,9u,20u})for(bool parallel:{false,true}){
    rns8_plan plan{};plan.backend=RNS8_BACKEND_HIP_DIRECT;plan.prefix=prefix;plan.desc.semantics=semantics;
    plan.desc.m=m;plan.desc.n=n;plan.desc.k=65537;plan.desc.flags=parallel?RNS8_PLAN_DIRECT_RNS_PARALLEL:0;
    const auto route=direct_rns_route(m,n,65537,65537,n,n,prefix,parallel);
    CHECK(api::selected_kernel_for_plan(plan)==direct_rns_route_name(route));
    if(!parallel)CHECK(route==(m*n<=64?DirectRnsRoute::persistent_small:DirectRnsRoute::grouped));
    if(parallel && n<=8)CHECK(route==(n==1?DirectRnsRoute::gemv_n1:DirectRnsRoute::gemv_small_n));
  }
  CHECK(vector_small_n_declared_lds(2,true,true)==12800);CHECK(vector_small_n_declared_lds(2,false,true)==12288);
  CHECK(vector_small_n_declared_lds(2,true,false)==51200);CHECK(vector_small_n_declared_lds(2,false,false)==49152);
  for(uint32_t n=2;n<=8;++n){CHECK(vector_small_n_width(n,true)==n);CHECK(vector_small_n_width(n,false)==8);}
}

TEST_CASE("narrow fused resource contracts omit native intermediate but retain proof scan", "[narrow][resources]") {
  using namespace rns8::detail;
  for(bool wmma:{false,true})for(uint32_t prefix:{1u,2u,9u,20u}) {
    const auto staged=query_narrow_rns_support(17,19,31,32,21,prefix,wmma,UINT64_MAX);
    const auto fused=query_narrow_rns_support(17,19,31,32,21,prefix,wmma,UINT64_MAX,true);
    CHECK(staged.status==RNS8_SUCCESS);CHECK(fused.status==RNS8_SUCCESS);
    CHECK(staged.scratch_bytes-fused.scratch_bytes==17*19*8);CHECK(fused.scratch_bytes==(wmma?32:16));
    CHECK(query_narrow_rns_support(17,19,31,32,21,prefix,wmma,fused.scratch_bytes-1,true).status==RNS8_WORKSPACE_TOO_SMALL);
  }
}
TEST_CASE("synchronous failure leases invalidate only written outputs and release clean failures", "[api][lifetime]") {
  using namespace rns8::detail::api;auto* ctx=create_cpu_context();rns8_matrix a{},b{},c{};rns8_workspace workspace{};
  c.host_residues_current=true;size_t token=0;REQUIRE(begin_synchronous_lease(*ctx,a,b,c,workspace,token)==RNS8_SUCCESS);
  CHECK(a.in_flight_readers==1);CHECK(c.in_flight_writer);CHECK(workspace.in_flight);
  REQUIRE(finish_synchronous_lease(*ctx,token,RNS8_RANGE_ERROR,false));CHECK(c.host_residues_current);CHECK_FALSE(c.in_flight_writer);
  REQUIRE(begin_synchronous_lease(*ctx,a,a,c,workspace,token)==RNS8_SUCCESS);CHECK(a.in_flight_readers==1);
  REQUIRE(finish_synchronous_lease(*ctx,token,RNS8_RANGE_ERROR,true));CHECK_FALSE(c.host_residues_current);CHECK(a.in_flight_readers==0);
  CHECK(ctx->failed_work_leases.empty());REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
