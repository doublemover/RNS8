#pragma once
#include "core/internal.hpp"
#include <rns8/experimental_product.h>
#include <cstring>
#include "core/hip_resources.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <vector>
namespace ragged_recipe_fixture {
inline void check(rns8_context* ctx,bool u,bool materialized=false) {
  using boost::multiprecision::cpp_int;constexpr int side=17;const auto semantics=u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;
  std::array<rns8_matrix*,5> matrices{};rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.semantics=semantics;md.bound_kind=RNS8_BOUND_NONE;
  md.rows=side;md.cols=side;md.logical_ld=side;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.max_prefix=20;
  for(auto& matrix:matrices)REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
  auto* source=matrices[0];auto* converted=matrices[1];auto* b=matrices[2];std::vector<uint64_t> native(side*side),ones(side*side,1);
  for(int row=0;row<side;++row)for(int col=0;col<side;++col){int64_t x=col<5?0:col<10?1:10000;if(!u && row%2)x=-x;native[row*side+col]=uint64_t(x);}
  const auto pack=[&](rns8_matrix* matrix,const std::vector<uint64_t>& x,uint64_t version){return u?rns8_pack_u64(ctx,matrix,x.data(),side,version):rns8_pack_i64(ctx,matrix,reinterpret_cast<const int64_t*>(x.data()),side,version);};
  REQUIRE(pack(source,native,1)==RNS8_SUCCESS);REQUIRE(pack(b,ones,1)==RNS8_SUCCESS);
  rns8_ragged_materialization* ragged=nullptr;REQUIRE(rns8_prepare_ragged_materialization(ctx,source,3,5,UINT64_MAX,&ragged)==RNS8_SUCCESS);
  rns8_ragged_resources rr{};rr.struct_size=sizeof(rr);rr.abi_version=RNS8_ABI_VERSION;REQUIRE(rns8_get_ragged_resources(ragged,&rr)==RNS8_SUCCESS);CHECK(rr.minimum_prefix<rr.maximum_prefix);
  const rns8_matrix* inputs[]{materialized?b:converted,materialized?converted:b};const rns8_product_node nodes[]{{0,1},{2,1},{0,2},{3,4}};const uint32_t outputs[]{3,5};
  rns8_product_recipe_desc desc{};desc.struct_size=sizeof(desc);desc.abi_version=RNS8_ABI_VERSION;desc.inputs=inputs;desc.input_count=2;desc.nodes=nodes;desc.node_count=4;desc.prefix_policy=RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX;desc.max_arena_and_lift_bytes=UINT64_MAX;
  rns8_ragged_recipe_binding binding{materialized?1u:0u,ragged,converted};rns8_product_recipe* recipe=nullptr;const auto original_generation=converted->content_generation;
  CHECK(rns8_create_product_recipe_ragged(ctx,&desc,&binding,1,outputs,2,0,&recipe)==RNS8_WORKSPACE_TOO_SMALL);CHECK(recipe==nullptr);CHECK(converted->content_generation==original_generation);
  if(materialized) {
    rns8_product_materialization_identity identity{};identity.struct_size=sizeof(identity);identity.abi_version=RNS8_ABI_VERSION;
    const auto before=rns8::detail::hip_direct_allocation_counters_snapshot();
    REQUIRE(rns8_query_product_materialization_identity_ragged(ctx,&desc,&binding,1,outputs,2,UINT64_MAX,&identity)==RNS8_SUCCESS);
    CHECK(identity.admission_status==RNS8_SUCCESS);CHECK(converted->content_generation==original_generation);
    CHECK(rns8::detail::hip_direct_allocation_counters_snapshot().allocate_calls==before.allocate_calls);
    std::array<rns8_product_materialization_node_costs,4> costs{};
    for(auto& cost:costs) {
      std::fill_n(cost.uniform_gemm_ns,21,UINT64_MAX);std::fill_n(cost.packed_gemm_ns,21,UINT64_MAX);
      std::fill_n(cost.lift_ns,441,1);std::fill_n(cost.pack_ns,441,1);
      for(uint32_t prefix=1;prefix<=20;++prefix)cost.packed_gemm_ns[prefix]=10*prefix;
    }
    rns8_product_materialization_profile profile{};profile.struct_size=sizeof(profile);profile.abi_version=RNS8_ABI_VERSION;
    profile.max_evaluations=64;profile.max_chain_frontier=64;profile.node_count=4;profile.identity=identity;profile.nodes=costs.data();
    std::strcpy(profile.calibration_identity,"synthetic-ragged-materialization-fixture");
    profile.identity.build_sha256[0]=profile.identity.build_sha256[0]=='0'?'1':'0';
    CHECK(rns8_create_materialized_product_recipe_ragged(ctx,&desc,&binding,1,outputs,2,UINT64_MAX,&profile,&recipe)==RNS8_INVALID_ARGUMENT);
    CHECK(recipe==nullptr);CHECK(converted->content_generation==original_generation);
    profile.identity=identity;
    REQUIRE(rns8_create_materialized_product_recipe_ragged(ctx,&desc,&binding,1,outputs,2,UINT64_MAX,&profile,&recipe)==RNS8_SUCCESS);
    std::array<rns8_product_materialization_choice,4> choices{};
    REQUIRE(rns8_get_product_materialization_choices(recipe,choices.data(),4)==RNS8_SUCCESS);
    for(const auto& choice:choices){CHECK(choice.packed_b==1);CHECK(choice.packed_bytes>0);}
  } else REQUIRE(rns8_create_product_recipe_ragged(ctx,&desc,&binding,1,outputs,2,UINT64_MAX,&recipe)==RNS8_SUCCESS);
  CHECK(rns8_destroy_ragged_materialization(ragged)==RNS8_BUSY);CHECK(rns8_destroy_matrix(converted)==RNS8_BUSY);
  rns8_product_recipe_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;REQUIRE(rns8_get_product_recipe_resources(recipe,&resources)==RNS8_SUCCESS);
  CHECK(resources.materialized_input_bytes>=side*side*20);CHECK(resources.ragged_retained_bytes>=rr.payload_bytes+rr.directory_bytes);CHECK(resources.arena_bytes>0);
  rns8_gemm_desc gd{};gd.struct_size=sizeof(gd);gd.abi_version=RNS8_ABI_VERSION;gd.semantics=semantics;gd.bound_kind=RNS8_BOUND_NONE;gd.requested_backend=ctx->backend;gd.m=side;gd.n=side;gd.k=side;gd.max_prefix=20;gd.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;rns8_plan* export_plan=nullptr;REQUIRE(rns8_create_plan(ctx,&gd,&export_plan)==RNS8_SUCCESS);
  rns8_matrix* destinations[]{matrices[3],matrices[4]};
  auto audit=[&](){std::vector<std::vector<cpp_int>> values(2,std::vector<cpp_int>(side*side));for(size_t i=0;i<native.size();++i){values[materialized?1:0][i]=native[i];if(!u && native[i]>>63)values[materialized?1:0][i]-=cpp_int(1)<<64;values[materialized?0:1][i]=1;}
    for(auto node:nodes){std::vector<cpp_int> value(side*side);for(int r=0;r<side;++r)for(int c=0;c<side;++c)for(int q=0;q<side;++q)value[r*side+c]+=values[node.a][r*side+q]*values[node.b][q*side+c];values.push_back(std::move(value));}
    for(unsigned which=0;which<2;++which){std::vector<uint64_t> limbs(side*(side+2)*3,99);REQUIRE((u?rns8_export_exact_wide_unsigned_limbs(ctx,export_plan,destinations[which],limbs.data(),side+2,3):rns8_export_exact_wide_signed_limbs(ctx,export_plan,destinations[which],limbs.data(),side+2,3))==RNS8_SUCCESS);
      for(int r=0;r<side;++r)for(int c=0;c<side+2;++c){const size_t at=(r*(side+2)+c)*3;if(c>=side){CHECK(limbs[at]==99);CHECK(limbs[at+1]==99);CHECK(limbs[at+2]==99);continue;}
        cpp_int got=cpp_int(limbs[at])+(cpp_int(limbs[at+1])<<64)+(cpp_int(limbs[at+2])<<128);if(!u && limbs[at+2]>>63)got-=cpp_int(1)<<192;CHECK(got==values[outputs[which]][r*side+c]);}}};
  for(int repeat=0;repeat<2;++repeat){const auto before=rns8::detail::hip_direct_allocation_counters_snapshot();REQUIRE(rns8_execute_product_recipe_outputs(recipe,destinations,2)==RNS8_SUCCESS);const auto after=rns8::detail::hip_direct_allocation_counters_snapshot();CHECK(after.allocate_calls==before.allocate_calls);CHECK(after.free_calls==before.free_calls);CHECK(after.h2d_bytes==before.h2d_bytes);audit();}
  const auto old_generation=destinations[1]->content_generation;for(auto& x:native)x=u?2:uint64_t(-2);REQUIRE(pack(source,native,2)==RNS8_SUCCESS);
  CHECK(rns8_execute_product_recipe_outputs(recipe,destinations,2)==RNS8_INVALID_ARGUMENT);CHECK(destinations[1]->content_generation==old_generation);
  CHECK(rns8_rebind_product_recipe_inputs(recipe,inputs,2)==RNS8_INVALID_ARGUMENT);
  rns8_ragged_materialization* refreshed=nullptr;REQUIRE(rns8_prepare_ragged_materialization(ctx,source,5,3,UINT64_MAX,&refreshed)==RNS8_SUCCESS);
  binding.source=refreshed;REQUIRE(rns8_refresh_product_recipe_ragged(recipe,&binding,1)==RNS8_SUCCESS);REQUIRE(rns8_destroy_ragged_materialization(ragged)==RNS8_SUCCESS);
  CHECK(rns8_destroy_ragged_materialization(refreshed)==RNS8_BUSY);REQUIRE(rns8_execute_product_recipe_outputs(recipe,destinations,2)==RNS8_SUCCESS);audit();
  REQUIRE(rns8_destroy_product_recipe(recipe)==RNS8_SUCCESS);REQUIRE(rns8_destroy_ragged_materialization(refreshed)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(export_plan)==RNS8_SUCCESS);
  for(auto* matrix:matrices)REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
}
}
