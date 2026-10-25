#pragma once
#include "core/api_internal.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include "rns8/experimental_product.h"
#include "experimental_basis_fixture.hpp"
#include <array>
#include <algorithm>
#include <cstring>
#include <vector>

inline void check_product_materializations(rns8_backend_kind backend,bool experimental) {
  using namespace rns8::detail;using boost::multiprecision::cpp_int;
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=backend;
  rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(backend==RNS8_BACKEND_CPU_REFERENCE?-1:0,&options,&ctx)==RNS8_SUCCESS);
  auto* basis=experimental?experimental_basis_fixture::create(experimental_basis_fixture::moduli()):nullptr;
  for(const auto semantics:{RNS8_EXACT_WIDE_SIGNED,RNS8_EXACT_WIDE_UNSIGNED})
    for(bool diamond:{false,true})for(const auto policy:{RNS8_PRODUCT_FULL_PREFIX,RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX}) {
      rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.rows=md.cols=2;
      md.semantics=semantics;md.bound_kind=RNS8_BOUND_NONE;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.max_prefix=20;
      std::array<rns8_matrix*,4> matrices{};
      for(auto& matrix:matrices)REQUIRE((experimental?rns8_create_basis_matrix(ctx,basis,&md,&matrix):rns8_create_matrix(ctx,&md,&matrix))==RNS8_SUCCESS);
      rns8_gemm_desc gd{};gd.struct_size=sizeof(gd);gd.abi_version=RNS8_ABI_VERSION;gd.m=gd.n=gd.k=2;
      gd.semantics=semantics;gd.bound_kind=RNS8_BOUND_NONE;gd.max_prefix=20;gd.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
      rns8_plan* export_plan=nullptr;rns8_basis_plan* basis_plan=nullptr;
      if(experimental)basis_plan=experimental_basis_fixture::plan(ctx,basis,semantics,2,2,2,20);
      else REQUIRE(rns8_create_plan(ctx,&gd,&export_plan)==RNS8_SUCCESS);
      const auto pack=[&](unsigned index,uint64_t version=1) {
        const uint64_t v=index?3:2;const uint64_t words[]{v,0,99,0,v,99};
        if(semantics==RNS8_EXACT_WIDE_UNSIGNED)return experimental?rns8_pack_basis_u64(basis_plan,matrices[index],words,3,version):rns8_pack_u64(ctx,matrices[index],words,3,version);
        return experimental?rns8_pack_basis_i64(basis_plan,matrices[index],reinterpret_cast<const int64_t*>(words),3,version):
            rns8_pack_i64(ctx,matrices[index],reinterpret_cast<const int64_t*>(words),3,version);
      };
      REQUIRE(pack(0)==RNS8_SUCCESS);REQUIRE(pack(1)==RNS8_SUCCESS);
      const rns8_matrix* inputs[]{matrices[0],matrices[1]};
      const std::vector<rns8_product_node> nodes=diamond?std::vector<rns8_product_node>{{0,1},{2,1},{2,1},{3,4}}:
          std::vector<rns8_product_node>{{0,1},{2,1},{3,1}};
      const std::array<uint32_t,2> outputs{2,static_cast<uint32_t>(nodes.size()+1)};
      rns8_product_recipe_desc desc{};desc.struct_size=sizeof(desc);desc.abi_version=RNS8_ABI_VERSION;desc.prefix_policy=policy;
      desc.inputs=inputs;desc.input_count=2;desc.nodes=nodes.data();desc.node_count=static_cast<uint32_t>(nodes.size());desc.max_arena_and_lift_bytes=UINT64_MAX;
      rns8_product_materialization_profile prices{};prices.struct_size=sizeof(prices);prices.abi_version=RNS8_ABI_VERSION;
      prices.cost_schema_version=RNS8_PRODUCT_MATERIALIZATION_COST_SCHEMA_VERSION;prices.node_costs_size=sizeof(rns8_product_materialization_node_costs);
      prices.node_count=desc.node_count;prices.max_evaluations=4096;prices.max_chain_frontier=256;
      {constexpr char label[]="synthetic-correctness-only-profile";std::copy_n(label,sizeof(label),prices.calibration_identity);}
      prices.identity.struct_size=sizeof(prices.identity);prices.identity.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_query_product_materialization_identity(ctx,&desc,outputs.data(),2,&prices.identity)==RNS8_SUCCESS);
      REQUIRE(prices.identity.admission_status==RNS8_SUCCESS);CHECK(prices.identity.blocking_node==UINT32_MAX);
      std::vector<rns8_product_materialization_node_costs> costs(nodes.size());
      for(auto& cost:costs) {
        std::fill_n(cost.uniform_gemm_ns,21,UINT64_MAX);std::fill_n(cost.packed_gemm_ns,441,UINT64_MAX);
        std::fill_n(cost.lift_ns,441,UINT64_MAX);std::fill_n(cost.pack_ns,9261,UINT64_MAX);
        for(uint32_t p=1;p<=20;++p) {
          cost.uniform_gemm_ns[p]=1000*p;
          for(uint32_t old=1;old<p;++old)cost.lift_ns[old*21+p]=1;
          for(uint32_t capacity=p;capacity<=20;++capacity) {
            cost.packed_gemm_ns[capacity*21+p]=10*p;
            for(uint32_t old=0;old<p;++old)cost.pack_ns[(capacity*21+old)*21+p]=1;
          }
        }
      }
      prices.nodes=costs.data();rns8_product_recipe* recipe=nullptr;
      // These malformed headers carry an unreadable node pointer so a widened
      // table read before schema/stride/outer-size validation is observable.
      for(uint32_t schema:{0u,1u,3u}) {
        auto malformed=prices;malformed.cost_schema_version=schema;
        malformed.nodes=reinterpret_cast<const rns8_product_materialization_node_costs*>(uintptr_t(1));
        CHECK(rns8_create_materialized_product_recipe(ctx,&desc,outputs.data(),2,&malformed,&recipe)==RNS8_INVALID_ARGUMENT);CHECK_FALSE(recipe);
      }
      for(uint32_t stride:{uint32_t(0),uint32_t((21+441+21+441)*sizeof(uint64_t)),uint32_t(sizeof(rns8_product_materialization_node_costs)-8),
          uint32_t(sizeof(rns8_product_materialization_node_costs)+8)}) {
        auto malformed=prices;malformed.node_costs_size=stride;
        malformed.nodes=reinterpret_cast<const rns8_product_materialization_node_costs*>(uintptr_t(1));
        CHECK(rns8_create_materialized_product_recipe(ctx,&desc,outputs.data(),2,&malformed,&recipe)==RNS8_INVALID_ARGUMENT);CHECK_FALSE(recipe);
      }
      struct ShortProfileHeader {uint64_t struct_size;uint32_t abi_version;};
      const ShortProfileHeader short_profile{sizeof(prices)-8,RNS8_ABI_VERSION};
      CHECK(rns8_create_materialized_product_recipe(ctx,&desc,outputs.data(),2,
          reinterpret_cast<const rns8_product_materialization_profile*>(&short_profile),&recipe)==RNS8_INVALID_ARGUMENT);CHECK_FALSE(recipe);
      auto stale=prices;stale.identity.build_sha256[0]=stale.identity.build_sha256[0]=='a'?'b':'a';
      CHECK(rns8_create_materialized_product_recipe(ctx,&desc,outputs.data(),2,&stale,&recipe)==RNS8_INVALID_ARGUMENT);CHECK_FALSE(recipe);
      REQUIRE(rns8_create_materialized_product_recipe(ctx,&desc,outputs.data(),2,&prices,&recipe)==RNS8_SUCCESS);
      rns8_product_materialization_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_get_product_materialization_info(recipe,&info)==RNS8_SUCCESS);CHECK(info.packed_payload_bytes>0);
      std::vector<rns8_product_materialization_choice> choices(nodes.size());
      REQUIRE(rns8_get_product_materialization_choices(recipe,choices.data(),static_cast<uint32_t>(choices.size()))==RNS8_SUCCESS);
      for(const auto& choice:choices) {
        CHECK(choice.packed_b==1);CHECK(choice.packed_capacity_prefix>=choice.compute_prefix);CHECK(choice.arena_offset+choice.packed_bytes<=info.retained_arena_and_lift_bytes);
        if(policy==RNS8_PRODUCT_FULL_PREFIX)CHECK(choice.compute_prefix==20);
      }
      const std::array<rns8_matrix*,2> destinations{matrices[2],matrices[3]};
      for(unsigned repeat=0;repeat<2;++repeat) {
        const auto before=hip_direct_allocation_counters_snapshot();
        REQUIRE(rns8_execute_product_recipe_outputs(recipe,destinations.data(),2)==RNS8_SUCCESS);
        const auto after=hip_direct_allocation_counters_snapshot();
        CHECK(after.allocate_calls==before.allocate_calls);CHECK(after.free_calls==before.free_calls);
        REQUIRE(rns8_get_product_materialization_info(recipe,&info)==RNS8_SUCCESS);
        if(policy==RNS8_PRODUCT_FULL_PREFIX)CHECK(info.completed_packing_calls==(diamond?2:1));
        CHECK(info.completed_packing_calls<=nodes.size());
        for(unsigned output=0;output<2;++output) {
          std::vector<uint64_t> limbs(2*3*3,UINT64_C(0xfeed));
          const auto status=experimental?rns8_export_basis_limbs(basis_plan,destinations[output],limbs.data(),3,3):
              semantics==RNS8_EXACT_WIDE_SIGNED?rns8_export_exact_wide_signed_limbs(ctx,export_plan,destinations[output],limbs.data(),3,3):
              rns8_export_exact_wide_unsigned_limbs(ctx,export_plan,destinations[output],limbs.data(),3,3);
          REQUIRE(status==RNS8_SUCCESS);const cpp_int expected=output?(diamond?324:54):6;
          for(unsigned row=0;row<2;++row)for(unsigned col=0;col<3;++col) {
            const auto index=(row*3+col)*3;
            if(col==2){for(unsigned limb=0;limb<3;++limb)CHECK(limbs[index+limb]==UINT64_C(0xfeed));continue;}
            cpp_int actual=cpp_int(limbs[index])+(cpp_int(limbs[index+1])<<64)+(cpp_int(limbs[index+2])<<128);
            if(semantics==RNS8_EXACT_WIDE_SIGNED && limbs[index+2]>>63)actual-=cpp_int(1)<<192;
            CHECK(actual==(row==col?expected:cpp_int(0)));
          }
        }
      }
      REQUIRE(pack(0,2)==RNS8_SUCCESS);const auto generation=matrices[2]->content_generation;
      CHECK(rns8_execute_product_recipe_outputs(recipe,destinations.data(),2)==RNS8_INVALID_ARGUMENT);
      CHECK(matrices[2]->content_generation==generation);
      REQUIRE(rns8_rebind_product_recipe_inputs(recipe,inputs,2)==RNS8_SUCCESS);
      REQUIRE(rns8_execute_product_recipe_outputs(recipe,destinations.data(),2)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_product_recipe(recipe)==RNS8_SUCCESS);
      if(basis_plan)REQUIRE(rns8_destroy_basis_plan(basis_plan)==RNS8_SUCCESS);
      if(export_plan)REQUIRE(rns8_destroy_plan(export_plan)==RNS8_SUCCESS);
      for(auto* matrix:matrices)REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
    }
  if(basis)REQUIRE(rns8_destroy_basis(basis)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
