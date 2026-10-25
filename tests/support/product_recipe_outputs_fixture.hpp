#pragma once
#include "core/api_internal.hpp"
#include "core/runtime_product_recipe.hpp"
#include <array>
inline void check_product_recipe_outputs(rns8_backend_kind backend) {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
  options.requested_backend=backend;
  rns8_context* ctx=nullptr;
  REQUIRE(rns8_create_context(backend==RNS8_BACKEND_HIP_DIRECT?0:-1,&options,&ctx)==RNS8_SUCCESS);
  for(auto semantics:{RNS8_EXACT_WIDE_SIGNED,RNS8_EXACT_WIDE_UNSIGNED})
    for(auto policy:{RNS8_PRODUCT_FULL_PREFIX,RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX}) {
      constexpr int n=3;
      rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;
      md.rows=md.cols=n;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;
      md.semantics=semantics;md.bound_kind=RNS8_BOUND_NONE;md.max_prefix=20;
      std::array<rns8_matrix*,5> matrices{};
      for(auto& matrix:matrices) REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
      std::vector<std::vector<cpp_int>> oracle(2,std::vector<cpp_int>(n*n));
      for(int input=0;input<2;++input) {
        std::array<int64_t,n*(n+2)> signed_values{};
        std::array<uint64_t,n*(n+2)> unsigned_values{};
        for(int r=0;r<n;++r) for(int c=0;c<n;++c) {
          const int value=(r*3+c+input)%5;
          signed_values[r*(n+2)+c]=value-2;unsigned_values[r*(n+2)+c]=value;
          oracle[input][r*n+c]=semantics==RNS8_EXACT_WIDE_SIGNED?value-2:value;
        }
        if(semantics==RNS8_EXACT_WIDE_SIGNED)
          REQUIRE(rns8_pack_i64(ctx,matrices[input],signed_values.data(),n+2,0)==RNS8_SUCCESS);
        else REQUIRE(rns8_pack_u64(ctx,matrices[input],unsigned_values.data(),n+2,0)==RNS8_SUCCESS);
      }
      const rns8_product_node nodes[]{{0,1},{2,1},{2,0},{3,4}};
      for(auto node:nodes) {
        std::vector<cpp_int> product(n*n);
        for(int r=0;r<n;++r) for(int c=0;c<n;++c) for(int q=0;q<n;++q)
          product[r*n+c]+=oracle[node.a][r*n+q]*oracle[node.b][q*n+c];
        oracle.push_back(std::move(product));
      }
      const rns8_matrix* inputs[]{matrices[0],matrices[1]};
      const uint32_t ids[]{5,2,3}; // deliberately non-topological publication order
      rns8_product_recipe_desc desc{};desc.struct_size=sizeof(desc);desc.abi_version=RNS8_ABI_VERSION;
      desc.inputs=inputs;desc.input_count=2;desc.nodes=nodes;desc.node_count=4;
      desc.prefix_policy=policy;desc.max_arena_and_lift_bytes=UINT64_MAX;
      rns8_product_recipe* recipe=nullptr;
      uint32_t invalid[]{2,2};
      CHECK(rns8_create_product_recipe_outputs(ctx,&desc,invalid,2,&recipe)==RNS8_INVALID_ARGUMENT);
      CHECK(recipe==nullptr);invalid[1]=0;
      CHECK(rns8_create_product_recipe_outputs(ctx,&desc,invalid,2,&recipe)==RNS8_INVALID_ARGUMENT);
      invalid[1]=6;
      CHECK(rns8_create_product_recipe_outputs(ctx,&desc,invalid,2,&recipe)==RNS8_INVALID_ARGUMENT);
      REQUIRE(rns8_create_product_recipe_outputs(ctx,&desc,ids,3,&recipe)==RNS8_SUCCESS);
      rns8_product_recipe_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_get_product_recipe_resources(recipe,&resources)==RNS8_SUCCESS);
      desc.max_arena_and_lift_bytes=resources.arena_bytes+resources.lift_scratch_bytes-1;
      rns8_product_recipe* rejected=nullptr;
      CHECK(rns8_create_product_recipe_outputs(ctx,&desc,ids,3,&rejected)==RNS8_WORKSPACE_TOO_SMALL);
      CHECK(rejected==nullptr);
      rns8_matrix* outputs[]{matrices[2],matrices[3],matrices[4]};
      CHECK(rns8_execute_product_recipe(recipe,outputs[0])==RNS8_INVALID_ARGUMENT);
      for(int repeat=0;repeat<2;++repeat) {
        const auto allocations_before=rns8::detail::hip_direct_allocation_counters_snapshot();
        REQUIRE(rns8_execute_product_recipe_outputs(recipe,outputs,3)==RNS8_SUCCESS);
        const auto allocations_after=rns8::detail::hip_direct_allocation_counters_snapshot();
        CHECK(allocations_after.allocate_calls==allocations_before.allocate_calls);
        CHECK(allocations_after.free_calls==allocations_before.free_calls);
        for(int i=0;i<3;++i) {
          std::vector<int8_t> residues(20*n*n);
          if(backend==RNS8_BACKEND_HIP_DIRECT)
            REQUIRE(hip_direct_copy_device_to_host(0,residues.data(),outputs[i]->hip_residues,residues.size())==RNS8_SUCCESS);
          else residues=outputs[i]->residues;
          for(uint32_t p=0;p<20;++p) for(int cell=0;cell<n*n;++cell)
            REQUIRE(residues[p*n*n+cell]==centered_residue(oracle[ids[i]][cell],kDefaultModuli[p]));
          CHECK(outputs[i]->exact_range_prefix==20);
          CHECK_FALSE(outputs[i]->in_flight_writer);CHECK(outputs[i]->prepared_references==0);
        }
      }
      std::array<uint64_t,3> generations{};
      for(int i=0;i<3;++i) generations[i]=outputs[i]->content_generation;
      outputs[2]->in_flight_readers=1;
      CHECK(rns8_execute_product_recipe_outputs(recipe,outputs,3)==RNS8_BUSY);
      outputs[2]->in_flight_readers=0;
      rns8_matrix* duplicate[]{outputs[0],outputs[1],outputs[0]};
      CHECK(rns8_execute_product_recipe_outputs(recipe,duplicate,3)==RNS8_INVALID_ARGUMENT);
      duplicate[2]=matrices[0];
      CHECK(rns8_execute_product_recipe_outputs(recipe,duplicate,3)==RNS8_INVALID_ARGUMENT);
      duplicate[2]=nullptr;
      CHECK(rns8_execute_product_recipe_outputs(recipe,duplicate,3)==RNS8_INVALID_ARGUMENT);
      for(int i=0;i<3;++i) CHECK(outputs[i]->content_generation==generations[i]);
      REQUIRE(rns8_destroy_product_recipe(recipe)==RNS8_SUCCESS);
      const std::vector<ProductNode> internal_nodes{{0,1},{2,1},{2,0},{3,4}};
      const std::vector<uint32_t> observed{5,2,3};
      ProductCostProfile costs;costs.backend=backend;costs.semantics=semantics;
      costs.target=backend==RNS8_BACKEND_HIP_DIRECT?ctx->device_info.gcn_arch:"cpu";
      costs.provenance="synthetic-multi-output-test";
      costs.graph_key=product_graph_key({{n,n},{n,n}},internal_nodes,20,observed);
      costs.nodes.resize(4);
      for(auto& node:costs.nodes) for(uint32_t p=1;p<=20;++p) {
        node.gemm[p]=100*p;
        for(uint32_t q=p+1;q<=20;++q) node.lift[p*21+q]=1;
      }
      std::unique_ptr<RuntimeProductRecipe> selected;
      REQUIRE(RuntimeProductRecipe::create(ctx,{inputs[0],inputs[1]},internal_nodes,
          UINT64_MAX,selected,false,&costs,observed)==RNS8_SUCCESS);
      CHECK_FALSE(selected->cost_selection().chain_dynamic_programming);
      CHECK(selected->cost_selection().peak_bytes==selected->allocation().arena_bytes+selected->lift_scratch_bytes());
      REQUIRE(selected->execute({outputs[0],outputs[1],outputs[2]})==RNS8_SUCCESS);
      for(int i=0;i<3;++i) {
        std::vector<int8_t> residues(20*n*n);
        if(backend==RNS8_BACKEND_HIP_DIRECT)
          REQUIRE(hip_direct_copy_device_to_host(0,residues.data(),outputs[i]->hip_residues,residues.size())==RNS8_SUCCESS);
        else residues=outputs[i]->residues;
        for(uint32_t p=0;p<20;++p) for(int cell=0;cell<n*n;++cell)
          CHECK(residues[p*n*n+cell]==centered_residue(oracle[ids[i]][cell],kDefaultModuli[p]));
      }
      selected.reset();
      for(auto* matrix:matrices) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
    }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
