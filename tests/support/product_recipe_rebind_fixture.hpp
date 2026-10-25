#pragma once
#include <array>
#include <vector>
inline void check_product_recipe_rebind(rns8_backend_kind backend) {
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=backend;
  rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(backend==RNS8_BACKEND_HIP_DIRECT?0:-1,&options,&ctx)==RNS8_SUCCESS);
  rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.rows=md.cols=md.logical_ld=1;
  md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.semantics=RNS8_EXACT_WIDE_SIGNED;md.bound_kind=RNS8_BOUND_NONE;md.max_prefix=20;
  std::array<rns8_matrix*,5> matrices{};
  for(auto& matrix:matrices) REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
  const std::array<int64_t,4> values={5,3,7,1};
  for(size_t i=0;i<4;++i) REQUIRE(rns8_pack_i64(ctx,matrices[i],&values[i],1,0)==RNS8_SUCCESS);
  const rns8_matrix* inputs[]={matrices[0],matrices[1]};
  rns8_product_node node{0,1};
  rns8_product_recipe_desc desc{};desc.struct_size=sizeof(desc);desc.abi_version=RNS8_ABI_VERSION;
  desc.inputs=inputs;desc.input_count=2;desc.nodes=&node;desc.node_count=1;
  desc.prefix_policy=RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX;desc.max_arena_and_lift_bytes=UINT64_MAX;
  rns8_product_recipe* recipe=nullptr;REQUIRE(rns8_create_product_recipe(ctx,&desc,&recipe)==RNS8_SUCCESS);
  rns8_product_recipe_resources before{};before.struct_size=sizeof(before);before.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_get_product_recipe_resources(recipe,&before)==RNS8_SUCCESS);
  const auto check_value=[&](int64_t expected) {
    const auto allocations_before=rns8::detail::hip_direct_allocation_counters_snapshot();
    REQUIRE(rns8_execute_product_recipe(recipe,matrices[4])==RNS8_SUCCESS);
    const auto allocations_after=rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(allocations_after.allocate_calls==allocations_before.allocate_calls);
    CHECK(allocations_after.free_calls==allocations_before.free_calls);
    std::vector<int8_t> residues(20);
    if(backend==RNS8_BACKEND_HIP_DIRECT)
      REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0,residues.data(),matrices[4]->hip_residues,20)==RNS8_SUCCESS);
    else residues=matrices[4]->residues;
    for(unsigned p=0;p<20;++p) {
      int64_t value=expected%rns8_default_modulus(p);if(value<0) value+=rns8_default_modulus(p);
      if(value>=(rns8_default_modulus(p)+1)/2) value-=rns8_default_modulus(p);
      REQUIRE(residues[p]==value);
    }
    for(auto* matrix:matrices) {REQUIRE_FALSE(matrix->in_flight_writer);REQUIRE(matrix->in_flight_readers==0);}
    REQUIRE(matrices[4]->prepared_references==0);
  };
  const rns8_matrix* replacement[]={matrices[2],matrices[3]};
  REQUIRE(rns8_rebind_product_recipe_inputs(recipe,nullptr,2)==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_rebind_product_recipe_inputs(recipe,replacement,1)==RNS8_INVALID_ARGUMENT);
  matrices[2]->in_flight_writer=true;
  REQUIRE(rns8_rebind_product_recipe_inputs(recipe,replacement,2)==RNS8_BUSY);
  matrices[2]->in_flight_writer=false;
  const auto original_refs=matrices[2]->prepared_references;
  matrices[2]->prepared_references=UINT64_MAX;
  // Range proof rejection takes precedence and cannot alter any reference.
  REQUIRE(rns8_rebind_product_recipe_inputs(recipe,replacement,2)==RNS8_RANGE_ERROR);
  REQUIRE(matrices[2]->prepared_references==UINT64_MAX);
  matrices[2]->prepared_references=original_refs;
  REQUIRE(rns8_rebind_product_recipe_inputs(recipe,replacement,2)==RNS8_RANGE_ERROR);
  REQUIRE(matrices[0]->prepared_references==1);REQUIRE(matrices[2]->prepared_references==0);check_value(15);
  int64_t value=2;REQUIRE(rns8_pack_i64(ctx,matrices[2],&value,1,0)==RNS8_SUCCESS);
  matrices[2]->prepared_references=UINT64_MAX;
  REQUIRE(rns8_rebind_product_recipe_inputs(recipe,replacement,2)==RNS8_RANGE_ERROR);
  REQUIRE(matrices[0]->prepared_references==1);
  matrices[2]->prepared_references=0;
  REQUIRE(rns8_rebind_product_recipe_inputs(recipe,replacement,2)==RNS8_SUCCESS);check_value(2);
  REQUIRE(matrices[0]->prepared_references==0);REQUIRE(matrices[1]->prepared_references==0);
  REQUIRE(matrices[2]->prepared_references==1);REQUIRE(matrices[3]->prepared_references==1);
  replacement[1]=matrices[2];REQUIRE(rns8_rebind_product_recipe_inputs(recipe,replacement,2)==RNS8_SUCCESS);check_value(4);
  REQUIRE(matrices[2]->prepared_references==1);REQUIRE(matrices[3]->prepared_references==0);
  value=1;REQUIRE(rns8_pack_i64(ctx,matrices[2],&value,1,0)==RNS8_SUCCESS);
  const auto generation=matrices[4]->content_generation;
  REQUIRE(rns8_execute_product_recipe(recipe,matrices[4])==RNS8_INVALID_ARGUMENT);REQUIRE(matrices[4]->content_generation==generation);
  REQUIRE(rns8_rebind_product_recipe_inputs(recipe,replacement,2)==RNS8_SUCCESS);check_value(1);
  REQUIRE(matrices[4]->exact_max_magnitude==15); // compiled envelope remains conservative
  rns8_product_recipe_resources after{};after.struct_size=sizeof(after);after.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_get_product_recipe_resources(recipe,&after)==RNS8_SUCCESS);
  REQUIRE(after.arena_bytes==before.arena_bytes);REQUIRE(after.lift_scratch_bytes==before.lift_scratch_bytes);
  REQUIRE(rns8_destroy_product_recipe(recipe)==RNS8_SUCCESS);
  for(auto* matrix:matrices) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
