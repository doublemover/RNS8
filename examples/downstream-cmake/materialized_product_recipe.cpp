#include <rns8/experimental_product.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "noninteractive_errors.hpp"

namespace {
void check(rns8_status status) {
  if(status!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(status));
}
struct Owners {
  rns8_context* ctx=nullptr;
  std::array<rns8_matrix*,3> matrices{};
  rns8_plan* plan=nullptr;
  rns8_product_recipe* recipe=nullptr;
  rns8_status close() noexcept {
    if(recipe){const auto s=rns8_destroy_product_recipe(recipe);if(s!=RNS8_SUCCESS)return s;recipe=nullptr;}
    for(auto& matrix:matrices)if(matrix){const auto s=rns8_destroy_matrix(matrix);if(s!=RNS8_SUCCESS)return s;matrix=nullptr;}
    if(plan){const auto s=rns8_destroy_plan(plan);if(s!=RNS8_SUCCESS)return s;plan=nullptr;}
    if(ctx){const auto s=rns8_destroy_context(ctx);if(s!=RNS8_SUCCESS)return s;ctx=nullptr;}
    return RNS8_SUCCESS;
  }
  ~Owners(){(void)close();}
};
}
int main() {
  rns8_example::configure_noninteractive_errors();
  try {
    Owners owner;rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=RNS8_BACKEND_CPU_REFERENCE;check(rns8_create_context(-1,&options,&owner.ctx));
    rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.rows=md.cols=1;
    md.semantics=RNS8_EXACT_WIDE_SIGNED;md.bound_kind=RNS8_BOUND_NONE;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.max_prefix=20;
    for(auto& matrix:owner.matrices)check(rns8_create_matrix(owner.ctx,&md,&matrix));
    const int64_t two=2,three=3;check(rns8_pack_i64(owner.ctx,owner.matrices[0],&two,1,1));
    check(rns8_pack_i64(owner.ctx,owner.matrices[1],&three,1,1));
    const rns8_matrix* inputs[]{owner.matrices[0],owner.matrices[1]};
    const rns8_product_node nodes[]{{0,1},{2,1},{3,1}};
    rns8_product_recipe_desc desc{};desc.struct_size=sizeof(desc);desc.abi_version=RNS8_ABI_VERSION;
    desc.inputs=inputs;desc.input_count=2;desc.nodes=nodes;desc.node_count=3;desc.prefix_policy=RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX;
    desc.max_arena_and_lift_bytes=1u<<20;
    rns8_product_materialization_profile prices{};prices.struct_size=sizeof(prices);prices.abi_version=RNS8_ABI_VERSION;
    prices.node_count=3;prices.max_evaluations=4096;prices.max_chain_frontier=256;
    prices.identity.struct_size=sizeof(prices.identity);prices.identity.abi_version=RNS8_ABI_VERSION;
    check(rns8_query_product_materialization_identity(owner.ctx,&desc,nullptr,0,&prices.identity));
    check(prices.identity.admission_status);
    // Illustrative estimates exercise the API; these are not measured timings.
    {constexpr char label[]="installed-CPU-example-synthetic-estimates";std::copy_n(label,sizeof(label),prices.calibration_identity);}
    std::vector<rns8_product_materialization_node_costs> costs(3);
    for(auto& cost:costs) {
      std::fill_n(cost.uniform_gemm_ns,21,UINT64_MAX);std::fill_n(cost.packed_gemm_ns,21,UINT64_MAX);
      std::fill_n(cost.lift_ns,441,UINT64_MAX);std::fill_n(cost.pack_ns,441,UINT64_MAX);
      for(uint32_t p=1;p<=20;++p) {
        cost.uniform_gemm_ns[p]=1000*p;cost.packed_gemm_ns[p]=10*p;
        for(uint32_t old=0;old<p;++old){cost.pack_ns[old*21+p]=1;if(old)cost.lift_ns[old*21+p]=1;}
      }
    }
    prices.nodes=costs.data();check(rns8_create_materialized_product_recipe(owner.ctx,&desc,nullptr,0,&prices,&owner.recipe));
    check(rns8_execute_product_recipe(owner.recipe,owner.matrices[2]));
    rns8_product_materialization_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    check(rns8_get_product_materialization_info(owner.recipe,&info));
    if(info.completed_packing_calls!=1)throw std::runtime_error("shared B did not use one packing");
    rns8_gemm_desc gd{};gd.struct_size=sizeof(gd);gd.abi_version=RNS8_ABI_VERSION;gd.m=gd.n=gd.k=1;
    gd.semantics=md.semantics;gd.bound_kind=RNS8_BOUND_NONE;gd.max_prefix=20;gd.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
    check(rns8_create_plan(owner.ctx,&gd,&owner.plan));uint64_t result=0;
    check(rns8_export_exact_wide_signed_limbs(owner.ctx,owner.plan,owner.matrices[2],&result,1,1));
    if(result!=54)throw std::runtime_error("exact chain result mismatch");
    check(owner.close());
    std::cout<<"materialized product: 2*3*3*3="<<result<<", completed packs="<<info.completed_packing_calls
        <<", retained bytes="<<info.retained_arena_and_lift_bytes<<'\n';
    return 0;
  } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
