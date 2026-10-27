#include "core/runtime_product_recipe.hpp"
#include "../support/product_materialization_fixture.hpp"
TEST_CASE("public HIP materialized product recipes preserve values reuse rebind and observed branches") {
  if(!hip_available())SKIP("no HIP device for materialized product recipes");
  check_product_materializations(RNS8_BACKEND_HIP_DIRECT,false);
  check_product_materializations(RNS8_BACKEND_HIP_DIRECT,true);
}
TEST_CASE("runtime product arena executes a HIP diamond with exact final publication") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  if(!hip_available()) SKIP("no HIP device for runtime product arena");
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr int side=17;
  auto desc=exact_signed_desc(side,side,side,RNS8_BACKEND_HIP_DIRECT);
  desc.max_prefix=20; desc.flags|=RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* plan=nullptr;
  REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  auto md=matrix_desc(side,side,RNS8_EXACT_WIDE_SIGNED,RNS8_BOUND_NONE); md.max_prefix=20;
  rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
  for(auto** matrix:{&a,&b,&c}) REQUIRE(rns8_create_matrix(ctx,&md,matrix)==RNS8_SUCCESS);
  std::vector<int64_t> av(side*side),bv(side*side);
  for(int row=0;row<side;++row) for(int col=0;col<side;++col) {
    av[row*side+col]=(row+col)%5-2;
    bv[row*side+col]=(row*3+col)%3-1;
  }
  REQUIRE(rns8_pack_i64(ctx,a,av.data(),side,0)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx,b,bv.data(),side,0)==RNS8_SUCCESS);
  std::unique_ptr<RuntimeProductRecipe> recipe;
  REQUIRE(RuntimeProductRecipe::create(ctx,{a,b},{{0,1},{2,1},{2,1},{3,4}},UINT64_MAX,recipe)==RNS8_SUCCESS);
  CHECK(recipe->allocation().arena_bytes<recipe->allocation().unreused_bytes);
  const auto multiply=[&](const std::vector<cpp_int>& a,const std::vector<cpp_int>& b) {
    std::vector<cpp_int> c(side*side);
    for(int row=0;row<side;++row) for(int col=0;col<side;++col)
      for(int q=0;q<side;++q) c[row*side+col]+=a[row*side+q]*b[q*side+col];
    return c;
  };
  const auto first=multiply(std::vector<cpp_int>(av.begin(),av.end()),std::vector<cpp_int>(bv.begin(),bv.end()));
  const auto second=multiply(first,std::vector<cpp_int>(bv.begin(),bv.end()));
  const auto expected=multiply(second,second);
  for(int repeat=0;repeat<6;++repeat) {
    if(repeat==2) {
      recipe.reset();
      REQUIRE(RuntimeProductRecipe::create(ctx,{a,b},{{0,1},{2,1},{2,1},{3,4}},UINT64_MAX,recipe,true)==RNS8_SUCCESS);
      CHECK(recipe->compute_prefixes().front()<20);
      CHECK(recipe->lift_scratch_bytes()>0);
      std::unique_ptr<RuntimeProductRecipe> tight;
      const uint64_t peak=recipe->allocation().arena_bytes+recipe->lift_scratch_bytes();
      CHECK(RuntimeProductRecipe::create(ctx,{a,b},{{0,1},{2,1},{2,1},{3,4}},peak-1,tight,true)==RNS8_WORKSPACE_TOO_SMALL);
      CHECK_FALSE(tight);
    }
    if(repeat==4) {
      recipe.reset();
      const std::vector<ProductNode> graph{{0,1},{2,1},{2,1},{3,4}};
      ProductCostProfile costs;costs.backend=RNS8_BACKEND_HIP_DIRECT;
      costs.target=ctx->device_info.gcn_arch;costs.provenance="synthetic-HIP-admission-profile";
      costs.graph_key=product_graph_key({{side,side},{side,side}},graph,20);costs.nodes.resize(4);
      for(auto& node:costs.nodes) for(uint32_t p=1;p<=20;++p) {
        node.gemm[p]=100*p;
        for(uint32_t q=p+1;q<=20;++q) node.lift[p*21+q]=1;
      }
      REQUIRE(RuntimeProductRecipe::create(ctx,{a,b},graph,UINT64_MAX,recipe,false,&costs)==RNS8_SUCCESS);
      CHECK(recipe->allocation().arena_bytes+recipe->lift_scratch_bytes()==recipe->cost_selection().peak_bytes);
    }
    REQUIRE(recipe->execute(c)==RNS8_SUCCESS);
    std::vector<uint64_t> limbs(side*side*3);
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c,limbs.data(),side,3)==RNS8_SUCCESS);
    for(int cell=0;cell<side*side;++cell) {
      cpp_int actual=cpp_int(limbs[cell*3])+(cpp_int(limbs[cell*3+1])<<64)+(cpp_int(limbs[cell*3+2])<<128);
      if(limbs[cell*3+2]>>63) actual-=cpp_int(1)<<192;
      CHECK(actual==expected[cell]);
    }
  }
  recipe.reset();
  const rns8_matrix* inputs[]{a,b};
  const rns8_product_node nodes[]{{0,1},{2,1},{2,1},{3,4}};
  rns8_product_recipe_desc public_desc{};
  public_desc.struct_size=sizeof(public_desc); public_desc.abi_version=RNS8_ABI_VERSION;
  public_desc.prefix_policy=RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX;
  public_desc.inputs=inputs; public_desc.input_count=2; public_desc.nodes=nodes; public_desc.node_count=4;
  public_desc.max_arena_and_lift_bytes=UINT64_MAX;
  rns8_product_recipe* public_recipe=nullptr;
  REQUIRE(rns8_create_product_recipe(ctx,&public_desc,&public_recipe)==RNS8_SUCCESS);
  REQUIRE(rns8_execute_product_recipe(public_recipe,c)==RNS8_SUCCESS);
  std::vector<uint64_t> final_limbs(side*side*3);
  REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c,final_limbs.data(),side,3)==RNS8_SUCCESS);
  for(int cell=0;cell<side*side;++cell) {
    cpp_int actual=cpp_int(final_limbs[cell*3])+(cpp_int(final_limbs[cell*3+1])<<64)+(cpp_int(final_limbs[cell*3+2])<<128);
    if(final_limbs[cell*3+2]>>63) actual-=cpp_int(1)<<192;
    CHECK(actual==expected[cell]);
  }
  REQUIRE(rns8_destroy_product_recipe(public_recipe)==RNS8_SUCCESS);
  for(auto* matrix:{a,b,c}) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

#include "../support/product_recipe_outputs_fixture.hpp"
TEST_CASE("public HIP product recipes publish multiple retained intermediate values") {
  if(!hip_available()) SKIP("no HIP device for multi-output product recipe");
  check_product_recipe_outputs(RNS8_BACKEND_HIP_DIRECT);
  check_product_recipe_outputs(RNS8_BACKEND_HIP_DIRECT,true);
}
