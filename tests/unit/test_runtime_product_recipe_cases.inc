#include "core/runtime_product_recipe.hpp"
TEST_CASE("runtime product arena respects every diamond reader") {
  using namespace rns8::detail;
  const std::vector<ProductShape> inputs{{16,16},{16,16}};
  const std::vector<ProductNode> chain{{0,1},{2,1},{3,1},{4,1}};
  const std::vector<ProductNode> diamond{{0,1},{2,1},{2,1},{3,4}};
  for(const auto& nodes:{chain,diamond}) {
    const auto plan=plan_product_arena(inputs,nodes,20,UINT64_MAX);
    REQUIRE(plan.status==RNS8_SUCCESS);
    CHECK(plan.arena_bytes==(nodes[2].a==2?3u:2u)*5120);
    CHECK(plan.unreused_bytes==4*5120);
    for(size_t i=0;i<nodes.size();++i) for(size_t j=i+1;j<nodes.size();++j) {
      const auto& a=plan.values[2+i]; const auto& b=plan.values[2+j];
      if(a.last_use>=j) CHECK((a.offset+a.bytes<=b.offset || b.offset+b.bytes<=a.offset));
    }
    CHECK(plan_product_arena(inputs,nodes,20,plan.arena_bytes-1).status==RNS8_WORKSPACE_TOO_SMALL);
  }
  CHECK(plan_product_arena({{1,1},{1,PTRDIFF_MAX}},{{0,1}},1,UINT64_MAX).status==RNS8_RANGE_ERROR);
  // Individually valid regions can exceed the whole-object difference domain.
  const int64_t huge=(PTRDIFF_MAX/2)&~int64_t{255};
  CHECK(plan_product_arena({{1,1},{1,huge}},{{0,1},{0,2},{0,2},{0,3}},1,UINT64_MAX).status==RNS8_RANGE_ERROR);
  CHECK(plan_product_arena(inputs,{{0,2}},20,UINT64_MAX).status==RNS8_INVALID_ARGUMENT);
  CHECK(plan_product_arena({{16,17},{16,16}},{{0,1}},20,UINT64_MAX).status==RNS8_INVALID_ARGUMENT);
}
TEST_CASE("runtime product recipe executes CPU chain and diamond in a reused arena") {
  using namespace rns8::detail;
  auto* ctx=create_cpu_context();
  auto desc=exact_desc(RNS8_EXACT_WIDE_SIGNED,16,16,16);
  desc.max_prefix=20; desc.flags|=RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* plan=nullptr;
  REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  auto md=exact_matrix_desc(16,16,RNS8_EXACT_WIDE_SIGNED); md.max_prefix=20;
  rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
  for(auto** matrix:{&a,&b,&c}) REQUIRE(rns8_create_matrix(ctx,&md,matrix)==RNS8_SUCCESS);
  std::vector<int64_t> av(256,0),bv(256,0);
  for(int i=0;i<16;++i) { av[i*16+i]=2; bv[i*16+i]=3; }
  REQUIRE(rns8_pack_i64(ctx,a,av.data(),16,0)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx,b,bv.data(),16,0)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx,c,av.data(),16,0)==RNS8_SUCCESS);
  for(int variant=0;variant<4;++variant) {
    const bool diamond=variant%2,adaptive=variant>=2;
    const std::vector<ProductNode> nodes=diamond?std::vector<ProductNode>{{0,1},{2,1},{2,1},{3,4}}
        :std::vector<ProductNode>{{0,1},{2,1},{3,1},{4,1}};
    std::unique_ptr<RuntimeProductRecipe> recipe;
    CHECK(RuntimeProductRecipe::create(ctx,{a,b},nodes,0,recipe,adaptive)==RNS8_WORKSPACE_TOO_SMALL);
    CHECK_FALSE(recipe);
    REQUIRE(RuntimeProductRecipe::create(ctx,{a,b},nodes,UINT64_MAX,recipe,adaptive)==RNS8_SUCCESS);
    REQUIRE(recipe->compute_prefixes().size()==nodes.size());
    CHECK(recipe->compute_prefixes().front()==(adaptive?1u:20u));
    CHECK(recipe->lift_scratch_bytes()==0); // CPU extends private cells directly
    CHECK(rns8_destroy_matrix(a)==RNS8_BUSY);
    const auto generation=c->content_generation;
    c->in_flight_readers=1;
    CHECK(recipe->execute(c)==RNS8_BUSY);
    c->in_flight_readers=0;
    CHECK(c->content_generation==generation);
    CHECK(recipe->execute(a)==RNS8_INVALID_ARGUMENT);
    for(int repeat=0;repeat<2;++repeat) {
      REQUIRE(recipe->execute(c)==RNS8_SUCCESS);
      std::vector<uint64_t> limbs(256);
      REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c,limbs.data(),16,1)==RNS8_SUCCESS);
      for(int row=0;row<16;++row) for(int col=0;col<16;++col)
        CHECK(limbs[row*16+col]==uint64_t(row==col?(diamond?324:162):0));
    }
    REQUIRE(rns8_pack_i64(ctx,a,av.data(),16,0)==RNS8_SUCCESS);
    const auto final_generation=c->content_generation;
    CHECK(recipe->execute(c)==RNS8_INVALID_ARGUMENT);
    CHECK(c->content_generation==final_generation);
  }
  for(auto* matrix:{a,b,c}) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
