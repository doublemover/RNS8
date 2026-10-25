#include "core/runtime_product_recipe.hpp"
#include "../support/product_materialization_fixture.hpp"
TEST_CASE("public CPU materialized product recipes preserve values reuse rebind and observed branches") {
  check_product_materializations(RNS8_BACKEND_CPU_REFERENCE,false);
  check_product_materializations(RNS8_BACKEND_CPU_REFERENCE,true);
}
namespace {
rns8::detail::ProductMaterializationProfile materialization_prices(
    const std::vector<rns8::detail::ProductShape>& inputs,
    const std::vector<rns8::detail::ProductNode>& nodes,uint32_t prefix,
    const std::vector<uint32_t>& outputs={}) {
  using namespace rns8::detail;
  ProductMaterializationProfile p;p.uniform.target="cpu";p.uniform.provenance="synthetic-materialization-fixture";
  p.uniform.graph_key=product_graph_key(inputs,nodes,prefix,outputs);p.uniform.nodes.resize(nodes.size());p.packed.resize(nodes.size());
  for(size_t i=0;i<nodes.size();++i)for(uint32_t q=1;q<=prefix;++q) {
    p.uniform.nodes[i].gemm[q]=100*q;p.packed[i].gemm[q]=10*q;
    for(uint32_t old=0;old<q;++old) {
      p.packed[i].pack[old*21+q]=20*(q-old);
      if(old)p.uniform.nodes[i].lift[old*21+q]=5*(q-old);
    }
  }
  return p;
}
}
TEST_CASE("joint prefix and B-layout planning charges shared growth and physical liveness") {
  using namespace rns8::detail;
  const std::vector<ProductShape> inputs{{16,16},{16,16}};
  for(bool diamond:{false,true}) {
    const std::vector<ProductNode> nodes=diamond?std::vector<ProductNode>{{0,1},{2,1},{2,1},{3,4}}:
        std::vector<ProductNode>{{0,1},{2,1},{3,1}};
    auto prices=materialization_prices(inputs,nodes,3);
    const std::vector<uint32_t> minimum(nodes.size(),1);
    const auto selected=select_product_materializations(inputs,nodes,3,minimum,UINT64_MAX,false,prices);
    REQUIRE(selected.status==RNS8_SUCCESS);CHECK(selected.exhaustive);CHECK(selected.cost_optimal);
    CHECK(selected.estimated_ns==(diamond?90:60));CHECK(selected.prefixes==minimum);
    CHECK(selected.packed_b==std::vector<uint8_t>(nodes.size(),1));
    CHECK(selected.packed[1].first==0);CHECK(selected.packed[1].last==2);
    CHECK(selected.packed[1].layout.prefix==1);
    if(diamond) {
      auto bounded_prices=prices;bounded_prices.uniform.max_evaluations=128;
      const auto bounded=select_product_materializations(inputs,nodes,3,minimum,UINT64_MAX,false,bounded_prices);
      REQUIRE(bounded.status==RNS8_SUCCESS);CHECK_FALSE(bounded.exhaustive);CHECK_FALSE(bounded.chain_dynamic_programming);
      CHECK_FALSE(bounded.cost_optimal);CHECK(bounded.evaluated<=128);CHECK(bounded.estimated_ns==selected.estimated_ns);
    }
    struct Region {uint64_t offset,bytes;uint32_t first,last;};std::vector<Region> regions;
    for(uint32_t i=0;i<nodes.size();++i) {
      const auto& v=selected.arena.values[inputs.size()+i];regions.push_back({v.offset,v.bytes,i,v.last_use});
    }
    for(const auto& v:selected.packed)if(v.used)regions.push_back({v.offset,v.layout.bytes,v.first,v.last});
    for(size_t i=0;i<regions.size();++i)for(size_t j=i+1;j<regions.size();++j) {
      const auto a=regions[i],b=regions[j];
      if(a.first<=b.last && b.first<=a.last)CHECK((a.offset+a.bytes<=b.offset || b.offset+b.bytes<=a.offset));
    }
    CHECK(select_product_materializations(inputs,nodes,3,minimum,0,false,prices).status==RNS8_WORKSPACE_TOO_SMALL);
    const auto exact=select_product_materializations(inputs,nodes,3,minimum,selected.arena.arena_bytes,false,prices);
    REQUIRE(exact.status==RNS8_SUCCESS);CHECK(exact.estimated_ns==selected.estimated_ns);
    for(auto& row:prices.packed){row.gemm.fill(UINT64_MAX);row.pack.fill(UINT64_MAX);}
    const auto unknown=select_product_materializations(inputs,nodes,3,minimum,UINT64_MAX,false,prices);
    REQUIRE(unknown.status==RNS8_SUCCESS);CHECK_FALSE(unknown.cost_optimal);
    CHECK(unknown.packed_b==std::vector<uint8_t>(nodes.size(),0));
    for(auto& row:prices.uniform.nodes)row.gemm.fill(UINT64_MAX-1);
    CHECK(select_product_materializations(inputs,nodes,3,minimum,UINT64_MAX,false,prices).status==RNS8_UNSUPPORTED_BACKEND);
  }
}
TEST_CASE("bounded chain materialization DP agrees with exhaustive tiny-domain cost") {
  using namespace rns8::detail;
  const std::vector<ProductShape> inputs{{2,2},{2,2}};
  const std::vector<ProductNode> nodes{{0,1},{2,1},{3,1}};
  auto prices=materialization_prices(inputs,nodes,3);const std::vector<uint32_t> minimum{1,1,1};
  const auto exact=select_product_materializations(inputs,nodes,3,minimum,UINT64_MAX,false,prices);
  prices.uniform.max_evaluations=8;
  const auto dp=select_product_materializations(inputs,nodes,3,minimum,UINT64_MAX,false,prices);
  REQUIRE(dp.status==RNS8_SUCCESS);CHECK(dp.chain_dynamic_programming);CHECK_FALSE(dp.exhaustive);
  CHECK(dp.evaluated<=8);CHECK(dp.estimated_ns==exact.estimated_ns);CHECK(dp.cost_optimal);
  CHECK(dp.chain_cost_lower_bound==exact.estimated_ns);
  prices.max_frontier=1;
  const auto capped=select_product_materializations(inputs,nodes,3,minimum,UINT64_MAX,false,prices);
  REQUIRE(capped.status==RNS8_SUCCESS);CHECK(capped.chain_cost_lower_bound==UINT64_MAX);CHECK_FALSE(capped.cost_optimal);
  prices.uniform.graph_key^=1;
  CHECK(select_product_materializations(inputs,nodes,3,minimum,UINT64_MAX,false,prices).status==RNS8_INVALID_ARGUMENT);
}
TEST_CASE("materialization identity reports the first blocking edge before unbounded squaring") {
  auto* ctx=create_cpu_context();auto md=exact_matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED);md.max_prefix=2;
  rns8_matrix* matrix=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
  const int64_t two=2;REQUIRE(rns8_pack_i64(ctx,matrix,&two,1,1)==RNS8_SUCCESS);
  const rns8_matrix* inputs[]{matrix};std::vector<rns8_product_node> nodes;
  for(uint32_t i=0;i<64;++i)nodes.push_back({i,i});
  rns8_product_recipe_desc desc{};desc.struct_size=sizeof(desc);desc.abi_version=RNS8_ABI_VERSION;
  desc.inputs=inputs;desc.input_count=1;desc.nodes=nodes.data();desc.node_count=64;desc.max_arena_and_lift_bytes=UINT64_MAX;
  rns8_product_materialization_profile prices{};prices.struct_size=sizeof(prices);prices.abi_version=RNS8_ABI_VERSION;
  prices.node_count=64;prices.max_evaluations=4;prices.max_chain_frontier=1;
  prices.identity.struct_size=sizeof(prices.identity);prices.identity.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_query_product_materialization_identity(ctx,&desc,nullptr,0,&prices.identity)==RNS8_SUCCESS);
  CHECK(prices.identity.admission_status==RNS8_RANGE_ERROR);CHECK(prices.identity.blocking_node==3);
  CHECK(prices.identity.minimum_prefixes[0]==1);CHECK(prices.identity.minimum_prefixes[1]==1);CHECK(prices.identity.minimum_prefixes[2]==2);
  for(unsigned i=3;i<64;++i)CHECK(prices.identity.minimum_prefixes[i]==0);
  std::vector<rns8_product_materialization_node_costs> costs(64);prices.nodes=costs.data();
  std::strcpy(prices.calibration_identity,"prices-cannot-waive-range-proof");prices.identity.admission_status=RNS8_SUCCESS;
  rns8_product_recipe* recipe=nullptr;
  CHECK(rns8_create_materialized_product_recipe(ctx,&desc,nullptr,0,&prices,&recipe)==RNS8_RANGE_ERROR);CHECK_FALSE(recipe);
  REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
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

TEST_CASE("public product recipe owns its arena and exposes exact prefix decisions") {
  auto* ctx=create_cpu_context();
  auto md=exact_matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED); md.max_prefix=20;
  rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
  for(auto** matrix:{&a,&b,&c}) REQUIRE(rns8_create_matrix(ctx,&md,matrix)==RNS8_SUCCESS);
  int64_t two=2,three=3;
  REQUIRE(rns8_pack_i64(ctx,a,&two,1,0)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx,b,&three,1,0)==RNS8_SUCCESS);
  const rns8_matrix* inputs[]{a,b};
  const rns8_product_node nodes[]{{0,1},{2,1},{2,1},{3,4}};
  rns8_product_recipe_desc desc{};
  desc.struct_size=sizeof(desc); desc.abi_version=RNS8_ABI_VERSION;
  desc.inputs=inputs; desc.input_count=2; desc.nodes=nodes; desc.node_count=4;
  desc.max_arena_and_lift_bytes=UINT64_MAX;
  desc.prefix_policy=RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX;
  rns8_product_recipe* recipe=nullptr;
  REQUIRE(rns8_create_product_recipe(ctx,&desc,&recipe)==RNS8_SUCCESS);
  rns8_product_recipe_resources info{}; info.struct_size=sizeof(info); info.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_get_product_recipe_resources(recipe,&info)==RNS8_SUCCESS);
  CHECK(info.node_count==4); CHECK(info.arena_bytes>0); CHECK(info.lift_scratch_bytes==0);
  uint32_t prefixes[4]{99,99,99,99};
  CHECK(rns8_get_product_recipe_prefixes(recipe,prefixes,3)==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(prefixes[0]==99);
  REQUIRE(rns8_get_product_recipe_prefixes(recipe,prefixes,4)==RNS8_SUCCESS);
  CHECK(prefixes[0]==1); CHECK(prefixes[3]==2);
  CHECK(rns8_destroy_matrix(a)==RNS8_BUSY);
  REQUIRE(rns8_execute_product_recipe(recipe,c)==RNS8_SUCCESS);
  auto pd=exact_desc(RNS8_EXACT_WIDE_SIGNED,1,1,1); pd.max_prefix=20; pd.flags|=RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* plan=nullptr; REQUIRE(rns8_create_plan(ctx,&pd,&plan)==RNS8_SUCCESS);
  uint64_t result=0;
  REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c,&result,1,1)==RNS8_SUCCESS);
  CHECK(result==324);
  REQUIRE(rns8_pack_i64(ctx,b,&three,1,0)==RNS8_SUCCESS);
  const auto generation=c->content_generation;
  CHECK(rns8_execute_product_recipe(recipe,c)==RNS8_INVALID_ARGUMENT);
  CHECK(c->content_generation==generation);
  REQUIRE(rns8_destroy_product_recipe(recipe)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_product_recipe(nullptr)==RNS8_SUCCESS);
  for(auto* matrix:{a,b,c}) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

TEST_CASE("prefix cost selection prices shared incremental lifts and exact arena budget") {
  using namespace rns8::detail;
  const std::vector<ProductShape> inputs{{16,16},{16,16}};
  const std::vector<ProductNode> nodes{{0,1},{2,1},{2,1},{3,4}};
  ProductCostProfile costs;costs.target="cpu";costs.provenance="synthetic-unit-profile";
  costs.graph_key=product_graph_key(inputs,nodes,3);costs.nodes.resize(4);
  const auto fill=[&](uint64_t lift) {
    for(auto& node:costs.nodes) for(uint32_t p=1;p<=3;++p) {
      node.gemm[p]=100*p;
      for(uint32_t q=p+1;q<=3;++q) node.lift[p*21+q]=lift;
    }
  };
  fill(1);
  costs.version=1;
  CHECK(select_product_prefixes(inputs,nodes,3,{1,1,1,1},UINT64_MAX,false,costs).status==RNS8_INVALID_ARGUMENT);
  costs.version=2;
  auto cheap=select_product_prefixes(inputs,nodes,3,{1,1,1,1},UINT64_MAX,false,costs);
  REQUIRE(cheap.status==RNS8_SUCCESS);CHECK(cheap.exhaustive);CHECK(cheap.evaluated==81);
  CHECK(cheap.prefixes==std::vector<uint32_t>{1,1,1,1});CHECK(cheap.estimated_ns==401);
  fill(10000);
  const auto expensive=select_product_prefixes(inputs,nodes,3,{1,1,1,1},UINT64_MAX,false,costs);
  REQUIRE(expensive.status==RNS8_SUCCESS);
  CHECK(expensive.prefixes==std::vector<uint32_t>{3,3,3,3});CHECK(expensive.estimated_ns==1200);
  CHECK(expensive.peak_bytes>cheap.peak_bytes);
  const auto constrained=select_product_prefixes(inputs,nodes,3,{1,1,1,1},cheap.peak_bytes,false,costs);
  REQUIRE(constrained.status==RNS8_SUCCESS);CHECK(constrained.peak_bytes<=cheap.peak_bytes);
  CHECK(constrained.prefixes!=expensive.prefixes);
  const auto proof=select_product_prefixes(inputs,nodes,3,{3,3,3,3},UINT64_MAX,false,costs);
  REQUIRE(proof.status==RNS8_SUCCESS);CHECK(proof.prefixes==std::vector<uint32_t>{3,3,3,3});
  // Only one completely priced assignment; shared producer grows twice.
  costs.nodes.assign(4,ProductNodeCosts{});
  const uint32_t selected[]={1,2,3,3};
  for(unsigned i=0;i<4;++i) costs.nodes[i].gemm[selected[i]]=0;
  costs.nodes[0].lift[1*21+2]=7;costs.nodes[0].lift[2*21+3]=11;
  costs.nodes[1].lift[2*21+3]=13;
  const auto shared=select_product_prefixes(inputs,nodes,3,{1,1,1,1},UINT64_MAX,true,costs);
  REQUIRE(shared.status==RNS8_SUCCESS);CHECK(shared.estimated_ns==31);
  CHECK(shared.lift_scratch_bytes==256*2+4);
  CHECK(shared.prefixes==std::vector<uint32_t>{1,2,3,3});
  costs.experimental_basis=true;
  const auto experimental=select_product_prefixes(inputs,nodes,3,{1,1,1,1},UINT64_MAX,true,costs);
  REQUIRE(experimental.status==RNS8_SUCCESS);
  CHECK(experimental.lift_scratch_bytes==basis_lift_storage(256,1,3).scratch_bytes);
  CHECK(experimental.peak_bytes>shared.peak_bytes);
  CHECK(select_product_prefixes(inputs,nodes,3,{1,1,1,1},experimental.peak_bytes-1,true,costs).status!=RNS8_SUCCESS);
  costs.experimental_basis=false;
  costs.nodes[0].lift[2*21+3]=UINT64_MAX;
  CHECK(select_product_prefixes(inputs,nodes,3,{1,1,1,1},UINT64_MAX,true,costs).status==RNS8_UNSUPPORTED_BACKEND);
  costs.graph_key^=1;
  CHECK(select_product_prefixes(inputs,nodes,3,{1,1,1,1},UINT64_MAX,true,costs).status==RNS8_INVALID_ARGUMENT);
}
TEST_CASE("runtime recipe cost estimates change work selection without changing exact output") {
  using namespace rns8::detail;
  auto* ctx=create_cpu_context();
  auto md=exact_matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED);md.max_prefix=20;
  rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
  for(auto** matrix:{&a,&b,&c}) REQUIRE(rns8_create_matrix(ctx,&md,matrix)==RNS8_SUCCESS);
  int64_t two=2,three=3;
  REQUIRE(rns8_pack_i64(ctx,a,&two,1,1)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx,b,&three,1,1)==RNS8_SUCCESS);
  const std::vector<ProductNode> nodes{{0,1},{2,1},{2,1},{3,4}};
  ProductCostProfile costs;costs.target="cpu";costs.provenance="synthetic-runtime-test";
  costs.nodes.resize(4);costs.graph_key=product_graph_key({{1,1},{1,1}},nodes,20);
  for(auto& node:costs.nodes) for(uint32_t p=1;p<=20;++p) {
    node.gemm[p]=100*p;
    for(uint32_t q=p+1;q<=20;++q) node.lift[p*21+q]=1;
  }
  std::unique_ptr<RuntimeProductRecipe> recipe;
  REQUIRE(RuntimeProductRecipe::create(ctx,{a,b},nodes,UINT64_MAX,recipe,false,&costs)==RNS8_SUCCESS);
  CHECK_FALSE(recipe->cost_selection().exhaustive);
  CHECK(recipe->cost_selection().evaluated<=costs.max_evaluations);
  CHECK(recipe->compute_prefixes()==std::vector<uint32_t>{1,1,1,2});
  CHECK(recipe->allocation().arena_bytes+recipe->lift_scratch_bytes()==recipe->cost_selection().peak_bytes);
  REQUIRE(recipe->execute(c)==RNS8_SUCCESS);
  for(uint32_t p=0;p<20;++p) {
    int expected=324%rns8_default_modulus(p);
    if(expected>=(rns8_default_modulus(p)+1)/2) expected-=rns8_default_modulus(p);
    CHECK(c->residues[p]==expected);
  }
  recipe.reset();costs.target="wrong-device";
  CHECK(RuntimeProductRecipe::create(ctx,{a,b},nodes,UINT64_MAX,recipe,false,&costs)==RNS8_INVALID_ARGUMENT);
  CHECK_FALSE(recipe);
  for(auto* matrix:{a,b,c}) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

TEST_CASE("chain prefix DP agrees with exhaustive costs and reaches a long-chain optimum") {
  using namespace rns8::detail;
  const std::vector<ProductShape> inputs{{1,1},{1,1}};
  const std::vector<ProductNode> short_chain{{0,1},{2,1},{1,3},{4,4}};
  ProductCostProfile costs;costs.target="cpu";costs.provenance="synthetic-chain-dp";
  costs.nodes.resize(4);costs.graph_key=product_graph_key(inputs,short_chain,4);
  for(unsigned seed=0;seed<8;++seed) {
    for(unsigned i=0;i<4;++i) for(uint32_t p=1;p<=4;++p) {
      costs.nodes[i].gemm[p]=((i+1)*11+p*7+seed*3)%37;
      for(uint32_t q=p+1;q<=4;++q) costs.nodes[i].lift[p*21+q]=
          (i+p+q+seed)%5==0?UINT64_MAX:((i+1)*5+p*3+q*11+seed)%19;
    }
    const auto result=select_product_prefixes(inputs,short_chain,4,{1,1,1,1},UINT64_MAX,false,costs);
    REQUIRE(result.status==RNS8_SUCCESS);REQUIRE(result.exhaustive);REQUIRE(result.chain_dynamic_programming);
    REQUIRE(result.cost_optimal);REQUIRE(result.estimated_ns==result.chain_cost_lower_bound);
  }
  std::vector<ProductNode> chain{{0,1}};
  for(uint32_t i=1;i<64;++i) chain.push_back({i+1,1});
  costs.nodes.assign(64,ProductNodeCosts{});costs.max_evaluations=2;costs.graph_key=product_graph_key(inputs,chain,20);
  for(size_t i=0;i<64;++i) for(uint32_t p=3;p<=20;++p) {
    costs.nodes[i].gemm[p]=p==3?10:100;
    for(uint32_t q=p+1;q<=20;++q) costs.nodes[i].lift[p*21+q]=i==63?10000:1;
  }
  costs.nodes.back().gemm[20]=5;
  const auto result=select_product_prefixes(inputs,chain,20,std::vector<uint32_t>(64,3),UINT64_MAX,true,costs);
  REQUIRE(result.status==RNS8_SUCCESS);REQUIRE(result.evaluated==2);REQUIRE_FALSE(result.exhaustive);
  REQUIRE(result.chain_dynamic_programming);REQUIRE(result.cost_optimal);REQUIRE(result.estimated_ns==636);
  for(size_t i=0;i<63;++i) REQUIRE(result.prefixes[i]==3);REQUIRE(result.prefixes.back()==20);
  const auto rejected=select_product_prefixes(inputs,chain,20,std::vector<uint32_t>(64,3),0,true,costs);
  REQUIRE(rejected.status!=RNS8_SUCCESS);REQUIRE_FALSE(rejected.cost_optimal);
  // An older producer reused outside the immediate chain is a real DAG and
  // must retain the general search, not acquire an invalid chain cost proof.
  chain.back()={2,64};costs.graph_key=product_graph_key(inputs,chain,20);
  const auto dag=select_product_prefixes(inputs,chain,20,std::vector<uint32_t>(64,3),UINT64_MAX,false,costs);
  REQUIRE_FALSE(dag.chain_dynamic_programming);
}

#include "../support/product_recipe_outputs_fixture.hpp"
TEST_CASE("public CPU product recipes publish multiple retained intermediate values") {
  check_product_recipe_outputs(RNS8_BACKEND_CPU_REFERENCE);
  check_product_recipe_outputs(RNS8_BACKEND_CPU_REFERENCE,true);
}
TEST_CASE("observable product values stay live through publication") {
  using namespace rns8::detail;
  const std::vector<ProductShape> inputs{{16,16},{16,16}};
  const std::vector<ProductNode> chain{{0,1},{2,1},{3,1},{4,1}};
  const std::vector<uint32_t> observed{2,3,5};
  const auto plan=plan_product_arena(inputs,chain,20,UINT64_MAX,{},observed);
  REQUIRE(plan.status==RNS8_SUCCESS);
  for(auto id:observed) CHECK(plan.values[id].last_use==chain.size());
  for(size_t i=0;i<chain.size();++i) for(size_t j=i+1;j<chain.size();++j) {
    const auto& a=plan.values[2+i];const auto& b=plan.values[2+j];
    if(a.last_use>=j) CHECK((a.offset+a.bytes<=b.offset || b.offset+b.bytes<=a.offset));
  }
  CHECK(plan_product_arena(inputs,chain,20,plan.arena_bytes-1,{},observed).status==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(plan_product_arena(inputs,chain,20,UINT64_MAX,{}, {2,2}).status==RNS8_INVALID_ARGUMENT);
  CHECK(plan_product_arena(inputs,chain,20,UINT64_MAX,{}, {0}).status==RNS8_INVALID_ARGUMENT);
  CHECK(plan_product_arena(inputs,chain,20,UINT64_MAX,{}, {6}).status==RNS8_INVALID_ARGUMENT);
}

TEST_CASE("multi-output cost selection charges retained lifts and binds observation order") {
  using namespace rns8::detail;
  const std::vector<ProductShape> inputs{{16,16},{16,16}};
  const std::vector<ProductNode> nodes{{0,1},{2,1},{3,1}};
  const std::vector<uint32_t> outputs{4,2};
  ProductCostProfile costs;costs.target="cpu";costs.provenance="synthetic-observation-test";
  costs.graph_key=product_graph_key(inputs,nodes,3,outputs);costs.nodes.resize(3);
  for(auto& node:costs.nodes) for(uint32_t p=1;p<=3;++p) {
    node.gemm[p]=100*p;
    for(uint32_t q=p+1;q<=3;++q) node.lift[p*21+q]=1;
  }
  const auto selected=select_product_prefixes(inputs,nodes,3,{1,1,1},UINT64_MAX,false,costs,outputs);
  REQUIRE(selected.status==RNS8_SUCCESS);CHECK(selected.exhaustive);CHECK(selected.cost_optimal);
  CHECK_FALSE(selected.chain_dynamic_programming);
  CHECK(selected.estimated_ns==302);
  CHECK(selected.prefixes==std::vector<uint32_t>{1,1,1});
  CHECK(selected.capacities==std::vector<uint32_t>{3,1,3});
  const auto arena=plan_product_arena(inputs,nodes,3,UINT64_MAX,selected.capacities,outputs);
  CHECK(selected.peak_bytes==arena.arena_bytes);
  CHECK(select_product_prefixes(inputs,nodes,3,{1,1,1},selected.peak_bytes-1,false,costs,outputs).status!=RNS8_SUCCESS);
  for(const auto& changed:std::vector<std::vector<uint32_t>>{{2,4},{4},{}})
    CHECK(select_product_prefixes(inputs,nodes,3,{1,1,1},UINT64_MAX,false,costs,changed).status==RNS8_INVALID_ARGUMENT);
  costs.graph_key=product_graph_key(inputs,nodes,3,{2,2});
  CHECK(select_product_prefixes(inputs,nodes,3,{1,1,1},UINT64_MAX,false,costs,{2,2}).status==RNS8_INVALID_ARGUMENT);
}
