#include "core/runtime_product_recipe.hpp"
#include "../support/product_materialization_fixture.hpp"
namespace {
struct OrderedRecipeFixture {
  rns8_context* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  std::array<rns8_matrix*,4> matrices{};
  rns8_product_recipe* recipe=nullptr;
  OrderedRecipeFixture(uint32_t prefix,rns8_semantics semantics,rns8_product_prefix_policy policy,uint64_t budget=UINT64_MAX) {
    auto md=matrix_desc(1,1,semantics,RNS8_BOUND_NONE);md.max_prefix=prefix;
    for(auto& matrix:matrices)REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
    const std::array<int64_t,4> values{2,3,7,11};
    for(size_t i=0;i<values.size();++i) {
      const uint64_t word=uint64_t(values[i]);
      REQUIRE((semantics==RNS8_EXACT_WIDE_SIGNED?rns8_pack_i64(ctx,matrices[i],&values[i],1,0):
          rns8_pack_u64(ctx,matrices[i],&word,1,0))==RNS8_SUCCESS);
    }
    const rns8_matrix* inputs[]{matrices[0],matrices[1]};
    const rns8_product_node nodes[]{{0,1},{2,1},{3,1}};const uint32_t outputs[]{2,4};
    rns8_product_recipe_desc desc{};desc.struct_size=sizeof(desc);desc.abi_version=RNS8_ABI_VERSION;
    desc.inputs=inputs;desc.input_count=2;desc.nodes=nodes;desc.node_count=3;
    desc.prefix_policy=policy;desc.max_arena_and_lift_bytes=budget;
    REQUIRE(rns8_create_product_recipe_outputs(ctx,&desc,outputs,2,&recipe)==RNS8_SUCCESS);
  }
  rns8_status execute() {rns8_matrix* outputs[]{matrices[2],matrices[3]};return rns8_execute_product_recipe_outputs(recipe,outputs,2);}
  rns8_product_recipe_execution_info info() const {
    rns8_product_recipe_execution_info out{};out.struct_size=sizeof(out);out.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_get_product_recipe_execution_info(recipe,&out)==RNS8_SUCCESS);return out;
  }
  std::vector<int8_t> bytes(size_t i) const {
    std::vector<int8_t> result(matrices[i]->hip_residue_bytes);
    REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0,result.data(),matrices[i]->hip_residues,result.size())==RNS8_SUCCESS);
    return result;
  }
  void close() {
    ctx->synchronous_test_hooks=nullptr;
    REQUIRE(rns8_destroy_product_recipe(recipe)==RNS8_SUCCESS);recipe=nullptr;
    for(auto* matrix:matrices)REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);ctx=nullptr;
  }
};
}
TEST_CASE("ordered HIP recipe retains one stream and preserves all publications after a failed lift") {
  using namespace rns8::detail;
  if(!hip_available())SKIP("no HIP device for ordered product recipe");
  for(auto prefix:{1u,9u,20u})for(auto semantics:{RNS8_EXACT_WIDE_SIGNED,RNS8_EXACT_WIDE_UNSIGNED})
    for(auto policy:{RNS8_PRODUCT_FULL_PREFIX,RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX}) {
      CAPTURE(prefix,semantics,policy);OrderedRecipeFixture fixture(prefix,semantics,policy);
      const bool guarded=prefix>1 && policy==RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX;
      if(guarded) {
        rns8_product_recipe_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
        REQUIRE(rns8_get_product_recipe_resources(fixture.recipe,&resources)==RNS8_SUCCESS);
        OrderedRecipeFixture tight(prefix,semantics,policy,resources.arena_bytes+resources.lift_scratch_bytes);
        const auto before=hip_direct_allocation_counters_snapshot();
        CHECK(rns8_set_product_recipe_execution_policy(tight.recipe,RNS8_PRODUCT_EXECUTION_ORDERED_HIP)==RNS8_WORKSPACE_TOO_SMALL);
        CHECK(before.allocate_calls==hip_direct_allocation_counters_snapshot().allocate_calls);
        CHECK(tight.info().policy==RNS8_PRODUCT_EXECUTION_SYNCHRONOUS_WRAPPERS);CHECK(tight.info().stream_prepared==0);
        tight.close();
      }
      REQUIRE(rns8_set_product_recipe_execution_policy(fixture.recipe,RNS8_PRODUCT_EXECUTION_ORDERED_HIP)==RNS8_SUCCESS);
      CHECK(rns8_set_product_recipe_execution_policy(fixture.recipe,static_cast<rns8_product_execution_policy>(UINT32_MAX))==RNS8_INVALID_ARGUMENT);
      CHECK(fixture.info().policy==RNS8_PRODUCT_EXECUTION_ORDERED_HIP);
      CHECK(fixture.info().stream_prepared==1);
      CHECK(fixture.info().retained_control_device_bytes==(guarded?sizeof(int):0));
      CHECK(fixture.info().retained_control_pinned_bytes==(guarded?sizeof(int):0));
      for(unsigned repeat=0;repeat<2;++repeat) {
        const auto before=hip_direct_allocation_counters_snapshot();REQUIRE(fixture.execute()==RNS8_SUCCESS);
        const auto after=hip_direct_allocation_counters_snapshot();
        CHECK(before.allocate_calls==after.allocate_calls);CHECK(before.free_calls==after.free_calls);
        const auto info=fixture.info();CHECK(info.stream_wait_calls==1);CHECK(info.ordered_gemm_calls==3);
        CHECK(info.ordered_copy_calls==2);CHECK(info.ordered_status_readback_bytes==(guarded?sizeof(int):0));
        for(size_t i=2;i<4;++i)for(uint32_t p=0;p<prefix;++p)
          CHECK(fixture.bytes(i)[p]==centered_residue(boost::multiprecision::cpp_int(i==2?6:54),kDefaultModuli[p]));
      }
      if(guarded) {
        const auto first=fixture.bytes(2),last=fixture.bytes(3),original=fixture.bytes(0);
        const auto first_gen=fixture.matrices[2]->content_generation,last_gen=fixture.matrices[3]->content_generation;
        const auto first_bound=fixture.matrices[2]->exact_max_magnitude,last_bound=fixture.matrices[3]->exact_max_magnitude;
        // Deliberately dishonest private storage exercises the device proof guard.
        // Public pack APIs always update the value bound and generation together.
        auto corrupted=original;for(uint32_t p=0;p<prefix;++p)corrupted[p]=centered_residue(boost::multiprecision::cpp_int(7),kDefaultModuli[p]);
        REQUIRE(hip_direct_copy_host_to_device(0,fixture.matrices[0]->hip_residues,corrupted.data(),corrupted.size())==RNS8_SUCCESS);
        REQUIRE(fixture.execute()==RNS8_RANGE_ERROR);
        CHECK(fixture.bytes(2)==first);CHECK(fixture.bytes(3)==last);
        CHECK(fixture.matrices[2]->content_generation==first_gen);CHECK(fixture.matrices[3]->content_generation==last_gen);
        CHECK(fixture.matrices[2]->exact_max_magnitude==first_bound);CHECK(fixture.matrices[3]->exact_max_magnitude==last_bound);
        CHECK_FALSE(fixture.ctx->hip_failure_poisoned);
        for(auto* matrix:fixture.matrices){CHECK(matrix->in_flight_readers==0);CHECK_FALSE(matrix->in_flight_writer);}
        REQUIRE(hip_direct_copy_host_to_device(0,fixture.matrices[0]->hip_residues,original.data(),original.size())==RNS8_SUCCESS);
        REQUIRE(fixture.execute()==RNS8_SUCCESS); // resets the sticky flag for the next replay
      }
      fixture.close();
    }
}
TEST_CASE("ordered HIP recipe failures retain resources until its own stream drains") {
  if(!hip_available())SKIP("no HIP device for ordered recipe lifetime checks");
  for(const char* stage:{"product_recipe_prepare","product_recipe_compute","product_recipe_publication"})
    for(bool throws:{false,true})for(bool fail_drain:{false,true}) {
      CAPTURE(stage,throws,fail_drain);OrderedRecipeFixture fixture(20,RNS8_EXACT_WIDE_SIGNED,RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX);
      REQUIRE(rns8_set_product_recipe_execution_policy(fixture.recipe,RNS8_PRODUCT_EXECUTION_ORDERED_HIP)==RNS8_SUCCESS);
      const auto old_first=fixture.bytes(2),old_last=fixture.bytes(3);
      const auto generation=fixture.matrices[2]->content_generation;
      struct Failure {
        OrderedRecipeFixture* fixture;const char* stage;bool throws,fail_drain;
        bool reservations=false;rns8_status busy=RNS8_INTERNAL_ERROR;unsigned drains=0;
        static rns8_status dispatch(void* opaque,const char* name,bool*) {
          auto& self=*static_cast<Failure*>(opaque);if(std::string(name)!=self.stage)return RNS8_SUCCESS;
          const auto& m=self.fixture->matrices;
          self.reservations=m[0]->in_flight_readers && m[1]->in_flight_readers && m[2]->in_flight_writer && m[3]->in_flight_writer;
          self.busy=rns8_destroy_product_recipe(self.fixture->recipe);
          if(self.throws)throw std::runtime_error("injected recipe checkpoint");
          return RNS8_BACKEND_FAILURE;
        }
        static rns8_status drain(void* opaque) noexcept {
          auto& self=*static_cast<Failure*>(opaque);++self.drains;
          return self.fail_drain?RNS8_BACKEND_FAILURE:RNS8_SUCCESS;
        }
      } failure{&fixture,stage,throws,fail_drain};
      const rns8_synchronous_test_hooks hooks{&failure,&Failure::dispatch,&Failure::drain};fixture.ctx->synchronous_test_hooks=&hooks;
      CHECK(fixture.execute()==(throws?RNS8_INTERNAL_ERROR:RNS8_BACKEND_FAILURE));
      CHECK(failure.reservations);CHECK(failure.busy==RNS8_BUSY);
      const bool queued=std::string(stage)!="product_recipe_prepare";
      const bool quarantined=queued && fail_drain;
      CHECK(fixture.ctx->hip_failure_poisoned==quarantined);
      CHECK(fixture.matrices[2]->content_generation==generation);
      if(!quarantined && std::string(stage)!="product_recipe_publication") {
        CHECK(fixture.bytes(2)==old_first);CHECK(fixture.bytes(3)==old_last);
      }
      if(std::string(stage)=="product_recipe_publication")for(size_t i=2;i<4;++i)CHECK_FALSE(fixture.matrices[i]->device_residues_current);
      if(quarantined) {
        CHECK(fixture.matrices[0]->in_flight_readers==1);CHECK(fixture.matrices[2]->in_flight_writer);
        CHECK(rns8_destroy_matrix(fixture.matrices[0])==RNS8_BUSY);
      }
      fixture.ctx->synchronous_test_hooks=nullptr;
      if(!quarantined)REQUIRE(fixture.execute()==RNS8_SUCCESS); // rejected BUSY close did not retire the handle
      fixture.close(); // retries the real owned-stream fence before releasing any hold
    }
}
TEST_CASE("public HIP materialized product recipes preserve values reuse rebind and observed branches") {
  if(!hip_available())SKIP("no HIP device for materialized product recipes");
  for(auto execution:{RNS8_PRODUCT_EXECUTION_SYNCHRONOUS_WRAPPERS,RNS8_PRODUCT_EXECUTION_ORDERED_HIP}) {
    check_product_materializations(RNS8_BACKEND_HIP_DIRECT,false,execution);
    check_product_materializations(RNS8_BACKEND_HIP_DIRECT,true,execution);
  }
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
    const auto execution=repeat%2?RNS8_PRODUCT_EXECUTION_ORDERED_HIP:RNS8_PRODUCT_EXECUTION_SYNCHRONOUS_WRAPPERS;
    REQUIRE(recipe->set_execution_policy(execution)==RNS8_SUCCESS);
    REQUIRE(recipe->execute(c)==RNS8_SUCCESS);CHECK(recipe->execution_info().stream_wait_calls==uint64_t(repeat%2));
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
  REQUIRE(rns8_set_product_recipe_execution_policy(public_recipe,RNS8_PRODUCT_EXECUTION_ORDERED_HIP)==RNS8_SUCCESS);
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
  for(auto execution:{RNS8_PRODUCT_EXECUTION_SYNCHRONOUS_WRAPPERS,RNS8_PRODUCT_EXECUTION_ORDERED_HIP}) {
    check_product_recipe_outputs(RNS8_BACKEND_HIP_DIRECT,false,execution);
    check_product_recipe_outputs(RNS8_BACKEND_HIP_DIRECT,true,execution);
  }
}
