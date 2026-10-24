#include "core/runtime_product_recipe.hpp"
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
  for(int repeat=0;repeat<2;++repeat) {
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
  for(auto* matrix:{a,b,c}) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
