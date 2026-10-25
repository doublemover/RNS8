#include "backend_common/rank23_slp.hpp"
#include "backend_common/rank23_slp_resources.hpp"
#include "backend_common/rank23_reference.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <random>

TEST_CASE("rank23 actual certificate SLP handles overwrites CSE and exact lifetime reuse") {
  using namespace rns8::detail::rank23::slp;
  REQUIRE(matches_coefficients());
  CHECK(u.source_additions + v.source_additions + w.source_additions == 56);
  RawStage duplicate{2,2,3,{{2,0,1,false},{3,1,0,false},{2,2,1,true}},{2,3}};
  const auto plan=compile(duplicate);
  REQUIRE(plan.valid);
  CHECK(plan.source_additions==3);
  CHECK(plan.count==2);
  const int8_t input[]{11,7}; int8_t output[2]{};
  evaluate(plan,input,output,251);
  CHECK(output[0]==11); CHECK(output[1]==18);
  CHECK(plan.results[0]!=plan.results[1]);
  RawStage chain{2,1,50,{}, {2}};
  chain.assignments[0]={2,0,1,false};
  for(int i=1;i<50;++i) chain.assignments[i]={2,2,1,true};
  const auto reuse=compile(chain);
  REQUIRE(reuse.valid);
  CHECK(reuse.slots==2);
  evaluate(reuse,input,output,251);
  CHECK(output[0]==center(11+7-49*7,251));
  for (auto invalid : {-2,64}) {
    auto bad=raw_u(); bad.assignments[0].left=invalid;
    CHECK_FALSE(compile(bad).valid);
  }
  auto bad=raw_u(); bad.assignments[0].left=50;
  CHECK_FALSE(compile(bad).valid);
  bad=raw_u(); bad.results[0]=63;
  CHECK_FALSE(compile(bad).valid);
}

TEST_CASE("rank23 pinned SLP matches independent Boost in every byte ring and aliases safely") {
  using boost::multiprecision::cpp_int;
  using namespace rns8::detail::rank23;
  std::mt19937 random(234851);
  for(uint32_t modulus=2;modulus<=256;++modulus) for(int sample=0;sample<24;++sample) {
    int8_t a[9]{},b[9]{},actual[9]{},coefficient[9]{};
    for(int i=0;i<9;++i) {
      a[i]=static_cast<int8_t>(sample==0?-128:sample==1?127:int(random()%256)-128);
      b[i]=static_cast<int8_t>(sample<2?(i%2?-128:127):int(random()%256)-128);
    }
    REQUIRE(slp::multiply3x3(a,b,actual,modulus));
    REQUIRE(multiply3x3_modular(a,b,coefficient,modulus));
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) {
      cpp_int expected=0;
      for(int q=0;q<3;++q) expected+=cpp_int(a[r*3+q])*b[q*3+c];
      expected%=modulus;
      if(expected<0) expected+=modulus;
      if(expected>=(modulus+1)/2) expected-=modulus;
      REQUIRE(cpp_int(actual[r*3+c])==expected);
      REQUIRE(actual[r*3+c]==coefficient[r*3+c]);
    }
    int8_t alias_a[9]{},alias_b[9]{};
    std::copy_n(a,9,alias_a); std::copy_n(b,9,alias_b);
    REQUIRE(slp::multiply3x3(alias_a,b,alias_a,modulus));
    REQUIRE(slp::multiply3x3(a,alias_b,alias_b,modulus));
    CHECK(std::equal(actual,actual+9,alias_a));
    CHECK(std::equal(actual,actual+9,alias_b));
  }
  int8_t input[9]{},output[9]{1,2,3,4,5,6,7,8,9};
  for(uint32_t bad : {0u,1u,257u,UINT32_MAX}) {
    REQUIRE_FALSE(slp::multiply3x3(input,input,output,bad));
    for(int i=0;i<9;++i) CHECK(output[i]==i+1);
  }
  CHECK_FALSE(slp::multiply3x3(nullptr,input,output,251));
  CHECK_FALSE(slp::multiply3x3(input,nullptr,output,251));
  CHECK_FALSE(slp::multiply3x3(input,input,nullptr,251));
}

TEST_CASE("rank23 SLP resource query accounts every retained operand and product") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for(int64_t size : {int64_t(1),int64_t(17),int64_t(16384)}) for(uint32_t prefix : {1u,9u,20u}) {
    const auto plan=query_rank23_slp_resources(size,size+1,size+2,size+3,size+4,size+5,prefix,UINT64_MAX);
    REQUIRE(plan.status==RNS8_SUCCESS);
    CHECK(cpp_int(plan.total_bytes)==23*(cpp_int(plan.a_bytes)+plan.b_bytes+plan.product_bytes)+plan.output_bytes);
    CHECK(plan.b_offset==23*plan.a_bytes);
    CHECK(plan.product_offset==23*(plan.a_bytes+plan.b_bytes));
    CHECK(plan.output_offset==23*(plan.a_bytes+plan.b_bytes+plan.product_bytes));
    CHECK(plan.output_offset+plan.output_bytes==plan.total_bytes);
    CHECK(query_rank23_slp_resources(size,size+1,size+2,size+3,size+4,size+5,prefix,plan.total_bytes).status==RNS8_SUCCESS);
    CHECK(query_rank23_slp_resources(size,size+1,size+2,size+3,size+4,size+5,prefix,plan.total_bytes-1).status==RNS8_WORKSPACE_TOO_SMALL);
  }
  CHECK(query_rank23_slp_resources(1,1,1,1,1,1,0,UINT64_MAX).status==RNS8_INVALID_ARGUMENT);
  CHECK(query_rank23_slp_resources(INT64_MAX,1,1,1,1,1,20,UINT64_MAX).status==RNS8_UNSUPPORTED_BACKEND);
  CHECK(query_rank23_slp_resources(32768,32768,32768,32768,32768,32768,20,UINT64_MAX).status==RNS8_UNSUPPORTED_BACKEND);
}

#include "backend_common/rank23_october55.hpp"
TEST_CASE("October55 executes its own signed DAG in every byte ring without int8 negation overflow") {
  using namespace rns8::detail::rank23;
  REQUIRE(october55::exact_ordered_tensor());
  for(int modulus=2;modulus<=256;++modulus) {
    for(int left=-(modulus/2);left<=(modulus-1)/2;++left)
      for(int right=-(modulus/2);right<=(modulus-1)/2;++right)
        for(int ls:{-1,1})for(int rs:{-1,1})
          REQUIRE(slp::center_binary(ls*left+rs*right,modulus)==slp::center(ls*left+rs*right,modulus));
    for(int sample=0;sample<8;++sample) {
      int8_t a[9]{},b[9]{},u[23]{},v[23]{},products[23]{},out[9]{};
      for(int i=0;i<9;++i){a[i]=int8_t(sample?((i*71+sample*23)%256)-128:-128);b[i]=int8_t(sample?((i*19+sample*47)%256)-128:127);}
      slp::evaluate(october55::u,a,u,modulus);slp::evaluate(october55::v,b,v,modulus);
      for(int p=0;p<23;++p)products[p]=int8_t(slp::center(int(u[p])*v[p],modulus));
      slp::evaluate(october55::w,products,out,modulus);
      for(int r=0;r<3;++r)for(int c=0;c<3;++c){int expected=0;for(int q=0;q<3;++q)expected+=int(a[r*3+q])*b[q*3+c];REQUIRE(out[r*3+c]==slp::center(expected,modulus));}
    }
  }
}
TEST_CASE("October55 recomputation removes only retained input buffers and respects exact budgets") {
  using namespace rns8::detail;
  for(uint32_t prefix:{1u,3u,9u,20u}) {
    const auto retained=query_rank23_slp_resources(17,19,31,34,22,24,prefix,UINT64_MAX);
    const auto recomputed=query_rank23_slp_resources(17,19,31,34,22,24,prefix,UINT64_MAX,true);
    REQUIRE(retained.status==RNS8_SUCCESS);REQUIRE(recomputed.status==RNS8_SUCCESS);
    REQUIRE(retained.total_bytes-recomputed.total_bytes==22*(retained.a_bytes+retained.b_bytes));
    REQUIRE(recomputed.product_offset==recomputed.a_bytes+recomputed.b_bytes);
    REQUIRE(recomputed.output_offset==recomputed.product_offset+23*recomputed.product_bytes);
    REQUIRE(query_rank23_slp_resources(17,19,31,34,22,24,prefix,recomputed.total_bytes,true).status==RNS8_SUCCESS);
    REQUIRE(query_rank23_slp_resources(17,19,31,34,22,24,prefix,recomputed.total_bytes-1,true).status==RNS8_WORKSPACE_TOO_SMALL);
  }
}
