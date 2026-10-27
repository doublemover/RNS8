#include <catch2/catch_test_macros.hpp>
#include "backend_common/fp8_modular.hpp"
TEST_CASE("FNUZ balanced digits preserve every signed byte and finite ring", "[fp8][cpu]") {
  namespace f=rns8::detail::fp8_modular;
  const uint8_t positive[]={0,0x40,0x48,0x4c,0x50,0x52,0x54,0x56,0x58};
  for(int value=-8;value<=8;++value) {
    CHECK(f::encode(value)==uint8_t(positive[value<0?-value:value]|(value<0?128:0)));
    CHECK(f::decode(f::encode(value))==value);
  }
  for(int modulus=2;modulus<=256;++modulus) {
    bool equal=true;
    for(int a=-128;a<=127;++a)for(int b=-128;b<=127;++b) {
      const int actual=f::low(a)*f::low(b)+16*f::low(a)*f::high(b)+16*f::high(a)*f::low(b)+256*f::high(a)*f::high(b);
      equal&=actual==a*b && (actual-a*b)%modulus==0;
    }
    CHECK(equal);
  }
  CHECK(f::algebra());CHECK(uint64_t(f::safe_k)*64<(uint64_t{1}<<24));
  CHECK(f::identity("gfx942","a")!=f::identity("gfx942","b"));
}
TEST_CASE("three FP8 products and RDNA4 fragments preserve exact byte arithmetic", "[fp8][cpu]") {
  namespace f=rns8::detail::fp8_modular;
  CHECK(f::Rdna4Layout::bijective());CHECK(f::three_algebra());
  for(bool ocp:{false,true})for(int x=-128;x<=127;++x)for(unsigned term=0;term<3;++term){
    const int digit=f::component(x,term,true);CHECK(f::decode(f::encode(digit,ocp),ocp)==digit);
  }
  for(int x=-128;x<=127;++x)for(int y=-128;y<=127;++y){
    int sum=0;for(unsigned term=0;term<3;++term)sum+=f::component(x,term,true)*f::component(y,term,true)*f::weight(term,true);
    REQUIRE(sum==x*y);
  }
  CHECK(f::encode(1,true)==0x38);CHECK(f::encode(15,true)==0x57);
  CHECK(f::identity("gfx1201","a",2,2)!=f::identity("gfx942","a",1,2));
  CHECK(f::identity("gfx942","a",1,0)!=f::identity("gfx942","a",1,2));
  int result=123;CHECK(f::integer_partial(14745600.0f,result,f::partial_bound(1,true)));
  CHECK_FALSE(f::integer_partial(14745601.0f,result,f::partial_bound(1,true)));
}
#include <limits>
TEST_CASE("FP8 publication guard rejects nonfinite fractional and out-of-model partials") {
  int result=123;
  for(float value:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),0.5f,-0.5f,4194305.0f,-4194305.0f}) {
    CHECK_FALSE(rns8::detail::fp8_modular::integer_partial(value,result));CHECK(result==123);
  }
  for(float value:{0.0f,-0.0f,4194304.0f,-4194304.0f,17.0f}) {
    REQUIRE(rns8::detail::fp8_modular::integer_partial(value,result));CHECK(result==int(value));
  }
}
