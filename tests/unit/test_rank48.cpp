#include <catch2/catch_test_macros.hpp>
#include "backend_common/rank48_program.hpp"
#include "backend_common/bilinear_selection.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
TEST_CASE("rank48 scaled dyadic coefficients preserve every ordered tensor coefficient") {
  const auto program=rns8::detail::rank48::program();
  for(unsigned c=0;c<16;++c)for(unsigned a=0;a<16;++a)for(unsigned b=0;b<16;++b) {
    int observed=0;for(const auto& term:program.products)observed+=int(term.a[a])*term.b[b]*term.c[c];
    CHECK(observed==8*int(a/4==c/4 && b%4==c%4 && a%4==b/4));
  }
  for(uint32_t method=0;method<12;++method){const auto selected=rns8::detail::bilinear_selection(method);REQUIRE(selected.valid);
    if(method>=10){CHECK(selected.blocks==4);CHECK_FALSE(selected.slp);CHECK(selected.matrix==(method==11));}}
  CHECK_FALSE(rns8::detail::bilinear_selection(UINT32_MAX).valid);
  CHECK(rns8::detail::query_bilinear_method_resources(10,32768,32768,32768,32768,32768,32768,20,UINT64_MAX).status==RNS8_SUCCESS);
}
TEST_CASE("rank48 evaluates odd composite and prime rings against independent integer products") {
  const auto program=rns8::detail::rank48::program();using boost::multiprecision::cpp_int;
  for(int modulus:{3,5,9,15,127,167,247,251,253,255})for(int seed:{0,1,17,255}) {
    std::array<int,16> a{},b{},out{};
    for(unsigned i=0;i<16;++i){a[i]=int((i*97+seed)%256)-128;b[i]=int((i*53+seed*3)%256)-128;}
    for(const auto& term:program.products){int left=0,right=0;
      for(unsigned i=0;i<16;++i){left+=rns8::detail::rank48::coefficient(term.a[i],modulus)*a[i];right+=rns8::detail::rank48::coefficient(term.b[i],modulus)*b[i];}
      const int product=(left%modulus)*(right%modulus)%modulus;
      for(unsigned c=0;c<16;++c)out[c]=(out[c]+rns8::detail::rank48::coefficient(term.c[c],modulus)*product)%modulus;
    }
    for(unsigned r=0;r<4;++r)for(unsigned c=0;c<4;++c){cpp_int expected=0;
      for(unsigned q=0;q<4;++q)expected+=cpp_int(a[r*4+q])*b[q*4+c];
      expected%=modulus;if(expected<0)expected+=modulus;
      CHECK((out[r*4+c]+modulus)%modulus==expected.convert_to<int>());
    }
  }
}
