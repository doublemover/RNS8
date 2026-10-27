#include <catch2/catch_test_macros.hpp>
#include "backend_common/rank48_program.hpp"
#include "backend_common/rank48_slp.hpp"
#include "backend_common/bilinear_selection.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
TEST_CASE("rank48 scaled dyadic coefficients preserve every ordered tensor coefficient") {
  const auto program=rns8::detail::rank48::program();
  for(unsigned c=0;c<16;++c)for(unsigned a=0;a<16;++a)for(unsigned b=0;b<16;++b) {
    int observed=0;for(const auto& term:program.products)observed+=int(term.a[a])*term.b[b]*term.c[c];
    CHECK(observed==8*int(a/4==c/4 && b%4==c%4 && a%4==b/4));
  }
  for(uint32_t method=0;method<16;++method){const auto selected=rns8::detail::bilinear_selection(method);REQUIRE(selected.valid);
    if(method>=10){CHECK(selected.blocks==4);CHECK(selected.slp==(method>=12));CHECK(selected.matrix==bool(method&1));}}
  CHECK_FALSE(rns8::detail::bilinear_selection(UINT32_MAX).valid);
  CHECK(rns8::detail::query_bilinear_method_resources(10,32768,32768,32768,32768,32768,32768,20,UINT64_MAX).status==RNS8_SUCCESS);
}
TEST_CASE("rank48 canonical SLP live registers match independent odd ring products") {
  using namespace rns8::detail::rank48::slp;
  for(int q=3;q<=255;q+=2)for(int seed:{0,1,127,255}) {
    int8_t a[capacity]{},b[capacity]{},c[capacity]{};int original_a[16]{},original_b[16]{};
    for(int i=0;i<16;++i){original_a[i]=int((i*97+seed)%256)-128;original_b[i]=int((i*53+seed*3)%256)-128;
      a[i]=int8_t(center(original_a[i],q));b[i]=int8_t(center(original_b[i],q));}
    evaluate(u,a,q);evaluate(v,b,q);
    for(int i=0;i<48;++i)c[i]=int8_t(center(int(a[u.results[i]])*b[v.results[i]],q));
    evaluate(w,c,q);
    for(int row=0;row<4;++row)for(int col=0;col<4;++col){int exact=0;
      for(int k=0;k<4;++k)exact+=original_a[row*4+k]*original_b[k*4+col];
      CHECK(int(c[w.results[row*4+col]])==center(exact,q));}
  }
  for(unsigned method:{12u,13u,14u,15u}) {
    const auto r=rns8::detail::query_bilinear_method_resources(method,17,19,33,34,21,22,20,UINT64_MAX);
    REQUIRE(r.status==RNS8_SUCCESS);
    CHECK(rns8::detail::query_bilinear_method_resources(method,17,19,33,34,21,22,20,r.total_bytes-1).status==RNS8_WORKSPACE_TOO_SMALL);
  }
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
