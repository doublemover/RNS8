#include "backend_common/word_mod_plan.hpp"
TEST_CASE("word modular arithmetic and decompositions preserve arbitrary rings") {
  using namespace rns8::detail::word_mod;
  using boost::multiprecision::cpp_int;
  for(uint64_t modulus:{UINT64_C(2),UINT64_C(251),UINT64_C(256),UINT64_C(65521),
      UINT64_C(65536),UINT64_C(2147483647),UINT64_C(4503599627370496),UINT64_MAX-58,UINT64_MAX}) {
    const std::array<uint64_t,6> values={0,1,modulus/2,modulus-1,UINT64_C(0x8080808080808080)%modulus,UINT64_C(0xdeadbeef12345678)%modulus};
    for(auto a:values) for(auto b:values) {
      const auto expected=(cpp_int(a)*b%modulus).convert_to<uint64_t>();
      REQUIRE(multiply(a,b,modulus)==expected);
      REQUIRE(add(a,b,modulus)==((cpp_int(a)+b)%modulus).convert_to<uint64_t>());
      for(const auto& d:candidates(modulus,65537,53)) {
        REQUIRE(decomposed_product(a,b,modulus,d.left_digits,d.right_digits)==expected);
        REQUIRE(d.exact_bound==(cpp_int(d.lambda)*d.alpha*d.beta+modulus-1 <= (cpp_int(1)<<53)));
      }
    }
    for(unsigned degree=1;degree<=4;++degree) {
      const auto root=ceil_root(modulus,degree);cpp_int below=1,at=1;
      for(unsigned i=0;i<degree;++i) {below*=root-1;at*=root;}
      REQUIRE(below<modulus);REQUIRE(at>=modulus);
    }
  }
  REQUIRE(prime(UINT64_C(18446744073709551557)));
  for(uint64_t composite:{UINT64_C(0),UINT64_C(1),UINT64_C(341),UINT64_C(561),UINT64_C(3215031751),UINT64_MAX}) REQUIRE_FALSE(prime(composite));
  REQUIRE_FALSE(decomposition(251,0,1,1,53).exact_bound);
  REQUIRE_FALSE(decomposition(251,1,1,0,53).exact_bound);
}
TEST_CASE("word modular CPU GEMM validates canonical inputs before publishing") {
  using namespace rns8::detail::word_mod;
  Problem p; p.modulus=UINT64_MAX;p.m=2;p.n=2;p.k=3;p.lda=4;p.ldb=3;p.ldc=3;
  const std::vector<uint64_t> a={0,UINT64_MAX-1,2,99,4,5,6,99};
  std::vector<uint64_t> b={7,8,99,9,10,99,11,12,99},c(6,77);
  REQUIRE(gemm_cpu(a.data(),b.data(),c.data(),p)==RNS8_SUCCESS);
  for(int i=0;i<2;++i) for(int j=0;j<2;++j) {
    boost::multiprecision::cpp_int sum=0;for(int h=0;h<3;++h) sum+=boost::multiprecision::cpp_int(a[i*4+h])*b[h*3+j];
    REQUIRE(c[i*3+j]==(sum%p.modulus).convert_to<uint64_t>());
  }
  REQUIRE(c[2]==77);REQUIRE(c[5]==77);
  const auto before=c;b[0]=UINT64_MAX;
  REQUIRE(gemm_cpu(a.data(),b.data(),c.data(),p)==RNS8_INVALID_ARGUMENT);REQUIRE(c==before);
  REQUIRE(gemm_cpu(c.data(),b.data(),c.data(),p)==RNS8_INVALID_ARGUMENT);
  p.domain=Domain::prime_field;REQUIRE(query(p).status==RNS8_INVALID_ARGUMENT);
  p.modulus=251;REQUIRE(query(p).status==RNS8_SUCCESS);
  p.m=INT64_MAX;REQUIRE(query(p).status==RNS8_INVALID_ARGUMENT);
}
