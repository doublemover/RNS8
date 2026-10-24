#include "backend_common/fixed_conversion.hpp"
TEST_CASE("fixed byte transform imports and CRT sums match independent integers") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  const uint64_t values[]={0,1,127,128,255,256,UINT64_MAX,UINT64_C(1)<<63,
      UINT64_C(0x7f8081ff007fff80),UINT64_C(0x8080808080808080)};
  for(unsigned prefix=1;prefix<=20;++prefix) {
    cpp_int product=1;for(unsigned p=0;p<prefix;++p) product*=kDefaultModuli[p];
    for(bool signed_values:{false,true}) for(auto raw:values) {
      cpp_int value=raw;if(signed_values && (raw>>63)) value-=cpp_int(1)<<64;
      uint32_t canonical[20]{};
      for(unsigned p=0;p<prefix;++p) {
        cpp_int residue=value%kDefaultModuli[p];if(residue<0) residue+=kDefaultModuli[p];
        canonical[p]=residue.convert_to<uint32_t>();
        REQUIRE(fixed_conversion::import_canonical(raw,p,signed_values)==canonical[p]);
      }
      const auto words=fixed_conversion::crt_unreduced(canonical,prefix);
      const cpp_int sum=cpp_int(words[0])+(cpp_int(words[1])<<64)+(cpp_int(words[2])<<128);
      cpp_int expected=0;
      for(unsigned p=0;p<prefix;++p) {
        const cpp_int weight=product/kDefaultModuli[p];
        const uint32_t coefficient=canonical[p]*direct_crt::kConstants.inverses[prefix][p]%kDefaultModuli[p];
        expected+=coefficient*weight;
      }
      REQUIRE(sum==expected);REQUIRE(sum<product*prefix);
      cpp_int reduced=value%product;if(reduced<0) reduced+=product;
      REQUIRE(sum%product==reduced);
    }
  }
  for(unsigned a=0;a<256;++a) for(unsigned b=0;b<256;++b)
    REQUIRE(fixed_conversion::dot4(a*UINT32_C(0x01010101),b*UINT32_C(0x01010101))==4*a*b);
}
