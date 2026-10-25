#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <limits>
#include "backend_common/residue_epilogue.hpp"

TEST_CASE("modular integer axpby epilogue preserves coefficient extremes and composite rings") {
  using boost::multiprecision::cpp_int;
  const int64_t coefficients[] = {INT64_MIN, INT64_MAX, -256, -1, 0, 1, 255};
  for (uint32_t modulus = 2; modulus <= 256; ++modulus)
    for (int32_t product : {-128, -127, -1, 0, 1, 127})
      for (int32_t addend : {-128, -1, 0, 127})
        for (const auto alpha : coefficients)
          for (const auto beta : coefficients) {
            cpp_int expected = (cpp_int(alpha) * product + cpp_int(beta) * addend) % modulus;
            if (expected < 0) expected += modulus;
            if (expected >= (modulus + 1) / 2) expected -= modulus;
            CHECK(rns8::detail::centered_residue_axpby(static_cast<int8_t>(product), static_cast<int8_t>(addend),
                                                     alpha, beta, modulus) == expected.convert_to<int>());
          }
}

TEST_CASE("shared lowered coefficients remove only modular dead channels and retain original integer extremes") {
  using namespace rns8::detail;
  for(uint32_t modulus=2;modulus<=256;++modulus)for(int64_t alpha:{int64_t{0},int64_t{modulus},-int64_t{modulus},INT64_MIN,INT64_MAX})for(int64_t beta:{int64_t{0},int64_t{1},int64_t{-1},INT64_MIN,INT64_MAX}){
    const auto lowered=lower_residue_axpby(alpha,beta,modulus);
    CHECK(lowered.alpha>=0);CHECK(lowered.alpha<int32_t(modulus));CHECK(lowered.beta>=0);CHECK(lowered.beta<int32_t(modulus));
    for(int product:{-128,-1,0,1,127})for(int addend=-128;addend<=127;++addend){
      const auto dead_aware=centered_residue_axpby_reduced(lowered.alpha?int8_t(product):int8_t{0},lowered.beta?int8_t(addend):int8_t{0},lowered);
      boost::multiprecision::cpp_int expected=(boost::multiprecision::cpp_int(alpha)*product+boost::multiprecision::cpp_int(beta)*addend)%modulus;
      if(expected<0)expected+=modulus;if(expected>=(modulus+1)/2)expected-=modulus;
      CHECK(dead_aware==expected.convert_to<int>());
    }
  }
  const uint16_t moduli[]{256,255,253};ResidueAxpbyTable table;
  REQUIRE(lower_residue_axpby_table(255,256,moduli,3,table));
  CHECK(table.channel[0].beta==0);CHECK(table.channel[1].alpha==0);CHECK(table.channel[2].alpha!=0);
  const auto before=table;const uint16_t bad[]{256,257};
  CHECK_FALSE(lower_residue_axpby_table(1,1,bad,2,table));CHECK(table.channel[1].alpha==before.channel[1].alpha);
  CHECK_FALSE(lower_residue_axpby_table(1,1,moduli,0,table));CHECK_FALSE(lower_residue_axpby_table(1,1,moduli,21,table));
}

#include "core/exact_range.hpp"
TEST_CASE("only original zero coefficients remove exact expression range contributions") {
  using namespace rns8::detail;
  rns8_plan plan;plan.desc.semantics=RNS8_EXACT_WIDE_SIGNED;plan.desc.m=plan.desc.n=plan.desc.k=1;plan.prefix=1;plan.modulus_product=256;
  rns8_matrix a,b,d;a.exact_range_prefix=b.exact_range_prefix=d.exact_range_prefix=1;
  a.exact_max_magnitude=b.exact_max_magnitude=2;d.exact_max_magnitude=1;
  boost::multiprecision::cpp_int bound;
  CHECK(exact_axpby_range(plan,a,b,d,256,0,bound)==RNS8_RANGE_ERROR);CHECK(bound==1024);
  CHECK(exact_axpby_range(plan,a,b,d,0,1,bound)==RNS8_SUCCESS);CHECK(bound==1);
  a.exact_max_magnitude=boost::multiprecision::cpp_int(1)<<512;
  CHECK(exact_axpby_range(plan,a,b,d,0,0,bound)==RNS8_SUCCESS);CHECK(bound==0);
  a.exact_range_prefix=0;CHECK(exact_axpby_range(plan,a,b,d,0,0,bound)==RNS8_RANGE_ERROR);
}
