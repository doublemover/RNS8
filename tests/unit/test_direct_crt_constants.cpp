#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <vector>
#include "backend_common/direct_crt_constants.hpp"

TEST_CASE("direct CRT constants and unreduced widths match exact integer construction") {
  using boost::multiprecision::cpp_int;
  using namespace rns8::detail;
  auto wide = [](const uint64_t* limbs) -> cpp_int {
    return cpp_int(limbs[0]) + (cpp_int(limbs[1]) << 64) + (cpp_int(limbs[2]) << 128);
  };
  cpp_int product = 1;
  for (uint32_t prefix = 1; prefix <= 20; ++prefix) {
    product *= garner::kConstants.moduli[prefix - 1];
    REQUIRE(wide(direct_crt::kConstants.products[prefix]) == product);
    cpp_int sum_max = 0;
    for (uint32_t p = 0; p < prefix; ++p) {
      const uint32_t modulus = garner::kConstants.moduli[p];
      const cpp_int weight = wide(direct_crt::kConstants.weights[prefix][p]);
      REQUIRE(weight == product / modulus);
      REQUIRE((weight * direct_crt::kConstants.inverses[prefix][p]) % modulus == 1);
      sum_max += (modulus - 1) * weight;
    }
    REQUIRE(sum_max < prefix * product);
    REQUIRE(sum_max < (cpp_int(1) << 165));
    for (const cpp_int value : std::vector<cpp_int>{0, 1, product / 2, product - 1}) {
      cpp_int sum = 0;
      for (uint32_t p = 0; p < prefix; ++p) {
        const uint32_t modulus = garner::kConstants.moduli[p];
        const cpp_int coefficient = ((value % modulus) * direct_crt::kConstants.inverses[prefix][p]) % modulus;
        sum += coefficient * wide(direct_crt::kConstants.weights[prefix][p]);
      }
      REQUIRE(sum <= sum_max);
      REQUIRE(sum % product == value);
    }
  }
}
