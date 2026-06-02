#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rns8/moduli.h"

TEST_CASE("default modulus ladder is stable and pairwise coprime") {
  const uint16_t expected[] = {256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
                               193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};
  REQUIRE(rns8_default_modulus_count() == 28);
  for (uint32_t i = 0; i < rns8_default_modulus_count(); ++i) {
    CHECK(rns8_default_modulus(i) == expected[i]);
  }
  CHECK(rns8_validate_default_moduli() == RNS8_SUCCESS);
}

TEST_CASE("prefix range bits match the research spec table") {
  CHECK(rns8_prefix_range_bits(4) == Catch::Approx(31.949).margin(0.002));
  CHECK(rns8_prefix_range_bits(5) == Catch::Approx(39.897).margin(0.002));
  CHECK(rns8_prefix_range_bits(6) == Catch::Approx(47.798).margin(0.002));
  CHECK(rns8_prefix_range_bits(7) == Catch::Approx(55.662).margin(0.002));
  CHECK(rns8_prefix_range_bits(8) == Catch::Approx(63.502).margin(0.002));
  CHECK(rns8_prefix_range_bits(9) == Catch::Approx(71.328).margin(0.002));
  CHECK(rns8_prefix_range_bits(10) == Catch::Approx(79.129).margin(0.002));
  CHECK(rns8_prefix_range_bits(12) == Catch::Approx(94.612).margin(0.002));
  CHECK(rns8_prefix_range_bits(16) == Catch::Approx(125.040).margin(0.002));
  CHECK(rns8_prefix_range_bits(18) == Catch::Approx(140.024).margin(0.002));
  CHECK(rns8_prefix_range_bits(19) == Catch::Approx(147.458).margin(0.002));
  CHECK(rns8_prefix_range_bits(20) == Catch::Approx(154.842).margin(0.002));
}

