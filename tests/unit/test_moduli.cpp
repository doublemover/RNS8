#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <numeric>

#include "core/internal.hpp"
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

TEST_CASE("default modulus ladder has explicit pairwise gcd and prefix product boundaries") {
  for (uint32_t i = 0; i < rns8_default_modulus_count(); ++i) {
    for (uint32_t j = i + 1; j < rns8_default_modulus_count(); ++j) {
      CHECK(std::gcd(rns8_default_modulus(i), rns8_default_modulus(j)) == 1);
    }
  }

  boost::multiprecision::cpp_int product = 1;
  for (uint32_t prefix = 1; prefix <= RNS8_MAX_SUPPORTED_PREFIX; ++prefix) {
    product *= rns8_default_modulus(prefix - 1u);
    CHECK(rns8::detail::modulus_product(prefix) == product);
    CHECK(rns8::detail::required_prefix_for_range(product - 1) == prefix);
    if (prefix < RNS8_MAX_SUPPORTED_PREFIX) {
      CHECK(rns8::detail::required_prefix_for_range(product) == prefix + 1u);
    } else {
      CHECK(rns8::detail::required_prefix_for_range(product) == 0);
    }
  }
}

TEST_CASE("default modulus ladder exposes exact prefix-9 bounded product contracts") {
  boost::multiprecision::cpp_int prefix9 = 1;
  for (uint32_t i = 0; i < RNS8_DEFAULT_BOUNDED_PREFIX; ++i) {
    prefix9 *= rns8_default_modulus(i);
  }
  CHECK(rns8::detail::modulus_product(RNS8_DEFAULT_BOUNDED_PREFIX) == prefix9);
  CHECK(prefix9 > (boost::multiprecision::cpp_int(1) << 64u));
  CHECK(rns8::detail::modulus_product(RNS8_DEFAULT_BOUNDED_PREFIX - 1u) <
        (boost::multiprecision::cpp_int(1) << 64u));
  CHECK(rns8::detail::required_prefix_for_range((boost::multiprecision::cpp_int(1) << 64u) - 1) ==
        RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(rns8::detail::required_prefix_for_range(boost::multiprecision::cpp_int(1) << 64u) ==
        RNS8_DEFAULT_BOUNDED_PREFIX);
}

TEST_CASE("default modulus public queries reject out-of-range prefixes and indices") {
  CHECK(rns8_default_modulus(RNS8_DEFAULT_MODULUS_COUNT) == 0);
  CHECK(rns8_prefix_range_bits(0) == 0.0);
  CHECK(rns8_prefix_range_bits(RNS8_DEFAULT_MODULUS_COUNT + 1u) == 0.0);
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

