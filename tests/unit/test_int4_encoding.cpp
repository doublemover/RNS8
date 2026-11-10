#include <catch2/catch_test_macros.hpp>
#include "../../src/backend_common/int4_encoding.hpp"

TEST_CASE("finite INT4 encoding preserves every signed byte for all supported moduli") {
  for (int modulus = 2; modulus <= 16; ++modulus)
    for (int value = -128; value <= 127; ++value) {
      const int encoded = rns8::detail::finite_int4_nibble(static_cast<int8_t>(value), modulus);
      const int decoded = encoded >= 8 ? encoded - 16 : encoded;
      REQUIRE(decoded >= -8);
      REQUIRE(decoded <= 7);
      REQUIRE((value - decoded) % modulus == 0);
      for (int other = -128; other <= 127; ++other) {
        const int packed_b = rns8::detail::finite_int4_nibble(static_cast<int8_t>(other), modulus);
        const int decoded_b = packed_b >= 8 ? packed_b - 16 : packed_b;
        REQUIRE((value * other - decoded * decoded_b) % modulus == 0);
      }
    }
}
