#include "encodings.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <vector>

namespace {
template<class Encoding, class T>
void check_generated_encoding() {
  using boost::multiprecision::cpp_int;
  const T values[] = {std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), T{0}, T{1}, T{127}, T{255}};
  for (std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{17},
                            std::size_t{65535}, std::size_t{65536}, std::size_t{65537}}) {
    std::vector<T> a(count), b(count);
    cpp_int sum = 0;
    for (std::size_t i = 0; i < count; ++i) {
      a[i] = values[i % 6]; b[i] = values[(i * 5 + 2) % 6];
      sum += cpp_int(a[i]) * b[i];
    }
    int expected = (sum % Encoding::modulus).template convert_to<int>();
    if (expected < 0) expected += static_cast<int>(Encoding::modulus);
    if (expected >= static_cast<int>((Encoding::modulus + 1) / 2)) expected -= static_cast<int>(Encoding::modulus);
    int8_t output = 99;
    REQUIRE(Encoding::dot(a.data(), b.data(), count, output));
    REQUIRE(output == expected);
  }
  T value = 1;
  int8_t output = 99;
  REQUIRE_FALSE(Encoding::dot(static_cast<const T*>(nullptr), &value, 1, output));
  REQUIRE(output == 99);
  REQUIRE_FALSE(Encoding::dot(&value, static_cast<const T*>(nullptr), 1, output));
  REQUIRE(output == 99);
}

template<class Encoding>
void check_encoding_variant() {
  check_generated_encoding<Encoding, int64_t>();
  check_generated_encoding<Encoding, uint64_t>();
  // Exhaust every encoded residue pair, independently of generated tables.
  for (int64_t a = 0; a < Encoding::modulus; ++a)
    for (int64_t b = 0; b < Encoding::modulus; ++b) {
      auto expected = (a * b) % Encoding::modulus;
      if (expected >= (Encoding::modulus + 1) / 2) expected -= Encoding::modulus;
      int8_t output = 99;
      REQUIRE(Encoding::dot(&a, &b, 1, output));
      REQUIRE(output == expected);
    }
}
}

TEST_CASE("generated AWE and base13 integer dots preserve exact division and slab boundaries") {
  using namespace rns8_encoding_reference;
  check_encoding_variant<rns8_awe6_251>();
  check_encoding_variant<rns8_awe6_256>();
  check_encoding_variant<rns8_base13_251>();
  check_encoding_variant<rns8_base13_256>();
  check_encoding_variant<rns8_exact_divide>();
}
