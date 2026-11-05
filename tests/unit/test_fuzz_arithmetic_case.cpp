#include <catch2/catch_test_macros.hpp>
#include "../fuzz/arithmetic_case.hpp"
#include <cmath>
#include <numeric>

namespace {
std::vector<uint8_t> encoded(uint8_t semantics, uint64_t a, uint64_t b) {
  std::vector<uint8_t> bytes(40);
  bytes[0] = 'R';
  bytes[1] = '8';
  bytes[2] = 'F';
  bytes[3] = '1';
  bytes[4] = semantics;
  bytes[6] = bytes[7] = bytes[8] = 1;
  bytes[15] = (semantics == 2 || semantics == 3) ? 0 : 20;
  if (semantics == 2)
    bytes[16] = 254;
  for (unsigned x = 0; x < 8; ++x) {
    bytes[24 + x] = uint8_t(a >> (8 * x));
    bytes[32 + x] = uint8_t(b >> (8 * x));
  }
  return bytes;
}
} // namespace
TEST_CASE(
    "fuzz arithmetic codec preserves literal domains and rejects malformed allocation requests",
    "[fuzz][contract]") {
  using namespace rns8_fuzz;
  auto bytes = encoded(0, uint64_t(INT64_MIN), 1);
  ArithmeticCase value;
  REQUIRE(decode(bytes.data(), bytes.size(), value));
  REQUIRE(signed_bits(value.a[0]) == INT64_MIN);
  REQUIRE(oracle(value).admitted);
  CHECK(oracle(value).cells[0] == cpp_int(INT64_MIN));
  bytes[18] = 1;
  CHECK_FALSE(decode(bytes.data(), bytes.size(), value));
  bytes[18] = 0;
  bytes[8] = bytes[9] = bytes[10] = bytes[11] = 255;
  CHECK_FALSE(decode(bytes.data(), bytes.size(), value));
  bytes = encoded(0, 1, 1);
  bytes.pop_back();
  CHECK_FALSE(decode(bytes.data(), bytes.size(), value));
  bytes = encoded(2, 256, 1);
  CHECK_FALSE(decode(bytes.data(), bytes.size(), value));
}
TEST_CASE("fuzz oracle distinguishes wrap finite and full signed wide arithmetic",
          "[fuzz][contract]") {
  using namespace rns8_fuzz;
  ArithmeticCase value;
  for (uint8_t semantic : {2, 3, 4, 5}) {
    const auto bytes = encoded(semantic, semantic == 2 ? 255 : UINT64_MAX, semantic == 2 ? 255 : 2);
    REQUIRE(decode(bytes.data(), bytes.size(), value));
    const auto result = oracle(value);
    REQUIRE(result.admitted);
    if (semantic == 2)
      CHECK(result.cells[0] == 1);
    if (semantic == 3)
      CHECK(result.cells[0] == cpp_int(UINT64_MAX) - 1);
    if (semantic == 4)
      CHECK(result.cells[0] == -2);
    if (semantic == 5)
      CHECK(result.cells[0] == cpp_int(UINT64_MAX) * 2);
  }
  auto bytes = encoded(0, uint64_t(INT64_MAX), 2);
  REQUIRE(decode(bytes.data(), bytes.size(), value));
  CHECK_FALSE(oracle(value).admitted);
  bytes = encoded(4, uint64_t(INT64_MIN), uint64_t(INT64_MIN));
  REQUIRE(decode(bytes.data(), bytes.size(), value));
  CHECK(oracle(value).cells[0] == (cpp_int(1) << 126));
}
TEST_CASE("independent fuzz ladder matches public products and pairwise coprimality",
          "[fuzz][contract]") {
  using namespace rns8_fuzz;
  cpp_int product = 1;
  for (uint32_t index = 0; index < oracle_moduli.size(); ++index) {
    CHECK(rns8_default_modulus(index) == oracle_moduli[index]);
    for (uint32_t earlier = 0; earlier < index; ++earlier)
      CHECK(std::gcd(oracle_moduli[index], oracle_moduli[earlier]) == 1);
    product *= oracle_moduli[index];
    CHECK(std::abs(rns8_prefix_range_bits(index + 1) - std::log2(product.convert_to<double>())) <
          1e-10);
  }
}
