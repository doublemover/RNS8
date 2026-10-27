#include "backend_common/finite_u8_reducer.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstring>

TEST_CASE("native reciprocal reduction preserves all byte moduli and full-width signedness") {
  using namespace rns8::detail::finite_u8;
  using boost::multiprecision::cpp_int;
  const cpp_int radix32 = cpp_int(1) << 32, radix64 = cpp_int(1) << 64;
  const std::array<uint64_t, 14> boundaries = {
      0, 1, 127, 128, 255, 256, UINT32_MAX, uint64_t{1} << 32,
      (uint64_t{1} << 32) + 1, (uint64_t{1} << 63) - 1,
      uint64_t{1} << 63, (uint64_t{1} << 63) + 1, UINT64_MAX - 1, UINT64_MAX};
  for (uint32_t modulus = 2; modulus <= 256; ++modulus) {
    INFO("modulus=" << modulus);
    const auto reducer = native64_reducer(modulus);
    REQUIRE(reducer.reciprocal == (radix32 / modulus).convert_to<uint32_t>());
    REQUIRE(reducer.radix32 == (radix32 % modulus).convert_to<uint32_t>());
    REQUIRE(reducer.signed_bias == (radix64 % modulus).convert_to<uint32_t>());
    const auto centered = [&](const cpp_int& value) {
      cpp_int residue = value % modulus;
      if (residue < 0) residue += modulus;
      if (residue >= (modulus + 1) / 2) residue -= modulus;
      return residue.convert_to<int>();
    };
    const auto check = [&](uint64_t bits) {
      int64_t signed_value;
      std::memcpy(&signed_value, &bits, sizeof(bits));
      REQUIRE(canonical_native_u64(bits, reducer) == (cpp_int(bits) % modulus).convert_to<uint32_t>());
      REQUIRE(static_cast<int>(center_native_u64(bits, reducer)) == centered(cpp_int(bits)));
      REQUIRE(static_cast<int>(center_native_i64(signed_value, reducer)) == centered(cpp_int(signed_value)));
    };
    for (auto value : boundaries) check(value);
    const uint64_t last_multiple = (UINT64_MAX / modulus) * modulus;
    check(last_multiple - 1); check(last_multiple);
    if (last_multiple < UINT64_MAX) check(last_multiple + 1);
    uint64_t state = 0x8b5ad4cef9a37126ULL;
    for (uint32_t sample = 0; sample < 1024; ++sample) {
      state ^= state << 13; state ^= state >> 7; state ^= state << 17;
      check(state);
    }
  }
}
