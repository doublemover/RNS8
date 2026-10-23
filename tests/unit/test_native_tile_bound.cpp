#include "backend_common/native_tile_bound.hpp"
#include "backend_common/garner_constants.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include "backend_common/native_narrow.hpp"
#include "backend_common/mixed_radix_lift.hpp"

TEST_CASE("mixed radix lift subtracts the complete source product for negatives") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  const auto& moduli = garner::kConstants.moduli;
  cpp_int product = 1;
  for (uint32_t count = 1; count < 20; ++count) {
    product *= moduli[count - 1];
    const std::vector<cpp_int> values{0, 1, product - 1, (product - 1) / 2, product / 2 + 1};
    for (const auto& canonical : values) {
      uint32_t digits[20]{};
      cpp_int remaining = canonical;
      for (uint32_t i = 0; i < count; ++i) {
        digits[i] = static_cast<uint32_t>(remaining % moduli[i]);
        remaining /= moduli[i];
      }
      REQUIRE(remaining == 0);
      for (bool negative : {false, true}) for (uint32_t p = count; p < 20; ++p) {
        const cpp_int value = negative ? cpp_int(canonical - product) : canonical;
        cpp_int expected = value % moduli[p];
        if (expected < 0) expected += moduli[p];
        REQUIRE(mixed_radix_lift_residue(digits, moduli, count, moduli[p], negative) == expected);
      }
    }
  }
  // X=P-1 represents -1: reducing X alone is a different residue for every
  // coprime suffix modulus and must not be accepted as signed extension.
  const uint32_t digits[]{255}, base[]{256};
  REQUIRE(mixed_radix_lift_residue(digits, base, 1, 255, false) == 0);
  REQUIRE(mixed_radix_lift_residue(digits, base, 1, 255, true) == 254);
}

TEST_CASE("native partial sum admission agrees with arbitrary precision at caps") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (uint64_t limit : {uint64_t{INT32_MAX}, uint64_t{INT64_MAX}, UINT64_MAX})
    for (uint64_t a : {uint64_t{0}, uint64_t{1}, uint64_t{128}, uint64_t{255}, uint64_t{1} << 63, UINT64_MAX})
      for (uint64_t b : {uint64_t{0}, uint64_t{1}, uint64_t{127}, uint64_t{255}, UINT64_MAX})
        for (uint64_t k : {uint64_t{1}, uint64_t{33025}, uint64_t{33026}, uint64_t{131071}, uint64_t{131072}, UINT64_MAX}) {
          const cpp_int expected = cpp_int(a) * b * k;
          uint64_t bound = 41;
          REQUIRE(native_partial_sum_bound(a, b, k, limit, bound) == (expected <= limit));
          REQUIRE(cpp_int(bound) == (expected <= limit ? expected : cpp_int(41)));
        }
  uint64_t bound = 41;
  REQUIRE_FALSE(native_partial_sum_bound(0, 0, 0, 0, bound));
  REQUIRE(bound == 41);
}

TEST_CASE("native L1 bounds retain all carry bits through full-width products") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  const auto decode = [](const NativeTileBound& value) -> cpp_int {
    return cpp_int(value.magnitude[0]) + (cpp_int(value.magnitude[1]) << 64) +
           (cpp_int(value.magnitude[2]) << 128);
  };
  for (uint64_t maximum : {uint64_t{0}, uint64_t{1}, uint64_t{1} << 63, UINT64_MAX})
    for (uint32_t k : {1u, 17u, 65536u, UINT32_MAX}) {
      const cpp_int sum = cpp_int(maximum) * k;
      const NativeAxisSummary axis{maximum, static_cast<uint64_t>(sum & UINT64_MAX),
                                   static_cast<uint64_t>(sum >> 64)};
      REQUIRE(decode(native_axis_l1_bound(axis, axis, k)) == sum * maximum);
    }
  std::mt19937_64 rng(340034);
  for (unsigned iteration = 0; iteration < 512; ++iteration) {
    const uint64_t maximum = rng(), low = rng(), high = static_cast<uint32_t>(rng());
    REQUIRE(decode(native_l1_product_bound(maximum, low, high)) ==
            cpp_int(maximum) * (cpp_int(low) + (cpp_int(high) << 64)));
  }
  const NativeAxisSummary one_outlier{100, 100, 0}, dense{3, 300, 0};
  REQUIRE(decode(native_axis_l1_bound(one_outlier, dense, 100)) == 300);
  REQUIRE(decode(native_tile_max_bound(100, 3, 100)) == 30000);
}

TEST_CASE("native tile maximum bounds have exact multiword products and strict prefixes") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr auto basis = garner::make_constants();
  for (uint64_t a : {uint64_t{0}, uint64_t{1}, uint64_t{128}, uint64_t{1} << 32, uint64_t{1} << 63, UINT64_MAX})
    for (uint64_t b : {uint64_t{0}, uint64_t{1}, uint64_t{255}, uint64_t{1} << 32, uint64_t{1} << 63, UINT64_MAX})
      for (uint32_t k : {1u, 65535u, 65536u, 65537u, UINT32_MAX}) {
        auto bound = native_tile_max_bound(a, b, k);
        const cpp_int exact = cpp_int(a) * b * k;
        const cpp_int observed = cpp_int(bound.magnitude[0]) + (cpp_int(bound.magnitude[1]) << 64) +
                                 (cpp_int(bound.magnitude[2]) << 128);
        REQUIRE(observed == exact);
        for (bool signed_result : {false, true})
          for (uint32_t capacity = 1; capacity <= 20; ++capacity) {
            select_native_bound_prefix(bound, signed_result, capacity, basis.products);
            cpp_int product = 1;
            uint32_t expected = exact == 0 ? 0 : 21;
            for (uint32_t p = 0; p < capacity && expected == 21; ++p) {
              product *= basis.moduli[p];
              if (product > exact * (signed_result ? 2 : 1)) expected = p + 1;
            }
            REQUIRE(bound.required_prefix == expected);
          }
      }
  auto equal = native_tile_max_bound(128, 1, 1);
  select_native_bound_prefix(equal, true, 1, basis.products);
  REQUIRE(equal.required_prefix == 21); // P == 2*bound is insufficient.
}

TEST_CASE("K-panel bound summation preserves carries and disjoint zero proofs") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  const auto decode = [](const NativeTileBound& value) -> cpp_int {
    return cpp_int(value.magnitude[0]) + (cpp_int(value.magnitude[1]) << 64) +
           (cpp_int(value.magnitude[2]) << 128);
  };
  for (uint64_t maximum : {uint64_t{0}, uint64_t{1}, uint64_t{1} << 63, UINT64_MAX}) {
    NativeTileBound accumulated{};
    uint64_t terms = 0;
    for (uint32_t panel : {1u, 65535u, 65536u, UINT32_MAX - 131072u}) {
      add_native_panel_bound(accumulated, native_tile_max_bound(maximum, maximum, panel));
      terms += panel;
      REQUIRE(decode(accumulated) == cpp_int(maximum) * maximum * terms);
    }
    REQUIRE(terms == UINT32_MAX);
  }
  const NativeAxisSummary zero{}, large{UINT64_MAX, UINT64_MAX, 0};
  NativeTileBound disjoint{};
  add_native_panel_bound(disjoint, native_axis_l1_bound(large, zero, 1));
  add_native_panel_bound(disjoint, native_axis_l1_bound(zero, large, 1));
  REQUIRE(decode(disjoint) == 0);
  REQUIRE(decode(native_axis_l1_bound(large, large, 2)) > 0);
}
