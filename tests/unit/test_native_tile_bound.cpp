#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include "../../src/backend_common/native_tile_bound.hpp"
#include "../../src/backend_common/garner_constants.hpp"

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
