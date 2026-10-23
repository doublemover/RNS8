#include "backend_common/native_tile_bound.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>

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
