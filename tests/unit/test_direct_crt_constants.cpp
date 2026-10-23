#include "backend_common/balanced_crt.hpp"
#include "backend_common/direct_crt_constants.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>

TEST_CASE("direct CRT constants and unreduced widths match exact integer "
          "construction") {
  using boost::multiprecision::cpp_int;
  using namespace rns8::detail;
  auto wide = [](const uint64_t *limbs) -> cpp_int {
    return cpp_int(limbs[0]) + (cpp_int(limbs[1]) << 64) +
           (cpp_int(limbs[2]) << 128);
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
      REQUIRE((weight * direct_crt::kConstants.inverses[prefix][p]) % modulus ==
              1);
      sum_max += (modulus - 1) * weight;
    }
    REQUIRE(sum_max < prefix * product);
    REQUIRE(sum_max < (cpp_int(1) << 165));
    for (const cpp_int value :
         std::vector<cpp_int>{0, 1, product / 2, product - 1}) {
      cpp_int sum = 0;
      for (uint32_t p = 0; p < prefix; ++p) {
        const uint32_t modulus = garner::kConstants.moduli[p];
        const cpp_int coefficient =
            ((value % modulus) * direct_crt::kConstants.inverses[prefix][p]) %
            modulus;
        sum += coefficient * wide(direct_crt::kConstants.weights[prefix][p]);
      }
      REQUIRE(sum <= sum_max);
      REQUIRE(sum % product == value);
    }
  }
}

TEST_CASE("balanced CRT levels and bounded arithmetic match Boost") {
  using boost::multiprecision::cpp_int;
  using namespace rns8::detail;
  const auto wide = [](balanced_crt::Wide x) -> cpp_int {
    return cpp_int(x.limb[0]) + (cpp_int(x.limb[1]) << 64) +
           (cpp_int(x.limb[2]) << 128);
  };
  std::mt19937_64 random(3902);
  for (uint32_t prefix = 1; prefix <= 20; ++prefix) {
    const auto &plan = balanced_crt::plan(prefix);
    REQUIRE(plan.count == prefix - 1);
    cpp_int product = 1;
    for (uint32_t p = 0; p < prefix; ++p)
      product *= garner::kConstants.moduli[p];
    for (int sample = 0; sample < 128; ++sample) {
      cpp_int value = (cpp_int(random()) + (cpp_int(random()) << 64) +
                       (cpp_int(random()) << 128)) %
                      product;
      if (sample == 0)
        value = 0;
      if (sample == 1)
        value = product - 1;
      if (sample == 2)
        value = product / 2;
      balanced_crt::Wide slots[20]{};
      uint32_t depths[20]{};
      for (uint32_t p = 0; p < prefix; ++p)
        slots[p].limb[0] =
            static_cast<uint64_t>(value % garner::kConstants.moduli[p]);
      for (uint32_t i = 0; i < plan.count; ++i) {
        const auto &node = plan.merges[i];
        const cpp_int a = wide(node.a), b = wide(node.b),
                      inv = wide(node.inverse);
        REQUIRE(a * b < (cpp_int(1) << 160));
        REQUIRE((a * inv) % b == 1);
        REQUIRE(wide(slots[node.left]) < a);
        REQUIRE(wide(slots[node.right]) < b);
        const cpp_int delta =
            (wide(slots[node.right]) - wide(slots[node.left]) % b + b) % b;
        const cpp_int expected =
            wide(slots[node.left]) + a * ((delta * inv) % b);
        slots[node.left] =
            balanced_crt::merge(slots[node.left], slots[node.right], node);
        REQUIRE(wide(slots[node.left]) == expected);
        REQUIRE(expected == value % (a * b));
        depths[node.left] = 1 + std::max(depths[node.left], depths[node.right]);
      }
      REQUIRE(wide(slots[0]) == value);
      uint32_t depth = 0;
      for (uint32_t n = 1; n < prefix; n *= 2)
        ++depth;
      REQUIRE(depths[0] == depth);
    }
  }
  REQUIRE_THROWS_AS(balanced_crt::plan(0), std::invalid_argument);
  REQUIRE_THROWS_AS(balanced_crt::plan(21), std::invalid_argument);
}
