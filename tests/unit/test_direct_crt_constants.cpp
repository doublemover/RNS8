#include "backend_common/balanced_crt.hpp"
#include "backend_common/direct_crt_constants.hpp"
#include "backend_common/strassen_program.hpp"
#include "backend_common/rank23_reference.hpp"
#include "backend_common/bilinear_resources.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>

TEST_CASE("bilinear resource queries separate full-stack offsets from sub-GEMM indexing") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (uint32_t blocks : {2u, 3u}) for (uint32_t prefix : {1u, 9u, 20u})
    for (int64_t size : {int64_t{1}, int64_t{17}, int64_t{16384}}) {
      const auto plan = query_bilinear_resources(size, size + 1, size + 2,
          size + 3, size + 4, size + 5, prefix, blocks, UINT64_MAX);
      REQUIRE(plan.status == RNS8_SUCCESS);
      const cpp_int hm = (size + blocks - 1) / blocks;
      const cpp_int hn = (size + 1 + blocks - 1) / blocks;
      const cpp_int hk = (size + 2 + blocks - 1) / blocks;
      REQUIRE(cpp_int(plan.a_bytes) == hm * hk * prefix);
      REQUIRE(cpp_int(plan.b_bytes) == hk * hn * prefix);
      REQUIRE(cpp_int(plan.product_bytes) == hm * hn * prefix);
      REQUIRE(cpp_int(plan.output_bytes) == cpp_int(size) * (size + 1) * prefix);
      REQUIRE(cpp_int(plan.total_bytes) == (hm * hk + hk * hn + hm * hn + cpp_int(size) * (size + 1)) * prefix);
      REQUIRE(plan.b_offset == plan.a_bytes);
      REQUIRE(plan.product_offset == plan.a_bytes + plan.b_bytes);
      REQUIRE(plan.output_offset + plan.output_bytes == plan.total_bytes);
      REQUIRE(cpp_int(plan.source_a.byte_extent) == (cpp_int(prefix) * size - 1) * (size + 3) + size + 2);
      REQUIRE(query_bilinear_resources(size, size + 1, size + 2, size + 3, size + 4, size + 5,
          prefix, blocks, plan.total_bytes).status == RNS8_SUCCESS);
      const auto insufficient = query_bilinear_resources(size, size + 1, size + 2, size + 3, size + 4, size + 5,
          prefix, blocks, plan.total_bytes - 1);
      REQUIRE(insufficient.status == RNS8_WORKSPACE_TOO_SMALL);
      REQUIRE(insufficient.total_bytes == plan.total_bytes);
    }
  REQUIRE_FALSE(qualified_i32_gemm_address(16384, 16384, 16384, 16384, 16384, 16384, 20));
  for (uint32_t blocks : {2u, 3u}) {
    const auto large = query_bilinear_resources(16384, 16384, 16384, 16384, 16384, 16384, 20, blocks, UINT64_MAX);
    REQUIRE(large.status == RNS8_SUCCESS);
    REQUIRE(large.source_a.byte_extent > INT32_MAX);
    REQUIRE(query_bilinear_resources(32768, 32768, 32768, 32768, 32768, 32768, 20, blocks, UINT64_MAX).status == RNS8_UNSUPPORTED_BACKEND);
  }
  REQUIRE(query_bilinear_resources(1, 1, 1, 1, 1, 1, 20, 1, UINT64_MAX).status == RNS8_INVALID_ARGUMENT);
  REQUIRE(query_bilinear_resources(INT64_MAX, 1, 1, 1, 1, 1, 20, 3, UINT64_MAX).status == RNS8_UNSUPPORTED_BACKEND);
  REQUIRE(query_bilinear_resources(1, 1, 1, 1, 1, 1, 0, 3, UINT64_MAX).status == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("rank23 exact coefficients reject every single coefficient mutation") {
  using namespace rns8::detail::rank23;
  REQUIRE(verified(program()));
  for (int p = 0; p < 23; ++p) for (int term = 0; term < 9; ++term) {
    auto changed = program(); ++changed.products[p].a[term]; REQUIRE_FALSE(verified(changed));
    changed = program(); ++changed.products[p].b[term]; REQUIRE_FALSE(verified(changed));
    changed = program(); ++changed.products[p].c[term]; REQUIRE_FALSE(verified(changed));
  }
}

TEST_CASE("rank23 scalar prototype preserves every byte modulus including composite rings") {
  using boost::multiprecision::cpp_int;
  using namespace rns8::detail::rank23;
  std::mt19937 random(4823);
  for (uint32_t modulus = 2; modulus <= 256; ++modulus) for (int sample = 0; sample < 24; ++sample) {
    int8_t a[9]{}, b[9]{}, actual[9]{};
    for (int i = 0; i < 9; ++i) {
      a[i] = static_cast<int8_t>(sample == 0 ? -128 : sample == 1 ? 127 : int(random() % 256) - 128);
      b[i] = static_cast<int8_t>(sample < 2 ? (i % 2 ? -128 : 127) : int(random() % 256) - 128);
    }
    REQUIRE(multiply3x3_modular(a, b, actual, modulus));
    for (int row = 0; row < 3; ++row) for (int col = 0; col < 3; ++col) {
      cpp_int expected = 0;
      for (int q = 0; q < 3; ++q) expected += cpp_int(a[row * 3 + q]) * b[q * 3 + col];
      expected %= modulus;
      if (expected < 0) expected += modulus;
      if (expected >= (modulus + 1) / 2) expected -= modulus;
      REQUIRE(cpp_int(actual[row * 3 + col]) == expected);
    }
  }
  int8_t zero[9]{}, untouched[9]{1,2,3,4,5,6,7,8,9};
  REQUIRE_FALSE(multiply3x3_modular(zero, zero, untouched, 257));
  for (int i = 0; i < 9; ++i) REQUIRE(untouched[i] == i + 1);
}

TEST_CASE("Strassen coefficient identity rejects mutations") {
  using namespace rns8::detail::strassen;
  REQUIRE(verified(program()));
  for (int p = 0; p < 7; ++p)
    for (int term = 0; term < 4; ++term) {
      auto changed = program();
      ++changed.products[p].a[term];
      REQUIRE_FALSE(verified(changed));
      changed = program();
      ++changed.products[p].b[term];
      REQUIRE_FALSE(verified(changed));
      changed = program();
      ++changed.products[p].c[term];
      REQUIRE_FALSE(verified(changed));
    }
}
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
