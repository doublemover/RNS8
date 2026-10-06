#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "backend_common/residue_dot.hpp"
#include "core/internal.hpp"

TEST_CASE("persistent residue dot reduces every safe K block", "[qualification][cpu]") {
  static_assert(rns8::detail::kResidueDotKBlock == RNS8_SAFE_INT32_K_BLOCK);
  for (const int64_t k : {65535, 65536, 65537, 131072, 131073, 262145}) {
    for (uint32_t p = 0; p < RNS8_MAX_SUPPORTED_PREFIX; ++p) {
      const uint16_t modulus = rns8::detail::kDefaultModuli[p];
      const int8_t lo = static_cast<int8_t>(-static_cast<int>(modulus / 2));
      for (const bool alternating : {false, true}) {
        INFO("k=" << k << " modulus=" << modulus << " alternating=" << alternating);
        std::vector<int8_t> a(static_cast<std::size_t>(k), lo);
        std::vector<int8_t> b(static_cast<std::size_t>(k * 3), 73);
        int64_t exact = 0;  // Test dimensions prove this sum fits int64_t.
        for (int64_t i = 0; i < k; ++i) {
          const int8_t value = alternating && i % 2 ? int8_t{1} : lo;
          b[static_cast<std::size_t>(i * 3)] = value;
          exact += static_cast<int64_t>(lo) * value;
        }
        const int8_t expected = rns8::detail::centered_residue(
            boost::multiprecision::cpp_int(exact), modulus);
        CHECK(rns8::detail::centered_residue_dot(a.data(), b.data(), k, 3, modulus) == expected);
      }
    }
  }
}
