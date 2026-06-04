#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/internal.hpp"

namespace {

std::vector<int8_t> residues_for(boost::multiprecision::cpp_int value, uint32_t prefix) {
  std::vector<int8_t> residues(prefix);
  for (uint32_t p = 0; p < prefix; ++p) {
    residues[p] = rns8::detail::centered_residue(value, rns8::detail::kDefaultModuli[p]);
  }
  return residues;
}

}  // namespace

TEST_CASE("CRT reconstruction recovers signed and unsigned bounded values") {
  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(-123456789), 9);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, 9, 123456789, value) == RNS8_SUCCESS);
    CHECK(value == -123456789);
  }

  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<int64_t>::min()), 9);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, 9, 1ull << 63u, value) == RNS8_SUCCESS);
    CHECK(value == std::numeric_limits<int64_t>::min());
  }

  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()), 9);
    uint64_t value = 0;
    CHECK(rns8::detail::reconstruct_unsigned(residues, 9, std::numeric_limits<uint64_t>::max(), value) ==
          RNS8_SUCCESS);
    CHECK(value == std::numeric_limits<uint64_t>::max());
  }
}

TEST_CASE("CRT canonical reconstruction matches exact representatives across prefixes") {
  for (uint32_t prefix : {1u, 2u, 4u, 8u, 9u, 12u, RNS8_MAX_SUPPORTED_PREFIX}) {
    const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(prefix);
    const std::vector<boost::multiprecision::cpp_int> values = {
        boost::multiprecision::cpp_int(0),
        boost::multiprecision::cpp_int(1),
        product / 2,
        product - 1,
    };
    for (const boost::multiprecision::cpp_int& value : values) {
      const auto residues = residues_for(value, prefix);
      CHECK(rns8::detail::reconstruct_canonical(residues, prefix) == value);
    }
  }
}

TEST_CASE("CRT signed reconstruction locks centered midpoint thresholds") {
  constexpr uint32_t prefix = 2;
  const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(prefix);
  const boost::multiprecision::cpp_int threshold = (product + 1) / 2;
  REQUIRE(product == 65280);
  REQUIRE(threshold == 32640);

  {
    const auto residues = residues_for(threshold - 1, prefix);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, prefix, 32639, value) == RNS8_SUCCESS);
    CHECK(value == 32639);
  }
  {
    const auto residues = residues_for(threshold + 1, prefix);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, prefix, 32639, value) == RNS8_SUCCESS);
    CHECK(value == -32639);
  }
  {
    const auto residues = residues_for(threshold, prefix);
    int64_t value = 123;
    CHECK(rns8::detail::reconstruct_signed(residues, prefix, 32640, value) == RNS8_RANGE_ERROR);
    CHECK(value == 123);
  }
  {
    const auto residues = residues_for(product - 1, prefix);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, prefix, 1, value) == RNS8_SUCCESS);
    CHECK(value == -1);
  }
}

TEST_CASE("CRT signed prefix-9 distinguishes int64 min from positive 2^63") {
  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<int64_t>::min()), 9);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, 9, 1ull << 63u, value) == RNS8_SUCCESS);
    CHECK(value == std::numeric_limits<int64_t>::min());
  }
  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(1) << 63u, 9);
    int64_t value = 77;
    CHECK(rns8::detail::reconstruct_signed(residues, 9, 1ull << 63u, value) == RNS8_RANGE_ERROR);
    CHECK(value == 77);
  }
}

TEST_CASE("CRT unsigned reconstruction is canonical and does not wrap past uint64") {
  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()), 9);
    uint64_t value = 0;
    CHECK(rns8::detail::reconstruct_unsigned(residues, 9, std::numeric_limits<uint64_t>::max(), value) ==
          RNS8_SUCCESS);
    CHECK(value == std::numeric_limits<uint64_t>::max());
  }
  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()) + 1, 9);
    uint64_t value = 123;
    CHECK(rns8::detail::reconstruct_unsigned(residues, 9, std::numeric_limits<uint64_t>::max(), value) ==
          RNS8_RANGE_ERROR);
    CHECK(value == 123);
  }
  {
    const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(9);
    const auto residues = residues_for(product - 1, 9);
    uint64_t value = 456;
    CHECK(rns8::detail::reconstruct_unsigned(residues, 9, std::numeric_limits<uint64_t>::max(), value) ==
          RNS8_RANGE_ERROR);
    CHECK(value == 456);
  }
}

TEST_CASE("CRT reconstruction reports insufficient range") {
  const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<int64_t>::max()), 8);
  int64_t value = 0;
  CHECK(rns8::detail::reconstruct_signed(
            residues, 8, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), value) == RNS8_RANGE_ERROR);
}

TEST_CASE("CRT reconstruction rejects malformed prefix inputs before range evaluation") {
  {
    const std::vector<int8_t> residues;
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(
              residues, 8, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), value) ==
          RNS8_INVALID_ARGUMENT);
  }
  {
    const std::vector<int8_t> residues(8, 0);
    uint64_t value = 0;
    CHECK(rns8::detail::reconstruct_unsigned(residues, 9, std::numeric_limits<uint64_t>::max(), value) ==
          RNS8_INVALID_ARGUMENT);
  }
  {
    const std::vector<int8_t> residues(1, 0);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, 0, 1, value) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8::detail::reconstruct_signed(residues, RNS8_MAX_SUPPORTED_PREFIX + 1, 1, value) ==
          RNS8_INVALID_ARGUMENT);
  }
}
