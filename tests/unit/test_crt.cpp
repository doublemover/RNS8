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
    const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()), 9);
    uint64_t value = 0;
    CHECK(rns8::detail::reconstruct_unsigned(residues, 9, std::numeric_limits<uint64_t>::max(), value) ==
          RNS8_SUCCESS);
    CHECK(value == std::numeric_limits<uint64_t>::max());
  }
}

TEST_CASE("CRT reconstruction reports insufficient range") {
  const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<int64_t>::max()), 8);
  int64_t value = 0;
  CHECK(rns8::detail::reconstruct_signed(
            residues, 8, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), value) == RNS8_RANGE_ERROR);
}

