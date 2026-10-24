#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <limits>
#include "backend_common/residue_epilogue.hpp"

TEST_CASE("modular integer axpby epilogue preserves coefficient extremes and composite rings") {
  using boost::multiprecision::cpp_int;
  const int64_t coefficients[] = {INT64_MIN, INT64_MAX, -256, -1, 0, 1, 255};
  for (uint32_t modulus = 2; modulus <= 256; ++modulus)
    for (int32_t product : {-128, -127, -1, 0, 1, 127})
      for (int32_t addend : {-128, -1, 0, 127})
        for (const auto alpha : coefficients)
          for (const auto beta : coefficients) {
            cpp_int expected = (cpp_int(alpha) * product + cpp_int(beta) * addend) % modulus;
            if (expected < 0) expected += modulus;
            if (expected >= (modulus + 1) / 2) expected -= modulus;
            CHECK(rns8::detail::centered_residue_axpby(static_cast<int8_t>(product), static_cast<int8_t>(addend),
                                                     alpha, beta, modulus) == expected.convert_to<int>());
          }
}
