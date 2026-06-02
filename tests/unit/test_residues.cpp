#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>
#include <limits>

#include "core/internal.hpp"

TEST_CASE("centered residues use the spec ranges") {
  using boost::multiprecision::cpp_int;
  using rns8::detail::canonical_from_centered;
  using rns8::detail::centered_residue;

  CHECK(centered_residue(cpp_int(127), 256) == 127);
  CHECK(centered_residue(cpp_int(128), 256) == -128);
  CHECK(centered_residue(cpp_int(-128), 256) == -128);
  CHECK(centered_residue(cpp_int(-129), 256) == 127);
  CHECK(canonical_from_centered(static_cast<int8_t>(-128), 256) == 128);

  CHECK(centered_residue(cpp_int(127), 255) == 127);
  CHECK(centered_residue(cpp_int(128), 255) == -127);
  CHECK(centered_residue(cpp_int(-127), 255) == -127);
  CHECK(centered_residue(cpp_int(-128), 255) == 127);

  CHECK(centered_residue(cpp_int(125), 251) == 125);
  CHECK(centered_residue(cpp_int(126), 251) == -125);
  CHECK(centered_residue(cpp_int(-125), 251) == -125);
}

TEST_CASE("full signed boundary inputs reduce deterministically") {
  using boost::multiprecision::cpp_int;
  const cpp_int min_value = cpp_int(std::numeric_limits<int64_t>::min());
  const int8_t residue = rns8::detail::centered_residue(min_value, 251);
  CHECK(residue >= -125);
  CHECK(residue <= 125);
}

