#include <catch2/catch_test_macros.hpp>

#include <array>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <limits>

#include "backend_common/finite_u8_reducer.hpp"
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

TEST_CASE("accelerator finite reducers match exact centered residues") {
  using boost::multiprecision::cpp_int;
  namespace finite = rns8::detail::finite_u8;

  constexpr std::array<uint32_t, 28> moduli = {
      256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
      193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};
  constexpr std::array<int32_t, 20> values = {
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::min() + 1,
      -536870912,
      -65537,
      -65536,
      -32769,
      -32768,
      -257,
      -256,
      -1,
      0,
      1,
      127,
      128,
      255,
      256,
      32767,
      32768,
      65536,
      std::numeric_limits<int32_t>::max()};

  for (const uint32_t modulus : moduli) {
    const uint32_t reciprocal = finite::modulus_reciprocal_u32(modulus);
    for (const int32_t value : values) {
      const int8_t expected =
          rns8::detail::centered_residue(cpp_int(value), static_cast<uint16_t>(modulus));
      CHECK(finite::reduce_to_centered_accelerator_i32(value, modulus, reciprocal) == expected);
      CHECK(finite::reduce_to_centered_ck_i32(value, modulus, reciprocal) == expected);
    }
  }
}

