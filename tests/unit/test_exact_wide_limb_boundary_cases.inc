TEST_CASE("exact-wide signed CRT export uses centered half-open representative") {
  constexpr uint32_t prefix = 3;
  const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(prefix);
  const boost::multiprecision::cpp_int half = product / 2;
  const auto residues = residues_for(half, prefix);

  uint64_t limb = 0;
  REQUIRE(rns8::detail::export_exact_wide_signed_limbs(residues, prefix, &limb, 1) == RNS8_SUCCESS);
  const auto expected = signed_twos_complement_limbs(-half, 1);
  CHECK(limb == expected[0]);

  int64_t bounded = 0;
  const boost::multiprecision::cpp_int below_half = half - 1;
  const auto bounded_residues = residues_for(below_half, prefix);
  CHECK(rns8::detail::reconstruct_signed(
            bounded_residues, prefix, static_cast<uint64_t>(below_half), bounded) == RNS8_SUCCESS);
  CHECK(boost::multiprecision::cpp_int(bounded) == below_half);
}

TEST_CASE("exact-wide signed max-prefix half-product export is fixed-width two's-complement") {
  constexpr uint32_t prefix = RNS8_MAX_SUPPORTED_PREFIX;
  const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(prefix);
  const boost::multiprecision::cpp_int half = product / 2;
  const auto residues = residues_for(half, prefix);

  constexpr uint64_t sentinel = 0xa3a3a3a3a3a3a3a3ull;
  uint64_t too_few[2] = {sentinel, sentinel};
  CHECK(rns8::detail::export_exact_wide_signed_limbs(residues, prefix, too_few, 2) == RNS8_RANGE_ERROR);
  CHECK(too_few[0] == sentinel);
  CHECK(too_few[1] == sentinel);

  uint64_t limbs[3] = {};
  REQUIRE(rns8::detail::export_exact_wide_signed_limbs(residues, prefix, limbs, 3) == RNS8_SUCCESS);
  const auto expected = signed_twos_complement_limbs(-half, 3);
  CHECK(std::vector<uint64_t>(limbs, limbs + 3) == expected);
}

TEST_CASE("exact-wide signed fixed-width limb export covers one-limb boundaries") {
  constexpr uint32_t prefix = RNS8_MAX_SUPPORTED_PREFIX;
  uint64_t limbs[32] = {};

  auto export_signed = [&](boost::multiprecision::cpp_int value, uint32_t limb_count) {
    const auto residues = residues_for(value, prefix);
    return rns8::detail::export_exact_wide_signed_limbs(residues, prefix, limbs, limb_count);
  };

  REQUIRE(export_signed(boost::multiprecision::cpp_int(std::numeric_limits<int64_t>::max()), 1) == RNS8_SUCCESS);
  CHECK(limbs[0] == 0x7fffffffffffffffull);

  REQUIRE(export_signed(boost::multiprecision::cpp_int(std::numeric_limits<int64_t>::min()), 1) == RNS8_SUCCESS);
  CHECK(limbs[0] == 0x8000000000000000ull);

  REQUIRE(export_signed(boost::multiprecision::cpp_int(-1), 1) == RNS8_SUCCESS);
  CHECK(limbs[0] == std::numeric_limits<uint64_t>::max());

  constexpr uint64_t sentinel = 0x8787878787878787ull;
  for (uint64_t& limb : limbs) {
    limb = sentinel;
  }
  CHECK(export_signed(boost::multiprecision::cpp_int(1) << 63u, 1) == RNS8_RANGE_ERROR);
  CHECK(limbs[0] == sentinel);

  for (uint64_t& limb : limbs) {
    limb = sentinel;
  }
  CHECK(export_signed(-((boost::multiprecision::cpp_int(1) << 63u) + 1), 1) == RNS8_RANGE_ERROR);
  CHECK(limbs[0] == sentinel);

  REQUIRE(export_signed(boost::multiprecision::cpp_int(-1), 32) == RNS8_SUCCESS);
  for (uint64_t limb : limbs) {
    CHECK(limb == std::numeric_limits<uint64_t>::max());
  }
}

TEST_CASE("exact-wide unsigned fixed-width limb export covers overflow and max width") {
  constexpr uint32_t prefix = RNS8_MAX_SUPPORTED_PREFIX;
  uint64_t limbs[32] = {};

  auto export_unsigned = [&](boost::multiprecision::cpp_int value, uint32_t limb_count) {
    const auto residues = residues_for(value, prefix);
    return rns8::detail::export_exact_wide_unsigned_limbs(residues, prefix, limbs, limb_count);
  };

  REQUIRE(export_unsigned(boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()), 1) == RNS8_SUCCESS);
  CHECK(limbs[0] == std::numeric_limits<uint64_t>::max());

  constexpr uint64_t sentinel = 0x9696969696969696ull;
  for (uint64_t& limb : limbs) {
    limb = sentinel;
  }
  CHECK(export_unsigned(boost::multiprecision::cpp_int(1) << 64u, 1) == RNS8_RANGE_ERROR);
  CHECK(limbs[0] == sentinel);

  REQUIRE(export_unsigned(boost::multiprecision::cpp_int(1) << 64u, 2) == RNS8_SUCCESS);
  CHECK(limbs[0] == 0);
  CHECK(limbs[1] == 1);

  REQUIRE(export_unsigned(boost::multiprecision::cpp_int(1), 32) == RNS8_SUCCESS);
  CHECK(limbs[0] == 1);
  for (uint32_t limb = 1; limb < 32; ++limb) {
    CHECK(limbs[limb] == 0);
  }
}

TEST_CASE("exact-wide unsigned high-bit magnitude export preserves fixed width") {
  constexpr uint32_t prefix = RNS8_MAX_SUPPORTED_PREFIX;
  const boost::multiprecision::cpp_int value = boost::multiprecision::cpp_int(1) << 127u;
  const auto residues = residues_for(value, prefix);

  constexpr uint64_t sentinel = 0xb4b4b4b4b4b4b4b4ull;
  uint64_t too_few[1] = {sentinel};
  CHECK(rns8::detail::export_exact_wide_unsigned_limbs(residues, prefix, too_few, 1) == RNS8_RANGE_ERROR);
  CHECK(too_few[0] == sentinel);

  uint64_t two_limbs[2] = {};
  REQUIRE(rns8::detail::export_exact_wide_unsigned_limbs(residues, prefix, two_limbs, 2) == RNS8_SUCCESS);
  CHECK(two_limbs[0] == 0);
  CHECK(two_limbs[1] == (uint64_t{1} << 63u));

  uint64_t wide[32] = {};
  REQUIRE(rns8::detail::export_exact_wide_unsigned_limbs(residues, prefix, wide, 32) == RNS8_SUCCESS);
  CHECK(wide[0] == 0);
  CHECK(wide[1] == (uint64_t{1} << 63u));
  for (uint32_t limb = 2; limb < 32; ++limb) {
    CHECK(wide[limb] == 0);
  }
}

