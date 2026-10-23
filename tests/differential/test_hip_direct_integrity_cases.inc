TEST_CASE("HIP independent native checks detect corruption and disclose "
          "invisible errors") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  if (!hip_available())
    SKIP("no HIP device for integrity sentinel");
  for (bool unsigned_inputs : {false, true}) {
    const uint64_t av = 2, bv = 3, bound[3] = {100000, 0, 0};
    IntegrityIdentity identity{1, 1, 7, 2, 8, 3, 1, 3};
    void *a = nullptr, *b = nullptr, *c = nullptr;
    REQUIRE(hip_direct_allocate(0, 8, &a) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, 8, &b) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, 3, &c) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, a, &av, 8) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, b, &bv, 8) == RNS8_SUCCESS);
    uint32_t flags = 0;
    for (int value : {6, 7, 6 + 257, 6 + 263, 6 + 257 * 263}) {
      int8_t residues[3];
      for (int p = 0; p < 3; ++p)
        residues[p] = centered_residue(cpp_int(value), kDefaultModuli[p]);
      REQUIRE(hip_direct_copy_host_to_device(0, c, residues, 3) ==
              RNS8_SUCCESS);
      const auto status = hip_direct_check_native_product_integrity(
          0, a, b, c, 1, 1, 1, 1, 1, unsigned_inputs, bound, identity, identity,
          flags);
      const bool invisible = value == 6 || value == 6 + 257 * 263;
      REQUIRE(status == (invisible ? RNS8_SUCCESS : RNS8_VERIFICATION_FAILED));
      REQUIRE((flags == 0) == invisible);
    }
    int8_t correct[3];
    for (int p = 0; p < 3; ++p)
      correct[p] = centered_residue(cpp_int(6), kDefaultModuli[p]);
    for (int p = 0; p < 3; ++p) {
      int8_t changed[3] = {correct[0], correct[1], correct[2]};
      changed[p] = centered_residue(cpp_int(7), kDefaultModuli[p]);
      REQUIRE(hip_direct_copy_host_to_device(0, c, changed, 3) == RNS8_SUCCESS);
      REQUIRE(hip_direct_check_native_product_integrity(
                  0, a, b, c, 1, 1, 1, 1, 1, unsigned_inputs, bound, identity,
                  identity, flags) == RNS8_VERIFICATION_FAILED);
    }
    REQUIRE(hip_direct_copy_host_to_device(0, c, correct, 3) == RNS8_SUCCESS);
    auto stale = identity;
    ++stale.a_generation;
    REQUIRE(hip_direct_check_native_product_integrity(
                0, a, b, c, 1, 1, 1, 1, 1, unsigned_inputs, bound, identity,
                stale, flags) == RNS8_INVALID_ARGUMENT);
    const uint64_t too_small[3] = {5, 0, 0};
    REQUIRE(hip_direct_check_native_product_integrity(
                0, a, b, c, 1, 1, 1, 1, 1, unsigned_inputs, too_small, identity,
                identity, flags) == RNS8_VERIFICATION_FAILED);
    REQUIRE((flags & 2) != 0);
    correct[1] = -128;
    REQUIRE(hip_direct_copy_host_to_device(0, c, correct, 3) == RNS8_SUCCESS);
    REQUIRE(hip_direct_check_native_product_integrity(
                0, a, b, c, 1, 1, 1, 1, 1, unsigned_inputs, bound, identity,
                identity, flags) == RNS8_VERIFICATION_FAILED);
    REQUIRE(flags == 1);
    CHECK(hip_direct_free(0, c) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, b) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, a) == RNS8_SUCCESS);
  }
}

TEST_CASE("HIP independent checks handle full-width signed and unsigned native "
          "operands") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  if (!hip_available())
    SKIP("no HIP device for full-width integrity");
  for (bool unsigned_inputs : {false, true}) {
    const uint64_t av[2] = {uint64_t{1} << 63, UINT64_MAX},
                   bv[2] = {UINT64_MAX, uint64_t{1} << 63};
    cpp_int expected = 0;
    for (int q = 0; q < 2; ++q) {
      cpp_int x = av[q], y = bv[q];
      if (!unsigned_inputs) {
        if (av[q] >> 63)
          x -= cpp_int(1) << 64;
        if (bv[q] >> 63)
          y -= cpp_int(1) << 64;
      }
      expected += x * y;
    }
    int8_t residues[20];
    for (int p = 0; p < 20; ++p)
      residues[p] = centered_residue(expected, kDefaultModuli[p]);
    const auto bound = native_tile_max_bound(UINT64_MAX, UINT64_MAX, 2);
    IntegrityIdentity identity{3, 2, 1, 0, 2, 0, 1, 20};
    void *a = nullptr, *b = nullptr, *c = nullptr;
    REQUIRE(hip_direct_allocate(0, sizeof(av), &a) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, sizeof(bv), &b) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, sizeof(residues), &c) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, a, av, sizeof(av)) ==
            RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, b, bv, sizeof(bv)) ==
            RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, c, residues, sizeof(residues)) ==
            RNS8_SUCCESS);
    uint32_t flags = 99;
    REQUIRE(hip_direct_check_native_product_integrity(
                0, a, b, c, 1, 1, 2, 2, 1, unsigned_inputs, bound.magnitude,
                identity, identity, flags) == RNS8_SUCCESS);
    REQUIRE(flags == 0);
    CHECK(hip_direct_free(0, c) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, b) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, a) == RNS8_SUCCESS);
  }
}
