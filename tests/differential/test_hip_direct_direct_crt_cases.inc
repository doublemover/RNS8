TEST_CASE(
    "direct HIP canonical CRT candidates match exact prefix representatives") {
  if (!hip_available())
    SKIP("no HIP device for direct CRT differential");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr std::size_t cells = 257;
  std::mt19937_64 rng(3901);
  for (uint32_t prefix = 1; prefix <= 20; ++prefix) {
    const cpp_int product = modulus_product(prefix);
    std::vector<cpp_int> values(cells);
    values[0] = 0;
    values[1] = 1;
    values[2] = product / 2;
    values[3] = product - 1;
    for (std::size_t i = 4; i < cells; ++i)
      values[i] =
          (cpp_int(rng()) + (cpp_int(rng()) << 64) + (cpp_int(rng()) << 128)) %
          product;
    std::vector<int8_t> residues(cells * prefix);
    for (uint32_t p = 0; p < prefix; ++p)
      for (std::size_t cell = 0; cell < cells; ++cell)
        residues[p * cells + cell] =
            centered_residue(values[cell], kDefaultModuli[p]);
    std::vector<uint64_t> limbs(cells * 3 + 3, UINT64_MAX);
    void *dr = nullptr, *dl = nullptr;
    REQUIRE(hip_direct_allocate(0, residues.size(), &dr) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, limbs.size() * sizeof(uint64_t), &dl) ==
            RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, dr, residues.data(),
                                           residues.size()) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, dl, limbs.data(),
                                           limbs.size() * sizeof(uint64_t)) ==
            RNS8_SUCCESS);
    for (const bool balanced : {false, true}) {
      std::fill(limbs.begin(), limbs.end(), UINT64_MAX);
      REQUIRE(hip_direct_copy_host_to_device(0, dl, limbs.data(),
                                             limbs.size() * sizeof(uint64_t)) ==
              RNS8_SUCCESS);
      REQUIRE((balanced ? hip_direct_export_canonical_balanced_crt_device(
                              0, dr, dl, cells, prefix)
                        : hip_direct_export_canonical_direct_crt_device(
                              0, dr, dl, cells, prefix)) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_device_to_host(0, limbs.data(), dl,
                                             limbs.size() * sizeof(uint64_t)) ==
              RNS8_SUCCESS);
      for (std::size_t cell = 0; cell < cells; ++cell) {
        const cpp_int actual = cpp_int(limbs[cell * 3]) +
                               (cpp_int(limbs[cell * 3 + 1]) << 64) +
                               (cpp_int(limbs[cell * 3 + 2]) << 128);
        REQUIRE(actual == values[cell]);
      }
      for (std::size_t i = cells * 3; i < limbs.size(); ++i)
        REQUIRE(limbs[i] == UINT64_MAX);
    }
    CHECK(hip_direct_free(0, dl) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, dr) == RNS8_SUCCESS);
  }
}
