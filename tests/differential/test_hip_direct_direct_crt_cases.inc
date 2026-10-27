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

TEST_CASE("descriptor CRT consumes tile-major and padded residues without conversion") {
  if (!hip_available()) SKIP("no HIP device for descriptor CRT");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr uint64_t rows=17, cols=19, cells=rows*cols;
  for (uint32_t prefix=1; prefix<=20; ++prefix) {
    const cpp_int product=modulus_product(prefix);
    std::vector<cpp_int> values(cells);
    for (uint64_t cell=0; cell<cells; ++cell)
      values[cell] = cell%3==0 ? cpp_int(product-1) : cell%3==1 ? cpp_int(product/2) : cpp_int(cell)%product;
    for (auto kind : {ResidueLayoutKind::modulus_major, ResidueLayoutKind::tile_major}) {
      ResidueLayout layout;
      const bool tiled=kind==ResidueLayoutKind::tile_major;
      REQUIRE(make_residue_layout(kind,rows,cols,tiled?cols:cols+3,prefix,tiled?4:0,tiled?8:0,layout));
      std::vector<int8_t> residues(layout.bytes,99);
      for (uint32_t p=0; p<prefix; ++p) for (uint64_t row=0; row<rows; ++row)
        for (uint64_t col=0; col<cols; ++col)
          residues[layout.offset(p,row,col)]=centered_residue(values[row*cols+col],kDefaultModuli[p]);
      hip_direct_device_buffer input,output;
      std::vector<uint64_t> limbs(cells*3+3,UINT64_MAX);
      REQUIRE(input.allocate(0,residues.size()) == RNS8_SUCCESS);
      REQUIRE(output.allocate(0,limbs.size()*8) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,input.get(),residues.data(),residues.size()) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,output.get(),limbs.data(),limbs.size()*8) == RNS8_SUCCESS);
      auto bad=layout; ++bad.basis_version;
      CHECK(hip_direct_export_canonical_balanced_crt_layout_device(0,input.get(),bad,output.get()) == RNS8_INVALID_ARGUMENT);
      CHECK(hip_direct_export_canonical_balanced_crt_layout_device(0,input.get(),layout,input.get()) == RNS8_INVALID_ARGUMENT);
      REQUIRE(hip_direct_copy_device_to_host(0,limbs.data(),output.get(),limbs.size()*8) == RNS8_SUCCESS);
      for (auto value:limbs) REQUIRE(value==UINT64_MAX);
      REQUIRE(hip_direct_export_canonical_balanced_crt_layout_device(0,input.get(),layout,output.get()) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_device_to_host(0,limbs.data(),output.get(),limbs.size()*8) == RNS8_SUCCESS);
      for (uint64_t cell=0; cell<cells; ++cell) {
        const cpp_int actual=cpp_int(limbs[cell*3])+(cpp_int(limbs[cell*3+1])<<64)+(cpp_int(limbs[cell*3+2])<<128);
        REQUIRE(actual==values[cell]);
      }
      for (std::size_t i=cells*3; i<limbs.size(); ++i) REQUIRE(limbs[i]==UINT64_MAX);
      std::vector<int8_t> after(residues.size());
      REQUIRE(hip_direct_copy_device_to_host(0,after.data(),input.get(),after.size()) == RNS8_SUCCESS);
      REQUIRE(after==residues);
    }
  }
}
