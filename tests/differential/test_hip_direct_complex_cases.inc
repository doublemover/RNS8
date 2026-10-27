TEST_CASE(
    "resident complex 3M and four-product baseline match Boost for rings and signed extrema") {
  if (!hip_available())
    SKIP("no HIP device for complex residue composition");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  const int64_t extrema[]{0, 1, -1, INT64_MIN, INT64_MAX, 127, -128};
  const auto center = [](const cpp_int &value, uint32_t modulus) {
    int residue = (value % modulus).convert_to<int>();
    if (residue < 0)
      residue += modulus;
    if (residue >= int((modulus + 1) / 2))
      residue -= modulus;
    return static_cast<int8_t>(residue);
  };
  for (uint32_t modulus : {0u, 2u, 4u, 6u, 127u, 251u, 256u})
    for (int shape = 0; shape < 8; ++shape) {
      DYNAMIC_SECTION("complex modulus " << modulus << " shape " << shape) {
        const int64_t m = shape == 2   ? 1
                          : shape == 1 ? 17
                                       : 3,
                      n = shape == 2   ? 1
                          : shape == 1 ? 19
                                       : 5,
                      k = shape == 2   ? 65537
                          : shape == 1 ? 15
                                       : 7;
        const uint32_t prefix = modulus ? 1 : shape == 5 ? 9 : 20;
        const ComplexResidueBasis basis{modulus      ? ComplexResidueDomain::finite_ring
                                        : shape == 5 ? ComplexResidueDomain::bounded_signed
                                                     : ComplexResidueDomain::wide_signed,
                                        1, prefix, modulus};
        const int64_t rows[]{m, m, k, k, m, m}, cols[]{k, k, n, n, n, n},
            ld[]{k + 1, k + 2, n + 1, n + 2, n + 3, n + 4};
        std::array<std::vector<cpp_int>, 4> native;
        std::array<std::vector<int8_t>, 6> planes;
        std::array<hip_direct_device_buffer, 6> buffers;
        for (std::size_t component = 0; component < 6; ++component) {
          planes[component].assign(std::size_t(prefix + 1) * rows[component] * ld[component], 99);
          if (component < 4) {
            native[component].resize(std::size_t(rows[component]) * cols[component]);
            for (int64_t row = 0; row < rows[component]; ++row)
              for (int64_t col = 0; col < cols[component]; ++col) {
                const bool zero_component = shape == 4 || (shape == 1 && component % 2 == 1) ||
                                            (shape == 3 && component % 2 == 0);
                const auto position = (row * 3 + col * 5 + component * 2) % 7;
                cpp_int value = shape == 6       ? INT64_MIN
                                : zero_component ? int64_t{0}
                                : shape == 5     ? int64_t(position) - 3
                                                 : extrema[position];
                if (shape == 7)
                  value = ((row + col + component) % 2 ? -1 : 1) * (cpp_int(1) << 70);
                native[component][row * cols[component] + col] = value;
                for (uint32_t p = 0; p < prefix; ++p)
                  planes[component][(p * rows[component] + row) * ld[component] + col] =
                      center(cpp_int(value), modulus ? modulus : rns8_default_modulus(p));
              }
          }
          REQUIRE(buffers[component].allocate(0, planes[component].size()) == RNS8_SUCCESS);
          REQUIRE(hip_direct_copy_host_to_device(0, buffers[component].get(),
                                                 planes[component].data(),
                                                 planes[component].size()) == RNS8_SUCCESS);
        }
        auto expected_real = planes[4], expected_imag = planes[5];
        std::array<std::vector<cpp_int>, 2> exact{
            {std::vector<cpp_int>(std::size_t(m) * n), std::vector<cpp_int>(std::size_t(m) * n)}};
        for (int64_t row = 0; row < m; ++row)
          for (int64_t col = 0; col < n; ++col) {
            cpp_int real = 0, imag = 0;
            for (int64_t q = 0; q < k; ++q) {
              const cpp_int ar = native[0][row * k + q], ai = native[1][row * k + q],
                            br = native[2][q * n + col], bi = native[3][q * n + col];
              real += ar * br - ai * bi;
              imag += ar * bi + ai * br;
            }
            exact[0][row * n + col] = real;
            exact[1][row * n + col] = imag;
            for (uint32_t p = 0; p < prefix; ++p) {
              const uint32_t ring = modulus ? modulus : rns8_default_modulus(p);
              expected_real[(p * m + row) * ld[4] + col] = center(real, ring);
              expected_imag[(p * m + row) * ld[5] + col] = center(imag, ring);
            }
          }
        const auto input = [&](std::size_t component) {
          return ComplexResidueView<const int8_t>{
              static_cast<const int8_t *>(buffers[component].get()), rows[component],
              cols[component], ld[component], basis};
        };
        const ComplexResiduePair<const int8_t> a{input(0), input(1)}, b{input(2), input(3)};
        const ComplexResiduePair<int8_t> c{
            {static_cast<int8_t *>(buffers[4].get()), m, n, ld[4], basis},
            {static_cast<int8_t *>(buffers[5].get()), m, n, ld[5], basis}};
        const cpp_int maximum = shape == 5 ? cpp_int(8) : cpp_int(1) << (shape == 7 ? 70 : 63);
        const ComplexMagnitudeBounds bounds{maximum, maximum, maximum, maximum};
        for (const auto method :
             {ComplexProductMethod::three_real, ComplexProductMethod::four_real}) {
          REQUIRE(hip_direct_copy_host_to_device(0, buffers[4].get(), planes[4].data(),
                                                 planes[4].size()) == RNS8_SUCCESS);
          REQUIRE(hip_direct_copy_host_to_device(0, buffers[5].get(), planes[5].data(),
                                                 planes[5].size()) == RNS8_SUCCESS);
          bool committed = true;
          CHECK(hip_direct_gemm_complex_residues(0, a, b, c, bounds, method, 0, committed) ==
                RNS8_WORKSPACE_TOO_SMALL);
          CHECK_FALSE(committed);
          auto collision = c;
          collision.imag.data = collision.real.data;
          CHECK(hip_direct_gemm_complex_residues(0, a, b, collision, bounds, method, UINT64_MAX,
                                                 committed) == RNS8_INVALID_ARGUMENT);
          CHECK_FALSE(committed);
          for (std::size_t component = 4; component < 6; ++component) {
            std::vector<int8_t> before(planes[component].size());
            REQUIRE(hip_direct_copy_device_to_host(0, before.data(), buffers[component].get(),
                                                   before.size()) == RNS8_SUCCESS);
            REQUIRE(before == planes[component]);
          }
          const auto resources=plan_complex_residue_product(a,b,c,bounds,method,UINT64_MAX);
          REQUIRE(resources.status==RNS8_SUCCESS);
          CHECK(resources.staged_real_offset()==resources.products_offset());
          CHECK(resources.staged_imag_offset()==resources.products_offset()+resources.product_bytes);
          REQUIRE(hip_direct_gemm_complex_residues(0, a, b, c, bounds, method, resources.scratch_bytes,
                                                   committed) == RNS8_SUCCESS);
          REQUIRE(committed);
          for (std::size_t component = 4; component < 6; ++component) {
            std::vector<int8_t> observed(planes[component].size());
            REQUIRE(hip_direct_copy_device_to_host(0, observed.data(), buffers[component].get(),
                                                   observed.size()) == RNS8_SUCCESS);
            REQUIRE(observed == (component == 4 ? expected_real : expected_imag));
            if (!modulus)
              for (int64_t row = 0; row < m; ++row)
                for (int64_t col = 0; col < n; ++col) {
                  std::vector<int8_t> residues(prefix);
                  for (uint32_t p = 0; p < prefix; ++p)
                    residues[p] = observed[(p * m + row) * ld[component] + col];
                  if (shape == 5) {
                    int64_t bounded = 0;
                    REQUIRE(reconstruct_signed(residues, prefix, uint64_t(k) * 128, bounded) ==
                            RNS8_SUCCESS);
                    REQUIRE(cpp_int(bounded) == exact[component - 4][row * n + col]);
                  }
                  uint64_t limbs[3]{};
                  REQUIRE(export_exact_wide_signed_limbs(residues, prefix, limbs, 3) ==
                          RNS8_SUCCESS);
                  cpp_int restored =
                      cpp_int(limbs[0]) + (cpp_int(limbs[1]) << 64) + (cpp_int(limbs[2]) << 128);
                  if (limbs[2] >> 63)
                    restored -= cpp_int(1) << 192;
                  REQUIRE(restored == exact[component - 4][row * n + col]);
                }
          }
        }
      }
    }
}
