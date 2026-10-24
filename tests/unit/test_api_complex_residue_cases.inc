TEST_CASE("internal complex owner binds both proofs and publishes paired generations",
          "[api][complex][hip]") {
  using namespace rns8::detail;
  using namespace rns8::detail::api;
  for (bool hip : {false, true}) {
    if (hip && !direct_hip_compiled())
      continue;
    rns8_context *ctx = nullptr;
    if (hip) {
      rns8_context_options options{};
      options.struct_size = sizeof(options);
      options.abi_version = RNS8_ABI_VERSION;
      options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
    } else
      ctx = create_cpu_context();
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 2, 2, 2);
    if (hip)
      desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    rns8_plan *plan = nullptr;
    rns8_workspace *workspace = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    const auto md = exact_matrix_desc(2, 2, RNS8_EXACT_WIDE_SIGNED);
    rns8_matrix *ar = nullptr, *ai = nullptr, *br = nullptr, *bi = nullptr, *cr = nullptr,
                *ci = nullptr;
    for (auto **value : {&ar, &ai, &br, &bi, &cr, &ci})
      REQUIRE(rns8_create_matrix(ctx, &md, value) == RNS8_SUCCESS);
    std::array<std::vector<int64_t>, 4> values{{{INT64_MIN, INT64_MAX, 0, -7},
                                                {4, -9, INT64_MAX, INT64_MIN},
                                                {2, -3, 7, 11},
                                                {-5, 1, 0, -6}}};
    const std::array<rns8_matrix *, 4> inputs{ar, ai, br, bi};
    for (std::size_t i = 0; i < inputs.size(); ++i)
      REQUIRE(rns8_pack_i64(ctx, inputs[i], values[i].data(), 2, 1) == RNS8_SUCCESS);
    const auto exported = [&](rns8_matrix *matrix) {
      std::array<uint64_t, 12> limbs{};
      REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, matrix, limbs.data(), 2, 3) ==
              RNS8_SUCCESS);
      std::vector<cpp_int> result;
      for (std::size_t cell = 0; cell < 4; ++cell) {
        cpp_int value = cpp_int(limbs[cell * 3]) + (cpp_int(limbs[cell * 3 + 1]) << 64) +
                        (cpp_int(limbs[cell * 3 + 2]) << 128);
        if (limbs[cell * 3 + 2] >> 63)
          value -= cpp_int(1) << 192;
        result.push_back(value);
      }
      return result;
    };
    for (auto method : {ComplexProductMethod::three_real, ComplexProductMethod::four_real}) {
      for (int changed = 0; changed < 2; ++changed) {
        if (changed) {
          ++values[1][0];
          REQUIRE(rns8_pack_i64(ctx, ai, values[1].data(), 2, 0) == RNS8_SUCCESS);
        }
        const auto rg = cr->content_generation, ig = ci->content_generation;
        REQUIRE(complex_exact_signed_gemm(ctx, plan, {ar, ai}, {br, bi}, {cr, ci}, method,
                                          UINT64_MAX, workspace) == RNS8_SUCCESS);
        REQUIRE(cr->content_generation == rg + 1);
        REQUIRE(ci->content_generation == ig + 1);
        REQUIRE(cr->exact_range_prefix == plan->prefix);
        REQUIRE(ci->exact_range_prefix == plan->prefix);
        REQUIRE_FALSE(cr->in_flight_writer);
        REQUIRE_FALSE(ci->in_flight_writer);
        REQUIRE_FALSE(workspace->in_flight);
        for (auto *input : inputs)
          REQUIRE(input->in_flight_readers == 0);
        const auto real = exported(cr), imag = exported(ci);
        for (int row = 0; row < 2; ++row)
          for (int col = 0; col < 2; ++col) {
            cpp_int expected_r = 0, expected_i = 0;
            for (int q = 0; q < 2; ++q) {
              expected_r += cpp_int(values[0][row * 2 + q]) * values[2][q * 2 + col] -
                            cpp_int(values[1][row * 2 + q]) * values[3][q * 2 + col];
              expected_i += cpp_int(values[0][row * 2 + q]) * values[3][q * 2 + col] +
                            cpp_int(values[1][row * 2 + q]) * values[2][q * 2 + col];
            }
            REQUIRE(real[row * 2 + col] == expected_r);
            REQUIRE(imag[row * 2 + col] == expected_i);
            REQUIRE(boost::multiprecision::abs(expected_r) <= cr->exact_max_magnitude);
            REQUIRE(boost::multiprecision::abs(expected_i) <= ci->exact_max_magnitude);
          }
      }
    }
    const auto old_r = exported(cr), old_i = exported(ci);
    const auto rg = cr->content_generation, ig = ci->content_generation;
    const auto call = [&](uint64_t budget = UINT64_MAX) {
      return complex_exact_signed_gemm(ctx, plan, {ar, ai}, {br, bi}, {cr, ci},
                                       ComplexProductMethod::three_real, budget, workspace);
    };
    CHECK(call(0) == RNS8_WORKSPACE_TOO_SMALL);
    for (auto *input : inputs) {
      CHECK(complex_exact_signed_gemm(ctx, plan, {ar, ai}, {br, bi}, {input, ci},
                                      ComplexProductMethod::three_real, UINT64_MAX,
                                      workspace) == RNS8_INVALID_ARGUMENT);
      CHECK(complex_exact_signed_gemm(ctx, plan, {ar, ai}, {br, bi}, {cr, input},
                                      ComplexProductMethod::three_real, UINT64_MAX,
                                      workspace) == RNS8_INVALID_ARGUMENT);
      input->in_flight_writer = true;
      CHECK(call() == RNS8_BUSY);
      input->in_flight_writer = false;
      input->in_flight_readers = UINT64_MAX;
      CHECK(call() == RNS8_BUSY);
      input->in_flight_readers = 0;
      const auto prefix = input->exact_range_prefix;
      input->exact_range_prefix = 0;
      CHECK(call() == RNS8_RANGE_ERROR);
      input->exact_range_prefix = prefix;
    }
    CHECK(complex_exact_signed_gemm(ctx, plan, {ar, ai}, {br, bi}, {cr, cr},
                                    ComplexProductMethod::three_real, UINT64_MAX,
                                    workspace) == RNS8_INVALID_ARGUMENT);
    for (auto *output : {cr, ci}) {
      output->in_flight_readers = 1;
      CHECK(call() == RNS8_BUSY);
      output->in_flight_readers = 0;
      output->in_flight_writer = true;
      CHECK(call() == RNS8_BUSY);
      output->in_flight_writer = false;
      output->generation_exhausted = true;
      CHECK(call() == RNS8_RANGE_ERROR);
      output->generation_exhausted = false;
    }
    workspace->in_flight = true;
    CHECK(call() == RNS8_BUSY);
    workspace->in_flight = false;
    REQUIRE(exported(cr) == old_r);
    REQUIRE(exported(ci) == old_i);
    REQUIRE(cr->content_generation == rg);
    REQUIRE(ci->content_generation == ig);
    // Legal shared readers are reserved once, and real/imaginary bounds remain
    // independent even when both components reference one input handle.
    REQUIRE(complex_exact_signed_gemm(ctx, plan, {ar, ar}, {br, br}, {cr, ci},
                                      ComplexProductMethod::three_real, UINT64_MAX,
                                      workspace) == RNS8_SUCCESS);
    REQUIRE(exported(cr) == std::vector<cpp_int>(4, 0));
    REQUIRE(ar->in_flight_readers == 0);
    REQUIRE(br->in_flight_readers == 0);
    // Produce genuine resident >64-bit inputs, then reject an unrepresentable
    // full complex expression without publishing either component.
    const std::vector<int64_t> large(4, int64_t{1} << 40);
    REQUIRE(rns8_pack_i64(ctx, ar, large.data(), 2, 0) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, br, large.data(), 2, 0) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, plan, ar, br, ai, workspace) == RNS8_SUCCESS);
    REQUIRE(call() == RNS8_SUCCESS);
    const auto wide_r = exported(cr), wide_i = exported(ci);
    for (int row = 0; row < 2; ++row)
      for (int col = 0; col < 2; ++col) {
        const cpp_int bi_sum = cpp_int(values[3][col]) + values[3][2 + col];
        REQUIRE(wide_r[row * 2 + col] == (cpp_int(1) << 81) - (cpp_int(1) << 81) * bi_sum);
        REQUIRE(wide_i[row * 2 + col] == (cpp_int(1) << 122) + (cpp_int(1) << 40) * bi_sum);
      }
    REQUIRE(rns8_gemm_rns(ctx, plan, ar, br, bi, workspace) == RNS8_SUCCESS);
    const auto before_r = exported(cr), before_i = exported(ci);
    CHECK(call() == RNS8_RANGE_ERROR);
    REQUIRE(exported(cr) == before_r);
    REQUIRE(exported(ci) == before_i);
    for (auto *matrix : {ar, ai, br, bi, cr, ci})
      REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  }
}
