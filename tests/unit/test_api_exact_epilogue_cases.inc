TEST_CASE(
    "exact epilogue admits the full expression and preserves rejected outputs",
    "[api][epilogue]") {
  using boost::multiprecision::cpp_int;
  for (bool hip : {false, true}) {
    if (hip && !rns8::detail::api::direct_hip_compiled())
      continue;
    for (bool signed_values : {false, true}) {
      rns8_context *ctx = nullptr;
      if (hip) {
        rns8_context_options options{};
        options.struct_size = sizeof(options);
        options.abi_version = RNS8_ABI_VERSION;
        options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
        REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
      } else
        ctx = create_cpu_context();
      const auto semantics =
          signed_values ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED;
      auto desc = exact_desc(semantics, 2, 2, 2);
      if (hip)
        desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      auto md = exact_matrix_desc(2, 2, semantics);
      rns8_plan *plan = nullptr;
      rns8_workspace *workspace = nullptr;
      REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
      rns8_matrix *a = nullptr, *b = nullptr, *d = nullptr, *c = nullptr;
      for (auto **matrix : {&a, &b, &d, &c})
        REQUIRE(rns8_create_matrix(ctx, &md, matrix) == RNS8_SUCCESS);
      const std::vector<int64_t> av{1, 2, 3, 4}, bv{5, 6, 7, 8},
          dv{9, 10, 11, 12};
      const auto pack = [&](rns8_matrix *matrix,
                            const std::vector<int64_t> &values) {
        if (signed_values)
          return rns8_pack_i64(ctx, matrix, values.data(), 2, 0);
        const std::vector<uint64_t> unsigned_data(values.begin(), values.end());
        return rns8_pack_u64(ctx, matrix, unsigned_data.data(), 2, 0);
      };
      REQUIRE(pack(a, av) == RNS8_SUCCESS);
      REQUIRE(pack(b, bv) == RNS8_SUCCESS);
      REQUIRE(pack(d, dv) == RNS8_SUCCESS);
      std::vector<std::pair<int64_t, int64_t>> coefficients{
          {0, 0}, {1, 0}, {0, 1}, {2, 3}, {255, 256}, {256, 255}, {INT64_MAX, INT64_MAX}};
      if (signed_values) {
        coefficients.emplace_back(INT64_MIN, INT64_MAX);
        coefficients.emplace_back(-3, -7);
      }
      for (const auto [alpha, beta] : coefficients) {
        REQUIRE(rns8_gemm_exact_axpby(ctx, plan, a, b, d, c, alpha, beta,
                                      workspace) == RNS8_SUCCESS);
        rns8_prepared_execution* prepared = nullptr;
        REQUIRE(rns8_prepare_exact_axpby(ctx, plan, a, b, d, c, alpha, beta, workspace, &prepared) == RNS8_SUCCESS);
        REQUIRE(rns8_destroy_matrix(d) == RNS8_BUSY);
        rns8_prepared_execution* conflicting = nullptr;
        REQUIRE(rns8_prepare_gemm(ctx, plan, a, b, d, workspace, &conflicting) == RNS8_SUCCESS);
        rns8_prepared_execution* members[]{prepared, conflicting};
        rns8_prepared_group* rejected = nullptr;
        REQUIRE(rns8_create_prepared_group(members, 2, &rejected) == RNS8_INVALID_ARGUMENT);
        REQUIRE(rejected == nullptr);
        REQUIRE(rns8_destroy_prepared_execution(conflicting) == RNS8_SUCCESS);
        rns8_completion_token token{};
        REQUIRE(rns8_execute_prepared(prepared, &token) == RNS8_SUCCESS);
        rns8_completion_info completion{};
        completion.struct_size = sizeof(completion); completion.abi_version = RNS8_ABI_VERSION;
        REQUIRE(rns8_get_prepared_completion(prepared, token, &completion) == RNS8_SUCCESS);
        REQUIRE(completion.has_exact_epilogue == 1);
        REQUIRE(completion.alpha == alpha);
        REQUIRE(completion.beta == beta);
        REQUIRE(completion.d_instance == d->matrix_instance_id);
        REQUIRE(completion.d_generation == d->content_generation);
        REQUIRE(completion.output_is_current);
        REQUIRE(rns8_destroy_prepared_execution(prepared) == RNS8_SUCCESS);
        uint64_t limbs[12]{};
        REQUIRE((signed_values
                     ? rns8_export_exact_wide_signed_limbs(ctx, plan, c, limbs,
                                                           2, 3)
                     : rns8_export_exact_wide_unsigned_limbs(
                           ctx, plan, c, limbs, 2, 3)) == RNS8_SUCCESS);
        for (int row = 0; row < 2; ++row)
          for (int col = 0; col < 2; ++col) {
            cpp_int sum = 0;
            for (int q = 0; q < 2; ++q)
              sum += cpp_int(av[row * 2 + q]) * bv[q * 2 + col];
            const cpp_int expected =
                cpp_int(alpha) * sum + cpp_int(beta) * dv[row * 2 + col];
            const int cell = row * 2 + col;
            cpp_int actual = cpp_int(limbs[3 * cell]) +
                             (cpp_int(limbs[3 * cell + 1]) << 64) +
                             (cpp_int(limbs[3 * cell + 2]) << 128);
            if (signed_values && (limbs[3 * cell + 2] >> 63))
              actual -= cpp_int(1) << 192;
            REQUIRE(actual == expected);
          }
      }
      const auto generation = c->content_generation;
      // Algebraic zeros do not waive admission of any original operand.
      const auto a_prefix=a->exact_range_prefix;a->exact_range_prefix=0;
      REQUIRE(rns8_gemm_exact_axpby(ctx,plan,a,b,d,c,0,0,workspace)==RNS8_RANGE_ERROR);
      a->exact_range_prefix=a_prefix;CHECK(c->content_generation==generation);
      d->in_flight_writer = true;
      REQUIRE(rns8_gemm_exact_axpby(ctx, plan, a, b, d, c, 1, 1, workspace) == RNS8_BUSY);
      d->in_flight_writer = false;
      REQUIRE(c->content_generation == generation);
      for (auto *alias : {a, b, d})
        REQUIRE(rns8_gemm_exact_axpby(ctx, plan, a, b, d, alias, 1, 1,
                                      workspace) == RNS8_INVALID_ARGUMENT);
      if (!signed_values)
        REQUIRE(rns8_gemm_exact_axpby(ctx, plan, a, b, d, c, -1, 1,
                                      workspace) == RNS8_INVALID_ARGUMENT);
      REQUIRE(c->content_generation == generation);
      std::vector<int64_t> large(4, INT64_MAX);
      REQUIRE(pack(a, large) == RNS8_SUCCESS);
      REQUIRE(pack(b, large) == RNS8_SUCCESS);
      REQUIRE(rns8_gemm_exact_axpby(ctx, plan, a, b, d, c, INT64_MAX, 1,
                                    workspace) == RNS8_RANGE_ERROR);
      REQUIRE(c->content_generation == generation);
      // The complete-expression bound admits alpha=0 even if AB itself would
      // exceed the root basis. No intermediate integer result is published.
      REQUIRE(rns8_gemm_exact_axpby(ctx, plan, a, b, d, c, 0, 1, workspace) ==
              RNS8_SUCCESS);
      for (auto *matrix : {a, b, d, c})
        REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
  }
}
