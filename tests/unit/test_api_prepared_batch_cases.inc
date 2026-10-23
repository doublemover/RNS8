TEST_CASE("HIP batches admit all members before reserving shared readers",
          "[api][async][group]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (bool signed_values : {false, true}) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    rns8_context *ctx = nullptr;
    REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
    const auto semantics =
        signed_values ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED;
    auto desc = exact_desc(semantics, 2, 2, 2);
    desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    auto md = exact_matrix_desc(2, 2, semantics);
    rns8_plan *plan = nullptr;
    rns8_workspace *work[2]{};
    rns8_matrix *a = nullptr, *d = nullptr, *c[3]{};
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    for (auto &w : work)
      REQUIRE(rns8_create_workspace(ctx, plan, &w) == RNS8_SUCCESS);
    for (auto **m : {&a, &d, &c[0], &c[1], &c[2]})
      REQUIRE(rns8_create_matrix(ctx, &md, m) == RNS8_SUCCESS);
    const int64_t av[]{1, 2, 3, 4}, dv[]{5, 6, 7, 8};
    const auto pack = [&](rns8_matrix *m, const int64_t *values) {
      if (signed_values)
        return rns8_pack_i64(ctx, m, values, 2, 0);
      const std::vector<uint64_t> converted(values, values + 4);
      return rns8_pack_u64(ctx, m, converted.data(), 2, 0);
    };
    REQUIRE(pack(a, av) == RNS8_SUCCESS);
    REQUIRE(pack(d, dv) == RNS8_SUCCESS);
    const int64_t alpha = signed_values ? INT64_MIN : INT64_MAX;
    rns8_prepared_execution *e[3]{};
    REQUIRE(rns8_prepare_gemm(ctx, plan, a, a, c[0], work[0], &e[0]) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_prepare_exact_axpby(ctx, plan, a, a, d, c[1], alpha, INT64_MAX,
                                     work[1], &e[1]) == RNS8_SUCCESS);
    REQUIRE(rns8_prepare_gemm(ctx, plan, a, a, c[2], work[0], &e[2]) ==
            RNS8_SUCCESS);
    for (auto *execution : e)
      REQUIRE(rns8_capture_prepared_hip_graph(execution) == RNS8_SUCCESS);
    rns8_prepared_group *group = nullptr, *collision = nullptr;
    rns8_prepared_execution *overlapping[]{e[0], e[2]};
    REQUIRE(rns8_create_prepared_group(e, 2, &group) == RNS8_SUCCESS);
    REQUIRE(rns8_create_prepared_group(overlapping, 2, &collision) ==
            RNS8_SUCCESS);
    hip_unique_stream s1, s2;
    REQUIRE(s1.create_non_blocking() == hipSuccess);
    REQUIRE(s2.create_non_blocking() == hipSuccess);
    void *streams[]{s1.get(), s2.get()};
    rns8_completion_token tokens[2]{{77, 88}, {77, 88}};
    uint32_t submitted = 999;
    for (bool graph : {false, true}) {
      const auto submit = [&](rns8_prepared_group *target) {
        return graph ? rns8_submit_prepared_group_hip_graph(
                           target, streams, 2, tokens, 2, &submitted)
                     : rns8_submit_prepared_group_hip(target, streams, 2,
                                                      tokens, 2, &submitted);
      };
      const auto before0 = c[0]->content_generation,
                 before1 = c[1]->content_generation;
      const auto old_token = tokens[0];
      streams[1] = nullptr;
      REQUIRE(submit(group) == RNS8_INVALID_ARGUMENT);
      REQUIRE(submitted == 0);
      streams[1] = s2.get();
      REQUIRE(submit(collision) == RNS8_INVALID_ARGUMENT);
      REQUIRE(submitted == 0);
      a->in_flight_readers = UINT64_MAX - 1;
      REQUIRE(submit(group) == RNS8_RANGE_ERROR);
      REQUIRE(submitted == 0);
      a->in_flight_readers = 0;
      REQUIRE(tokens[0].prepared_id == old_token.prepared_id);
      REQUIRE(tokens[0].epoch == old_token.epoch);
      REQUIRE(c[0]->content_generation == before0);
      REQUIRE(c[1]->content_generation == before1);
      REQUIRE_FALSE(work[0]->in_flight);
      REQUIRE_FALSE(work[1]->in_flight);
      REQUIRE(submit(group) == RNS8_SUCCESS);
      REQUIRE(submitted == 2);
      REQUIRE(a->in_flight_readers == 2);
      REQUIRE(d->in_flight_readers == 1);
      REQUIRE(rns8_destroy_prepared_group(group) == RNS8_BUSY);
      REQUIRE(rns8_refresh_prepared_group_inputs(group) == RNS8_BUSY);
      REQUIRE_FALSE(c[0]->device_residues_current);
      REQUIRE_FALSE(c[1]->device_residues_current);
      rns8_completion_info info{};
      info.struct_size = sizeof(info);
      info.abi_version = RNS8_ABI_VERSION;
      // Observe the second stream first: its event must release only its
      // readers.
      for (int i : {1, 0}) {
        REQUIRE(rns8_wait_prepared_hip(e[i], tokens[i], &info) == RNS8_SUCCESS);
        REQUIRE(info.operation_status == RNS8_SUCCESS);
        REQUIRE(info.output_is_current);
        REQUIRE(a->in_flight_readers == static_cast<uint64_t>(i));
        REQUIRE(d->in_flight_readers == 0);
        uint64_t limbs[12]{};
        REQUIRE((signed_values
                     ? rns8_export_exact_wide_signed_limbs(ctx, plan, c[i],
                                                           limbs, 2, 3)
                     : rns8_export_exact_wide_unsigned_limbs(
                           ctx, plan, c[i], limbs, 2, 3)) == RNS8_SUCCESS);
        for (int row = 0; row < 2; ++row)
          for (int col = 0; col < 2; ++col) {
            const int cell = row * 2 + col;
            cpp_int expected = 0;
            for (int q = 0; q < 2; ++q)
              expected += cpp_int(av[row * 2 + q]) * av[q * 2 + col];
            if (i)
              expected =
                  cpp_int(alpha) * expected + cpp_int(INT64_MAX) * dv[cell];
            cpp_int actual = cpp_int(limbs[3 * cell]) +
                             (cpp_int(limbs[3 * cell + 1]) << 64) +
                             (cpp_int(limbs[3 * cell + 2]) << 128);
            if (signed_values && limbs[3 * cell + 2] >> 63)
              actual -= cpp_int(1) << 192;
            REQUIRE(actual == expected);
          }
      }
      // The first member is admissible, but the scaled second product is not.
      const int64_t huge[]{INT64_MAX, INT64_MAX, INT64_MAX, INT64_MAX};
      REQUIRE(pack(a, huge) == RNS8_SUCCESS);
      REQUIRE(rns8_refresh_prepared_group_inputs(group) == RNS8_SUCCESS);
      const auto completed0 = c[0]->content_generation;
      const auto completed1 = c[1]->content_generation;
      const auto previous = tokens[0];
      REQUIRE(submit(group) == RNS8_RANGE_ERROR);
      REQUIRE(submitted == 0);
      REQUIRE(tokens[0].epoch == previous.epoch);
      REQUIRE(c[0]->content_generation == completed0);
      REQUIRE(c[1]->content_generation == completed1);
      REQUIRE(c[0]->device_residues_current);
      REQUIRE(pack(a, av) == RNS8_SUCCESS);
      REQUIRE(rns8_refresh_prepared_group_inputs(group) == RNS8_SUCCESS);
      REQUIRE(rns8_refresh_prepared_group_inputs(collision) == RNS8_SUCCESS);
    }
    REQUIRE(rns8_destroy_prepared_group(collision) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_prepared_group(group) == RNS8_SUCCESS);
    for (auto *execution : e)
      REQUIRE(rns8_destroy_prepared_execution(execution) == RNS8_SUCCESS);
    for (auto *m : {a, d, c[0], c[1], c[2]})
      REQUIRE(rns8_destroy_matrix(m) == RNS8_SUCCESS);
    for (auto *w : work)
      REQUIRE(rns8_destroy_workspace(w) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  }
#else
  SKIP("HIP batch submission unavailable in this build");
#endif
}
