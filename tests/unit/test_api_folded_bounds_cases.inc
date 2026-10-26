namespace {
void check_folded_tile_bounds(rns8_backend_kind backend) {
  rns8_context_options options{};
  options.struct_size = sizeof(options); options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = backend;
  rns8_context* raw_context = nullptr;
  REQUIRE(rns8_create_context(backend == RNS8_BACKEND_CPU_REFERENCE ? -1 : 0,
                              &options, &raw_context) == RNS8_SUCCESS);
  std::unique_ptr<rns8_context, decltype(&rns8_destroy_context)> ctx(raw_context, rns8_destroy_context);
  for (bool is_unsigned : {false, true}) for (int contract = 0; contract < 4; ++contract) {
    CAPTURE(is_unsigned, contract);
    const uint64_t limit = is_unsigned ? UINT64_MAX : static_cast<uint64_t>(INT64_MAX);
    std::vector<uint64_t> bounds = contract == 0 ? std::vector<uint64_t>{0, 0}
                                : contract == 1 ? std::vector<uint64_t>{1, 1}
                                : contract == 2 ? std::vector<uint64_t>{limit, limit}
                                                : std::vector<uint64_t>{1, 2};
    const auto semantics = is_unsigned ? RNS8_BOUNDED_U64 : RNS8_BOUNDED_I64;
    auto desc = bounded_gemm_desc(semantics, 65, 3, 1);
    desc.requested_backend = backend; desc.bound = 0;
    desc.bound_kind = is_unsigned ? RNS8_BOUND_PER_TILE_MAX_UNSIGNED : RNS8_BOUND_PER_TILE_MAX_ABS;
    desc.tile_m = desc.tile_n = 64;
    desc.tile_bounds = bounds.data(); desc.tile_bounds_count = bounds.size();
    // Select exactly the required basis so equal bounds exercise physical folding.
    const rns8::detail::cpp_int range = rns8::detail::cpp_int(bounds.back()) * (is_unsigned ? 1 : 2);
    desc.max_prefix = 1;
    while (rns8::detail::modulus_product(desc.max_prefix) <= range) ++desc.max_prefix;
    desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    rns8_plan* raw_plan = nullptr;
    REQUIRE(rns8_create_plan(ctx.get(), &desc, &raw_plan) == RNS8_SUCCESS);
    std::unique_ptr<rns8_plan, decltype(&rns8_destroy_plan)> plan(raw_plan, rns8_destroy_plan);
    CHECK(plan->desc.bound == 0);
    CHECK(plan->tile_bounds == bounds);
    CHECK(plan->tile_schedule.empty() == (contract != 3));
    if (contract == 1) {
      plan->tile_bounds[1] = 2;
      CHECK_FALSE(rns8::detail::api::revalidate_plan_schedule_contract(*plan));
      plan->tile_bounds[1] = 1;
    }
    rns8_workspace* raw_workspace = nullptr;
    REQUIRE(rns8_create_workspace(ctx.get(), plan.get(), &raw_workspace) == RNS8_SUCCESS);
    std::unique_ptr<rns8_workspace, decltype(&rns8_destroy_workspace)> workspace(raw_workspace, rns8_destroy_workspace);
    auto create_matrix = [&](int64_t rows, int64_t cols) {
      auto md = bounded_matrix_desc(rows, cols, semantics);
      md.bound_kind = desc.bound_kind; md.tile_m = md.tile_n = 64; md.max_prefix = desc.max_prefix;
      rns8_matrix* raw = nullptr;
      REQUIRE(rns8_create_matrix(ctx.get(), &md, &raw) == RNS8_SUCCESS);
      return std::unique_ptr<rns8_matrix, decltype(&rns8_destroy_matrix)>(raw, rns8_destroy_matrix);
    };
    auto a = create_matrix(65, 1), b = create_matrix(1, 3), c = create_matrix(65, 3);
    std::vector<uint64_t> au(65), bu(3, 1), output_u(65 * 5, 37);
    std::vector<int64_t> ai(65), bi(3, 1), output_i(65 * 5, 37);
    for (int row = 0; row < 65; ++row) {
      au[row] = bounds[row / 64];
      if (!is_unsigned)
        ai[row] = row % 2 ? -static_cast<int64_t>(bounds[row / 64]) : static_cast<int64_t>(bounds[row / 64]);
    }
    if (is_unsigned) {
      REQUIRE(rns8_pack_u64(ctx.get(), a.get(), au.data(), 1, 1) == RNS8_SUCCESS);
      REQUIRE(rns8_pack_u64(ctx.get(), b.get(), bu.data(), 3, 1) == RNS8_SUCCESS);
    } else {
      REQUIRE(rns8_pack_i64(ctx.get(), a.get(), ai.data(), 1, 1) == RNS8_SUCCESS);
      REQUIRE(rns8_pack_i64(ctx.get(), b.get(), bi.data(), 3, 1) == RNS8_SUCCESS);
    }
    REQUIRE(rns8_gemm_rns(ctx.get(), plan.get(), a.get(), b.get(), c.get(), workspace.get()) == RNS8_SUCCESS);
    const auto export_output = [&] {
      return is_unsigned ? rns8_export_u64(ctx.get(), plan.get(), c.get(), output_u.data(), 5)
                         : rns8_export_i64(ctx.get(), plan.get(), c.get(), output_i.data(), 5);
    };
    REQUIRE(export_output() == RNS8_SUCCESS);
    for (int row = 0; row < 65; ++row) for (int col = 0; col < 5; ++col) {
      if (is_unsigned) CHECK(output_u[row * 5 + col] == (col < 3 ? au[row] : 37));
      else CHECK(output_i[row * 5 + col] == (col < 3 ? ai[row] : 37));
    }
    if (contract != 2) {
      // Import an out-of-contract result without depending on producer admission.
      std::vector<uint64_t> bad_u(65 * 3, 0);
      std::vector<int64_t> bad_i(65 * 3, 0);
      bad_u[0] = bounds.front() + 1; bad_i[0] = -static_cast<int64_t>(bounds.front() + 1);
      if (is_unsigned) REQUIRE(rns8_pack_u64(ctx.get(), c.get(), bad_u.data(), 3, 2) == RNS8_SUCCESS);
      else REQUIRE(rns8_pack_i64(ctx.get(), c.get(), bad_i.data(), 3, 2) == RNS8_SUCCESS);
      std::fill(output_u.begin(), output_u.end(), 37);
      std::fill(output_i.begin(), output_i.end(), 37);
      REQUIRE(export_output() == RNS8_RANGE_ERROR);
      CHECK(output_u == std::vector<uint64_t>(65 * 5, 37));
      CHECK(output_i == std::vector<int64_t>(65 * 5, 37));
    }
  }
}
} // namespace

TEST_CASE("CPU folded and heterogeneous tile bounds retain exact export contracts", "[api][schedule][export]") {
  check_folded_tile_bounds(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("HIP folded and heterogeneous tile bounds retain exact export contracts", "[api][schedule][export][hip]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!hip_available()) SKIP("HIP device unavailable");
  check_folded_tile_bounds(RNS8_BACKEND_HIP_DIRECT);
#else
  SKIP("HIP backend unavailable in this build");
#endif
}
