TEST_CASE("direct HIP per-tile bounded signed K-split preserves centered cancellation") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP per-tile signed K-split cancellation smoke");
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  const int64_t lda = k + 1;
  constexpr int64_t ldb = n + 1;
  constexpr int64_t ldc = n + 1;
  constexpr int64_t expected = 127 * 127;
  constexpr int64_t sentinel = INT64_C(-0x55aa55aa);
  std::vector<int64_t> A(static_cast<std::size_t>(m * lda), sentinel);
  std::vector<int64_t> B(static_cast<std::size_t>(k * ldb), sentinel);
  std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<int64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);
  for (int64_t kk = 0; kk < k; ++kk) {
    A[static_cast<std::size_t>(kk)] = kk % 2 == 0 ? 127 : -127;
    B[static_cast<std::size_t>(kk * ldb)] = 127;
    B[static_cast<std::size_t>(kk * ldb + 1)] = -127;
  }

  const std::vector<uint64_t> bounds = {static_cast<uint64_t>(expected)};
  auto cpu_desc = per_tile_signed_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = per_tile_signed_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* hip_plan = nullptr;
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(hip_plan->prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  REQUIRE(hip_plan->tile_schedule.size() == 1);
  CHECK(hip_plan->tile_schedule[0].selected_prefix == hip_plan->tile_schedule[0].required_prefix);
  CHECK(hip_plan->tile_schedule[0].selected_prefix < hip_plan->prefix);
  rns8_destroy_plan(hip_plan);

  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_i64_oneshot(hip, &hip_desc, A.data(), lda, B.data(), ldb, hip_c.data(), ldc) == RNS8_SUCCESS);
  CHECK(hip_c[0] == cpu_c[0]);
  CHECK(hip_c[1] == cpu_c[1]);
  CHECK(hip_c[0] == expected);
  CHECK(hip_c[1] == -expected);
  CHECK(cpu_c[2] == sentinel);
  CHECK(hip_c[2] == sentinel);

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP private tiled bounded wrappers reject malformed selected-prefix metadata") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP tiled selected-prefix rejection smoke");
  }

  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  constexpr int8_t residue_sentinel = -91;
  const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};

  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto hip_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* hip_plan = nullptr;
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE_FALSE(hip_plan->tile_schedule.empty());

  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  std::fill(hip_c->residues.begin(), hip_c->residues.end(), residue_sentinel);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(
              hip->device_id, hip_c->hip_residues, hip_c->residues.data(), hip_c->hip_residue_bytes) ==
          RNS8_SUCCESS);

  rns8::detail::hip_direct_allocation_counters_reset();
  const auto initial_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  auto check_bad_entries = [&](const std::vector<rns8_plan_tile_schedule_entry>& bad_entries) {
    CHECK(rns8::detail::hip_direct_gemm_rns_tiled_device(
              hip->device_id,
              hip_a->hip_residues,
              hip_b->hip_residues,
              hip_c->hip_residues,
              m,
              n,
              k,
              k,
              n,
              n,
              bad_entries.data(),
              bad_entries.size()) == RNS8_INVALID_ARGUMENT);
    REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
                hip->device_id, hip_c->residues.data(), hip_c->hip_residues, hip_c->hip_residue_bytes) ==
            RNS8_SUCCESS);
    CHECK(std::all_of(hip_c->residues.begin(), hip_c->residues.end(), [&](int8_t value) {
      return value == residue_sentinel;
    }));

    const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(repeated_allocations.allocate_calls == initial_allocations.allocate_calls);
    CHECK(repeated_allocations.free_calls == initial_allocations.free_calls);
    CHECK(repeated_allocations.allocated_bytes == initial_allocations.allocated_bytes);
  };

  {
    std::vector<rns8_plan_tile_schedule_entry> bad_entries = hip_plan->tile_schedule;
    bad_entries[0].required_prefix = bad_entries[0].selected_prefix + 1;
    check_bad_entries(bad_entries);
  }
  {
    std::vector<rns8_plan_tile_schedule_entry> bad_entries = hip_plan->tile_schedule;
    bad_entries[0].group_index += 1;
    check_bad_entries(bad_entries);
  }
  {
    std::vector<rns8_plan_tile_schedule_entry> bad_entries = hip_plan->tile_schedule;
    bad_entries[1].tile_row = bad_entries[0].tile_row;
    bad_entries[1].tile_col = bad_entries[0].tile_col;
    check_bad_entries(bad_entries);
  }
  {
    std::vector<rns8_plan_tile_schedule_entry> bad_entries = hip_plan->tile_schedule;
    bad_entries.back().tile_col += 1;
    check_bad_entries(bad_entries);
  }
  {
    std::vector<rns8_plan_tile_schedule_entry> bad_entries = hip_plan->tile_schedule;
    bad_entries[0].row_extent -= 1;
    check_bad_entries(bad_entries);
  }
  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_context(hip);
}

TEST_CASE("direct HIP per-tile bounded GEMM leaves skipped residue planes untouched") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP per-tile residue skip smoke");
  }

  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  constexpr int8_t sentinel = 42;
  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  std::vector<uint64_t> A(m * k);
  std::vector<uint64_t> B(k * n);
  for (int64_t row = 0; row < m; ++row) {
    A[static_cast<std::size_t>(row)] = row < 64 ? 1 : 1000000;
  }
  for (int64_t col = 0; col < n; ++col) {
    B[static_cast<std::size_t>(col)] = col < 64 ? 0 : 1000;
  }
  const std::vector<uint64_t> bounds = {0, 1000, 0, 1000000000};
  auto cpu_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);
  cpu_desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
  hip_desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;

  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(cpu_plan->tile_schedule.size() == hip_plan->tile_schedule.size());
  for (std::size_t i = 0; i < cpu_plan->tile_schedule.size(); ++i) {
    CHECK(cpu_plan->tile_schedule[i].required_prefix == hip_plan->tile_schedule[i].required_prefix);
    CHECK(cpu_plan->tile_schedule[i].selected_prefix == hip_plan->tile_schedule[i].selected_prefix);
    CHECK(cpu_plan->tile_schedule[i].group_index == hip_plan->tile_schedule[i].group_index);
    CHECK(cpu_plan->tile_schedule[i].flags == hip_plan->tile_schedule[i].flags);
  }
  CHECK(hip_plan->tile_schedule[0].flags == RNS8_TILE_SCHEDULE_ZERO_OUTPUT);
  CHECK(hip_plan->tile_schedule[2].flags == RNS8_TILE_SCHEDULE_ZERO_OUTPUT);
  rns8_plan_backend_info zero_skip_info{};
  zero_skip_info.struct_size = sizeof(zero_skip_info);
  zero_skip_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(hip_plan, &zero_skip_info) == RNS8_SUCCESS);
  constexpr const char* zero_skip_kernel = "direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3";
  CHECK(std::string(zero_skip_info.selected_kernel) == zero_skip_kernel);
  CHECK(std::string(zero_skip_info.autotune_key).find(std::string("kernel=") + zero_skip_kernel + ";") !=
        std::string::npos);
  uint64_t nonzero_active_entry_count = 0;
  for (const auto& entry : hip_plan->tile_schedule) {
    if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) == 0) {
      nonzero_active_entry_count += entry.selected_prefix;
    }
  }
  const uint64_t expected_zero_skip_workspace_bytes =
      (static_cast<uint64_t>(hip_plan->tile_schedule.size()) + nonzero_active_entry_count) *
      sizeof(rns8_plan_tile_schedule_entry);
  CHECK(zero_skip_info.workspace_required_bytes == expected_zero_skip_workspace_bytes);

  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);
  CHECK(hip_workspace->hip_tile_schedule != nullptr);
  CHECK(hip_workspace->hip_tile_schedule_count == hip_plan->tile_schedule.size());
  CHECK(
      hip_workspace->hip_tile_schedule_bytes ==
      hip_plan->tile_schedule.size() * sizeof(rns8_plan_tile_schedule_entry));
  CHECK(hip_workspace->hip_tile_schedule_active_entries_count == nonzero_active_entry_count);
  CHECK(
      hip_workspace->hip_tile_schedule_active_entries_bytes ==
      nonzero_active_entry_count * sizeof(rns8_plan_tile_schedule_entry));

  std::fill(hip_c->residues.begin(), hip_c->residues.end(), sentinel);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(
              0, hip_c->hip_residues, hip_c->residues.data(), hip_c->hip_residue_bytes) == RNS8_SUCCESS);
  rns8::test::set_host_residues_current(*hip_c, true);
  rns8::test::set_device_residues_current(*hip_c, true);

  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
  const auto zero_skip_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(has_timing_label(zero_skip_events, "direct_hip_zero_output_tile_memset"));
  CHECK(has_timing_label(zero_skip_events, "rns_gemm_kernel_group"));
  REQUIRE_FALSE(hip_c->host_residues_current);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
              0, hip_c->residues.data(), hip_c->hip_residues, hip_c->hip_residue_bytes) == RNS8_SUCCESS);

  for (const auto& entry : hip_plan->tile_schedule) {
    for (uint32_t p = 0; p < hip_plan->prefix; ++p) {
      for (int64_t row = entry.row_offset; row < entry.row_offset + entry.row_extent; ++row) {
        for (int64_t col = entry.col_offset; col < entry.col_offset + entry.col_extent; ++col) {
          const auto index = rns8::detail::residue_index(*hip_c, p, row, col);
          if (p < entry.selected_prefix) {
            CHECK(hip_c->residues[index] == cpu_c->residues[rns8::detail::residue_index(*cpu_c, p, row, col)]);
          } else {
            CHECK(hip_c->residues[index] == sentinel);
          }
        }
      }
    }
  }

  rns8_destroy_workspace(hip_workspace);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP per-tile bounded zero row and column proof masks match CPU") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP zero row/column proof-mask smoke");
  }

  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 3;
  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  std::vector<uint64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * n), 0);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t kk = 0; kk < k; ++kk) {
      A[static_cast<std::size_t>(row * k + kk)] =
          row == 0 ? 0 : static_cast<uint64_t>((row % 11) + kk + 1);
    }
  }
  for (int64_t kk = 0; kk < k; ++kk) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(kk * n + col)] =
          col == 64 ? 0 : static_cast<uint64_t>((col % 13) + kk + 2);
    }
  }

  const std::vector<uint64_t> bounds(4, 1000000);
  std::vector<uint8_t> zero_a_rows(65, 0);
  std::vector<uint8_t> zero_b_cols(65, 0);
  zero_a_rows[0] = 1;
  zero_b_cols[64] = 1;
  auto cpu_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);
  for (auto* desc : {&cpu_desc, &hip_desc}) {
    desc->flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS;
    desc->zero_a_rows = zero_a_rows.data();
    desc->zero_a_rows_count = zero_a_rows.size();
    desc->zero_b_cols = zero_b_cols.data();
    desc->zero_b_cols_count = zero_b_cols.size();
  }

  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(hip_plan->tile_schedule.size() == bounds.size());
  CHECK(hip_plan->schedule_flags == RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT);
  CHECK(hip_plan->zero_a_row_count == 1);
  CHECK(hip_plan->zero_b_col_count == 1);
  CHECK(hip_plan->zero_row_col_product_count == 129);

  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);
  CHECK(hip_workspace->hip_zero_a_rows != nullptr);
  CHECK(hip_workspace->hip_zero_a_rows_bytes == zero_a_rows.size());
  CHECK(hip_workspace->hip_zero_b_cols != nullptr);
  CHECK(hip_workspace->hip_zero_b_cols_bytes == zero_b_cols.size());

  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
  const auto gemm_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(has_timing_label(gemm_events, "rns_gemm_kernel_group"));
  CHECK_FALSE(has_timing_label(gemm_events, "direct_hip_zero_output_tile_memset"));

  std::vector<uint64_t> cpu_out(static_cast<std::size_t>(m * n), 0xababababababababull);
  std::vector<uint64_t> hip_out(static_cast<std::size_t>(m * n), 0xcdcdcdcdcdcdcdcdull);
  REQUIRE(rns8_export_u64(cpu, cpu_plan, cpu_c, cpu_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), n) == RNS8_SUCCESS);
  CHECK(hip_c->hip_export_zero_a_rows != nullptr);
  CHECK(hip_c->hip_export_zero_a_rows_bytes == zero_a_rows.size());
  CHECK(hip_c->hip_export_zero_b_cols != nullptr);
  CHECK(hip_c->hip_export_zero_b_cols_bytes == zero_b_cols.size());
  CHECK(hip_out == cpu_out);
  for (int64_t row = 0; row < m; ++row) {
    CHECK(hip_out[static_cast<std::size_t>(row * n + 64)] == 0);
  }
  for (int64_t col = 0; col < n; ++col) {
    CHECK(hip_out[static_cast<std::size_t>(col)] == 0);
  }

  rns8_destroy_workspace(hip_workspace);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP all-zero per-tile bounded export skips status traffic") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP all-zero per-tile export smoke");
  }

  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 3;
  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  std::vector<uint64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * n), 0);
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * n + col)] = static_cast<uint64_t>((row + 1) * (col + 3));
    }
  }
  const std::vector<uint64_t> bounds(4, 0);
  auto cpu_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);
  cpu_desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
  hip_desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;

  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(hip_plan->tile_schedule.size() == bounds.size());
  for (const auto& entry : hip_plan->tile_schedule) {
    CHECK(entry.flags == RNS8_TILE_SCHEDULE_ZERO_OUTPUT);
  }
  rns8_plan_backend_info hip_info{};
  hip_info.struct_size = sizeof(hip_info);
  hip_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(hip_plan, &hip_info) == RNS8_SUCCESS);
  CHECK(hip_info.workspace_required_bytes == 0);

  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);
  CHECK(hip_workspace->hip_tile_schedule == nullptr);
  CHECK(hip_workspace->hip_tile_schedule_bytes == 0);
  CHECK(hip_workspace->hip_tile_schedule_count == 0);
  CHECK(hip_workspace->hip_tile_schedule_active_entries_count == 0);
  CHECK(hip_workspace->hip_tile_schedule_active_entries_bytes == 0);
  CHECK(hip_workspace->hip_tile_schedule_active_entries == nullptr);

  CHECK_FALSE(hip_a->device_residues_current);
  CHECK_FALSE(hip_b->device_residues_current);
  std::vector<uint64_t> hip_unpacked_out(static_cast<std::size_t>(m * n), 0xefefefefefefefefull);
  REQUIRE(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, hip_plan, hip_c, hip_unpacked_out.data(), n) == RNS8_SUCCESS);
  CHECK(std::all_of(hip_unpacked_out.begin(), hip_unpacked_out.end(), [](uint64_t value) { return value == 0; }));

  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
  const auto gemm_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(has_timing_label(gemm_events, "direct_hip_zero_output_tile_memset"));
  CHECK_FALSE(has_timing_label(gemm_events, "rns_gemm_kernel_group"));

  std::vector<uint64_t> cpu_out(static_cast<std::size_t>(m * n), 0xababababababababull);
  std::vector<uint64_t> hip_out(static_cast<std::size_t>(m * n), 0xcdcdcdcdcdcdcdcdull);
  REQUIRE(rns8_export_u64(cpu, cpu_plan, cpu_c, cpu_out.data(), n) == RNS8_SUCCESS);
  CHECK(hip_c->hip_status_buffer == nullptr);
  CHECK(hip_c->hip_export_tile_schedule == nullptr);
  CHECK(hip_c->hip_export_tile_bounds == nullptr);
  CHECK(hip_c->hip_export_tile_schedule_count == 0);
  CHECK(hip_c->hip_export_tile_bounds_count == 0);
  CHECK(hip_c->hip_export_tile_max_elements == 0);
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), n) == RNS8_SUCCESS);
  const auto export_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(hip_c->hip_status_buffer == nullptr);
  CHECK(hip_c->hip_export_tile_schedule == nullptr);
  CHECK(hip_c->hip_export_tile_bounds == nullptr);
  CHECK(hip_c->hip_export_tile_schedule_count == 0);
  CHECK(hip_c->hip_export_tile_bounds_count == 0);
  CHECK(hip_c->hip_export_tile_max_elements == 0);
  CHECK(has_timing_label(export_events, "crt_export_kernel"));
  CHECK(has_timing_label(export_events, "crt_export_d2h"));
  CHECK_FALSE(has_timing_label(export_events, "crt_export_status_memset"));
  CHECK_FALSE(has_timing_label(export_events, "crt_export_status_d2h"));
  CHECK(hip_out == cpu_out);
  CHECK(std::all_of(hip_out.begin(), hip_out.end(), [](uint64_t value) { return value == 0; }));

  rns8_destroy_workspace(hip_workspace);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}
