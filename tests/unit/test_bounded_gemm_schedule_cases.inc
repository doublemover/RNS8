TEST_CASE("bounded CPU K-split edge cases preserve padded output and exact cancellation") {
  rns8_context* ctx = create_cpu();
  for (int64_t k : {static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) - 1,
                    static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK),
                    static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1}) {
    {
      constexpr int64_t m = 2;
      constexpr int64_t n = 2;
      const int64_t lda = k + 1;
      constexpr int64_t ldb = 3;
      constexpr int64_t ldc = 4;
      constexpr int64_t product = 127 * 127;
      constexpr int64_t sentinel = INT64_C(-0x51f15e5);
      std::vector<int64_t> A(static_cast<std::size_t>(m * lda), INT64_C(0x5a5a5a5a));
      std::vector<int64_t> B(static_cast<std::size_t>(k * ldb), INT64_C(-0x1234567));
      std::vector<int64_t> C(static_cast<std::size_t>(m * ldc), sentinel);
      for (int64_t kk = 0; kk < k; ++kk) {
        A[static_cast<std::size_t>(kk)] = kk % 2 == 0 ? 127 : -127;
        A[static_cast<std::size_t>(lda + kk)] = 127;
        B[static_cast<std::size_t>(kk * ldb)] = 127;
        B[static_cast<std::size_t>(kk * ldb + 1)] = -127;
      }

      auto desc = i64_desc(m, n, k, static_cast<uint64_t>(k * product));
      REQUIRE(rns8_gemm_i64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, C.data(), ldc) == RNS8_SUCCESS);
      const int64_t alternating = k % 2 == 0 ? 0 : product;
      CHECK(C[0] == alternating);
      CHECK(C[1] == -alternating);
      CHECK(C[2] == sentinel);
      CHECK(C[3] == sentinel);
      CHECK(C[static_cast<std::size_t>(ldc)] == k * product);
      CHECK(C[static_cast<std::size_t>(ldc + 1)] == -(k * product));
      CHECK(C[static_cast<std::size_t>(ldc + 2)] == sentinel);
      CHECK(C[static_cast<std::size_t>(ldc + 3)] == sentinel);
    }

    {
      constexpr int64_t m = 2;
      constexpr int64_t n = 2;
      const int64_t lda = k + 1;
      constexpr int64_t ldb = 3;
      constexpr int64_t ldc = 4;
      constexpr uint64_t sentinel = 0xc001d00dcafef00dull;
      std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), 0xfeedfacefeedfaceull);
      std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), 0xdeadbeefdeadbeefull);
      std::vector<uint64_t> C(static_cast<std::size_t>(m * ldc), sentinel);
      for (int64_t kk = 0; kk < k; ++kk) {
        A[static_cast<std::size_t>(kk)] = 255;
        A[static_cast<std::size_t>(lda + kk)] = 1;
        B[static_cast<std::size_t>(kk * ldb)] = 1;
        B[static_cast<std::size_t>(kk * ldb + 1)] = 255;
      }

      auto desc = u64_desc(m, n, k, static_cast<uint64_t>(k) * 255u * 255u);
      REQUIRE(rns8_gemm_u64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, C.data(), ldc) == RNS8_SUCCESS);
      CHECK(C[0] == static_cast<uint64_t>(k) * 255u);
      CHECK(C[1] == static_cast<uint64_t>(k) * 255u * 255u);
      CHECK(C[2] == sentinel);
      CHECK(C[3] == sentinel);
      CHECK(C[static_cast<std::size_t>(ldc)] == static_cast<uint64_t>(k));
      CHECK(C[static_cast<std::size_t>(ldc + 1)] == static_cast<uint64_t>(k) * 255u);
      CHECK(C[static_cast<std::size_t>(ldc + 2)] == sentinel);
      CHECK(C[static_cast<std::size_t>(ldc + 3)] == sentinel);
    }
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded plan schedule exposes tile grid and minimum proven prefix groups") {
  rns8_context* ctx = create_cpu();
  auto desc = u64_desc(130, 129, 7, 1000);
  desc.tile_m = 64;
  desc.tile_n = 64;
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.tile_m == 64);
  CHECK(info.tile_n == 64);
  CHECK(info.tile_rows == 3);
  CHECK(info.tile_cols == 3);
  CHECK(info.tile_count == 9);
  CHECK(info.min_required_prefix == 2);
  CHECK(info.max_required_prefix == 2);
  CHECK(info.min_selected_prefix == 2);
  CHECK(info.max_selected_prefix == 2);
  CHECK(info.prefix_group_count == 1);
  CHECK(info.adaptive_prefix_active == 0);
  CHECK(info.adaptive_skip_active == 1);
  CHECK(info.range_bit_length == 10);

  uint64_t written = 0;
  REQUIRE(rns8_get_plan_tile_schedule(plan, nullptr, 0, &written) == RNS8_SUCCESS);
  CHECK(written == 9);
  std::vector<rns8_plan_tile_schedule_entry> too_small(8);
  CHECK(rns8_get_plan_tile_schedule(plan, too_small.data(), too_small.size(), &written) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(written == 9);

  std::vector<rns8_plan_tile_schedule_entry> entries(9);
  REQUIRE(rns8_get_plan_tile_schedule(plan, entries.data(), entries.size(), &written) == RNS8_SUCCESS);
  REQUIRE(written == entries.size());
  CHECK(entries.front().tile_row == 0);
  CHECK(entries.front().tile_col == 0);
  CHECK(entries.front().row_offset == 0);
  CHECK(entries.front().col_offset == 0);
  CHECK(entries.front().row_extent == 64);
  CHECK(entries.front().col_extent == 64);
  CHECK(entries.front().required_prefix == 2);
  CHECK(entries.front().selected_prefix == 2);
  CHECK(entries.front().group_index == 0);
  CHECK(entries.front().range_bit_length == 10);
  CHECK(entries.back().tile_row == 2);
  CHECK(entries.back().tile_col == 2);
  CHECK(entries.back().row_offset == 128);
  CHECK(entries.back().col_offset == 128);
  CHECK(entries.back().row_extent == 2);
  CHECK(entries.back().col_extent == 1);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded CPU plan schedule uses copied per-tile unsigned bounds") {
  rns8_context* ctx = create_cpu();
  std::vector<uint64_t> bounds = {1, 1000, 1000000, uint64_t{1} << 40u, 5,
                                  2000, 70000,  uint64_t{1} << 20u, uint64_t{1} << 32u};
  auto desc = u64_desc(130, 129, 7, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  bounds[3] = 1;

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.tile_m == 64);
  CHECK(info.tile_n == 64);
  CHECK(info.tile_rows == 3);
  CHECK(info.tile_cols == 3);
  CHECK(info.tile_count == 9);
  CHECK(info.min_required_prefix == 1);
  CHECK(info.max_required_prefix == 6);
  CHECK(info.min_selected_prefix == 1);
  CHECK(info.max_selected_prefix == 6);
  CHECK(info.prefix_group_count == 5);
  CHECK(info.adaptive_prefix_active == 1);
  CHECK(info.adaptive_skip_active == 1);
  CHECK(info.range_bit_length == 41);

  uint64_t written = 0;
  std::vector<rns8_plan_tile_schedule_entry> entries(9);
  REQUIRE(rns8_get_plan_tile_schedule(plan, entries.data(), entries.size(), &written) == RNS8_SUCCESS);
  REQUIRE(written == entries.size());
  CHECK(entries[0].required_prefix == 1);
  CHECK(entries[0].selected_prefix == 1);
  CHECK(entries[0].group_index == 0);
  CHECK(entries[1].required_prefix == 2);
  CHECK(entries[1].selected_prefix == 2);
  CHECK(entries[1].group_index == 1);
  CHECK(entries[2].required_prefix == 3);
  CHECK(entries[2].selected_prefix == 3);
  CHECK(entries[2].group_index == 2);
  CHECK(entries[3].required_prefix == 6);
  CHECK(entries[3].selected_prefix == 6);
  CHECK(entries[3].group_index == 4);
  CHECK(entries[8].required_prefix == 5);
  CHECK(entries[8].selected_prefix == 5);
  CHECK(entries[8].group_index == 3);
  CHECK(entries[8].row_offset == 128);
  CHECK(entries[8].col_offset == 128);
  CHECK(entries[8].row_extent == 2);
  CHECK(entries[8].col_extent == 1);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded per-tile zero output skips require proven-zero opt-in") {
  rns8_context* ctx = create_cpu();
  const std::vector<uint64_t> bounds = {0, 1000, 0, 1000000};

  auto desc = u64_desc(65, 65, 1, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.flags == 0);

  std::vector<rns8_plan_tile_schedule_entry> entries(4);
  uint64_t written = 0;
  REQUIRE(rns8_get_plan_tile_schedule(plan, entries.data(), entries.size(), &written) == RNS8_SUCCESS);
  REQUIRE(written == entries.size());
  CHECK(entries[0].range_bit_length == 0);
  CHECK(entries[0].required_prefix == 1);
  CHECK(entries[0].selected_prefix == 1);
  CHECK(entries[0].flags == 0);
  CHECK(entries[2].flags == 0);
  rns8_destroy_plan(plan);

  auto proven_desc = u64_desc(65, 65, 1, 0);
  proven_desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  proven_desc.tile_m = 64;
  proven_desc.tile_n = 64;
  proven_desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
  proven_desc.tile_bounds = bounds.data();
  proven_desc.tile_bounds_count = bounds.size();
  rns8_plan* proven_plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &proven_desc, &proven_plan) == RNS8_SUCCESS);

  info = {};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(proven_plan, &info) == RNS8_SUCCESS);
  CHECK(info.flags == RNS8_TILE_SCHEDULE_ZERO_OUTPUT);

  entries.assign(4, {});
  written = 0;
  REQUIRE(rns8_get_plan_tile_schedule(proven_plan, entries.data(), entries.size(), &written) == RNS8_SUCCESS);
  REQUIRE(written == entries.size());
  CHECK(entries[0].flags == RNS8_TILE_SCHEDULE_ZERO_OUTPUT);
  CHECK(entries[1].flags == 0);
  CHECK(entries[2].flags == RNS8_TILE_SCHEDULE_ZERO_OUTPUT);
  CHECK(entries[3].flags == 0);
  CHECK(entries[0].range_bit_length == 0);
  CHECK(entries[2].range_bit_length == 0);
  rns8_destroy_plan(proven_plan);

  auto global_desc = u64_desc(1, 1, 1, 1);
  global_desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
  rns8_plan* invalid_plan = nullptr;
  CHECK(rns8_create_plan(ctx, &global_desc, &invalid_plan) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_context(ctx);
}

TEST_CASE("bounded per-tile zero row and column product skips require explicit proof masks") {
  rns8_context* ctx = create_cpu();
  const std::vector<uint64_t> bounds = {1000, 1000, 1000, 1000};
  std::vector<uint8_t> zero_a_rows(65, 0);
  std::vector<uint8_t> zero_b_cols(65, 0);
  zero_a_rows[0] = 1;
  zero_b_cols[64] = 1;

  auto desc = u64_desc(65, 65, 1, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS;
  desc.zero_a_rows = zero_a_rows.data();
  desc.zero_a_rows_count = zero_a_rows.size();
  desc.zero_b_cols = zero_b_cols.data();
  desc.zero_b_cols_count = zero_b_cols.size();
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  CHECK(plan->desc.zero_a_rows == nullptr);
  CHECK(plan->desc.zero_a_rows_count == zero_a_rows.size());
  CHECK(plan->desc.zero_b_cols == nullptr);
  CHECK(plan->desc.zero_b_cols_count == zero_b_cols.size());
  CHECK(plan->zero_a_row_count == 1);
  CHECK(plan->zero_b_col_count == 1);
  CHECK(plan->zero_row_col_product_count == 129);

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.flags == RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT);
  CHECK(info.zero_a_row_count == 1);
  CHECK(info.zero_b_col_count == 1);
  CHECK(info.zero_row_col_product_count == 129);

  std::vector<rns8_plan_tile_schedule_entry> entries(4);
  uint64_t written = 0;
  REQUIRE(rns8_get_plan_tile_schedule(plan, entries.data(), entries.size(), &written) == RNS8_SUCCESS);
  REQUIRE(written == entries.size());
  CHECK(entries[0].flags == RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT);
  CHECK(entries[1].flags == RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT);
  CHECK(entries[2].flags == 0);
  CHECK(entries[3].flags == RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT);
  rns8_destroy_plan(plan);

  auto missing_flag = desc;
  missing_flag.flags = 0;
  rns8_plan* invalid_plan = nullptr;
  CHECK(rns8_create_plan(ctx, &missing_flag, &invalid_plan) == RNS8_INVALID_ARGUMENT);

  auto global_desc = u64_desc(65, 65, 1, 1000);
  global_desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS;
  global_desc.zero_a_rows = zero_a_rows.data();
  global_desc.zero_a_rows_count = zero_a_rows.size();
  global_desc.zero_b_cols = zero_b_cols.data();
  global_desc.zero_b_cols_count = zero_b_cols.size();
  CHECK(rns8_create_plan(ctx, &global_desc, &invalid_plan) == RNS8_INVALID_ARGUMENT);

  auto bad_mask = desc;
  zero_a_rows[1] = 2;
  bad_mask.zero_a_rows = zero_a_rows.data();
  CHECK(rns8_create_plan(ctx, &bad_mask, &invalid_plan) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_context(ctx);
}

TEST_CASE("bounded per-tile schedule collapses to fixed prefix when every tile needs the full prefix") {
  rns8_context* ctx = create_cpu();
  const auto prefix8_product = rns8::detail::modulus_product(RNS8_DEFAULT_BOUNDED_PREFIX - 1u);
  const uint64_t first_full_signed_bound = (prefix8_product / 2).convert_to<uint64_t>();
  constexpr uint64_t max_signed_bound = uint64_t{1} << 63u;
  const uint32_t first_full_range_bits = rns8::detail::bit_length(prefix8_product);
  const uint32_t max_signed_range_bits = rns8::detail::bit_length(
      rns8::detail::cpp_int(2) * rns8::detail::cpp_int(max_signed_bound));
  std::vector<uint64_t> bounds = {
      max_signed_bound,
      first_full_signed_bound,
      max_signed_bound,
      first_full_signed_bound,
  };
  auto desc = i64_desc(65, 65, 7, 0);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_ABS;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  bounds[0] = 1;
  REQUIRE(plan->tile_bounds.size() == 4);
  CHECK(plan->desc.tile_bounds == nullptr);
  CHECK(plan->desc.tile_bounds_count == 4);
  CHECK(plan->tile_schedule.empty());

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.tile_m == 64);
  CHECK(info.tile_n == 64);
  CHECK(info.tile_rows == 2);
  CHECK(info.tile_cols == 2);
  CHECK(info.tile_count == 4);
  CHECK(info.min_required_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(info.max_required_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(info.prefix_group_count == 1);
  CHECK(info.adaptive_prefix_active == 0);
  CHECK(info.adaptive_skip_active == 0);
  CHECK(info.range_bit_length == max_signed_range_bits);

  uint64_t written = 0;
  REQUIRE(rns8_get_plan_tile_schedule(plan, nullptr, 0, &written) == RNS8_SUCCESS);
  CHECK(written == 4);

  std::vector<rns8_plan_tile_schedule_entry> entries(4);
  REQUIRE(rns8_get_plan_tile_schedule(plan, entries.data(), entries.size(), &written) == RNS8_SUCCESS);
  REQUIRE(written == entries.size());
  for (const auto& entry : entries) {
    CHECK(entry.required_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(entry.selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(entry.group_index == 0);
  }
  CHECK(entries[0].range_bit_length == max_signed_range_bits);
  CHECK(entries[1].range_bit_length == first_full_range_bits);
  CHECK(entries[0].row_offset == 0);
  CHECK(entries[0].col_offset == 0);
  CHECK(entries[0].row_extent == 64);
  CHECK(entries[0].col_extent == 64);
  CHECK(entries[3].tile_row == 1);
  CHECK(entries[3].tile_col == 1);
  CHECK(entries[3].row_offset == 64);
  CHECK(entries[3].col_offset == 64);
  CHECK(entries[3].row_extent == 1);
  CHECK(entries[3].col_extent == 1);

  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("phase 2 fixed 9-modulus per-tile schedules preserve signed and unsigned parity") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  const std::vector<uint64_t> unsigned_bounds = {7, 1000, 7000000, 1000000000};
  const std::vector<uint64_t> signed_bounds = {6, 2000, 3000000, 1000000000};

  auto unsigned_desc = u64_desc(m, n, k, 0);
  unsigned_desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  unsigned_desc.tile_m = 64;
  unsigned_desc.tile_n = 64;
  unsigned_desc.tile_bounds = unsigned_bounds.data();
  unsigned_desc.tile_bounds_count = unsigned_bounds.size();
  auto signed_desc = i64_desc(m, n, k, 0);
  signed_desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_ABS;
  signed_desc.tile_m = 64;
  signed_desc.tile_n = 64;
  signed_desc.tile_bounds = signed_bounds.data();
  signed_desc.tile_bounds_count = signed_bounds.size();

  rns8_plan* unsigned_plan = nullptr;
  rns8_plan* signed_plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &unsigned_desc, &unsigned_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(ctx, &signed_desc, &signed_plan) == RNS8_SUCCESS);

  rns8_plan_schedule_info unsigned_info{};
  unsigned_info.struct_size = sizeof(unsigned_info);
  unsigned_info.abi_version = RNS8_ABI_VERSION;
  rns8_plan_schedule_info signed_info{};
  signed_info.struct_size = sizeof(signed_info);
  signed_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(unsigned_plan, &unsigned_info) == RNS8_SUCCESS);
  REQUIRE(rns8_get_plan_schedule_info(signed_plan, &signed_info) == RNS8_SUCCESS);
  CHECK(unsigned_info.tile_rows == 2);
  CHECK(unsigned_info.tile_cols == 2);
  CHECK(unsigned_info.tile_count == 4);
  CHECK(signed_info.tile_rows == unsigned_info.tile_rows);
  CHECK(signed_info.tile_cols == unsigned_info.tile_cols);
  CHECK(signed_info.tile_count == unsigned_info.tile_count);
  CHECK(unsigned_info.min_selected_prefix == 1);
  CHECK(unsigned_info.max_selected_prefix == 4);
  CHECK(signed_info.min_selected_prefix == unsigned_info.min_selected_prefix);
  CHECK(signed_info.max_selected_prefix == unsigned_info.max_selected_prefix);
  CHECK(unsigned_info.prefix_group_count == 4);
  CHECK(signed_info.prefix_group_count == unsigned_info.prefix_group_count);
  CHECK(unsigned_info.adaptive_prefix_active == 1);
  CHECK(signed_info.adaptive_prefix_active == 1);
  CHECK(unsigned_info.adaptive_skip_active == 1);
  CHECK(signed_info.adaptive_skip_active == 1);

  uint64_t unsigned_written = 0;
  uint64_t signed_written = 0;
  std::vector<rns8_plan_tile_schedule_entry> unsigned_entries(4);
  std::vector<rns8_plan_tile_schedule_entry> signed_entries(4);
  REQUIRE(rns8_get_plan_tile_schedule(
              unsigned_plan, unsigned_entries.data(), unsigned_entries.size(), &unsigned_written) == RNS8_SUCCESS);
  REQUIRE(rns8_get_plan_tile_schedule(signed_plan, signed_entries.data(), signed_entries.size(), &signed_written) ==
          RNS8_SUCCESS);
  REQUIRE(unsigned_written == unsigned_entries.size());
  REQUIRE(signed_written == signed_entries.size());

  const uint32_t expected_prefixes[] = {1, 2, 3, 4};
  for (std::size_t index = 0; index < unsigned_entries.size(); ++index) {
    CHECK(unsigned_entries[index].tile_row == signed_entries[index].tile_row);
    CHECK(unsigned_entries[index].tile_col == signed_entries[index].tile_col);
    CHECK(unsigned_entries[index].row_offset == signed_entries[index].row_offset);
    CHECK(unsigned_entries[index].col_offset == signed_entries[index].col_offset);
    CHECK(unsigned_entries[index].row_extent == signed_entries[index].row_extent);
    CHECK(unsigned_entries[index].col_extent == signed_entries[index].col_extent);
    CHECK(unsigned_entries[index].required_prefix == expected_prefixes[index]);
    CHECK(unsigned_entries[index].selected_prefix == expected_prefixes[index]);
    CHECK(signed_entries[index].required_prefix == expected_prefixes[index]);
    CHECK(signed_entries[index].selected_prefix == expected_prefixes[index]);
    CHECK(unsigned_entries[index].group_index == index);
    CHECK(signed_entries[index].group_index == index);
  }

  rns8_destroy_plan(signed_plan);
  rns8_destroy_plan(unsigned_plan);
  rns8_destroy_context(ctx);
}

