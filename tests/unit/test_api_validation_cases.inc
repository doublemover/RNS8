TEST_CASE("public one-shot APIs reject invalid leading dimensions before access") {
  rns8_context* cpu = create_cpu_context();
  const int64_t a_i64[] = {1, 2, 3, 4};
  const int64_t b_i64[] = {5, 6, 7, 8};
  int64_t c_i64[] = {-1, -1, -1, -1};
  const uint64_t a_u64[] = {1, 2, 3, 4};
  const uint64_t b_u64[] = {5, 6, 7, 8};
  uint64_t c_u64[] = {99, 99, 99, 99};
  const uint8_t a_u8[] = {1, 2, 3, 4};
  const uint8_t b_u8[] = {5, 6, 7, 8};
  uint8_t c_u8[] = {77, 77, 77, 77};

  auto i64_desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 2, 2, 2);
  CHECK(rns8_gemm_i64_oneshot(cpu, &i64_desc, a_i64, 1, b_i64, 2, c_i64, 2) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_i64_oneshot(cpu, &i64_desc, a_i64, 2, b_i64, 1, c_i64, 2) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_i64_oneshot(cpu, &i64_desc, a_i64, 2, b_i64, 2, c_i64, 1) == RNS8_INVALID_ARGUMENT);

  auto u64_desc = bounded_gemm_desc(RNS8_BOUNDED_U64, 2, 2, 2);
  CHECK(rns8_gemm_u64_oneshot(cpu, &u64_desc, a_u64, 1, b_u64, 2, c_u64, 2) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_u64_oneshot(cpu, &u64_desc, a_u64, 2, b_u64, 1, c_u64, 2) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_u64_oneshot(cpu, &u64_desc, a_u64, 2, b_u64, 2, c_u64, 1) == RNS8_INVALID_ARGUMENT);

  auto ring_desc = finite_gemm_desc(RNS8_FINITE_RING_U8, 2, 2, 2, 251);
  CHECK(rns8_gemm_finite_ring_u8_oneshot(cpu, &ring_desc, 251, a_u8, 1, b_u8, 2, c_u8, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_finite_ring_u8_oneshot(cpu, &ring_desc, 251, a_u8, 2, b_u8, 1, c_u8, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_finite_ring_u8_oneshot(cpu, &ring_desc, 251, a_u8, 2, b_u8, 2, c_u8, 1) ==
        RNS8_INVALID_ARGUMENT);

  auto field_desc = finite_gemm_desc(RNS8_FINITE_FIELD_U8, 2, 2, 2, 251);
  CHECK(rns8_gemm_finite_field_u8_oneshot(cpu, &field_desc, 251, a_u8, 1, b_u8, 2, c_u8, 2) ==
        RNS8_INVALID_ARGUMENT);

  rns8_context* wrap = create_wrap_context();
  auto wrap_desc = wrap_gemm_desc(2, 2, 2);
  CHECK(rns8_gemm_wrap_u64_oneshot(wrap, &wrap_desc, a_u64, 1, b_u64, 2, c_u64, 2) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_wrap_u64_oneshot(wrap, &wrap_desc, a_u64, 2, b_u64, 1, c_u64, 2) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_wrap_u64_oneshot(wrap, &wrap_desc, a_u64, 2, b_u64, 2, c_u64, 1) == RNS8_INVALID_ARGUMENT);

  CHECK(c_i64[0] == -1);
  CHECK(c_u64[0] == 99);
  CHECK(c_u8[0] == 77);
  rns8_destroy_context(wrap);
  rns8_destroy_context(cpu);
}

TEST_CASE("public APIs reject overflow dimensions before allocation or access") {
  rns8_context* cpu = create_cpu_context();
  const int64_t max_i64 = std::numeric_limits<int64_t>::max();
  const int64_t huge = max_i64 / 2 + 1;

  const int64_t a_i64[] = {1, 2, 3, 4};
  const int64_t b_i64[] = {5, 6, 7, 8};
  int64_t c_i64[] = {-7, -7, -7, -7};
  auto address_overflow = bounded_gemm_desc(RNS8_BOUNDED_I64, huge, huge, 2);
  CHECK(rns8_gemm_i64_oneshot(cpu, &address_overflow, a_i64, 2, b_i64, huge, c_i64, huge) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(c_i64[0] == -7);

  auto per_tile = bounded_gemm_desc(RNS8_BOUNDED_I64, max_i64, max_i64, 1);
  per_tile.bound_kind = RNS8_BOUND_PER_TILE_MAX_ABS;
  per_tile.bound = 0;
  per_tile.tile_m = 64;
  per_tile.tile_n = 64;
  const uint64_t tile_bound = 1;
  per_tile.tile_bounds = &tile_bound;
  per_tile.tile_bounds_count = std::numeric_limits<uint64_t>::max();
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(cpu, &per_tile, &plan) == RNS8_RANGE_ERROR);
  CHECK(plan == nullptr);

  rns8_matrix* matrix = nullptr;
  auto rns_matrix = bounded_matrix_desc(huge, 3, RNS8_BOUNDED_I64);
  CHECK(rns8_create_matrix(cpu, &rns_matrix, &matrix) == RNS8_RANGE_ERROR);
  CHECK(matrix == nullptr);

  auto finite_matrix = finite_matrix_desc(max_i64, max_i64, RNS8_FINITE_RING_U8);
  CHECK(rns8_create_matrix(cpu, &finite_matrix, &matrix) == RNS8_RANGE_ERROR);
  CHECK(matrix == nullptr);

  rns8_context* wrap = create_wrap_context();
  auto wrap_matrix = wrap_matrix_desc(huge, 3);
  CHECK(rns8_create_matrix(wrap, &wrap_matrix, &matrix) == RNS8_RANGE_ERROR);
  CHECK(matrix == nullptr);

  auto bad_ld = bounded_matrix_desc(2, 3, RNS8_BOUNDED_I64);
  bad_ld.logical_ld = 2;
  CHECK(rns8_create_matrix(cpu, &bad_ld, &matrix) == RNS8_INVALID_ARGUMENT);
  CHECK(matrix == nullptr);

  auto bad_finite_ld = finite_matrix_desc(2, 3, RNS8_FINITE_RING_U8);
  bad_finite_ld.logical_ld = 2;
  CHECK(rns8_create_matrix(cpu, &bad_finite_ld, &matrix) == RNS8_INVALID_ARGUMENT);
  CHECK(matrix == nullptr);

  auto bad_wrap_ld = wrap_matrix_desc(2, 3);
  bad_wrap_ld.logical_ld = 2;
  CHECK(rns8_create_matrix(wrap, &bad_wrap_ld, &matrix) == RNS8_INVALID_ARGUMENT);
  CHECK(matrix == nullptr);

  rns8_destroy_context(wrap);
  rns8_destroy_context(cpu);
}

