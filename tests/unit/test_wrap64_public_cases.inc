TEST_CASE("public wrap64 one-shot uses byte-limb low-64-bit semantics") {
  constexpr int64_t m = 2;
  constexpr int64_t n = 3;
  constexpr int64_t k = 5;
  constexpr int64_t ldc = 4;
  const std::vector<uint64_t> A = {
      0,
      1,
      std::numeric_limits<uint64_t>::max(),
      0x8080808080808080ull,
      0x0102030405060708ull,
      255,
      256,
      std::numeric_limits<uint64_t>::max() - 1,
      0x7f7f7f7f7f7f7f7full,
      17};
  const std::vector<uint64_t> B = {
      3,
      std::numeric_limits<uint64_t>::max(),
      0x1112131415161718ull,
      29,
      0x8080808080808080ull,
      31,
      0x0101010101010101ull,
      0xfefdfcfbfaf9f8f7ull,
      37,
      41,
      43,
      47,
      53,
      59,
      61};
  std::vector<uint64_t> C(static_cast<std::size_t>(m * ldc), 0xdeadbeefdeadbeefull);

  rns8_context* ctx = create_wrap64();
  auto desc = wrap_desc(m, n, k);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(ctx, &desc, A.data(), k, B.data(), n, C.data(), ldc) == RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      boost::multiprecision::cpp_int exact = 0;
      for (int64_t kk = 0; kk < k; ++kk) {
        exact += boost::multiprecision::cpp_int(A[row * k + kk]) * boost::multiprecision::cpp_int(B[kk * n + col]);
      }
      CHECK(C[row * ldc + col] == low64(exact));
    }
    CHECK(C[row * ldc + n] == 0xdeadbeefdeadbeefull);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("public wrap64 one-shot matches byte-limb oracle for carry-heavy padded data") {
  constexpr int64_t m = 3;
  constexpr int64_t n = 2;
  constexpr int64_t k = 17;
  constexpr int64_t lda = 19;
  constexpr int64_t ldb = 5;
  constexpr int64_t ldc = 4;
  constexpr uint64_t c_sentinel = 0xfacefeedfacefeedull;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> C(static_cast<std::size_t>(m * ldc), c_sentinel);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] =
          row == 0 ? std::numeric_limits<uint64_t>::max()
                   : row == 1 ? 0x8080808080808080ull : 0xfefdfcfbfaf9f8f7ull;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] =
          col == 0 ? std::numeric_limits<uint64_t>::max() : 0x7f807f807f807f80ull;
    }
  }

  rns8_context* ctx = create_wrap64();
  auto desc = wrap_desc(m, n, k);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, C.data(), ldc) == RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected =
          rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      CHECK(C[static_cast<std::size_t>(row * ldc + col)] == expected);
      CHECK(C[static_cast<std::size_t>(row * ldc + col)] ==
            rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k));
    }
    for (int64_t col = n; col < ldc; ++col) {
      CHECK(C[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
    }
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("public wrap64 persistent matrices use byte-limb low-64-bit semantics") {
  constexpr int64_t m = 2;
  constexpr int64_t n = 3;
  constexpr int64_t k = 5;
  constexpr int64_t ldc = 4;
  const std::vector<uint64_t> A = {
      0,
      1,
      std::numeric_limits<uint64_t>::max(),
      0x8080808080808080ull,
      0x0102030405060708ull,
      255,
      256,
      std::numeric_limits<uint64_t>::max() - 1,
      0x7f7f7f7f7f7f7f7full,
      17};
  const std::vector<uint64_t> B = {
      3,
      std::numeric_limits<uint64_t>::max(),
      0x1112131415161718ull,
      29,
      0x8080808080808080ull,
      31,
      0x0101010101010101ull,
      0xfefdfcfbfaf9f8f7ull,
      37,
      41,
      43,
      47,
      53,
      59,
      61};
  std::vector<uint64_t> C(static_cast<std::size_t>(m * ldc), 0xdeadbeefdeadbeefull);

  rns8_context* ctx = create_wrap64();
  auto desc = wrap_desc(m, n, k);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(plan->prefix == 0);
  REQUIRE(plan->modulus_product == 0);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  auto a_desc = wrap_matrix_desc(m, k);
  auto b_desc = wrap_matrix_desc(k, n);
  auto c_desc = wrap_matrix_desc(m, n);
  c_desc.logical_ld = ldc;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(a_matrix->byte_limbs.size() == static_cast<std::size_t>(m * k * 8));
  REQUIRE(a_matrix->residues.empty());

  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A.data(), k, 7) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B.data(), n, 11) == RNS8_SUCCESS);
  CHECK(a_matrix->source_version == 7);
  CHECK(b_matrix->source_version == 11);

  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  CHECK(rns8_export_u64(ctx, plan, c_matrix, C.data(), ldc) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_export_wrap_u64(ctx, plan, c_matrix, C.data(), ldc) == RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      boost::multiprecision::cpp_int exact = 0;
      for (int64_t kk = 0; kk < k; ++kk) {
        exact += boost::multiprecision::cpp_int(A[row * k + kk]) * boost::multiprecision::cpp_int(B[kk * n + col]);
      }
      CHECK(C[row * ldc + col] == low64(exact));
    }
    CHECK(C[row * ldc + n] == 0xdeadbeefdeadbeefull);
  }

  rns8_destroy_workspace(workspace);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("public wrap64 persistent matrices use compact byte limbs independent of padded host storage") {
  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 3;
  constexpr int64_t lda = 5;
  constexpr int64_t ldb = 4;
  constexpr int64_t ldc = 3;
  constexpr uint64_t c_sentinel = 0xfeedfacefeedfaceull;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> C(static_cast<std::size_t>(m * ldc), c_sentinel);

  A[0] = 0;
  A[1] = std::numeric_limits<uint64_t>::max();
  A[2] = 0x8080808080808080ull;
  A[static_cast<std::size_t>(lda)] = 0x0102030405060708ull;
  A[static_cast<std::size_t>(lda + 1)] = std::numeric_limits<uint64_t>::max() - 1;
  A[static_cast<std::size_t>(lda + 2)] = 0x7f807f807f807f80ull;
  B[0] = std::numeric_limits<uint64_t>::max();
  B[1] = 3;
  B[static_cast<std::size_t>(ldb)] = 0xfefdfcfbfaf9f8f7ull;
  B[static_cast<std::size_t>(ldb + 1)] = 5;
  B[static_cast<std::size_t>(2 * ldb)] = 0x1112131415161718ull;
  B[static_cast<std::size_t>(2 * ldb + 1)] = 7;
  const std::vector<uint64_t> packed_A = A;
  const std::vector<uint64_t> packed_B = B;

  rns8_context* ctx = create_wrap64();
  auto desc = wrap_desc(m, n, k);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  auto a_desc = wrap_matrix_desc(m, k);
  auto b_desc = wrap_matrix_desc(k, n);
  auto c_desc = wrap_matrix_desc(m, n);
  a_desc.logical_ld = lda;
  b_desc.logical_ld = ldb;
  c_desc.logical_ld = ldc;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(a_matrix->byte_limbs.size() == static_cast<std::size_t>(m * k * 8));
  REQUIRE(b_matrix->byte_limbs.size() == static_cast<std::size_t>(k * n * 8));
  REQUIRE(c_matrix->byte_limbs.size() == static_cast<std::size_t>(m * n * 8));

  REQUIRE(rns8_pack_u64(ctx, a_matrix, A.data(), lda, 10) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B.data(), ldb, 11) == RNS8_SUCCESS);
  std::fill(A.begin(), A.end(), 0x1111111111111111ull);
  std::fill(B.begin(), B.end(), 0x2222222222222222ull);

  REQUIRE(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(ctx, plan, c_matrix, C.data(), ldc) == RNS8_SUCCESS);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(C[static_cast<std::size_t>(row * ldc + col)] ==
            rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(
                packed_A.data(), lda, packed_B.data(), ldb, row, col, k));
    }
    CHECK(C[static_cast<std::size_t>(row * ldc + n)] == c_sentinel);
  }

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("public wrap64 CPU path reuses resident byte-limb storage for padded full-width data") {
  constexpr int64_t m = 5;
  constexpr int64_t n = 4;
  constexpr int64_t k = 9;
  constexpr int64_t lda = 11;
  constexpr int64_t ldb = 7;
  constexpr int64_t ldc = 6;
  constexpr uint64_t c_sentinel = 0xc0dec0dec0dec0deull;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> C(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::mt19937_64 rng(0x7772617036345f63ull);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] = rng();
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] = rng();
    }
  }
  A[0] = 0;
  A[1] = std::numeric_limits<uint64_t>::max();
  A[2] = 0x8080808080808080ull;
  A[static_cast<std::size_t>((m - 1) * lda + (k - 1))] = std::numeric_limits<uint64_t>::max() - 1;
  B[0] = std::numeric_limits<uint64_t>::max();
  B[1] = 1;
  B[2] = 0xfefdfcfbfaf9f8f7ull;
  B[static_cast<std::size_t>((k - 1) * ldb + (n - 1))] = 0x7f807f807f807f80ull;

  rns8_context* ctx = create_wrap64();
  auto desc = wrap_desc(m, n, k);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  auto a_desc = wrap_matrix_desc(m, k);
  auto b_desc = wrap_matrix_desc(k, n);
  auto c_desc = wrap_matrix_desc(m, n);
  a_desc.logical_ld = lda;
  b_desc.logical_ld = ldb;
  c_desc.logical_ld = ldc;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  const auto* a_bytes = a_matrix->byte_limbs.data();
  const auto* b_bytes = b_matrix->byte_limbs.data();
  const auto* c_bytes = c_matrix->byte_limbs.data();
  REQUIRE(a_matrix->byte_limbs.size() == static_cast<std::size_t>(m * k * 8));
  REQUIRE(b_matrix->byte_limbs.size() == static_cast<std::size_t>(k * n * 8));
  REQUIRE(c_matrix->byte_limbs.size() == static_cast<std::size_t>(m * n * 8));
  CHECK(a_matrix->desc.logical_ld == lda);
  CHECK(b_matrix->desc.logical_ld == ldb);
  CHECK(c_matrix->desc.logical_ld == ldc);
  CHECK(a_matrix->residues.empty());
  CHECK(b_matrix->residues.empty());
  CHECK(c_matrix->residues.empty());

  auto run_and_check = [&](uint64_t a_version, uint64_t b_version) {
    std::fill(C.begin(), C.end(), c_sentinel);
    std::fill(oneshot.begin(), oneshot.end(), c_sentinel);
    REQUIRE(rns8_pack_u64(ctx, a_matrix, A.data(), lda, a_version) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(ctx, b_matrix, B.data(), ldb, b_version) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_wrap_u64(ctx, plan, c_matrix, C.data(), ldc) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_wrap_u64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, oneshot.data(), ldc) ==
            RNS8_SUCCESS);
    CHECK(a_matrix->byte_limbs.data() == a_bytes);
    CHECK(b_matrix->byte_limbs.data() == b_bytes);
    CHECK(c_matrix->byte_limbs.data() == c_bytes);
    CHECK(a_matrix->source_version == a_version);
    CHECK(b_matrix->source_version == b_version);
    CHECK(C == oneshot);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const std::size_t out_index = static_cast<std::size_t>(row * ldc + col);
        const uint64_t expected = expected_wrap_cell(A, lda, B, ldb, row, col, k);
        CHECK(C[out_index] == expected);
        CHECK(C[out_index] == rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k));
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(C[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
      }
    }
  };

  run_and_check(101, 102);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] ^=
          0x9e3779b97f4a7c15ull + static_cast<uint64_t>(row * 131 + col);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] ^=
          0xbf58476d1ce4e5b9ull + static_cast<uint64_t>(row * 257 + col);
    }
  }
  run_and_check(201, 202);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

