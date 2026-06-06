TEST_CASE("wrap64 byte-limb product matches low 64-bit multiprecision product") {
  const std::vector<uint64_t> values = {
      0,
      1,
      255,
      256,
      0x0102030405060708ull,
      0x8080808080808080ull,
      std::numeric_limits<uint64_t>::max() - 1,
      std::numeric_limits<uint64_t>::max()};

  for (const uint64_t a : values) {
    for (const uint64_t b : values) {
      const uint64_t expected = low64(boost::multiprecision::cpp_int(a) * boost::multiprecision::cpp_int(b));
      CHECK(rns8::detail::wrap64_byte_limb_product(a, b) == expected);
    }
  }
}

TEST_CASE("wrap64 signed-int8 byte correction recovers unsigned byte products") {
  CHECK(rns8::detail::wrap64_signed_i8_product_correction(0x7fu, 0x7fu) == 0);
  CHECK(rns8::detail::wrap64_signed_i8_product_correction(0x80u, 0x7fu) == 256 * 0x7f);
  CHECK(rns8::detail::wrap64_signed_i8_product_correction(0x7fu, 0x80u) == 256 * 0x7f);
  CHECK(rns8::detail::wrap64_signed_i8_product_correction(0xffu, 0xffu) == 256 * 0xff + 256 * 0xff - 65536);

  for (uint32_t a = 0; a <= 0xffu; ++a) {
    for (uint32_t b = 0; b <= 0xffu; ++b) {
      const auto a_byte = static_cast<uint8_t>(a);
      const auto b_byte = static_cast<uint8_t>(b);
      const int32_t signed_product = rns8::detail::wrap64_signed_i8_lane_value(a_byte) *
                                     rns8::detail::wrap64_signed_i8_lane_value(b_byte);
      const int32_t corrected = signed_product + rns8::detail::wrap64_signed_i8_product_correction(a_byte, b_byte);
      CHECK(corrected == static_cast<int32_t>(a * b));
      CHECK(rns8::detail::wrap64_unsigned_byte_product_from_signed_i8(a_byte, b_byte) == a * b);
    }
  }
}

TEST_CASE("wrap64 signed-int8 correction composes through Comba diagonals") {
  const std::vector<uint64_t> values = {
      0x0000000000000000ull,
      0x00000000000000ffull,
      0x8080808080808080ull,
      0x7f807f807f807f80ull,
      0xfefdfcfbfaf9f8f7ull,
      std::numeric_limits<uint64_t>::max()};

  for (const uint64_t a : values) {
    for (const uint64_t b : values) {
      CHECK(corrected_signed_i8_comba_product(a, b) == rns8::detail::wrap64_byte_limb_product(a, b));
    }
  }
}

TEST_CASE("wrap64 byte-limb GEMM cell matches multiprecision modulo 2^64") {
  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 3;
  const uint64_t A[] = {
      std::numeric_limits<uint64_t>::max(),
      0x0102030405060708ull,
      17,
      0x8080808080808080ull,
      std::numeric_limits<uint64_t>::max() - 1,
      255};
  const uint64_t B[] = {
      3,
      std::numeric_limits<uint64_t>::max(),
      0x1112131415161718ull,
      29,
      0x8080808080808080ull,
      31};

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      boost::multiprecision::cpp_int exact = 0;
      for (int64_t kk = 0; kk < k; ++kk) {
        exact += boost::multiprecision::cpp_int(A[row * k + kk]) * boost::multiprecision::cpp_int(B[kk * n + col]);
      }
      CHECK(rns8::detail::wrap64_byte_limb_gemm_cell(A, k, B, n, row, col, k) == low64(exact));
    }
  }
}

TEST_CASE("wrap64 36-byte-pair oracle matches Comba and multiprecision") {
  constexpr int64_t m = 2;
  constexpr int64_t n = 3;
  constexpr int64_t k = 4;
  constexpr int64_t lda = 5;
  constexpr int64_t ldb = 4;
  const std::vector<uint64_t> A = {
      0,
      std::numeric_limits<uint64_t>::max(),
      0x8080808080808080ull,
      0x0102030405060708ull,
      0xaaaaaaaaaaaaaaaaull,
      0xfefdfcfbfaf9f8f7ull,
      17,
      0x7f807f807f807f80ull,
      255,
      0xbbbbbbbbbbbbbbbbull};
  const std::vector<uint64_t> B = {
      3,
      std::numeric_limits<uint64_t>::max(),
      0x1112131415161718ull,
      0xccccccccccccccccull,
      0x8080808080808080ull,
      31,
      0xfefdfcfbfaf9f8f7ull,
      0xddddddddddddddddull,
      0x0101010101010101ull,
      37,
      41,
      0xeeeeeeeeeeeeeeeeull,
      43,
      47,
      53,
      0xffffffffffffffffull};

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      boost::multiprecision::cpp_int exact = 0;
      for (int64_t kk = 0; kk < k; ++kk) {
        exact += boost::multiprecision::cpp_int(A[row * lda + kk]) * boost::multiprecision::cpp_int(B[kk * ldb + col]);
      }
      const uint64_t oracle =
          rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      CHECK(oracle == low64(exact));
      CHECK(oracle == rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k));
    }
  }
}

TEST_CASE("wrap64 36-byte-pair oracle covers full-width randomized signed-int8 correction") {
  constexpr int64_t m = 4;
  constexpr int64_t n = 5;
  constexpr int64_t k = 13;
  constexpr int64_t lda = 17;
  constexpr int64_t ldb = 9;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), 0xbbbbbbbbbbbbbbbbull);
  std::mt19937_64 rng(0x36345f7369676e64ull);

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
  A[0] = std::numeric_limits<uint64_t>::max();
  A[1] = 0x8080808080808080ull;
  A[2] = 0x7f807f807f807f80ull;
  B[0] = std::numeric_limits<uint64_t>::max();
  B[1] = 0xfefdfcfbfaf9f8f7ull;
  B[2] = 0x0102030405060708ull;

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t oracle =
          rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      CHECK(oracle == expected_wrap_cell(A, lda, B, ldb, row, col, k));
      CHECK(oracle == rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k));
    }
  }
}

