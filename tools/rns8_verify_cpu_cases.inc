bool checked_u64_bound(const cpp_int& value, uint64_t& out) {
  if (value < 0 || value > cpp_int(std::numeric_limits<uint64_t>::max())) {
    return false;
  }
  out = static_cast<uint64_t>(value);
  return true;
}

bool verify_signed_public_case(
    rns8_context* ctx,
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<int64_t>& A,
    int64_t lda,
    const std::vector<int64_t>& B,
    int64_t ldb,
    const char* label) {
  std::vector<cpp_int> expected(static_cast<std::size_t>(m * n));
  cpp_int max_abs = 0;
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const cpp_int value = rns8::detail::exact_i64_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      if (value < i64_min_cpp() || value > cpp_int(std::numeric_limits<int64_t>::max())) {
        std::cerr << label << " expected signed output exceeds int64 range\n";
        return false;
      }
      expected[static_cast<std::size_t>(row * n + col)] = value;
      max_abs = std::max(max_abs, abs_cpp(value));
    }
  }

  uint64_t bound = 0;
  if (!checked_u64_bound(max_abs, bound)) {
    std::cerr << label << " signed bound exceeds uint64 range\n";
    return false;
  }

  std::vector<int64_t> C(static_cast<std::size_t>(m * n), 0);
  auto desc = signed_desc(m, n, k, bound);
  const rns8_status status = rns8_gemm_i64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, C.data(), n);
  if (status != RNS8_SUCCESS) {
    std::cerr << label << " bounded i64 status failed: " << rns8_status_string(status) << "\n";
    return false;
  }

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      if (cpp_int(C[static_cast<std::size_t>(row * n + col)]) !=
          expected[static_cast<std::size_t>(row * n + col)]) {
        std::cerr << label << " bounded i64 value mismatch at (" << row << ", " << col << ")\n";
        return false;
      }
    }
  }
  return true;
}

bool verify_unsigned_public_case(
    rns8_context* ctx,
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<uint64_t>& A,
    int64_t lda,
    const std::vector<uint64_t>& B,
    int64_t ldb,
    const char* label) {
  std::vector<cpp_int> expected(static_cast<std::size_t>(m * n));
  cpp_int max_value = 0;
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const cpp_int value = rns8::detail::exact_u64_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      if (value < 0 || value > cpp_int(std::numeric_limits<uint64_t>::max())) {
        std::cerr << label << " expected unsigned output exceeds uint64 range\n";
        return false;
      }
      expected[static_cast<std::size_t>(row * n + col)] = value;
      max_value = std::max(max_value, value);
    }
  }

  uint64_t bound = 0;
  if (!checked_u64_bound(max_value, bound)) {
    std::cerr << label << " unsigned bound exceeds uint64 range\n";
    return false;
  }

  std::vector<uint64_t> C(static_cast<std::size_t>(m * n), 0);
  auto desc = unsigned_desc(m, n, k, bound);
  const rns8_status status = rns8_gemm_u64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, C.data(), n);
  if (status != RNS8_SUCCESS) {
    std::cerr << label << " bounded u64 status failed: " << rns8_status_string(status) << "\n";
    return false;
  }

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      if (cpp_int(C[static_cast<std::size_t>(row * n + col)]) !=
          expected[static_cast<std::size_t>(row * n + col)]) {
        std::cerr << label << " bounded u64 value mismatch at (" << row << ", " << col << ")\n";
        return false;
      }
    }
  }
  return true;
}

int64_t centered_fixture_value(int64_t value, int64_t modulus, int64_t center) {
  int64_t residue = value % modulus;
  if (residue < 0) {
    residue += modulus;
  }
  return residue - center;
}

bool verify_tiny_dimension_sweep(rns8_context* ctx) {
  for (int64_t m = 1; m <= 8; ++m) {
    for (int64_t n = 1; n <= 8; ++n) {
      for (int64_t k = 1; k <= 8; ++k) {
        std::vector<int64_t> signed_a(static_cast<std::size_t>(m * k));
        std::vector<int64_t> signed_b(static_cast<std::size_t>(k * n));
        std::vector<uint64_t> unsigned_a(static_cast<std::size_t>(m * k));
        std::vector<uint64_t> unsigned_b(static_cast<std::size_t>(k * n));
        for (int64_t row = 0; row < m; ++row) {
          for (int64_t col = 0; col < k; ++col) {
            signed_a[static_cast<std::size_t>(row * k + col)] =
                centered_fixture_value(row * 17 + col * 5 + m * 3 - n * 2, 17, 8);
            unsigned_a[static_cast<std::size_t>(row * k + col)] =
                static_cast<uint64_t>((row * 19 + col * 3 + m + 1) % 17);
          }
        }
        for (int64_t row = 0; row < k; ++row) {
          for (int64_t col = 0; col < n; ++col) {
            signed_b[static_cast<std::size_t>(row * n + col)] =
                centered_fixture_value(row * 7 - col * 13 + k * 2 + n, 19, 9);
            unsigned_b[static_cast<std::size_t>(row * n + col)] =
                static_cast<uint64_t>((row * 5 + col * 11 + n + 2) % 19);
          }
        }
        if (!verify_signed_public_case(ctx, m, n, k, signed_a, k, signed_b, n, "tiny dimension sweep") ||
            !verify_unsigned_public_case(ctx, m, n, k, unsigned_a, k, unsigned_b, n, "tiny dimension sweep")) {
          return false;
        }
      }
    }
  }
  return true;
}

bool verify_fixed_seed_random(rns8_context* ctx) {
  std::mt19937_64 rng(0x8a5cd13f00dULL);
  std::uniform_int_distribution<int64_t> signed_dist(-31, 31);
  std::uniform_int_distribution<uint64_t> unsigned_dist(0, 63);

  for (int trial = 0; trial < 32; ++trial) {
    const int64_t m = 1 + static_cast<int64_t>(rng() % 8);
    const int64_t n = 1 + static_cast<int64_t>(rng() % 8);
    const int64_t k = 1 + static_cast<int64_t>(rng() % 8);
    const int64_t lda = k + 1;
    const int64_t ldb = n + 1;
    std::vector<int64_t> signed_a(static_cast<std::size_t>(m * lda), 12345);
    std::vector<int64_t> signed_b(static_cast<std::size_t>(k * ldb), -12345);
    std::vector<uint64_t> unsigned_a(static_cast<std::size_t>(m * lda), 99999);
    std::vector<uint64_t> unsigned_b(static_cast<std::size_t>(k * ldb), 99999);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        signed_a[static_cast<std::size_t>(row * lda + col)] = signed_dist(rng);
        unsigned_a[static_cast<std::size_t>(row * lda + col)] = unsigned_dist(rng);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        signed_b[static_cast<std::size_t>(row * ldb + col)] = signed_dist(rng);
        unsigned_b[static_cast<std::size_t>(row * ldb + col)] = unsigned_dist(rng);
      }
    }
    if (!verify_signed_public_case(ctx, m, n, k, signed_a, lda, signed_b, ldb, "fixed-seed random sweep") ||
        !verify_unsigned_public_case(ctx, m, n, k, unsigned_a, lda, unsigned_b, ldb, "fixed-seed random sweep")) {
      return false;
    }
  }
  return true;
}

bool verify_k_block_cases(rns8_context* ctx) {
  for (int64_t k : {static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK),
                    static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1}) {
    {
      std::vector<int64_t> A(static_cast<std::size_t>(k), 127);
      std::vector<int64_t> B(static_cast<std::size_t>(k), 127);
      if (!verify_signed_public_case(ctx, 1, 1, k, A, k, B, 1, "signed positive K-block sweep")) {
        return false;
      }
    }
    {
      std::vector<int64_t> A(static_cast<std::size_t>(k), -128);
      std::vector<int64_t> B(static_cast<std::size_t>(k), 127);
      if (!verify_signed_public_case(ctx, 1, 1, k, A, k, B, 1, "signed negative K-block sweep")) {
        return false;
      }
    }
    {
      std::vector<uint64_t> A(static_cast<std::size_t>(k), 255);
      std::vector<uint64_t> B(static_cast<std::size_t>(k), 255);
      if (!verify_unsigned_public_case(ctx, 1, 1, k, A, k, B, 1, "unsigned K-block sweep")) {
        return false;
      }
    }
  }
  return true;
}

bool verify_cpu() {
  if (rns8_validate_default_moduli() != RNS8_SUCCESS) {
    std::cerr << "default modulus ladder is not pairwise coprime\n";
    return false;
  }

  rns8_context* ctx = create_cpu_context();
  if (!ctx) {
    std::cerr << "failed to create CPU reference context\n";
    return false;
  }

  {
    const int64_t A[] = {7, -3, 5, -11, 13, 17};
    const int64_t B[] = {2, -5, 19, 23, -29, 31};
    int64_t C[4] = {};
    auto desc = signed_desc(2, 2, 3, 100000);
    const rns8_status status = rns8_gemm_i64_oneshot(ctx, &desc, A, 3, B, 2, C, 2);
    if (status != RNS8_SUCCESS || C[0] != -188 || C[1] != 51 || C[2] != -268 || C[3] != 881) {
      std::cerr << "bounded i64 verification failed: " << rns8_status_string(status) << "\n";
      rns8_destroy_context(ctx);
      return false;
    }
  }

  {
    const int64_t A[] = {std::numeric_limits<int64_t>::max()};
    const int64_t B[] = {1};
    int64_t C[] = {0};
    auto desc = signed_desc(1, 1, 1, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    const rns8_status status = rns8_gemm_i64_oneshot(ctx, &desc, A, 1, B, 1, C, 1);
    if (status != RNS8_SUCCESS || C[0] != std::numeric_limits<int64_t>::max()) {
      std::cerr << "bounded i64 boundary verification failed: " << rns8_status_string(status) << "\n";
      rns8_destroy_context(ctx);
      return false;
    }
  }

  {
    const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
    const uint64_t B[] = {1};
    uint64_t C[] = {0};
    auto desc = unsigned_desc(1, 1, 1, std::numeric_limits<uint64_t>::max());
    const rns8_status status = rns8_gemm_u64_oneshot(ctx, &desc, A, 1, B, 1, C, 1);
    if (status != RNS8_SUCCESS || C[0] != std::numeric_limits<uint64_t>::max()) {
      std::cerr << "bounded u64 boundary verification failed: " << rns8_status_string(status) << "\n";
      rns8_destroy_context(ctx);
      return false;
    }
  }

  if (!verify_tiny_dimension_sweep(ctx)) {
    rns8_destroy_context(ctx);
    return false;
  }

  if (!verify_fixed_seed_random(ctx)) {
    rns8_destroy_context(ctx);
    return false;
  }

  if (!verify_k_block_cases(ctx)) {
    rns8_destroy_context(ctx);
    return false;
  }

  rns8_destroy_context(ctx);
  return true;
}

