#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "rns8/rns8.h"

namespace {

using boost::multiprecision::cpp_int;

rns8_context* create_cpu() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8_gemm_desc i64_desc(int64_t m, int64_t n, int64_t k, uint64_t bound) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc u64_desc(int64_t m, int64_t n, int64_t k, uint64_t bound) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

cpp_int abs_cpp(cpp_int value) {
  return value < 0 ? -value : value;
}

cpp_int exact_i64_cell(
    const std::vector<int64_t>& A,
    int64_t lda,
    const std::vector<int64_t>& B,
    int64_t ldb,
    int64_t row,
    int64_t col,
    int64_t k) {
  cpp_int acc = 0;
  for (int64_t kk = 0; kk < k; ++kk) {
    acc += cpp_int(A[static_cast<std::size_t>(row * lda + kk)]) *
           cpp_int(B[static_cast<std::size_t>(kk * ldb + col)]);
  }
  return acc;
}

cpp_int exact_u64_cell(
    const std::vector<uint64_t>& A,
    int64_t lda,
    const std::vector<uint64_t>& B,
    int64_t ldb,
    int64_t row,
    int64_t col,
    int64_t k) {
  cpp_int acc = 0;
  for (int64_t kk = 0; kk < k; ++kk) {
    acc += cpp_int(A[static_cast<std::size_t>(row * lda + kk)]) *
           cpp_int(B[static_cast<std::size_t>(kk * ldb + col)]);
  }
  return acc;
}

uint64_t checked_u64_bound(const cpp_int& value) {
  REQUIRE(value >= 0);
  REQUIRE(value <= cpp_int(std::numeric_limits<uint64_t>::max()));
  return static_cast<uint64_t>(value);
}

void assert_i64_public_matches_exact(
    rns8_context* ctx,
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<int64_t>& A,
    int64_t lda,
    const std::vector<int64_t>& B,
    int64_t ldb) {
  std::vector<cpp_int> exact(static_cast<std::size_t>(m * n));
  cpp_int max_abs = 0;
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      cpp_int value = exact_i64_cell(A, lda, B, ldb, row, col, k);
      exact[static_cast<std::size_t>(row * n + col)] = value;
      max_abs = std::max(max_abs, abs_cpp(value));
    }
  }

  std::vector<int64_t> C(static_cast<std::size_t>(m * n), 0);
  auto desc = i64_desc(m, n, k, checked_u64_bound(max_abs));
  REQUIRE(rns8_gemm_i64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, C.data(), n) == RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const cpp_int got = C[static_cast<std::size_t>(row * n + col)];
      CHECK(got == exact[static_cast<std::size_t>(row * n + col)]);
    }
  }
}

void assert_u64_public_matches_exact(
    rns8_context* ctx,
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<uint64_t>& A,
    int64_t lda,
    const std::vector<uint64_t>& B,
    int64_t ldb) {
  std::vector<cpp_int> exact(static_cast<std::size_t>(m * n));
  cpp_int max_value = 0;
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      cpp_int value = exact_u64_cell(A, lda, B, ldb, row, col, k);
      exact[static_cast<std::size_t>(row * n + col)] = value;
      max_value = std::max(max_value, value);
    }
  }

  std::vector<uint64_t> C(static_cast<std::size_t>(m * n), 0);
  auto desc = u64_desc(m, n, k, checked_u64_bound(max_value));
  REQUIRE(rns8_gemm_u64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, C.data(), n) == RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const cpp_int got = C[static_cast<std::size_t>(row * n + col)];
      CHECK(got == exact[static_cast<std::size_t>(row * n + col)]);
    }
  }
}

}  // namespace

TEST_CASE("bounded i64 CPU reference covers all tiny dimensions 1 through 8") {
  rns8_context* ctx = create_cpu();
  for (int64_t m = 1; m <= 8; ++m) {
    for (int64_t n = 1; n <= 8; ++n) {
      for (int64_t k = 1; k <= 8; ++k) {
        std::vector<int64_t> A(static_cast<std::size_t>(m * k));
        std::vector<int64_t> B(static_cast<std::size_t>(k * n));
        for (int64_t row = 0; row < m; ++row) {
          for (int64_t col = 0; col < k; ++col) {
            A[static_cast<std::size_t>(row * k + col)] = ((row * 17 + col * 5 + m - n) % 11) - 5;
          }
        }
        for (int64_t row = 0; row < k; ++row) {
          for (int64_t col = 0; col < n; ++col) {
            B[static_cast<std::size_t>(row * n + col)] = ((row * 7 - col * 13 + k + n) % 13) - 6;
          }
        }
        assert_i64_public_matches_exact(ctx, m, n, k, A, k, B, n);
      }
    }
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded u64 CPU reference covers all tiny dimensions 1 through 8") {
  rns8_context* ctx = create_cpu();
  for (int64_t m = 1; m <= 8; ++m) {
    for (int64_t n = 1; n <= 8; ++n) {
      for (int64_t k = 1; k <= 8; ++k) {
        std::vector<uint64_t> A(static_cast<std::size_t>(m * k));
        std::vector<uint64_t> B(static_cast<std::size_t>(k * n));
        for (int64_t row = 0; row < m; ++row) {
          for (int64_t col = 0; col < k; ++col) {
            A[static_cast<std::size_t>(row * k + col)] = static_cast<uint64_t>((row * 19 + col * 3 + m + 1) % 17);
          }
        }
        for (int64_t row = 0; row < k; ++row) {
          for (int64_t col = 0; col < n; ++col) {
            B[static_cast<std::size_t>(row * n + col)] = static_cast<uint64_t>((row * 5 + col * 11 + n + 2) % 19);
          }
        }
        assert_u64_public_matches_exact(ctx, m, n, k, A, k, B, n);
      }
    }
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded CPU reference matches fixed-seed random signed and unsigned cases") {
  rns8_context* ctx = create_cpu();
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

    assert_i64_public_matches_exact(ctx, m, n, k, signed_a, lda, signed_b, ldb);
    assert_u64_public_matches_exact(ctx, m, n, k, unsigned_a, lda, unsigned_b, ldb);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded i64 CPU reference handles worst-case centered accumulation around K block") {
  rns8_context* ctx = create_cpu();
  for (int64_t k : {static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK),
                    static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1}) {
    {
      std::vector<int64_t> A(static_cast<std::size_t>(k), 127);
      std::vector<int64_t> B(static_cast<std::size_t>(k), 127);
      assert_i64_public_matches_exact(ctx, 1, 1, k, A, k, B, 1);
    }
    {
      std::vector<int64_t> A(static_cast<std::size_t>(k), -128);
      std::vector<int64_t> B(static_cast<std::size_t>(k), 127);
      assert_i64_public_matches_exact(ctx, 1, 1, k, A, k, B, 1);
    }
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded u64 CPU reference handles accumulation around K block") {
  rns8_context* ctx = create_cpu();
  for (int64_t k : {static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK),
                    static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1}) {
    std::vector<uint64_t> A(static_cast<std::size_t>(k), 255);
    std::vector<uint64_t> B(static_cast<std::size_t>(k), 255);
    assert_u64_public_matches_exact(ctx, 1, 1, k, A, k, B, 1);
  }
  rns8_destroy_context(ctx);
}
