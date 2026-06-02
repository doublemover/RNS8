#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "core/internal.hpp"
#include "rns8/rns8.h"

namespace {

uint64_t low64(boost::multiprecision::cpp_int value) {
  const boost::multiprecision::cpp_int modulus = boost::multiprecision::cpp_int(1) << 64;
  value %= modulus;
  if (value < 0) {
    value += modulus;
  }
  return static_cast<uint64_t>(value);
}

rns8_context* create_wrap64() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8_context* create_cpu() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8_gemm_desc wrap_desc(int64_t m, int64_t n, int64_t k) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  return desc;
}

rns8_matrix_desc wrap_matrix_desc(int64_t rows, int64_t cols) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  return desc;
}

uint64_t corrected_signed_i8_comba_product(uint64_t a, uint64_t b) {
  uint64_t out = 0;
  uint64_t carry = 0;
  for (uint32_t diagonal = 0; diagonal < 8; ++diagonal) {
    uint64_t column = carry;
    for (uint32_t i = 0; i <= diagonal; ++i) {
      const uint32_t j = diagonal - i;
      const auto a_byte = static_cast<uint8_t>((a >> (8u * i)) & 0xffu);
      const auto b_byte = static_cast<uint8_t>((b >> (8u * j)) & 0xffu);
      column += rns8::detail::wrap64_unsigned_byte_product_from_signed_i8(a_byte, b_byte);
    }
    out |= (column & 0xffu) << (8u * diagonal);
    carry = column >> 8u;
  }
  return out;
}

uint64_t expected_wrap_cell(
    const std::vector<uint64_t>& A,
    int64_t lda,
    const std::vector<uint64_t>& B,
    int64_t ldb,
    int64_t row,
    int64_t col,
    int64_t k) {
  boost::multiprecision::cpp_int exact = 0;
  for (int64_t kk = 0; kk < k; ++kk) {
    exact += boost::multiprecision::cpp_int(A[static_cast<std::size_t>(row * lda + kk)]) *
             boost::multiprecision::cpp_int(B[static_cast<std::size_t>(kk * ldb + col)]);
  }
  return low64(exact);
}

}  // namespace

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
  for (uint32_t a = 0; a <= 0xffu; ++a) {
    for (uint32_t b = 0; b <= 0xffu; ++b) {
      const auto a_byte = static_cast<uint8_t>(a);
      const auto b_byte = static_cast<uint8_t>(b);
      const int32_t signed_product =
          static_cast<int32_t>(static_cast<int8_t>(a_byte)) * static_cast<int32_t>(static_cast<int8_t>(b_byte));
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

TEST_CASE("wrap64 36-byte-GEMM oracle matches Comba and multiprecision") {
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
      const uint64_t oracle = rns8::detail::wrap64_byte_gemm36_cell(A.data(), lda, B.data(), ldb, row, col, k);
      CHECK(oracle == low64(exact));
      CHECK(oracle == rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k));
    }
  }
}

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
      const uint64_t expected = rns8::detail::wrap64_byte_gemm36_cell(A.data(), lda, B.data(), ldb, row, col, k);
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

TEST_CASE("public wrap64 CPU path rejects residue-backed and stale byte-limb matrices") {
  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 1;
  const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
  const uint64_t B[] = {3};

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
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B, n, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  c_matrix->residues.assign(1, 0);
  CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  c_matrix->residues.clear();

  c_matrix->byte_limbs.clear();
  CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  c_matrix->byte_limbs.assign(static_cast<std::size_t>(m * n * 8), 0);

  a_matrix->host_byte_limbs_current = false;
  CHECK(rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  a_matrix->host_byte_limbs_current = true;

  c_matrix->host_byte_limbs_current = false;
  uint64_t C[] = {0};
  CHECK(rns8_export_wrap_u64(ctx, plan, c_matrix, C, n) == RNS8_INTERNAL_ERROR);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("public wrap64 path rejects CRT metadata and RNS APIs") {
  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 1;
  const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
  const uint64_t B[] = {2};
  uint64_t C[] = {0};

  rns8_context* wrap_ctx = create_wrap64();
  rns8_context* cpu_ctx = create_cpu();
  auto desc = wrap_desc(m, n, k);

  CHECK(rns8_gemm_wrap_u64_oneshot(cpu_ctx, &desc, A, k, B, n, C, n) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(rns8_gemm_u64_oneshot(wrap_ctx, &desc, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
  const int64_t signed_A[] = {-1};
  const int64_t signed_B[] = {2};
  int64_t signed_C[] = {0};
  CHECK(rns8_gemm_i64_oneshot(wrap_ctx, &desc, signed_A, k, signed_B, n, signed_C, n) == RNS8_INVALID_ARGUMENT);

  auto bounded_looking = desc;
  bounded_looking.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  bounded_looking.bound = std::numeric_limits<uint64_t>::max();
  CHECK(rns8_gemm_wrap_u64_oneshot(wrap_ctx, &bounded_looking, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(wrap_ctx, &bounded_looking, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto prefixed = desc;
  prefixed.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  CHECK(rns8_gemm_wrap_u64_oneshot(wrap_ctx, &prefixed, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_create_plan(wrap_ctx, &prefixed, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  CHECK(rns8_create_plan(cpu_ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);

  auto bound_only = desc;
  bound_only.bound = 1;
  CHECK(rns8_gemm_wrap_u64_oneshot(wrap_ctx, &bound_only, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_create_plan(wrap_ctx, &bound_only, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  rns8_gemm_desc bounded_desc{};
  bounded_desc.struct_size = sizeof(bounded_desc);
  bounded_desc.abi_version = RNS8_ABI_VERSION;
  bounded_desc.semantics = RNS8_BOUNDED_U64;
  bounded_desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  bounded_desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  bounded_desc.m = m;
  bounded_desc.n = n;
  bounded_desc.k = k;
  bounded_desc.bound = 2;
  CHECK(rns8_gemm_wrap_u64_oneshot(wrap_ctx, &bounded_desc, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_create_plan(wrap_ctx, &bounded_desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);

  rns8_matrix_desc matrix{};
  matrix.struct_size = sizeof(matrix);
  matrix.abi_version = RNS8_ABI_VERSION;
  matrix.rows = 1;
  matrix.cols = 1;
  matrix.logical_ld = 1;
  matrix.semantics = RNS8_WRAP_U64_MOD_2_64;
  matrix.bound_kind = RNS8_BOUND_NONE;
  matrix.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  rns8_matrix* storage = nullptr;
  CHECK(rns8_create_matrix(cpu_ctx, &matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(storage == nullptr);
  matrix.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  CHECK(rns8_create_matrix(wrap_ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
  CHECK(storage == nullptr);
  matrix.bound_kind = RNS8_BOUND_NONE;
  matrix.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  CHECK(rns8_create_matrix(wrap_ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
  CHECK(storage == nullptr);

  rns8_plan* valid_plan = nullptr;
  REQUIRE(rns8_create_plan(wrap_ctx, &desc, &valid_plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(wrap_ctx, valid_plan, &workspace) == RNS8_SUCCESS);
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  auto a_desc = wrap_matrix_desc(m, k);
  auto b_desc = wrap_matrix_desc(k, n);
  auto c_desc = wrap_matrix_desc(m, n);
  REQUIRE(rns8_create_matrix(wrap_ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(wrap_ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(wrap_ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  uint64_t limbs[] = {0};
  CHECK(rns8_pack_i64(wrap_ctx, a_matrix, signed_A, k, 0) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(wrap_ctx, valid_plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(wrap_ctx, valid_plan, c_matrix, limbs, n, 1) == RNS8_INVALID_ARGUMENT);

  auto bounded_matrix = wrap_matrix_desc(m, n);
  bounded_matrix.semantics = RNS8_BOUNDED_U64;
  bounded_matrix.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  bounded_matrix.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  CHECK(rns8_create_matrix(wrap_ctx, &bounded_matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(storage == nullptr);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(valid_plan);

  rns8_destroy_context(cpu_ctx);
  rns8_destroy_context(wrap_ctx);
}
