#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <cstdint>
#include <limits>
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

  auto bounded_looking = desc;
  bounded_looking.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  bounded_looking.bound = std::numeric_limits<uint64_t>::max();
  CHECK(rns8_gemm_wrap_u64_oneshot(wrap_ctx, &bounded_looking, A, k, B, n, C, n) == RNS8_UNSUPPORTED_BACKEND);
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(wrap_ctx, &bounded_looking, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);

  auto prefixed = desc;
  prefixed.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  CHECK(rns8_gemm_wrap_u64_oneshot(wrap_ctx, &prefixed, A, k, B, n, C, n) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(rns8_create_plan(wrap_ctx, &prefixed, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);

  CHECK(rns8_create_plan(cpu_ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
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
  CHECK(rns8_create_matrix(wrap_ctx, &matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(storage == nullptr);

  rns8_destroy_context(cpu_ctx);
  rns8_destroy_context(wrap_ctx);
}
