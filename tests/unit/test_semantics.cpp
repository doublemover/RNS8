#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/internal.hpp"
#include "rns8/rns8.h"

namespace {

rns8_context* create_cpu() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
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

template <typename Enum>
Enum invalid_public_enum_value() {
  volatile unsigned value = 0x7fffu;
  return static_cast<Enum>(value);
}

rns8_gemm_desc gemm_desc(rns8_semantics semantics, rns8_bound_kind bound_kind) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = bound_kind;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = 1;
  desc.n = 1;
  desc.k = 1;
  desc.bound = 1;
  return desc;
}

rns8_gemm_desc finite_desc(rns8_semantics semantics, int64_t m, int64_t n, int64_t k, uint16_t modulus = 0) {
  auto desc = gemm_desc(semantics, RNS8_BOUND_NONE);
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = 0;
  desc.max_prefix = 0;
  desc.finite_modulus =
      modulus != 0 ? modulus : (semantics == RNS8_FINITE_FIELD_U8 ? uint16_t{251} : uint16_t{255});
  return desc;
}

uint8_t exact_finite_cell(
    const std::vector<uint8_t>& A,
    const std::vector<uint8_t>& B,
    int64_t lda,
    int64_t ldb,
    int64_t k,
    int64_t row,
    int64_t col,
    uint16_t modulus) {
  uint64_t acc = 0;
  for (int64_t kk = 0; kk < k; ++kk) {
    acc += static_cast<uint64_t>(A[static_cast<std::size_t>(row * lda + kk)] % modulus) *
           static_cast<uint64_t>(B[static_cast<std::size_t>(kk * ldb + col)] % modulus);
    acc %= modulus;
  }
  return static_cast<uint8_t>(acc % modulus);
}

rns8_gemm_desc bounded_looking_desc(
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    rns8_backend_kind requested_backend = RNS8_BACKEND_CPU_REFERENCE) {
  auto desc = gemm_desc(semantics, bound_kind);
  desc.requested_backend = requested_backend;
  desc.bound = UINT64_MAX;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

rns8_matrix_desc matrix_desc(rns8_semantics semantics, rns8_bound_kind bound_kind) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = 1;
  desc.cols = 1;
  desc.logical_ld = 1;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind;
  return desc;
}

rns8_matrix_desc matrix_desc(
    int64_t rows,
    int64_t cols,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t max_prefix = 0) {
  auto desc = matrix_desc(semantics, bound_kind);
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.max_prefix = max_prefix;
  return desc;
}

rns8_matrix_desc bounded_looking_matrix_desc(rns8_semantics semantics, rns8_bound_kind bound_kind) {
  auto desc = matrix_desc(semantics, bound_kind);
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

}  // namespace

TEST_CASE("bounded semantics reject bound none explicitly") {
  rns8_context* ctx = create_cpu();
  {
    auto desc = gemm_desc(RNS8_BOUNDED_I64, RNS8_BOUND_NONE);
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_NONE);
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("reserved public ABI flags are hard rejected") {
  {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    options.flags = 1;
    rns8_context* ctx = nullptr;
    CHECK(rns8_create_context(-1, &options, &ctx) == RNS8_INVALID_ARGUMENT);
    CHECK(ctx == nullptr);
  }
  {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_HIPBLASLT;
    options.flags = 1;
    rns8_context* ctx = nullptr;
    CHECK(rns8_create_context(0, &options, &ctx) == RNS8_INVALID_ARGUMENT);
    CHECK(ctx == nullptr);
  }

  rns8_context* ctx = create_cpu();
  {
    auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    desc.flags = 0x80000000u;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    auto matrix = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    matrix.flags = 1;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("bounded one-shot APIs validate ABI and leading dimensions before dispatch") {
  rns8_context* ctx = create_cpu();
  int64_t signed_a[4] = {1, 2, 3, 4};
  int64_t signed_b[4] = {5, 6, 7, 8};
  int64_t signed_c[4] = {};
  uint64_t unsigned_a[4] = {1, 2, 3, 4};
  uint64_t unsigned_b[4] = {5, 6, 7, 8};
  uint64_t unsigned_c[4] = {};

  auto signed_desc = gemm_desc(RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  signed_desc.m = 2;
  signed_desc.n = 2;
  signed_desc.k = 2;
  signed_desc.bound = 100;
  auto invalid_signed_abi = signed_desc;
  invalid_signed_abi.struct_size = 0;
  CHECK(rns8_gemm_i64_oneshot(ctx, &invalid_signed_abi, signed_a, 2, signed_b, 2, signed_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_i64_oneshot(ctx, &signed_desc, signed_a, 1, signed_b, 2, signed_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_i64_oneshot(ctx, &signed_desc, signed_a, 2, signed_b, 1, signed_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_i64_oneshot(ctx, &signed_desc, signed_a, 2, signed_b, 2, signed_c, 1) ==
        RNS8_INVALID_ARGUMENT);
  auto malformed_future_signed = signed_desc;
  malformed_future_signed.requested_backend = RNS8_BACKEND_HIPBLASLT;
  malformed_future_signed.bound_kind = static_cast<rns8_bound_kind>(0x7fffu);
  CHECK(rns8_gemm_i64_oneshot(ctx, &malformed_future_signed, signed_a, 2, signed_b, 2, signed_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  auto valid_future_signed = signed_desc;
  valid_future_signed.requested_backend = RNS8_BACKEND_HIPBLASLT;
  CHECK(rns8_gemm_i64_oneshot(ctx, &valid_future_signed, signed_a, 2, signed_b, 2, signed_c, 2) ==
        RNS8_UNSUPPORTED_BACKEND);

  auto unsigned_desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  unsigned_desc.m = 2;
  unsigned_desc.n = 2;
  unsigned_desc.k = 2;
  unsigned_desc.bound = 100;
  auto invalid_unsigned_abi = unsigned_desc;
  invalid_unsigned_abi.abi_version = 0;
  CHECK(rns8_gemm_u64_oneshot(ctx, &invalid_unsigned_abi, unsigned_a, 2, unsigned_b, 2, unsigned_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_u64_oneshot(ctx, &unsigned_desc, unsigned_a, 1, unsigned_b, 2, unsigned_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_u64_oneshot(ctx, &unsigned_desc, unsigned_a, 2, unsigned_b, 1, unsigned_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_u64_oneshot(ctx, &unsigned_desc, unsigned_a, 2, unsigned_b, 2, unsigned_c, 1) ==
        RNS8_INVALID_ARGUMENT);
  auto malformed_future_unsigned = unsigned_desc;
  malformed_future_unsigned.requested_backend = RNS8_BACKEND_CK;
  malformed_future_unsigned.bound_kind = static_cast<rns8_bound_kind>(0x7fffu);
  CHECK(rns8_gemm_u64_oneshot(ctx, &malformed_future_unsigned, unsigned_a, 2, unsigned_b, 2, unsigned_c, 2) ==
        RNS8_INVALID_ARGUMENT);
  auto valid_future_unsigned = unsigned_desc;
  valid_future_unsigned.requested_backend = RNS8_BACKEND_CK;
  CHECK(rns8_gemm_u64_oneshot(ctx, &valid_future_unsigned, unsigned_a, 2, unsigned_b, 2, unsigned_c, 2) ==
        RNS8_UNSUPPORTED_BACKEND);

  rns8_destroy_context(ctx);
}

TEST_CASE("unsupported semantic contracts do not fall through to bounded CRT") {
  rns8_context* ctx = create_cpu();
  for (const rns8_semantics semantics : {RNS8_WRAP_U64_MOD_2_64}) {
    auto desc = gemm_desc(semantics, RNS8_BOUND_NONE);
    desc.bound = 0;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(plan == nullptr);

    auto matrix = matrix_desc(semantics, RNS8_BOUND_NONE);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(storage == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("public finite ring and field u8 oneshot use explicit modulus contracts") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 2;
  constexpr int64_t n = 3;
  constexpr int64_t k = 4;
  constexpr int64_t lda = 6;
  constexpr int64_t ldb = 5;
  constexpr int64_t ldc = 5;
  const std::vector<uint8_t> A = {
      254, 128, 7, 3, 0xaa, 0xaa,
      5, 250, 251, 1, 0xaa, 0xaa,
  };
  const std::vector<uint8_t> B = {
      2, 3, 4, 0xbb, 0xbb,
      250, 11, 9, 0xbb, 0xbb,
      13, 17, 19, 0xbb, 0xbb,
      23, 29, 31, 0xbb, 0xbb,
  };

  {
    auto desc = finite_desc(RNS8_FINITE_RING_U8, m, n, k);
    std::vector<uint8_t> C(static_cast<std::size_t>(m * ldc), 0xcc);
    REQUIRE(rns8_gemm_finite_ring_u8_oneshot(ctx, &desc, 255, A.data(), lda, B.data(), ldb, C.data(), ldc) ==
            RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(C[static_cast<std::size_t>(row * ldc + col)] == exact_finite_cell(A, B, lda, ldb, k, row, col, 255));
      }
      CHECK(C[static_cast<std::size_t>(row * ldc + n)] == 0xcc);
    }
  }

  {
    auto desc = finite_desc(RNS8_FINITE_FIELD_U8, m, n, k);
    std::vector<uint8_t> C(static_cast<std::size_t>(m * ldc), 0xdd);
    REQUIRE(rns8_gemm_finite_field_u8_oneshot(ctx, &desc, 251, A.data(), lda, B.data(), ldb, C.data(), ldc) ==
            RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(C[static_cast<std::size_t>(row * ldc + col)] == exact_finite_cell(A, B, lda, ldb, k, row, col, 251));
      }
      CHECK(C[static_cast<std::size_t>(row * ldc + n)] == 0xdd);
    }
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("public finite ring and field u8 persistent matrices use explicit modulus contracts") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 2;
  constexpr int64_t n = 3;
  constexpr int64_t k = 4;
  constexpr int64_t lda = 6;
  constexpr int64_t ldb = 5;
  constexpr int64_t ldc = 5;
  const std::vector<uint8_t> A = {
      254, 128, 7, 3, 0xaa, 0xaa,
      5, 250, 251, 1, 0xaa, 0xaa,
  };
  const std::vector<uint8_t> B = {
      2, 3, 4, 0xbb, 0xbb,
      250, 11, 9, 0xbb, 0xbb,
      13, 17, 19, 0xbb, 0xbb,
      23, 29, 31, 0xbb, 0xbb,
  };

  for (const auto item : {std::pair{RNS8_FINITE_RING_U8, uint16_t{255}},
                          std::pair{RNS8_FINITE_FIELD_U8, uint16_t{251}}}) {
    auto desc = finite_desc(item.first, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    rns8_plan_schedule_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
    CHECK(info.min_required_prefix == 0);
    CHECK(info.max_selected_prefix == 0);
    CHECK(info.prefix_group_count == 0);

    auto a_desc = matrix_desc(m, k, item.first, RNS8_BOUND_NONE);
    auto b_desc = matrix_desc(k, n, item.first, RNS8_BOUND_NONE);
    auto c_desc = matrix_desc(m, n, item.first, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(ctx, a_matrix, item.second, A.data(), lda, 11) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(ctx, b_matrix, item.second, B.data(), ldb, 12) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_u8(ctx, plan, item.second, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

    std::vector<uint8_t> C(static_cast<std::size_t>(m * ldc), 0xcc);
    REQUIRE(rns8_export_finite_u8(ctx, plan, item.second, c_matrix, C.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(C[static_cast<std::size_t>(row * ldc + col)] ==
              exact_finite_cell(A, B, lda, ldb, k, row, col, item.second));
      }
      CHECK(C[static_cast<std::size_t>(row * ldc + n)] == 0xcc);
    }

    std::vector<uint64_t> wrong_export(static_cast<std::size_t>(m * ldc), 0);
    CHECK(rns8_export_u64(ctx, plan, c_matrix, wrong_export.data(), ldc) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);

    rns8_destroy_workspace(workspace);
    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("finite u8 oneshot descriptors reject stale CRT metadata and invalid fields") {
  rns8_context* ctx = create_cpu();
  const uint8_t A[1] = {3};
  const uint8_t B[1] = {5};
  uint8_t C[1] = {0};

  auto ring = finite_desc(RNS8_FINITE_RING_U8, 1, 1, 1);
  CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &ring, 1, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &ring, 251, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &ring, 257, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);

  auto field = finite_desc(RNS8_FINITE_FIELD_U8, 1, 1, 1);
  CHECK(rns8_gemm_finite_field_u8_oneshot(ctx, &field, 241, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_finite_field_u8_oneshot(ctx, &field, 255, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_finite_field_u8_oneshot(ctx, &field, 251, A, 0, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);

  auto stale_prefix = ring;
  stale_prefix.max_prefix = 1;
  CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &stale_prefix, 255, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);

  auto stale_bound = ring;
  stale_bound.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  stale_bound.bound = 15;
  CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &stale_bound, 255, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);

  CHECK(rns8_gemm_finite_field_u8_oneshot(ctx, &ring, 251, A, 1, B, 1, C, 1) == RNS8_INVALID_ARGUMENT);

  ring.requested_backend = RNS8_BACKEND_CK;
  CHECK(rns8_gemm_finite_ring_u8_oneshot(ctx, &ring, 255, A, 1, B, 1, C, 1) == RNS8_UNSUPPORTED_BACKEND);

  rns8_destroy_context(ctx);
}

TEST_CASE("finite u8 persistent descriptors reject stale CRT metadata and modulus mismatches") {
  rns8_context* ctx = create_cpu();
  const uint8_t A[1] = {3};
  const uint8_t B[1] = {5};
  uint8_t C[1] = {0};

  auto stale_prefix = finite_desc(RNS8_FINITE_RING_U8, 1, 1, 1);
  stale_prefix.max_prefix = 1;
  rns8_plan* rejected_plan = nullptr;
  CHECK(rns8_create_plan(ctx, &stale_prefix, &rejected_plan) == RNS8_INVALID_ARGUMENT);
  CHECK(rejected_plan == nullptr);

  auto missing_modulus = finite_desc(RNS8_FINITE_RING_U8, 1, 1, 1);
  missing_modulus.finite_modulus = 0;
  CHECK(rns8_create_plan(ctx, &missing_modulus, &rejected_plan) == RNS8_INVALID_ARGUMENT);
  CHECK(rejected_plan == nullptr);

  auto stale_nonfinite_modulus = gemm_desc(RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  stale_nonfinite_modulus.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  stale_nonfinite_modulus.finite_modulus = 251;
  CHECK(rns8_create_plan(ctx, &stale_nonfinite_modulus, &rejected_plan) == RNS8_INVALID_ARGUMENT);
  CHECK(rejected_plan == nullptr);

  auto stale_matrix_desc = matrix_desc(1, 1, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  stale_matrix_desc.max_prefix = 1;
  rns8_matrix* rejected_matrix = nullptr;
  CHECK(rns8_create_matrix(ctx, &stale_matrix_desc, &rejected_matrix) == RNS8_INVALID_ARGUMENT);
  CHECK(rejected_matrix == nullptr);

  auto accelerator = finite_desc(RNS8_FINITE_RING_U8, 1, 1, 1);
  accelerator.requested_backend = RNS8_BACKEND_CK;
  CHECK(rns8_create_plan(ctx, &accelerator, &rejected_plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(rejected_plan == nullptr);

  auto desc = finite_desc(RNS8_FINITE_RING_U8, 1, 1, 1);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  auto matrix = matrix_desc(1, 1, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &matrix, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &matrix, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &matrix, &c_matrix) == RNS8_SUCCESS);

  CHECK(rns8_export_finite_u8(ctx, plan, 255, c_matrix, C, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_pack_finite_u8(ctx, a_matrix, 1, A, 1, 1) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_pack_finite_u8(ctx, a_matrix, 255, A, 1, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(ctx, b_matrix, 255, B, 1, 1) == RNS8_SUCCESS);
  CHECK(rns8_gemm_finite_u8(ctx, plan, 251, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_gemm_finite_u8(ctx, plan, 255, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  CHECK(rns8_export_finite_u8(ctx, plan, 251, c_matrix, C, 1) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_export_finite_u8(ctx, plan, 255, c_matrix, C, 1) == RNS8_SUCCESS);
  CHECK(C[0] == 15);

  auto field_desc = finite_desc(RNS8_FINITE_FIELD_U8, 1, 1, 1);
  rns8_plan* field_plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &field_desc, &field_plan) == RNS8_SUCCESS);
  auto field_matrix_desc = matrix_desc(1, 1, RNS8_FINITE_FIELD_U8, RNS8_BOUND_NONE);
  rns8_matrix* field_matrix = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &field_matrix_desc, &field_matrix) == RNS8_SUCCESS);
  CHECK(rns8_pack_finite_u8(ctx, field_matrix, 255, A, 1, 1) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(field_matrix);
  rns8_destroy_plan(field_plan);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("unknown public enum values are invalid before backend routing") {
  rns8_context* ctx = create_cpu();
  const auto unknown_semantics = invalid_public_enum_value<rns8_semantics>();
  const auto unknown_bound_kind = invalid_public_enum_value<rns8_bound_kind>();
  const auto unknown_layout = invalid_public_enum_value<rns8_layout>();
  const auto unknown_backend = invalid_public_enum_value<rns8_backend_kind>();

  {
    auto desc = gemm_desc(RNS8_BOUNDED_U64, unknown_bound_kind);
    desc.requested_backend = RNS8_BACKEND_HIPBLASLT;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    auto desc = gemm_desc(unknown_semantics, RNS8_BOUND_NONE);
    desc.bound = 0;
    desc.requested_backend = RNS8_BACKEND_CK;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    desc.requested_backend = unknown_backend;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(plan == nullptr);
  }
  {
    auto matrix = matrix_desc(RNS8_BOUNDED_U64, unknown_bound_kind);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  {
    auto matrix = matrix_desc(unknown_semantics, RNS8_BOUND_NONE);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  {
    auto matrix = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    matrix.logical_layout = unknown_layout;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  {
    auto desc = gemm_desc(RNS8_FINITE_RING_U8, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    desc.requested_backend = RNS8_BACKEND_ROCWMMA;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  {
    auto matrix = matrix_desc(RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    matrix.logical_layout = RNS8_LAYOUT_COLUMN_MAJOR;
    matrix.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = unknown_backend;
    rns8_context* unknown_ctx = nullptr;
    CHECK(rns8_create_context(0, &options, &unknown_ctx) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(unknown_ctx == nullptr);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("known but unimplemented descriptor contracts report unsupported status") {
  rns8_context* ctx = create_cpu();
  {
    auto column_major = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    column_major.logical_layout = RNS8_LAYOUT_COLUMN_MAJOR;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &column_major, &storage) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(storage == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide RNS output accepts only explicit unbounded semantics") {
  rns8_context* ctx = create_cpu();
  for (const rns8_semantics semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    auto desc = gemm_desc(semantics, RNS8_BOUND_NONE);
    desc.bound = 0;
    desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    rns8_destroy_plan(plan);

    auto matrix = matrix_desc(semantics, RNS8_BOUND_NONE);
    matrix.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_SUCCESS);
    rns8_destroy_matrix(storage);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("unbounded semantic descriptors reject stale bound metadata") {
  rns8_context* ctx = create_cpu();
  for (const rns8_semantics semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    auto desc = gemm_desc(semantics, RNS8_BOUND_NONE);
    desc.bound = 1;
    desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("global descriptors reject stray per-tile bound storage") {
  rns8_context* ctx = create_cpu();
  const uint64_t stale_bound = 100;
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.tile_bounds = &stale_bound;
  desc.tile_bounds_count = 1;
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto exact = gemm_desc(RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  exact.bound = 0;
  exact.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  exact.tile_bounds_count = 1;
  CHECK(rns8_create_plan(ctx, &exact, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide semantics reject bounded-looking CRT metadata") {
  rns8_context* ctx = create_cpu();
  for (const rns8_semantics semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    for (const rns8_bound_kind bound_kind :
         {RNS8_BOUND_GLOBAL_MAX_ABS,
          RNS8_BOUND_GLOBAL_MAX_UNSIGNED,
          RNS8_BOUND_PER_TILE_MAX_ABS,
          RNS8_BOUND_PER_TILE_MAX_UNSIGNED,
          RNS8_BOUND_INPUT_RANGE_AND_K}) {
      auto desc = bounded_looking_desc(semantics, bound_kind);
      rns8_plan* plan = nullptr;
      CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
      CHECK(plan == nullptr);

      auto matrix = bounded_looking_matrix_desc(semantics, bound_kind);
      rns8_matrix* storage = nullptr;
      CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
      CHECK(storage == nullptr);
    }
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("strict wraparound is not accepted as bounded odd-modulus CRT") {
  rns8_context* ctx = create_cpu();
  for (const rns8_bound_kind bound_kind :
       {RNS8_BOUND_NONE,
        RNS8_BOUND_GLOBAL_MAX_UNSIGNED,
        RNS8_BOUND_PER_TILE_MAX_ABS,
        RNS8_BOUND_PER_TILE_MAX_UNSIGNED,
        RNS8_BOUND_INPUT_RANGE_AND_K}) {
    auto desc = bounded_looking_desc(RNS8_WRAP_U64_MOD_2_64, bound_kind);
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto matrix = bounded_looking_matrix_desc(RNS8_WRAP_U64_MOD_2_64, bound_kind);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("wrap64 byte-limb context reports explicit CPU reference backend") {
  rns8_context* ctx = create_wrap64();
  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(ctx, &info) == RNS8_SUCCESS);
  CHECK(info.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB);
  CHECK(info.device_id == -1);
  CHECK(info.hip_available == 0);
  rns8_destroy_context(ctx);
}

TEST_CASE("auto backend selection never routes across explicit semantic backends") {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_AUTO;
  rns8_context* auto_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &auto_ctx) == RNS8_SUCCESS);

  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(auto_ctx, &info) == RNS8_SUCCESS);
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  CHECK(info.backend == RNS8_BACKEND_HIP_DIRECT);
#else
  CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);
#endif

  auto wrap = gemm_desc(RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  wrap.bound = 0;
  wrap.requested_backend = RNS8_BACKEND_AUTO;
  rns8_plan* plan = nullptr;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  CHECK(rns8_create_plan(auto_ctx, &wrap, &plan) == RNS8_SUCCESS);
  rns8_destroy_plan(plan);
  plan = nullptr;
#else
  CHECK(rns8_create_plan(auto_ctx, &wrap, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);
#endif

  auto exact = gemm_desc(RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  exact.bound = 0;
  exact.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  exact.requested_backend = RNS8_BACKEND_AUTO;
  REQUIRE(rns8_create_plan(auto_ctx, &exact, &plan) == RNS8_SUCCESS);
  rns8_destroy_plan(plan);
  plan = nullptr;
  rns8_destroy_context(auto_ctx);

  rns8_context* wrap_ctx = create_wrap64();
  CHECK(rns8_create_plan(wrap_ctx, &wrap, &plan) == RNS8_SUCCESS);
  rns8_destroy_plan(plan);

  auto bounded = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  bounded.requested_backend = RNS8_BACKEND_AUTO;
  CHECK(rns8_create_plan(wrap_ctx, &bounded, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);
  CHECK(rns8_create_plan(wrap_ctx, &exact, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);

  auto exact_matrix = matrix_desc(RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  exact_matrix.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  rns8_matrix* storage = nullptr;
  CHECK(rns8_create_matrix(wrap_ctx, &exact_matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(storage == nullptr);

  auto bounded_matrix = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  CHECK(rns8_create_matrix(wrap_ctx, &bounded_matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(storage == nullptr);
  rns8_destroy_context(wrap_ctx);
}

TEST_CASE("future backend context kinds report unsupported status") {
  std::vector<rns8_backend_kind> backends;
#if !defined(RNS8_ENABLE_ROCWMMA) || !RNS8_ENABLE_ROCWMMA
  backends.push_back(RNS8_BACKEND_ROCWMMA);
#endif
#if !defined(RNS8_ENABLE_CK) || !RNS8_ENABLE_CK
  backends.push_back(RNS8_BACKEND_CK);
#endif
#if !defined(RNS8_ENABLE_HIPBLASLT) || !RNS8_ENABLE_HIPBLASLT
  backends.push_back(RNS8_BACKEND_HIPBLASLT);
#endif
  for (const rns8_backend_kind backend : backends) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = backend;
    rns8_context* ctx = nullptr;
    CHECK(rns8_create_context(0, &options, &ctx) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(ctx == nullptr);

    auto invalid_size = options;
    invalid_size.struct_size = 0;
    CHECK(rns8_create_context(0, &invalid_size, &ctx) == RNS8_INVALID_ARGUMENT);
    CHECK(ctx == nullptr);

    auto invalid_version = options;
    invalid_version.abi_version = 0;
    CHECK(rns8_create_context(0, &invalid_version, &ctx) == RNS8_INVALID_ARGUMENT);
    CHECK(ctx == nullptr);
  }
}

TEST_CASE("tile dimensions are powers of two from 64 to 512") {
  rns8_context* ctx = create_cpu();
  auto valid = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  valid.tile_m = 256;
  valid.tile_n = 512;
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(ctx, &valid, &plan) == RNS8_SUCCESS);
  rns8_destroy_plan(plan);

  for (const uint32_t bad_tile : {32u, 96u, 513u}) {
    auto bad_plan = valid;
    bad_plan.tile_m = bad_tile;
    plan = nullptr;
    CHECK(rns8_create_plan(ctx, &bad_plan, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto bad_matrix = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    bad_matrix.tile_n = bad_tile;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &bad_matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("persistent RNS GEMM rejects incompatible matrix and workspace contracts") {
  rns8_context* ctx = create_cpu();
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.m = 2;
  desc.n = 2;
  desc.k = 2;
  desc.bound = 100000;

  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto b_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  auto smaller_desc = desc;
  smaller_desc.k = 1;
  rns8_plan* smaller_plan = nullptr;
  rns8_workspace* wrong_workspace = nullptr;
  REQUIRE(rns8_create_plan(ctx, &smaller_desc, &smaller_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, smaller_plan, &wrong_workspace) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, wrong_workspace) == RNS8_WORKSPACE_TOO_SMALL);

  rns8_matrix* wrong_b_shape = nullptr;
  auto wrong_b_shape_desc = matrix_desc(3, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &wrong_b_shape_desc, &wrong_b_shape) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, wrong_b_shape, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);

  rns8_matrix* wrong_c_prefix = nullptr;
  auto wrong_c_prefix_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, 1);
  REQUIRE(rns8_create_matrix(ctx, &wrong_c_prefix_desc, &wrong_c_prefix) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, wrong_c_prefix, workspace) == RNS8_INVALID_ARGUMENT);

  rns8_matrix* wrong_a_bound_kind = nullptr;
  auto wrong_a_bound_kind_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &wrong_a_bound_kind_desc, &wrong_a_bound_kind) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan, wrong_a_bound_kind, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(wrong_a_bound_kind);
  rns8_destroy_matrix(wrong_c_prefix);
  rns8_destroy_matrix(wrong_b_shape);
  rns8_destroy_workspace(wrong_workspace);
  rns8_destroy_plan(smaller_plan);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent bounded RNS GEMM stamps output source versions from packed inputs") {
  rns8_context* ctx = create_cpu();
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.m = 2;
  desc.n = 2;
  desc.k = 2;
  desc.bound = 1000;

  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_workspace* wrong_workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto smaller_desc = desc;
  smaller_desc.k = 1;
  rns8_plan* smaller_plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &smaller_desc, &smaller_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, smaller_plan, &wrong_workspace) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto b_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

  const uint64_t A0[] = {1, 2, 3, 4};
  const uint64_t B0[] = {5, 6, 7, 8};
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A0, 2, 17) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B0, 2, 23) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  const uint64_t first_output_version = c_matrix->source_version;
  CHECK(first_output_version != 0);
  CHECK(first_output_version != a_matrix->source_version);
  CHECK(first_output_version != b_matrix->source_version);

  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  CHECK(c_matrix->source_version == first_output_version);

  const uint64_t B1[] = {8, 7, 6, 5};
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B1, 2, 24) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  const uint64_t second_output_version = c_matrix->source_version;
  CHECK(second_output_version != 0);
  CHECK(second_output_version != first_output_version);

  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, wrong_workspace) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(c_matrix->source_version == second_output_version);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(wrong_workspace);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(smaller_plan);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent RNS GEMM rejects same-shape workspaces from different semantic contracts") {
  rns8_context* ctx = create_cpu();

  auto bounded = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  bounded.m = 2;
  bounded.n = 2;
  bounded.k = 1;
  bounded.bound = 100;
  bounded.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;

  auto exact = gemm_desc(RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  exact.m = bounded.m;
  exact.n = bounded.n;
  exact.k = bounded.k;
  exact.bound = 0;
  exact.max_prefix = bounded.max_prefix;

  const uint64_t tile_bound = 100;
  auto per_tile = bounded;
  per_tile.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  per_tile.bound = 0;
  per_tile.tile_m = 64;
  per_tile.tile_n = 64;
  per_tile.tile_bounds = &tile_bound;
  per_tile.tile_bounds_count = 1;

  rns8_plan* bounded_plan = nullptr;
  rns8_plan* exact_plan = nullptr;
  rns8_plan* per_tile_plan = nullptr;
  rns8_workspace* bounded_workspace = nullptr;
  rns8_workspace* exact_workspace = nullptr;
  rns8_workspace* per_tile_workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;

  REQUIRE(rns8_create_plan(ctx, &bounded, &bounded_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(ctx, &exact, &exact_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(ctx, &per_tile, &per_tile_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, bounded_plan, &bounded_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, exact_plan, &exact_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, per_tile_plan, &per_tile_workspace) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(2, 1, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_MAX_SUPPORTED_PREFIX);
  auto b_desc = matrix_desc(1, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_MAX_SUPPORTED_PREFIX);
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_MAX_SUPPORTED_PREFIX);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

  CHECK(rns8_gemm_rns(ctx, bounded_plan, a_matrix, b_matrix, c_matrix, bounded_workspace) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, bounded_plan, a_matrix, b_matrix, c_matrix, exact_workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(ctx, bounded_plan, a_matrix, b_matrix, c_matrix, per_tile_workspace) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(per_tile_workspace);
  rns8_destroy_workspace(exact_workspace);
  rns8_destroy_workspace(bounded_workspace);
  rns8_destroy_plan(per_tile_plan);
  rns8_destroy_plan(exact_plan);
  rns8_destroy_plan(bounded_plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent bounded CPU matrices preserve source versions and currentness across reuse") {
  rns8_context* ctx = create_cpu();
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.m = 2;
  desc.n = 2;
  desc.k = 2;
  desc.bound = 1000;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;

  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
  auto b_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

  const uint64_t A0[] = {1, 2, 3, 4};
  const uint64_t B0[] = {5, 6, 7, 8};
  uint64_t C[4] = {};
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A0, 2, 101) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B0, 2, 202) == RNS8_SUCCESS);
  CHECK(a_matrix->source_version == 101);
  CHECK(b_matrix->source_version == 202);
  CHECK(a_matrix->host_residues_current);
  CHECK_FALSE(a_matrix->device_residues_current);
  CHECK(b_matrix->host_residues_current);
  CHECK_FALSE(b_matrix->device_residues_current);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(ctx, plan, c_matrix, C, 2) == RNS8_SUCCESS);
  CHECK(C[0] == 19);
  CHECK(C[1] == 22);
  CHECK(C[2] == 43);
  CHECK(C[3] == 50);
  CHECK(a_matrix->source_version == 101);
  CHECK(b_matrix->source_version == 202);
  CHECK(c_matrix->host_residues_current);
  CHECK_FALSE(c_matrix->device_residues_current);

  const uint64_t A1[] = {9, 1, 2, 3};
  const uint64_t B1[] = {4, 5, 6, 7};
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A1, 2, 303) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B1, 2, 404) == RNS8_SUCCESS);
  CHECK(a_matrix->source_version == 303);
  CHECK(b_matrix->source_version == 404);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(ctx, plan, c_matrix, C, 2) == RNS8_SUCCESS);
  CHECK(C[0] == 42);
  CHECK(C[1] == 52);
  CHECK(C[2] == 26);
  CHECK(C[3] == 31);
  c_matrix->host_residues_current = false;
  CHECK(rns8_export_u64(ctx, plan, c_matrix, C, 2) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent bounded RNS GEMM rejects same-shape stale schedule workspaces and matrix tiles") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  const uint64_t bounds_a[4] = {0, 0, 7000000, 1000000000};
  const uint64_t bounds_b[4] = {0, 0, 7000000, 999999999};
  const uint64_t bounds_tile[1] = {1000000000};
  std::vector<uint64_t> A(static_cast<std::size_t>(m), 0);
  std::vector<uint64_t> B(static_cast<std::size_t>(n), 7);
  A.back() = 1000000;
  B.back() = 1000;

  auto desc_a = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  desc_a.m = m;
  desc_a.n = n;
  desc_a.k = k;
  desc_a.bound = 0;
  desc_a.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  desc_a.tile_m = 64;
  desc_a.tile_n = 64;
  desc_a.tile_bounds = bounds_a;
  desc_a.tile_bounds_count = 4;

  auto desc_b = desc_a;
  desc_b.tile_bounds = bounds_b;

  auto desc_tile = desc_a;
  desc_tile.tile_m = 128;
  desc_tile.tile_n = 128;
  desc_tile.tile_bounds = bounds_tile;
  desc_tile.tile_bounds_count = 1;

  rns8_plan* plan_a = nullptr;
  rns8_plan* plan_b = nullptr;
  rns8_plan* plan_tile = nullptr;
  rns8_workspace* workspace_a = nullptr;
  rns8_workspace* workspace_b = nullptr;
  rns8_workspace* workspace_tile = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  rns8_matrix* wrong_a_tile = nullptr;
  rns8_matrix* wrong_c_tile = nullptr;

  REQUIRE(rns8_create_plan(ctx, &desc_a, &plan_a) == RNS8_SUCCESS);
  REQUIRE_FALSE(plan_a->tile_schedule.empty());
  CHECK(plan_a->tile_schedule[0].range_bit_length == 0);
  REQUIRE(rns8_create_plan(ctx, &desc_b, &plan_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(ctx, &desc_tile, &plan_tile) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan_a, &workspace_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan_b, &workspace_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan_tile, &workspace_tile) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B.data(), n, 1) == RNS8_SUCCESS);

  CHECK(rns8_gemm_rns(ctx, plan_a, a_matrix, b_matrix, c_matrix, workspace_a) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan_a, a_matrix, b_matrix, c_matrix, workspace_b) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(ctx, plan_a, a_matrix, b_matrix, c_matrix, workspace_tile) == RNS8_INVALID_ARGUMENT);

  auto wrong_a_desc = a_desc;
  wrong_a_desc.tile_m = 128;
  wrong_a_desc.tile_n = 128;
  auto wrong_c_desc = c_desc;
  wrong_c_desc.tile_m = 128;
  wrong_c_desc.tile_n = 128;
  REQUIRE(rns8_create_matrix(ctx, &wrong_a_desc, &wrong_a_tile) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &wrong_c_desc, &wrong_c_tile) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(ctx, plan_a, wrong_a_tile, b_matrix, c_matrix, workspace_a) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(ctx, plan_a, a_matrix, b_matrix, wrong_c_tile, workspace_a) == RNS8_INVALID_ARGUMENT);
  uint64_t out[4] = {};
  CHECK(rns8_export_u64(ctx, plan_a, wrong_c_tile, out, n) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(wrong_c_tile);
  rns8_destroy_matrix(wrong_a_tile);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace_tile);
  rns8_destroy_workspace(workspace_b);
  rns8_destroy_workspace(workspace_a);
  rns8_destroy_plan(plan_tile);
  rns8_destroy_plan(plan_b);
  rns8_destroy_plan(plan_a);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent RNS APIs reject stale plan schedule metadata before dispatch") {
  rns8_context* ctx = create_cpu();
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.m = 1;
  desc.n = 1;
  desc.k = 1;
  desc.bound = 100000;

  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);

  plan->schedule_prefix_group_count = 0;
  CHECK(rns8_get_plan_schedule_info(plan, &info) == RNS8_INVALID_ARGUMENT);
  uint64_t written = 0;
  CHECK(rns8_get_plan_tile_schedule(plan, nullptr, 0, &written) == RNS8_INVALID_ARGUMENT);
  rns8_workspace* workspace = nullptr;
  CHECK(rns8_create_workspace(ctx, plan, &workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(workspace == nullptr);
  plan->schedule_prefix_group_count = 1;

  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  auto a_desc = matrix_desc(1, 1, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto b_desc = matrix_desc(1, 1, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto c_desc = matrix_desc(1, 1, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  const uint64_t a = 3;
  const uint64_t b = 5;
  uint64_t c = 0;
  REQUIRE(rns8_pack_u64(ctx, a_matrix, &a, 1, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, &b, 1, 1) == RNS8_SUCCESS);

  const uint32_t saved_min_selected = plan->schedule_min_selected_prefix;
  plan->schedule_min_selected_prefix = 0;
  CHECK(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_u64(ctx, plan, c_matrix, &c, 1) == RNS8_INVALID_ARGUMENT);
  plan->schedule_min_selected_prefix = saved_min_selected;

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("persistent bounded export rejects output matrices outside the plan contract") {
  rns8_context* ctx = create_cpu();
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.m = 2;
  desc.n = 2;
  desc.k = 1;
  desc.bound = 100000;
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  uint64_t dst[4] = {};
  rns8_matrix* c_matrix = nullptr;
  auto c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  CHECK(rns8_export_u64(ctx, plan, c_matrix, dst, 2) == RNS8_SUCCESS);

  rns8_matrix* wrong_c_prefix = nullptr;
  auto wrong_c_prefix_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, 1);
  REQUIRE(rns8_create_matrix(ctx, &wrong_c_prefix_desc, &wrong_c_prefix) == RNS8_SUCCESS);
  CHECK(rns8_export_u64(ctx, plan, wrong_c_prefix, dst, 2) == RNS8_INVALID_ARGUMENT);

  rns8_matrix* wrong_c_bound_kind = nullptr;
  auto wrong_c_bound_kind_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &wrong_c_bound_kind_desc, &wrong_c_bound_kind) == RNS8_SUCCESS);
  CHECK(rns8_export_u64(ctx, plan, wrong_c_bound_kind, dst, 2) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(wrong_c_bound_kind);
  rns8_destroy_matrix(wrong_c_prefix);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("matrix creation rejects descriptor sizes that overflow owned storage") {
  {
    rns8_context* ctx = create_cpu();
    auto desc = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    desc.rows = std::numeric_limits<int64_t>::max() / 2;
    desc.cols = 8;
    desc.logical_ld = desc.cols;
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &desc, &storage) == RNS8_RANGE_ERROR);
    CHECK(storage == nullptr);
    rns8_destroy_context(ctx);
  }

  {
    rns8_context* ctx = create_wrap64();
    auto desc = matrix_desc(RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
    desc.rows = std::numeric_limits<int64_t>::max() / 2;
    desc.cols = 8;
    desc.logical_ld = desc.cols;
    desc.max_prefix = 0;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &desc, &storage) == RNS8_RANGE_ERROR);
    CHECK(storage == nullptr);
    rns8_destroy_context(ctx);
  }
}
