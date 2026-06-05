#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <vector>

#include "core/internal.hpp"
#include "../support/currentness_test_helpers.hpp"
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

rns8_gemm_desc exact_desc(rns8_semantics semantics, int64_t m, int64_t n, int64_t k) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_matrix_desc exact_matrix_desc(int64_t rows, int64_t cols, rns8_semantics semantics) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

void check_signed_residues(
    const rns8_matrix& matrix,
    const int64_t* A,
    int64_t lda,
    const int64_t* B,
    int64_t ldb,
    int64_t m,
    int64_t n,
    int64_t k) {
  for (uint32_t p = 0; p < matrix.prefix; ++p) {
    const uint16_t modulus = rns8::detail::kDefaultModuli[p];
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const auto exact = rns8::detail::exact_i64_gemm_cell(A, lda, B, ldb, row, col, k);
        CHECK(matrix.residues[rns8::detail::residue_index(matrix, p, row, col)] ==
              rns8::detail::centered_residue(exact, modulus));
      }
    }
  }
}

void check_unsigned_residues(
    const rns8_matrix& matrix,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    int64_t m,
    int64_t n,
    int64_t k) {
  for (uint32_t p = 0; p < matrix.prefix; ++p) {
    const uint16_t modulus = rns8::detail::kDefaultModuli[p];
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const auto exact = rns8::detail::exact_u64_gemm_cell(A, lda, B, ldb, row, col, k);
        CHECK(matrix.residues[rns8::detail::residue_index(matrix, p, row, col)] ==
              rns8::detail::centered_residue(exact, modulus));
      }
    }
  }
}

std::vector<uint64_t> unsigned_limbs(boost::multiprecision::cpp_int value, uint32_t limb_count) {
  std::vector<uint64_t> limbs(limb_count);
  for (uint32_t limb = 0; limb < limb_count; ++limb) {
    limbs[limb] = static_cast<uint64_t>(value & boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()));
    value >>= 64u;
  }
  return limbs;
}

std::vector<uint64_t> signed_twos_complement_limbs(boost::multiprecision::cpp_int value, uint32_t limb_count) {
  if (value < 0) {
    value += boost::multiprecision::cpp_int(1) << (64u * limb_count);
  }
  return unsigned_limbs(value, limb_count);
}

std::vector<int8_t> residues_for(boost::multiprecision::cpp_int value, uint32_t prefix) {
  std::vector<int8_t> residues(prefix);
  for (uint32_t p = 0; p < prefix; ++p) {
    residues[p] = rns8::detail::centered_residue(value, rns8::detail::kDefaultModuli[p]);
  }
  return residues;
}

void fill_exact_residue_matrix(rns8_matrix* matrix, const std::vector<boost::multiprecision::cpp_int>& values) {
  REQUIRE(matrix != nullptr);
  REQUIRE(values.size() == static_cast<std::size_t>(matrix->desc.rows * matrix->desc.cols));
  std::vector<std::vector<int8_t>> residues;
  residues.reserve(values.size());
  for (const auto& value : values) {
    residues.push_back(residues_for(value, matrix->prefix));
  }
  for (uint32_t p = 0; p < matrix->prefix; ++p) {
    for (int64_t row = 0; row < matrix->desc.rows; ++row) {
      for (int64_t col = 0; col < matrix->desc.cols; ++col) {
        const std::size_t value_index = static_cast<std::size_t>(row * matrix->desc.cols + col);
        const std::size_t residue_index = rns8::detail::residue_index(*matrix, p, row, col);
        matrix->residues[residue_index] = residues[value_index][p];
      }
    }
  }
  rns8::test::set_host_residues_current(*matrix, true);
  rns8::test::set_device_residues_current(*matrix, false);
  rns8::test::set_host_byte_limbs_current(*matrix, false);
  rns8::test::set_device_byte_limbs_current(*matrix, false);
}

}  // namespace

TEST_CASE("exact-wide signed CRT export uses centered half-open representative") {
  constexpr uint32_t prefix = 3;
  const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(prefix);
  const boost::multiprecision::cpp_int half = product / 2;
  const auto residues = residues_for(half, prefix);

  uint64_t limb = 0;
  REQUIRE(rns8::detail::export_exact_wide_signed_limbs(residues, prefix, &limb, 1) == RNS8_SUCCESS);
  const auto expected = signed_twos_complement_limbs(-half, 1);
  CHECK(limb == expected[0]);

  int64_t bounded = 0;
  const boost::multiprecision::cpp_int below_half = half - 1;
  const auto bounded_residues = residues_for(below_half, prefix);
  CHECK(rns8::detail::reconstruct_signed(
            bounded_residues, prefix, static_cast<uint64_t>(below_half), bounded) == RNS8_SUCCESS);
  CHECK(boost::multiprecision::cpp_int(bounded) == below_half);
}

TEST_CASE("exact-wide signed max-prefix half-product export is fixed-width two's-complement") {
  constexpr uint32_t prefix = RNS8_MAX_SUPPORTED_PREFIX;
  const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(prefix);
  const boost::multiprecision::cpp_int half = product / 2;
  const auto residues = residues_for(half, prefix);

  constexpr uint64_t sentinel = 0xa3a3a3a3a3a3a3a3ull;
  uint64_t too_few[2] = {sentinel, sentinel};
  CHECK(rns8::detail::export_exact_wide_signed_limbs(residues, prefix, too_few, 2) == RNS8_RANGE_ERROR);
  CHECK(too_few[0] == sentinel);
  CHECK(too_few[1] == sentinel);

  uint64_t limbs[3] = {};
  REQUIRE(rns8::detail::export_exact_wide_signed_limbs(residues, prefix, limbs, 3) == RNS8_SUCCESS);
  const auto expected = signed_twos_complement_limbs(-half, 3);
  CHECK(std::vector<uint64_t>(limbs, limbs + 3) == expected);
}

TEST_CASE("exact-wide signed fixed-width limb export covers one-limb boundaries") {
  constexpr uint32_t prefix = RNS8_MAX_SUPPORTED_PREFIX;
  uint64_t limbs[32] = {};

  auto export_signed = [&](boost::multiprecision::cpp_int value, uint32_t limb_count) {
    const auto residues = residues_for(value, prefix);
    return rns8::detail::export_exact_wide_signed_limbs(residues, prefix, limbs, limb_count);
  };

  REQUIRE(export_signed(boost::multiprecision::cpp_int(std::numeric_limits<int64_t>::max()), 1) == RNS8_SUCCESS);
  CHECK(limbs[0] == 0x7fffffffffffffffull);

  REQUIRE(export_signed(boost::multiprecision::cpp_int(std::numeric_limits<int64_t>::min()), 1) == RNS8_SUCCESS);
  CHECK(limbs[0] == 0x8000000000000000ull);

  REQUIRE(export_signed(boost::multiprecision::cpp_int(-1), 1) == RNS8_SUCCESS);
  CHECK(limbs[0] == std::numeric_limits<uint64_t>::max());

  constexpr uint64_t sentinel = 0x8787878787878787ull;
  for (uint64_t& limb : limbs) {
    limb = sentinel;
  }
  CHECK(export_signed(boost::multiprecision::cpp_int(1) << 63u, 1) == RNS8_RANGE_ERROR);
  CHECK(limbs[0] == sentinel);

  for (uint64_t& limb : limbs) {
    limb = sentinel;
  }
  CHECK(export_signed(-((boost::multiprecision::cpp_int(1) << 63u) + 1), 1) == RNS8_RANGE_ERROR);
  CHECK(limbs[0] == sentinel);

  REQUIRE(export_signed(boost::multiprecision::cpp_int(-1), 32) == RNS8_SUCCESS);
  for (uint64_t limb : limbs) {
    CHECK(limb == std::numeric_limits<uint64_t>::max());
  }
}

TEST_CASE("exact-wide unsigned fixed-width limb export covers overflow and max width") {
  constexpr uint32_t prefix = RNS8_MAX_SUPPORTED_PREFIX;
  uint64_t limbs[32] = {};

  auto export_unsigned = [&](boost::multiprecision::cpp_int value, uint32_t limb_count) {
    const auto residues = residues_for(value, prefix);
    return rns8::detail::export_exact_wide_unsigned_limbs(residues, prefix, limbs, limb_count);
  };

  REQUIRE(export_unsigned(boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()), 1) == RNS8_SUCCESS);
  CHECK(limbs[0] == std::numeric_limits<uint64_t>::max());

  constexpr uint64_t sentinel = 0x9696969696969696ull;
  for (uint64_t& limb : limbs) {
    limb = sentinel;
  }
  CHECK(export_unsigned(boost::multiprecision::cpp_int(1) << 64u, 1) == RNS8_RANGE_ERROR);
  CHECK(limbs[0] == sentinel);

  REQUIRE(export_unsigned(boost::multiprecision::cpp_int(1) << 64u, 2) == RNS8_SUCCESS);
  CHECK(limbs[0] == 0);
  CHECK(limbs[1] == 1);

  REQUIRE(export_unsigned(boost::multiprecision::cpp_int(1), 32) == RNS8_SUCCESS);
  CHECK(limbs[0] == 1);
  for (uint32_t limb = 1; limb < 32; ++limb) {
    CHECK(limbs[limb] == 0);
  }
}

TEST_CASE("exact-wide unsigned high-bit magnitude export preserves fixed width") {
  constexpr uint32_t prefix = RNS8_MAX_SUPPORTED_PREFIX;
  const boost::multiprecision::cpp_int value = boost::multiprecision::cpp_int(1) << 127u;
  const auto residues = residues_for(value, prefix);

  constexpr uint64_t sentinel = 0xb4b4b4b4b4b4b4b4ull;
  uint64_t too_few[1] = {sentinel};
  CHECK(rns8::detail::export_exact_wide_unsigned_limbs(residues, prefix, too_few, 1) == RNS8_RANGE_ERROR);
  CHECK(too_few[0] == sentinel);

  uint64_t two_limbs[2] = {};
  REQUIRE(rns8::detail::export_exact_wide_unsigned_limbs(residues, prefix, two_limbs, 2) == RNS8_SUCCESS);
  CHECK(two_limbs[0] == 0);
  CHECK(two_limbs[1] == (uint64_t{1} << 63u));

  uint64_t wide[32] = {};
  REQUIRE(rns8::detail::export_exact_wide_unsigned_limbs(residues, prefix, wide, 32) == RNS8_SUCCESS);
  CHECK(wide[0] == 0);
  CHECK(wide[1] == (uint64_t{1} << 63u));
  for (uint32_t limb = 2; limb < 32; ++limb) {
    CHECK(wide[limb] == 0);
  }
}

TEST_CASE("exact-wide signed CPU RNS output matches multiprecision residues") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 2;
  const int64_t A[] = {
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max() - 17,
      -std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max() - 31};
  const int64_t B[] = {
      std::numeric_limits<int64_t>::max() - 5,
      -std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max() - 11,
      std::numeric_limits<int64_t>::max() - 13};

  auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
  c_desc.max_prefix = plan->prefix;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  check_signed_residues(*c_matrix, A, k, B, n, m, n, k);
  constexpr uint32_t limb_count = 3;
  std::vector<uint64_t> limbs(static_cast<std::size_t>(m * n * limb_count), 0);
  REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), n, limb_count) == RNS8_SUCCESS);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const auto exact = rns8::detail::exact_i64_gemm_cell(A, k, B, n, row, col, k);
      const auto expected = signed_twos_complement_limbs(exact, limb_count);
      const std::size_t offset = static_cast<std::size_t>((row * n + col) * limb_count);
      CHECK(std::vector<uint64_t>(limbs.begin() + offset, limbs.begin() + offset + limb_count) == expected);
    }
  }
  constexpr uint64_t range_sentinel = 0x123456789abcdef0ull;
  std::vector<uint64_t> too_few(static_cast<std::size_t>(m * n), range_sentinel);
  CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, too_few.data(), n, 1) == RNS8_RANGE_ERROR);
  for (const uint64_t limb : too_few) {
    CHECK(limb == range_sentinel);
  }
  CHECK(rns8_export_i64(ctx, plan, c_matrix, reinterpret_cast<int64_t*>(limbs.data()), n) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide signed CPU export rejects bounded unsigned and wrap interpretations") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 1;
  const int64_t n = 1;
  const int64_t k = 1;
  const int64_t A[] = {-7};
  const int64_t B[] = {11};

  auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  uint64_t limbs[2] = {0xddddddddddddddddull, 0xeeeeeeeeeeeeeeeeull};
  CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs, n, 2) == RNS8_INVALID_ARGUMENT);
  CHECK(limbs[0] == 0xddddddddddddddddull);
  CHECK(limbs[1] == 0xeeeeeeeeeeeeeeeeull);
  CHECK(rns8_export_u64(ctx, plan, c_matrix, limbs, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_wrap_u64(ctx, plan, c_matrix, limbs, n) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide signed CPU limb export preserves padded destination cells") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 1;
  const int64_t ld = 3;
  const uint64_t sentinel = 0xdecafbaddecafbadull;
  const int64_t A[] = {5, -7};
  const int64_t B[] = {11, -13};

  auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  constexpr uint32_t limb_count = 2;
  std::vector<uint64_t> limbs(static_cast<std::size_t>(m * ld * limb_count), sentinel);
  REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), ld, limb_count) == RNS8_SUCCESS);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const auto exact = rns8::detail::exact_i64_gemm_cell(A, k, B, n, row, col, k);
      const auto expected = signed_twos_complement_limbs(exact, limb_count);
      const std::size_t offset = static_cast<std::size_t>((row * ld + col) * limb_count);
      CHECK(std::vector<uint64_t>(limbs.begin() + offset, limbs.begin() + offset + limb_count) == expected);
    }
    const std::size_t padding = static_cast<std::size_t>((row * ld + n) * limb_count);
    CHECK(limbs[padding] == sentinel);
    CHECK(limbs[padding + 1] == sentinel);
  }

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide unsigned CPU RNS output matches multiprecision residues") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 1;
  const int64_t n = 2;
  const int64_t k = 2;
  const uint64_t A[] = {std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max() - 3};
  const uint64_t B[] = {
      std::numeric_limits<uint64_t>::max() - 5,
      std::numeric_limits<uint64_t>::max() - 7,
      std::numeric_limits<uint64_t>::max() - 11,
      std::numeric_limits<uint64_t>::max() - 13};

  auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
  c_desc.max_prefix = plan->prefix;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  check_unsigned_residues(*c_matrix, A, k, B, n, m, n, k);
  constexpr uint32_t limb_count = 3;
  std::vector<uint64_t> limbs(static_cast<std::size_t>(m * n * limb_count), 0);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), n, limb_count) == RNS8_SUCCESS);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const auto exact = rns8::detail::exact_u64_gemm_cell(A, k, B, n, row, col, k);
      const auto expected = unsigned_limbs(exact, limb_count);
      const std::size_t offset = static_cast<std::size_t>((row * n + col) * limb_count);
      CHECK(std::vector<uint64_t>(limbs.begin() + offset, limbs.begin() + offset + limb_count) == expected);
    }
  }
  constexpr uint64_t range_sentinel = 0x0fedcba987654321ull;
  std::vector<uint64_t> too_few(static_cast<std::size_t>(m * n), range_sentinel);
  CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, too_few.data(), n, 1) == RNS8_RANGE_ERROR);
  for (const uint64_t limb : too_few) {
    CHECK(limb == range_sentinel);
  }
  CHECK(rns8_export_u64(ctx, plan, c_matrix, limbs.data(), n) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide unsigned CPU export rejects signed bounded and wrap interpretations") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 1;
  const int64_t n = 1;
  const int64_t k = 1;
  const uint64_t A[] = {17};
  const uint64_t B[] = {19};

  auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  uint64_t limbs[2] = {0xccccccccccccccccull, 0xbbbbbbbbbbbbbbbbull};
  CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs, n, 2) == RNS8_INVALID_ARGUMENT);
  CHECK(limbs[0] == 0xccccccccccccccccull);
  CHECK(limbs[1] == 0xbbbbbbbbbbbbbbbbull);
  CHECK(rns8_export_i64(ctx, plan, c_matrix, reinterpret_cast<int64_t*>(limbs), n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_wrap_u64(ctx, plan, c_matrix, limbs, n) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide unsigned CPU limb export preserves padded destination cells") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 1;
  const int64_t ld = 4;
  const uint64_t sentinel = 0xf00df00df00df00dull;
  const uint64_t A[] = {17, 19};
  const uint64_t B[] = {23, 29};

  auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED);
  auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED);
  auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b_matrix, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

  constexpr uint32_t limb_count = 2;
  std::vector<uint64_t> limbs(static_cast<std::size_t>(m * ld * limb_count), sentinel);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), ld, limb_count) == RNS8_SUCCESS);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const auto exact = rns8::detail::exact_u64_gemm_cell(A, k, B, n, row, col, k);
      const auto expected = unsigned_limbs(exact, limb_count);
      const std::size_t offset = static_cast<std::size_t>((row * ld + col) * limb_count);
      CHECK(std::vector<uint64_t>(limbs.begin() + offset, limbs.begin() + offset + limb_count) == expected);
    }
    for (int64_t col = n; col < ld; ++col) {
      const std::size_t padding = static_cast<std::size_t>((row * ld + col) * limb_count);
      CHECK(limbs[padding] == sentinel);
      CHECK(limbs[padding + 1] == sentinel);
    }
  }

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide CPU max-width public limb export preserves padded element stride") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 1;
  constexpr int64_t ld = 3;
  constexpr uint32_t limb_count = 32;

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
    const std::vector<boost::multiprecision::cpp_int> values = {
        boost::multiprecision::cpp_int(-1),
        -(boost::multiprecision::cpp_int(1) << 63u),
        boost::multiprecision::cpp_int(0),
        boost::multiprecision::cpp_int(1),
    };
    fill_exact_residue_matrix(c_matrix, values);

    constexpr uint64_t sentinel = 0xa5a5a5a5a5a5a5a5ull;
    std::vector<uint64_t> limbs(static_cast<std::size_t>(m * ld * limb_count), sentinel);
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), ld, limb_count) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const auto expected = signed_twos_complement_limbs(values[static_cast<std::size_t>(row * n + col)], limb_count);
        const std::size_t offset = static_cast<std::size_t>((row * ld + col) * limb_count);
        CHECK(std::vector<uint64_t>(limbs.begin() + offset, limbs.begin() + offset + limb_count) == expected);
      }
      const std::size_t padding = static_cast<std::size_t>((row * ld + n) * limb_count);
      for (uint32_t limb = 0; limb < limb_count; ++limb) {
        CHECK(limbs[padding + limb] == sentinel);
      }
    }

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
    const std::vector<boost::multiprecision::cpp_int> values = {
        boost::multiprecision::cpp_int(0),
        boost::multiprecision::cpp_int(1) << 63u,
        boost::multiprecision::cpp_int(1) << 127u,
        boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()),
    };
    fill_exact_residue_matrix(c_matrix, values);

    constexpr uint64_t sentinel = 0x5a5a5a5a5a5a5a5aull;
    std::vector<uint64_t> limbs(static_cast<std::size_t>(m * ld * limb_count), sentinel);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), ld, limb_count) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const auto expected = unsigned_limbs(values[static_cast<std::size_t>(row * n + col)], limb_count);
        const std::size_t offset = static_cast<std::size_t>((row * ld + col) * limb_count);
        CHECK(std::vector<uint64_t>(limbs.begin() + offset, limbs.begin() + offset + limb_count) == expected);
      }
      const std::size_t padding = static_cast<std::size_t>((row * ld + n) * limb_count);
      for (uint32_t limb = 0; limb < limb_count; ++limb) {
        CHECK(limbs[padding + limb] == sentinel);
      }
    }

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide CPU range errors preserve every destination cell") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  constexpr int64_t k = 1;

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(
        c_matrix,
        {boost::multiprecision::cpp_int(1), boost::multiprecision::cpp_int(1) << 63u});

    constexpr uint64_t sentinel = 0x3131313131313131ull;
    std::vector<uint64_t> limbs(static_cast<std::size_t>(m * n), sentinel);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), n, 1) == RNS8_RANGE_ERROR);
    CHECK(limbs == std::vector<uint64_t>(static_cast<std::size_t>(m * n), sentinel));

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(
        c_matrix,
        {boost::multiprecision::cpp_int(1), boost::multiprecision::cpp_int(1) << 64u});

    constexpr uint64_t sentinel = 0x4141414141414141ull;
    std::vector<uint64_t> limbs(static_cast<std::size_t>(m * n), sentinel);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), n, 1) == RNS8_RANGE_ERROR);
    CHECK(limbs == std::vector<uint64_t>(static_cast<std::size_t>(m * n), sentinel));

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide CPU export rejects stale or non-RNS matrix state") {
  rns8_context* ctx = create_cpu();
  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 1;

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(c_matrix, {boost::multiprecision::cpp_int(-1)});

    uint64_t limbs[2] = {0x5151515151515151ull, 0x5252525252525252ull};
    rns8::test::set_host_residues_current(*c_matrix, false);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs, n, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0x5151515151515151ull);
    CHECK(limbs[1] == 0x5252525252525252ull);
    rns8::test::set_host_residues_current(*c_matrix, true);

    c_matrix->byte_limbs.push_back(0);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs, n, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0x5151515151515151ull);
    CHECK(limbs[1] == 0x5252525252525252ull);
    c_matrix->byte_limbs.clear();

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(c_matrix, {boost::multiprecision::cpp_int(1)});

    uint64_t limbs[2] = {0x6161616161616161ull, 0x6262626262626262ull};
    rns8::test::set_host_residues_current(*c_matrix, false);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs, n, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0x6161616161616161ull);
    CHECK(limbs[1] == 0x6262626262626262ull);
    rns8::test::set_host_residues_current(*c_matrix, true);

    rns8::test::set_host_byte_limbs_current(*c_matrix, true);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs, n, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0x6161616161616161ull);
    CHECK(limbs[1] == 0x6262626262626262ull);
    rns8::test::set_host_byte_limbs_current(*c_matrix, false);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide CPU descriptors reject bounded metadata") {
  rns8_context* ctx = create_cpu();

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 1, 1, 1);
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    desc.bound = 1;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto matrix = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
    matrix.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    rns8_matrix* out = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &out) == RNS8_INVALID_ARGUMENT);
    CHECK(out == nullptr);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, 1, 1, 1);
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
    desc.bound = 1;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto matrix = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED);
    matrix.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
    rns8_matrix* out = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &out) == RNS8_INVALID_ARGUMENT);
    CHECK(out == nullptr);
  }

  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide CPU export rejects invalid fixed-width ABI parameters") {
  rns8_context* ctx = create_cpu();
  const int64_t m = 1;
  const int64_t n = 2;
  const int64_t k = 1;
  constexpr uint64_t sentinel = 0x4d4d4d4d4d4d4d4dull;
  std::vector<uint64_t> limbs(66, sentinel);

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), n, 0) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), n, 33) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, limbs.data(), n - 1, 2) == RNS8_INVALID_ARGUMENT);
    for (const uint64_t limb : limbs) {
      CHECK(limb == sentinel);
    }

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), n, 0) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), n, 33) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, limbs.data(), n - 1, 2) ==
          RNS8_INVALID_ARGUMENT);
    for (const uint64_t limb : limbs) {
      CHECK(limb == sentinel);
    }

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(ctx);
}
