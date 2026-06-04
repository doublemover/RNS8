#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "rns8/rns8.h"

namespace {

#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
rns8_context* create_backend_context(rns8_backend_kind backend) {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = backend;
  rns8_context* ctx = nullptr;
  const int device_id = backend == RNS8_BACKEND_CPU_REFERENCE ? -1 : 0;
  REQUIRE(rns8_create_context(device_id, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

bool ck_available() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CK;
  rns8_context* ctx = nullptr;
  const rns8_status status = rns8_create_context(0, &options, &ctx);
  rns8_destroy_context(ctx);
  return status == RNS8_SUCCESS;
}

rns8_gemm_desc bounded_i64_desc(int64_t m, int64_t n, int64_t k, uint64_t bound, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc bounded_u64_desc(int64_t m, int64_t n, int64_t k, uint64_t bound, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc per_tile_u64_desc(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<uint64_t>& bounds,
    rns8_backend_kind backend) {
  rns8_gemm_desc desc = bounded_u64_desc(m, n, k, 0, backend);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  return desc;
}

rns8_gemm_desc finite_desc(
    int64_t m,
    int64_t n,
    int64_t k,
    rns8_semantics semantics,
    rns8_backend_kind backend,
    uint16_t modulus = 0) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.finite_modulus =
      modulus != 0 ? modulus : (semantics == RNS8_FINITE_FIELD_U8 ? uint16_t{251} : uint16_t{255});
  return desc;
}

rns8_gemm_desc exact_signed_desc(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_EXACT_WIDE_SIGNED;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_matrix_desc matrix_desc(
    int64_t rows,
    int64_t cols,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t prefix = 0) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind;
  desc.max_prefix = prefix;
  return desc;
}

void require_same_i64(const std::vector<int64_t>& expected, const std::vector<int64_t>& actual) {
  REQUIRE(expected.size() == actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(actual[i] == expected[i]);
  }
}

void require_same_u64(const std::vector<uint64_t>& expected, const std::vector<uint64_t>& actual) {
  REQUIRE(expected.size() == actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(actual[i] == expected[i]);
  }
}
#endif

}  // namespace

#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
TEST_CASE("CK fused bounded RNS backend matches CPU and direct HIP") {
  if (!ck_available()) {
    SKIP("CK backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* ck = create_backend_context(RNS8_BACKEND_CK);

  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;
  std::vector<int64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<int64_t> B(static_cast<std::size_t>(k * n), 0);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * k + col)] = (row + col) % 5 == 0 ? -3 : (row - col) % 7;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * n + col)] = (row * 3 + col) % 11 - 5;
    }
  }
  std::vector<int64_t> cpu_out(static_cast<std::size_t>(m * n), 0);
  std::vector<int64_t> hip_out(static_cast<std::size_t>(m * n), 0);
  std::vector<int64_t> ck_out(static_cast<std::size_t>(m * n), 0);
  auto cpu_desc = bounded_i64_desc(m, n, k, 1000000, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = bounded_i64_desc(m, n, k, 1000000, RNS8_BACKEND_HIP_DIRECT);
  auto ck_desc = bounded_i64_desc(m, n, k, 1000000, RNS8_BACKEND_CK);

  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_i64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_i64_oneshot(ck, &ck_desc, A.data(), k, B.data(), n, ck_out.data(), n) == RNS8_SUCCESS);
  require_same_i64(cpu_out, hip_out);
  require_same_i64(cpu_out, ck_out);

  rns8_destroy_context(ck);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("CK fused adaptive bounded schedule matches CPU and direct HIP") {
  if (!ck_available()) {
    SKIP("CK backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* ck = create_backend_context(RNS8_BACKEND_CK);

  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 64;
  std::vector<uint64_t> bounds = {64, 1024, uint64_t{1} << 32u, uint64_t{1} << 40u};
  std::vector<uint64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * n), 0);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * k + col)] = static_cast<uint64_t>((row + col) & 1);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * n + col)] = static_cast<uint64_t>(((row ^ col) & 1) == 0 ? 1 : 0);
    }
  }
  std::vector<uint64_t> cpu_out(static_cast<std::size_t>(m * n), 0);
  std::vector<uint64_t> hip_out(static_cast<std::size_t>(m * n), 0);
  std::vector<uint64_t> ck_out(static_cast<std::size_t>(m * n), 0);
  auto cpu_desc = per_tile_u64_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = per_tile_u64_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);
  auto ck_desc = per_tile_u64_desc(m, n, k, bounds, RNS8_BACKEND_CK);

  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_u64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_u64_oneshot(ck, &ck_desc, A.data(), k, B.data(), n, ck_out.data(), n) == RNS8_SUCCESS);
  require_same_u64(cpu_out, hip_out);
  require_same_u64(cpu_out, ck_out);

  rns8_destroy_context(ck);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("CK exact-wide RNS output matches CPU and direct HIP limbs") {
  if (!ck_available()) {
    SKIP("CK backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* ck = create_backend_context(RNS8_BACKEND_CK);
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;
  std::vector<int64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<int64_t> B(static_cast<std::size_t>(k * n), 0);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * k + col)] = (row + col) % 3 == 0 ? -7 : 5;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * n + col)] = (row + 2 * col) % 5 - 2;
    }
  }
  constexpr uint32_t limb_count = 2;
  constexpr int64_t limb_ld = n;

  auto run_backend = [&](rns8_context* ctx, rns8_backend_kind backend) {
    auto desc = exact_signed_desc(m, n, k, backend);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a = nullptr;
    rns8_matrix* b = nullptr;
    rns8_matrix* c = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    if (backend == RNS8_BACKEND_CK) {
      CHECK(
          std::string(info.selected_kernel) ==
          "ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2");
      CHECK(std::string(info.epilogue_mode) == "ck_fused_i32_to_centered_residue_rns_output");
      CHECK(info.workspace_required_bytes > 0);
      CHECK(info.accumulator_uses_int32_inner_product == 1);
      CHECK(info.accumulator_k_block_size == static_cast<uint64_t>(k));
      CHECK(info.accumulator_k_block_cap == 32768);
      CHECK(std::string(info.accumulator_type) == "int32");
      CHECK(std::string(info.accumulator_safety_status) == "safe_int32_k_block_split");
    }
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, a, A.data(), k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, b, B.data(), n, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, plan, a, b, c, workspace) == RNS8_SUCCESS);
    std::vector<uint64_t> limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c, limbs.data(), limb_ld, limb_count) == RNS8_SUCCESS);
    rns8_destroy_matrix(c);
    rns8_destroy_matrix(b);
    rns8_destroy_matrix(a);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
    return limbs;
  };

  const auto cpu_limbs = run_backend(cpu, RNS8_BACKEND_CPU_REFERENCE);
  const auto hip_limbs = run_backend(hip, RNS8_BACKEND_HIP_DIRECT);
  const auto ck_limbs = run_backend(ck, RNS8_BACKEND_CK);
  require_same_u64(cpu_limbs, hip_limbs);
  require_same_u64(cpu_limbs, ck_limbs);

  rns8_destroy_context(ck);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("CK finite u8 backend matches CPU and direct HIP across padded strides") {
  if (!ck_available()) {
    SKIP("CK backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* ck = create_backend_context(RNS8_BACKEND_CK);
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  constexpr int64_t k = 64;
  constexpr int64_t lda = k + 3;
  constexpr int64_t ldb = n + 5;
  constexpr int64_t ldc = n + 7;
  std::vector<uint8_t> A(static_cast<std::size_t>(m * lda), 0xa5);
  std::vector<uint8_t> B(static_cast<std::size_t>(k * ldb), 0x5a);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] = static_cast<uint8_t>((row * 17 + col * 23) % 251);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] = static_cast<uint8_t>((row * 29 + col * 31 + 7) % 251);
    }
  }
  std::vector<uint8_t> cpu_out(static_cast<std::size_t>(m * ldc), 0xcc);
  std::vector<uint8_t> hip_out(static_cast<std::size_t>(m * ldc), 0xcc);
  std::vector<uint8_t> ck_out(static_cast<std::size_t>(m * ldc), 0xcc);
  auto cpu_desc = finite_desc(m, n, k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = finite_desc(m, n, k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_HIP_DIRECT);
  auto ck_desc = finite_desc(m, n, k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_CK);

  REQUIRE(rns8_gemm_finite_field_u8_oneshot(
              cpu, &cpu_desc, 251, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_field_u8_oneshot(
              hip, &hip_desc, 251, A.data(), lda, B.data(), ldb, hip_out.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_field_u8_oneshot(
              ck, &ck_desc, 251, A.data(), lda, B.data(), ldb, ck_out.data(), ldc) == RNS8_SUCCESS);
  CHECK(hip_out == cpu_out);
  CHECK(ck_out == cpu_out);

  rns8_destroy_context(ck);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("CK finite u8 common moduli report specialized reducer kernels") {
  if (!ck_available()) {
    SKIP("CK backend is not available on this device");
  }

  rns8_context* ck = create_backend_context(RNS8_BACKEND_CK);
  struct Case {
    rns8_semantics semantics;
    uint16_t modulus;
    const char* selected_kernel;
  };
  const Case cases[] = {
      {RNS8_FINITE_FIELD_U8, 251, "ck_wmma_cshuffle_finite_u8_mod251_centered_epilogue_v2"},
      {RNS8_FINITE_RING_U8, 255, "ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2"},
      {RNS8_FINITE_RING_U8, 256, "ck_wmma_cshuffle_finite_u8_mod256_centered_epilogue_v2"},
  };
  for (const auto& item : cases) {
    auto desc = finite_desc(64, 64, 64, item.semantics, RNS8_BACKEND_CK, item.modulus);
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(ck, &desc, &plan) == RNS8_SUCCESS);
    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    CHECK(std::string(info.selected_kernel) == item.selected_kernel);
    CHECK(std::string(info.epilogue_mode) == "ck_fused_i32_to_centered_residue_then_canonical_u8_export");
    rns8_destroy_plan(plan);
  }
  rns8_destroy_context(ck);
}

TEST_CASE("CK finite u8 K-split preserves centered accumulation") {
  if (!ck_available()) {
    SKIP("CK backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* ck = create_backend_context(RNS8_BACKEND_CK);
  constexpr int64_t m = 64;
  constexpr int64_t n = 128;
  const int64_t k = 32769;
  std::vector<uint8_t> A(static_cast<std::size_t>(m * k), 255);
  std::vector<uint8_t> B(static_cast<std::size_t>(k * n), 255);
  std::vector<uint8_t> cpu_out(static_cast<std::size_t>(m * n), 0);
  std::vector<uint8_t> hip_out(static_cast<std::size_t>(m * n), 0);
  std::vector<uint8_t> ck_out(static_cast<std::size_t>(m * n), 0);
  auto cpu_desc = finite_desc(m, n, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_CPU_REFERENCE, 256);
  auto hip_desc = finite_desc(m, n, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_HIP_DIRECT, 256);
  auto ck_desc = finite_desc(m, n, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_CK, 256);

  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(cpu, &cpu_desc, 256, A.data(), k, B.data(), n, cpu_out.data(), n) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(hip, &hip_desc, 256, A.data(), k, B.data(), n, hip_out.data(), n) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(ck, &ck_desc, 256, A.data(), k, B.data(), n, ck_out.data(), n) ==
          RNS8_SUCCESS);
  CHECK(hip_out == cpu_out);
  CHECK(ck_out == cpu_out);

  rns8_destroy_context(ck);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}
#endif
