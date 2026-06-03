#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "backend_hipblaslt/hipblaslt_backend.hpp"
#include "core/internal.hpp"
#include "rns8/rns8.h"

namespace {

#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
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

bool hipblaslt_available() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_HIPBLASLT;
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

std::vector<int64_t> reference_i64(
    const std::vector<int64_t>& A,
    const std::vector<int64_t>& B,
    int64_t m,
    int64_t n,
    int64_t k) {
  std::vector<int64_t> out(static_cast<std::size_t>(m * n), 0);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      int64_t acc = 0;
      for (int64_t kk = 0; kk < k; ++kk) {
        acc += A[static_cast<std::size_t>(row * k + kk)] * B[static_cast<std::size_t>(kk * n + col)];
      }
      out[static_cast<std::size_t>(row * n + col)] = acc;
    }
  }
  return out;
}
#endif

}  // namespace

#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
TEST_CASE("hipBLASLt bounded baseline one-shot matches CPU and direct HIP") {
  if (!hipblaslt_available()) {
    SKIP("hipBLASLt backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* hipblaslt = create_backend_context(RNS8_BACKEND_HIPBLASLT);

  {
    constexpr int64_t m = 3;
    constexpr int64_t n = 2;
    constexpr int64_t k = 4;
    const int64_t A[m * k] = {
        std::numeric_limits<int64_t>::min(), 7, -3, 5,
        9, -11, 13, -17,
        19, 23, -29, 31};
    const int64_t B[k * n] = {
        1, 0,
        0, 3,
        0, -5,
        0, 7};
    std::vector<int64_t> cpu_out(m * n, 0);
    std::vector<int64_t> hip_out(m * n, 0);
    std::vector<int64_t> hipblaslt_out(m * n, 0);
    auto cpu_desc = bounded_i64_desc(m, n, k, uint64_t{1} << 63u, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = bounded_i64_desc(m, n, k, uint64_t{1} << 63u, RNS8_BACKEND_HIP_DIRECT);
    auto hipblaslt_desc = bounded_i64_desc(m, n, k, uint64_t{1} << 63u, RNS8_BACKEND_HIPBLASLT);

    REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A, k, B, n, cpu_out.data(), n) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_i64_oneshot(hip, &hip_desc, A, k, B, n, hip_out.data(), n) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_i64_oneshot(hipblaslt, &hipblaslt_desc, A, k, B, n, hipblaslt_out.data(), n) ==
            RNS8_SUCCESS);
    require_same_i64(cpu_out, hip_out);
    require_same_i64(cpu_out, hipblaslt_out);
  }

  {
    constexpr int64_t m = 2;
    constexpr int64_t n = 3;
    constexpr int64_t k = 3;
    const uint64_t A[m * k] = {
        std::numeric_limits<uint64_t>::max(), 0, 17,
        5, 9, 13};
    const uint64_t B[k * n] = {
        1, 0, 0,
        0, 5, 6,
        0, 8, 9};
    std::vector<uint64_t> cpu_out(m * n, 0);
    std::vector<uint64_t> hip_out(m * n, 0);
    std::vector<uint64_t> hipblaslt_out(m * n, 0);
    auto cpu_desc = bounded_u64_desc(m, n, k, std::numeric_limits<uint64_t>::max(), RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = bounded_u64_desc(m, n, k, std::numeric_limits<uint64_t>::max(), RNS8_BACKEND_HIP_DIRECT);
    auto hipblaslt_desc = bounded_u64_desc(m, n, k, std::numeric_limits<uint64_t>::max(), RNS8_BACKEND_HIPBLASLT);

    REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A, k, B, n, cpu_out.data(), n) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_u64_oneshot(hip, &hip_desc, A, k, B, n, hip_out.data(), n) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_u64_oneshot(hipblaslt, &hipblaslt_desc, A, k, B, n, hipblaslt_out.data(), n) ==
            RNS8_SUCCESS);
    require_same_u64(cpu_out, hip_out);
    require_same_u64(cpu_out, hipblaslt_out);
  }

  rns8_destroy_context(hipblaslt);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("hipBLASLt workspace B prepack cache reuses stable source versions") {
  if (!hipblaslt_available()) {
    SKIP("hipBLASLt backend is not available on this device");
  }

  rns8_context* hipblaslt = create_backend_context(RNS8_BACKEND_HIPBLASLT);
  constexpr int64_t m = 4;
  constexpr int64_t n = 3;
  constexpr int64_t k = 5;
  const std::vector<int64_t> A0 = {
      1, -2, 3, -4, 5,
      6, 7, -8, 9, -10,
      -11, 12, 13, -14, 15,
      16, -17, 18, 19, -20};
  const std::vector<int64_t> A1 = {
      -3, 4, 5, -6, 7,
      8, -9, 10, 11, -12,
      13, 14, -15, 16, -17,
      -18, 19, 20, -21, 22};
  const std::vector<int64_t> B0 = {
      2, -3, 4,
      -5, 6, -7,
      8, 9, -10,
      -11, 12, 13,
      14, -15, 16};
  const std::vector<int64_t> B1 = {
      -2, 5, -8,
      11, -14, 17,
      -20, 23, -26,
      29, -32, 35,
      -38, 41, -44};

  auto desc = bounded_i64_desc(m, n, k, 10000, RNS8_BACKEND_HIPBLASLT);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(hipblaslt, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hipblaslt, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  REQUIRE(rns8_create_matrix(hipblaslt, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hipblaslt, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hipblaslt, &c_desc, &c_matrix) == RNS8_SUCCESS);
  CHECK(workspace->hipblaslt_b_prepack_cache == nullptr);
  CHECK_FALSE(workspace->hipblaslt_b_prepack_current);

  std::vector<int64_t> out(static_cast<std::size_t>(m * n), 0);
  REQUIRE(rns8_pack_i64(hipblaslt, a_matrix, A0.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hipblaslt, b_matrix, B0.data(), n, 10) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hipblaslt, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(hipblaslt, plan, c_matrix, out.data(), n) == RNS8_SUCCESS);
  require_same_i64(reference_i64(A0, B0, m, n, k), out);
  REQUIRE(workspace->hipblaslt_b_prepack_cache != nullptr);
  CHECK(workspace->hipblaslt_b_prepack_current);
  CHECK(workspace->hipblaslt_b_prepack_source_version == 10);
  CHECK(workspace->hipblaslt_b_prepack_k == k);
  CHECK(workspace->hipblaslt_b_prepack_n == n);
  CHECK(workspace->hipblaslt_b_prepack_ldb == n);
  CHECK(workspace->hipblaslt_b_prepack_prefix == plan->prefix);
  const void* first_cache_ptr = workspace->hipblaslt_b_prepack_cache;
  const std::size_t first_cache_bytes = workspace->hipblaslt_b_prepack_cache_bytes;

  REQUIRE(rns8_pack_i64(hipblaslt, a_matrix, A1.data(), k, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hipblaslt, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(hipblaslt, plan, c_matrix, out.data(), n) == RNS8_SUCCESS);
  require_same_i64(reference_i64(A1, B0, m, n, k), out);
  CHECK(workspace->hipblaslt_b_prepack_cache == first_cache_ptr);
  CHECK(workspace->hipblaslt_b_prepack_cache_bytes == first_cache_bytes);
  CHECK(workspace->hipblaslt_b_prepack_current);
  CHECK(workspace->hipblaslt_b_prepack_source_version == 10);

  REQUIRE(rns8_pack_i64(hipblaslt, b_matrix, B1.data(), n, 11) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hipblaslt, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(hipblaslt, plan, c_matrix, out.data(), n) == RNS8_SUCCESS);
  require_same_i64(reference_i64(A1, B1, m, n, k), out);
  CHECK(workspace->hipblaslt_b_prepack_cache == first_cache_ptr);
  CHECK(workspace->hipblaslt_b_prepack_cache_bytes == first_cache_bytes);
  CHECK(workspace->hipblaslt_b_prepack_current);
  CHECK(workspace->hipblaslt_b_prepack_source_version == 11);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(hipblaslt);
}

TEST_CASE("hipBLASLt baseline preserves K-split modular accumulation") {
  if (!hipblaslt_available()) {
    SKIP("hipBLASLt backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* hipblaslt = create_backend_context(RNS8_BACKEND_HIPBLASLT);
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  std::vector<int64_t> A(static_cast<std::size_t>(k), 1);
  std::vector<int64_t> B(static_cast<std::size_t>(k), 1);
  int64_t cpu_out = 0;
  int64_t hip_out = 0;
  int64_t hipblaslt_out = 0;
  auto cpu_desc = bounded_i64_desc(1, 1, k, static_cast<uint64_t>(k), RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = bounded_i64_desc(1, 1, k, static_cast<uint64_t>(k), RNS8_BACKEND_HIP_DIRECT);
  auto hipblaslt_desc = bounded_i64_desc(1, 1, k, static_cast<uint64_t>(k), RNS8_BACKEND_HIPBLASLT);

  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), 1, &cpu_out, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_i64_oneshot(hip, &hip_desc, A.data(), k, B.data(), 1, &hip_out, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_i64_oneshot(hipblaslt, &hipblaslt_desc, A.data(), k, B.data(), 1, &hipblaslt_out, 1) ==
          RNS8_SUCCESS);
  CHECK(cpu_out == k);
  CHECK(hip_out == cpu_out);
  CHECK(hipblaslt_out == cpu_out);

  {
    constexpr int64_t m = 2;
    constexpr int64_t n = 3;
    constexpr int64_t small_k = 5;
    constexpr int64_t lda = 7;
    constexpr int64_t ldb = 4;
    constexpr int64_t ldc = 5;
    std::vector<uint8_t> high_a(static_cast<std::size_t>(m * lda), 0xa5);
    std::vector<uint8_t> high_b(static_cast<std::size_t>(small_k * ldb), 0x5a);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < small_k; ++col) {
        high_a[static_cast<std::size_t>(row * lda + col)] =
            static_cast<uint8_t>((180 + row * 31 + col * 19) % 251);
      }
    }
    for (int64_t row = 0; row < small_k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        high_b[static_cast<std::size_t>(row * ldb + col)] =
            static_cast<uint8_t>((170 + row * 23 + col * 41) % 251);
      }
    }
    std::vector<uint8_t> cpu_high_out(static_cast<std::size_t>(m * ldc), 0xcc);
    std::vector<uint8_t> hip_high_out(static_cast<std::size_t>(m * ldc), 0xcc);
    std::vector<uint8_t> hipblaslt_high_out(static_cast<std::size_t>(m * ldc), 0xcc);
    auto cpu_field_desc = finite_desc(m, n, small_k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_field_desc = finite_desc(m, n, small_k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_HIP_DIRECT);
    auto hipblaslt_field_desc = finite_desc(m, n, small_k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_HIPBLASLT);

    REQUIRE(rns8_gemm_finite_field_u8_oneshot(
                cpu, &cpu_field_desc, 251, high_a.data(), lda, high_b.data(), ldb, cpu_high_out.data(), ldc) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_field_u8_oneshot(
                hip, &hip_field_desc, 251, high_a.data(), lda, high_b.data(), ldb, hip_high_out.data(), ldc) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_field_u8_oneshot(
                hipblaslt,
                &hipblaslt_field_desc,
                251,
                high_a.data(),
                lda,
                high_b.data(),
                ldb,
                hipblaslt_high_out.data(),
                ldc) == RNS8_SUCCESS);
    CHECK(hip_high_out == cpu_high_out);
    CHECK(hipblaslt_high_out == cpu_high_out);
  }

  rns8_destroy_context(hipblaslt);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("hipBLASLt finite u8 baseline matches CPU and direct HIP across K split") {
  if (!hipblaslt_available()) {
    SKIP("hipBLASLt backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* hipblaslt = create_backend_context(RNS8_BACKEND_HIPBLASLT);
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  std::vector<uint8_t> A(static_cast<std::size_t>(k), 255);
  std::vector<uint8_t> B(static_cast<std::size_t>(k), 255);
  uint8_t cpu_out = 0;
  uint8_t hip_out = 0;
  uint8_t hipblaslt_out = 0;
  auto cpu_desc = finite_desc(1, 1, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_CPU_REFERENCE, 256);
  auto hip_desc = finite_desc(1, 1, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_HIP_DIRECT, 256);
  auto hipblaslt_desc = finite_desc(1, 1, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_HIPBLASLT, 256);

  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(cpu, &cpu_desc, 256, A.data(), k, B.data(), 1, &cpu_out, 1) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(hip, &hip_desc, 256, A.data(), k, B.data(), 1, &hip_out, 1) ==
          RNS8_SUCCESS);
  REQUIRE(
      rns8_gemm_finite_ring_u8_oneshot(hipblaslt, &hipblaslt_desc, 256, A.data(), k, B.data(), 1, &hipblaslt_out, 1) ==
      RNS8_SUCCESS);
  CHECK(hip_out == cpu_out);
  CHECK(hipblaslt_out == cpu_out);

  rns8_destroy_context(hipblaslt);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("hipBLASLt exact-wide RNS output matches CPU and direct HIP limbs") {
  if (!hipblaslt_available()) {
    SKIP("hipBLASLt backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* hipblaslt = create_backend_context(RNS8_BACKEND_HIPBLASLT);
  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 3;
  const int64_t A[m * k] = {-7, 11, -13, 17, -19, 23};
  const int64_t B[k * n] = {29, -31, 37, -41, 43, -47};
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
    if (backend == RNS8_BACKEND_HIPBLASLT) {
      CHECK(std::string(info.selected_kernel) == "hipblaslt_int8_i32_scratch_reduce_baseline_v1");
      CHECK(info.workspace_required_bytes > rns8::detail::kHipblasLtBaselineWorkspaceBytes);
    }
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, RNS8_MAX_SUPPORTED_PREFIX);
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, a, A, k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, b, B, n, 2) == RNS8_SUCCESS);
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
  const auto hipblaslt_limbs = run_backend(hipblaslt, RNS8_BACKEND_HIPBLASLT);
  require_same_u64(cpu_limbs, hip_limbs);
  require_same_u64(cpu_limbs, hipblaslt_limbs);

  rns8_destroy_context(hipblaslt);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("hipBLASLt rejects adaptive per-tile bounded schedules") {
  if (!hipblaslt_available()) {
    SKIP("hipBLASLt backend is not available on this device");
  }

  rns8_context* hipblaslt = create_backend_context(RNS8_BACKEND_HIPBLASLT);
  std::vector<uint64_t> bounds = {100, 200, 300, 400};
  auto desc = bounded_u64_desc(65, 65, 3, 0, RNS8_BACKEND_HIPBLASLT);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(hipblaslt, &desc, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);

  rns8_destroy_context(hipblaslt);
}
#endif
