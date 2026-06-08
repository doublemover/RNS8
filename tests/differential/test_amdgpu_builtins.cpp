#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "core/internal.hpp"
#include "rns8/rns8.h"

namespace {

rns8_context* create_context_or_null(rns8_backend_kind backend) {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = backend;
  rns8_context* ctx = nullptr;
  const int device_id = backend == RNS8_BACKEND_CPU_REFERENCE ? -1 : 0;
  if (rns8_create_context(device_id, &options, &ctx) != RNS8_SUCCESS) {
    return nullptr;
  }
  return ctx;
}

rns8_context* require_context(rns8_backend_kind backend) {
  rns8_context* ctx = create_context_or_null(backend);
  REQUIRE(ctx != nullptr);
  return ctx;
}

bool amdgpu_builtins_available() {
  rns8_context* ctx = create_context_or_null(RNS8_BACKEND_AMDGPU_BUILTINS);
  const bool available = ctx != nullptr;
  rns8_destroy_context(ctx);
  return available;
}

std::string context_target(rns8_context* ctx) {
  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(ctx, &info) == RNS8_SUCCESS);
  return info.gcn_arch;
}

bool sparse_runtime_target(const std::string& target) {
  return target.rfind("gfx942", 0) == 0 || target.rfind("gfx1200", 0) == 0 || target.rfind("gfx1201", 0) == 0;
}

bool has_timing_label(
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    const std::string& label) {
  for (const auto& sample : samples) {
    if (sample.label == label) {
      return true;
    }
  }
  return false;
}

uint8_t nonzero_mod251(int64_t value) {
  const uint8_t residue = static_cast<uint8_t>(value % 251);
  return residue == 0 ? uint8_t{1} : residue;
}

rns8_gemm_desc bounded_i64_desc(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = 4096;
  return desc;
}

rns8_gemm_desc bounded_u64_desc(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = 4096;
  return desc;
}

rns8_gemm_desc finite_desc(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_FINITE_FIELD_U8;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.finite_modulus = 251;
  return desc;
}

rns8_gemm_desc exact_wide_desc(
    int64_t m,
    int64_t n,
    int64_t k,
    rns8_semantics semantics,
    rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_matrix_desc finite_matrix_desc(int64_t rows, int64_t cols) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = RNS8_FINITE_FIELD_U8;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  return desc;
}

rns8_matrix_desc exact_wide_matrix_desc(int64_t rows, int64_t cols, rns8_semantics semantics) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_sparse_matrix_desc exact_wide_sparse_a_desc(int64_t rows, int64_t expanded_k, rns8_semantics semantics) {
  rns8_sparse_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
  desc.rows = rows;
  desc.expanded_k = expanded_k;
  desc.group_size = 4;
  desc.nonzeros_per_group = 2;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_sparse_matrix_desc bounded_sparse_a_desc(
    int64_t rows,
    int64_t expanded_k,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t max_prefix) {
  rns8_sparse_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = bound_kind;
  desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
  desc.rows = rows;
  desc.expanded_k = expanded_k;
  desc.group_size = 4;
  desc.nonzeros_per_group = 2;
  desc.max_prefix = max_prefix;
  return desc;
}

rns8_matrix_desc bounded_i64_matrix_desc(int64_t rows, int64_t cols) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

rns8_matrix_desc bounded_u64_matrix_desc(int64_t rows, int64_t cols) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

void require_same_u64(const std::vector<uint64_t>& expected, const std::vector<uint64_t>& actual) {
  REQUIRE(expected.size() == actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (actual[i] != expected[i]) {
      CAPTURE(i);
      CHECK(actual[i] == expected[i]);
      return;
    }
  }
}

}  // namespace

#if defined(RNS8_ENABLE_AMDGPU_BUILTINS) && RNS8_ENABLE_AMDGPU_BUILTINS && \
    defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE

TEST_CASE("AMDGPU builtin dense bounded RNS backend matches CPU") {
  if (!amdgpu_builtins_available()) {
    SKIP("AMDGPU builtin backend is not available on this device");
  }

  rns8_context* cpu = require_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* amdgpu = require_context(RNS8_BACKEND_AMDGPU_BUILTINS);
  constexpr int64_t m = 16;
  constexpr int64_t n = 16;
  constexpr int64_t k = 32;
  std::vector<int64_t> A(static_cast<std::size_t>(m * k));
  std::vector<int64_t> B(static_cast<std::size_t>(k * n));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * k + col)] = static_cast<int64_t>((row * 3 + col * 5) % 9) - 4;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * n + col)] = static_cast<int64_t>((row * 7 + col * 11 + 1) % 9) - 4;
    }
  }
  std::vector<int64_t> cpu_out(static_cast<std::size_t>(m * n), 0);
  std::vector<int64_t> gpu_out(static_cast<std::size_t>(m * n), 0);
  auto cpu_desc = bounded_i64_desc(m, n, k, RNS8_BACKEND_CPU_REFERENCE);
  auto gpu_desc = bounded_i64_desc(m, n, k, RNS8_BACKEND_AMDGPU_BUILTINS);

  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_out.data(), n) == RNS8_SUCCESS);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(amdgpu, &gpu_desc, &plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(amdgpu, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = bounded_i64_matrix_desc(m, k);
  auto b_desc = bounded_i64_matrix_desc(k, n);
  auto c_desc = bounded_i64_matrix_desc(m, n);
  rns8_matrix* a = nullptr;
  rns8_matrix* b = nullptr;
  rns8_matrix* c = nullptr;
  REQUIRE(rns8_create_matrix(amdgpu, &a_desc, &a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(amdgpu, &b_desc, &b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &c) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(amdgpu, a, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(amdgpu, b, B.data(), n, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(amdgpu, plan, a, b, c, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(amdgpu, plan, c, gpu_out.data(), n) == RNS8_SUCCESS);
  CHECK(gpu_out == cpu_out);

  rns8_destroy_matrix(c);
  rns8_destroy_matrix(b);
  rns8_destroy_matrix(a);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(amdgpu);
  rns8_destroy_context(cpu);
}

TEST_CASE("AMDGPU builtin dense bounded RNS backend handles tile and K tails") {
  if (!amdgpu_builtins_available()) {
    SKIP("AMDGPU builtin backend is not available on this device");
  }

  rns8_context* cpu = require_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* amdgpu = require_context(RNS8_BACKEND_AMDGPU_BUILTINS);
  struct Shape {
    int64_t m;
    int64_t n;
    int64_t k;
  };
  const Shape shapes[] = {
      {17, 19, 35},
      {33, 18, 47},
  };
  for (const Shape shape : shapes) {
    CAPTURE(shape.m, shape.n, shape.k);
    std::vector<int64_t> A(static_cast<std::size_t>(shape.m * shape.k));
    std::vector<int64_t> B(static_cast<std::size_t>(shape.k * shape.n));
    for (int64_t row = 0; row < shape.m; ++row) {
      for (int64_t col = 0; col < shape.k; ++col) {
        A[static_cast<std::size_t>(row * shape.k + col)] =
            static_cast<int64_t>((row * 5 + col * 7 + 3) % 11) - 5;
      }
    }
    for (int64_t row = 0; row < shape.k; ++row) {
      for (int64_t col = 0; col < shape.n; ++col) {
        B[static_cast<std::size_t>(row * shape.n + col)] =
            static_cast<int64_t>((row * 13 + col * 17 + 1) % 13) - 6;
      }
    }
    std::vector<int64_t> cpu_out(static_cast<std::size_t>(shape.m * shape.n), 0);
    std::vector<int64_t> gpu_out(static_cast<std::size_t>(shape.m * shape.n), 0);
    auto cpu_desc = bounded_i64_desc(shape.m, shape.n, shape.k, RNS8_BACKEND_CPU_REFERENCE);
    auto gpu_desc = bounded_i64_desc(shape.m, shape.n, shape.k, RNS8_BACKEND_AMDGPU_BUILTINS);

    REQUIRE(
        rns8_gemm_i64_oneshot(
            cpu, &cpu_desc, A.data(), shape.k, B.data(), shape.n, cpu_out.data(), shape.n) == RNS8_SUCCESS);
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(amdgpu, &gpu_desc, &plan) == RNS8_SUCCESS);
    rns8_workspace* workspace = nullptr;
    REQUIRE(rns8_create_workspace(amdgpu, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = bounded_i64_matrix_desc(shape.m, shape.k);
    auto b_desc = bounded_i64_matrix_desc(shape.k, shape.n);
    auto c_desc = bounded_i64_matrix_desc(shape.m, shape.n);
    rns8_matrix* a = nullptr;
    rns8_matrix* b = nullptr;
    rns8_matrix* c = nullptr;
    REQUIRE(rns8_create_matrix(amdgpu, &a_desc, &a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &b_desc, &b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(amdgpu, a, A.data(), shape.k, 11) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(amdgpu, b, B.data(), shape.n, 12) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(amdgpu, plan, a, b, c, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_i64(amdgpu, plan, c, gpu_out.data(), shape.n) == RNS8_SUCCESS);
    CHECK(gpu_out == cpu_out);

    rns8_destroy_matrix(c);
    rns8_destroy_matrix(b);
    rns8_destroy_matrix(a);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(amdgpu);
  rns8_destroy_context(cpu);
}

TEST_CASE("AMDGPU builtin dense exact-wide RNS backend matches CPU") {
  if (!amdgpu_builtins_available()) {
    SKIP("AMDGPU builtin backend is not available on this device");
  }

  rns8_context* cpu = require_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* amdgpu = require_context(RNS8_BACKEND_AMDGPU_BUILTINS);
  constexpr int64_t m = 16;
  constexpr int64_t n = 16;
  constexpr int64_t k = 32;
  constexpr uint32_t limb_count = 3;
  constexpr int64_t limb_ld = n;
  std::vector<int64_t> signed_a(static_cast<std::size_t>(m * k), 0);
  std::vector<int64_t> signed_b(static_cast<std::size_t>(k * n), 0);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      signed_a[static_cast<std::size_t>(row * k + col)] =
          static_cast<int64_t>((row * 5 + col * 7 + 3) % 17) - 8;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      signed_b[static_cast<std::size_t>(row * n + col)] =
          static_cast<int64_t>((row * 11 + col * 13 + 5) % 19) - 9;
    }
  }
  std::vector<uint64_t> unsigned_a(static_cast<std::size_t>(m * k), 0);
  std::vector<uint64_t> unsigned_b(static_cast<std::size_t>(k * n), 0);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      unsigned_a[static_cast<std::size_t>(row * k + col)] =
          static_cast<uint64_t>((row * 17 + col * 19 + 7) % 251 + 1);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      unsigned_b[static_cast<std::size_t>(row * n + col)] =
          static_cast<uint64_t>((row * 23 + col * 29 + 11) % 239 + 1);
    }
  }

  auto run_signed_backend = [&](rns8_context* ctx, rns8_backend_kind backend) {
    auto desc = exact_wide_desc(m, n, k, RNS8_EXACT_WIDE_SIGNED, backend);
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
    if (backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
      CHECK(std::string(info.selected_kernel).rfind("amdgpu_builtin_", 0) == 0);
      CHECK(std::string(info.epilogue_mode) == "amdgpu_builtin_fused_i32_to_centered_residue_rns_output");
      CHECK(info.accumulator_uses_int32_inner_product == 1);
      CHECK(std::string(info.accumulator_type) == "int32");
    }
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = exact_wide_matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED);
    auto b_desc = exact_wide_matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED);
    auto c_desc = exact_wide_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, a, signed_a.data(), k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, b, signed_b.data(), n, 2) == RNS8_SUCCESS);
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

  auto run_unsigned_backend = [&](rns8_context* ctx, rns8_backend_kind backend) {
    auto desc = exact_wide_desc(m, n, k, RNS8_EXACT_WIDE_UNSIGNED, backend);
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
    if (backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
      CHECK(std::string(info.selected_kernel).rfind("amdgpu_builtin_", 0) == 0);
      CHECK(std::string(info.epilogue_mode) == "amdgpu_builtin_fused_i32_to_centered_residue_rns_output");
      CHECK(info.accumulator_uses_int32_inner_product == 1);
      CHECK(std::string(info.accumulator_type) == "int32");
    }
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = exact_wide_matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED);
    auto b_desc = exact_wide_matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED);
    auto c_desc = exact_wide_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(ctx, a, unsigned_a.data(), k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(ctx, b, unsigned_b.data(), n, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, plan, a, b, c, workspace) == RNS8_SUCCESS);
    std::vector<uint64_t> limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c, limbs.data(), limb_ld, limb_count) == RNS8_SUCCESS);
    rns8_destroy_matrix(c);
    rns8_destroy_matrix(b);
    rns8_destroy_matrix(a);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
    return limbs;
  };

  require_same_u64(run_signed_backend(cpu, RNS8_BACKEND_CPU_REFERENCE),
                   run_signed_backend(amdgpu, RNS8_BACKEND_AMDGPU_BUILTINS));
  require_same_u64(run_unsigned_backend(cpu, RNS8_BACKEND_CPU_REFERENCE),
                   run_unsigned_backend(amdgpu, RNS8_BACKEND_AMDGPU_BUILTINS));

  rns8_destroy_context(amdgpu);
  rns8_destroy_context(cpu);
}

TEST_CASE("AMDGPU builtin dense finite u8 backend matches CPU") {
  if (!amdgpu_builtins_available()) {
    SKIP("AMDGPU builtin backend is not available on this device");
  }

  rns8_context* cpu = require_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* amdgpu = require_context(RNS8_BACKEND_AMDGPU_BUILTINS);
  constexpr int64_t m = 16;
  constexpr int64_t n = 16;
  constexpr int64_t k = 32;
  std::vector<uint8_t> A(static_cast<std::size_t>(m * k));
  std::vector<uint8_t> B(static_cast<std::size_t>(k * n));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * k + col)] = static_cast<uint8_t>((row * 13 + col * 17) % 251);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * n + col)] = static_cast<uint8_t>((row * 19 + col * 23 + 3) % 251);
    }
  }
  std::vector<uint8_t> cpu_out(static_cast<std::size_t>(m * n), 0);
  std::vector<uint8_t> gpu_out(static_cast<std::size_t>(m * n), 0);
  auto cpu_desc = finite_desc(m, n, k, RNS8_BACKEND_CPU_REFERENCE);
  auto gpu_desc = finite_desc(m, n, k, RNS8_BACKEND_AMDGPU_BUILTINS);

  REQUIRE(rns8_gemm_finite_field_u8_oneshot(cpu, &cpu_desc, 251, A.data(), k, B.data(), n, cpu_out.data(), n) ==
          RNS8_SUCCESS);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(amdgpu, &gpu_desc, &plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(amdgpu, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = finite_matrix_desc(m, k);
  auto b_desc = finite_matrix_desc(k, n);
  auto c_desc = finite_matrix_desc(m, n);
  rns8_matrix* a = nullptr;
  rns8_matrix* b = nullptr;
  rns8_matrix* c = nullptr;
  REQUIRE(rns8_create_matrix(amdgpu, &a_desc, &a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(amdgpu, &b_desc, &b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &c) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(amdgpu, a, 251, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(amdgpu, b, 251, B.data(), n, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_u8(amdgpu, plan, 251, a, b, c, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_finite_u8(amdgpu, plan, 251, c, gpu_out.data(), n) == RNS8_SUCCESS);
  CHECK(gpu_out == cpu_out);

  rns8_destroy_matrix(c);
  rns8_destroy_matrix(b);
  rns8_destroy_matrix(a);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(amdgpu);
  rns8_destroy_context(cpu);
}

TEST_CASE("AMDGPU builtin explicit sparse-A bounded backend matches dense backend") {
  if (!amdgpu_builtins_available()) {
    SKIP("AMDGPU builtin backend is not available on this device");
  }

  rns8_context* cpu = require_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* amdgpu = require_context(RNS8_BACKEND_AMDGPU_BUILTINS);
  if (!sparse_runtime_target(context_target(amdgpu))) {
    rns8_destroy_context(amdgpu);
    rns8_destroy_context(cpu);
    SKIP("AMDGPU sparse-A builtin runtime requires CDNA3 SMFMAC or RDNA4 SWMMAC target");
  }

  constexpr int64_t m = 16;
  constexpr int64_t n = 16;
  constexpr int64_t k = 32;

  auto run_signed = [&]() {
    std::vector<int64_t> A(static_cast<std::size_t>(m * k), 0);
    std::vector<int64_t> B(static_cast<std::size_t>(k * n), 0);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t group = 0; group < k / 4; ++group) {
        A[static_cast<std::size_t>(row * k + group * 4 + 0)] =
            static_cast<int64_t>((row * 3 + group * 5 + 1) % 11) - 5;
        A[static_cast<std::size_t>(row * k + group * 4 + 2)] =
            static_cast<int64_t>((row * 7 + group * 11 + 3) % 13) - 6;
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * n + col)] =
            static_cast<int64_t>((row * 13 + col * 17 + 5) % 15) - 7;
      }
    }

    auto cpu_a_desc = bounded_i64_matrix_desc(m, k);
    rns8_matrix* cpu_a = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &cpu_a_desc, &cpu_a) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(cpu, cpu_a, A.data(), k, 101) == RNS8_SUCCESS);

    auto gemm = bounded_i64_desc(m, n, k, RNS8_BACKEND_AMDGPU_BUILTINS);
    gemm.bound = 8192;
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    REQUIRE(rns8_create_plan(amdgpu, &gemm, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(amdgpu, plan, &workspace) == RNS8_SUCCESS);

    auto a_desc = bounded_i64_matrix_desc(m, k);
    auto b_desc = bounded_i64_matrix_desc(k, n);
    auto c_desc = bounded_i64_matrix_desc(m, n);
    rns8_matrix* dense_a = nullptr;
    rns8_matrix* dense_b = nullptr;
    rns8_matrix* dense_c = nullptr;
    rns8_matrix* sparse_c = nullptr;
    REQUIRE(rns8_create_matrix(amdgpu, &a_desc, &dense_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &b_desc, &dense_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &dense_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &sparse_c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(amdgpu, dense_a, A.data(), k, 101) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(amdgpu, dense_b, B.data(), n, 202) == RNS8_SUCCESS);

    auto sparse_desc = bounded_sparse_a_desc(
        m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS, RNS8_DEFAULT_BOUNDED_PREFIX);
    rns8_sparse_matrix* sparse_a = nullptr;
    REQUIRE(rns8_create_sparse_matrix(amdgpu, &sparse_desc, &sparse_a) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(
                amdgpu,
                sparse_a,
                reinterpret_cast<const uint8_t*>(cpu_a->residues.data()),
                cpu_a->desc.cols,
                cpu_a->source_version) == RNS8_SUCCESS);

    REQUIRE(rns8_gemm_rns(amdgpu, plan, dense_a, dense_b, dense_c, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns_sparse_a(amdgpu, plan, sparse_a, dense_b, sparse_c, workspace) == RNS8_SUCCESS);
    std::vector<int64_t> dense_out(static_cast<std::size_t>(m * n), 0);
    std::vector<int64_t> sparse_out(static_cast<std::size_t>(m * n), 0);
    REQUIRE(rns8_export_i64(amdgpu, plan, dense_c, dense_out.data(), n) == RNS8_SUCCESS);
    REQUIRE(rns8_export_i64(amdgpu, plan, sparse_c, sparse_out.data(), n) == RNS8_SUCCESS);
    CHECK(sparse_out == dense_out);

    rns8_destroy_sparse_matrix(sparse_a);
    rns8_destroy_matrix(sparse_c);
    rns8_destroy_matrix(dense_c);
    rns8_destroy_matrix(dense_b);
    rns8_destroy_matrix(dense_a);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
    rns8_destroy_matrix(cpu_a);
  };

  auto run_unsigned = [&]() {
    std::vector<uint64_t> A(static_cast<std::size_t>(m * k), 0);
    std::vector<uint64_t> B(static_cast<std::size_t>(k * n), 0);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t group = 0; group < k / 4; ++group) {
        A[static_cast<std::size_t>(row * k + group * 4 + 0)] =
            static_cast<uint64_t>((row * 3 + group * 5 + 1) % 17 + 1);
        A[static_cast<std::size_t>(row * k + group * 4 + 2)] =
            static_cast<uint64_t>((row * 7 + group * 11 + 3) % 19 + 1);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * n + col)] =
            static_cast<uint64_t>((row * 13 + col * 17 + 5) % 19 + 1);
      }
    }

    auto cpu_a_desc = bounded_u64_matrix_desc(m, k);
    rns8_matrix* cpu_a = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &cpu_a_desc, &cpu_a) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), k, 301) == RNS8_SUCCESS);

    auto gemm = bounded_u64_desc(m, n, k, RNS8_BACKEND_AMDGPU_BUILTINS);
    gemm.bound = 8192;
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    REQUIRE(rns8_create_plan(amdgpu, &gemm, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(amdgpu, plan, &workspace) == RNS8_SUCCESS);

    auto a_desc = bounded_u64_matrix_desc(m, k);
    auto b_desc = bounded_u64_matrix_desc(k, n);
    auto c_desc = bounded_u64_matrix_desc(m, n);
    rns8_matrix* dense_a = nullptr;
    rns8_matrix* dense_b = nullptr;
    rns8_matrix* dense_c = nullptr;
    rns8_matrix* sparse_c = nullptr;
    REQUIRE(rns8_create_matrix(amdgpu, &a_desc, &dense_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &b_desc, &dense_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &dense_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &sparse_c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(amdgpu, dense_a, A.data(), k, 301) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(amdgpu, dense_b, B.data(), n, 402) == RNS8_SUCCESS);

    auto sparse_desc = bounded_sparse_a_desc(
        m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED, RNS8_DEFAULT_BOUNDED_PREFIX);
    rns8_sparse_matrix* sparse_a = nullptr;
    REQUIRE(rns8_create_sparse_matrix(amdgpu, &sparse_desc, &sparse_a) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(
                amdgpu,
                sparse_a,
                reinterpret_cast<const uint8_t*>(cpu_a->residues.data()),
                cpu_a->desc.cols,
                cpu_a->source_version) == RNS8_SUCCESS);

    REQUIRE(rns8_gemm_rns(amdgpu, plan, dense_a, dense_b, dense_c, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns_sparse_a(amdgpu, plan, sparse_a, dense_b, sparse_c, workspace) == RNS8_SUCCESS);
    std::vector<uint64_t> dense_out(static_cast<std::size_t>(m * n), 0);
    std::vector<uint64_t> sparse_out(static_cast<std::size_t>(m * n), 0);
    REQUIRE(rns8_export_u64(amdgpu, plan, dense_c, dense_out.data(), n) == RNS8_SUCCESS);
    REQUIRE(rns8_export_u64(amdgpu, plan, sparse_c, sparse_out.data(), n) == RNS8_SUCCESS);
    CHECK(sparse_out == dense_out);

    rns8_destroy_sparse_matrix(sparse_a);
    rns8_destroy_matrix(sparse_c);
    rns8_destroy_matrix(dense_c);
    rns8_destroy_matrix(dense_b);
    rns8_destroy_matrix(dense_a);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
    rns8_destroy_matrix(cpu_a);
  };

  run_signed();
  run_unsigned();

  rns8_destroy_context(amdgpu);
  rns8_destroy_context(cpu);
}

TEST_CASE("AMDGPU builtin explicit sparse-A finite u8 backend matches dense backend") {
  if (!amdgpu_builtins_available()) {
    SKIP("AMDGPU builtin backend is not available on this device");
  }

  rns8_context* amdgpu = require_context(RNS8_BACKEND_AMDGPU_BUILTINS);
  if (!sparse_runtime_target(context_target(amdgpu))) {
    rns8_destroy_context(amdgpu);
    SKIP("AMDGPU sparse-A builtin runtime requires CDNA3 SMFMAC or RDNA4 SWMMAC target");
  }

  constexpr int64_t m = 16;
  constexpr int64_t n = 16;
  constexpr int64_t k = 32;
  std::vector<uint8_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<uint8_t> B(static_cast<std::size_t>(k * n));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t group = 0; group < k / 4; ++group) {
      A[static_cast<std::size_t>(row * k + group * 4 + 0)] =
          nonzero_mod251(row * 13 + group * 17 + 1);
      A[static_cast<std::size_t>(row * k + group * 4 + 2)] =
          nonzero_mod251(row * 19 + group * 23 + 3);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * n + col)] = static_cast<uint8_t>((row * 29 + col * 31 + 5) % 251);
    }
  }

  auto gemm = finite_desc(m, n, k, RNS8_BACKEND_AMDGPU_BUILTINS);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(amdgpu, &gemm, &plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(amdgpu, plan, &workspace) == RNS8_SUCCESS);

  auto a_desc = finite_matrix_desc(m, k);
  auto b_desc = finite_matrix_desc(k, n);
  auto c_desc = finite_matrix_desc(m, n);
  rns8_matrix* dense_a = nullptr;
  rns8_matrix* dense_b = nullptr;
  rns8_matrix* dense_c = nullptr;
  rns8_matrix* sparse_c = nullptr;
  REQUIRE(rns8_create_matrix(amdgpu, &a_desc, &dense_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(amdgpu, &b_desc, &dense_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &dense_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &sparse_c) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(amdgpu, dense_a, 251, A.data(), k, 11) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(amdgpu, dense_b, 251, B.data(), n, 22) == RNS8_SUCCESS);

  rns8_sparse_matrix_desc sparse_desc{};
  sparse_desc.struct_size = sizeof(sparse_desc);
  sparse_desc.abi_version = RNS8_ABI_VERSION;
  sparse_desc.semantics = RNS8_FINITE_FIELD_U8;
  sparse_desc.bound_kind = RNS8_BOUND_NONE;
  sparse_desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  sparse_desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  sparse_desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  sparse_desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8;
  sparse_desc.rows = m;
  sparse_desc.expanded_k = k;
  sparse_desc.group_size = 4;
  sparse_desc.nonzeros_per_group = 2;
  sparse_desc.finite_modulus = 251;
  rns8_sparse_matrix* sparse_a = nullptr;
  REQUIRE(rns8_create_sparse_matrix(amdgpu, &sparse_desc, &sparse_a) == RNS8_SUCCESS);

  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(amdgpu, sparse_a, A.data(), k, 11) == RNS8_SUCCESS);
  const auto sparse_pack_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(has_timing_label(sparse_pack_events, "sparse_a_values_h2d"));
  CHECK(has_timing_label(sparse_pack_events, "sparse_a_indices_h2d"));

  REQUIRE(rns8_gemm_finite_u8(amdgpu, plan, 251, dense_a, dense_b, dense_c, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_u8_sparse_a(amdgpu, plan, 251, sparse_a, dense_b, sparse_c, workspace) == RNS8_SUCCESS);
  std::vector<uint8_t> dense_out(static_cast<std::size_t>(m * n), 0);
  std::vector<uint8_t> sparse_out(static_cast<std::size_t>(m * n), 0);
  REQUIRE(rns8_export_finite_u8(amdgpu, plan, 251, dense_c, dense_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_export_finite_u8(amdgpu, plan, 251, sparse_c, sparse_out.data(), n) == RNS8_SUCCESS);
  CHECK(sparse_out == dense_out);

  rns8_destroy_sparse_matrix(sparse_a);
  rns8_destroy_matrix(sparse_c);
  rns8_destroy_matrix(dense_c);
  rns8_destroy_matrix(dense_b);
  rns8_destroy_matrix(dense_a);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(amdgpu);
}

TEST_CASE("AMDGPU builtin explicit sparse-A exact-wide backend matches dense backend") {
  if (!amdgpu_builtins_available()) {
    SKIP("AMDGPU builtin backend is not available on this device");
  }

  rns8_context* cpu = require_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* amdgpu = require_context(RNS8_BACKEND_AMDGPU_BUILTINS);
  if (!sparse_runtime_target(context_target(amdgpu))) {
    rns8_destroy_context(amdgpu);
    rns8_destroy_context(cpu);
    SKIP("AMDGPU sparse-A builtin runtime requires CDNA3 SMFMAC or RDNA4 SWMMAC target");
  }

  constexpr int64_t m = 16;
  constexpr int64_t n = 16;
  constexpr int64_t k = 32;
  constexpr uint32_t limb_count = 3;
  constexpr int64_t limb_ld = n;

  auto run_signed = [&]() {
    std::vector<int64_t> A(static_cast<std::size_t>(m * k), 0);
    std::vector<int64_t> B(static_cast<std::size_t>(k * n), 0);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t group = 0; group < k / 4; ++group) {
        A[static_cast<std::size_t>(row * k + group * 4 + 0)] =
            static_cast<int64_t>((row * 5 + group * 7 + 3) % 17) - 8;
        A[static_cast<std::size_t>(row * k + group * 4 + 2)] =
            static_cast<int64_t>((row * 11 + group * 13 + 5) % 19) - 9;
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * n + col)] =
            static_cast<int64_t>((row * 17 + col * 23 + 7) % 23) - 11;
      }
    }

    auto cpu_a_desc = exact_wide_matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED);
    rns8_matrix* cpu_a = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &cpu_a_desc, &cpu_a) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(cpu, cpu_a, A.data(), k, 101) == RNS8_SUCCESS);

    auto gemm = exact_wide_desc(m, n, k, RNS8_EXACT_WIDE_SIGNED, RNS8_BACKEND_AMDGPU_BUILTINS);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    REQUIRE(rns8_create_plan(amdgpu, &gemm, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(amdgpu, plan, &workspace) == RNS8_SUCCESS);

    auto a_desc = exact_wide_matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED);
    auto b_desc = exact_wide_matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED);
    auto c_desc = exact_wide_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    rns8_matrix* dense_a = nullptr;
    rns8_matrix* dense_b = nullptr;
    rns8_matrix* dense_c = nullptr;
    rns8_matrix* sparse_c = nullptr;
    REQUIRE(rns8_create_matrix(amdgpu, &a_desc, &dense_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &b_desc, &dense_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &dense_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &sparse_c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(amdgpu, dense_a, A.data(), k, 101) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(amdgpu, dense_b, B.data(), n, 202) == RNS8_SUCCESS);

    auto sparse_desc = exact_wide_sparse_a_desc(m, k, RNS8_EXACT_WIDE_SIGNED);
    rns8_sparse_matrix* sparse_a = nullptr;
    REQUIRE(rns8_create_sparse_matrix(amdgpu, &sparse_desc, &sparse_a) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(
                amdgpu,
                sparse_a,
                reinterpret_cast<const uint8_t*>(cpu_a->residues.data()),
                cpu_a->desc.cols,
                cpu_a->source_version) == RNS8_SUCCESS);

    REQUIRE(rns8_gemm_rns(amdgpu, plan, dense_a, dense_b, dense_c, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns_sparse_a(amdgpu, plan, sparse_a, dense_b, sparse_c, workspace) == RNS8_SUCCESS);
    std::vector<uint64_t> dense_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    std::vector<uint64_t> sparse_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    REQUIRE(rns8_export_exact_wide_signed_limbs(amdgpu, plan, dense_c, dense_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_signed_limbs(amdgpu, plan, sparse_c, sparse_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    CHECK(sparse_limbs == dense_limbs);

    rns8_destroy_sparse_matrix(sparse_a);
    rns8_destroy_matrix(sparse_c);
    rns8_destroy_matrix(dense_c);
    rns8_destroy_matrix(dense_b);
    rns8_destroy_matrix(dense_a);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
    rns8_destroy_matrix(cpu_a);
  };

  auto run_unsigned = [&]() {
    std::vector<uint64_t> A(static_cast<std::size_t>(m * k), 0);
    std::vector<uint64_t> B(static_cast<std::size_t>(k * n), 0);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t group = 0; group < k / 4; ++group) {
        A[static_cast<std::size_t>(row * k + group * 4 + 0)] =
            static_cast<uint64_t>((row * 5 + group * 7 + 3) % 251 + 1);
        A[static_cast<std::size_t>(row * k + group * 4 + 2)] =
            static_cast<uint64_t>((row * 11 + group * 13 + 5) % 239 + 1);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * n + col)] =
            static_cast<uint64_t>((row * 17 + col * 23 + 7) % 251 + 1);
      }
    }

    auto cpu_a_desc = exact_wide_matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED);
    rns8_matrix* cpu_a = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &cpu_a_desc, &cpu_a) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), k, 301) == RNS8_SUCCESS);

    auto gemm = exact_wide_desc(m, n, k, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BACKEND_AMDGPU_BUILTINS);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    REQUIRE(rns8_create_plan(amdgpu, &gemm, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(amdgpu, plan, &workspace) == RNS8_SUCCESS);

    auto a_desc = exact_wide_matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED);
    auto b_desc = exact_wide_matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED);
    auto c_desc = exact_wide_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
    rns8_matrix* dense_a = nullptr;
    rns8_matrix* dense_b = nullptr;
    rns8_matrix* dense_c = nullptr;
    rns8_matrix* sparse_c = nullptr;
    REQUIRE(rns8_create_matrix(amdgpu, &a_desc, &dense_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &b_desc, &dense_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &dense_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(amdgpu, &c_desc, &sparse_c) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(amdgpu, dense_a, A.data(), k, 301) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(amdgpu, dense_b, B.data(), n, 402) == RNS8_SUCCESS);

    auto sparse_desc = exact_wide_sparse_a_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED);
    rns8_sparse_matrix* sparse_a = nullptr;
    REQUIRE(rns8_create_sparse_matrix(amdgpu, &sparse_desc, &sparse_a) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(
                amdgpu,
                sparse_a,
                reinterpret_cast<const uint8_t*>(cpu_a->residues.data()),
                cpu_a->desc.cols,
                cpu_a->source_version) == RNS8_SUCCESS);

    REQUIRE(rns8_gemm_rns(amdgpu, plan, dense_a, dense_b, dense_c, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns_sparse_a(amdgpu, plan, sparse_a, dense_b, sparse_c, workspace) == RNS8_SUCCESS);
    std::vector<uint64_t> dense_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    std::vector<uint64_t> sparse_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(amdgpu, plan, dense_c, dense_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(amdgpu, plan, sparse_c, sparse_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    CHECK(sparse_limbs == dense_limbs);

    rns8_destroy_sparse_matrix(sparse_a);
    rns8_destroy_matrix(sparse_c);
    rns8_destroy_matrix(dense_c);
    rns8_destroy_matrix(dense_b);
    rns8_destroy_matrix(dense_a);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
    rns8_destroy_matrix(cpu_a);
  };

  run_signed();
  run_unsigned();

  rns8_destroy_context(amdgpu);
  rns8_destroy_context(cpu);
}

#endif
