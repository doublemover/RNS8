#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <vector>

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
  REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(amdgpu, sparse_a, A.data(), k, 11) == RNS8_SUCCESS);

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

#endif
