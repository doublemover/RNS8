#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <iterator>
#include <limits>
#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "core/internal.hpp"
#include "rns8/rns8.h"

namespace {

bool hip_available() {
  if (!rns8::detail::hip_direct_compiled()) {
    return false;
  }
  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  return rns8::detail::hip_direct_probe(0, info) == RNS8_SUCCESS;
}

rns8_context* create_context(rns8_backend_kind backend) {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = backend;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(backend == RNS8_BACKEND_HIP_DIRECT ? 0 : -1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8_gemm_desc signed_desc(int64_t m, int64_t n, int64_t k, uint64_t bound, rns8_backend_kind backend) {
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

rns8_gemm_desc unsigned_desc(int64_t m, int64_t n, int64_t k, uint64_t bound, rns8_backend_kind backend) {
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

rns8_matrix_desc matrix_desc(int64_t rows, int64_t cols, rns8_semantics semantics, rns8_bound_kind bound_kind) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind;
  desc.tile_m = 128;
  desc.tile_n = 128;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

}  // namespace

TEST_CASE("direct HIP ring GEMM matches CPU reference for one modulus") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP smoke");
  }

  const int64_t m = 2;
  const int64_t n = 3;
  const int64_t k = 4;
  const uint16_t modulus = 255;
  const std::vector<int8_t> A = {1, -2, 3, -4, -5, 6, -7, 8};
  const std::vector<int8_t> B = {9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20};
  std::vector<int8_t> cpu(static_cast<std::size_t>(m * n), 0);
  std::vector<int8_t> gpu(static_cast<std::size_t>(m * n), 0);

  rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, k, n, n, modulus);
  CHECK(rns8::detail::hip_direct_ring_gemm_i8(0, A.data(), B.data(), gpu.data(), m, n, k, k, n, n, modulus) ==
        RNS8_SUCCESS);
  CHECK(gpu == cpu);
}

TEST_CASE("direct HIP ring GEMM splits K above the int32 safe block") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP split smoke");
  }

  const int64_t m = 1;
  const int64_t n = 1;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  const uint16_t modulus = 251;
  std::vector<int8_t> A(static_cast<std::size_t>(k), 127);
  std::vector<int8_t> B(static_cast<std::size_t>(k), 127);
  std::vector<int8_t> cpu(1, 0);
  std::vector<int8_t> gpu(1, 0);

  rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, k, n, n, modulus);
  CHECK(rns8::detail::hip_direct_ring_gemm_i8(0, A.data(), B.data(), gpu.data(), m, n, k, k, n, n, modulus) ==
        RNS8_SUCCESS);
  CHECK(gpu == cpu);
}

TEST_CASE("direct HIP residue packing matches CPU reference for i64 and u64") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP pack smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    const int64_t rows = 3;
    const int64_t cols = 4;
    const int64_t ld = 5;
    const std::vector<int64_t> src = {
        0,
        1,
        -1,
        127,
        999,
        128,
        -128,
        -129,
        std::numeric_limits<int64_t>::max(),
        999,
        -std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::min(),
        251,
        -251,
        999};
    auto desc = matrix_desc(rows, cols, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    rns8_matrix* cpu_matrix = nullptr;
    rns8_matrix* hip_matrix = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &desc, &cpu_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &desc, &hip_matrix) == RNS8_SUCCESS);
    CHECK(rns8_pack_i64(cpu, cpu_matrix, src.data(), ld, 11) == RNS8_SUCCESS);
    CHECK(rns8_pack_i64(hip, hip_matrix, src.data(), ld, 11) == RNS8_SUCCESS);
    CHECK(hip_matrix->residues == cpu_matrix->residues);
    CHECK(hip_matrix->source_version == 11);
    rns8_destroy_matrix(hip_matrix);
    rns8_destroy_matrix(cpu_matrix);
  }

  {
    const int64_t rows = 2;
    const int64_t cols = 5;
    const int64_t ld = 6;
    const std::vector<uint64_t> src = {
        0,
        1,
        127,
        128,
        255,
        999,
        256,
        257,
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max() - 1,
        251,
        999};
    auto desc = matrix_desc(rows, cols, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    rns8_matrix* cpu_matrix = nullptr;
    rns8_matrix* hip_matrix = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &desc, &cpu_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &desc, &hip_matrix) == RNS8_SUCCESS);
    CHECK(rns8_pack_u64(cpu, cpu_matrix, src.data(), ld, 19) == RNS8_SUCCESS);
    CHECK(rns8_pack_u64(hip, hip_matrix, src.data(), ld, 19) == RNS8_SUCCESS);
    CHECK(hip_matrix->residues == cpu_matrix->residues);
    CHECK(hip_matrix->source_version == 19);
    rns8_destroy_matrix(hip_matrix);
    rns8_destroy_matrix(cpu_matrix);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP bounded oneshot matches CPU for signed and unsigned APIs") {
  if (!hip_available()) {
    SKIP("no HIP device available for public bounded GEMM smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    const int64_t m = 2;
    const int64_t n = 2;
    const int64_t k = 3;
    const int64_t A[] = {7, -3, 5, -11, 13, 17};
    const int64_t B[] = {2, -5, 19, 23, -29, 31};
    int64_t cpu_c[4] = {};
    int64_t hip_c[4] = {};
    auto cpu_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_HIP_DIRECT);
    CHECK(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A, k, B, n, cpu_c, n) == RNS8_SUCCESS);
    CHECK(rns8_gemm_i64_oneshot(hip, &hip_desc, A, k, B, n, hip_c, n) == RNS8_SUCCESS);
    CHECK(std::vector<int64_t>(std::begin(hip_c), std::end(hip_c)) ==
          std::vector<int64_t>(std::begin(cpu_c), std::end(cpu_c)));
  }

  {
    const int64_t m = 1;
    const int64_t n = 1;
    const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
    const uint64_t expected_bound = static_cast<uint64_t>(k) * 127u * 127u;
    std::vector<int64_t> A(static_cast<std::size_t>(k), 127);
    std::vector<int64_t> B(static_cast<std::size_t>(k), 127);
    int64_t cpu_c[1] = {};
    int64_t hip_c[1] = {};
    auto cpu_desc = signed_desc(m, n, k, expected_bound, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(m, n, k, expected_bound, RNS8_BACKEND_HIP_DIRECT);
    CHECK(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c, n) == RNS8_SUCCESS);
    CHECK(rns8_gemm_i64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_c, n) == RNS8_SUCCESS);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[0] == static_cast<int64_t>(expected_bound));
  }

  {
    const uint64_t A[] = {17, 3, 255, 9, 41, 5};
    const uint64_t B[] = {11, 7, 13, 19, 23, 29};
    uint64_t cpu_c[4] = {};
    uint64_t hip_c[4] = {};
    auto cpu_desc = unsigned_desc(2, 2, 3, 20000, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = unsigned_desc(2, 2, 3, 20000, RNS8_BACKEND_HIP_DIRECT);
    CHECK(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A, 3, B, 2, cpu_c, 2) == RNS8_SUCCESS);
    CHECK(rns8_gemm_u64_oneshot(hip, &hip_desc, A, 3, B, 2, hip_c, 2) == RNS8_SUCCESS);
    CHECK(std::vector<uint64_t>(std::begin(hip_c), std::end(hip_c)) ==
          std::vector<uint64_t>(std::begin(cpu_c), std::end(cpu_c)));
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}
