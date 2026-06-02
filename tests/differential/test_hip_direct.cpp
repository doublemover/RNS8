#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "backend_wrap64/wrap64_hip.hpp"
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

rns8_gemm_desc exact_unsigned_desc(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_EXACT_WIDE_UNSIGNED;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_gemm_desc wrap_desc(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
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
  if (semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED) {
    desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  } else if (semantics == RNS8_WRAP_U64_MOD_2_64) {
    desc.max_prefix = 0;
  } else {
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  }
  return desc;
}

void store_u64_limbs(std::vector<uint8_t>& dst, int64_t cell, uint64_t value) {
  for (uint32_t limb = 0; limb < 8; ++limb) {
    dst[static_cast<std::size_t>(cell * 8 + limb)] = static_cast<uint8_t>((value >> (8u * limb)) & 0xffu);
  }
}

uint64_t load_u64_limbs(const std::vector<uint8_t>& src, int64_t cell) {
  uint64_t value = 0;
  for (uint32_t limb = 0; limb < 8; ++limb) {
    value |= static_cast<uint64_t>(src[static_cast<std::size_t>(cell * 8 + limb)]) << (8u * limb);
  }
  return value;
}

bool has_timing_label(const std::vector<rns8::detail::hip_direct_timing_sample>& samples, const std::string& label) {
  for (const auto& sample : samples) {
    if (sample.label == label) {
      return true;
    }
  }
  return false;
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

TEST_CASE("direct HIP ring GEMM covers centered correction boundaries") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP centered-boundary smoke");
  }

  const int64_t m = 1;
  const int64_t n = 4;
  const int64_t k = 2;
  const uint16_t modulus = 255;
  const std::vector<int8_t> A = {1, 1};
  const std::vector<int8_t> B = {
      64,
      127,
      127,
      -64,
      64,
      0,
      127,
      -64};
  std::vector<int8_t> cpu(static_cast<std::size_t>(m * n), 0);
  std::vector<int8_t> gpu(static_cast<std::size_t>(m * n), 0);

  rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, k, n, n, modulus);
  REQUIRE(cpu[0] == -127);
  REQUIRE(cpu[1] == 127);
  REQUIRE(cpu[2] == -1);
  REQUIRE(cpu[3] == 127);
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

TEST_CASE("private HIP wrap64 byte-limb GEMM matches CPU reference") {
  if (!hip_available()) {
    SKIP("no HIP device available for private wrap64 HIP smoke");
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 3;
  constexpr int64_t k = 5;
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
  std::vector<uint8_t> a_limbs(static_cast<std::size_t>(m * k * 8));
  std::vector<uint8_t> b_limbs(static_cast<std::size_t>(k * n * 8));
  std::vector<uint8_t> c_limbs(static_cast<std::size_t>(m * n * 8), 0xff);
  for (int64_t cell = 0; cell < m * k; ++cell) {
    store_u64_limbs(a_limbs, cell, A[static_cast<std::size_t>(cell)]);
  }
  for (int64_t cell = 0; cell < k * n; ++cell) {
    store_u64_limbs(b_limbs, cell, B[static_cast<std::size_t>(cell)]);
  }

  REQUIRE(rns8::detail::wrap64_hip_gemm_byte_limbs(0, a_limbs.data(), b_limbs.data(), c_limbs.data(), m, n, k) ==
          RNS8_SUCCESS);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected = rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), k, B.data(), n, row, col, k);
      CHECK(load_u64_limbs(c_limbs, row * n + col) == expected);
    }
  }
}

TEST_CASE("direct HIP public wrap64 byte-limb path matches CPU reference") {
  if (!hip_available()) {
    SKIP("no HIP device available for public wrap64 HIP smoke");
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 3;
  constexpr int64_t k = 5;
  constexpr int64_t lda = 6;
  constexpr int64_t ldb = 4;
  constexpr int64_t ldc = 5;
  const std::vector<uint64_t> A = {
      0,
      1,
      std::numeric_limits<uint64_t>::max(),
      0x8080808080808080ull,
      0x0102030405060708ull,
      0xaaaaull,
      255,
      256,
      std::numeric_limits<uint64_t>::max() - 1,
      0x7f7f7f7f7f7f7f7full,
      17,
      0xbbbbull};
  const std::vector<uint64_t> B = {
      3,
      std::numeric_limits<uint64_t>::max(),
      0x1112131415161718ull,
      0xccccull,
      29,
      0x8080808080808080ull,
      31,
      0xddddull,
      0x0101010101010101ull,
      0xfefdfcfbfaf9f8f7ull,
      37,
      0xeeeeull,
      41,
      43,
      47,
      0xffffull,
      53,
      59,
      61,
      0x1234ull};
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), 0xdeadbeefdeadbeefull);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), 0xdeadbeefdeadbeefull);

  rns8_context* cpu = create_context(RNS8_BACKEND_WRAP64_BYTE_LIMB);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  auto cpu_desc = wrap_desc(m, n, k, RNS8_BACKEND_WRAP64_BYTE_LIMB);
  auto hip_desc = wrap_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_out = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_out = nullptr;

  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  CHECK(cpu_plan->prefix == 0);
  CHECK(hip_plan->prefix == 0);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(m, k, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(m, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_out) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_out) == RNS8_SUCCESS);
  REQUIRE(hip_a->hip_byte_limbs != nullptr);
  REQUIRE(hip_b->hip_byte_limbs != nullptr);
  REQUIRE(hip_out->hip_byte_limbs != nullptr);
  CHECK(hip_a->hip_residues == nullptr);
  CHECK(hip_b->hip_residues == nullptr);
  CHECK(hip_out->hip_residues == nullptr);
  void* hip_a_bytes = hip_a->hip_byte_limbs;
  void* hip_b_bytes = hip_b->hip_byte_limbs;
  void* hip_out_bytes = hip_out->hip_byte_limbs;

  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), lda, 7) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), ldb, 11) == RNS8_SUCCESS);
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), lda, 7) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), ldb, 11) == RNS8_SUCCESS);
  auto hip_pack_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(has_timing_label(hip_pack_events, "pack_h2d"));
  CHECK(has_timing_label(hip_pack_events, "pack_kernel"));
  CHECK(hip_a->hip_byte_limbs == hip_a_bytes);
  CHECK(hip_b->hip_byte_limbs == hip_b_bytes);
  CHECK(hip_a->device_byte_limbs_current);
  CHECK(hip_b->device_byte_limbs_current);
  CHECK_FALSE(hip_a->host_byte_limbs_current);
  CHECK_FALSE(hip_b->host_byte_limbs_current);

  CHECK(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_gemm_wrap_u64(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_gemm_wrap_u64(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
  auto hip_gemm_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(has_timing_label(hip_gemm_events, "wrap64_comba_gemm_kernel"));
  CHECK(hip_out->hip_byte_limbs == hip_out_bytes);
  CHECK(hip_out->device_byte_limbs_current);
  CHECK_FALSE(hip_out->host_byte_limbs_current);

  CHECK(rns8_export_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_export_wrap_u64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  auto hip_export_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);
  CHECK(has_timing_label(hip_export_events, "wrap64_export_kernel"));
  CHECK(has_timing_label(hip_export_events, "wrap64_export_d2h"));
  CHECK(hip_out->hip_export_buffer != nullptr);
  CHECK_FALSE(hip_out->host_byte_limbs_current);
  CHECK(hip_c == cpu_c);

  std::vector<uint64_t> cpu_oneshot(static_cast<std::size_t>(m * ldc), 0xababababababababull);
  std::vector<uint64_t> hip_oneshot(static_cast<std::size_t>(m * ldc), 0xababababababababull);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_oneshot.data(), ldc) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(hip, &hip_desc, A.data(), lda, B.data(), ldb, hip_oneshot.data(), ldc) ==
          RNS8_SUCCESS);
  CHECK(hip_oneshot == cpu_oneshot);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected = rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == expected);
    }
    CHECK(hip_c[static_cast<std::size_t>(row * ldc + n)] == 0xdeadbeefdeadbeefull);
    CHECK(hip_oneshot[static_cast<std::size_t>(row * ldc + n)] == 0xababababababababull);
  }

  rns8_destroy_matrix(hip_out);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_matrix(cpu_out);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_workspace(hip_workspace);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
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
    CHECK(hip_matrix->hip_residues != nullptr);
    CHECK(hip_matrix->device_residues_current);
    CHECK_FALSE(hip_matrix->host_residues_current);
    CHECK(rns8::detail::hip_direct_copy_device_to_host(
              hip_matrix->hip_device_id,
              hip_matrix->residues.data(),
              hip_matrix->hip_residues,
              hip_matrix->hip_residue_bytes) == RNS8_SUCCESS);
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
    CHECK(hip_matrix->hip_residues != nullptr);
    CHECK(hip_matrix->device_residues_current);
    CHECK_FALSE(hip_matrix->host_residues_current);
    CHECK(rns8::detail::hip_direct_copy_device_to_host(
              hip_matrix->hip_device_id,
              hip_matrix->residues.data(),
              hip_matrix->hip_residues,
              hip_matrix->hip_residue_bytes) == RNS8_SUCCESS);
    CHECK(hip_matrix->residues == cpu_matrix->residues);
    CHECK(hip_matrix->source_version == 19);
    rns8_destroy_matrix(hip_matrix);
    rns8_destroy_matrix(cpu_matrix);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP persistent RNS matrices keep device storage through GEMM") {
  if (!hip_available()) {
    SKIP("no HIP device available for persistent direct HIP smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  const int64_t m = 2;
  const int64_t n = 3;
  const int64_t k = 4;
  const int64_t A[] = {5, -7, 11, 13, -17, 19, 23, -29};
  const int64_t B[] = {3, -5, 7, 11, 13, -17, 19, 23, -29, 31, 37, -41};
  int64_t cpu_c[6] = {};
  int64_t hip_c[6] = {};

  auto cpu_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_HIP_DIRECT);
  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A, k, B, n, cpu_c, n) == RNS8_SUCCESS);

  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(hip, &hip_desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

  REQUIRE(a_matrix->hip_residues != nullptr);
  REQUIRE(b_matrix->hip_residues != nullptr);
  REQUIRE(c_matrix->hip_residues != nullptr);
  void* a_device_residues = a_matrix->hip_residues;
  void* b_device_residues = b_matrix->hip_residues;
  void* c_device_residues = c_matrix->hip_residues;

  REQUIRE(rns8_pack_i64(hip, a_matrix, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, b_matrix, B, n, 2) == RNS8_SUCCESS);
  CHECK(a_matrix->hip_residues == a_device_residues);
  CHECK(b_matrix->hip_residues == b_device_residues);
  CHECK(a_matrix->device_residues_current);
  CHECK(b_matrix->device_residues_current);

  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  CHECK(c_matrix->hip_residues == c_device_residues);
  CHECK(c_matrix->device_residues_current);
  CHECK_FALSE(c_matrix->host_residues_current);

  REQUIRE(rns8_export_i64(hip, plan, c_matrix, hip_c, n) == RNS8_SUCCESS);
  CHECK_FALSE(c_matrix->host_residues_current);
  CHECK(c_matrix->hip_export_buffer != nullptr);
  CHECK(c_matrix->hip_status_buffer != nullptr);
  CHECK(std::vector<int64_t>(std::begin(hip_c), std::end(hip_c)) ==
        std::vector<int64_t>(std::begin(cpu_c), std::end(cpu_c)));

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide RNS output matches CPU residues") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide direct HIP smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  const int64_t m = 1;
  const int64_t n = 2;
  const int64_t k = 2;
  const int64_t A[] = {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max() - 19};
  const int64_t B[] = {
      std::numeric_limits<int64_t>::max() - 3,
      -std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max() - 7,
      std::numeric_limits<int64_t>::max() - 11};

  rns8_plan* cpu_plan = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  auto cpu_desc = exact_signed_desc(m, n, k, RNS8_BACKEND_CPU_REFERENCE);
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(cpu, cpu_a, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(cpu, cpu_b, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);

  rns8_plan* hip_plan = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  auto hip_desc = exact_signed_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, hip_a, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, hip_b, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
              hip_c->hip_device_id, hip_c->residues.data(), hip_c->hip_residues, hip_c->hip_residue_bytes) ==
          RNS8_SUCCESS);
  CHECK(hip_c->residues == cpu_c->residues);
  constexpr uint32_t limb_count = 3;
  constexpr int64_t limb_ld = 3;
  std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), 0xaaaaaaaaaaaaaaaaull);
  REQUIRE(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  hip_c->host_residues_current = false;
  REQUIRE(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  CHECK_FALSE(hip_c->host_residues_current);
  CHECK(hip_c->hip_export_buffer != nullptr);
  CHECK(hip_c->hip_status_buffer != nullptr);
  CHECK(hip_limbs == cpu_limbs);
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 2) * limb_count)] == 0xaaaaaaaaaaaaaaaaull);
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 1) * limb_count + 2)] == UINT64_MAX);
  std::vector<uint64_t> too_few_cpu(static_cast<std::size_t>(m * n), 0);
  std::vector<uint64_t> too_few_hip(static_cast<std::size_t>(m * n), 0);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, too_few_cpu.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, too_few_hip.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK_FALSE(hip_c->host_residues_current);

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_workspace(hip_workspace);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide unsigned RNS output matches CPU residues") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide unsigned direct HIP smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  const int64_t m = 1;
  const int64_t n = 2;
  const int64_t k = 2;
  const uint64_t A[] = {std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max() - 19};
  const uint64_t B[] = {
      std::numeric_limits<uint64_t>::max() - 3,
      std::numeric_limits<uint64_t>::max() - 5,
      0x8080808080808080ull,
      0x0102030405060708ull};

  rns8_plan* cpu_plan = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  auto cpu_desc = exact_unsigned_desc(m, n, k, RNS8_BACKEND_CPU_REFERENCE);
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_a, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);

  rns8_plan* hip_plan = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  auto hip_desc = exact_unsigned_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B, n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
              hip_c->hip_device_id, hip_c->residues.data(), hip_c->hip_residues, hip_c->hip_residue_bytes) ==
          RNS8_SUCCESS);
  CHECK(hip_c->residues == cpu_c->residues);
  constexpr uint32_t limb_count = 3;
  constexpr int64_t limb_ld = 3;
  std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), 0xbbbbbbbbbbbbbbbbull);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  hip_c->host_residues_current = false;
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  CHECK_FALSE(hip_c->host_residues_current);
  CHECK(hip_c->hip_export_buffer != nullptr);
  CHECK(hip_c->hip_status_buffer != nullptr);
  CHECK(hip_limbs == cpu_limbs);
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 2) * limb_count)] == 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> too_few_cpu(static_cast<std::size_t>(m * n), 0);
  std::vector<uint64_t> too_few_hip(static_cast<std::size_t>(m * n), 0);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, too_few_cpu.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, too_few_hip.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK_FALSE(hip_c->host_residues_current);

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_workspace(hip_workspace);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_plan(cpu_plan);
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
    const int64_t A[] = {std::numeric_limits<int64_t>::min()};
    const int64_t B[] = {1};
    int64_t cpu_c[1] = {};
    int64_t hip_c[1] = {};
    auto cpu_desc = signed_desc(1, 1, 1, 1ull << 63u, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(1, 1, 1, 1ull << 63u, RNS8_BACKEND_HIP_DIRECT);
    cpu_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    hip_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    CHECK(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A, 1, B, 1, cpu_c, 1) == RNS8_SUCCESS);
    CHECK(rns8_gemm_i64_oneshot(hip, &hip_desc, A, 1, B, 1, hip_c, 1) == RNS8_SUCCESS);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[0] == std::numeric_limits<int64_t>::min());
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

  {
    const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
    const uint64_t B[] = {1};
    uint64_t cpu_c[1] = {};
    uint64_t hip_c[1] = {};
    auto cpu_desc = unsigned_desc(1, 1, 1, std::numeric_limits<uint64_t>::max(), RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = unsigned_desc(1, 1, 1, std::numeric_limits<uint64_t>::max(), RNS8_BACKEND_HIP_DIRECT);
    cpu_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    hip_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    CHECK(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A, 1, B, 1, cpu_c, 1) == RNS8_SUCCESS);
    CHECK(rns8_gemm_u64_oneshot(hip, &hip_desc, A, 1, B, 1, hip_c, 1) == RNS8_SUCCESS);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[0] == std::numeric_limits<uint64_t>::max());
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}
