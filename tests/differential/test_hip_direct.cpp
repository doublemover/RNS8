#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
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

rns8_gemm_desc per_tile_signed_desc(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<uint64_t>& bounds,
    rns8_backend_kind backend) {
  rns8_gemm_desc desc = signed_desc(m, n, k, 0, backend);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_ABS;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  return desc;
}

rns8_gemm_desc per_tile_unsigned_desc(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<uint64_t>& bounds,
    rns8_backend_kind backend) {
  rns8_gemm_desc desc = unsigned_desc(m, n, k, 0, backend);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
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

void fill_wrap64_carry_heavy_inputs(
    std::vector<uint64_t>& A,
    int64_t m,
    int64_t k,
    int64_t lda,
    std::vector<uint64_t>& B,
    int64_t n,
    int64_t ldb) {
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] =
          row % 3 == 0 ? std::numeric_limits<uint64_t>::max()
                       : row % 3 == 1 ? 0x8080808080808080ull : 0xfefdfcfbfaf9f8f7ull;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] =
          col % 3 == 0 ? std::numeric_limits<uint64_t>::max()
                       : col % 3 == 1 ? 0x7f807f807f807f80ull : 0x0102030405060708ull;
    }
  }
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

TEST_CASE("private HIP wrap64 byte-limb GEMM matches CPU reference for carry-heavy tiled data") {
  if (!hip_available()) {
    SKIP("no HIP device available for private wrap64 HIP carry smoke");
  }

  constexpr int64_t m = 17;
  constexpr int64_t n = 17;
  constexpr int64_t k = 19;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * n), 0);
  fill_wrap64_carry_heavy_inputs(A, m, k, k, B, n, n);

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
      const uint64_t expected = rns8::detail::wrap64_byte_gemm36_cell(A.data(), k, B.data(), n, row, col, k);
      CHECK(load_u64_limbs(c_limbs, row * n + col) == expected);
    }
  }
}

TEST_CASE("private HIP wrap64 helpers reject invalid contracts before launch") {
  std::vector<uint8_t> limbs(8, 0);
  uint64_t src = 0;
  uint64_t dst = 0;
  void* buffer = nullptr;
  std::size_t bytes = 0;
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs(0, nullptr, limbs.data(), limbs.data(), 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs(0, limbs.data(), limbs.data(), limbs.data(), 0, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, limbs.data(), limbs.data(), limbs.data(), 1, 0, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(
            0,
            &src,
            &buffer,
            &bytes,
            limbs.data(),
            2,
            1,
            std::numeric_limits<int64_t>::max()) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(
            0,
            limbs.data(),
            &buffer,
            &bytes,
            2,
            1,
            &dst,
            std::numeric_limits<int64_t>::max()) == RNS8_INVALID_ARGUMENT);
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
  a_desc.logical_ld = lda;
  b_desc.logical_ld = ldb;
  c_desc.logical_ld = ldc;
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_out) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_out) == RNS8_SUCCESS);
  REQUIRE(hip_a->hip_byte_limbs != nullptr);
  REQUIRE(hip_b->hip_byte_limbs != nullptr);
  REQUIRE(hip_out->hip_byte_limbs != nullptr);
  CHECK(hip_a->hip_byte_limb_bytes == static_cast<std::size_t>(m * k * 8));
  CHECK(hip_b->hip_byte_limb_bytes == static_cast<std::size_t>(k * n * 8));
  CHECK(hip_out->hip_byte_limb_bytes == static_cast<std::size_t>(m * n * 8));
  CHECK(hip_a->desc.logical_ld == lda);
  CHECK(hip_b->desc.logical_ld == ldb);
  CHECK(hip_out->desc.logical_ld == ldc);
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
  CHECK(has_timing_label(hip_gemm_events, "wrap64_tiled_byte_gemm_kernel"));
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
  void* hip_export = hip_out->hip_export_buffer;
  const std::size_t hip_export_bytes = hip_out->hip_export_bytes;
  CHECK_FALSE(hip_out->host_byte_limbs_current);
  CHECK(hip_c == cpu_c);

  std::fill(hip_c.begin(), hip_c.end(), 0xfeedfeedfeedfeedull);
  REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  CHECK(hip_out->hip_export_buffer == hip_export);
  CHECK(hip_out->hip_export_bytes == hip_export_bytes);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == cpu_c[static_cast<std::size_t>(row * ldc + col)]);
    }
    for (int64_t col = n; col < ldc; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == 0xfeedfeedfeedfeedull);
    }
  }

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
      const uint64_t decomposition =
          rns8::detail::wrap64_byte_gemm36_cell(A.data(), lda, B.data(), ldb, row, col, k);
      CHECK(decomposition == expected);
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == expected);
    }
    CHECK(hip_c[static_cast<std::size_t>(row * ldc + n)] == 0xfeedfeedfeedfeedull);
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

TEST_CASE("direct HIP public wrap64 tiled byte-limb path matches CPU for random padded layouts") {
  if (!hip_available()) {
    SKIP("no HIP device available for public wrap64 HIP tiled smoke");
  }

  constexpr int64_t m = 19;
  constexpr int64_t n = 18;
  constexpr int64_t k = 33;
  constexpr int64_t lda = 37;
  constexpr int64_t ldb = 23;
  constexpr int64_t ldc = 21;
  constexpr uint64_t c_sentinel = 0x5a5a5a5a5a5a5a5aull;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> cpu_oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::mt19937_64 rng(0x726e73385f777261ull);

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
  A[static_cast<std::size_t>((m / 2) * lda + (k / 2))] = 0x0102030405060708ull;
  A[static_cast<std::size_t>((m - 1) * lda + (k - 1))] = std::numeric_limits<uint64_t>::max() - 1;
  B[0] = std::numeric_limits<uint64_t>::max();
  B[1] = 1;
  B[2] = 0xfefdfcfbfaf9f8f7ull;
  B[static_cast<std::size_t>((k / 2) * ldb + (n / 2))] = 0x7f807f807f807f80ull;
  B[static_cast<std::size_t>((k - 1) * ldb + (n - 1))] = std::numeric_limits<uint64_t>::max();

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
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(m, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  a_desc.logical_ld = lda;
  b_desc.logical_ld = ldb;
  c_desc.logical_ld = ldc;
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_out) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_out) == RNS8_SUCCESS);
  REQUIRE(hip_a->hip_byte_limbs != nullptr);
  REQUIRE(hip_b->hip_byte_limbs != nullptr);
  REQUIRE(hip_out->hip_byte_limbs != nullptr);
  CHECK(hip_a->hip_byte_limb_bytes == static_cast<std::size_t>(m * k * 8));
  CHECK(hip_b->hip_byte_limb_bytes == static_cast<std::size_t>(k * n * 8));
  CHECK(hip_out->hip_byte_limb_bytes == static_cast<std::size_t>(m * n * 8));
  CHECK(hip_a->desc.logical_ld == lda);
  CHECK(hip_b->desc.logical_ld == ldb);
  CHECK(hip_out->desc.logical_ld == ldc);
  CHECK(hip_a->hip_residues == nullptr);
  CHECK(hip_b->hip_residues == nullptr);
  CHECK(hip_out->hip_residues == nullptr);
  void* hip_a_bytes = hip_a->hip_byte_limbs;
  void* hip_b_bytes = hip_b->hip_byte_limbs;
  void* hip_out_bytes = hip_out->hip_byte_limbs;

  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), lda, 101) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), ldb, 102) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), lda, 101) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), ldb, 102) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  void* hip_export = hip_out->hip_export_buffer;
  const std::size_t hip_export_bytes = hip_out->hip_export_bytes;

  REQUIRE(rns8_gemm_wrap_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_oneshot.data(), ldc) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(hip, &hip_desc, A.data(), lda, B.data(), ldb, hip_oneshot.data(), ldc) ==
          RNS8_SUCCESS);
  CHECK(hip_c == cpu_c);
  CHECK(hip_oneshot == cpu_oneshot);
  CHECK(hip_a->hip_byte_limbs == hip_a_bytes);
  CHECK(hip_b->hip_byte_limbs == hip_b_bytes);
  CHECK(hip_out->hip_byte_limbs == hip_out_bytes);

  std::fill(cpu_c.begin(), cpu_c.end(), c_sentinel);
  std::fill(hip_c.begin(), hip_c.end(), c_sentinel);
  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), lda, 201) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), ldb, 202) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), lda, 201) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), ldb, 202) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  CHECK(hip_c == cpu_c);
  CHECK(hip_a->hip_byte_limbs == hip_a_bytes);
  CHECK(hip_b->hip_byte_limbs == hip_b_bytes);
  CHECK(hip_out->hip_byte_limbs == hip_out_bytes);
  CHECK(hip_out->hip_export_buffer == hip_export);
  CHECK(hip_out->hip_export_bytes == hip_export_bytes);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected = rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      const std::size_t out_index = static_cast<std::size_t>(row * ldc + col);
      CHECK(cpu_c[out_index] == expected);
      CHECK(hip_c[out_index] == expected);
      CHECK(hip_oneshot[out_index] == expected);
    }
    for (int64_t col = n; col < ldc; ++col) {
      const std::size_t out_index = static_cast<std::size_t>(row * ldc + col);
      CHECK(hip_c[out_index] == c_sentinel);
      CHECK(hip_oneshot[out_index] == c_sentinel);
    }
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

TEST_CASE("direct HIP public wrap64 path matches CPU for carry-heavy tiled export") {
  if (!hip_available()) {
    SKIP("no HIP device available for public wrap64 HIP carry smoke");
  }

  constexpr int64_t m = 17;
  constexpr int64_t n = 17;
  constexpr int64_t k = 19;
  constexpr int64_t lda = 23;
  constexpr int64_t ldb = 21;
  constexpr int64_t ldc = 20;
  constexpr uint64_t c_sentinel = 0x6464646464646464ull;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> cpu_oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  fill_wrap64_carry_heavy_inputs(A, m, k, lda, B, n, ldb);

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
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(m, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  a_desc.logical_ld = lda;
  b_desc.logical_ld = ldb;
  c_desc.logical_ld = ldc;
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_out) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_out) == RNS8_SUCCESS);
  CHECK(hip_a->hip_byte_limb_bytes == static_cast<std::size_t>(m * k * 8));
  CHECK(hip_b->hip_byte_limb_bytes == static_cast<std::size_t>(k * n * 8));
  CHECK(hip_out->hip_byte_limb_bytes == static_cast<std::size_t>(m * n * 8));
  CHECK(hip_a->hip_residues == nullptr);
  CHECK(hip_b->hip_residues == nullptr);
  CHECK(hip_out->hip_residues == nullptr);

  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), lda, 301) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), ldb, 302) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), lda, 301) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), ldb, 302) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_oneshot.data(), ldc) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(hip, &hip_desc, A.data(), lda, B.data(), ldb, hip_oneshot.data(), ldc) ==
          RNS8_SUCCESS);
  CHECK(hip_c == cpu_c);
  CHECK(hip_oneshot == cpu_oneshot);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected = rns8::detail::wrap64_byte_gemm36_cell(A.data(), lda, B.data(), ldb, row, col, k);
      const std::size_t out_index = static_cast<std::size_t>(row * ldc + col);
      CHECK(hip_c[out_index] == expected);
      CHECK(hip_oneshot[out_index] == expected);
    }
    for (int64_t col = n; col < ldc; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
      CHECK(hip_oneshot[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
    }
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

TEST_CASE("direct HIP wrap64 rejects CRT-style descriptors") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP wrap64 descriptor rejection smoke");
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 1;
  const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
  const uint64_t B[] = {2};
  uint64_t C[] = {0};
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto desc = wrap_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);

  auto bounded_looking = desc;
  bounded_looking.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  bounded_looking.bound = std::numeric_limits<uint64_t>::max();
  CHECK(rns8_gemm_wrap_u64_oneshot(hip, &bounded_looking, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(hip, &bounded_looking, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto prefixed = desc;
  prefixed.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  CHECK(rns8_gemm_wrap_u64_oneshot(hip, &prefixed, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_create_plan(hip, &prefixed, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto matrix = matrix_desc(m, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  rns8_matrix* storage = nullptr;
  matrix.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  CHECK(rns8_create_matrix(hip, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
  CHECK(storage == nullptr);
  matrix.bound_kind = RNS8_BOUND_NONE;
  matrix.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  CHECK(rns8_create_matrix(hip, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
  CHECK(storage == nullptr);

  rns8_plan* valid_plan = nullptr;
  REQUIRE(rns8_create_plan(hip, &desc, &valid_plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(hip, valid_plan, &workspace) == RNS8_SUCCESS);
  rns8_matrix* wrap_a = nullptr;
  rns8_matrix* wrap_b = nullptr;
  rns8_matrix* wrap_c = nullptr;
  auto a_desc = matrix_desc(m, k, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(m, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &wrap_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &wrap_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &wrap_c) == RNS8_SUCCESS);
  const int64_t signed_A[] = {-1};
  uint64_t limbs[] = {0};
  CHECK(rns8_pack_i64(hip, wrap_a, signed_A, k, 0) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, valid_plan, wrap_c, limbs, n, 1) == RNS8_INVALID_ARGUMENT);

  rns8_matrix* residue_a = nullptr;
  rns8_matrix* residue_b = nullptr;
  rns8_matrix* residue_c = nullptr;
  auto residue_a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto residue_b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto residue_c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  REQUIRE(rns8_create_matrix(hip, &residue_a_desc, &residue_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &residue_b_desc, &residue_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &residue_c_desc, &residue_c) == RNS8_SUCCESS);
  CHECK(residue_a->hip_residues != nullptr);
  CHECK(residue_a->hip_byte_limbs == nullptr);
  CHECK(rns8_gemm_wrap_u64(hip, valid_plan, residue_a, residue_b, residue_c, workspace) == RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(residue_c);
  rns8_destroy_matrix(residue_b);
  rns8_destroy_matrix(residue_a);
  rns8_destroy_matrix(wrap_c);
  rns8_destroy_matrix(wrap_b);
  rns8_destroy_matrix(wrap_a);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(valid_plan);

  rns8_destroy_context(hip);
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

  rns8::detail::hip_direct_allocation_counters_reset();
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

  void* a_upload = a_matrix->hip_upload_buffer;
  void* b_upload = b_matrix->hip_upload_buffer;
  void* c_export = c_matrix->hip_export_buffer;
  void* c_status = c_matrix->hip_status_buffer;
  const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  REQUIRE(warmed_allocations.allocate_calls > 0);
  REQUIRE(warmed_allocations.allocated_bytes > 0);
  std::fill(std::begin(hip_c), std::end(hip_c), int64_t{0});

  REQUIRE(rns8_pack_i64(hip, a_matrix, A, k, 3) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, b_matrix, B, n, 4) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(hip, plan, c_matrix, hip_c, n) == RNS8_SUCCESS);
  const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
  CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
  CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
  CHECK(a_matrix->hip_residues == a_device_residues);
  CHECK(b_matrix->hip_residues == b_device_residues);
  CHECK(c_matrix->hip_residues == c_device_residues);
  CHECK(a_matrix->hip_upload_buffer == a_upload);
  CHECK(b_matrix->hip_upload_buffer == b_upload);
  CHECK(c_matrix->hip_export_buffer == c_export);
  CHECK(c_matrix->hip_status_buffer == c_status);
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

TEST_CASE("direct HIP persistent per-tile bounded u64 reuses resident storage across same-shape calls") {
  if (!hip_available()) {
    SKIP("no HIP device available for persistent per-tile direct HIP bounded smoke");
  }

  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 2;
  constexpr int64_t lda = k + 1;
  constexpr int64_t ldb = n + 2;
  constexpr int64_t ldc = n + 3;
  constexpr uint64_t sentinel = 0xbad0bad0bad0bad0ull;
  const std::vector<uint64_t> bounds = {64, 4096, 8000000, 2000000000};

  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), sentinel);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), sentinel);
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);

  auto fill_inputs = [&](int variant) {
    std::fill(A.begin(), A.end(), sentinel);
    std::fill(B.begin(), B.end(), sentinel);
    for (int64_t row = 0; row < m; ++row) {
      A[static_cast<std::size_t>(row * lda)] = row < 64 ? (variant == 0 ? 1 : 3) : (variant == 0 ? 1000000 : 500000);
      A[static_cast<std::size_t>(row * lda + 1)] = row < 64 ? (variant == 0 ? 2 : 4) : (variant == 0 ? 3 : 5);
    }
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(col)] = col < 64 ? (variant == 0 ? 7 : 5) : (variant == 0 ? 1000 : 999);
      B[static_cast<std::size_t>(ldb + col)] = col < 64 ? (variant == 0 ? 11 : 6) : (variant == 0 ? 13 : 7);
    }
  };

  auto compare_outputs = [&]() {
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_c[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t pad = n; pad < ldc; ++pad) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + pad)] == sentinel);
        CHECK(cpu_c[static_cast<std::size_t>(row * ldc + pad)] == sentinel);
      }
    }
  };

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);

  rns8::detail::hip_direct_allocation_counters_reset();
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(hip, &hip_desc, &plan) == RNS8_SUCCESS);
  REQUIRE(plan->prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.tile_count == bounds.size());
  CHECK(info.adaptive_prefix_active == 1);
  CHECK(info.adaptive_skip_active == 1);
  CHECK(info.min_selected_prefix >= 1);
  CHECK(info.min_selected_prefix < info.max_selected_prefix);
  CHECK(info.max_selected_prefix < plan->prefix);

  REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

  void* a_device_residues = a_matrix->hip_residues;
  void* b_device_residues = b_matrix->hip_residues;
  void* c_device_residues = c_matrix->hip_residues;
  REQUIRE(a_device_residues != nullptr);
  REQUIRE(b_device_residues != nullptr);
  REQUIRE(c_device_residues != nullptr);

  fill_inputs(0);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, a_matrix, A.data(), lda, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, b_matrix, B.data(), ldb, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs();

  void* a_upload = a_matrix->hip_upload_buffer;
  void* b_upload = b_matrix->hip_upload_buffer;
  void* c_export = c_matrix->hip_export_buffer;
  void* c_status = c_matrix->hip_status_buffer;
  const std::size_t a_upload_bytes = a_matrix->hip_upload_bytes;
  const std::size_t b_upload_bytes = b_matrix->hip_upload_bytes;
  const std::size_t c_export_bytes = c_matrix->hip_export_bytes;
  const std::size_t c_status_bytes = c_matrix->hip_status_bytes;
  const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  REQUIRE(a_upload != nullptr);
  REQUIRE(b_upload != nullptr);
  REQUIRE(c_export != nullptr);
  REQUIRE(c_status != nullptr);
  REQUIRE(warmed_allocations.allocate_calls > 0);
  REQUIRE(warmed_allocations.allocated_bytes > 0);

  fill_inputs(1);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, a_matrix, A.data(), lda, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, b_matrix, B.data(), ldb, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs();

  const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
  CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
  CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
  CHECK(a_matrix->hip_residues == a_device_residues);
  CHECK(b_matrix->hip_residues == b_device_residues);
  CHECK(c_matrix->hip_residues == c_device_residues);
  CHECK(a_matrix->hip_upload_buffer == a_upload);
  CHECK(b_matrix->hip_upload_buffer == b_upload);
  CHECK(c_matrix->hip_export_buffer == c_export);
  CHECK(c_matrix->hip_status_buffer == c_status);
  CHECK(a_matrix->hip_upload_bytes == a_upload_bytes);
  CHECK(b_matrix->hip_upload_bytes == b_upload_bytes);
  CHECK(c_matrix->hip_export_bytes == c_export_bytes);
  CHECK(c_matrix->hip_status_bytes == c_status_bytes);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP persistent bounded i64 K-split reuses resident storage with padded layouts") {
  if (!hip_available()) {
    SKIP("no HIP device available for persistent direct HIP bounded K-split smoke");
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  const int64_t lda = k + 3;
  constexpr int64_t ldb = n + 2;
  constexpr int64_t ldc = n + 2;
  constexpr int64_t sentinel = INT64_C(-0x123456789abc);
  const uint64_t bound = static_cast<uint64_t>(k) * 127u * 127u;

  std::vector<int64_t> A(static_cast<std::size_t>(m * lda), sentinel);
  std::vector<int64_t> B(static_cast<std::size_t>(k * ldb), sentinel);
  std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<int64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);

  auto fill_inputs = [&](int variant) {
    std::fill(A.begin(), A.end(), sentinel);
    std::fill(B.begin(), B.end(), sentinel);
    for (int64_t kk = 0; kk < k; ++kk) {
      A[static_cast<std::size_t>(kk)] = variant == 0 ? 127 : (kk % 2 == 0 ? 127 : -127);
      B[static_cast<std::size_t>(kk * ldb)] = 127;
      B[static_cast<std::size_t>(kk * ldb + 1)] = -127;
    }
  };

  auto compare_outputs = [&](int64_t expected_col0, int64_t expected_col1) {
    CHECK(cpu_c[0] == expected_col0);
    CHECK(cpu_c[1] == expected_col1);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[1] == cpu_c[1]);
    CHECK(cpu_c[2] == sentinel);
    CHECK(cpu_c[3] == sentinel);
    CHECK(hip_c[2] == sentinel);
    CHECK(hip_c[3] == sentinel);
  };

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = signed_desc(m, n, k, bound, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = signed_desc(m, n, k, bound, RNS8_BACKEND_HIP_DIRECT);

  rns8::detail::hip_direct_allocation_counters_reset();
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  REQUIRE(rns8_create_plan(hip, &hip_desc, &plan) == RNS8_SUCCESS);
  REQUIRE(plan->prefix == RNS8_DEFAULT_BOUNDED_PREFIX);

  rns8_plan_schedule_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &info) == RNS8_SUCCESS);
  CHECK(info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  CHECK(info.adaptive_prefix_active == 0);

  REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

  void* a_device_residues = a_matrix->hip_residues;
  void* b_device_residues = b_matrix->hip_residues;
  void* c_device_residues = c_matrix->hip_residues;
  REQUIRE(a_device_residues != nullptr);
  REQUIRE(b_device_residues != nullptr);
  REQUIRE(c_device_residues != nullptr);

  fill_inputs(0);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, a_matrix, A.data(), lda, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, b_matrix, B.data(), ldb, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs(static_cast<int64_t>(bound), -static_cast<int64_t>(bound));

  void* a_upload = a_matrix->hip_upload_buffer;
  void* b_upload = b_matrix->hip_upload_buffer;
  void* c_export = c_matrix->hip_export_buffer;
  void* c_status = c_matrix->hip_status_buffer;
  const std::size_t a_upload_bytes = a_matrix->hip_upload_bytes;
  const std::size_t b_upload_bytes = b_matrix->hip_upload_bytes;
  const std::size_t c_export_bytes = c_matrix->hip_export_bytes;
  const std::size_t c_status_bytes = c_matrix->hip_status_bytes;
  const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  REQUIRE(a_upload != nullptr);
  REQUIRE(b_upload != nullptr);
  REQUIRE(c_export != nullptr);
  REQUIRE(c_status != nullptr);
  REQUIRE(warmed_allocations.allocate_calls > 0);
  REQUIRE(warmed_allocations.allocated_bytes > 0);

  fill_inputs(1);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, a_matrix, A.data(), lda, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(hip, b_matrix, B.data(), ldb, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs(127 * 127, -(127 * 127));

  const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
  CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
  CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
  CHECK(a_matrix->hip_residues == a_device_residues);
  CHECK(b_matrix->hip_residues == b_device_residues);
  CHECK(c_matrix->hip_residues == c_device_residues);
  CHECK(a_matrix->hip_upload_buffer == a_upload);
  CHECK(b_matrix->hip_upload_buffer == b_upload);
  CHECK(c_matrix->hip_export_buffer == c_export);
  CHECK(c_matrix->hip_status_buffer == c_status);
  CHECK(a_matrix->hip_upload_bytes == a_upload_bytes);
  CHECK(b_matrix->hip_upload_bytes == b_upload_bytes);
  CHECK(c_matrix->hip_export_bytes == c_export_bytes);
  CHECK(c_matrix->hip_status_bytes == c_status_bytes);

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
  constexpr uint64_t signed_range_sentinel = 0x1212121212121212ull;
  std::vector<uint64_t> too_few_cpu(static_cast<std::size_t>(m * n), signed_range_sentinel);
  std::vector<uint64_t> too_few_hip(static_cast<std::size_t>(m * n), signed_range_sentinel);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, too_few_cpu.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, too_few_hip.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(too_few_cpu == std::vector<uint64_t>(static_cast<std::size_t>(m * n), signed_range_sentinel));
  CHECK(too_few_hip == std::vector<uint64_t>(static_cast<std::size_t>(m * n), signed_range_sentinel));
  CHECK_FALSE(hip_c->host_residues_current);
  std::vector<uint64_t> invalid_limb_layout(66, signed_range_sentinel);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, invalid_limb_layout.data(), limb_ld, 0) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, invalid_limb_layout.data(), limb_ld, 0) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, invalid_limb_layout.data(), limb_ld, 33) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, invalid_limb_layout.data(), limb_ld, 33) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, invalid_limb_layout.data(), n - 1, limb_count) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, invalid_limb_layout.data(), n - 1, limb_count) ==
        RNS8_INVALID_ARGUMENT);

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
  constexpr uint64_t unsigned_range_sentinel = 0x3434343434343434ull;
  std::vector<uint64_t> too_few_cpu(static_cast<std::size_t>(m * n), unsigned_range_sentinel);
  std::vector<uint64_t> too_few_hip(static_cast<std::size_t>(m * n), unsigned_range_sentinel);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, too_few_cpu.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, too_few_hip.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(too_few_cpu == std::vector<uint64_t>(static_cast<std::size_t>(m * n), unsigned_range_sentinel));
  CHECK(too_few_hip == std::vector<uint64_t>(static_cast<std::size_t>(m * n), unsigned_range_sentinel));
  CHECK_FALSE(hip_c->host_residues_current);
  std::vector<uint64_t> invalid_limb_layout(66, unsigned_range_sentinel);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, invalid_limb_layout.data(), limb_ld, 0) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, invalid_limb_layout.data(), limb_ld, 0) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, invalid_limb_layout.data(), limb_ld, 33) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, invalid_limb_layout.data(), limb_ld, 33) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, invalid_limb_layout.data(), n - 1, limb_count) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, invalid_limb_layout.data(), n - 1, limb_count) ==
        RNS8_INVALID_ARGUMENT);

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

TEST_CASE("direct HIP bounded descriptors hard-reject invalid bound contracts") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP bounded descriptor rejection smoke");
  }

  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* plan = nullptr;
  rns8_matrix* matrix = nullptr;

  auto signed_without_bound = signed_desc(1, 1, 1, 0, RNS8_BACKEND_HIP_DIRECT);
  signed_without_bound.bound_kind = RNS8_BOUND_NONE;
  CHECK(rns8_create_plan(hip, &signed_without_bound, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto signed_with_unsigned_bound = signed_desc(1, 1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
  signed_with_unsigned_bound.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  CHECK(rns8_create_plan(hip, &signed_with_unsigned_bound, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto unsigned_with_signed_bound = unsigned_desc(1, 1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
  unsigned_with_signed_bound.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  CHECK(rns8_create_plan(hip, &unsigned_with_signed_bound, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};
  auto per_tile_with_global_bound = per_tile_unsigned_desc(65, 65, 1, bounds, RNS8_BACKEND_HIP_DIRECT);
  per_tile_with_global_bound.bound = 1;
  CHECK(rns8_create_plan(hip, &per_tile_with_global_bound, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  const std::vector<uint64_t> too_few_bounds = {7, 1000, 7000000};
  auto per_tile_too_few = per_tile_unsigned_desc(65, 65, 1, too_few_bounds, RNS8_BACKEND_HIP_DIRECT);
  CHECK(rns8_create_plan(hip, &per_tile_too_few, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto insufficient_prefix = signed_desc(
      1, 1, 1, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), RNS8_BACKEND_HIP_DIRECT);
  insufficient_prefix.max_prefix = 8;
  CHECK(rns8_create_plan(hip, &insufficient_prefix, &plan) == RNS8_RANGE_ERROR);
  CHECK(plan == nullptr);

  auto bad_matrix_desc = matrix_desc(1, 1, RNS8_BOUNDED_U64, RNS8_BOUND_NONE);
  CHECK(rns8_create_matrix(hip, &bad_matrix_desc, &matrix) == RNS8_INVALID_ARGUMENT);
  CHECK(matrix == nullptr);

  rns8_destroy_context(hip);
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

TEST_CASE("direct HIP bounded oneshot matches CPU for cancellation and full-width stressors") {
  if (!hip_available()) {
    SKIP("no HIP device available for public bounded GEMM stress smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    constexpr int64_t m = 2;
    constexpr int64_t n = 2;
    constexpr int64_t k = 4;
    const int64_t A[] = {
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::min() + 1,
        37,
        -37,
        1000000000,
        -1000000000,
        -12345,
        12345};
    const int64_t B[] = {1, 1, 1, 1, 1000000, 1000000, 1000000, -1000000};
    int64_t cpu_c[4] = {};
    int64_t hip_c[4] = {};
    auto cpu_desc = signed_desc(m, n, k, 30000000000ull, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(m, n, k, 30000000000ull, RNS8_BACKEND_HIP_DIRECT);
    CHECK(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A, k, B, n, cpu_c, n) == RNS8_SUCCESS);
    CHECK(rns8_gemm_i64_oneshot(hip, &hip_desc, A, k, B, n, hip_c, n) == RNS8_SUCCESS);
    CHECK(std::vector<int64_t>(std::begin(hip_c), std::end(hip_c)) ==
          std::vector<int64_t>(std::begin(cpu_c), std::end(cpu_c)));
    CHECK(hip_c[0] == 0);
    CHECK(hip_c[1] == 74000000);
    CHECK(hip_c[2] == 0);
    CHECK(hip_c[3] == -24690000000LL);
  }

  {
    const int64_t A[] = {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    const int64_t B[] = {1, 0, 0, 1};
    int64_t cpu_c[2] = {};
    int64_t hip_c[2] = {};
    auto cpu_desc = signed_desc(1, 2, 2, 1ull << 63u, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(1, 2, 2, 1ull << 63u, RNS8_BACKEND_HIP_DIRECT);
    cpu_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    hip_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    CHECK(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A, 2, B, 2, cpu_c, 2) == RNS8_SUCCESS);
    CHECK(rns8_gemm_i64_oneshot(hip, &hip_desc, A, 2, B, 2, hip_c, 2) == RNS8_SUCCESS);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[1] == cpu_c[1]);
    CHECK(hip_c[0] == std::numeric_limits<int64_t>::min());
    CHECK(hip_c[1] == std::numeric_limits<int64_t>::max());
  }

  {
    const uint64_t A[] = {std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max() - 42};
    const uint64_t B[] = {1, 0, 0, 1};
    uint64_t cpu_c[2] = {};
    uint64_t hip_c[2] = {};
    auto cpu_desc = unsigned_desc(1, 2, 2, std::numeric_limits<uint64_t>::max(), RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = unsigned_desc(1, 2, 2, std::numeric_limits<uint64_t>::max(), RNS8_BACKEND_HIP_DIRECT);
    cpu_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    hip_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    CHECK(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A, 2, B, 2, cpu_c, 2) == RNS8_SUCCESS);
    CHECK(rns8_gemm_u64_oneshot(hip, &hip_desc, A, 2, B, 2, hip_c, 2) == RNS8_SUCCESS);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[1] == cpu_c[1]);
    CHECK(hip_c[0] == std::numeric_limits<uint64_t>::max());
    CHECK(hip_c[1] == std::numeric_limits<uint64_t>::max() - 42);
  }

  {
    const int64_t m = 1;
    const int64_t n = 2;
    const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
    std::vector<uint64_t> A(static_cast<std::size_t>(k), 255);
    std::vector<uint64_t> B(static_cast<std::size_t>(k * n));
    for (int64_t kk = 0; kk < k; ++kk) {
      B[static_cast<std::size_t>(kk * n)] = 1;
      B[static_cast<std::size_t>(kk * n + 1)] = 2;
    }
    uint64_t cpu_c[2] = {};
    uint64_t hip_c[2] = {};
    const uint64_t bound = static_cast<uint64_t>(k) * 255u * 2u;
    auto cpu_desc = unsigned_desc(m, n, k, bound, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = unsigned_desc(m, n, k, bound, RNS8_BACKEND_HIP_DIRECT);
    CHECK(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c, n) == RNS8_SUCCESS);
    CHECK(rns8_gemm_u64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_c, n) == RNS8_SUCCESS);
    CHECK(hip_c[0] == cpu_c[0]);
    CHECK(hip_c[1] == cpu_c[1]);
    CHECK(hip_c[0] == static_cast<uint64_t>(k) * 255u);
    CHECK(hip_c[1] == bound);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP per-tile bounded oneshot matches CPU for signed and unsigned APIs") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP per-tile bounded GEMM smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    constexpr int64_t m = 65;
    constexpr int64_t n = 65;
    constexpr int64_t k = 1;
    constexpr int64_t ldc = n + 1;
    std::vector<uint64_t> A(m * k);
    std::vector<uint64_t> B(k * n);
    std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), 0xdeadbeefdeadbeefull);
    std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), 0xdeadbeefdeadbeefull);
    for (int64_t row = 0; row < m; ++row) {
      A[static_cast<std::size_t>(row)] = row < 64 ? 1 : 1000000;
    }
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(col)] = col < 64 ? 7 : 1000;
    }
    const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};
    auto cpu_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);
    REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), ldc) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_u64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_c.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_c[static_cast<std::size_t>(row * ldc + col)]);
      }
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + n)] == 0xdeadbeefdeadbeefull);
    }

    const std::vector<uint64_t> too_small_bounds = {6, 1000, 7000000, 1000000000};
    auto bad_cpu_desc = per_tile_unsigned_desc(m, n, k, too_small_bounds, RNS8_BACKEND_CPU_REFERENCE);
    auto bad_hip_desc = per_tile_unsigned_desc(m, n, k, too_small_bounds, RNS8_BACKEND_HIP_DIRECT);
    CHECK(rns8_gemm_u64_oneshot(cpu, &bad_cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), ldc) ==
          RNS8_RANGE_ERROR);
    CHECK(rns8_gemm_u64_oneshot(hip, &bad_hip_desc, A.data(), k, B.data(), n, hip_c.data(), ldc) ==
          RNS8_RANGE_ERROR);
  }

  {
    constexpr int64_t m = 65;
    constexpr int64_t n = 65;
    constexpr int64_t k = 1;
    constexpr int64_t ldc = n + 1;
    std::vector<int64_t> A(m * k);
    std::vector<int64_t> B(k * n);
    std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * ldc), INT64_C(0x123456789abcdef));
    std::vector<int64_t> hip_c(static_cast<std::size_t>(m * ldc), INT64_C(0x123456789abcdef));
    for (int64_t row = 0; row < m; ++row) {
      A[static_cast<std::size_t>(row)] = row < 64 ? -2 : -1000000;
    }
    A[0] = 0;
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(col)] = col < 64 ? 3 : -1000;
    }
    const std::vector<uint64_t> bounds = {6, 2000, 3000000, 1000000000};
    auto cpu_desc = per_tile_signed_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = per_tile_signed_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);
    REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), ldc) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_i64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_c.data(), ldc) == RNS8_SUCCESS);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_c[static_cast<std::size_t>(row * ldc + col)]);
      }
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + n)] == INT64_C(0x123456789abcdef));
    }
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP per-tile bounded K-split oneshot matches CPU selected prefix") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP per-tile bounded K-split selected-prefix smoke");
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  constexpr int64_t ldc = n + 1;
  const uint64_t ku = static_cast<uint64_t>(k);
  std::vector<uint64_t> A(static_cast<std::size_t>(m * k));
  std::vector<uint64_t> B(static_cast<std::size_t>(k * n));
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), 0xccccccccccccccccull);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), 0xccccccccccccccccull);
  for (int64_t kk = 0; kk < k; ++kk) {
    A[static_cast<std::size_t>(kk)] = 1;
    A[static_cast<std::size_t>(k + kk)] = 2;
    B[static_cast<std::size_t>(kk * n)] = 3;
    B[static_cast<std::size_t>(kk * n + 1)] = 5;
  }

  const std::vector<uint64_t> bounds = {10u * ku};
  auto cpu_desc = unsigned_desc(m, n, k, 0, RNS8_BACKEND_CPU_REFERENCE);
  cpu_desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  cpu_desc.tile_m = 64;
  cpu_desc.tile_n = 64;
  cpu_desc.tile_bounds = bounds.data();
  cpu_desc.tile_bounds_count = bounds.size();
  auto hip_desc = cpu_desc;
  hip_desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_u64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_c.data(), ldc) == RNS8_SUCCESS);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
            cpu_c[static_cast<std::size_t>(row * ldc + col)]);
    }
    CHECK(hip_c[static_cast<std::size_t>(row * ldc + n)] == 0xccccccccccccccccull);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP per-tile bounded signed K-split preserves centered cancellation") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP per-tile signed K-split cancellation smoke");
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  const int64_t lda = k + 1;
  constexpr int64_t ldb = n + 1;
  constexpr int64_t ldc = n + 1;
  constexpr int64_t expected = 127 * 127;
  constexpr int64_t sentinel = INT64_C(-0x55aa55aa);
  std::vector<int64_t> A(static_cast<std::size_t>(m * lda), sentinel);
  std::vector<int64_t> B(static_cast<std::size_t>(k * ldb), sentinel);
  std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<int64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);
  for (int64_t kk = 0; kk < k; ++kk) {
    A[static_cast<std::size_t>(kk)] = kk % 2 == 0 ? 127 : -127;
    B[static_cast<std::size_t>(kk * ldb)] = 127;
    B[static_cast<std::size_t>(kk * ldb + 1)] = -127;
  }

  const std::vector<uint64_t> bounds = {static_cast<uint64_t>(expected)};
  auto cpu_desc = per_tile_signed_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = per_tile_signed_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* hip_plan = nullptr;
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(hip_plan->prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
  REQUIRE(hip_plan->tile_schedule.size() == 1);
  CHECK(hip_plan->tile_schedule[0].selected_prefix == hip_plan->tile_schedule[0].required_prefix);
  CHECK(hip_plan->tile_schedule[0].selected_prefix < hip_plan->prefix);
  rns8_destroy_plan(hip_plan);

  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_i64_oneshot(hip, &hip_desc, A.data(), lda, B.data(), ldb, hip_c.data(), ldc) == RNS8_SUCCESS);
  CHECK(hip_c[0] == cpu_c[0]);
  CHECK(hip_c[1] == cpu_c[1]);
  CHECK(hip_c[0] == expected);
  CHECK(hip_c[1] == -expected);
  CHECK(cpu_c[2] == sentinel);
  CHECK(hip_c[2] == sentinel);

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP per-tile bounded GEMM leaves skipped residue planes untouched") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP per-tile residue skip smoke");
  }

  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  constexpr int8_t sentinel = 42;
  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  std::vector<uint64_t> A(m * k);
  std::vector<uint64_t> B(k * n);
  for (int64_t row = 0; row < m; ++row) {
    A[static_cast<std::size_t>(row)] = row < 64 ? 1 : 1000000;
  }
  for (int64_t col = 0; col < n; ++col) {
    B[static_cast<std::size_t>(col)] = col < 64 ? 7 : 1000;
  }
  const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};
  auto cpu_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);

  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(cpu_plan->tile_schedule.size() == hip_plan->tile_schedule.size());
  for (std::size_t i = 0; i < cpu_plan->tile_schedule.size(); ++i) {
    CHECK(cpu_plan->tile_schedule[i].required_prefix == hip_plan->tile_schedule[i].required_prefix);
    CHECK(cpu_plan->tile_schedule[i].selected_prefix == hip_plan->tile_schedule[i].selected_prefix);
    CHECK(cpu_plan->tile_schedule[i].group_index == hip_plan->tile_schedule[i].group_index);
  }

  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);

  std::fill(hip_c->residues.begin(), hip_c->residues.end(), sentinel);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(
              0, hip_c->hip_residues, hip_c->residues.data(), hip_c->hip_residue_bytes) == RNS8_SUCCESS);
  hip_c->host_residues_current = true;
  hip_c->device_residues_current = true;

  REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), n, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(cpu, cpu_plan, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
  REQUIRE_FALSE(hip_c->host_residues_current);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
              0, hip_c->residues.data(), hip_c->hip_residues, hip_c->hip_residue_bytes) == RNS8_SUCCESS);

  for (const auto& entry : hip_plan->tile_schedule) {
    for (uint32_t p = 0; p < hip_plan->prefix; ++p) {
      for (int64_t row = entry.row_offset; row < entry.row_offset + entry.row_extent; ++row) {
        for (int64_t col = entry.col_offset; col < entry.col_offset + entry.col_extent; ++col) {
          const auto index = rns8::detail::residue_index(*hip_c, p, row, col);
          if (p < entry.selected_prefix) {
            CHECK(hip_c->residues[index] == cpu_c->residues[rns8::detail::residue_index(*cpu_c, p, row, col)]);
          } else {
            CHECK(hip_c->residues[index] == sentinel);
          }
        }
      }
    }
  }

  rns8_destroy_workspace(hip_workspace);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}
