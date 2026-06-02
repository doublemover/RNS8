#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "core/internal.hpp"
#include "rns8/rns8.h"

namespace {

rns8_context* create_cpu_context() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  return rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS ? ctx : nullptr;
}

rns8_context* create_hip_context() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx = nullptr;
  return rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS ? ctx : nullptr;
}

rns8_context* create_wrap64_context() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  rns8_context* ctx = nullptr;
  return rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS ? ctx : nullptr;
}

rns8_gemm_desc signed_desc(int64_t m, int64_t n, int64_t k, uint64_t bound) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc signed_desc_for_backend(int64_t m, int64_t n, int64_t k, uint64_t bound, rns8_backend_kind backend) {
  auto desc = signed_desc(m, n, k, bound);
  desc.requested_backend = backend;
  return desc;
}

rns8_gemm_desc unsigned_desc(int64_t m, int64_t n, int64_t k, uint64_t bound) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc unsigned_desc_for_backend(
    int64_t m,
    int64_t n,
    int64_t k,
    uint64_t bound,
    rns8_backend_kind backend) {
  auto desc = unsigned_desc(m, n, k, bound);
  desc.requested_backend = backend;
  return desc;
}

rns8_gemm_desc wrap_desc_for_backend(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
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
  desc.max_prefix = semantics == RNS8_WRAP_U64_MOD_2_64 ? 0 : RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

using rns8::detail::cpp_int;

cpp_int abs_cpp(cpp_int value) {
  return value < 0 ? -value : value;
}

cpp_int i64_min_cpp() {
  return -cpp_int(std::numeric_limits<int64_t>::max()) - 1;
}

bool checked_u64_bound(const cpp_int& value, uint64_t& out) {
  if (value < 0 || value > cpp_int(std::numeric_limits<uint64_t>::max())) {
    return false;
  }
  out = static_cast<uint64_t>(value);
  return true;
}

bool verify_signed_public_case(
    rns8_context* ctx,
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<int64_t>& A,
    int64_t lda,
    const std::vector<int64_t>& B,
    int64_t ldb,
    const char* label) {
  std::vector<cpp_int> expected(static_cast<std::size_t>(m * n));
  cpp_int max_abs = 0;
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const cpp_int value = rns8::detail::exact_i64_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      if (value < i64_min_cpp() || value > cpp_int(std::numeric_limits<int64_t>::max())) {
        std::cerr << label << " expected signed output exceeds int64 range\n";
        return false;
      }
      expected[static_cast<std::size_t>(row * n + col)] = value;
      max_abs = std::max(max_abs, abs_cpp(value));
    }
  }

  uint64_t bound = 0;
  if (!checked_u64_bound(max_abs, bound)) {
    std::cerr << label << " signed bound exceeds uint64 range\n";
    return false;
  }

  std::vector<int64_t> C(static_cast<std::size_t>(m * n), 0);
  auto desc = signed_desc(m, n, k, bound);
  const rns8_status status = rns8_gemm_i64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, C.data(), n);
  if (status != RNS8_SUCCESS) {
    std::cerr << label << " bounded i64 status failed: " << rns8_status_string(status) << "\n";
    return false;
  }

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      if (cpp_int(C[static_cast<std::size_t>(row * n + col)]) !=
          expected[static_cast<std::size_t>(row * n + col)]) {
        std::cerr << label << " bounded i64 value mismatch at (" << row << ", " << col << ")\n";
        return false;
      }
    }
  }
  return true;
}

bool verify_unsigned_public_case(
    rns8_context* ctx,
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<uint64_t>& A,
    int64_t lda,
    const std::vector<uint64_t>& B,
    int64_t ldb,
    const char* label) {
  std::vector<cpp_int> expected(static_cast<std::size_t>(m * n));
  cpp_int max_value = 0;
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const cpp_int value = rns8::detail::exact_u64_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
      if (value < 0 || value > cpp_int(std::numeric_limits<uint64_t>::max())) {
        std::cerr << label << " expected unsigned output exceeds uint64 range\n";
        return false;
      }
      expected[static_cast<std::size_t>(row * n + col)] = value;
      max_value = std::max(max_value, value);
    }
  }

  uint64_t bound = 0;
  if (!checked_u64_bound(max_value, bound)) {
    std::cerr << label << " unsigned bound exceeds uint64 range\n";
    return false;
  }

  std::vector<uint64_t> C(static_cast<std::size_t>(m * n), 0);
  auto desc = unsigned_desc(m, n, k, bound);
  const rns8_status status = rns8_gemm_u64_oneshot(ctx, &desc, A.data(), lda, B.data(), ldb, C.data(), n);
  if (status != RNS8_SUCCESS) {
    std::cerr << label << " bounded u64 status failed: " << rns8_status_string(status) << "\n";
    return false;
  }

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      if (cpp_int(C[static_cast<std::size_t>(row * n + col)]) !=
          expected[static_cast<std::size_t>(row * n + col)]) {
        std::cerr << label << " bounded u64 value mismatch at (" << row << ", " << col << ")\n";
        return false;
      }
    }
  }
  return true;
}

int64_t centered_fixture_value(int64_t value, int64_t modulus, int64_t center) {
  int64_t residue = value % modulus;
  if (residue < 0) {
    residue += modulus;
  }
  return residue - center;
}

bool verify_tiny_dimension_sweep(rns8_context* ctx) {
  for (int64_t m = 1; m <= 8; ++m) {
    for (int64_t n = 1; n <= 8; ++n) {
      for (int64_t k = 1; k <= 8; ++k) {
        std::vector<int64_t> signed_a(static_cast<std::size_t>(m * k));
        std::vector<int64_t> signed_b(static_cast<std::size_t>(k * n));
        std::vector<uint64_t> unsigned_a(static_cast<std::size_t>(m * k));
        std::vector<uint64_t> unsigned_b(static_cast<std::size_t>(k * n));
        for (int64_t row = 0; row < m; ++row) {
          for (int64_t col = 0; col < k; ++col) {
            signed_a[static_cast<std::size_t>(row * k + col)] =
                centered_fixture_value(row * 17 + col * 5 + m * 3 - n * 2, 17, 8);
            unsigned_a[static_cast<std::size_t>(row * k + col)] =
                static_cast<uint64_t>((row * 19 + col * 3 + m + 1) % 17);
          }
        }
        for (int64_t row = 0; row < k; ++row) {
          for (int64_t col = 0; col < n; ++col) {
            signed_b[static_cast<std::size_t>(row * n + col)] =
                centered_fixture_value(row * 7 - col * 13 + k * 2 + n, 19, 9);
            unsigned_b[static_cast<std::size_t>(row * n + col)] =
                static_cast<uint64_t>((row * 5 + col * 11 + n + 2) % 19);
          }
        }
        if (!verify_signed_public_case(ctx, m, n, k, signed_a, k, signed_b, n, "tiny dimension sweep") ||
            !verify_unsigned_public_case(ctx, m, n, k, unsigned_a, k, unsigned_b, n, "tiny dimension sweep")) {
          return false;
        }
      }
    }
  }
  return true;
}

bool verify_fixed_seed_random(rns8_context* ctx) {
  std::mt19937_64 rng(0x8a5cd13f00dULL);
  std::uniform_int_distribution<int64_t> signed_dist(-31, 31);
  std::uniform_int_distribution<uint64_t> unsigned_dist(0, 63);

  for (int trial = 0; trial < 32; ++trial) {
    const int64_t m = 1 + static_cast<int64_t>(rng() % 8);
    const int64_t n = 1 + static_cast<int64_t>(rng() % 8);
    const int64_t k = 1 + static_cast<int64_t>(rng() % 8);
    const int64_t lda = k + 1;
    const int64_t ldb = n + 1;
    std::vector<int64_t> signed_a(static_cast<std::size_t>(m * lda), 12345);
    std::vector<int64_t> signed_b(static_cast<std::size_t>(k * ldb), -12345);
    std::vector<uint64_t> unsigned_a(static_cast<std::size_t>(m * lda), 99999);
    std::vector<uint64_t> unsigned_b(static_cast<std::size_t>(k * ldb), 99999);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        signed_a[static_cast<std::size_t>(row * lda + col)] = signed_dist(rng);
        unsigned_a[static_cast<std::size_t>(row * lda + col)] = unsigned_dist(rng);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        signed_b[static_cast<std::size_t>(row * ldb + col)] = signed_dist(rng);
        unsigned_b[static_cast<std::size_t>(row * ldb + col)] = unsigned_dist(rng);
      }
    }
    if (!verify_signed_public_case(ctx, m, n, k, signed_a, lda, signed_b, ldb, "fixed-seed random sweep") ||
        !verify_unsigned_public_case(ctx, m, n, k, unsigned_a, lda, unsigned_b, ldb, "fixed-seed random sweep")) {
      return false;
    }
  }
  return true;
}

bool verify_k_block_cases(rns8_context* ctx) {
  for (int64_t k : {static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK),
                    static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1}) {
    {
      std::vector<int64_t> A(static_cast<std::size_t>(k), 127);
      std::vector<int64_t> B(static_cast<std::size_t>(k), 127);
      if (!verify_signed_public_case(ctx, 1, 1, k, A, k, B, 1, "signed positive K-block sweep")) {
        return false;
      }
    }
    {
      std::vector<int64_t> A(static_cast<std::size_t>(k), -128);
      std::vector<int64_t> B(static_cast<std::size_t>(k), 127);
      if (!verify_signed_public_case(ctx, 1, 1, k, A, k, B, 1, "signed negative K-block sweep")) {
        return false;
      }
    }
    {
      std::vector<uint64_t> A(static_cast<std::size_t>(k), 255);
      std::vector<uint64_t> B(static_cast<std::size_t>(k), 255);
      if (!verify_unsigned_public_case(ctx, 1, 1, k, A, k, B, 1, "unsigned K-block sweep")) {
        return false;
      }
    }
  }
  return true;
}

bool verify_cpu() {
  if (rns8_validate_default_moduli() != RNS8_SUCCESS) {
    std::cerr << "default modulus ladder is not pairwise coprime\n";
    return false;
  }

  rns8_context* ctx = create_cpu_context();
  if (!ctx) {
    std::cerr << "failed to create CPU reference context\n";
    return false;
  }

  {
    const int64_t A[] = {7, -3, 5, -11, 13, 17};
    const int64_t B[] = {2, -5, 19, 23, -29, 31};
    int64_t C[4] = {};
    auto desc = signed_desc(2, 2, 3, 100000);
    const rns8_status status = rns8_gemm_i64_oneshot(ctx, &desc, A, 3, B, 2, C, 2);
    if (status != RNS8_SUCCESS || C[0] != -188 || C[1] != 51 || C[2] != -268 || C[3] != 881) {
      std::cerr << "bounded i64 verification failed: " << rns8_status_string(status) << "\n";
      rns8_destroy_context(ctx);
      return false;
    }
  }

  {
    const int64_t A[] = {std::numeric_limits<int64_t>::max()};
    const int64_t B[] = {1};
    int64_t C[] = {0};
    auto desc = signed_desc(1, 1, 1, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    const rns8_status status = rns8_gemm_i64_oneshot(ctx, &desc, A, 1, B, 1, C, 1);
    if (status != RNS8_SUCCESS || C[0] != std::numeric_limits<int64_t>::max()) {
      std::cerr << "bounded i64 boundary verification failed: " << rns8_status_string(status) << "\n";
      rns8_destroy_context(ctx);
      return false;
    }
  }

  {
    const uint64_t A[] = {std::numeric_limits<uint64_t>::max()};
    const uint64_t B[] = {1};
    uint64_t C[] = {0};
    auto desc = unsigned_desc(1, 1, 1, std::numeric_limits<uint64_t>::max());
    const rns8_status status = rns8_gemm_u64_oneshot(ctx, &desc, A, 1, B, 1, C, 1);
    if (status != RNS8_SUCCESS || C[0] != std::numeric_limits<uint64_t>::max()) {
      std::cerr << "bounded u64 boundary verification failed: " << rns8_status_string(status) << "\n";
      rns8_destroy_context(ctx);
      return false;
    }
  }

  if (!verify_tiny_dimension_sweep(ctx)) {
    rns8_destroy_context(ctx);
    return false;
  }

  if (!verify_fixed_seed_random(ctx)) {
    rns8_destroy_context(ctx);
    return false;
  }

  if (!verify_k_block_cases(ctx)) {
    rns8_destroy_context(ctx);
    return false;
  }

  rns8_destroy_context(ctx);
  return true;
}

bool verify_hip_smoke() {
  if (!rns8::detail::hip_direct_compiled()) {
    std::cerr << "direct HIP backend was not compiled\n";
    return false;
  }

  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  const rns8_status probe = rns8::detail::hip_direct_probe(0, info);
  if (probe != RNS8_SUCCESS) {
    std::cerr << "direct HIP probe failed: " << rns8_status_string(probe) << "\n";
    return false;
  }

  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 4;
  const uint16_t modulus = 251;
  const std::vector<int8_t> A = {1, -2, 3, -4, -5, 6, -7, 8};
  const std::vector<int8_t> B = {9, -10, 11, -12, -13, 14, 15, -16};
  std::vector<int8_t> cpu(4, 0);
  std::vector<int8_t> gpu(4, 0);
  rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, k, n, n, modulus);
  const rns8_status status =
      rns8::detail::hip_direct_ring_gemm_i8(0, A.data(), B.data(), gpu.data(), m, n, k, k, n, n, modulus);
  if (status != RNS8_SUCCESS || cpu != gpu) {
    std::cerr << "direct HIP ring GEMM smoke failed: " << rns8_status_string(status) << "\n";
    return false;
  }

  const int64_t split_k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  const uint64_t split_bound = static_cast<uint64_t>(split_k) * 127u * 127u;
  std::vector<int64_t> split_a(static_cast<std::size_t>(split_k), 127);
  std::vector<int64_t> split_b(static_cast<std::size_t>(split_k), 127);

  rns8_context* cpu_ctx = create_cpu_context();
  rns8_context* hip_ctx = create_hip_context();
  rns8_context* wrap_ctx = create_wrap64_context();
  if (!cpu_ctx || !hip_ctx || !wrap_ctx) {
    std::cerr << "failed to create CPU, wrap64, or direct HIP context for smoke\n";
    rns8_destroy_context(wrap_ctx);
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  {
    const int64_t rows = 2;
    const int64_t cols = 4;
    const int64_t ld = 5;
    const std::vector<int64_t> src = {
        0,
        1,
        -1,
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
    rns8_status cpu_status = rns8_create_matrix(cpu_ctx, &desc, &cpu_matrix);
    rns8_status hip_status = rns8_create_matrix(hip_ctx, &desc, &hip_matrix);
    if (cpu_status == RNS8_SUCCESS) {
      cpu_status = rns8_pack_i64(cpu_ctx, cpu_matrix, src.data(), ld, 1);
    }
    if (hip_status == RNS8_SUCCESS) {
      hip_status = rns8_pack_i64(hip_ctx, hip_matrix, src.data(), ld, 1);
    }
    if (hip_status == RNS8_SUCCESS && hip_matrix) {
      hip_status = rns8::detail::hip_direct_copy_device_to_host(
          hip_matrix->hip_device_id,
          hip_matrix->residues.data(),
          hip_matrix->hip_residues,
          hip_matrix->hip_residue_bytes);
    }
    const bool equal = cpu_matrix && hip_matrix && cpu_matrix->residues == hip_matrix->residues;
    rns8_destroy_matrix(hip_matrix);
    rns8_destroy_matrix(cpu_matrix);
    if (cpu_status != RNS8_SUCCESS || hip_status != RNS8_SUCCESS || !equal) {
      std::cerr << "direct HIP i64 residue pack smoke failed: CPU=" << rns8_status_string(cpu_status)
                << " HIP=" << rns8_status_string(hip_status) << "\n";
      rns8_destroy_context(wrap_ctx);
      rns8_destroy_context(hip_ctx);
      rns8_destroy_context(cpu_ctx);
      return false;
    }
  }

  {
    const int64_t rows = 2;
    const int64_t cols = 4;
    const int64_t ld = 5;
    const std::vector<uint64_t> src = {
        0,
        1,
        127,
        128,
        999,
        255,
        256,
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max() - 1,
        999};
    auto desc = matrix_desc(rows, cols, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    rns8_matrix* cpu_matrix = nullptr;
    rns8_matrix* hip_matrix = nullptr;
    rns8_status cpu_status = rns8_create_matrix(cpu_ctx, &desc, &cpu_matrix);
    rns8_status hip_status = rns8_create_matrix(hip_ctx, &desc, &hip_matrix);
    if (cpu_status == RNS8_SUCCESS) {
      cpu_status = rns8_pack_u64(cpu_ctx, cpu_matrix, src.data(), ld, 1);
    }
    if (hip_status == RNS8_SUCCESS) {
      hip_status = rns8_pack_u64(hip_ctx, hip_matrix, src.data(), ld, 1);
    }
    if (hip_status == RNS8_SUCCESS && hip_matrix) {
      hip_status = rns8::detail::hip_direct_copy_device_to_host(
          hip_matrix->hip_device_id,
          hip_matrix->residues.data(),
          hip_matrix->hip_residues,
          hip_matrix->hip_residue_bytes);
    }
    const bool equal = cpu_matrix && hip_matrix && cpu_matrix->residues == hip_matrix->residues;
    rns8_destroy_matrix(hip_matrix);
    rns8_destroy_matrix(cpu_matrix);
    if (cpu_status != RNS8_SUCCESS || hip_status != RNS8_SUCCESS || !equal) {
      std::cerr << "direct HIP u64 residue pack smoke failed: CPU=" << rns8_status_string(cpu_status)
                << " HIP=" << rns8_status_string(hip_status) << "\n";
      rns8_destroy_context(wrap_ctx);
      rns8_destroy_context(hip_ctx);
      rns8_destroy_context(cpu_ctx);
      return false;
    }
  }

  int64_t cpu_split_c[1] = {};
  int64_t hip_split_c[1] = {};
  auto cpu_split_desc = signed_desc_for_backend(1, 1, split_k, split_bound, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_split_desc = signed_desc_for_backend(1, 1, split_k, split_bound, RNS8_BACKEND_HIP_DIRECT);
  const rns8_status cpu_split_status =
      rns8_gemm_i64_oneshot(cpu_ctx, &cpu_split_desc, split_a.data(), split_k, split_b.data(), 1, cpu_split_c, 1);
  const rns8_status hip_split_status =
      rns8_gemm_i64_oneshot(hip_ctx, &hip_split_desc, split_a.data(), split_k, split_b.data(), 1, hip_split_c, 1);
  if (cpu_split_status != RNS8_SUCCESS || hip_split_status != RNS8_SUCCESS || cpu_split_c[0] != hip_split_c[0] ||
      hip_split_c[0] != static_cast<int64_t>(split_bound)) {
    std::cerr << "direct HIP bounded i64 split smoke failed: CPU=" << rns8_status_string(cpu_split_status)
              << " HIP=" << rns8_status_string(hip_split_status) << "\n";
    rns8_destroy_context(wrap_ctx);
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  const uint64_t u_a[] = {17, 3, 255, 9, 41, 5};
  const uint64_t u_b[] = {11, 7, 13, 19, 23, 29};
  uint64_t cpu_u_c[4] = {};
  uint64_t hip_u_c[4] = {};
  auto cpu_u_desc = unsigned_desc_for_backend(2, 2, 3, 20000, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_u_desc = unsigned_desc_for_backend(2, 2, 3, 20000, RNS8_BACKEND_HIP_DIRECT);
  const rns8_status cpu_u_status = rns8_gemm_u64_oneshot(cpu_ctx, &cpu_u_desc, u_a, 3, u_b, 2, cpu_u_c, 2);
  const rns8_status hip_u_status = rns8_gemm_u64_oneshot(hip_ctx, &hip_u_desc, u_a, 3, u_b, 2, hip_u_c, 2);
  if (cpu_u_status != RNS8_SUCCESS || hip_u_status != RNS8_SUCCESS ||
      std::vector<uint64_t>(std::begin(cpu_u_c), std::end(cpu_u_c)) !=
          std::vector<uint64_t>(std::begin(hip_u_c), std::end(hip_u_c))) {
    std::cerr << "direct HIP bounded u64 smoke failed: CPU=" << rns8_status_string(cpu_u_status)
              << " HIP=" << rns8_status_string(hip_u_status) << "\n";
    rns8_destroy_context(wrap_ctx);
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  const int64_t adaptive_m = 65;
  const int64_t adaptive_n = 65;
  const int64_t adaptive_k = 1;
  std::vector<uint64_t> adaptive_a(static_cast<std::size_t>(adaptive_m * adaptive_k));
  std::vector<uint64_t> adaptive_b(static_cast<std::size_t>(adaptive_k * adaptive_n));
  std::vector<uint64_t> cpu_adaptive_c(static_cast<std::size_t>(adaptive_m * adaptive_n), 0);
  std::vector<uint64_t> hip_adaptive_c(static_cast<std::size_t>(adaptive_m * adaptive_n), 0);
  for (int64_t row = 0; row < adaptive_m; ++row) {
    adaptive_a[static_cast<std::size_t>(row)] = row < 64 ? 1 : 1000000;
  }
  for (int64_t col = 0; col < adaptive_n; ++col) {
    adaptive_b[static_cast<std::size_t>(col)] = col < 64 ? 7 : 1000;
  }
  const std::vector<uint64_t> adaptive_bounds = {7, 1000, 7000000, 1000000000};
  auto cpu_adaptive_desc =
      unsigned_desc_for_backend(adaptive_m, adaptive_n, adaptive_k, 0, RNS8_BACKEND_CPU_REFERENCE);
  cpu_adaptive_desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  cpu_adaptive_desc.tile_m = 64;
  cpu_adaptive_desc.tile_n = 64;
  cpu_adaptive_desc.tile_bounds = adaptive_bounds.data();
  cpu_adaptive_desc.tile_bounds_count = adaptive_bounds.size();
  auto hip_adaptive_desc = cpu_adaptive_desc;
  hip_adaptive_desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
  const rns8_status cpu_adaptive_status = rns8_gemm_u64_oneshot(
      cpu_ctx,
      &cpu_adaptive_desc,
      adaptive_a.data(),
      adaptive_k,
      adaptive_b.data(),
      adaptive_n,
      cpu_adaptive_c.data(),
      adaptive_n);
  const rns8_status hip_adaptive_status = rns8_gemm_u64_oneshot(
      hip_ctx,
      &hip_adaptive_desc,
      adaptive_a.data(),
      adaptive_k,
      adaptive_b.data(),
      adaptive_n,
      hip_adaptive_c.data(),
      adaptive_n);
  if (cpu_adaptive_status != RNS8_SUCCESS || hip_adaptive_status != RNS8_SUCCESS ||
      cpu_adaptive_c != hip_adaptive_c) {
    std::cerr << "direct HIP per-tile bounded u64 smoke failed: CPU="
              << rns8_status_string(cpu_adaptive_status) << " HIP=" << rns8_status_string(hip_adaptive_status)
              << "\n";
    rns8_destroy_context(wrap_ctx);
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  const int64_t wrap_m = 2;
  const int64_t wrap_n = 3;
  const int64_t wrap_k = 4;
  const int64_t wrap_ldc = 4;
  const std::vector<uint64_t> wrap_a = {
      0,
      1,
      std::numeric_limits<uint64_t>::max(),
      0x8080808080808080ull,
      255,
      256,
      std::numeric_limits<uint64_t>::max() - 1,
      0x0102030405060708ull};
  const std::vector<uint64_t> wrap_b = {
      3,
      std::numeric_limits<uint64_t>::max(),
      0x1112131415161718ull,
      29,
      0x8080808080808080ull,
      31,
      37,
      41,
      43,
      47,
      53,
      59};
  std::vector<uint64_t> cpu_wrap_c(static_cast<std::size_t>(wrap_m * wrap_ldc), 0xfeedfacefeedfaceull);
  std::vector<uint64_t> hip_wrap_c(static_cast<std::size_t>(wrap_m * wrap_ldc), 0xfeedfacefeedfaceull);
  auto cpu_wrap_desc = wrap_desc_for_backend(wrap_m, wrap_n, wrap_k, RNS8_BACKEND_WRAP64_BYTE_LIMB);
  auto hip_wrap_desc = wrap_desc_for_backend(wrap_m, wrap_n, wrap_k, RNS8_BACKEND_HIP_DIRECT);
  const rns8_status cpu_wrap_status = rns8_gemm_wrap_u64_oneshot(
      wrap_ctx, &cpu_wrap_desc, wrap_a.data(), wrap_k, wrap_b.data(), wrap_n, cpu_wrap_c.data(), wrap_ldc);
  const rns8_status hip_wrap_status = rns8_gemm_wrap_u64_oneshot(
      hip_ctx, &hip_wrap_desc, wrap_a.data(), wrap_k, wrap_b.data(), wrap_n, hip_wrap_c.data(), wrap_ldc);
  if (cpu_wrap_status != RNS8_SUCCESS || hip_wrap_status != RNS8_SUCCESS || cpu_wrap_c != hip_wrap_c) {
    std::cerr << "direct HIP wrap64 byte-limb smoke failed: CPU=" << rns8_status_string(cpu_wrap_status)
              << " HIP=" << rns8_status_string(hip_wrap_status) << "\n";
    rns8_destroy_context(wrap_ctx);
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  rns8_destroy_context(wrap_ctx);
  rns8_destroy_context(hip_ctx);
  rns8_destroy_context(cpu_ctx);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  bool hip_smoke = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--hip-smoke") {
      hip_smoke = true;
    } else if (arg == "--help") {
      std::cout << "usage: rns8-verify [--hip-smoke]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }

  if (!verify_cpu()) {
    return 1;
  }
  std::cout << "CPU reference verification: PASS\n";

  if (hip_smoke) {
    if (!verify_hip_smoke()) {
      return 1;
    }
    std::cout << "Direct HIP pack, ring, bounded GEMM, adaptive bounded GEMM, and wrap64 smoke: PASS\n";
  }

  return 0;
}
