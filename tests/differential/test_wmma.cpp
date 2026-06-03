#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "backend_wmma/wmma_backend.hpp"
#include "backend_wrap64/wrap64_hip.hpp"
#include "core/internal.hpp"
#include "rns8/rns8.h"

#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
#  include <hip/hip_runtime_api.h>
#endif

namespace {

#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
class HipBuffer {
 public:
  explicit HipBuffer(std::size_t bytes) {
    REQUIRE(bytes > 0);
    REQUIRE(hipMalloc(&ptr_, bytes) == hipSuccess);
  }

  ~HipBuffer() {
    if (ptr_) {
      (void)hipFree(ptr_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;

  void* get() const {
    return ptr_;
  }

 private:
  void* ptr_ = nullptr;
};

class HipScratchBuffer {
 public:
  ~HipScratchBuffer() {
    if (ptr_) {
      (void)hipFree(ptr_);
    }
  }

  HipScratchBuffer(const HipScratchBuffer&) = delete;
  HipScratchBuffer& operator=(const HipScratchBuffer&) = delete;

  HipScratchBuffer() = default;

  void** ptr_address() {
    return &ptr_;
  }

  std::size_t* bytes_address() {
    return &bytes_;
  }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

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

bool wmma_available() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_WMMA;
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

void require_wrap64_output_matches_oracle(
    const std::vector<uint64_t>& A,
    int64_t lda,
    const std::vector<uint64_t>& B,
    int64_t ldb,
    const std::vector<uint64_t>& C,
    int64_t ldc,
    int64_t m,
    int64_t n,
    int64_t k) {
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(C[static_cast<std::size_t>(row * ldc + col)] ==
            rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k));
    }
  }
}

struct Wrap64CandidateShape {
  const char* name;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t lda_padding;
  int64_t ldb_padding;
  int64_t ldc_padding;
  uint64_t seed;
};

uint64_t wrap64_candidate_a_value(int64_t row, int64_t col, std::mt19937_64& rng) {
  const auto selector = static_cast<uint32_t>((row * 13 + col * 17) % 11);
  if (selector == 0) {
    return std::numeric_limits<uint64_t>::max();
  }
  if (selector == 1) {
    return 0x8080808080808080ull;
  }
  if (selector == 2) {
    return 0x7f807f807f807f80ull;
  }
  if (selector == 3) {
    return 0x0000000000000001ull << static_cast<uint32_t>((row + col) % 63);
  }
  return rng();
}

uint64_t wrap64_candidate_b_value(int64_t row, int64_t col, std::mt19937_64& rng) {
  const auto selector = static_cast<uint32_t>((row * 19 + col * 23) % 11);
  if (selector == 0) {
    return std::numeric_limits<uint64_t>::max() - 1;
  }
  if (selector == 1) {
    return 0xfefdfcfbfaf9f8f7ull;
  }
  if (selector == 2) {
    return 0x0102030405060708ull;
  }
  if (selector == 3) {
    return 0x8000000000000000ull >> static_cast<uint32_t>((row + col) % 63);
  }
  return rng();
}

void require_wrap64_wmma_candidate_matches_direct_hip_and_oracle(const Wrap64CandidateShape& shape) {
  INFO(shape.name);

  constexpr int device_id = 0;
  constexpr uint64_t sentinel = 0xfeedfacecafebeefull;
  const int64_t lda = shape.k + shape.lda_padding;
  const int64_t ldb = shape.n + shape.ldb_padding;
  const int64_t ldc = shape.n + shape.ldc_padding;
  REQUIRE(lda >= shape.k);
  REQUIRE(ldb >= shape.n);
  REQUIRE(ldc >= shape.n);

  std::vector<uint64_t> A(static_cast<std::size_t>(shape.m * lda), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> B(static_cast<std::size_t>(shape.k * ldb), 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> direct_out(static_cast<std::size_t>(shape.m * ldc), sentinel);
  std::vector<uint64_t> wmma_out(static_cast<std::size_t>(shape.m * ldc), sentinel);
  std::mt19937_64 rng(shape.seed);

  for (int64_t row = 0; row < shape.m; ++row) {
    for (int64_t col = 0; col < shape.k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] = wrap64_candidate_a_value(row, col, rng);
    }
  }
  for (int64_t row = 0; row < shape.k; ++row) {
    for (int64_t col = 0; col < shape.n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] = wrap64_candidate_b_value(row, col, rng);
    }
  }

  const std::size_t a_limb_bytes = static_cast<std::size_t>(shape.m * shape.k * 8);
  const std::size_t b_limb_bytes = static_cast<std::size_t>(shape.k * shape.n * 8);
  const std::size_t c_limb_bytes = static_cast<std::size_t>(shape.m * shape.n * 8);
  HipBuffer a_limbs(a_limb_bytes);
  HipBuffer b_limbs(b_limb_bytes);
  HipBuffer direct_limbs(c_limb_bytes);
  HipBuffer wmma_limbs(c_limb_bytes);
  HipScratchBuffer upload_buffer;
  HipScratchBuffer export_buffer;

  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              device_id,
              A.data(),
              upload_buffer.ptr_address(),
              upload_buffer.bytes_address(),
              a_limbs.get(),
              shape.m,
              shape.k,
              lda) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              device_id,
              B.data(),
              upload_buffer.ptr_address(),
              upload_buffer.bytes_address(),
              b_limbs.get(),
              shape.k,
              shape.n,
              ldb) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(
              device_id, a_limbs.get(), b_limbs.get(), direct_limbs.get(), shape.m, shape.n, shape.k) ==
          RNS8_SUCCESS);
  REQUIRE(rns8::detail::wmma_wrap64_gemm_byte_limbs_candidate_device(
              device_id, a_limbs.get(), b_limbs.get(), wmma_limbs.get(), shape.m, shape.n, shape.k) ==
          RNS8_SUCCESS);
  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              device_id,
              direct_limbs.get(),
              export_buffer.ptr_address(),
              export_buffer.bytes_address(),
              shape.m,
              shape.n,
              direct_out.data(),
              ldc) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              device_id,
              wmma_limbs.get(),
              export_buffer.ptr_address(),
              export_buffer.bytes_address(),
              shape.m,
              shape.n,
              wmma_out.data(),
              ldc) == RNS8_SUCCESS);

  for (int64_t row = 0; row < shape.m; ++row) {
    for (int64_t col = shape.n; col < ldc; ++col) {
      CHECK(direct_out[static_cast<std::size_t>(row * ldc + col)] == sentinel);
      CHECK(wmma_out[static_cast<std::size_t>(row * ldc + col)] == sentinel);
    }
  }
  require_same_u64(direct_out, wmma_out);
  require_wrap64_output_matches_oracle(A, lda, B, ldb, wmma_out, ldc, shape.m, shape.n, shape.k);
}
#endif

}  // namespace

#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
TEST_CASE("rocWMMA fused bounded RNS backend matches CPU and direct HIP") {
  if (!wmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* wmma = create_backend_context(RNS8_BACKEND_WMMA);

  constexpr int64_t m = 65;
  constexpr int64_t n = 79;
  constexpr int64_t k = 33;
  std::vector<int64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<int64_t> B(static_cast<std::size_t>(k * n), 0);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * k + col)] = (row + col) % 5 == 0 ? -9 : (row - 2 * col) % 11;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * n + col)] = (row * 3 + col * 5) % 17 - 8;
    }
  }
  std::vector<int64_t> cpu_out(static_cast<std::size_t>(m * n), 0);
  std::vector<int64_t> hip_out(static_cast<std::size_t>(m * n), 0);
  std::vector<int64_t> wmma_out(static_cast<std::size_t>(m * n), 0);
  auto cpu_desc = bounded_i64_desc(m, n, k, 2000000, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = bounded_i64_desc(m, n, k, 2000000, RNS8_BACKEND_HIP_DIRECT);
  auto wmma_desc = bounded_i64_desc(m, n, k, 2000000, RNS8_BACKEND_WMMA);

  REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_i64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_i64_oneshot(wmma, &wmma_desc, A.data(), k, B.data(), n, wmma_out.data(), n) == RNS8_SUCCESS);
  require_same_i64(cpu_out, hip_out);
  require_same_i64(cpu_out, wmma_out);

  rns8_destroy_context(wmma);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("rocWMMA fused adaptive bounded schedule matches CPU and direct HIP") {
  if (!wmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* wmma = create_backend_context(RNS8_BACKEND_WMMA);

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
  std::vector<uint64_t> wmma_out(static_cast<std::size_t>(m * n), 0);
  auto cpu_desc = per_tile_u64_desc(m, n, k, bounds, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = per_tile_u64_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);
  auto wmma_desc = per_tile_u64_desc(m, n, k, bounds, RNS8_BACKEND_WMMA);

  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_u64_oneshot(hip, &hip_desc, A.data(), k, B.data(), n, hip_out.data(), n) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_u64_oneshot(wmma, &wmma_desc, A.data(), k, B.data(), n, wmma_out.data(), n) == RNS8_SUCCESS);
  require_same_u64(cpu_out, hip_out);
  require_same_u64(cpu_out, wmma_out);

  rns8_destroy_context(wmma);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("rocWMMA exact-wide RNS output matches CPU and direct HIP limbs") {
  if (!wmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* wmma = create_backend_context(RNS8_BACKEND_WMMA);
  constexpr int64_t m = 48;
  constexpr int64_t n = 80;
  constexpr int64_t k = 32;
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
    if (backend == RNS8_BACKEND_WMMA) {
      CHECK(std::string(info.selected_kernel) == "rocwmma_i8_i32_signed_hot_residue_v1");
      CHECK(std::string(info.epilogue_mode) == "rocwmma_fused_i32_to_centered_residue_rns_output");
      CHECK(info.workspace_required_bytes > 0);
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
  const auto wmma_limbs = run_backend(wmma, RNS8_BACKEND_WMMA);
  require_same_u64(cpu_limbs, hip_limbs);
  require_same_u64(cpu_limbs, wmma_limbs);

  rns8_destroy_context(wmma);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("rocWMMA finite u8 backend matches CPU and direct HIP across padded strides") {
  if (!wmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* wmma = create_backend_context(RNS8_BACKEND_WMMA);
  constexpr int64_t m = 48;
  constexpr int64_t n = 80;
  constexpr int64_t k = 32;
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
  std::vector<uint8_t> wmma_out(static_cast<std::size_t>(m * ldc), 0xcc);
  auto cpu_desc = finite_desc(m, n, k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = finite_desc(m, n, k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_HIP_DIRECT);
  auto wmma_desc = finite_desc(m, n, k, RNS8_FINITE_FIELD_U8, RNS8_BACKEND_WMMA);

  REQUIRE(rns8_gemm_finite_field_u8_oneshot(
              cpu, &cpu_desc, 251, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_field_u8_oneshot(
              hip, &hip_desc, 251, A.data(), lda, B.data(), ldb, hip_out.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_field_u8_oneshot(
              wmma, &wmma_desc, 251, A.data(), lda, B.data(), ldb, wmma_out.data(), ldc) == RNS8_SUCCESS);
  CHECK(hip_out == cpu_out);
  CHECK(wmma_out == cpu_out);

  rns8_destroy_context(wmma);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("rocWMMA finite u8 K-split preserves signed centered accumulation") {
  if (!wmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  rns8_context* cpu = create_backend_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_backend_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_context* wmma = create_backend_context(RNS8_BACKEND_WMMA);
  constexpr int64_t m = 16;
  constexpr int64_t n = 16;
  const int64_t k = 65537;
  std::vector<uint8_t> A(static_cast<std::size_t>(m * k), 255);
  std::vector<uint8_t> B(static_cast<std::size_t>(k * n), 255);
  std::vector<uint8_t> cpu_out(static_cast<std::size_t>(m * n), 0);
  std::vector<uint8_t> hip_out(static_cast<std::size_t>(m * n), 0);
  std::vector<uint8_t> wmma_out(static_cast<std::size_t>(m * n), 0);
  auto cpu_desc = finite_desc(m, n, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_CPU_REFERENCE, 256);
  auto hip_desc = finite_desc(m, n, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_HIP_DIRECT, 256);
  auto wmma_desc = finite_desc(m, n, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_WMMA, 256);

  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(cpu, &cpu_desc, 256, A.data(), k, B.data(), n, cpu_out.data(), n) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(hip, &hip_desc, 256, A.data(), k, B.data(), n, hip_out.data(), n) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(wmma, &wmma_desc, 256, A.data(), k, B.data(), n, wmma_out.data(), n) ==
          RNS8_SUCCESS);
  CHECK(hip_out == cpu_out);
  CHECK(wmma_out == cpu_out);

  rns8_destroy_context(wmma);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("rocWMMA wrap64 byte-GEMM36 candidate matches direct HIP and CPU oracle across tails") {
  if (!wmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  const std::vector<Wrap64CandidateShape> cases = {
      {"single-cell K tail", 1, 1, 1, 1, 1, 1, 0x6436776d6d617572ull},
      {"exact WMMA tile", 16, 16, 16, 2, 3, 4, 0x77726f6336343136ull},
      {"padded carry-heavy tile tails", 17, 19, 33, 3, 5, 7, 0x6436776d6d617572ull},
      {"ragged two-tile output", 31, 33, 47, 1, 2, 3, 0x7461696c73363477ull},
  };
  for (const auto& candidate_case : cases) {
    require_wrap64_wmma_candidate_matches_direct_hip_and_oracle(candidate_case);
  }
}

TEST_CASE("rocWMMA wrap64 byte-GEMM36 candidate enforces K boundary") {
  if (!wmma_available()) {
    SKIP("rocWMMA backend is not available on this device");
  }

  constexpr int device_id = 0;
  require_wrap64_wmma_candidate_matches_direct_hip_and_oracle(
      {"maximum accepted K", 1, 1, 32768, 0, 0, 0, 0x3332373638776d6dull});

  HipBuffer a_limbs(8);
  HipBuffer b_limbs(8);
  HipBuffer c_limbs(8);
  CHECK(rns8::detail::wmma_wrap64_gemm_byte_limbs_candidate_device(
            device_id, a_limbs.get(), b_limbs.get(), c_limbs.get(), 1, 1, 32769) == RNS8_INVALID_ARGUMENT);
}
#endif
