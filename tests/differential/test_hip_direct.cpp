#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>

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

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_direct_ring_gemm_i8_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int modulus_index,
    int selected_prefix,
    int safe_k_block);
#endif

uint32_t reciprocal_for_modulus(uint16_t modulus) {
  return static_cast<uint32_t>((uint64_t{1} << 32u) / static_cast<uint32_t>(modulus));
}

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
  desc.max_prefix = 0;
  desc.finite_modulus =
      modulus != 0 ? modulus : (semantics == RNS8_FINITE_FIELD_U8 ? uint16_t{251} : uint16_t{255});
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
  } else if (semantics == RNS8_WRAP_U64_MOD_2_64 || semantics == RNS8_FINITE_RING_U8 ||
             semantics == RNS8_FINITE_FIELD_U8) {
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

void run_wrap64_resident_device_gemm(
    const std::vector<uint8_t>& a_limbs,
    const std::vector<uint8_t>& b_limbs,
    std::vector<uint8_t>& c_limbs,
    int64_t m,
    int64_t n,
    int64_t k) {
  void* device_a = nullptr;
  void* device_b = nullptr;
  void* device_c = nullptr;
  REQUIRE(rns8::detail::hip_direct_allocate(0, a_limbs.size(), &device_a) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, b_limbs.size(), &device_b) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, c_limbs.size(), &device_c) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, device_a, a_limbs.data(), a_limbs.size()) ==
          RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, device_b, b_limbs.data(), b_limbs.size()) ==
          RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, device_c, c_limbs.data(), c_limbs.size()) ==
          RNS8_SUCCESS);

  REQUIRE(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, device_a, device_b, device_c, m, n, k) ==
          RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, c_limbs.data(), device_c, c_limbs.size()) ==
          RNS8_SUCCESS);

  CHECK(rns8::detail::hip_direct_free(0, device_c) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, device_b) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, device_a) == RNS8_SUCCESS);
}

void run_resident_ring_gemm(
    const std::vector<int8_t>& A,
    const std::vector<int8_t>& B,
    std::vector<int8_t>& C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus,
    uint32_t modulus_index,
    uint32_t selected_prefix) {
  const std::size_t a_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(lda) * sizeof(int8_t);
  const std::size_t b_bytes = static_cast<std::size_t>(k) * static_cast<std::size_t>(ldb) * sizeof(int8_t);
  const std::size_t c_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(ldc) * sizeof(int8_t);
  REQUIRE(A.size() * sizeof(int8_t) >= a_bytes);
  REQUIRE(B.size() * sizeof(int8_t) >= b_bytes);
  REQUIRE(C.size() * sizeof(int8_t) >= c_bytes);

  void* device_a = nullptr;
  void* device_b = nullptr;
  void* device_c = nullptr;
  REQUIRE(rns8::detail::hip_direct_allocate(0, a_bytes, &device_a) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, b_bytes, &device_b) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, c_bytes, &device_c) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, device_a, A.data(), a_bytes) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, device_b, B.data(), b_bytes) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, device_c, C.data(), c_bytes) == RNS8_SUCCESS);

  REQUIRE(rns8::detail::hip_direct_ring_gemm_i8_device(
              0,
              device_a,
              device_b,
              device_c,
              m,
              n,
              k,
              lda,
              ldb,
              ldc,
              modulus,
              modulus_index,
              selected_prefix) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, C.data(), device_c, c_bytes) == RNS8_SUCCESS);

  CHECK(rns8::detail::hip_direct_free(0, device_c) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, device_b) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, device_a) == RNS8_SUCCESS);
}

std::vector<int8_t> exact_residues_for(boost::multiprecision::cpp_int value, uint32_t prefix) {
  std::vector<int8_t> residues(prefix);
  for (uint32_t p = 0; p < prefix; ++p) {
    residues[p] = rns8::detail::centered_residue(value, rns8::detail::kDefaultModuli[p]);
  }
  return residues;
}

void fill_exact_residue_matrix(rns8_matrix* matrix, const std::vector<boost::multiprecision::cpp_int>& values) {
  REQUIRE(matrix != nullptr);
  REQUIRE(values.size() == static_cast<std::size_t>(matrix->desc.rows * matrix->desc.cols));
  for (uint32_t p = 0; p < matrix->prefix; ++p) {
    for (int64_t row = 0; row < matrix->desc.rows; ++row) {
      for (int64_t col = 0; col < matrix->desc.cols; ++col) {
        const std::size_t value_index = static_cast<std::size_t>(row * matrix->desc.cols + col);
        const auto residues = exact_residues_for(values[value_index], matrix->prefix);
        const std::size_t residue = rns8::detail::residue_index(*matrix, p, row, col);
        matrix->residues[residue] = residues[p];
      }
    }
  }
  matrix->host_residues_current = true;
  matrix->device_residues_current = false;
}

void upload_exact_residues_to_hip(rns8_matrix* matrix) {
  REQUIRE(matrix != nullptr);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(
              matrix->hip_device_id, matrix->hip_residues, matrix->residues.data(), matrix->hip_residue_bytes) ==
          RNS8_SUCCESS);
  matrix->host_residues_current = false;
  matrix->device_residues_current = true;
}

struct BoundedHipResidentSnapshot {
  rns8::detail::hip_direct_allocation_counters allocations{};
  void* a_residues = nullptr;
  void* b_residues = nullptr;
  void* c_residues = nullptr;
  void* a_upload = nullptr;
  void* b_upload = nullptr;
  void* c_upload = nullptr;
  void* c_export = nullptr;
  void* c_status = nullptr;
  std::size_t a_residue_bytes = 0;
  std::size_t b_residue_bytes = 0;
  std::size_t c_residue_bytes = 0;
  std::size_t a_upload_bytes = 0;
  std::size_t b_upload_bytes = 0;
  std::size_t c_upload_bytes = 0;
  std::size_t c_export_bytes = 0;
  std::size_t c_status_bytes = 0;
  uint64_t workspace_bound = 0;
  uint32_t workspace_tile_m = 0;
  uint32_t workspace_tile_n = 0;
  uint64_t workspace_schedule_tile_rows = 0;
  uint64_t workspace_schedule_tile_cols = 0;
  uint64_t workspace_schedule_tile_count = 0;
  uint32_t workspace_schedule_min_required_prefix = 0;
  uint32_t workspace_schedule_max_required_prefix = 0;
  uint32_t workspace_schedule_min_selected_prefix = 0;
  uint32_t workspace_schedule_max_selected_prefix = 0;
  uint32_t workspace_schedule_prefix_group_count = 0;
  uint32_t workspace_schedule_range_bit_length = 0;
  uint32_t workspace_schedule_adaptive_prefix_active = 0;
  uint32_t workspace_schedule_adaptive_skip_active = 0;
  uint32_t workspace_schedule_flags = 0;
  uint64_t workspace_schedule_fingerprint = 0;
};

BoundedHipResidentSnapshot capture_bounded_resident_snapshot(
    const rns8_matrix* A,
    const rns8_matrix* B,
    const rns8_matrix* C,
    const rns8_workspace* workspace) {
  REQUIRE(A != nullptr);
  REQUIRE(B != nullptr);
  REQUIRE(C != nullptr);
  REQUIRE(workspace != nullptr);
  BoundedHipResidentSnapshot snapshot{};
  snapshot.allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  snapshot.a_residues = A->hip_residues;
  snapshot.b_residues = B->hip_residues;
  snapshot.c_residues = C->hip_residues;
  snapshot.a_upload = A->hip_upload_buffer;
  snapshot.b_upload = B->hip_upload_buffer;
  snapshot.c_upload = C->hip_upload_buffer;
  snapshot.c_export = C->hip_export_buffer;
  snapshot.c_status = C->hip_status_buffer;
  snapshot.a_residue_bytes = A->hip_residue_bytes;
  snapshot.b_residue_bytes = B->hip_residue_bytes;
  snapshot.c_residue_bytes = C->hip_residue_bytes;
  snapshot.a_upload_bytes = A->hip_upload_bytes;
  snapshot.b_upload_bytes = B->hip_upload_bytes;
  snapshot.c_upload_bytes = C->hip_upload_bytes;
  snapshot.c_export_bytes = C->hip_export_bytes;
  snapshot.c_status_bytes = C->hip_status_bytes;
  snapshot.workspace_bound = workspace->bound;
  snapshot.workspace_tile_m = workspace->tile_m;
  snapshot.workspace_tile_n = workspace->tile_n;
  snapshot.workspace_schedule_tile_rows = workspace->schedule_tile_rows;
  snapshot.workspace_schedule_tile_cols = workspace->schedule_tile_cols;
  snapshot.workspace_schedule_tile_count = workspace->schedule_tile_count;
  snapshot.workspace_schedule_min_required_prefix = workspace->schedule_min_required_prefix;
  snapshot.workspace_schedule_max_required_prefix = workspace->schedule_max_required_prefix;
  snapshot.workspace_schedule_min_selected_prefix = workspace->schedule_min_selected_prefix;
  snapshot.workspace_schedule_max_selected_prefix = workspace->schedule_max_selected_prefix;
  snapshot.workspace_schedule_prefix_group_count = workspace->schedule_prefix_group_count;
  snapshot.workspace_schedule_range_bit_length = workspace->schedule_range_bit_length;
  snapshot.workspace_schedule_adaptive_prefix_active = workspace->schedule_adaptive_prefix_active;
  snapshot.workspace_schedule_adaptive_skip_active = workspace->schedule_adaptive_skip_active;
  snapshot.workspace_schedule_flags = workspace->schedule_flags;
  snapshot.workspace_schedule_fingerprint = workspace->schedule_fingerprint;
  return snapshot;
}

void check_bounded_resident_snapshot_unchanged(
    const BoundedHipResidentSnapshot& snapshot,
    const rns8_matrix* A,
    const rns8_matrix* B,
    const rns8_matrix* C,
    const rns8_workspace* workspace) {
  const auto repeated = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated.allocate_calls == snapshot.allocations.allocate_calls);
  CHECK(repeated.free_calls == snapshot.allocations.free_calls);
  CHECK(repeated.allocated_bytes == snapshot.allocations.allocated_bytes);
  CHECK(A->hip_residues == snapshot.a_residues);
  CHECK(B->hip_residues == snapshot.b_residues);
  CHECK(C->hip_residues == snapshot.c_residues);
  CHECK(A->hip_upload_buffer == snapshot.a_upload);
  CHECK(B->hip_upload_buffer == snapshot.b_upload);
  CHECK(C->hip_upload_buffer == snapshot.c_upload);
  CHECK(C->hip_export_buffer == snapshot.c_export);
  CHECK(C->hip_status_buffer == snapshot.c_status);
  CHECK(A->hip_residue_bytes == snapshot.a_residue_bytes);
  CHECK(B->hip_residue_bytes == snapshot.b_residue_bytes);
  CHECK(C->hip_residue_bytes == snapshot.c_residue_bytes);
  CHECK(A->hip_upload_bytes == snapshot.a_upload_bytes);
  CHECK(B->hip_upload_bytes == snapshot.b_upload_bytes);
  CHECK(C->hip_upload_bytes == snapshot.c_upload_bytes);
  CHECK(C->hip_export_bytes == snapshot.c_export_bytes);
  CHECK(C->hip_status_bytes == snapshot.c_status_bytes);
  CHECK(workspace->bound == snapshot.workspace_bound);
  CHECK(workspace->tile_m == snapshot.workspace_tile_m);
  CHECK(workspace->tile_n == snapshot.workspace_tile_n);
  CHECK(workspace->schedule_tile_rows == snapshot.workspace_schedule_tile_rows);
  CHECK(workspace->schedule_tile_cols == snapshot.workspace_schedule_tile_cols);
  CHECK(workspace->schedule_tile_count == snapshot.workspace_schedule_tile_count);
  CHECK(workspace->schedule_min_required_prefix == snapshot.workspace_schedule_min_required_prefix);
  CHECK(workspace->schedule_max_required_prefix == snapshot.workspace_schedule_max_required_prefix);
  CHECK(workspace->schedule_min_selected_prefix == snapshot.workspace_schedule_min_selected_prefix);
  CHECK(workspace->schedule_max_selected_prefix == snapshot.workspace_schedule_max_selected_prefix);
  CHECK(workspace->schedule_prefix_group_count == snapshot.workspace_schedule_prefix_group_count);
  CHECK(workspace->schedule_range_bit_length == snapshot.workspace_schedule_range_bit_length);
  CHECK(workspace->schedule_adaptive_prefix_active == snapshot.workspace_schedule_adaptive_prefix_active);
  CHECK(workspace->schedule_adaptive_skip_active == snapshot.workspace_schedule_adaptive_skip_active);
  CHECK(workspace->schedule_flags == snapshot.workspace_schedule_flags);
  CHECK(workspace->schedule_fingerprint == snapshot.workspace_schedule_fingerprint);
}

struct Wrap64HipResidentSnapshot {
  rns8::detail::hip_direct_allocation_counters allocations{};
  void* a_byte_limbs = nullptr;
  void* b_byte_limbs = nullptr;
  void* c_byte_limbs = nullptr;
  void* a_upload = nullptr;
  void* b_upload = nullptr;
  void* c_upload = nullptr;
  void* c_export = nullptr;
  void* c_status = nullptr;
  std::size_t a_byte_limb_bytes = 0;
  std::size_t b_byte_limb_bytes = 0;
  std::size_t c_byte_limb_bytes = 0;
  std::size_t a_upload_bytes = 0;
  std::size_t b_upload_bytes = 0;
  std::size_t c_upload_bytes = 0;
  std::size_t c_export_bytes = 0;
  std::size_t c_status_bytes = 0;
  uint64_t workspace_bound = 0;
  uint32_t workspace_tile_m = 0;
  uint32_t workspace_tile_n = 0;
  uint64_t workspace_schedule_tile_rows = 0;
  uint64_t workspace_schedule_tile_cols = 0;
  uint64_t workspace_schedule_tile_count = 0;
  uint32_t workspace_schedule_min_required_prefix = 0;
  uint32_t workspace_schedule_max_required_prefix = 0;
  uint32_t workspace_schedule_min_selected_prefix = 0;
  uint32_t workspace_schedule_max_selected_prefix = 0;
  uint32_t workspace_schedule_prefix_group_count = 0;
  uint32_t workspace_schedule_range_bit_length = 0;
  uint32_t workspace_schedule_adaptive_prefix_active = 0;
  uint32_t workspace_schedule_adaptive_skip_active = 0;
  uint32_t workspace_schedule_flags = 0;
  uint64_t workspace_schedule_fingerprint = 0;
};

Wrap64HipResidentSnapshot capture_wrap64_resident_snapshot(
    const rns8_matrix* A,
    const rns8_matrix* B,
    const rns8_matrix* C,
    const rns8_workspace* workspace) {
  REQUIRE(A != nullptr);
  REQUIRE(B != nullptr);
  REQUIRE(C != nullptr);
  REQUIRE(workspace != nullptr);
  Wrap64HipResidentSnapshot snapshot{};
  snapshot.allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  snapshot.a_byte_limbs = A->hip_byte_limbs;
  snapshot.b_byte_limbs = B->hip_byte_limbs;
  snapshot.c_byte_limbs = C->hip_byte_limbs;
  snapshot.a_upload = A->hip_upload_buffer;
  snapshot.b_upload = B->hip_upload_buffer;
  snapshot.c_upload = C->hip_upload_buffer;
  snapshot.c_export = C->hip_export_buffer;
  snapshot.c_status = C->hip_status_buffer;
  snapshot.a_byte_limb_bytes = A->hip_byte_limb_bytes;
  snapshot.b_byte_limb_bytes = B->hip_byte_limb_bytes;
  snapshot.c_byte_limb_bytes = C->hip_byte_limb_bytes;
  snapshot.a_upload_bytes = A->hip_upload_bytes;
  snapshot.b_upload_bytes = B->hip_upload_bytes;
  snapshot.c_upload_bytes = C->hip_upload_bytes;
  snapshot.c_export_bytes = C->hip_export_bytes;
  snapshot.c_status_bytes = C->hip_status_bytes;
  snapshot.workspace_bound = workspace->bound;
  snapshot.workspace_tile_m = workspace->tile_m;
  snapshot.workspace_tile_n = workspace->tile_n;
  snapshot.workspace_schedule_tile_rows = workspace->schedule_tile_rows;
  snapshot.workspace_schedule_tile_cols = workspace->schedule_tile_cols;
  snapshot.workspace_schedule_tile_count = workspace->schedule_tile_count;
  snapshot.workspace_schedule_min_required_prefix = workspace->schedule_min_required_prefix;
  snapshot.workspace_schedule_max_required_prefix = workspace->schedule_max_required_prefix;
  snapshot.workspace_schedule_min_selected_prefix = workspace->schedule_min_selected_prefix;
  snapshot.workspace_schedule_max_selected_prefix = workspace->schedule_max_selected_prefix;
  snapshot.workspace_schedule_prefix_group_count = workspace->schedule_prefix_group_count;
  snapshot.workspace_schedule_range_bit_length = workspace->schedule_range_bit_length;
  snapshot.workspace_schedule_adaptive_prefix_active = workspace->schedule_adaptive_prefix_active;
  snapshot.workspace_schedule_adaptive_skip_active = workspace->schedule_adaptive_skip_active;
  snapshot.workspace_schedule_flags = workspace->schedule_flags;
  snapshot.workspace_schedule_fingerprint = workspace->schedule_fingerprint;
  return snapshot;
}

void check_wrap64_resident_snapshot_unchanged(
    const Wrap64HipResidentSnapshot& snapshot,
    const rns8_matrix* A,
    const rns8_matrix* B,
    const rns8_matrix* C,
    const rns8_workspace* workspace) {
  const auto repeated = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated.allocate_calls == snapshot.allocations.allocate_calls);
  CHECK(repeated.free_calls == snapshot.allocations.free_calls);
  CHECK(repeated.allocated_bytes == snapshot.allocations.allocated_bytes);
  CHECK(A->hip_residues == nullptr);
  CHECK(B->hip_residues == nullptr);
  CHECK(C->hip_residues == nullptr);
  CHECK(A->hip_byte_limbs == snapshot.a_byte_limbs);
  CHECK(B->hip_byte_limbs == snapshot.b_byte_limbs);
  CHECK(C->hip_byte_limbs == snapshot.c_byte_limbs);
  CHECK(A->hip_upload_buffer == snapshot.a_upload);
  CHECK(B->hip_upload_buffer == snapshot.b_upload);
  CHECK(C->hip_upload_buffer == snapshot.c_upload);
  CHECK(C->hip_export_buffer == snapshot.c_export);
  CHECK(C->hip_status_buffer == snapshot.c_status);
  CHECK(A->hip_byte_limb_bytes == snapshot.a_byte_limb_bytes);
  CHECK(B->hip_byte_limb_bytes == snapshot.b_byte_limb_bytes);
  CHECK(C->hip_byte_limb_bytes == snapshot.c_byte_limb_bytes);
  CHECK(A->hip_upload_bytes == snapshot.a_upload_bytes);
  CHECK(B->hip_upload_bytes == snapshot.b_upload_bytes);
  CHECK(C->hip_upload_bytes == snapshot.c_upload_bytes);
  CHECK(C->hip_export_bytes == snapshot.c_export_bytes);
  CHECK(C->hip_status_bytes == snapshot.c_status_bytes);
  CHECK(A->device_byte_limbs_current);
  CHECK(B->device_byte_limbs_current);
  CHECK(C->device_byte_limbs_current);
  CHECK_FALSE(A->host_byte_limbs_current);
  CHECK_FALSE(B->host_byte_limbs_current);
  CHECK_FALSE(C->host_byte_limbs_current);
  CHECK(workspace->bound == snapshot.workspace_bound);
  CHECK(workspace->tile_m == snapshot.workspace_tile_m);
  CHECK(workspace->tile_n == snapshot.workspace_tile_n);
  CHECK(workspace->schedule_tile_rows == snapshot.workspace_schedule_tile_rows);
  CHECK(workspace->schedule_tile_cols == snapshot.workspace_schedule_tile_cols);
  CHECK(workspace->schedule_tile_count == snapshot.workspace_schedule_tile_count);
  CHECK(workspace->schedule_min_required_prefix == snapshot.workspace_schedule_min_required_prefix);
  CHECK(workspace->schedule_max_required_prefix == snapshot.workspace_schedule_max_required_prefix);
  CHECK(workspace->schedule_min_selected_prefix == snapshot.workspace_schedule_min_selected_prefix);
  CHECK(workspace->schedule_max_selected_prefix == snapshot.workspace_schedule_max_selected_prefix);
  CHECK(workspace->schedule_prefix_group_count == snapshot.workspace_schedule_prefix_group_count);
  CHECK(workspace->schedule_range_bit_length == snapshot.workspace_schedule_range_bit_length);
  CHECK(workspace->schedule_adaptive_prefix_active == snapshot.workspace_schedule_adaptive_prefix_active);
  CHECK(workspace->schedule_adaptive_skip_active == snapshot.workspace_schedule_adaptive_skip_active);
  CHECK(workspace->schedule_flags == snapshot.workspace_schedule_flags);
  CHECK(workspace->schedule_fingerprint == snapshot.workspace_schedule_fingerprint);
}

struct FiniteHipResidentSnapshot {
  rns8::detail::hip_direct_allocation_counters allocations{};
  void* a_residues = nullptr;
  void* b_residues = nullptr;
  void* c_residues = nullptr;
  void* a_upload = nullptr;
  void* b_upload = nullptr;
  void* c_upload = nullptr;
  void* c_export = nullptr;
  void* c_status = nullptr;
  std::size_t a_residue_bytes = 0;
  std::size_t b_residue_bytes = 0;
  std::size_t c_residue_bytes = 0;
  std::size_t a_upload_bytes = 0;
  std::size_t b_upload_bytes = 0;
  std::size_t c_upload_bytes = 0;
  std::size_t c_export_bytes = 0;
  std::size_t c_status_bytes = 0;
  uint16_t a_modulus = 0;
  uint16_t b_modulus = 0;
  uint16_t c_modulus = 0;
  uint64_t workspace_bound = 0;
  uint32_t workspace_prefix = 0;
  uint32_t workspace_schedule_min_required_prefix = 0;
  uint32_t workspace_schedule_max_required_prefix = 0;
  uint32_t workspace_schedule_min_selected_prefix = 0;
  uint32_t workspace_schedule_max_selected_prefix = 0;
  uint32_t workspace_schedule_prefix_group_count = 0;
};

FiniteHipResidentSnapshot capture_finite_resident_snapshot(
    const rns8_matrix* A,
    const rns8_matrix* B,
    const rns8_matrix* C,
    const rns8_workspace* workspace) {
  REQUIRE(A != nullptr);
  REQUIRE(B != nullptr);
  REQUIRE(C != nullptr);
  REQUIRE(workspace != nullptr);
  FiniteHipResidentSnapshot snapshot{};
  snapshot.allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  snapshot.a_residues = A->hip_residues;
  snapshot.b_residues = B->hip_residues;
  snapshot.c_residues = C->hip_residues;
  snapshot.a_upload = A->hip_upload_buffer;
  snapshot.b_upload = B->hip_upload_buffer;
  snapshot.c_upload = C->hip_upload_buffer;
  snapshot.c_export = C->hip_export_buffer;
  snapshot.c_status = C->hip_status_buffer;
  snapshot.a_residue_bytes = A->hip_residue_bytes;
  snapshot.b_residue_bytes = B->hip_residue_bytes;
  snapshot.c_residue_bytes = C->hip_residue_bytes;
  snapshot.a_upload_bytes = A->hip_upload_bytes;
  snapshot.b_upload_bytes = B->hip_upload_bytes;
  snapshot.c_upload_bytes = C->hip_upload_bytes;
  snapshot.c_export_bytes = C->hip_export_bytes;
  snapshot.c_status_bytes = C->hip_status_bytes;
  snapshot.a_modulus = A->finite_modulus;
  snapshot.b_modulus = B->finite_modulus;
  snapshot.c_modulus = C->finite_modulus;
  snapshot.workspace_bound = workspace->bound;
  snapshot.workspace_prefix = workspace->prefix;
  snapshot.workspace_schedule_min_required_prefix = workspace->schedule_min_required_prefix;
  snapshot.workspace_schedule_max_required_prefix = workspace->schedule_max_required_prefix;
  snapshot.workspace_schedule_min_selected_prefix = workspace->schedule_min_selected_prefix;
  snapshot.workspace_schedule_max_selected_prefix = workspace->schedule_max_selected_prefix;
  snapshot.workspace_schedule_prefix_group_count = workspace->schedule_prefix_group_count;
  return snapshot;
}

void check_finite_resident_snapshot_unchanged(
    const FiniteHipResidentSnapshot& snapshot,
    const rns8_matrix* A,
    const rns8_matrix* B,
    const rns8_matrix* C,
    const rns8_workspace* workspace) {
  const auto repeated = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated.allocate_calls == snapshot.allocations.allocate_calls);
  CHECK(repeated.free_calls == snapshot.allocations.free_calls);
  CHECK(repeated.allocated_bytes == snapshot.allocations.allocated_bytes);
  CHECK(A->hip_residues == snapshot.a_residues);
  CHECK(B->hip_residues == snapshot.b_residues);
  CHECK(C->hip_residues == snapshot.c_residues);
  CHECK(A->hip_upload_buffer == snapshot.a_upload);
  CHECK(B->hip_upload_buffer == snapshot.b_upload);
  CHECK(C->hip_upload_buffer == snapshot.c_upload);
  CHECK(C->hip_export_buffer == snapshot.c_export);
  CHECK(C->hip_status_buffer == snapshot.c_status);
  CHECK(A->hip_residue_bytes == snapshot.a_residue_bytes);
  CHECK(B->hip_residue_bytes == snapshot.b_residue_bytes);
  CHECK(C->hip_residue_bytes == snapshot.c_residue_bytes);
  CHECK(A->hip_upload_bytes == snapshot.a_upload_bytes);
  CHECK(B->hip_upload_bytes == snapshot.b_upload_bytes);
  CHECK(C->hip_upload_bytes == snapshot.c_upload_bytes);
  CHECK(C->hip_export_bytes == snapshot.c_export_bytes);
  CHECK(C->hip_status_bytes == snapshot.c_status_bytes);
  CHECK(A->hip_byte_limbs == nullptr);
  CHECK(B->hip_byte_limbs == nullptr);
  CHECK(C->hip_byte_limbs == nullptr);
  CHECK(A->device_residues_current);
  CHECK(B->device_residues_current);
  CHECK(C->device_residues_current);
  CHECK_FALSE(A->host_residues_current);
  CHECK_FALSE(B->host_residues_current);
  CHECK_FALSE(C->host_residues_current);
  CHECK(A->finite_modulus == snapshot.a_modulus);
  CHECK(B->finite_modulus == snapshot.b_modulus);
  CHECK(C->finite_modulus == snapshot.c_modulus);
  CHECK(workspace->bound == snapshot.workspace_bound);
  CHECK(workspace->prefix == snapshot.workspace_prefix);
  CHECK(workspace->schedule_min_required_prefix == snapshot.workspace_schedule_min_required_prefix);
  CHECK(workspace->schedule_max_required_prefix == snapshot.workspace_schedule_max_required_prefix);
  CHECK(workspace->schedule_min_selected_prefix == snapshot.workspace_schedule_min_selected_prefix);
  CHECK(workspace->schedule_max_selected_prefix == snapshot.workspace_schedule_max_selected_prefix);
  CHECK(workspace->schedule_prefix_group_count == snapshot.workspace_schedule_prefix_group_count);
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
  run_resident_ring_gemm(A, B, gpu, m, n, k, k, n, n, modulus, 1, 2);
  CHECK(gpu == cpu);
}

TEST_CASE("direct HIP ring GEMM reciprocal reduction matches CPU across supported default moduli") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP reciprocal ladder smoke");
  }

  const int64_t m = 5;
  const int64_t n = 7;
  const int64_t k = 67;
  const int64_t lda = k + 3;
  const int64_t ldb = n + 2;
  const int64_t ldc = n + 4;
  const int8_t a_sentinel = static_cast<int8_t>(-88);
  const int8_t b_sentinel = static_cast<int8_t>(99);
  const int8_t c_sentinel = static_cast<int8_t>(-33);
  std::vector<int8_t> A(static_cast<std::size_t>(m * lda), a_sentinel);
  std::vector<int8_t> B(static_cast<std::size_t>(k * ldb), b_sentinel);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t kk = 0; kk < k; ++kk) {
      const int pattern = static_cast<int>((row * 37 + kk * 19) % 11);
      const int mixed = static_cast<int>((row + 113 - (2 * kk) % 113) % 113);
      A[static_cast<std::size_t>(row * lda + kk)] =
          pattern == 0 ? static_cast<int8_t>(127)
                       : pattern == 1 ? static_cast<int8_t>(-127)
                                      : static_cast<int8_t>(mixed - 56);
    }
  }
  for (int64_t kk = 0; kk < k; ++kk) {
    for (int64_t col = 0; col < n; ++col) {
      const int pattern = static_cast<int>((kk * 23 + col * 29) % 13);
      B[static_cast<std::size_t>(kk * ldb + col)] =
          pattern == 0 ? static_cast<int8_t>(126)
                       : pattern == 1 ? static_cast<int8_t>(-128)
                                      : static_cast<int8_t>(((3 * kk + col) % 127) - 63);
    }
  }

  for (uint32_t p = 0; p < RNS8_MAX_SUPPORTED_PREFIX; ++p) {
    const uint16_t modulus = rns8::detail::kDefaultModuli[p];
    std::vector<int8_t> cpu(static_cast<std::size_t>(m * ldc), c_sentinel);
    std::vector<int8_t> gpu(static_cast<std::size_t>(m * ldc), c_sentinel);

    rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, lda, ldb, ldc, modulus);
    run_resident_ring_gemm(A, B, gpu, m, n, k, lda, ldb, ldc, modulus, p, p + 1);

    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CAPTURE(p, modulus, row, col);
        CHECK(gpu[static_cast<std::size_t>(row * ldc + col)] ==
              cpu[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CAPTURE(p, modulus, row, col);
        CHECK(cpu[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
        CHECK(gpu[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
      }
    }
  }
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
  run_resident_ring_gemm(A, B, gpu, m, n, k, k, n, n, modulus, 1, 2);
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
  run_resident_ring_gemm(A, B, gpu, m, n, k, k, n, n, modulus, 3, 4);
  CHECK(gpu == cpu);
}

TEST_CASE("direct HIP tiled ring GEMM covers shared-memory tile tails with padded layouts") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP tiled-tail smoke");
  }

  const int64_t m = 17;
  const int64_t n = 19;
  const int64_t k = 130;
  const int64_t lda = 137;
  const int64_t ldb = 23;
  const int64_t ldc = 29;
  const uint16_t modulus = 253;
  const int8_t a_sentinel = static_cast<int8_t>(-99);
  const int8_t b_sentinel = static_cast<int8_t>(77);
  const int8_t c_sentinel = static_cast<int8_t>(44);
  std::vector<int8_t> A(static_cast<std::size_t>(m * lda), a_sentinel);
  std::vector<int8_t> B(static_cast<std::size_t>(k * ldb), b_sentinel);
  std::vector<int8_t> cpu(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<int8_t> gpu(static_cast<std::size_t>(m * ldc), c_sentinel);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t kk = 0; kk < k; ++kk) {
      switch ((row * 17 + kk * 5) % 7) {
        case 0:
          A[static_cast<std::size_t>(row * lda + kk)] = 127;
          break;
        case 1:
          A[static_cast<std::size_t>(row * lda + kk)] = -126;
          break;
        case 2:
          A[static_cast<std::size_t>(row * lda + kk)] = 64;
          break;
        case 3:
          A[static_cast<std::size_t>(row * lda + kk)] = -63;
          break;
        default:
          A[static_cast<std::size_t>(row * lda + kk)] = static_cast<int8_t>((row - kk) % 11);
          break;
      }
    }
  }
  for (int64_t kk = 0; kk < k; ++kk) {
    for (int64_t col = 0; col < n; ++col) {
      switch ((kk * 11 + col * 3) % 7) {
        case 0:
          B[static_cast<std::size_t>(kk * ldb + col)] = 126;
          break;
        case 1:
          B[static_cast<std::size_t>(kk * ldb + col)] = -127;
          break;
        case 2:
          B[static_cast<std::size_t>(kk * ldb + col)] = 63;
          break;
        case 3:
          B[static_cast<std::size_t>(kk * ldb + col)] = -64;
          break;
        default:
          B[static_cast<std::size_t>(kk * ldb + col)] = static_cast<int8_t>((kk + col) % 13 - 6);
          break;
      }
    }
  }

  rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, lda, ldb, ldc, modulus);
  run_resident_ring_gemm(A, B, gpu, m, n, k, lda, ldb, ldc, modulus, 2, 3);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(gpu[static_cast<std::size_t>(row * ldc + col)] ==
            cpu[static_cast<std::size_t>(row * ldc + col)]);
    }
    for (int64_t col = n; col < ldc; ++col) {
      CHECK(cpu[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
      CHECK(gpu[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
    }
  }
}

TEST_CASE("direct HIP finite u8 one-shot matches CPU for explicit ring and field moduli") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP finite u8 smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr int64_t m = 17;
  constexpr int64_t n = 19;
  constexpr int64_t k = 33;
  constexpr int64_t lda = 40;
  constexpr int64_t ldb = 23;
  constexpr int64_t ldc = 29;
  std::vector<uint8_t> A(static_cast<std::size_t>(m * lda), 0xa5);
  std::vector<uint8_t> B(static_cast<std::size_t>(k * ldb), 0x5a);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] =
          static_cast<uint8_t>((row * 37 + col * 19 + (row ^ col) * 11) & 0xff);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] =
          static_cast<uint8_t>((row * 23 + col * 41 + (row + col) * 7) & 0xff);
    }
  }

  struct Case {
    rns8_semantics semantics;
    uint16_t modulus;
  };
  for (const Case item : {
           Case{RNS8_FINITE_RING_U8, 255},
           Case{RNS8_FINITE_RING_U8, 256},
           Case{RNS8_FINITE_FIELD_U8, 251},
       }) {
    auto cpu_desc = finite_desc(m, n, k, item.semantics, RNS8_BACKEND_CPU_REFERENCE, item.modulus);
    auto hip_desc = finite_desc(m, n, k, item.semantics, RNS8_BACKEND_HIP_DIRECT, item.modulus);
    std::vector<uint8_t> cpu_out(static_cast<std::size_t>(m * ldc), 0xcc);
    std::vector<uint8_t> hip_out(static_cast<std::size_t>(m * ldc), 0xcc);

    if (item.semantics == RNS8_FINITE_FIELD_U8) {
      REQUIRE(rns8_gemm_finite_field_u8_oneshot(
                  cpu, &cpu_desc, item.modulus, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) ==
              RNS8_SUCCESS);
      rns8::detail::hip_direct_timing_set_enabled(true);
      rns8::detail::hip_direct_timing_reset();
      REQUIRE(rns8_gemm_finite_field_u8_oneshot(
                  hip, &hip_desc, item.modulus, A.data(), lda, B.data(), ldb, hip_out.data(), ldc) ==
              RNS8_SUCCESS);
    } else {
      REQUIRE(rns8_gemm_finite_ring_u8_oneshot(
                  cpu, &cpu_desc, item.modulus, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) ==
              RNS8_SUCCESS);
      rns8::detail::hip_direct_timing_set_enabled(true);
      rns8::detail::hip_direct_timing_reset();
      REQUIRE(rns8_gemm_finite_ring_u8_oneshot(
                  hip, &hip_desc, item.modulus, A.data(), lda, B.data(), ldb, hip_out.data(), ldc) ==
              RNS8_SUCCESS);
    }
    const auto hip_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(hip_events, "finite_pack_kernel"));
    CHECK(has_timing_label(hip_events, "finite_resident_gemm_kernel"));
    CHECK(has_timing_label(hip_events, "finite_export_kernel"));
    CHECK_FALSE(has_timing_label(hip_events, "finite_ring_gemm_kernel"));

    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(cpu_out[static_cast<std::size_t>(row * ldc + col)] == 0xcc);
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xcc);
      }
    }
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP finite u8 persistent matrices match CPU and reuse resident storage") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP finite u8 persistent smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr int64_t m = 13;
  constexpr int64_t n = 11;
  constexpr int64_t k = 17;
  constexpr int64_t lda = 23;
  constexpr int64_t ldb = 19;
  constexpr int64_t ldc = 17;
  std::vector<uint8_t> A(static_cast<std::size_t>(m * lda), 0xa5);
  std::vector<uint8_t> B(static_cast<std::size_t>(k * ldb), 0x5a);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] =
          static_cast<uint8_t>((row * 31 + col * 47 + (row ^ (col * 3)) * 5) & 0xff);
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] =
          static_cast<uint8_t>((row * 29 + col * 43 + (row + col * 7) * 9) & 0xff);
    }
  }

  struct Case {
    rns8_semantics semantics;
    uint16_t modulus;
  };
  for (const Case item : {
           Case{RNS8_FINITE_RING_U8, 255},
           Case{RNS8_FINITE_RING_U8, 256},
           Case{RNS8_FINITE_FIELD_U8, 251},
       }) {
    auto cpu_desc = finite_desc(m, n, k, item.semantics, RNS8_BACKEND_CPU_REFERENCE, item.modulus);
    auto hip_desc = finite_desc(m, n, k, item.semantics, RNS8_BACKEND_HIP_DIRECT, item.modulus);
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* hip_plan = nullptr;
    rns8_workspace* cpu_workspace = nullptr;
    rns8_workspace* hip_workspace = nullptr;
    rns8_matrix* cpu_a = nullptr;
    rns8_matrix* cpu_b = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* hip_a = nullptr;
    rns8_matrix* hip_b = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);

    auto a_desc = matrix_desc(m, k, item.semantics, RNS8_BOUND_NONE);
    auto b_desc = matrix_desc(k, n, item.semantics, RNS8_BOUND_NONE);
    auto c_desc = matrix_desc(m, n, item.semantics, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

    std::vector<uint8_t> cpu_out(static_cast<std::size_t>(m * ldc), 0xcc);
    std::vector<uint8_t> hip_out(static_cast<std::size_t>(m * ldc), 0xcc);
    REQUIRE(rns8_pack_finite_u8(cpu, cpu_a, item.modulus, A.data(), lda, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(cpu, cpu_b, item.modulus, B.data(), ldb, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_u8(cpu, cpu_plan, item.modulus, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_finite_u8(cpu, cpu_plan, item.modulus, cpu_c, cpu_out.data(), ldc) == RNS8_SUCCESS);

    rns8::detail::hip_direct_allocation_counters_reset();
    REQUIRE(rns8_pack_finite_u8(hip, hip_a, item.modulus, A.data(), lda, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(hip, hip_b, item.modulus, B.data(), ldb, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_u8(hip, hip_plan, item.modulus, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_finite_u8(hip, hip_plan, item.modulus, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    const auto warmed_snapshot = capture_finite_resident_snapshot(hip_a, hip_b, hip_c, hip_workspace);

    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(cpu_out[static_cast<std::size_t>(row * ldc + col)] == 0xcc);
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xcc);
      }
    }

    std::fill(hip_out.begin(), hip_out.end(), 0xdd);
    REQUIRE(rns8_pack_finite_u8(hip, hip_a, item.modulus, A.data(), lda, 3) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_finite_u8(hip, hip_b, item.modulus, B.data(), ldb, 4) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_finite_u8(hip, hip_plan, item.modulus, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_finite_u8(hip, hip_plan, item.modulus, hip_c, hip_out.data(), ldc) == RNS8_SUCCESS);
    check_finite_resident_snapshot_unchanged(warmed_snapshot, hip_a, hip_b, hip_c, hip_workspace);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xdd);
      }
    }
    CHECK(rns8_gemm_rns(hip, hip_plan, hip_a, hip_b, hip_c, hip_workspace) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_finite_u8(hip, hip_plan, item.modulus == 251 ? 255 : 251, hip_c, hip_out.data(), ldc) ==
          RNS8_INVALID_ARGUMENT);

    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(hip_b);
    rns8_destroy_matrix(hip_a);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_matrix(cpu_b);
    rns8_destroy_matrix(cpu_a);
    rns8_destroy_workspace(hip_workspace);
    rns8_destroy_workspace(cpu_workspace);
    rns8_destroy_plan(hip_plan);
    rns8_destroy_plan(cpu_plan);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP finite u8 one-shot preserves K-split semantics") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP finite u8 K-split smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  std::vector<uint8_t> A(static_cast<std::size_t>(k), 255);
  std::vector<uint8_t> B(static_cast<std::size_t>(k), 255);
  uint8_t cpu_out = 0;
  uint8_t hip_out = 0;
  auto cpu_desc = finite_desc(1, 1, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_CPU_REFERENCE, 256);
  auto hip_desc = finite_desc(1, 1, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_HIP_DIRECT, 256);

  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(cpu, &cpu_desc, 256, A.data(), k, B.data(), 1, &cpu_out, 1) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_ring_u8_oneshot(hip, &hip_desc, 256, A.data(), k, B.data(), 1, &hip_out, 1) ==
          RNS8_SUCCESS);
  CHECK(hip_out == cpu_out);

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP fixed-prefix plans advertise grouped GEMM kernels") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP fixed-prefix metadata smoke");
  }

  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr const char* grouped_kernel = "direct_hip_prefix9_grouped_rns_gemm_v1";
  for (const rns8_semantics semantics : {RNS8_BOUNDED_I64, RNS8_BOUNDED_U64}) {
    CAPTURE(semantics);
    auto desc = semantics == RNS8_BOUNDED_I64
                    ? signed_desc(32, 32, 32, 32u * 127u * 127u, RNS8_BACKEND_HIP_DIRECT)
                    : unsigned_desc(32, 32, 32, 32u * 127u * 127u, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(plan->prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    CHECK(plan->tile_schedule.empty());

    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    CHECK(std::string(info.selected_kernel) == grouped_kernel);
    CHECK(std::string(info.autotune_key).find(std::string("kernel=") + grouped_kernel + ";") != std::string::npos);

    rns8_destroy_plan(plan);
  }

  constexpr const char* exact_grouped_kernel = "direct_hip_prefix20_grouped_rns_gemm_v1";
  for (const rns8_semantics semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    CAPTURE(semantics);
    auto desc = semantics == RNS8_EXACT_WIDE_SIGNED ? exact_signed_desc(16, 16, 16, RNS8_BACKEND_HIP_DIRECT)
                                                    : exact_unsigned_desc(16, 16, 16, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(plan->prefix == RNS8_MAX_SUPPORTED_PREFIX);
    CHECK(plan->tile_schedule.empty());

    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(plan, &info) == RNS8_SUCCESS);
    CHECK(std::string(info.selected_kernel) == exact_grouped_kernel);
    CHECK(std::string(info.autotune_key).find(std::string("kernel=") + exact_grouped_kernel + ";") !=
          std::string::npos);

    rns8_destroy_plan(plan);
  }

  const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};
  auto adaptive_desc = per_tile_signed_desc(65, 65, 64, bounds, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* adaptive_plan = nullptr;
  REQUIRE(rns8_create_plan(hip, &adaptive_desc, &adaptive_plan) == RNS8_SUCCESS);
  REQUIRE_FALSE(adaptive_plan->tile_schedule.empty());
  rns8_plan_backend_info adaptive_info{};
  adaptive_info.struct_size = sizeof(adaptive_info);
  adaptive_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(adaptive_plan, &adaptive_info) == RNS8_SUCCESS);
  CHECK(std::string(adaptive_info.selected_kernel) == "direct_hip_tiled_rns_gemm_v1");
  rns8_destroy_plan(adaptive_plan);
  rns8_destroy_context(hip);
}

TEST_CASE("direct HIP finite u8 fixed-modulus kernels preserve K-split semantics") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP finite reducer smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr int64_t m = 3;
  constexpr int64_t n = 2;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 17;
  constexpr int64_t lda = k + 3;
  constexpr int64_t ldb = 5;
  constexpr int64_t ldc = 4;

  struct Case {
    rns8_semantics semantics;
    uint16_t modulus;
    const char* kernel;
  };
  for (const Case item : {
           Case{RNS8_FINITE_FIELD_U8, 251, "direct_hip_tiled_finite_u8_gemm_mod251_v1"},
           Case{RNS8_FINITE_RING_U8, 255, "direct_hip_tiled_finite_u8_gemm_mod255_v1"},
           Case{RNS8_FINITE_RING_U8, 256, "direct_hip_tiled_finite_u8_gemm_mod256_v1"},
       }) {
    CAPTURE(item.modulus);
    std::vector<uint8_t> A(static_cast<std::size_t>(m * lda), 0xa5);
    std::vector<uint8_t> B(static_cast<std::size_t>(k * ldb), 0x5a);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        A[static_cast<std::size_t>(row * lda + col)] =
            static_cast<uint8_t>((row * 97 + col * 53 + ((row + 3) ^ col) * 11) % item.modulus);
      }
    }
    for (int64_t row = 0; row < k; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        B[static_cast<std::size_t>(row * ldb + col)] =
            static_cast<uint8_t>((row * 47 + col * 89 + (row ^ (col + 7)) * 13) % item.modulus);
      }
    }

    auto cpu_desc = finite_desc(m, n, k, item.semantics, RNS8_BACKEND_CPU_REFERENCE, item.modulus);
    auto hip_desc = finite_desc(m, n, k, item.semantics, RNS8_BACKEND_HIP_DIRECT, item.modulus);
    rns8_plan* hip_plan = nullptr;
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_backend_info(hip_plan, &info) == RNS8_SUCCESS);
    CHECK(std::string(info.selected_kernel) == item.kernel);
    CHECK(std::string(info.autotune_key).find(std::string("kernel=") + item.kernel + ";") != std::string::npos);
    CHECK(std::string(info.isa_evidence) == "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide");

    std::vector<uint8_t> cpu_out(static_cast<std::size_t>(m * ldc), 0xcc);
    std::vector<uint8_t> hip_out(static_cast<std::size_t>(m * ldc), 0xcc);
    if (item.semantics == RNS8_FINITE_FIELD_U8) {
      REQUIRE(rns8_gemm_finite_field_u8_oneshot(
                  cpu, &cpu_desc, item.modulus, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) ==
              RNS8_SUCCESS);
      rns8::detail::hip_direct_timing_set_enabled(true);
      rns8::detail::hip_direct_timing_reset();
      REQUIRE(rns8_gemm_finite_field_u8_oneshot(
                  hip, &hip_desc, item.modulus, A.data(), lda, B.data(), ldb, hip_out.data(), ldc) ==
              RNS8_SUCCESS);
    } else {
      REQUIRE(rns8_gemm_finite_ring_u8_oneshot(
                  cpu, &cpu_desc, item.modulus, A.data(), lda, B.data(), ldb, cpu_out.data(), ldc) ==
              RNS8_SUCCESS);
      rns8::detail::hip_direct_timing_set_enabled(true);
      rns8::detail::hip_direct_timing_reset();
      REQUIRE(rns8_gemm_finite_ring_u8_oneshot(
                  hip, &hip_desc, item.modulus, A.data(), lda, B.data(), ldb, hip_out.data(), ldc) ==
              RNS8_SUCCESS);
    }
    const auto hip_events = rns8::detail::hip_direct_timing_snapshot();
    rns8::detail::hip_direct_timing_set_enabled(false);
    CHECK(has_timing_label(hip_events, "finite_pack_kernel"));
    CHECK(has_timing_label(hip_events, "finite_resident_gemm_kernel"));
    CHECK(has_timing_label(hip_events, "finite_export_kernel"));
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_out[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(cpu_out[static_cast<std::size_t>(row * ldc + col)] == 0xcc);
        CHECK(hip_out[static_cast<std::size_t>(row * ldc + col)] == 0xcc);
      }
    }

    rns8_destroy_plan(hip_plan);
  }
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP finite u8 persistent path preserves K-split semantics") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP finite u8 persistent K-split smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  std::vector<uint8_t> A(static_cast<std::size_t>(k), 255);
  std::vector<uint8_t> B(static_cast<std::size_t>(k), 255);
  uint8_t cpu_out = 0;
  uint8_t hip_out = 0;
  auto cpu_desc = finite_desc(1, 1, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_CPU_REFERENCE, 256);
  auto hip_desc = finite_desc(1, 1, k, RNS8_FINITE_RING_U8, RNS8_BACKEND_HIP_DIRECT, 256);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_workspace* cpu_workspace = nullptr;
  rns8_workspace* hip_workspace = nullptr;
  rns8_matrix* cpu_a = nullptr;
  rns8_matrix* cpu_b = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(cpu, cpu_plan, &cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, hip_plan, &hip_workspace) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(1, k, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  auto b_desc = matrix_desc(k, 1, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  auto c_desc = matrix_desc(1, 1, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &a_desc, &cpu_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &b_desc, &cpu_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  REQUIRE(rns8_pack_finite_u8(cpu, cpu_a, 256, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(cpu, cpu_b, 256, B.data(), 1, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_u8(cpu, cpu_plan, 256, cpu_a, cpu_b, cpu_c, cpu_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_finite_u8(cpu, cpu_plan, 256, cpu_c, &cpu_out, 1) == RNS8_SUCCESS);

  REQUIRE(rns8_pack_finite_u8(hip, hip_a, 256, A.data(), k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(hip, hip_b, 256, B.data(), 1, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_u8(hip, hip_plan, 256, hip_a, hip_b, hip_c, hip_workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_finite_u8(hip, hip_plan, 256, hip_c, &hip_out, 1) == RNS8_SUCCESS);
  CHECK(hip_out == cpu_out);

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_matrix(cpu_b);
  rns8_destroy_matrix(cpu_a);
  rns8_destroy_workspace(hip_workspace);
  rns8_destroy_workspace(cpu_workspace);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("private HIP ring GEMM rejects mismatched modulus metadata before launch") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP metadata-contract smoke");
  }

  constexpr int64_t bytes = 1;
  const int8_t a = 1;
  const int8_t b = 1;
  const int8_t sentinel = -77;
  int8_t out = 0;
  void* d_a = nullptr;
  void* d_b = nullptr;
  void* d_c = nullptr;
  REQUIRE(rns8::detail::hip_direct_allocate(0, bytes, &d_a) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, bytes, &d_b) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, bytes, &d_c) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_a, &a, bytes) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_b, &b, bytes) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, d_c, &sentinel, bytes) == RNS8_SUCCESS);

  CHECK(rns8_hip_direct_ring_gemm_i8_device(
            static_cast<const int8_t*>(d_a),
            static_cast<const int8_t*>(d_b),
            static_cast<int8_t*>(d_c),
            1,
            1,
            1,
            1,
            1,
            1,
            251,
            reciprocal_for_modulus(251),
            1,
            RNS8_DEFAULT_BOUNDED_PREFIX,
            RNS8_SAFE_INT32_K_BLOCK) != 0);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, &out, d_c, bytes) == RNS8_SUCCESS);
  CHECK(out == sentinel);

  const uint16_t valid_modulus = rns8::detail::kDefaultModuli[0];
  CHECK(rns8_hip_direct_ring_gemm_i8_device(
            static_cast<const int8_t*>(d_a),
            static_cast<const int8_t*>(d_b),
            static_cast<int8_t*>(d_c),
            1,
            1,
            1,
            1,
            1,
            1,
            valid_modulus,
            reciprocal_for_modulus(valid_modulus) + 1u,
            0,
            1,
            RNS8_SAFE_INT32_K_BLOCK) != 0);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, &out, d_c, bytes) == RNS8_SUCCESS);
  CHECK(out == sentinel);

  CHECK(rns8::detail::hip_direct_free(0, d_c) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, d_b) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, d_a) == RNS8_SUCCESS);
#else
  SKIP("direct HIP private metadata-contract smoke requires a HIP build");
#endif
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

  run_wrap64_resident_device_gemm(a_limbs, b_limbs, c_limbs, m, n, k);

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

  run_wrap64_resident_device_gemm(a_limbs, b_limbs, c_limbs, m, n, k);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected =
          rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), k, B.data(), n, row, col, k);
      CHECK(load_u64_limbs(c_limbs, row * n + col) == expected);
    }
  }
}

TEST_CASE("private HIP wrap64 byte-limb GEMM covers tile tails and signed-int8 correction") {
  if (!hip_available()) {
    SKIP("no HIP device available for private wrap64 HIP tile-tail smoke");
  }

  constexpr int64_t m = 31;
  constexpr int64_t n = 30;
  constexpr int64_t k = 33;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * n), 0);
  std::mt19937_64 rng(0x7772617036345f74ull);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      uint64_t value = rng();
      if ((row + col) % 7 == 0) {
        value = std::numeric_limits<uint64_t>::max();
      } else if ((row + col) % 7 == 1) {
        value = 0x8080808080808080ull;
      } else if ((row + col) % 7 == 2) {
        value = 0x7f807f807f807f80ull;
      }
      A[static_cast<std::size_t>(row * k + col)] = value;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      uint64_t value = rng();
      if ((row * 3 + col) % 11 == 0) {
        value = std::numeric_limits<uint64_t>::max();
      } else if ((row * 3 + col) % 11 == 1) {
        value = 0xfefdfcfbfaf9f8f7ull;
      } else if ((row * 3 + col) % 11 == 2) {
        value = 0x0102030405060708ull;
      }
      B[static_cast<std::size_t>(row * n + col)] = value;
    }
  }

  std::vector<uint8_t> a_limbs(static_cast<std::size_t>(m * k * 8));
  std::vector<uint8_t> b_limbs(static_cast<std::size_t>(k * n * 8));
  std::vector<uint8_t> c_limbs(static_cast<std::size_t>(m * n * 8), 0xa5);
  for (int64_t cell = 0; cell < m * k; ++cell) {
    store_u64_limbs(a_limbs, cell, A[static_cast<std::size_t>(cell)]);
  }
  for (int64_t cell = 0; cell < k * n; ++cell) {
    store_u64_limbs(b_limbs, cell, B[static_cast<std::size_t>(cell)]);
  }

  run_wrap64_resident_device_gemm(a_limbs, b_limbs, c_limbs, m, n, k);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected =
          rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), k, B.data(), n, row, col, k);
      CHECK(load_u64_limbs(c_limbs, row * n + col) == expected);
      CHECK(expected == rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), k, B.data(), n, row, col, k));
    }
  }
}

TEST_CASE("private HIP wrap64 helpers reject invalid contracts before launch") {
  std::vector<uint8_t> limbs(8, 0);
  uint64_t src = 0;
  uint64_t dst = 0;
  void* buffer = nullptr;
  std::size_t bytes = 0;
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, nullptr, limbs.data(), limbs.data(), 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, limbs.data(), nullptr, limbs.data(), 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(
            0, limbs.data(), limbs.data(), limbs.data(), -1, 1, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, limbs.data(), limbs.data(), limbs.data(), 0, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, limbs.data(), limbs.data(), limbs.data(), 1, 0, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, limbs.data(), limbs.data(), limbs.data(), 1, 1, 0) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, limbs.data(), limbs.data(), nullptr, 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(
            0,
            limbs.data(),
            limbs.data(),
            limbs.data(),
            static_cast<int64_t>(std::numeric_limits<int>::max()) + 1,
            1,
            1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(
            0,
            limbs.data(),
            limbs.data(),
            limbs.data(),
            1,
            1,
            static_cast<int64_t>(std::numeric_limits<int>::max()) + 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, nullptr, &buffer, &bytes, limbs.data(), 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, &src, nullptr, &bytes, limbs.data(), 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, &src, &buffer, nullptr, limbs.data(), 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, &src, &buffer, &bytes, limbs.data(), 1, 2, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, &src, &buffer, &bytes, limbs.data(), -1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, &src, &buffer, &bytes, limbs.data(), 0, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, &src, &buffer, &bytes, nullptr, 1, 1, 1) ==
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
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, limbs.data(), nullptr, &bytes, 1, 1, &dst, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, limbs.data(), &buffer, nullptr, 1, 1, &dst, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, limbs.data(), &buffer, &bytes, 1, 1, nullptr, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, limbs.data(), &buffer, &bytes, 1, 2, &dst, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, limbs.data(), &buffer, &bytes, -1, 1, &dst, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, limbs.data(), &buffer, &bytes, 0, 1, &dst, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, nullptr, &buffer, &bytes, 1, 1, &dst, 1) ==
        RNS8_INVALID_ARGUMENT);
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

TEST_CASE("private HIP wrap64 pack and export helpers preserve compact device byte-limb layout") {
  if (!hip_available()) {
    SKIP("no HIP device available for private wrap64 HIP pack/export layout smoke");
  }

  constexpr int64_t rows = 3;
  constexpr int64_t cols = 2;
  constexpr int64_t ld = 4;
  constexpr uint64_t sentinel = 0xccccccccccccccccull;
  std::vector<uint64_t> src(static_cast<std::size_t>(rows * ld), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> out(static_cast<std::size_t>(rows * ld), sentinel);
  src[0] = 0;
  src[1] = std::numeric_limits<uint64_t>::max();
  src[static_cast<std::size_t>(ld)] = 0x8080808080808080ull;
  src[static_cast<std::size_t>(ld + 1)] = 0x7f807f807f807f80ull;
  src[static_cast<std::size_t>(2 * ld)] = 0xfefdfcfbfaf9f8f7ull;
  src[static_cast<std::size_t>(2 * ld + 1)] = 0x0102030405060708ull;

  std::vector<uint8_t> expected_limbs(static_cast<std::size_t>(rows * cols * 8), 0);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      store_u64_limbs(expected_limbs, row * cols + col, src[static_cast<std::size_t>(row * ld + col)]);
    }
  }

  void* device_limbs = nullptr;
  void* upload = nullptr;
  std::size_t upload_bytes = 0;
  void* export_buffer = nullptr;
  std::size_t export_bytes = 0;
  const std::size_t compact_bytes = expected_limbs.size();
  REQUIRE(rns8::detail::hip_direct_allocate(0, compact_bytes, &device_limbs) == RNS8_SUCCESS);

  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              0,
              src.data(),
              &upload,
              &upload_bytes,
              device_limbs,
              rows,
              cols,
              ld) == RNS8_SUCCESS);
  REQUIRE(upload != nullptr);
  REQUIRE(upload_bytes >= static_cast<std::size_t>(rows * ld * static_cast<int64_t>(sizeof(uint64_t))));
  std::vector<uint8_t> device_copy(compact_bytes, 0);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, device_copy.data(), device_limbs, compact_bytes) ==
          RNS8_SUCCESS);
  CHECK(device_copy == expected_limbs);

  void* first_upload = upload;
  const std::size_t first_upload_bytes = upload_bytes;
  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              0,
              src.data(),
              &upload,
              &upload_bytes,
              device_limbs,
              rows,
              cols,
              ld) == RNS8_SUCCESS);
  CHECK(upload == first_upload);
  CHECK(upload_bytes == first_upload_bytes);

  std::fill(src.begin(), src.end(), 0x1111111111111111ull);
  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              0,
              device_limbs,
              &export_buffer,
              &export_bytes,
              rows,
              cols,
              out.data(),
              ld) == RNS8_SUCCESS);
  REQUIRE(export_buffer != nullptr);
  REQUIRE(export_bytes >= static_cast<std::size_t>(rows * cols * static_cast<int64_t>(sizeof(uint64_t))));
  void* first_export = export_buffer;
  const std::size_t first_export_bytes = export_bytes;

  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      CHECK(out[static_cast<std::size_t>(row * ld + col)] ==
            load_u64_limbs(expected_limbs, row * cols + col));
    }
    for (int64_t col = cols; col < ld; ++col) {
      CHECK(out[static_cast<std::size_t>(row * ld + col)] == sentinel);
    }
  }

  std::fill(out.begin(), out.end(), sentinel);
  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              0,
              device_limbs,
              &export_buffer,
              &export_bytes,
              rows,
              cols,
              out.data(),
              ld) == RNS8_SUCCESS);
  CHECK(export_buffer == first_export);
  CHECK(export_bytes == first_export_bytes);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      CHECK(out[static_cast<std::size_t>(row * ld + col)] ==
            load_u64_limbs(expected_limbs, row * cols + col));
    }
    for (int64_t col = cols; col < ld; ++col) {
      CHECK(out[static_cast<std::size_t>(row * ld + col)] == sentinel);
    }
  }

  if (export_buffer) {
    CHECK(rns8::detail::hip_direct_free(0, export_buffer) == RNS8_SUCCESS);
  }
  if (upload) {
    CHECK(rns8::detail::hip_direct_free(0, upload) == RNS8_SUCCESS);
  }
  CHECK(rns8::detail::hip_direct_free(0, device_limbs) == RNS8_SUCCESS);
}

TEST_CASE("private HIP wrap64 pack and export helpers cover tile-tail padded layout reuse") {
  if (!hip_available()) {
    SKIP("no HIP device available for private wrap64 HIP padded tile-tail layout smoke");
  }

  constexpr int64_t rows = 17;
  constexpr int64_t cols = 19;
  constexpr int64_t ld = 23;
  constexpr uint64_t padding = 0xaaaaaaaaaaaaaaaaull;
  constexpr uint64_t sentinel = 0x6d6d6d6d6d6d6d6dull;
  std::vector<uint64_t> src(static_cast<std::size_t>(rows * ld), padding);
  std::vector<uint64_t> out(static_cast<std::size_t>(rows * ld), sentinel);
  std::vector<uint8_t> expected_limbs(static_cast<std::size_t>(rows * cols * 8), 0);

  auto fill_src = [&](uint64_t salt) {
    std::mt19937_64 rng(0x7772617036345f70ull ^ salt);
    std::fill(src.begin(), src.end(), padding ^ salt);
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t col = 0; col < cols; ++col) {
        uint64_t value = rng() ^ (salt + static_cast<uint64_t>(row * 257 + col * 19));
        if ((row + col) % 11 == 0) {
          value = std::numeric_limits<uint64_t>::max();
        } else if ((row + col) % 11 == 1) {
          value = 0;
        } else if ((row + col) % 11 == 2) {
          value = 0x8080808080808080ull;
        } else if ((row + col) % 11 == 3) {
          value = 0xfefdfcfbfaf9f8f7ull;
        }
        src[static_cast<std::size_t>(row * ld + col)] = value;
      }
    }
  };

  auto refresh_expected_limbs = [&]() {
    std::fill(expected_limbs.begin(), expected_limbs.end(), 0);
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t col = 0; col < cols; ++col) {
        store_u64_limbs(expected_limbs, row * cols + col, src[static_cast<std::size_t>(row * ld + col)]);
      }
    }
  };

  void* device_limbs = nullptr;
  void* upload = nullptr;
  std::size_t upload_bytes = 0;
  void* export_buffer = nullptr;
  std::size_t export_bytes = 0;
  const std::size_t compact_bytes = expected_limbs.size();
  REQUIRE(rns8::detail::hip_direct_allocate(0, compact_bytes, &device_limbs) == RNS8_SUCCESS);

  fill_src(0x1111);
  refresh_expected_limbs();
  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              0,
              src.data(),
              &upload,
              &upload_bytes,
              device_limbs,
              rows,
              cols,
              ld) == RNS8_SUCCESS);
  REQUIRE(upload != nullptr);
  REQUIRE(upload_bytes >= static_cast<std::size_t>(rows * ld * static_cast<int64_t>(sizeof(uint64_t))));
  std::vector<uint8_t> device_copy(compact_bytes, 0);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, device_copy.data(), device_limbs, compact_bytes) ==
          RNS8_SUCCESS);
  CHECK(device_copy == expected_limbs);
  void* first_upload = upload;
  const std::size_t first_upload_bytes = upload_bytes;

  fill_src(0x2222);
  refresh_expected_limbs();
  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              0,
              src.data(),
              &upload,
              &upload_bytes,
              device_limbs,
              rows,
              cols,
              ld) == RNS8_SUCCESS);
  CHECK(upload == first_upload);
  CHECK(upload_bytes == first_upload_bytes);
  std::fill(device_copy.begin(), device_copy.end(), 0);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, device_copy.data(), device_limbs, compact_bytes) ==
          RNS8_SUCCESS);
  CHECK(device_copy == expected_limbs);

  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              0,
              device_limbs,
              &export_buffer,
              &export_bytes,
              rows,
              cols,
              out.data(),
              ld) == RNS8_SUCCESS);
  REQUIRE(export_buffer != nullptr);
  REQUIRE(export_bytes >= static_cast<std::size_t>(rows * cols * static_cast<int64_t>(sizeof(uint64_t))));
  void* first_export = export_buffer;
  const std::size_t first_export_bytes = export_bytes;

  auto check_export = [&]() {
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t col = 0; col < cols; ++col) {
        CHECK(out[static_cast<std::size_t>(row * ld + col)] ==
              load_u64_limbs(expected_limbs, row * cols + col));
      }
      for (int64_t col = cols; col < ld; ++col) {
        CHECK(out[static_cast<std::size_t>(row * ld + col)] == sentinel);
      }
    }
  };
  check_export();

  std::fill(out.begin(), out.end(), sentinel);
  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              0,
              device_limbs,
              &export_buffer,
              &export_bytes,
              rows,
              cols,
              out.data(),
              ld) == RNS8_SUCCESS);
  CHECK(export_buffer == first_export);
  CHECK(export_bytes == first_export_bytes);
  check_export();

  if (export_buffer) {
    CHECK(rns8::detail::hip_direct_free(0, export_buffer) == RNS8_SUCCESS);
  }
  if (upload) {
    CHECK(rns8::detail::hip_direct_free(0, upload) == RNS8_SUCCESS);
  }
  CHECK(rns8::detail::hip_direct_free(0, device_limbs) == RNS8_SUCCESS);
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
  rns8_plan_backend_info hip_info{};
  hip_info.struct_size = sizeof(hip_info);
  hip_info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(hip_plan, &hip_info) == RNS8_SUCCESS);
  CHECK(std::string(hip_info.selected_kernel) == "direct_hip_wrap64_byte_gemm36_tiled_2d_v3");
  CHECK(std::string(hip_info.isa_evidence) ==
        "wrap64_byte_gemm36_isa_gate_no_variable_divide_no_matrix_engine");
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
  CHECK_FALSE(hip_a->host_byte_limbs_current);
  CHECK_FALSE(hip_b->host_byte_limbs_current);
  CHECK_FALSE(hip_out->host_byte_limbs_current);
  CHECK_FALSE(hip_a->device_byte_limbs_current);
  CHECK_FALSE(hip_b->device_byte_limbs_current);
  CHECK_FALSE(hip_out->device_byte_limbs_current);
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
  CHECK(has_timing_label(hip_gemm_events, "wrap64_byte_gemm36_tiled_2d_kernel"));
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
          rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
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
  rns8::detail::hip_direct_allocation_counters_reset();
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
  CHECK_FALSE(hip_a->host_byte_limbs_current);
  CHECK_FALSE(hip_b->host_byte_limbs_current);
  CHECK_FALSE(hip_out->host_byte_limbs_current);
  CHECK_FALSE(hip_a->device_byte_limbs_current);
  CHECK_FALSE(hip_b->device_byte_limbs_current);
  CHECK_FALSE(hip_out->device_byte_limbs_current);
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
  const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  REQUIRE(warmed_allocations.allocate_calls > 0);
  REQUIRE(warmed_allocations.allocated_bytes > 0);
  const auto warmed_snapshot = capture_wrap64_resident_snapshot(hip_a, hip_b, hip_out, hip_workspace);

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
  const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
  CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
  CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
  check_wrap64_resident_snapshot_unchanged(warmed_snapshot, hip_a, hip_b, hip_out, hip_workspace);

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

TEST_CASE("direct HIP public wrap64 resident path is device-current for large padded tiles") {
  if (!hip_available()) {
    SKIP("no HIP device available for public wrap64 HIP resident stress smoke");
  }

  constexpr int64_t m = 33;
  constexpr int64_t n = 35;
  constexpr int64_t k = 65;
  constexpr int64_t lda = 71;
  constexpr int64_t ldb = 43;
  constexpr int64_t ldc = 39;
  constexpr uint64_t a_pad = 0xaaaaaaaaaaaaaaaaull;
  constexpr uint64_t b_pad = 0xbbbbbbbbbbbbbbbbull;
  constexpr uint64_t c_sentinel = 0x9191919191919191ull;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), a_pad);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), b_pad);
  std::vector<uint64_t> latest_A;
  std::vector<uint64_t> latest_B;
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), c_sentinel);

  auto fill_inputs = [&](uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::fill(A.begin(), A.end(), a_pad);
    std::fill(B.begin(), B.end(), b_pad);
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
    A[3] = 0x7f807f807f807f80ull;
    A[static_cast<std::size_t>((m / 2) * lda + (k / 2))] = 0xfefdfcfbfaf9f8f7ull;
    A[static_cast<std::size_t>((m - 1) * lda + (k - 1))] = std::numeric_limits<uint64_t>::max() - 1;
    B[0] = std::numeric_limits<uint64_t>::max();
    B[1] = 1;
    B[2] = 0x0102030405060708ull;
    B[3] = 0x8080808080808080ull;
    B[static_cast<std::size_t>((k / 2) * ldb + (n / 2))] = 0x7f807f807f807f80ull;
    B[static_cast<std::size_t>((k - 1) * ldb + (n - 1))] = std::numeric_limits<uint64_t>::max();
  };

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
  CHECK(hip_a->hip_residues == nullptr);
  CHECK(hip_b->hip_residues == nullptr);
  CHECK(hip_out->hip_residues == nullptr);
  void* hip_a_bytes = hip_a->hip_byte_limbs;
  void* hip_b_bytes = hip_b->hip_byte_limbs;
  void* hip_out_bytes = hip_out->hip_byte_limbs;

  rns8::detail::hip_direct_allocation_counters_reset();
  auto run_persistent = [&](uint64_t seed, uint64_t a_version, uint64_t b_version) {
    fill_inputs(seed);
    latest_A = A;
    latest_B = B;
    std::fill(cpu_c.begin(), cpu_c.end(), c_sentinel);
    std::fill(hip_c.begin(), hip_c.end(), c_sentinel);

    REQUIRE(rns8_pack_u64(cpu, cpu_a, A.data(), lda, a_version) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(cpu, cpu_b, B.data(), ldb, b_version) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(hip, hip_a, A.data(), lda, a_version) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(hip, hip_b, B.data(), ldb, b_version) == RNS8_SUCCESS);
    CHECK(hip_a->device_byte_limbs_current);
    CHECK(hip_b->device_byte_limbs_current);
    CHECK_FALSE(hip_a->host_byte_limbs_current);
    CHECK_FALSE(hip_b->host_byte_limbs_current);

    std::fill(A.begin(), A.end(), 0x1111111111111111ull);
    std::fill(B.begin(), B.end(), 0x2222222222222222ull);
    REQUIRE(rns8_gemm_wrap_u64(cpu, cpu_plan, cpu_a, cpu_b, cpu_out, cpu_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_wrap_u64(hip, hip_plan, hip_a, hip_b, hip_out, hip_workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_wrap_u64(cpu, cpu_plan, cpu_out, cpu_c.data(), ldc) == RNS8_SUCCESS);
    REQUIRE(rns8_export_wrap_u64(hip, hip_plan, hip_out, hip_c.data(), ldc) == RNS8_SUCCESS);
    CHECK(hip_c == cpu_c);
    CHECK(hip_a->source_version == a_version);
    CHECK(hip_b->source_version == b_version);
    CHECK(hip_a->hip_byte_limbs == hip_a_bytes);
    CHECK(hip_b->hip_byte_limbs == hip_b_bytes);
    CHECK(hip_out->hip_byte_limbs == hip_out_bytes);
    CHECK(hip_out->device_byte_limbs_current);
    CHECK_FALSE(hip_out->host_byte_limbs_current);

    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const uint64_t expected =
            rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(
                latest_A.data(), lda, latest_B.data(), ldb, row, col, k);
        const std::size_t out_index = static_cast<std::size_t>(row * ldc + col);
        CHECK(hip_c[out_index] == expected);
      }
      for (int64_t col = n; col < ldc; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] == c_sentinel);
      }
    }
  };

  run_persistent(0x7772617036345f31ull, 301, 302);
  void* hip_a_upload = hip_a->hip_upload_buffer;
  void* hip_b_upload = hip_b->hip_upload_buffer;
  void* hip_export = hip_out->hip_export_buffer;
  const std::size_t hip_a_upload_bytes = hip_a->hip_upload_bytes;
  const std::size_t hip_b_upload_bytes = hip_b->hip_upload_bytes;
  const std::size_t hip_export_bytes = hip_out->hip_export_bytes;
  const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  REQUIRE(hip_a_upload != nullptr);
  REQUIRE(hip_b_upload != nullptr);
  REQUIRE(hip_export != nullptr);
  REQUIRE(warmed_allocations.allocate_calls > 0);
  REQUIRE(warmed_allocations.allocated_bytes > 0);
  const auto warmed_snapshot = capture_wrap64_resident_snapshot(hip_a, hip_b, hip_out, hip_workspace);

  run_persistent(0x7772617036345f32ull, 401, 402);
  CHECK(hip_a->hip_upload_buffer == hip_a_upload);
  CHECK(hip_b->hip_upload_buffer == hip_b_upload);
  CHECK(hip_out->hip_export_buffer == hip_export);
  CHECK(hip_a->hip_upload_bytes == hip_a_upload_bytes);
  CHECK(hip_b->hip_upload_bytes == hip_b_upload_bytes);
  CHECK(hip_out->hip_export_bytes == hip_export_bytes);
  const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
  CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
  CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
  check_wrap64_resident_snapshot_unchanged(warmed_snapshot, hip_a, hip_b, hip_out, hip_workspace);

  std::vector<uint64_t> cpu_oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  std::vector<uint64_t> hip_oneshot(static_cast<std::size_t>(m * ldc), c_sentinel);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(
              cpu, &cpu_desc, latest_A.data(), lda, latest_B.data(), ldb, cpu_oneshot.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_wrap_u64_oneshot(
              hip, &hip_desc, latest_A.data(), lda, latest_B.data(), ldb, hip_oneshot.data(), ldc) == RNS8_SUCCESS);
  CHECK(cpu_oneshot == cpu_c);
  CHECK(hip_oneshot == hip_c);

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
      const uint64_t expected =
          rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k);
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

  rns8::detail::hip_direct_allocation_counters_reset();
  const auto rejected_start = rns8::detail::hip_direct_allocation_counters_snapshot();
  auto check_reject_allocations_unchanged = [&] {
    const auto rejected_now = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(rejected_now.allocate_calls == rejected_start.allocate_calls);
    CHECK(rejected_now.free_calls == rejected_start.free_calls);
    CHECK(rejected_now.allocated_bytes == rejected_start.allocated_bytes);
  };
  auto check_invalid_wrap_descriptor = [&](const rns8_gemm_desc& bad_desc) {
    rns8_plan* rejected_plan = nullptr;
    CHECK(rns8_gemm_wrap_u64_oneshot(hip, &bad_desc, A, k, B, n, C, n) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_create_plan(hip, &bad_desc, &rejected_plan) == RNS8_INVALID_ARGUMENT);
    CHECK(rejected_plan == nullptr);
    if (rejected_plan) {
      rns8_destroy_plan(rejected_plan);
    }
    check_reject_allocations_unchanged();
  };
  auto check_invalid_wrap_access = [&](int64_t bad_lda, int64_t bad_ldb, int64_t bad_ldc) {
    C[0] = 0x123456789abcdef0ull;
    CHECK(rns8_gemm_wrap_u64_oneshot(hip, &desc, A, bad_lda, B, bad_ldb, C, bad_ldc) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(C[0] == 0x123456789abcdef0ull);
    check_reject_allocations_unchanged();
  };
  check_invalid_wrap_access(0, n, n);
  check_invalid_wrap_access(k, 0, n);
  check_invalid_wrap_access(k, n, 0);

  auto bounded_looking = desc;
  bounded_looking.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  bounded_looking.bound = std::numeric_limits<uint64_t>::max();
  check_invalid_wrap_descriptor(bounded_looking);

  auto prefixed = desc;
  prefixed.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  check_invalid_wrap_descriptor(prefixed);

  auto bound_only = desc;
  bound_only.bound = 1;
  check_invalid_wrap_descriptor(bound_only);

  auto flagged = desc;
  flagged.flags = 1;
  check_invalid_wrap_descriptor(flagged);

  auto bad_tile_m = desc;
  bad_tile_m.tile_m = 32;
  check_invalid_wrap_descriptor(bad_tile_m);

  auto bad_tile_n = desc;
  bad_tile_n.tile_n = 96;
  check_invalid_wrap_descriptor(bad_tile_n);

  const uint64_t tile_bound = 1;
  auto tile_bounded = desc;
  tile_bounded.tile_bounds = &tile_bound;
  tile_bounded.tile_bounds_count = 1;
  check_invalid_wrap_descriptor(tile_bounded);

  auto tile_count_only = desc;
  tile_count_only.tile_bounds_count = 1;
  check_invalid_wrap_descriptor(tile_count_only);

  auto per_tile_wrap = desc;
  per_tile_wrap.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  per_tile_wrap.tile_m = 64;
  per_tile_wrap.tile_n = 64;
  per_tile_wrap.tile_bounds = &tile_bound;
  per_tile_wrap.tile_bounds_count = 1;
  check_invalid_wrap_descriptor(per_tile_wrap);

  auto matrix = matrix_desc(m, n, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  auto check_invalid_wrap_matrix_descriptor = [&](const rns8_matrix_desc& bad_desc) {
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(hip, &bad_desc, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
    if (storage) {
      rns8_destroy_matrix(storage);
    }
    check_reject_allocations_unchanged();
  };
  auto bounded_matrix_descriptor = matrix;
  bounded_matrix_descriptor.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  check_invalid_wrap_matrix_descriptor(bounded_matrix_descriptor);
  auto prefixed_matrix_descriptor = matrix;
  prefixed_matrix_descriptor.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  check_invalid_wrap_matrix_descriptor(prefixed_matrix_descriptor);
  auto flagged_matrix_descriptor = matrix;
  flagged_matrix_descriptor.flags = 1;
  check_invalid_wrap_matrix_descriptor(flagged_matrix_descriptor);
  auto bad_ld_matrix_descriptor = matrix;
  bad_ld_matrix_descriptor.logical_ld = -1;
  check_invalid_wrap_matrix_descriptor(bad_ld_matrix_descriptor);
  auto bad_tile_matrix_descriptor = matrix;
  bad_tile_matrix_descriptor.tile_m = 32;
  check_invalid_wrap_matrix_descriptor(bad_tile_matrix_descriptor);

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
  CHECK_FALSE(wrap_a->host_byte_limbs_current);
  CHECK_FALSE(wrap_b->host_byte_limbs_current);
  CHECK_FALSE(wrap_c->host_byte_limbs_current);
  CHECK_FALSE(wrap_a->device_byte_limbs_current);
  CHECK_FALSE(wrap_b->device_byte_limbs_current);
  CHECK_FALSE(wrap_c->device_byte_limbs_current);
  const int64_t signed_A[] = {-1};
  uint64_t limbs[] = {0};
  int64_t signed_out[] = {0};
  CHECK(rns8_pack_i64(hip, wrap_a, signed_A, k, 0) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_i64(hip, valid_plan, wrap_c, signed_out, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_u64(hip, valid_plan, wrap_c, C, n) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, valid_plan, wrap_c, limbs, n, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, valid_plan, wrap_c, limbs, n, 1) == RNS8_INVALID_ARGUMENT);

  REQUIRE(rns8_pack_u64(hip, wrap_a, A, k, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, wrap_b, B, n, 2) == RNS8_SUCCESS);

  wrap_a->host_byte_limbs_current = true;
  CHECK(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
  wrap_a->host_byte_limbs_current = false;

  auto bounded_workspace_desc = unsigned_desc(m, n, k, 2, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* bounded_plan = nullptr;
  rns8_workspace* bounded_workspace = nullptr;
  REQUIRE(rns8_create_plan(hip, &bounded_workspace_desc, &bounded_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, bounded_plan, &bounded_workspace) == RNS8_SUCCESS);
  CHECK(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, bounded_workspace) == RNS8_INVALID_ARGUMENT);
  rns8_destroy_workspace(bounded_workspace);
  rns8_destroy_plan(bounded_plan);

  wrap_a->host_residues_current = true;
  CHECK(rns8_pack_u64(hip, wrap_a, A, k, 3) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
  wrap_a->host_residues_current = false;

  wrap_b->device_residues_current = true;
  CHECK(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
  wrap_b->device_residues_current = false;

  wrap_a->host_byte_limbs_current = true;
  wrap_a->device_byte_limbs_current = false;
  CHECK(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
  wrap_a->host_byte_limbs_current = false;
  wrap_a->device_byte_limbs_current = true;

  REQUIRE(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(hip, valid_plan, wrap_c, C, n) == RNS8_SUCCESS);
  CHECK(C[0] == rns8::detail::wrap64_byte_limb_gemm_cell(A, k, B, n, 0, 0, k));

  rns8_plan_tile_schedule_entry stale_entry{};
  stale_entry.struct_size = sizeof(stale_entry);
  stale_entry.abi_version = RNS8_ABI_VERSION;
  stale_entry.required_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  stale_entry.selected_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  const uint32_t original_tile_m = valid_plan->desc.tile_m;
  const uint32_t original_tile_n = valid_plan->desc.tile_n;
  auto check_rejects_plan_mutation = [&](auto mutate, auto restore) {
    mutate();
    CHECK(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8_export_wrap_u64(hip, valid_plan, wrap_c, C, n) == RNS8_INVALID_ARGUMENT);
    restore();
  };
  check_rejects_plan_mutation(
      [&] { valid_plan->desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED; },
      [&] { valid_plan->desc.bound_kind = RNS8_BOUND_NONE; });
  check_rejects_plan_mutation([&] { valid_plan->desc.bound = 1; }, [&] { valid_plan->desc.bound = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX; },
      [&] { valid_plan->desc.max_prefix = 0; });
  check_rejects_plan_mutation([&] { valid_plan->desc.flags = 1; }, [&] { valid_plan->desc.flags = 0; });
  check_rejects_plan_mutation([&] { valid_plan->desc.tile_m = 64; }, [&] { valid_plan->desc.tile_m = original_tile_m; });
  check_rejects_plan_mutation([&] { valid_plan->desc.tile_n = 64; }, [&] { valid_plan->desc.tile_n = original_tile_n; });
  check_rejects_plan_mutation([&] { valid_plan->prefix = 1; }, [&] { valid_plan->prefix = 0; });
  check_rejects_plan_mutation([&] { valid_plan->modulus_product = 1; }, [&] { valid_plan->modulus_product = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->desc.tile_bounds = &tile_bound; },
      [&] { valid_plan->desc.tile_bounds = nullptr; });
  check_rejects_plan_mutation([&] { valid_plan->desc.tile_bounds_count = 1; }, [&] { valid_plan->desc.tile_bounds_count = 0; });
  check_rejects_plan_mutation([&] { valid_plan->tile_bounds.push_back(1); }, [&] { valid_plan->tile_bounds.clear(); });
  check_rejects_plan_mutation(
      [&] { valid_plan->tile_schedule.push_back(stale_entry); },
      [&] { valid_plan->tile_schedule.clear(); });
  check_rejects_plan_mutation([&] { ++valid_plan->schedule_tile_rows; }, [&] { --valid_plan->schedule_tile_rows; });
  check_rejects_plan_mutation([&] { ++valid_plan->schedule_tile_cols; }, [&] { --valid_plan->schedule_tile_cols; });
  check_rejects_plan_mutation([&] { ++valid_plan->schedule_tile_count; }, [&] { --valid_plan->schedule_tile_count; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_min_required_prefix = 1; },
      [&] { valid_plan->schedule_min_required_prefix = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_max_required_prefix = 1; },
      [&] { valid_plan->schedule_max_required_prefix = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_min_selected_prefix = 1; },
      [&] { valid_plan->schedule_min_selected_prefix = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_max_selected_prefix = 1; },
      [&] { valid_plan->schedule_max_selected_prefix = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_prefix_group_count = 1; },
      [&] { valid_plan->schedule_prefix_group_count = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_range_bit_length = 64; },
      [&] { valid_plan->schedule_range_bit_length = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_adaptive_prefix_active = 1; },
      [&] { valid_plan->schedule_adaptive_prefix_active = 0; });
  check_rejects_plan_mutation(
      [&] { valid_plan->schedule_adaptive_skip_active = 1; },
      [&] { valid_plan->schedule_adaptive_skip_active = 0; });
  check_rejects_plan_mutation([&] { valid_plan->schedule_flags = 1; }, [&] { valid_plan->schedule_flags = 0; });

  REQUIRE(rns8_gemm_wrap_u64(hip, valid_plan, wrap_a, wrap_b, wrap_c, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_wrap_u64(hip, valid_plan, wrap_c, C, n) == RNS8_SUCCESS);
  CHECK(C[0] == rns8::detail::wrap64_byte_limb_gemm_cell(A, k, B, n, 0, 0, k));

  wrap_c->host_residues_current = true;
  CHECK(rns8_export_wrap_u64(hip, valid_plan, wrap_c, C, n) == RNS8_INVALID_ARGUMENT);
  wrap_c->host_residues_current = false;

  wrap_c->host_byte_limbs_current = true;
  CHECK(rns8_export_wrap_u64(hip, valid_plan, wrap_c, C, n) == RNS8_INVALID_ARGUMENT);
  wrap_c->host_byte_limbs_current = false;

  wrap_c->host_byte_limbs_current = true;
  wrap_c->device_byte_limbs_current = false;
  CHECK(rns8_export_wrap_u64(hip, valid_plan, wrap_c, C, n) == RNS8_INVALID_ARGUMENT);
  wrap_c->host_byte_limbs_current = false;
  wrap_c->device_byte_limbs_current = true;

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
  CHECK(rns8_export_wrap_u64(hip, valid_plan, residue_c, C, n) == RNS8_INVALID_ARGUMENT);

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

  {
    const int64_t rows = 2;
    const int64_t cols = 3;
    const int64_t ld = 4;
    const std::vector<int64_t> src = {
        std::numeric_limits<int64_t>::min(),
        -9223372036854775807LL,
        -257,
        999,
        -1,
        0,
        std::numeric_limits<int64_t>::max(),
        999};
    auto desc = matrix_desc(rows, cols, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    rns8_matrix* cpu_matrix = nullptr;
    rns8_matrix* hip_matrix = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &desc, &cpu_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &desc, &hip_matrix) == RNS8_SUCCESS);
    CHECK(rns8_pack_i64(cpu, cpu_matrix, src.data(), ld, 23) == RNS8_SUCCESS);
    CHECK(rns8_pack_i64(hip, hip_matrix, src.data(), ld, 23) == RNS8_SUCCESS);
    CHECK(rns8::detail::hip_direct_copy_device_to_host(
              hip_matrix->hip_device_id,
              hip_matrix->residues.data(),
              hip_matrix->hip_residues,
              hip_matrix->hip_residue_bytes) == RNS8_SUCCESS);
    CHECK(hip_matrix->residues == cpu_matrix->residues);
    CHECK(hip_matrix->source_version == 23);
    rns8_destroy_matrix(hip_matrix);
    rns8_destroy_matrix(cpu_matrix);
  }

  {
    const int64_t rows = 2;
    const int64_t cols = 3;
    const int64_t ld = 4;
    const std::vector<uint64_t> src = {
        0,
        1,
        257,
        999,
        std::numeric_limits<uint64_t>::max() - 1,
        std::numeric_limits<uint64_t>::max(),
        1234567890123456789ull,
        999};
    auto desc = matrix_desc(rows, cols, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    rns8_matrix* cpu_matrix = nullptr;
    rns8_matrix* hip_matrix = nullptr;
    REQUIRE(rns8_create_matrix(cpu, &desc, &cpu_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &desc, &hip_matrix) == RNS8_SUCCESS);
    CHECK(rns8_pack_u64(cpu, cpu_matrix, src.data(), ld, 29) == RNS8_SUCCESS);
    CHECK(rns8_pack_u64(hip, hip_matrix, src.data(), ld, 29) == RNS8_SUCCESS);
    CHECK(rns8::detail::hip_direct_copy_device_to_host(
              hip_matrix->hip_device_id,
              hip_matrix->residues.data(),
              hip_matrix->hip_residues,
              hip_matrix->hip_residue_bytes) == RNS8_SUCCESS);
    CHECK(hip_matrix->residues == cpu_matrix->residues);
    CHECK(hip_matrix->source_version == 29);
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
  CHECK_FALSE(a_matrix->host_residues_current);
  CHECK_FALSE(a_matrix->device_residues_current);
  CHECK_FALSE(b_matrix->host_residues_current);
  CHECK_FALSE(b_matrix->device_residues_current);
  CHECK_FALSE(c_matrix->host_residues_current);
  CHECK_FALSE(c_matrix->device_residues_current);
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

TEST_CASE("AUTO direct-HIP RNS GEMM converts current native bounded inputs to RNS residues") {
  if (!hip_available()) {
    SKIP("no HIP device available for AUTO native-to-RNS conversion smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context_options auto_options{};
  auto_options.struct_size = sizeof(auto_options);
  auto_options.abi_version = RNS8_ABI_VERSION;
  auto_options.requested_backend = RNS8_BACKEND_AUTO;
  rns8_context* auto_ctx = nullptr;
  REQUIRE(rns8_create_context(0, &auto_options, &auto_ctx) == RNS8_SUCCESS);
  REQUIRE(auto_ctx->auto_backend_selection);
  REQUIRE(auto_ctx->backend == RNS8_BACKEND_HIP_DIRECT);

  {
    const int64_t m = 2;
    const int64_t n = 3;
    const int64_t k = 4;
    const std::vector<int64_t> A = {5, -7, 11, 13, -17, 19, 23, -29};
    const std::vector<int64_t> B = {3, -5, 7, 11, 13, -17, 19, 23, -29, 31, 37, -41};
    std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * n), 0);
    std::vector<int64_t> auto_c(static_cast<std::size_t>(m * n), 0);

    auto cpu_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_CPU_REFERENCE);
    auto auto_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_HIP_DIRECT);
    REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), n) == RNS8_SUCCESS);

    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(auto_ctx, &auto_desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(auto_ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(auto_ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(auto_ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(auto_ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

    REQUIRE(rns8_pack_i64(auto_ctx, a_matrix, A.data(), k, 101) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(auto_ctx, b_matrix, B.data(), n, 202) == RNS8_SUCCESS);
    REQUIRE(a_matrix->device_native_current);
    REQUIRE(b_matrix->device_native_current);
    REQUIRE(a_matrix->device_residues_current);
    REQUIRE(b_matrix->device_residues_current);

    a_matrix->device_residues_current = false;
    b_matrix->device_residues_current = false;
    REQUIRE(rns8_gemm_rns(auto_ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    CHECK(a_matrix->device_residues_current);
    CHECK(b_matrix->device_residues_current);
    CHECK(a_matrix->device_native_current);
    CHECK(b_matrix->device_native_current);
    CHECK(c_matrix->device_residues_current);
    REQUIRE(rns8_export_i64(auto_ctx, plan, c_matrix, auto_c.data(), n) == RNS8_SUCCESS);
    CHECK(auto_c == cpu_c);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    const int64_t m = 2;
    const int64_t n = 2;
    const int64_t k = 3;
    const std::vector<uint64_t> A = {5, 7, 11, 13, 17, 19};
    const std::vector<uint64_t> B = {23, 29, 31, 37, 41, 43};
    std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * n), 0);
    std::vector<uint64_t> auto_c(static_cast<std::size_t>(m * n), 0);

    auto cpu_desc = unsigned_desc(m, n, k, 10000, RNS8_BACKEND_CPU_REFERENCE);
    auto auto_desc = unsigned_desc(m, n, k, 10000, RNS8_BACKEND_HIP_DIRECT);
    REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), n) == RNS8_SUCCESS);

    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(auto_ctx, &auto_desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(auto_ctx, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(auto_ctx, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(auto_ctx, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(auto_ctx, &c_desc, &c_matrix) == RNS8_SUCCESS);

    REQUIRE(rns8_pack_u64(auto_ctx, a_matrix, A.data(), k, 301) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(auto_ctx, b_matrix, B.data(), n, 302) == RNS8_SUCCESS);
    REQUIRE(a_matrix->device_native_current);
    REQUIRE(b_matrix->device_native_current);
    REQUIRE(a_matrix->device_residues_current);
    REQUIRE(b_matrix->device_residues_current);

    a_matrix->device_residues_current = false;
    b_matrix->device_residues_current = false;
    REQUIRE(rns8_gemm_rns(auto_ctx, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    CHECK(a_matrix->device_residues_current);
    CHECK(b_matrix->device_residues_current);
    CHECK(a_matrix->device_native_current);
    CHECK(b_matrix->device_native_current);
    CHECK(c_matrix->device_residues_current);
    REQUIRE(rns8_export_u64(auto_ctx, plan, c_matrix, auto_c.data(), n) == RNS8_SUCCESS);
    CHECK(auto_c == cpu_c);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(auto_ctx);
  rns8_destroy_context(cpu);
}

TEST_CASE("vector ALU backend keeps native bounded storage through persistent GEMM") {
  if (!hip_available()) {
    SKIP("no HIP device available for vector ALU persistent bounded smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* vector = create_context(RNS8_BACKEND_HIP_VECTOR_ALU_INT64);

  {
    const int64_t m = 2;
    const int64_t n = 3;
    const int64_t k = 4;
    const std::vector<int64_t> A = {5, -7, 11, 13, -17, 19, 23, -29};
    const std::vector<int64_t> B = {3, -5, 7, 11, 13, -17, 19, 23, -29, 31, 37, -41};
    std::vector<int64_t> cpu_c(static_cast<std::size_t>(m * n), 0);
    std::vector<int64_t> vector_c(static_cast<std::size_t>(m * n), 0);

    auto cpu_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_CPU_REFERENCE);
    auto vector_desc = signed_desc(m, n, k, 100000, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    REQUIRE(rns8_gemm_i64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), n) == RNS8_SUCCESS);

    rns8::detail::hip_direct_allocation_counters_reset();
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(vector, &vector_desc, &plan) == RNS8_SUCCESS);
    rns8_plan_packing_info packing_info{};
    packing_info.struct_size = sizeof(packing_info);
    packing_info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_packing_info(plan, &packing_info) == RNS8_SUCCESS);
    CHECK(packing_info.input_domain == RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64);
    CHECK(packing_info.output_domain == RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64);
    CHECK(packing_info.output_host_current == 0);
    CHECK(packing_info.output_device_current == 1);
    CHECK((packing_info.next_op_flags & RNS8_NEXT_OP_NATIVE_GEMM) != 0);
    CHECK((packing_info.next_op_flags & RNS8_NEXT_OP_NATIVE_TO_RNS_CONVERTIBLE) != 0);
    CHECK(std::string(packing_info.output_domain_name) == "native_i64_u64_current");
    CHECK(std::string(packing_info.next_op_hint).find("native-to-RNS conversion") != std::string::npos);
    REQUIRE(rns8_create_workspace(vector, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(vector, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &c_desc, &c_matrix) == RNS8_SUCCESS);

    REQUIRE(a_matrix->hip_native_i64 != nullptr);
    REQUIRE(b_matrix->hip_native_i64 != nullptr);
    REQUIRE(c_matrix->hip_native_i64 != nullptr);
    CHECK(a_matrix->hip_residues == nullptr);
    CHECK(b_matrix->hip_residues == nullptr);
    CHECK(c_matrix->hip_residues == nullptr);
    CHECK(a_matrix->hip_native_i64_bytes == static_cast<std::size_t>(m * k) * sizeof(int64_t));
    CHECK(b_matrix->hip_native_i64_bytes == static_cast<std::size_t>(k * n) * sizeof(int64_t));
    CHECK(c_matrix->hip_native_i64_bytes == static_cast<std::size_t>(m * n) * sizeof(int64_t));

    rns8_matrix_storage_info storage_info{};
    storage_info.struct_size = sizeof(storage_info);
    storage_info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_matrix_storage_info(a_matrix, &storage_info) == RNS8_SUCCESS);
    CHECK(storage_info.uses_native_storage == 1);
    CHECK(storage_info.uses_residue_storage == 0);
    CHECK(storage_info.device_native_current == 0);
    CHECK(storage_info.device_native_bytes == static_cast<uint64_t>(m * k * sizeof(int64_t)));
    CHECK(std::string(storage_info.layout_version) == "native_i64_rowmajor_v1");
    CHECK(std::string(storage_info.storage_scope) == "native_device_storage");

    void* a_native = a_matrix->hip_native_i64;
    void* b_native = b_matrix->hip_native_i64;
    void* c_native = c_matrix->hip_native_i64;
    REQUIRE(rns8_pack_i64(vector, a_matrix, A.data(), k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(vector, b_matrix, B.data(), n, 2) == RNS8_SUCCESS);
    CHECK(a_matrix->hip_native_i64 == a_native);
    CHECK(b_matrix->hip_native_i64 == b_native);
    CHECK(a_matrix->device_native_current);
    CHECK(b_matrix->device_native_current);
    CHECK_FALSE(a_matrix->device_residues_current);
    CHECK_FALSE(b_matrix->device_residues_current);

    REQUIRE(rns8_gemm_rns(vector, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    CHECK(c_matrix->hip_native_i64 == c_native);
    CHECK(c_matrix->device_native_current);
    CHECK_FALSE(c_matrix->device_residues_current);
    CHECK(c_matrix->hip_status_buffer != nullptr);

    REQUIRE(rns8_export_i64(vector, plan, c_matrix, vector_c.data(), n) == RNS8_SUCCESS);
    CHECK(vector_c == cpu_c);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    REQUIRE(rns8_get_matrix_storage_info(c_matrix, &storage_info) == RNS8_SUCCESS);
    CHECK(storage_info.device_native_current == 1);
    CHECK(storage_info.device_residues_current == 0);

    void* status_buffer = c_matrix->hip_status_buffer;
    const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    REQUIRE(warmed_allocations.allocate_calls > 0);
    std::fill(vector_c.begin(), vector_c.end(), int64_t{0});

    REQUIRE(rns8_pack_i64(vector, a_matrix, A.data(), k, 3) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(vector, b_matrix, B.data(), n, 4) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(vector, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_export_i64(vector, plan, c_matrix, vector_c.data(), n) == RNS8_SUCCESS);
    const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
    CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
    CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
    CHECK(a_matrix->hip_native_i64 == a_native);
    CHECK(b_matrix->hip_native_i64 == b_native);
    CHECK(c_matrix->hip_native_i64 == c_native);
    CHECK(c_matrix->hip_status_buffer == status_buffer);
    CHECK(vector_c == cpu_c);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    const int64_t m = 2;
    const int64_t n = 2;
    const int64_t k = 3;
    const std::vector<uint64_t> A = {5, 7, 11, 13, 17, 19};
    const std::vector<uint64_t> B = {23, 29, 31, 37, 41, 43};
    std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * n), 0);
    std::vector<uint64_t> vector_c(static_cast<std::size_t>(m * n), 0);

    auto cpu_desc = unsigned_desc(m, n, k, 10000, RNS8_BACKEND_CPU_REFERENCE);
    auto vector_desc = unsigned_desc(m, n, k, 10000, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), k, B.data(), n, cpu_c.data(), n) == RNS8_SUCCESS);

    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(vector, &vector_desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(vector, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(vector, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &c_desc, &c_matrix) == RNS8_SUCCESS);
    REQUIRE(a_matrix->hip_native_u64 != nullptr);
    REQUIRE(b_matrix->hip_native_u64 != nullptr);
    REQUIRE(c_matrix->hip_native_u64 != nullptr);
    CHECK(a_matrix->hip_residues == nullptr);
    CHECK(b_matrix->hip_residues == nullptr);
    CHECK(c_matrix->hip_residues == nullptr);

    rns8_matrix_storage_info storage_info{};
    storage_info.struct_size = sizeof(storage_info);
    storage_info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_matrix_storage_info(a_matrix, &storage_info) == RNS8_SUCCESS);
    CHECK(storage_info.uses_native_storage == 1);
    CHECK(storage_info.uses_residue_storage == 0);
    CHECK(std::string(storage_info.layout_version) == "native_u64_rowmajor_v1");

    REQUIRE(rns8_pack_u64(vector, a_matrix, A.data(), k, 11) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(vector, b_matrix, B.data(), n, 12) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(vector, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
    CHECK(c_matrix->device_native_current);
    CHECK_FALSE(c_matrix->device_residues_current);
    REQUIRE(rns8_export_u64(vector, plan, c_matrix, vector_c.data(), n) == RNS8_SUCCESS);
    CHECK(vector_c == cpu_c);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    rns8_plan* rejected = nullptr;
    auto exact = exact_signed_desc(1, 1, 1, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    CHECK(rns8_create_plan(vector, &exact, &rejected) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(rejected == nullptr);

    auto wrap = wrap_desc(1, 1, 1, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    CHECK(rns8_create_plan(vector, &wrap, &rejected) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(rejected == nullptr);
  }

  rns8_destroy_context(vector);
  rns8_destroy_context(cpu);
}

TEST_CASE("vector ALU native export range errors leave destination unchanged") {
  if (!hip_available()) {
    SKIP("no HIP device available for vector ALU native export range smoke");
  }

  rns8_context* vector = create_context(RNS8_BACKEND_HIP_VECTOR_ALU_INT64);

  {
    constexpr int64_t m = 1;
    constexpr int64_t n = 2;
    constexpr int64_t k = 1;
    const int64_t A[] = {1};
    const int64_t B[] = {7, 11};
    auto desc = signed_desc(m, n, k, 10, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(vector, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(vector, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(vector, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &c_desc, &c_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(vector, a_matrix, A, k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(vector, b_matrix, B, n, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(vector, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

    std::vector<int64_t> out(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212));
    CHECK(rns8_export_i64(vector, plan, c_matrix, out.data(), n) == RNS8_RANGE_ERROR);
    CHECK(out == std::vector<int64_t>(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212)));
    CHECK(c_matrix->device_native_current);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    constexpr int64_t m = 1;
    constexpr int64_t n = 2;
    constexpr int64_t k = 1;
    const uint64_t A[] = {1};
    const uint64_t B[] = {7, 11};
    auto desc = unsigned_desc(m, n, k, 10, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(vector, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(vector, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(vector, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(vector, &c_desc, &c_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(vector, a_matrix, A, k, 1) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(vector, b_matrix, B, n, 2) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(vector, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);

    std::vector<uint64_t> out(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe));
    CHECK(rns8_export_u64(vector, plan, c_matrix, out.data(), n) == RNS8_RANGE_ERROR);
    CHECK(out == std::vector<uint64_t>(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe)));
    CHECK(c_matrix->device_native_current);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(vector);
}

TEST_CASE("direct HIP bounded GEMM rejects host-current stale device inputs") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP bounded stale-input GEMM rejection smoke");
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 2;
  constexpr int8_t c_sentinel = -33;
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    auto desc = signed_desc(m, n, k, 100, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

    fill_exact_residue_matrix(a_matrix, {boost::multiprecision::cpp_int(1), boost::multiprecision::cpp_int(2)});
    fill_exact_residue_matrix(b_matrix, {boost::multiprecision::cpp_int(3), boost::multiprecision::cpp_int(4)});
    upload_exact_residues_to_hip(b_matrix);
    std::fill(c_matrix->residues.begin(), c_matrix->residues.end(), c_sentinel);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(
                hip->device_id, c_matrix->hip_residues, c_matrix->residues.data(), c_matrix->hip_residue_bytes) ==
            RNS8_SUCCESS);
    c_matrix->host_residues_current = false;
    c_matrix->device_residues_current = true;

    CHECK(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
    CHECK(a_matrix->host_residues_current);
    CHECK_FALSE(a_matrix->device_residues_current);
    CHECK_FALSE(b_matrix->host_residues_current);
    CHECK(b_matrix->device_residues_current);
    CHECK(c_matrix->hip_upload_buffer == nullptr);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    CHECK(c_matrix->hip_status_buffer == nullptr);
    REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
                hip->device_id, c_matrix->residues.data(), c_matrix->hip_residues, c_matrix->hip_residue_bytes) ==
            RNS8_SUCCESS);
    CHECK(std::all_of(c_matrix->residues.begin(), c_matrix->residues.end(), [&](int8_t value) {
      return value == c_sentinel;
    }));

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = unsigned_desc(m, n, k, 100, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

    fill_exact_residue_matrix(a_matrix, {boost::multiprecision::cpp_int(5), boost::multiprecision::cpp_int(6)});
    fill_exact_residue_matrix(b_matrix, {boost::multiprecision::cpp_int(7), boost::multiprecision::cpp_int(8)});
    upload_exact_residues_to_hip(a_matrix);
    std::fill(c_matrix->residues.begin(), c_matrix->residues.end(), c_sentinel);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(
                hip->device_id, c_matrix->hip_residues, c_matrix->residues.data(), c_matrix->hip_residue_bytes) ==
            RNS8_SUCCESS);
    c_matrix->host_residues_current = false;
    c_matrix->device_residues_current = true;

    CHECK(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
    CHECK_FALSE(a_matrix->host_residues_current);
    CHECK(a_matrix->device_residues_current);
    CHECK(b_matrix->host_residues_current);
    CHECK_FALSE(b_matrix->device_residues_current);
    CHECK(c_matrix->hip_upload_buffer == nullptr);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    CHECK(c_matrix->hip_status_buffer == nullptr);
    REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
                hip->device_id, c_matrix->residues.data(), c_matrix->hip_residues, c_matrix->hip_residue_bytes) ==
            RNS8_SUCCESS);
    CHECK(std::all_of(c_matrix->residues.begin(), c_matrix->residues.end(), [&](int8_t value) {
      return value == c_sentinel;
    }));

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(hip);
}

TEST_CASE("direct HIP bounded exports reject host-current stale device residues") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP bounded stale-device export rejection smoke");
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    const std::vector<int64_t> expected = {-31, 0, 17, 999};
    auto desc = signed_desc(m, n, 1, 1000, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(
        c_matrix,
        {boost::multiprecision::cpp_int(expected[0]),
         boost::multiprecision::cpp_int(expected[1]),
         boost::multiprecision::cpp_int(expected[2]),
         boost::multiprecision::cpp_int(expected[3])});

    std::vector<int64_t> out(static_cast<std::size_t>(m * n), INT64_C(0x7f7f7f7f));
    CHECK(rns8_export_i64(hip, plan, c_matrix, out.data(), n) == RNS8_INVALID_ARGUMENT);
    CHECK(out == std::vector<int64_t>(static_cast<std::size_t>(m * n), INT64_C(0x7f7f7f7f)));
    CHECK(c_matrix->host_residues_current);
    CHECK_FALSE(c_matrix->device_residues_current);
    CHECK(c_matrix->hip_upload_buffer == nullptr);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    CHECK(c_matrix->hip_status_buffer == nullptr);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  {
    const std::vector<uint64_t> expected = {0, 17, 999, 1000};
    auto desc = unsigned_desc(m, n, 1, 1000, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(
        c_matrix,
        {boost::multiprecision::cpp_int(expected[0]),
         boost::multiprecision::cpp_int(expected[1]),
         boost::multiprecision::cpp_int(expected[2]),
         boost::multiprecision::cpp_int(expected[3])});

    std::vector<uint64_t> out(static_cast<std::size_t>(m * n), UINT64_C(0xffffffffffffffff));
    CHECK(rns8_export_u64(hip, plan, c_matrix, out.data(), n) == RNS8_INVALID_ARGUMENT);
    CHECK(out == std::vector<uint64_t>(static_cast<std::size_t>(m * n), UINT64_C(0xffffffffffffffff)));
    CHECK(c_matrix->host_residues_current);
    CHECK_FALSE(c_matrix->device_residues_current);
    CHECK(c_matrix->hip_upload_buffer == nullptr);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    CHECK(c_matrix->hip_status_buffer == nullptr);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(hip);
}

TEST_CASE("direct HIP bounded export range errors preserve host output and reuse resident buffers") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP bounded range-error export smoke");
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    auto cpu_desc = signed_desc(m, n, 1, 10, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = signed_desc(m, n, 1, 10, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* hip_plan = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    REQUIRE(cpu_plan->prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    REQUIRE(hip_plan->prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

    fill_exact_residue_matrix(cpu_c, {boost::multiprecision::cpp_int(-11), boost::multiprecision::cpp_int(7)});
    fill_exact_residue_matrix(hip_c, {boost::multiprecision::cpp_int(-11), boost::multiprecision::cpp_int(7)});
    upload_exact_residues_to_hip(hip_c);

    std::vector<int64_t> cpu_out(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212));
    std::vector<int64_t> hip_out(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212));
    CHECK(rns8_export_i64(cpu, cpu_plan, cpu_c, cpu_out.data(), n) == RNS8_RANGE_ERROR);
    CHECK(rns8_export_i64(hip, hip_plan, hip_c, hip_out.data(), n) == RNS8_RANGE_ERROR);
    CHECK(cpu_out == std::vector<int64_t>(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212)));
    CHECK(hip_out == std::vector<int64_t>(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212)));
    CHECK_FALSE(hip_c->host_residues_current);
    CHECK(hip_c->device_residues_current);
    CHECK(hip_c->hip_upload_buffer == nullptr);
    REQUIRE(hip_c->hip_export_buffer != nullptr);
    REQUIRE(hip_c->hip_status_buffer != nullptr);
    void* export_buffer = hip_c->hip_export_buffer;
    void* status_buffer = hip_c->hip_status_buffer;
    const std::size_t export_bytes = hip_c->hip_export_bytes;
    const std::size_t status_bytes = hip_c->hip_status_bytes;
    const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();

    CHECK(rns8_export_i64(hip, hip_plan, hip_c, hip_out.data(), n) == RNS8_RANGE_ERROR);
    CHECK_FALSE(hip_c->host_residues_current);
    CHECK(hip_c->device_residues_current);
    const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
    CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
    CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
    CHECK(hip_c->hip_upload_buffer == nullptr);
    CHECK(hip_c->hip_export_buffer == export_buffer);
    CHECK(hip_c->hip_status_buffer == status_buffer);
    CHECK(hip_c->hip_upload_bytes == 0);
    CHECK(hip_c->hip_export_bytes == export_bytes);
    CHECK(hip_c->hip_status_bytes == status_bytes);
    CHECK(hip_out == std::vector<int64_t>(static_cast<std::size_t>(m * n), INT64_C(0x1212121212121212)));

    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_plan(hip_plan);
    rns8_destroy_plan(cpu_plan);
  }

  {
    auto cpu_desc = unsigned_desc(m, n, 1, 10, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = unsigned_desc(m, n, 1, 10, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* hip_plan = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    REQUIRE(cpu_plan->prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    REQUIRE(hip_plan->prefix == RNS8_DEFAULT_BOUNDED_PREFIX);
    auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

    fill_exact_residue_matrix(cpu_c, {boost::multiprecision::cpp_int(11), boost::multiprecision::cpp_int(3)});
    fill_exact_residue_matrix(hip_c, {boost::multiprecision::cpp_int(11), boost::multiprecision::cpp_int(3)});
    upload_exact_residues_to_hip(hip_c);

    std::vector<uint64_t> cpu_out(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe));
    std::vector<uint64_t> hip_out(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe));
    CHECK(rns8_export_u64(cpu, cpu_plan, cpu_c, cpu_out.data(), n) == RNS8_RANGE_ERROR);
    CHECK(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), n) == RNS8_RANGE_ERROR);
    CHECK(cpu_out == std::vector<uint64_t>(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe)));
    CHECK(hip_out == std::vector<uint64_t>(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe)));
    CHECK_FALSE(hip_c->host_residues_current);
    CHECK(hip_c->device_residues_current);
    CHECK(hip_c->hip_upload_buffer == nullptr);
    REQUIRE(hip_c->hip_export_buffer != nullptr);
    REQUIRE(hip_c->hip_status_buffer != nullptr);
    void* export_buffer = hip_c->hip_export_buffer;
    void* status_buffer = hip_c->hip_status_buffer;
    const std::size_t export_bytes = hip_c->hip_export_bytes;
    const std::size_t status_bytes = hip_c->hip_status_bytes;
    const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();

    CHECK(rns8_export_u64(hip, hip_plan, hip_c, hip_out.data(), n) == RNS8_RANGE_ERROR);
    CHECK_FALSE(hip_c->host_residues_current);
    CHECK(hip_c->device_residues_current);
    const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(repeated_allocations.allocate_calls == warmed_allocations.allocate_calls);
    CHECK(repeated_allocations.free_calls == warmed_allocations.free_calls);
    CHECK(repeated_allocations.allocated_bytes == warmed_allocations.allocated_bytes);
    CHECK(hip_c->hip_upload_buffer == nullptr);
    CHECK(hip_c->hip_export_buffer == export_buffer);
    CHECK(hip_c->hip_status_buffer == status_buffer);
    CHECK(hip_c->hip_upload_bytes == 0);
    CHECK(hip_c->hip_export_bytes == export_bytes);
    CHECK(hip_c->hip_status_bytes == status_bytes);
    CHECK(hip_out == std::vector<uint64_t>(static_cast<std::size_t>(m * n), UINT64_C(0xfefefefefefefefe)));

    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_plan(hip_plan);
    rns8_destroy_plan(cpu_plan);
  }

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
  void* c_upload = c_matrix->hip_upload_buffer;
  void* c_export = c_matrix->hip_export_buffer;
  void* c_status = c_matrix->hip_status_buffer;
  const std::size_t a_upload_bytes = a_matrix->hip_upload_bytes;
  const std::size_t b_upload_bytes = b_matrix->hip_upload_bytes;
  const std::size_t c_upload_bytes = c_matrix->hip_upload_bytes;
  const std::size_t c_export_bytes = c_matrix->hip_export_bytes;
  const std::size_t c_status_bytes = c_matrix->hip_status_bytes;
  const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  REQUIRE(a_upload != nullptr);
  REQUIRE(b_upload != nullptr);
  CHECK(c_upload == nullptr);
  CHECK(c_upload_bytes == 0);
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
  CHECK(c_matrix->hip_upload_buffer == c_upload);
  CHECK(c_matrix->hip_export_buffer == c_export);
  CHECK(c_matrix->hip_status_buffer == c_status);
  CHECK(a_matrix->hip_upload_bytes == a_upload_bytes);
  CHECK(b_matrix->hip_upload_bytes == b_upload_bytes);
  CHECK(c_matrix->hip_upload_bytes == c_upload_bytes);
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

TEST_CASE("direct HIP bounded per-tile workspace and matrix schedule metadata are part of residency contract") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP bounded per-tile schedule contract smoke");
  }

  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  constexpr int64_t lda = 1;
  constexpr int64_t ldb = n;
  constexpr int64_t ldc = n;
  constexpr uint64_t sentinel = 0xdadadadadadadadaull;
  const std::vector<uint64_t> bounds_a = {0, 0, 7000000, 1000000000};
  const std::vector<uint64_t> bounds_b = {0, 0, 7000000, 999999999};
  const std::vector<uint64_t> bounds_tile = {1000000000};

  auto make_desc = [&](const std::vector<uint64_t>& bounds, uint32_t tile_m, uint32_t tile_n, rns8_backend_kind backend) {
    auto desc = unsigned_desc(m, n, k, 0, backend);
    desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
    desc.tile_m = tile_m;
    desc.tile_n = tile_n;
    desc.tile_bounds = bounds.data();
    desc.tile_bounds_count = static_cast<uint64_t>(bounds.size());
    return desc;
  };

  std::vector<uint64_t> A(static_cast<std::size_t>(m), 0);
  std::vector<uint64_t> B(static_cast<std::size_t>(n), 7);
  A.back() = 1000000;
  B.back() = 1000;
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = make_desc(bounds_a, 64, 64, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = make_desc(bounds_a, 64, 64, RNS8_BACKEND_HIP_DIRECT);
  auto hip_desc_b = make_desc(bounds_b, 64, 64, RNS8_BACKEND_HIP_DIRECT);
  auto hip_desc_tile = make_desc(bounds_tile, 128, 128, RNS8_BACKEND_HIP_DIRECT);

  rns8::detail::hip_direct_allocation_counters_reset();
  rns8_plan* plan = nullptr;
  rns8_plan* plan_b = nullptr;
  rns8_plan* plan_tile = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_workspace* workspace_b = nullptr;
  rns8_workspace* workspace_tile = nullptr;
  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  rns8_matrix* wrong_a_tile = nullptr;
  rns8_matrix* wrong_c_tile = nullptr;

  REQUIRE(rns8_create_plan(hip, &hip_desc, &plan) == RNS8_SUCCESS);
  REQUIRE_FALSE(plan->tile_schedule.empty());
  CHECK(plan->tile_schedule[0].range_bit_length == 0);
  REQUIRE(rns8_create_plan(hip, &hip_desc_b, &plan_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc_tile, &plan_tile) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, plan_b, &workspace_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(hip, plan_tile, &workspace_tile) == RNS8_SUCCESS);

  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

  auto wrong_a_desc = a_desc;
  wrong_a_desc.tile_m = 128;
  wrong_a_desc.tile_n = 128;
  auto wrong_c_desc = c_desc;
  wrong_c_desc.tile_m = 128;
  wrong_c_desc.tile_n = 128;
  REQUIRE(rns8_create_matrix(hip, &wrong_a_desc, &wrong_a_tile) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &wrong_c_desc, &wrong_c_tile) == RNS8_SUCCESS);

  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, a_matrix, A.data(), lda, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, b_matrix, B.data(), ldb, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, wrong_a_tile, A.data(), lda, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  CHECK(hip_c == cpu_c);

  const auto warmed_snapshot = capture_bounded_resident_snapshot(a_matrix, b_matrix, c_matrix, workspace);
  const uint64_t warmed_output_version = c_matrix->source_version;
  CHECK(warmed_output_version != 0);
  CHECK(warmed_snapshot.c_upload == nullptr);
  CHECK(warmed_snapshot.c_upload_bytes == 0);
  CHECK(workspace->schedule_tile_count == bounds_a.size());
  CHECK(workspace->schedule_prefix_group_count >= 1);
  CHECK(workspace->schedule_fingerprint != workspace_b->schedule_fingerprint);
  CHECK(workspace->schedule_fingerprint != workspace_tile->schedule_fingerprint);

  CHECK(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace_b) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace_tile) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_rns(hip, plan, wrong_a_tile, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_u64(hip, plan, wrong_c_tile, hip_c.data(), ldc) == RNS8_INVALID_ARGUMENT);
  check_bounded_resident_snapshot_unchanged(warmed_snapshot, a_matrix, b_matrix, c_matrix, workspace);
  CHECK(c_matrix->source_version == warmed_output_version);

  rns8_destroy_matrix(wrong_c_tile);
  rns8_destroy_matrix(wrong_a_tile);
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace_tile);
  rns8_destroy_workspace(workspace_b);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan_tile);
  rns8_destroy_plan(plan_b);
  rns8_destroy_plan(plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP persistent per-tile bounded u64 K-split reuses resident buffers across mixed prefixes") {
  if (!hip_available()) {
    SKIP("no HIP device available for persistent per-tile direct HIP bounded K-split reuse smoke");
  }

  constexpr int64_t m = 65;
  constexpr int64_t n = 2;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  const int64_t lda = k + 1;
  constexpr int64_t ldb = n + 1;
  constexpr int64_t ldc = n + 2;
  constexpr uint64_t sentinel = 0xa5a5a5a5a5a5a5a5ull;
  const uint64_t ku = static_cast<uint64_t>(k);
  const std::vector<uint64_t> bounds = {2u * ku * 1000u, ku * 1000000u * 1000u};

  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), sentinel);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), sentinel);
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);

  auto fill_inputs = [&](int variant) {
    std::fill(A.begin(), A.end(), sentinel);
    std::fill(B.begin(), B.end(), sentinel);
    for (int64_t row = 0; row < m; ++row) {
      const uint64_t value = row < 64 ? (variant == 0 ? 1u : 2u) : (variant == 0 ? 1000000u : 500000u);
      for (int64_t kk = 0; kk < k; ++kk) {
        A[static_cast<std::size_t>(row * lda + kk)] = value;
      }
    }
    for (int64_t kk = 0; kk < k; ++kk) {
      B[static_cast<std::size_t>(kk * ldb)] = variant == 0 ? 1000u : 997u;
      B[static_cast<std::size_t>(kk * ldb + 1)] = variant == 0 ? 999u : 991u;
    }
  };

  auto compare_outputs = [&]() {
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_c[static_cast<std::size_t>(row * ldc + col)]);
      }
      for (int64_t pad = n; pad < ldc; ++pad) {
        CHECK(cpu_c[static_cast<std::size_t>(row * ldc + pad)] == sentinel);
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + pad)] == sentinel);
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
  CHECK(info.max_selected_prefix < RNS8_DEFAULT_BOUNDED_PREFIX);

  REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

  fill_inputs(0);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, a_matrix, A.data(), lda, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, b_matrix, B.data(), ldb, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs();
  const uint64_t first_output_version = c_matrix->source_version;
  CHECK(first_output_version != 0);
  CHECK(first_output_version != a_matrix->source_version);
  CHECK(first_output_version != b_matrix->source_version);
  const auto warmed_snapshot = capture_bounded_resident_snapshot(a_matrix, b_matrix, c_matrix, workspace);
  REQUIRE(warmed_snapshot.a_upload != nullptr);
  REQUIRE(warmed_snapshot.b_upload != nullptr);
  CHECK(warmed_snapshot.c_upload == nullptr);
  CHECK(warmed_snapshot.c_upload_bytes == 0);
  REQUIRE(warmed_snapshot.c_export != nullptr);
  REQUIRE(warmed_snapshot.c_status != nullptr);
  REQUIRE(warmed_snapshot.allocations.allocate_calls > 0);
  REQUIRE(warmed_snapshot.allocations.allocated_bytes > 0);

  fill_inputs(1);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, a_matrix, A.data(), lda, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, b_matrix, B.data(), ldb, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs();
  CHECK(c_matrix->source_version != 0);
  CHECK(c_matrix->source_version != first_output_version);
  check_bounded_resident_snapshot_unchanged(warmed_snapshot, a_matrix, b_matrix, c_matrix, workspace);
  CHECK(a_matrix->device_residues_current);
  CHECK_FALSE(a_matrix->host_residues_current);
  CHECK(b_matrix->device_residues_current);
  CHECK_FALSE(b_matrix->host_residues_current);
  CHECK(c_matrix->device_residues_current);
  CHECK_FALSE(c_matrix->host_residues_current);

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
  void* c_upload = c_matrix->hip_upload_buffer;
  void* c_export = c_matrix->hip_export_buffer;
  void* c_status = c_matrix->hip_status_buffer;
  const std::size_t a_upload_bytes = a_matrix->hip_upload_bytes;
  const std::size_t b_upload_bytes = b_matrix->hip_upload_bytes;
  const std::size_t c_upload_bytes = c_matrix->hip_upload_bytes;
  const std::size_t c_export_bytes = c_matrix->hip_export_bytes;
  const std::size_t c_status_bytes = c_matrix->hip_status_bytes;
  const auto warmed_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  REQUIRE(a_upload != nullptr);
  REQUIRE(b_upload != nullptr);
  CHECK(c_upload == nullptr);
  CHECK(c_upload_bytes == 0);
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
  CHECK(c_matrix->hip_upload_buffer == c_upload);
  CHECK(c_matrix->hip_export_buffer == c_export);
  CHECK(c_matrix->hip_status_buffer == c_status);
  CHECK(a_matrix->hip_upload_bytes == a_upload_bytes);
  CHECK(b_matrix->hip_upload_bytes == b_upload_bytes);
  CHECK(c_matrix->hip_upload_bytes == c_upload_bytes);
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

TEST_CASE("direct HIP persistent bounded u64 prefix-9 covers exact K-block boundary with padded layouts") {
  if (!hip_available()) {
    SKIP("no HIP device available for persistent direct HIP bounded u64 K-boundary smoke");
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  const int64_t k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK);
  const int64_t lda = k + 1;
  constexpr int64_t ldb = n + 1;
  constexpr int64_t ldc = n + 1;
  constexpr uint64_t sentinel = 0xd1ffd1ffd1ffd1ffull;
  const uint64_t bound = static_cast<uint64_t>(k) * 12u;

  std::vector<uint64_t> A(static_cast<std::size_t>(m * lda), sentinel);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * ldb), sentinel);
  std::vector<uint64_t> cpu_c(static_cast<std::size_t>(m * ldc), sentinel);
  std::vector<uint64_t> hip_c(static_cast<std::size_t>(m * ldc), sentinel);

  auto fill_inputs = [&](int variant) {
    std::fill(A.begin(), A.end(), sentinel);
    std::fill(B.begin(), B.end(), sentinel);
    for (int64_t kk = 0; kk < k; ++kk) {
      A[static_cast<std::size_t>(kk)] = variant == 0 ? 1 : 2;
      A[static_cast<std::size_t>(lda + kk)] = variant == 0 ? 2 : 3;
      B[static_cast<std::size_t>(kk * ldb)] = variant == 0 ? 3 : 1;
      B[static_cast<std::size_t>(kk * ldb + 1)] = variant == 0 ? 5 : 4;
    }
  };

  auto compare_outputs = [&]() {
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        CHECK(hip_c[static_cast<std::size_t>(row * ldc + col)] ==
              cpu_c[static_cast<std::size_t>(row * ldc + col)]);
      }
      CHECK(cpu_c[static_cast<std::size_t>(row * ldc + n)] == sentinel);
      CHECK(hip_c[static_cast<std::size_t>(row * ldc + n)] == sentinel);
    }
  };

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = unsigned_desc(m, n, k, bound, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = unsigned_desc(m, n, k, bound, RNS8_BACKEND_HIP_DIRECT);

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
  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
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
  const uint64_t first_output_version = c_matrix->source_version;
  CHECK(first_output_version != 0);
  CHECK(first_output_version != a_matrix->source_version);
  CHECK(first_output_version != b_matrix->source_version);

  const auto warmed_snapshot = capture_bounded_resident_snapshot(a_matrix, b_matrix, c_matrix, workspace);
  REQUIRE(warmed_snapshot.a_upload != nullptr);
  REQUIRE(warmed_snapshot.b_upload != nullptr);
  CHECK(warmed_snapshot.c_upload == nullptr);
  CHECK(warmed_snapshot.c_upload_bytes == 0);
  REQUIRE(warmed_snapshot.c_export != nullptr);
  REQUIRE(warmed_snapshot.c_status != nullptr);
  REQUIRE(warmed_snapshot.allocations.allocate_calls > 0);
  REQUIRE(warmed_snapshot.allocations.allocated_bytes > 0);

  fill_inputs(1);
  std::fill(cpu_c.begin(), cpu_c.end(), sentinel);
  std::fill(hip_c.begin(), hip_c.end(), sentinel);
  REQUIRE(rns8_gemm_u64_oneshot(cpu, &cpu_desc, A.data(), lda, B.data(), ldb, cpu_c.data(), ldc) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, a_matrix, A.data(), lda, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(hip, b_matrix, B.data(), ldb, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(hip, plan, c_matrix, hip_c.data(), ldc) == RNS8_SUCCESS);
  compare_outputs();
  CHECK(c_matrix->source_version != 0);
  CHECK(c_matrix->source_version != first_output_version);

  check_bounded_resident_snapshot_unchanged(warmed_snapshot, a_matrix, b_matrix, c_matrix, workspace);
  CHECK(a_matrix->hip_residues == a_device_residues);
  CHECK(b_matrix->hip_residues == b_device_residues);
  CHECK(c_matrix->hip_residues == c_device_residues);

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

TEST_CASE("direct HIP exact-wide GEMM rejects host-current stale device inputs") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide stale-input GEMM rejection smoke");
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 1;
  constexpr int64_t k = 2;
  constexpr int8_t c_sentinel = -41;
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    auto desc = exact_signed_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

    fill_exact_residue_matrix(a_matrix, {boost::multiprecision::cpp_int(-3), boost::multiprecision::cpp_int(5)});
    fill_exact_residue_matrix(b_matrix, {boost::multiprecision::cpp_int(7), boost::multiprecision::cpp_int(-11)});
    upload_exact_residues_to_hip(b_matrix);
    std::fill(c_matrix->residues.begin(), c_matrix->residues.end(), c_sentinel);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(
                hip->device_id, c_matrix->hip_residues, c_matrix->residues.data(), c_matrix->hip_residue_bytes) ==
            RNS8_SUCCESS);
    c_matrix->host_residues_current = false;
    c_matrix->device_residues_current = true;

    CHECK(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
    CHECK(a_matrix->host_residues_current);
    CHECK_FALSE(a_matrix->device_residues_current);
    CHECK_FALSE(b_matrix->host_residues_current);
    CHECK(b_matrix->device_residues_current);
    REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
                hip->device_id, c_matrix->residues.data(), c_matrix->hip_residues, c_matrix->hip_residue_bytes) ==
            RNS8_SUCCESS);
    CHECK(std::all_of(c_matrix->residues.begin(), c_matrix->residues.end(), [&](int8_t value) {
      return value == c_sentinel;
    }));

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_unsigned_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(hip, plan, &workspace) == RNS8_SUCCESS);
    auto a_desc = matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    auto b_desc = matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(hip, &a_desc, &a_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &b_desc, &b_matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);

    fill_exact_residue_matrix(a_matrix, {boost::multiprecision::cpp_int(13), boost::multiprecision::cpp_int(17)});
    fill_exact_residue_matrix(b_matrix, {boost::multiprecision::cpp_int(19), boost::multiprecision::cpp_int(23)});
    upload_exact_residues_to_hip(a_matrix);
    std::fill(c_matrix->residues.begin(), c_matrix->residues.end(), c_sentinel);
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(
                hip->device_id, c_matrix->hip_residues, c_matrix->residues.data(), c_matrix->hip_residue_bytes) ==
            RNS8_SUCCESS);
    c_matrix->host_residues_current = false;
    c_matrix->device_residues_current = true;

    CHECK(rns8_gemm_rns(hip, plan, a_matrix, b_matrix, c_matrix, workspace) == RNS8_INVALID_ARGUMENT);
    CHECK_FALSE(a_matrix->host_residues_current);
    CHECK(a_matrix->device_residues_current);
    CHECK(b_matrix->host_residues_current);
    CHECK_FALSE(b_matrix->device_residues_current);
    REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
                hip->device_id, c_matrix->residues.data(), c_matrix->hip_residues, c_matrix->hip_residue_bytes) ==
            RNS8_SUCCESS);
    CHECK(std::all_of(c_matrix->residues.begin(), c_matrix->residues.end(), [&](int8_t value) {
      return value == c_sentinel;
    }));

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(hip);
}

TEST_CASE("direct HIP exact-wide signed export matches CPU fixed-width boundary ABI") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide signed export boundary smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  const int64_t m = 1;
  const int64_t n = 2;
  auto cpu_desc = exact_signed_desc(m, n, 1, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = exact_signed_desc(m, n, 1, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(cpu_plan->prefix);
  const boost::multiprecision::cpp_int half = product / 2;
  fill_exact_residue_matrix(cpu_c, {half, boost::multiprecision::cpp_int(-1)});
  fill_exact_residue_matrix(hip_c, {half, boost::multiprecision::cpp_int(-1)});
  upload_exact_residues_to_hip(hip_c);

  constexpr uint32_t limb_count = 3;
  constexpr int64_t limb_ld = 3;
  std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), 0x9a9a9a9a9a9a9a9aull);
  std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), 0x9a9a9a9a9a9a9a9aull);
  REQUIRE(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  CHECK(hip_limbs == cpu_limbs);
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + n) * limb_count)] == 0x9a9a9a9a9a9a9a9aull);
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 1) * limb_count)] ==
        std::numeric_limits<uint64_t>::max());
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 1) * limb_count + 1)] ==
        std::numeric_limits<uint64_t>::max());
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 1) * limb_count + 2)] ==
        std::numeric_limits<uint64_t>::max());

  std::vector<uint64_t> cpu_one_limb(static_cast<std::size_t>(m * n), 0x1010101010101010ull);
  std::vector<uint64_t> hip_one_limb(static_cast<std::size_t>(m * n), 0x1010101010101010ull);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, cpu_one_limb.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, hip_one_limb.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(cpu_one_limb == std::vector<uint64_t>(static_cast<std::size_t>(m * n), 0x1010101010101010ull));
  CHECK(hip_one_limb == std::vector<uint64_t>(static_cast<std::size_t>(m * n), 0x1010101010101010ull));

  std::vector<uint64_t> wrong_export(static_cast<std::size_t>(m * n * 2), 0x2020202020202020ull);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, wrong_export.data(), n, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, wrong_export.data(), n, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(wrong_export == std::vector<uint64_t>(static_cast<std::size_t>(m * n * 2), 0x2020202020202020ull));

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide fixed limb export widths match CPU") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide fixed limb-width smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = exact_signed_desc(1, 1, 1, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = exact_signed_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  auto c_desc = matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);
  fill_exact_residue_matrix(cpu_c, {boost::multiprecision::cpp_int(-1)});
  fill_exact_residue_matrix(hip_c, {boost::multiprecision::cpp_int(-1)});
  upload_exact_residues_to_hip(hip_c);

  for (const uint32_t limb_count : {1u, 2u, 4u, 8u, 16u, 32u}) {
    std::vector<uint64_t> cpu_limbs(limb_count, 0);
    std::vector<uint64_t> hip_limbs(limb_count, 0);
    REQUIRE(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), 1, limb_count) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, hip_limbs.data(), 1, limb_count) ==
            RNS8_SUCCESS);
    CHECK(hip_limbs == cpu_limbs);
    for (uint64_t limb : hip_limbs) {
      CHECK(limb == std::numeric_limits<uint64_t>::max());
    }
  }

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);

  auto cpu_unsigned_desc = exact_unsigned_desc(1, 1, 1, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_unsigned_desc = exact_unsigned_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_unsigned_plan = nullptr;
  rns8_plan* hip_unsigned_plan = nullptr;
  rns8_matrix* cpu_unsigned_c = nullptr;
  rns8_matrix* hip_unsigned_c = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_unsigned_desc, &cpu_unsigned_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_unsigned_desc, &hip_unsigned_plan) == RNS8_SUCCESS);
  auto unsigned_c_desc = matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &unsigned_c_desc, &cpu_unsigned_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &unsigned_c_desc, &hip_unsigned_c) == RNS8_SUCCESS);
  fill_exact_residue_matrix(cpu_unsigned_c, {boost::multiprecision::cpp_int(1)});
  fill_exact_residue_matrix(hip_unsigned_c, {boost::multiprecision::cpp_int(1)});
  upload_exact_residues_to_hip(hip_unsigned_c);

  for (const uint32_t limb_count : {1u, 2u, 4u, 8u, 16u, 32u}) {
    std::vector<uint64_t> cpu_limbs(limb_count, 0x4444444444444444ull);
    std::vector<uint64_t> hip_limbs(limb_count, 0x5555555555555555ull);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(
                cpu, cpu_unsigned_plan, cpu_unsigned_c, cpu_limbs.data(), 1, limb_count) == RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(
                hip, hip_unsigned_plan, hip_unsigned_c, hip_limbs.data(), 1, limb_count) == RNS8_SUCCESS);
    CHECK(hip_limbs == cpu_limbs);
    CHECK(hip_limbs[0] == 1);
    for (uint32_t limb = 1; limb < limb_count; ++limb) {
      CHECK(hip_limbs[limb] == 0);
    }
  }

  rns8_destroy_matrix(hip_unsigned_c);
  rns8_destroy_matrix(cpu_unsigned_c);
  rns8_destroy_plan(hip_unsigned_plan);
  rns8_destroy_plan(cpu_unsigned_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide max-width padded export matches CPU ABI") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide max-width padded export smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 1;
  constexpr int64_t limb_ld = 3;
  constexpr uint32_t limb_count = 32;

  {
    auto cpu_desc = exact_signed_desc(m, n, k, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = exact_signed_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* hip_plan = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

    const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(cpu_plan->prefix);
    const boost::multiprecision::cpp_int half = product / 2;
    fill_exact_residue_matrix(
        cpu_c,
        {half,
         boost::multiprecision::cpp_int(-1),
         -(boost::multiprecision::cpp_int(1) << 63u),
         boost::multiprecision::cpp_int(1)});
    fill_exact_residue_matrix(
        hip_c,
        {half,
         boost::multiprecision::cpp_int(-1),
         -(boost::multiprecision::cpp_int(1) << 63u),
         boost::multiprecision::cpp_int(1)});
    upload_exact_residues_to_hip(hip_c);

    constexpr uint64_t sentinel = 0x1919191919191919ull;
    std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), sentinel);
    std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), sentinel);
    REQUIRE(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    CHECK(hip_limbs == cpu_limbs);
    CHECK_FALSE(hip_c->host_residues_current);
    for (int64_t row = 0; row < m; ++row) {
      const std::size_t padding = static_cast<std::size_t>((row * limb_ld + n) * limb_count);
      for (uint32_t limb = 0; limb < limb_count; ++limb) {
        CHECK(hip_limbs[padding + limb] == sentinel);
      }
    }
    const std::size_t minus_one = static_cast<std::size_t>((0 * limb_ld + 1) * limb_count);
    for (uint32_t limb = 0; limb < limb_count; ++limb) {
      CHECK(hip_limbs[minus_one + limb] == std::numeric_limits<uint64_t>::max());
    }

    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_plan(hip_plan);
    rns8_destroy_plan(cpu_plan);
  }

  {
    auto cpu_desc = exact_unsigned_desc(m, n, k, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_desc = exact_unsigned_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* cpu_plan = nullptr;
    rns8_plan* hip_plan = nullptr;
    rns8_matrix* cpu_c = nullptr;
    rns8_matrix* hip_c = nullptr;
    REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
    auto c_desc = matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

    fill_exact_residue_matrix(
        cpu_c,
        {boost::multiprecision::cpp_int(0),
         boost::multiprecision::cpp_int(1) << 63u,
         boost::multiprecision::cpp_int(1) << 127u,
         boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max())});
    fill_exact_residue_matrix(
        hip_c,
        {boost::multiprecision::cpp_int(0),
         boost::multiprecision::cpp_int(1) << 63u,
         boost::multiprecision::cpp_int(1) << 127u,
         boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max())});
    upload_exact_residues_to_hip(hip_c);

    constexpr uint64_t sentinel = 0x2929292929292929ull;
    std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), sentinel);
    std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(m * limb_ld * limb_count), sentinel);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
            RNS8_SUCCESS);
    CHECK(hip_limbs == cpu_limbs);
    CHECK_FALSE(hip_c->host_residues_current);
    for (int64_t row = 0; row < m; ++row) {
      const std::size_t padding = static_cast<std::size_t>((row * limb_ld + n) * limb_count);
      for (uint32_t limb = 0; limb < limb_count; ++limb) {
        CHECK(hip_limbs[padding + limb] == sentinel);
      }
    }
    const std::size_t high_bit = static_cast<std::size_t>((1 * limb_ld + 0) * limb_count);
    CHECK(hip_limbs[high_bit] == 0);
    CHECK(hip_limbs[high_bit + 1] == (uint64_t{1} << 63u));
    for (uint32_t limb = 2; limb < limb_count; ++limb) {
      CHECK(hip_limbs[high_bit + limb] == 0);
    }

    rns8_destroy_matrix(hip_c);
    rns8_destroy_matrix(cpu_c);
    rns8_destroy_plan(hip_plan);
    rns8_destroy_plan(cpu_plan);
  }

  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide export requires device-current resident residues") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide device-current export smoke");
  }

  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);

  {
    auto desc = exact_signed_desc(1, 2, 1, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = matrix_desc(1, 2, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(c_matrix, {boost::multiprecision::cpp_int(-1), boost::multiprecision::cpp_int(1)});

    constexpr uint64_t sentinel = 0x6a6a6a6a6a6a6a6aull;
    std::vector<uint64_t> limbs(4, sentinel);
    CHECK(rns8_export_exact_wide_signed_limbs(hip, plan, c_matrix, limbs.data(), 2, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs == std::vector<uint64_t>(4, sentinel));
    CHECK(c_matrix->host_residues_current);
    CHECK_FALSE(c_matrix->device_residues_current);
    CHECK(c_matrix->hip_upload_buffer == nullptr);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    CHECK(c_matrix->hip_status_buffer == nullptr);

    upload_exact_residues_to_hip(c_matrix);
    REQUIRE(rns8_export_exact_wide_signed_limbs(hip, plan, c_matrix, limbs.data(), 2, 2) == RNS8_SUCCESS);
    CHECK(limbs[0] == std::numeric_limits<uint64_t>::max());
    CHECK(limbs[1] == std::numeric_limits<uint64_t>::max());
    CHECK(limbs[2] == 1);
    CHECK(limbs[3] == 0);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_unsigned_desc(1, 2, 1, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_matrix* c_matrix = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    auto c_desc = matrix_desc(1, 2, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    REQUIRE(rns8_create_matrix(hip, &c_desc, &c_matrix) == RNS8_SUCCESS);
    fill_exact_residue_matrix(c_matrix, {boost::multiprecision::cpp_int(1), boost::multiprecision::cpp_int(0)});

    constexpr uint64_t sentinel = 0x7b7b7b7b7b7b7b7bull;
    std::vector<uint64_t> limbs(4, sentinel);
    CHECK(rns8_export_exact_wide_unsigned_limbs(hip, plan, c_matrix, limbs.data(), 2, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs == std::vector<uint64_t>(4, sentinel));
    CHECK(c_matrix->host_residues_current);
    CHECK_FALSE(c_matrix->device_residues_current);
    CHECK(c_matrix->hip_upload_buffer == nullptr);
    CHECK(c_matrix->hip_export_buffer == nullptr);
    CHECK(c_matrix->hip_status_buffer == nullptr);

    upload_exact_residues_to_hip(c_matrix);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, plan, c_matrix, limbs.data(), 2, 2) == RNS8_SUCCESS);
    CHECK(limbs[0] == 1);
    CHECK(limbs[1] == 0);
    CHECK(limbs[2] == 0);
    CHECK(limbs[3] == 0);

    rns8_destroy_matrix(c_matrix);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(hip);
}

TEST_CASE("direct HIP exact-wide unsigned export matches CPU fixed-width boundary ABI") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide unsigned export boundary smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = exact_unsigned_desc(1, 2, 1, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = exact_unsigned_desc(1, 2, 1, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  auto c_desc = matrix_desc(1, 2, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  const boost::multiprecision::cpp_int two_to_64 = boost::multiprecision::cpp_int(1) << 64;
  const boost::multiprecision::cpp_int max_u64 = two_to_64 - 1;
  fill_exact_residue_matrix(cpu_c, {two_to_64, max_u64});
  fill_exact_residue_matrix(hip_c, {two_to_64, max_u64});
  upload_exact_residues_to_hip(hip_c);

  constexpr uint32_t limb_count = 2;
  constexpr int64_t limb_ld = 3;
  std::vector<uint64_t> cpu_limbs(static_cast<std::size_t>(limb_ld * limb_count), 0x4b4b4b4b4b4b4b4bull);
  std::vector<uint64_t> hip_limbs(static_cast<std::size_t>(limb_ld * limb_count), 0x4b4b4b4b4b4b4b4bull);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_limbs.data(), limb_ld, limb_count) ==
          RNS8_SUCCESS);
  CHECK(hip_limbs == cpu_limbs);
  CHECK(hip_limbs[0] == 0);
  CHECK(hip_limbs[1] == 1);
  CHECK(hip_limbs[2] == std::numeric_limits<uint64_t>::max());
  CHECK(hip_limbs[3] == 0);
  CHECK(hip_limbs[static_cast<std::size_t>(2 * limb_count)] == 0x4b4b4b4b4b4b4b4bull);
  CHECK(hip_limbs[static_cast<std::size_t>(2 * limb_count + 1)] == 0x4b4b4b4b4b4b4b4bull);

  std::vector<uint64_t> cpu_one_limb(2, 0x5151515151515151ull);
  std::vector<uint64_t> hip_one_limb(2, 0x5151515151515151ull);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_one_limb.data(), 2, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_one_limb.data(), 2, 1) == RNS8_RANGE_ERROR);
  CHECK(cpu_one_limb == std::vector<uint64_t>(2, 0x5151515151515151ull));
  CHECK(hip_one_limb == std::vector<uint64_t>(2, 0x5151515151515151ull));

  std::vector<uint64_t> cpu_wide(64, 0x6262626262626262ull);
  std::vector<uint64_t> hip_wide(64, 0x6262626262626262ull);
  fill_exact_residue_matrix(cpu_c, {boost::multiprecision::cpp_int(1), boost::multiprecision::cpp_int(0)});
  fill_exact_residue_matrix(hip_c, {boost::multiprecision::cpp_int(1), boost::multiprecision::cpp_int(0)});
  upload_exact_residues_to_hip(hip_c);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_wide.data(), 2, 32) == RNS8_SUCCESS);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_wide.data(), 2, 32) == RNS8_SUCCESS);
  CHECK(hip_wide == cpu_wide);
  CHECK(hip_wide[0] == 1);
  for (std::size_t limb = 1; limb < 32; ++limb) {
    CHECK(hip_wide[limb] == 0);
  }

  std::vector<uint64_t> wrong_export(4, 0x7373737373737373ull);
  CHECK(rns8_export_exact_wide_signed_limbs(cpu, cpu_plan, cpu_c, wrong_export.data(), 2, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8_export_exact_wide_signed_limbs(hip, hip_plan, hip_c, wrong_export.data(), 2, 2) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(wrong_export == std::vector<uint64_t>(4, 0x7373737373737373ull));

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide unsigned high-bit magnitude matches CPU fixed-width ABI") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide unsigned high-bit smoke");
  }

  rns8_context* cpu = create_context(RNS8_BACKEND_CPU_REFERENCE);
  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto cpu_desc = exact_unsigned_desc(1, 2, 1, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_desc = exact_unsigned_desc(1, 2, 1, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* cpu_plan = nullptr;
  rns8_plan* hip_plan = nullptr;
  rns8_matrix* cpu_c = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_plan(cpu, &cpu_desc, &cpu_plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  auto c_desc = matrix_desc(1, 2, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  REQUIRE(rns8_create_matrix(cpu, &c_desc, &cpu_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  const boost::multiprecision::cpp_int high_bit = boost::multiprecision::cpp_int(1) << 127u;
  fill_exact_residue_matrix(cpu_c, {high_bit, boost::multiprecision::cpp_int(0)});
  fill_exact_residue_matrix(hip_c, {high_bit, boost::multiprecision::cpp_int(0)});
  upload_exact_residues_to_hip(hip_c);

  std::vector<uint64_t> cpu_two(4, 0x8c8c8c8c8c8c8c8cull);
  std::vector<uint64_t> hip_two(4, 0x8c8c8c8c8c8c8c8cull);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_two.data(), 2, 2) == RNS8_SUCCESS);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_two.data(), 2, 2) == RNS8_SUCCESS);
  CHECK(hip_two == cpu_two);
  CHECK(hip_two[0] == 0);
  CHECK(hip_two[1] == (uint64_t{1} << 63u));
  CHECK(hip_two[2] == 0);
  CHECK(hip_two[3] == 0);

  std::vector<uint64_t> cpu_one(2, 0x9d9d9d9d9d9d9d9dull);
  std::vector<uint64_t> hip_one(2, 0x9d9d9d9d9d9d9d9dull);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_one.data(), 2, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_one.data(), 2, 1) == RNS8_RANGE_ERROR);
  CHECK(cpu_one == std::vector<uint64_t>(2, 0x9d9d9d9d9d9d9d9dull));
  CHECK(hip_one == std::vector<uint64_t>(2, 0x9d9d9d9d9d9d9d9dull));

  std::vector<uint64_t> cpu_wide(64, 0xaeaeaeaeaeaeaeaeull);
  std::vector<uint64_t> hip_wide(64, 0xaeaeaeaeaeaeaeaeull);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, cpu_wide.data(), 2, 32) == RNS8_SUCCESS);
  REQUIRE(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, hip_wide.data(), 2, 32) == RNS8_SUCCESS);
  CHECK(hip_wide == cpu_wide);
  CHECK(hip_wide[0] == 0);
  CHECK(hip_wide[1] == (uint64_t{1} << 63u));
  for (std::size_t limb = 2; limb < 32; ++limb) {
    CHECK(hip_wide[limb] == 0);
  }

  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(cpu_c);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_plan(cpu_plan);
  rns8_destroy_context(hip);
  rns8_destroy_context(cpu);
}

TEST_CASE("direct HIP exact-wide descriptors reject stale bounded metadata") {
  if (!hip_available()) {
    SKIP("no HIP device available for exact-wide descriptor rejection smoke");
  }

  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr uint64_t tile_bound = 1;

  {
    auto desc = exact_signed_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    desc.bound = 1;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(hip, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    desc = exact_signed_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    desc.tile_bounds = &tile_bound;
    desc.tile_bounds_count = 1;
    CHECK(rns8_create_plan(hip, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto matrix = matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_GLOBAL_MAX_ABS);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(hip, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }

  {
    auto desc = exact_unsigned_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
    desc.bound = 1;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(hip, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    desc = exact_unsigned_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    desc.tile_bounds = &tile_bound;
    desc.tile_bounds_count = 1;
    CHECK(rns8_create_plan(hip, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto matrix = matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(hip, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }

  {
    auto desc = exact_signed_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_matrix* storage = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    auto matrix = matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
    matrix.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    REQUIRE(rns8_create_matrix(hip, &matrix, &storage) == RNS8_SUCCESS);
    uint64_t limbs[2] = {0xcdcdcdcdcdcdcdcdull, 0xcecececececececeull};
    CHECK(rns8_export_exact_wide_signed_limbs(hip, plan, storage, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0xcdcdcdcdcdcdcdcdull);
    CHECK(limbs[1] == 0xcecececececececeull);
    rns8_destroy_matrix(storage);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_unsigned_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    rns8_matrix* storage = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);
    auto matrix = matrix_desc(1, 1, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
    matrix.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
    REQUIRE(rns8_create_matrix(hip, &matrix, &storage) == RNS8_SUCCESS);
    uint64_t limbs[2] = {0xdfdfdfdfdfdfdfdfull, 0xe0e0e0e0e0e0e0e0ull};
    CHECK(rns8_export_exact_wide_unsigned_limbs(hip, plan, storage, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0xdfdfdfdfdfdfdfdfull);
    CHECK(limbs[1] == 0xe0e0e0e0e0e0e0e0ull);
    rns8_destroy_matrix(storage);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_signed_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);

    auto bounded = matrix_desc(1, 1, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    rns8_matrix* bounded_storage = nullptr;
    REQUIRE(rns8_create_matrix(hip, &bounded, &bounded_storage) == RNS8_SUCCESS);
    uint64_t limbs[2] = {0xababababababababull, 0xbcbcbcbcbcbcbcbcull};
    CHECK(rns8_export_exact_wide_signed_limbs(hip, plan, bounded_storage, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0xababababababababull);
    CHECK(limbs[1] == 0xbcbcbcbcbcbcbcbcull);

    auto wrap = matrix_desc(1, 1, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
    rns8_matrix* wrap_storage = nullptr;
    REQUIRE(rns8_create_matrix(hip, &wrap, &wrap_storage) == RNS8_SUCCESS);
    CHECK(rns8_export_exact_wide_signed_limbs(hip, plan, wrap_storage, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0xababababababababull);
    CHECK(limbs[1] == 0xbcbcbcbcbcbcbcbcull);

    rns8_destroy_matrix(wrap_storage);
    rns8_destroy_matrix(bounded_storage);
    rns8_destroy_plan(plan);
  }

  {
    auto desc = exact_unsigned_desc(1, 1, 1, RNS8_BACKEND_HIP_DIRECT);
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(hip, &desc, &plan) == RNS8_SUCCESS);

    auto bounded = matrix_desc(1, 1, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    rns8_matrix* bounded_storage = nullptr;
    REQUIRE(rns8_create_matrix(hip, &bounded, &bounded_storage) == RNS8_SUCCESS);
    uint64_t limbs[2] = {0xcbcbcbcbcbcbcbcbull, 0xdcdcdcdcdcdcdcdcull};
    CHECK(rns8_export_exact_wide_unsigned_limbs(hip, plan, bounded_storage, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0xcbcbcbcbcbcbcbcbull);
    CHECK(limbs[1] == 0xdcdcdcdcdcdcdcdcull);

    auto wrap = matrix_desc(1, 1, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
    rns8_matrix* wrap_storage = nullptr;
    REQUIRE(rns8_create_matrix(hip, &wrap, &wrap_storage) == RNS8_SUCCESS);
    CHECK(rns8_export_exact_wide_unsigned_limbs(hip, plan, wrap_storage, limbs, 1, 2) == RNS8_INVALID_ARGUMENT);
    CHECK(limbs[0] == 0xcbcbcbcbcbcbcbcbull);
    CHECK(limbs[1] == 0xdcdcdcdcdcdcdcdcull);

    rns8_destroy_matrix(wrap_storage);
    rns8_destroy_matrix(bounded_storage);
    rns8_destroy_plan(plan);
  }

  rns8_destroy_context(hip);
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
  CHECK(hip_c->hip_status_buffer == nullptr);
  CHECK(hip_limbs == cpu_limbs);
  CHECK(hip_limbs[static_cast<std::size_t>((0 * limb_ld + 2) * limb_count)] == 0xbbbbbbbbbbbbbbbbull);
  constexpr uint64_t unsigned_range_sentinel = 0x3434343434343434ull;
  std::vector<uint64_t> too_few_cpu(static_cast<std::size_t>(m * n), unsigned_range_sentinel);
  std::vector<uint64_t> too_few_hip(static_cast<std::size_t>(m * n), unsigned_range_sentinel);
  CHECK(rns8_export_exact_wide_unsigned_limbs(cpu, cpu_plan, cpu_c, too_few_cpu.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(rns8_export_exact_wide_unsigned_limbs(hip, hip_plan, hip_c, too_few_hip.data(), n, 1) == RNS8_RANGE_ERROR);
  CHECK(hip_c->hip_status_buffer != nullptr);
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

TEST_CASE("direct HIP private tiled bounded wrappers reject malformed selected-prefix metadata") {
  if (!hip_available()) {
    SKIP("no HIP device available for direct HIP tiled selected-prefix rejection smoke");
  }

  constexpr int64_t m = 65;
  constexpr int64_t n = 65;
  constexpr int64_t k = 1;
  constexpr int8_t residue_sentinel = -91;
  const std::vector<uint64_t> bounds = {7, 1000, 7000000, 1000000000};

  rns8_context* hip = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto hip_desc = per_tile_unsigned_desc(m, n, k, bounds, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* hip_plan = nullptr;
  REQUIRE(rns8_create_plan(hip, &hip_desc, &hip_plan) == RNS8_SUCCESS);
  REQUIRE_FALSE(hip_plan->tile_schedule.empty());

  auto a_desc = matrix_desc(m, k, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto b_desc = matrix_desc(k, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  auto c_desc = matrix_desc(m, n, RNS8_BOUNDED_U64, RNS8_BOUND_PER_TILE_MAX_UNSIGNED);
  a_desc.tile_m = b_desc.tile_m = c_desc.tile_m = 64;
  a_desc.tile_n = b_desc.tile_n = c_desc.tile_n = 64;
  rns8_matrix* hip_a = nullptr;
  rns8_matrix* hip_b = nullptr;
  rns8_matrix* hip_c = nullptr;
  REQUIRE(rns8_create_matrix(hip, &a_desc, &hip_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &b_desc, &hip_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(hip, &c_desc, &hip_c) == RNS8_SUCCESS);

  std::fill(hip_c->residues.begin(), hip_c->residues.end(), residue_sentinel);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(
              hip->device_id, hip_c->hip_residues, hip_c->residues.data(), hip_c->hip_residue_bytes) ==
          RNS8_SUCCESS);

  rns8::detail::hip_direct_allocation_counters_reset();
  const auto initial_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  auto check_bad_entries = [&](const std::vector<rns8_plan_tile_schedule_entry>& bad_entries) {
    CHECK(rns8::detail::hip_direct_gemm_rns_tiled_device(
              hip->device_id,
              hip_a->hip_residues,
              hip_b->hip_residues,
              hip_c->hip_residues,
              m,
              n,
              k,
              k,
              n,
              n,
              bad_entries.data(),
              bad_entries.size()) == RNS8_INVALID_ARGUMENT);
    REQUIRE(rns8::detail::hip_direct_copy_device_to_host(
                hip->device_id, hip_c->residues.data(), hip_c->hip_residues, hip_c->hip_residue_bytes) ==
            RNS8_SUCCESS);
    CHECK(std::all_of(hip_c->residues.begin(), hip_c->residues.end(), [&](int8_t value) {
      return value == residue_sentinel;
    }));

    const auto repeated_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(repeated_allocations.allocate_calls == initial_allocations.allocate_calls);
    CHECK(repeated_allocations.free_calls == initial_allocations.free_calls);
    CHECK(repeated_allocations.allocated_bytes == initial_allocations.allocated_bytes);
  };

  {
    std::vector<rns8_plan_tile_schedule_entry> bad_entries = hip_plan->tile_schedule;
    bad_entries[0].required_prefix = bad_entries[0].selected_prefix + 1;
    check_bad_entries(bad_entries);
  }
  {
    std::vector<rns8_plan_tile_schedule_entry> bad_entries = hip_plan->tile_schedule;
    bad_entries[0].group_index += 1;
    check_bad_entries(bad_entries);
  }
  {
    std::vector<rns8_plan_tile_schedule_entry> bad_entries = hip_plan->tile_schedule;
    bad_entries[1].tile_row = bad_entries[0].tile_row;
    bad_entries[1].tile_col = bad_entries[0].tile_col;
    check_bad_entries(bad_entries);
  }
  {
    std::vector<rns8_plan_tile_schedule_entry> bad_entries = hip_plan->tile_schedule;
    bad_entries.back().tile_col += 1;
    check_bad_entries(bad_entries);
  }
  {
    std::vector<rns8_plan_tile_schedule_entry> bad_entries = hip_plan->tile_schedule;
    bad_entries[0].row_extent -= 1;
    check_bad_entries(bad_entries);
  }
  rns8_destroy_matrix(hip_c);
  rns8_destroy_matrix(hip_b);
  rns8_destroy_matrix(hip_a);
  rns8_destroy_plan(hip_plan);
  rns8_destroy_context(hip);
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
