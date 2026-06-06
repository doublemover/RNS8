#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rns8/rns8.h"

namespace {

uint64_t read_u64(const uint8_t*& cursor, const uint8_t* end, uint64_t fallback = 0) {
  uint64_t value = fallback;
  for (uint32_t i = 0; i < 8 && cursor < end; ++i) {
    value ^= static_cast<uint64_t>(*cursor++) << ((i % 8) * 8);
  }
  return value;
}

int64_t small_dim(uint64_t value) {
  return static_cast<int64_t>((value % 4u) + 1u);
}

rns8_matrix_desc matrix_desc(
    int64_t rows,
    int64_t cols,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t prefix) {
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

void destroy_matrices(rns8_matrix* a, rns8_matrix* b, rns8_matrix* c) {
  (void)rns8_destroy_matrix(a);
  (void)rns8_destroy_matrix(b);
  (void)rns8_destroy_matrix(c);
}

void fuzz_bounded_export(
    rns8_context* ctx,
    const uint8_t* data,
    std::size_t size,
    rns8_semantics semantics) {
  const uint8_t* cursor = data;
  const uint8_t* end = data + size;
  const int64_t m = small_dim(read_u64(cursor, end));
  const int64_t n = small_dim(read_u64(cursor, end));
  const int64_t k = small_dim(read_u64(cursor, end));
  const rns8_bound_kind bound_kind =
      semantics == RNS8_BOUNDED_I64 ? RNS8_BOUND_GLOBAL_MAX_ABS : RNS8_BOUND_GLOBAL_MAX_UNSIGNED;

  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = bound_kind;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = 4096;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;

  rns8_plan* plan = nullptr;
  if (rns8_create_plan(ctx, &desc, &plan) != RNS8_SUCCESS || !plan) {
    return;
  }

  rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr;
  const rns8_matrix_desc a_desc = matrix_desc(m, k, semantics, bound_kind, RNS8_DEFAULT_BOUNDED_PREFIX);
  const rns8_matrix_desc b_desc = matrix_desc(k, n, semantics, bound_kind, RNS8_DEFAULT_BOUNDED_PREFIX);
  const rns8_matrix_desc c_desc = matrix_desc(m, n, semantics, bound_kind, RNS8_DEFAULT_BOUNDED_PREFIX);
  if (rns8_create_matrix(ctx, &a_desc, &a) != RNS8_SUCCESS ||
      rns8_create_matrix(ctx, &b_desc, &b) != RNS8_SUCCESS ||
      rns8_create_matrix(ctx, &c_desc, &c) != RNS8_SUCCESS) {
    destroy_matrices(a, b, c);
    (void)rns8_destroy_plan(plan);
    return;
  }
  rns8_workspace* workspace = nullptr;
  if (rns8_create_workspace(ctx, plan, &workspace) != RNS8_SUCCESS || !workspace) {
    destroy_matrices(a, b, c);
    (void)rns8_destroy_plan(plan);
    return;
  }

  const std::size_t a_count = static_cast<std::size_t>(m * k);
  const std::size_t b_count = static_cast<std::size_t>(k * n);
  std::vector<int64_t> a_i64(a_count, 0);
  std::vector<int64_t> b_i64(b_count, 0);
  std::vector<uint64_t> a_u64(a_count, 0);
  std::vector<uint64_t> b_u64(b_count, 0);
  for (std::size_t i = 0; i < a_count; ++i) {
    const uint64_t raw = read_u64(cursor, end);
    a_i64[i] = static_cast<int64_t>(raw % 7u) - 3;
    a_u64[i] = raw % 7u;
  }
  for (std::size_t i = 0; i < b_count; ++i) {
    const uint64_t raw = read_u64(cursor, end);
    b_i64[i] = static_cast<int64_t>(raw % 7u) - 3;
    b_u64[i] = raw % 7u;
  }

  rns8_status status = RNS8_INVALID_ARGUMENT;
  if (semantics == RNS8_BOUNDED_I64) {
    status = rns8_pack_i64(ctx, a, a_i64.data(), k, 1);
    if (status == RNS8_SUCCESS) {
      status = rns8_pack_i64(ctx, b, b_i64.data(), n, 1);
    }
    if (status == RNS8_SUCCESS) {
      status = rns8_gemm_rns(ctx, plan, a, b, c, workspace);
    }
    if (status == RNS8_SUCCESS) {
      std::vector<int64_t> out(static_cast<std::size_t>(m * n), 0);
      (void)rns8_export_i64(ctx, plan, c, out.data(), n);
      (void)rns8_export_i64(ctx, plan, c, out.data(), std::max<int64_t>(1, n - 1));
    }
  } else {
    status = rns8_pack_u64(ctx, a, a_u64.data(), k, 1);
    if (status == RNS8_SUCCESS) {
      status = rns8_pack_u64(ctx, b, b_u64.data(), n, 1);
    }
    if (status == RNS8_SUCCESS) {
      status = rns8_gemm_rns(ctx, plan, a, b, c, workspace);
    }
    if (status == RNS8_SUCCESS) {
      std::vector<uint64_t> out(static_cast<std::size_t>(m * n), 0);
      (void)rns8_export_u64(ctx, plan, c, out.data(), n);
      (void)rns8_export_u64(ctx, plan, c, out.data(), std::max<int64_t>(1, n - 1));
    }
  }

  (void)rns8_destroy_workspace(workspace);
  destroy_matrices(a, b, c);
  (void)rns8_destroy_plan(plan);
}

void fuzz_export_contract(const uint8_t* data, std::size_t size) {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  if (rns8_create_context(-1, &options, &ctx) != RNS8_SUCCESS || !ctx) {
    return;
  }
  const rns8_semantics semantics =
      (size == 0 || (data[0] & 1u) == 0) ? RNS8_BOUNDED_I64 : RNS8_BOUNDED_U64;
  fuzz_bounded_export(ctx, data, size, semantics);
  (void)rns8_destroy_context(ctx);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
  fuzz_export_contract(data, size);
  return 0;
}
