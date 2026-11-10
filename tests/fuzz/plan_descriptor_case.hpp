#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "rns8/rns8.h"
#include "lifecycle_case.hpp"

namespace rns8_fuzz::descriptor_support {
inline void check(rns8_status actual, rns8_status expected, const char* phase) {
  if (actual != expected) {
    std::fprintf(stderr, "descriptor fuzz %s: status %u expected %u\n", phase, unsigned(actual),
                 unsigned(expected));
    std::abort();
  }
}
inline void require(bool value, const char* phase) {
  if (!value) {
    std::fprintf(stderr, "descriptor fuzz %s\n", phase);
    std::abort();
  }
}
template <class T> T abi() {
  T v{};
  v.struct_size = sizeof(v);
  v.abi_version = RNS8_ABI_VERSION;
  return v;
}
struct Reader {
  const uint8_t* data;
  std::size_t size, position = 0;
  uint8_t byte() { return position < size ? data[position++] : 0; }
  uint64_t integer() {
    uint64_t v = 0;
    for (unsigned x = 0; x < 8; ++x)
      v |= uint64_t(byte()) << (8 * x);
    return v;
  }
};
inline void replay(const uint8_t* data, std::size_t size) {
  Reader input{data, size};
  const auto mode = input.byte() % 6;
  auto options = abi<rns8_context_options>();
  options.requested_backend =
      mode == 3 ? RNS8_BACKEND_WRAP64_BYTE_LIMB : RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* context = nullptr;
  check(rns8_create_context(-1, &options, &context), RNS8_SUCCESS, "create context");
  require(context, "null context");
  auto desc = abi<rns8_gemm_desc>();
  desc.requested_backend = options.requested_backend;
  desc.m = 1 + input.byte() % 8;
  desc.n = 1 + input.byte() % 8;
  desc.k = 1 + input.byte() % 8;
  desc.tile_m = desc.tile_n = 64;
  constexpr rns8_semantics semantic[] = {RNS8_BOUNDED_I64,       RNS8_BOUNDED_U64,
                                         RNS8_FINITE_RING_U8,    RNS8_WRAP_U64_MOD_2_64,
                                         RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED};
  desc.semantics = semantic[mode];
  if (mode < 2) {
    desc.bound_kind = mode ? RNS8_BOUND_GLOBAL_MAX_UNSIGNED : RNS8_BOUND_GLOBAL_MAX_ABS;
    desc.bound = 1 + input.integer() % 4096;
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  } else {
    desc.bound_kind = RNS8_BOUND_NONE;
    if (mode == 2)
      desc.finite_modulus = 2 + input.byte() % 255;
    if (mode >= 4) {
      desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
      desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    }
  }
  std::vector<uint64_t> bounds;
  if (mode < 2 && (input.byte() & 1)) {
    desc.bound_kind = mode ? RNS8_BOUND_PER_TILE_MAX_UNSIGNED : RNS8_BOUND_PER_TILE_MAX_ABS;
    desc.bound = 0;
    desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
    bounds = {input.integer() % 4096};
    desc.tile_bounds = bounds.data();
    desc.tile_bounds_count = 1;
  }
  rns8_plan* plan = nullptr;
  check(rns8_create_plan(context, &desc, &plan), RNS8_SUCCESS, "valid plan");
  require(plan, "null valid plan");
  auto before = abi<rns8_plan_schedule_info>();
  check(rns8_get_plan_schedule_info(plan, &before), RNS8_SUCCESS, "schedule");
  auto backend = abi<rns8_plan_backend_info>();
  check(rns8_get_plan_backend_info(plan, &backend), RNS8_SUCCESS, "backend");
  require(backend.backend == options.requested_backend, "explicit backend changed");
  rns8_workspace* workspace = nullptr;
  check(rns8_create_workspace(context, plan, &workspace), RNS8_SUCCESS, "workspace");
  require(workspace, "null workspace");
  auto invalid = desc;
  switch (input.byte() % 15) {
  case 0:
    invalid.m = 0;
    break;
  case 1:
    invalid.n = 0;
    break;
  case 2:
    invalid.k = 0;
    break;
  case 3:
    invalid.struct_size = sizeof(invalid) - 1;
    break;
  case 4:
    invalid.abi_version = RNS8_ABI_VERSION + 1;
    break;
  case 5:
    invalid.semantics = static_cast<rns8_semantics>(UINT32_MAX);
    break;
  case 6:
    invalid.bound_kind = static_cast<rns8_bound_kind>(UINT32_MAX);
    break;
  case 7:
    invalid.tile_m = 3;
    break;
  case 8:
    invalid.tile_n = 3;
    break;
  case 9:
    invalid.flags = UINT32_MAX;
    break;
  case 10:
    invalid.lhs_bound = 1;
    break;
  case 11:
    invalid.rhs_bound = 1;
    break;
  case 12:
    invalid.zero_a_rows_count = 1;
    break;
  case 13:
    invalid.zero_b_cols_count = 1;
    break;
  default:
    invalid.bound_kind = RNS8_BOUND_NONE;
    invalid.tile_bounds = bounds.data();
    invalid.tile_bounds_count = 1;
    break;
  }
  // Rejected descriptor preflight preserves the caller's output slot and must
  // not alter the live plan that supplied its initial value.
  rns8_plan* rejected = plan;
  check(rns8_create_plan(context, &invalid, &rejected), RNS8_INVALID_ARGUMENT, "invalid plan");
  require(rejected == plan, "rejected output slot changed");
  auto after = abi<rns8_plan_schedule_info>();
  check(rns8_get_plan_schedule_info(plan, &after), RNS8_SUCCESS, "valid plan after rejection");
  require(before.tile_count == after.tile_count &&
              before.effective_bound == after.effective_bound &&
              before.min_selected_prefix == after.min_selected_prefix &&
              before.max_selected_prefix == after.max_selected_prefix,
          "rejection changed live plan");
  check(rns8_destroy_workspace(workspace), RNS8_SUCCESS, "destroy workspace");
  check(rns8_destroy_plan(plan), RNS8_SUCCESS, "destroy plan");
  check(rns8_destroy_context(context), RNS8_SUCCESS, "destroy context");
  if (input.position < size)
    rns8_fuzz::history_support::replay(data + input.position, size - input.position);
}
} // namespace rns8_fuzz::descriptor_support
