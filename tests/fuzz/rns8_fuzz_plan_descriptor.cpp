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
  return static_cast<int64_t>((value % 8u) + 1u);
}

void fuzz_plan_descriptor(const uint8_t* data, std::size_t size) {
  const uint8_t* cursor = data;
  const uint8_t* end = data + size;

  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;

  rns8_context* ctx = nullptr;
  if (rns8_create_context(-1, &options, &ctx) != RNS8_SUCCESS || !ctx) {
    return;
  }

  const uint64_t mode = read_u64(cursor, end);
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = small_dim(read_u64(cursor, end));
  desc.n = small_dim(read_u64(cursor, end));
  desc.k = small_dim(read_u64(cursor, end));
  desc.tile_m = 64;
  desc.tile_n = 64;

  std::vector<uint64_t> tile_bounds;
  switch (mode % 5u) {
    case 0:
      desc.semantics = RNS8_BOUNDED_I64;
      desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
      desc.bound = (read_u64(cursor, end) % 4096u) + 1u;
      desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
      break;
    case 1:
      desc.semantics = RNS8_BOUNDED_U64;
      desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
      desc.bound = (read_u64(cursor, end) % 4096u) + 1u;
      desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
      break;
    case 2:
      desc.semantics = RNS8_EXACT_WIDE_SIGNED;
      desc.bound_kind = RNS8_BOUND_NONE;
      desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
      break;
    case 3:
      desc.semantics = RNS8_FINITE_RING_U8;
      desc.bound_kind = RNS8_BOUND_NONE;
      desc.finite_modulus = static_cast<uint32_t>((read_u64(cursor, end) % 255u) + 2u);
      break;
    default:
      desc.semantics = RNS8_WRAP_U64_MOD_2_64;
      desc.bound_kind = RNS8_BOUND_NONE;
      desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
      options.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
      rns8_destroy_context(ctx);
      ctx = nullptr;
      if (rns8_create_context(-1, &options, &ctx) != RNS8_SUCCESS || !ctx) {
        return;
      }
      break;
  }

  if ((read_u64(cursor, end) & 1u) != 0 && desc.semantics != RNS8_WRAP_U64_MOD_2_64 &&
      desc.semantics != RNS8_FINITE_RING_U8) {
    desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_ABS;
    desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
    const uint64_t tile_count = 1;
    tile_bounds.assign(static_cast<std::size_t>(tile_count), (read_u64(cursor, end) % 4096u) + 1u);
    desc.tile_bounds = tile_bounds.data();
    desc.tile_bounds_count = tile_count;
  }

  rns8_plan* plan = nullptr;
  const rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status == RNS8_SUCCESS && plan) {
    rns8_plan_schedule_info schedule{};
    schedule.struct_size = sizeof(schedule);
    schedule.abi_version = RNS8_ABI_VERSION;
    (void)rns8_get_plan_schedule_info(plan, &schedule);

    rns8_plan_backend_info backend{};
    backend.struct_size = sizeof(backend);
    backend.abi_version = RNS8_ABI_VERSION;
    (void)rns8_get_plan_backend_info(plan, &backend);

    rns8_workspace* workspace = nullptr;
    if (rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS) {
      (void)rns8_destroy_workspace(workspace);
    }
  }
  (void)rns8_destroy_plan(plan);
  (void)rns8_destroy_context(ctx);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
  fuzz_plan_descriptor(data, size);
  return 0;
}
