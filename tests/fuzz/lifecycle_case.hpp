#pragma once

#include "arithmetic_replay.hpp"
#include <array>
#include <cstring>

namespace rns8_fuzz::history_support {
using replay_support::abi;
using replay_support::check;
using replay_support::fail;

// R8H1 followed by 1..32 bytes. Low nibble selects one of twelve legal API
// transitions; the high nibble maps the next packed scalar into [-4,4].
inline bool admitted(const uint8_t* data, std::size_t size) {
  return data && size >= 5 && size <= 36 && std::memcmp(data, "R8H1", 4) == 0;
}
struct Observation {
  uint32_t transitions = 0, stale_rejections = 0, successful_executions = 0, busy_checks = 0;
  uint32_t cache_queries = 0, unsupported_caches = 0;
};
inline Observation replay(const uint8_t* data, std::size_t size) {
  Observation observed;
  if (!admitted(data, size))
    return observed;
  replay_support::Owner owner;
  auto options = abi<rns8_context_options>();
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  check(rns8_create_context(-1, &options, &owner.ctx), RNS8_SUCCESS, "history context");
  auto plan = abi<rns8_gemm_desc>();
  plan.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  plan.semantics = RNS8_BOUNDED_I64;
  plan.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  plan.bound = 32;
  plan.m = plan.n = plan.k = 1;
  plan.tile_m = plan.tile_n = 128;
  plan.max_prefix = 20;
  check(rns8_create_plan(owner.ctx, &plan, &owner.plan), RNS8_SUCCESS, "history plan");
  auto matrix = abi<rns8_matrix_desc>();
  matrix.rows = matrix.cols = matrix.logical_ld = 1;
  matrix.semantics = plan.semantics;
  matrix.bound_kind = plan.bound_kind;
  matrix.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  matrix.max_prefix = plan.max_prefix;
  matrix.tile_m = matrix.tile_n = 128;
  for (auto slot : {&owner.a, &owner.b, &owner.c})
    check(rns8_create_matrix(owner.ctx, &matrix, slot), RNS8_SUCCESS, "history matrix");
  check(rns8_create_workspace(owner.ctx, owner.plan, &owner.work), RNS8_SUCCESS,
        "history workspace");
  int64_t a = 2, b = 3;
  check(rns8_pack_i64(owner.ctx, owner.a, &a, 1, 0), RNS8_SUCCESS, "history initial A");
  check(rns8_pack_i64(owner.ctx, owner.b, &b, 1, 0), RNS8_SUCCESS, "history initial B");
  auto a_info = abi<rns8_matrix_storage_info>(), b_info = a_info;
  check(rns8_get_matrix_storage_info(owner.a, &a_info), RNS8_SUCCESS, "history A identity");
  check(rns8_get_matrix_storage_info(owner.b, &b_info), RNS8_SUCCESS, "history B identity");
  if (!a_info.matrix_instance_id || !b_info.matrix_instance_id ||
      a_info.matrix_instance_id == b_info.matrix_instance_id ||
      a_info.source_version != b_info.source_version)
    fail({"state", "history distinct same-version instances"});
  uint64_t a_generation = 1, b_generation = 1, prepared_a = 0, prepared_b = 0;
  // Matrix creation publishes a complete zero basis before any producer runs.
  bool current = true;
  cpp_int result = 0;
  rns8_completion_token previous{};
  auto export_value = [&]() {
    std::array<int64_t, 3> output{91, 91, 91};
    check(rns8_export_i64(owner.ctx, owner.plan, owner.c, &output[1], 1),
          current ? RNS8_SUCCESS : RNS8_INVALID_ARGUMENT, "history export status");
    const auto expected = current ? result.convert_to<int64_t>() : int64_t(91);
    if (output != std::array<int64_t, 3>{91, expected, 91})
      fail({"value_or_padding", "history export"});
  };
  for (std::size_t index = 4; index < size; ++index) {
    const auto byte = data[index];
    switch ((byte & 15) % 12) {
    case 0:
      a = int64_t((byte >> 4) % 9) - 4;
      check(rns8_pack_i64(owner.ctx, owner.a, &a, 1, 0), RNS8_SUCCESS, "history repack A");
      ++a_generation;
      break;
    case 1:
      b = int64_t((byte >> 4) % 9) - 4;
      check(rns8_pack_i64(owner.ctx, owner.b, &b, 1, 0), RNS8_SUCCESS, "history repack B");
      ++b_generation;
      break;
    case 2:
      check(rns8_gemm_rns(owner.ctx, owner.plan, owner.a, owner.b, owner.c, owner.work),
            RNS8_SUCCESS, "history GEMM");
      result = cpp_int(a) * cpp_int(b);
      current = true;
      ++observed.successful_executions;
      break;
    case 3:
      export_value();
      break;
    case 4:
      check(rns8_destroy_prepared_execution(owner.prepared), RNS8_SUCCESS,
            "history replace prepared");
      owner.prepared = nullptr;
      check(rns8_prepare_gemm(owner.ctx, owner.plan, owner.a, owner.b, owner.c, owner.work,
                              &owner.prepared),
            RNS8_SUCCESS, "history prepare");
      prepared_a = a_generation;
      prepared_b = b_generation;
      if (previous.epoch) {
        auto completion = abi<rns8_completion_info>();
        check(rns8_get_prepared_completion(owner.prepared, previous, &completion),
              RNS8_INVALID_ARGUMENT, "history foreign prepared token");
      }
      break;
    case 5:
      if (owner.prepared) {
        const bool fresh = a_generation == prepared_a && b_generation == prepared_b;
        rns8_completion_token token{17, 19};
        check(rns8_execute_prepared(owner.prepared, &token),
              fresh ? RNS8_SUCCESS : RNS8_INVALID_ARGUMENT, "history prepared currentness");
        if (fresh) {
          if (!token.prepared_id || !token.epoch)
            fail({"state", "history missing completion token"});
          previous = token;
          result = cpp_int(a) * cpp_int(b);
          current = true;
          ++observed.successful_executions;
          for (unsigned query = 0; query < 2; ++query) {
            auto completion = abi<rns8_completion_info>();
            check(rns8_get_prepared_completion(owner.prepared, token, &completion), RNS8_SUCCESS,
                  "history repeat completion query");
            if (!completion.completed || !completion.output_is_current ||
                completion.operation_status != RNS8_SUCCESS)
              fail({"state", "history completion publication"});
          }
        } else {
          if (token.prepared_id != 17 || token.epoch != 19)
            fail({"no_write", "history rejected token"});
          ++observed.stale_rejections;
        }
      }
      break;
    case 6:
      if (owner.prepared) {
        check(rns8_refresh_prepared_inputs(owner.prepared), RNS8_SUCCESS, "history refresh");
        prepared_a = a_generation;
        prepared_b = b_generation;
      }
      break;
    case 7: {
      rns8_prepared_execution* rejected = owner.prepared;
      check(rns8_prepare_gemm(owner.ctx, owner.plan, owner.a, owner.b, owner.a, owner.work,
                              &rejected),
            RNS8_INVALID_ARGUMENT, "history alias preparation");
      if (rejected)
        fail({"no_write", "history rejected prepared output"});
      break;
    }
    case 8:
      check(rns8_destroy_prepared_execution(owner.prepared), RNS8_SUCCESS,
            "history close prepared");
      owner.prepared = nullptr;
      break;
    case 9:
      if (owner.prepared) {
        check(rns8_destroy_matrix(owner.a), RNS8_BUSY, "history retained matrix");
        check(rns8_destroy_plan(owner.plan), RNS8_BUSY, "history retained plan");
        check(rns8_destroy_workspace(owner.work), RNS8_BUSY, "history retained workspace");
        ++observed.busy_checks;
      }
      break;
    case 10: {
      auto key = abi<rns8_prepack_cache_key_info>();
      check(rns8_get_prepack_cache_key_info(owner.plan, owner.b, RNS8_OPERAND_B, &key),
            RNS8_SUCCESS, "history diagnostic cache key");
      if (!key.cache_key_valid || key.source_version || key.reusable_prepack_cache_available ||
          key.production_prepack_cache_available)
        fail({"state", "history CPU cache eligibility"});
      std::array<unsigned char, sizeof(key)> unchanged{};
      std::memcpy(unchanged.data(), &key, sizeof(key));
      check(rns8_get_prepack_cache_key_info(owner.plan, owner.b,
                                            static_cast<rns8_operand_role>(UINT32_MAX), &key),
            RNS8_INVALID_ARGUMENT, "history invalid cache role");
      if (std::memcmp(&key, unchanged.data(), sizeof(key)))
        fail({"no_write", "history invalid cache key"});
      ++observed.cache_queries;
      break;
    }
    case 11: {
      rns8_prepack_cache* cache = nullptr;
      check(rns8_create_prepack_cache(owner.ctx, owner.plan, owner.b, RNS8_OPERAND_B, &cache),
            RNS8_UNSUPPORTED_BACKEND, "history explicit CPU prepack unsupported");
      if (cache)
        fail({"state", "history unsupported cache output"});
      ++observed.unsupported_caches;
      break;
    }
    }
    export_value();
    ++observed.transitions;
  }
  return observed;
}
} // namespace rns8_fuzz::history_support
