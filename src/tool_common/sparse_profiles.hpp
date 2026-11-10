#pragma once

#include "rns8/rns8.h"
#include "rns8/generated/metadata_registry.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rns8::tools {
inline nlohmann::json operation_support_json(const rns8_operation_support& r,
                                             const std::string& backend_name) {
  return {{"operation", r.operation},
          {"backend", backend_name},
          {"backend_id", r.backend},
          {"status", rns8_status_string(r.status)},
          {"status_code", uint32_t(r.status)},
          {"reason", r.reason_code},
          {"source", r.source_implemented},
          {"compiled", r.compiled_inventory},
          {"runtime_observed", r.runtime_observed},
          {"correctness_qualified", r.correctness_qualified},
          {"performance_qualified", r.performance_qualified},
          {"requires_binding", r.requires_binding},
          {"execution_modes", r.supported_execution_modes},
          {"resource_status", r.resource_status},
          {"device_bytes", r.required_device_bytes},
          {"host_bytes", r.required_host_bytes},
          {"pinned_host_bytes", r.required_pinned_host_bytes},
          {"setup_bytes_lower_bound", r.setup_bytes_lower_bound},
          {"alignment", r.alignment},
          {"safe_k", r.safe_k},
          {"target", r.target_id},
          {"offending_field", r.offending_field},
          {"explanation", r.explanation},
          {"provider_sha256", r.provider_identity_sha256}};
}

// These are concrete query inputs, never a second backend capability table.
// Actual values, structure, generation, budget and method still bind at prepare.
struct SparseProfile {
  const char* id;
  rns8_semantics semantics;
  uint32_t modulus;
  int64_t m, n, k, ldb;
};
inline constexpr std::array<SparseProfile, 18> sparse_profiles{{
    {"bounded-i64", RNS8_BOUNDED_I64, 0, 17, 19, 32, 0},
    {"bounded-u64", RNS8_BOUNDED_U64, 0, 17, 19, 32, 0},
    {"wide-signed-p20", RNS8_EXACT_WIDE_SIGNED, 0, 17, 19, 32, 0},
    {"wide-unsigned-p20", RNS8_EXACT_WIDE_UNSIGNED, 0, 17, 19, 32, 0},
    {"finite-prime-q127", RNS8_FINITE_FIELD_U8, 127, 17, 19, 32, 0},
    {"finite-prime-q163", RNS8_FINITE_FIELD_U8, 163, 17, 19, 32, 0},
    {"finite-prime-q167", RNS8_FINITE_FIELD_U8, 167, 17, 19, 32, 0},
    {"finite-prime-q251", RNS8_FINITE_FIELD_U8, 251, 17, 19, 32, 0},
    {"finite-composite-q255", RNS8_FINITE_RING_U8, 255, 17, 19, 32, 0},
    {"finite-even-q256", RNS8_FINITE_RING_U8, 256, 17, 19, 32, 0},
    {"finite-q250-provider-control", RNS8_FINITE_RING_U8, 250, 17, 19, 32, 0},
    {"finite-q2-provider-control", RNS8_FINITE_FIELD_U8, 2, 17, 19, 32, 0},
    {"bounded-k33-tail", RNS8_BOUNDED_I64, 0, 17, 19, 33, 0},
    {"bounded-k65536", RNS8_BOUNDED_I64, 0, 1, 1, 65536, 0},
    {"bounded-k65537", RNS8_BOUNDED_I64, 0, 1, 1, 65537, 0},
    {"wide-large-family", RNS8_EXACT_WIDE_SIGNED, 0, 128, 128, 128, 0},
    {"finite-large-family", RNS8_FINITE_RING_U8, 255, 128, 128, 128, 0},
    {"bounded-padded-address-control", RNS8_BOUNDED_I64, 0, 17, 19, 32, INT32_MAX},
}};

inline rns8_operation_query sparse_profile_query(const SparseProfile& profile, uint32_t operation) {
  rns8_operation_query q{};
  q.struct_size = sizeof(q);
  q.abi_version = RNS8_ABI_VERSION;
  q.operation = operation;
  q.execution_modes = RNS8_EXEC_SYNC;
  q.scratch_budget = UINT64_MAX;
  q.ldb = profile.ldb;
  auto& d = q.problem;
  d.struct_size = sizeof(d);
  d.abi_version = RNS8_ABI_VERSION;
  d.requested_backend = RNS8_BACKEND_AUTO; // Inventory all providers, not dispatch.
  d.semantics = profile.semantics;
  d.bound_kind = profile.semantics == RNS8_BOUNDED_I64   ? RNS8_BOUND_GLOBAL_MAX_ABS
                 : profile.semantics == RNS8_BOUNDED_U64 ? RNS8_BOUND_GLOBAL_MAX_UNSIGNED
                                                         : RNS8_BOUND_NONE;
  d.bound = d.bound_kind == RNS8_BOUND_NONE ? 0 : 65536;
  d.max_prefix =
      profile.semantics == RNS8_EXACT_WIDE_SIGNED || profile.semantics == RNS8_EXACT_WIDE_UNSIGNED
          ? 20
          : 0;
  d.finite_modulus = profile.modulus;
  d.m = profile.m;
  d.n = profile.n;
  d.k = profile.k;
  return q;
}

inline nlohmann::json sparse_profile_report(const rns8_context* ctx) {
  if (!ctx)
    throw std::invalid_argument("sparse profile inventory requires an actual context");
  nlohmann::json rows = nlohmann::json::array();
  for (const auto& profile : sparse_profiles)
    for (uint32_t operation : {uint32_t(RNS8_OP_GEMM), uint32_t(RNS8_OP_STRUCTURED_SPARSE_A),
                               uint32_t(RNS8_OP_BLOCK_SPARSE)}) {
      const auto query = sparse_profile_query(profile, operation);
      uint32_t required = 0;
      if (rns8_query_operation_support(ctx, &query, nullptr, 0, &required) != RNS8_SUCCESS ||
          !required || required > 32)
        throw std::runtime_error("sparse profile provider inventory is empty or unbounded");
      std::vector<rns8_operation_support> support(required);
      const auto capacity = required;
      if (rns8_query_operation_support(ctx, &query, support.data(), capacity, &required) !=
              RNS8_SUCCESS ||
          required != capacity)
        throw std::runtime_error("sparse profile provider inventory changed during enumeration");
      for (const auto& item : support) {
        rns8_backend_capability_info capability{};
        capability.struct_size = sizeof(capability);
        capability.abi_version = RNS8_ABI_VERSION;
        if (rns8_get_backend_capability_info(static_cast<rns8_backend_kind>(item.backend),
                                             &capability) != RNS8_SUCCESS)
          throw std::runtime_error("sparse profile backend capability is unavailable");
        auto row = operation_support_json(item, capability.backend_name);
        row["profile"] = profile.id;
        row["representation"] = operation == RNS8_OP_GEMM ? "dense"
                                : operation == RNS8_OP_STRUCTURED_SPARSE_A
                                    ? "structured_sparse_a_v1"
                                    : "native_integer_zero_block_sparse";
        row["problem"] = {{"semantics", uint32_t(profile.semantics)},
                          {"finite_modulus", profile.modulus},
                          {"m", profile.m},
                          {"n", profile.n},
                          {"k", profile.k},
                          {"ldb", profile.ldb},
                          {"max_prefix", query.problem.max_prefix},
                          {"bound_kind", uint32_t(query.problem.bound_kind)},
                          {"bound", query.problem.bound},
                          {"algorithm", query.algorithm},
                          {"execution_modes", query.execution_modes},
                          {"proof_flags", query.proof_flags}};
        row["backend_isa_evidence"] = capability.isa_evidence;
        row["dispatched_symbol"] = nullptr;
        rows.push_back(std::move(row));
      }
    }
  nlohmann::json kernels = nlohmann::json::array();
  for (auto kernel : rns8::generated_metadata::selected_kernels)
    if (kernel.find("sparse") != std::string_view::npos)
      kernels.push_back(std::string(kernel));
  return {
      {"schema", "rns8_sparse_support_profiles_v1"},
      {"profile_count", sparse_profiles.size()},
      {"operation_count", 3},
      {"row_count", rows.size()},
      {"rows", std::move(rows)},
      {"registered_sparse_kernel_catalog", std::move(kernels)},
      {"kernel_catalog_scope", "source_registry_not_selected_or_executed_symbols"},
      {"query_scope",
       "actual_context_target_all_source_providers_no_matrix_allocation_or_execution"},
      {"basis_policy",
       "default_ordered_basis_only; wide_prefix20; bounded_default_prefix; finite_one_plane"},
      {"structured_policy", "exactly_two_stored_nonzero_bytes_per_four_K_positions_per_plane"},
      {"block_method_scope",
       "unbound_scalar_operation_query; WMMA/hybrid need explicit prepared owners and compiled capability probe"},
      {"qualification", "per_row_receipt_flags_only; unknown_is_not_failed_or_passed"},
      {"production_latency", false}};
}
} // namespace rns8::tools
