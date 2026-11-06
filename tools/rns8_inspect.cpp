#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "core/autotune_cache.hpp"
#include <nlohmann/json.hpp>
#include "core/plan_lowering.hpp"
#include "rns8/rns8.h"
#include "tool_common/cli_integer.hpp"
#include "tool_common/build_identity.hpp"
#include "tool_common/sparse_profiles.hpp"
#include "tool_common/device_identity_json.hpp"

namespace {
#include "rns8_inspect_parse_helpers.inc"
#include "rns8_inspect_plan_print.inc"
#include "rns8_inspect_autotune_print.inc"
#include "rns8_inspect_selector_print.inc"
int main(int argc, char** argv) {
  const int identity_status = rns8::tools::emit_build_identity(argc, argv, std::cout);
  if (identity_status >= 0)
    return identity_status;
  bool json = false;
  bool operations = false;
  bool sparse_profiles = false;
  bool show_autotune_cache = false;
  bool selector_shadow = false;
  int device_id = -1;
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;
  std::string autotune_key;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--json") {
      json = true;
    } else if (arg == "--sparse-profiles") {
      sparse_profiles = true;
    } else if (arg == "--operations") {
      operations = true;
    } else if (arg == "--show-autotune-cache") {
      show_autotune_cache = true;
    } else if (arg == "--selector-shadow") {
      selector_shadow = true;
    } else if (arg == "--autotune-key" && i + 1 < argc) {
      autotune_key = argv[++i];
    } else if (arg == "--backend" && i + 1 < argc) {
      const std::string value = argv[++i];
      if (!parse_backend(value, backend)) {
        std::cerr << "invalid backend string: " << value
                  << " (unknown names are not routed to auto; choose an explicit listed backend)\n";
        print_usage(std::cerr);
        return 2;
      }
    } else if (arg == "--device") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for --device\n";
        print_usage(std::cerr);
        return 2;
      }
      const auto* value = argv[++i];
      if (!rns8::cli::parse_device(value, device_id)) {
        std::cerr << "invalid device for --device: " << value
                  << " (expected -1 or a nonnegative int ordinal)\n";
        print_usage(std::cerr);
        return 2;
      }
    } else if (arg == "--help") {
      print_usage(std::cout);
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }

  if (sparse_profiles &&
      (operations || show_autotune_cache || selector_shadow || !autotune_key.empty())) {
    std::cerr << "--sparse-profiles is a separate inspection mode\n";
    return 2;
  }

  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = backend;

  rns8_status status = RNS8_SUCCESS;
  rns8_backend_capability_info requested_capability{};
  requested_capability.struct_size = sizeof(requested_capability);
  requested_capability.abi_version = RNS8_ABI_VERSION;
  status = rns8_get_backend_capability_info(backend, &requested_capability);
  if (status != RNS8_SUCCESS) {
    std::cerr << "rns8_get_backend_capability_info(" << backend_name(backend)
              << "): " << rns8_status_string(status) << "\n";
    return 1;
  }

  rns8_context* ctx = nullptr;
  status = rns8_create_context(device_id, &options, &ctx);
  if (status != RNS8_SUCCESS) {
    std::cerr << "rns8_create_context(" << backend_name(backend)
              << "): " << rns8_status_string(status) << "\n";
    std::cerr << "capability_status: " << requested_capability.status << "\n";
    std::cerr << "capability_detail: " << requested_capability.detail << "\n";
    if (requested_capability.enable_flag[0] != '\0') {
      std::cerr << "enable_flag: " << requested_capability.enable_flag << "\n";
    }
    if (status == RNS8_UNSUPPORTED_BACKEND && evidence_only_accelerator_backend(backend)) {
      std::cerr
          << "requested accelerator is evidence-only; runtime remains unsupported until a real exact "
             "correctness backend, compiled kernels, and ISA evidence exist\n";
    } else if (status == RNS8_UNSUPPORTED_BACKEND) {
      std::cerr
          << "backend is not available for this context; RNS8_BACKEND_AUTO does not route across "
             "bounded, exact-wide, or wrap64 semantic backends\n";
    }
    return 1;
  }

  if (sparse_profiles) {
    int result = 0;
    try {
      auto report = rns8::tools::sparse_profile_report(ctx);
#ifdef RNS8_BUILD_MANIFEST_JSON
      report["build_manifest"] = nlohmann::json::parse(RNS8_BUILD_MANIFEST_JSON);
#else
      report["build_manifest"] = nullptr;
#endif
      if (json)
        std::cout << report.dump(2) << '\n';
      else {
        std::cout << "profile representation backend status reason safe_k target\n";
        for (const auto& row : report["rows"])
          std::cout << row["profile"] << ' ' << row["representation"] << ' ' << row["backend"]
                    << ' ' << row["status"] << ' ' << row["reason"] << ' ' << row["safe_k"] << ' '
                    << row["target"] << '\n';
      }
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      result = 1;
    }
    return rns8_destroy_context(ctx) == RNS8_SUCCESS ? result : 1;
  }

  if (operations) {
    nlohmann::json rows = nlohmann::json::array();
    for (uint32_t operation = 0; operation < RNS8_OP_COUNT; ++operation) {
      rns8_operation_query q{};
      q.struct_size = sizeof(q);
      q.abi_version = RNS8_ABI_VERSION;
      q.operation = operation;
      q.execution_modes = RNS8_EXEC_SYNC;
      q.scratch_budget = UINT64_MAX;
      q.device_count = 1;
      q.problem.struct_size = sizeof(q.problem);
      q.problem.abi_version = RNS8_ABI_VERSION;
      q.problem.requested_backend = RNS8_BACKEND_AUTO;
      q.problem.semantics = RNS8_EXACT_WIDE_SIGNED;
      q.problem.bound_kind = RNS8_BOUND_NONE;
      q.problem.m = q.problem.n = q.problem.k = 16;
      q.problem.max_prefix = 20;
      q.proof_flags = 15;
      uint32_t count = 0;
      status = rns8_query_operation_support(ctx, &q, nullptr, 0, &count);
      if (status != RNS8_SUCCESS) {
        rns8_destroy_context(ctx);
        return 1;
      }
      std::vector<rns8_operation_support> records(count);
      status = rns8_query_operation_support(ctx, &q, records.data(), count, &count);
      if (status != RNS8_SUCCESS) {
        rns8_destroy_context(ctx);
        return 1;
      }
      for (uint32_t i = 0; i < count; ++i) {
        const auto& r = records[i];
        rows.push_back(rns8::tools::operation_support_json(
            r, backend_name(static_cast<rns8_backend_kind>(r.backend))));
      }
    }
    if (json)
      std::cout
          << nlohmann::json(
                 {{"schema", "rns8-operation-support-v1"},
                  {"contract",
                   "16x16x16 signed exact-wide prefix20; hypothetical proof flags; not qualification"},
                  {"rows", rows}})
                 .dump(2)
          << "\n";
    else {
      std::cout
          << "Operation support: 16x16x16 signed exact-wide prefix20, hypothetical proofs. No qualification inferred.\n";
      for (const auto& r : rows)
        std::cout << r["operation"] << " " << r["backend"] << " " << r["status"] << " "
                  << r["explanation"] << "\n";
    }
    return rns8_destroy_context(ctx) == RNS8_SUCCESS ? 0 : 1;
  }

  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  status = rns8_get_device_info(ctx, &info);
  if (status != RNS8_SUCCESS) {
    rns8_destroy_context(ctx);
    std::cerr << "rns8_get_device_info: " << rns8_status_string(status) << "\n";
    return 1;
  }

  rns8_backend_capability_info selected_capability{};
  selected_capability.struct_size = sizeof(selected_capability);
  selected_capability.abi_version = RNS8_ABI_VERSION;
  status = rns8_get_backend_capability_info(info.backend, &selected_capability);
  if (status != RNS8_SUCCESS) {
    rns8_destroy_context(ctx);
    std::cerr << "rns8_get_backend_capability_info(" << backend_name(info.backend)
              << "): " << rns8_status_string(status) << "\n";
    return 1;
  }

  rns8::detail::AutotuneCacheSnapshot snapshot{};
  const bool inspect_autotune = show_autotune_cache || !autotune_key.empty() || selector_shadow;
  if (inspect_autotune) {
    snapshot = rns8::detail::read_autotune_cache();
  }

  auto runtime = autotune_runtime_identity(info, selected_capability);
  rns8_plan_packing_info autotune_plan_packing{};
  bool autotune_plan_packing_available = false;
  rns8::detail::PlanLoweringDescription autotune_plan_lowering{};
  bool autotune_plan_lowering_available = false;
  if (inspect_autotune && !autotune_key.empty()) {
    runtime = runtime_identity_with_plan_version(
        ctx, info.backend, snapshot, autotune_key, runtime, &autotune_plan_packing,
        &autotune_plan_packing_available, &autotune_plan_lowering,
        &autotune_plan_lowering_available);
  }
  const rns8_plan_packing_info* plan_packing =
      autotune_plan_packing_available ? &autotune_plan_packing : nullptr;
  const rns8::detail::PlanLoweringDescription* plan_lowering =
      autotune_plan_lowering_available ? &autotune_plan_lowering : nullptr;

  if (json) {
    print_json(info, selected_capability, inspect_autotune ? &snapshot : nullptr, autotune_key,
               runtime, plan_packing, plan_lowering, show_autotune_cache, selector_shadow);
  } else {
    print_text(info, selected_capability, inspect_autotune ? &snapshot : nullptr, autotune_key,
               runtime, plan_packing, plan_lowering, show_autotune_cache, selector_shadow);
  }
  rns8_destroy_context(ctx);
  return 0;
}
