#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "core/autotune_cache.hpp"
#include "core/plan_lowering.hpp"
#include "rns8/rns8.h"

namespace {
#include "rns8_inspect_parse_helpers.inc"
#include "rns8_inspect_plan_print.inc"
#include "rns8_inspect_autotune_print.inc"
#include "rns8_inspect_selector_print.inc"
int main(int argc, char** argv) {
  bool json = false;
  bool show_autotune_cache = false;
  bool selector_shadow = false;
  int device_id = -1;
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;
  std::string autotune_key;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--json") {
      json = true;
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
    } else if (arg == "--device" && i + 1 < argc) {
      device_id = std::atoi(argv[++i]);
    } else if (arg == "--help") {
      print_usage(std::cout);
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
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
    std::cerr << "rns8_get_backend_capability_info(" << backend_name(backend) << "): "
              << rns8_status_string(status) << "\n";
    return 1;
  }

  rns8_context* ctx = nullptr;
  status = rns8_create_context(device_id, &options, &ctx);
  if (status != RNS8_SUCCESS) {
    std::cerr << "rns8_create_context(" << backend_name(backend) << "): " << rns8_status_string(status) << "\n";
    std::cerr << "capability_status: " << requested_capability.status << "\n";
    std::cerr << "capability_detail: " << requested_capability.detail << "\n";
    if (requested_capability.enable_flag[0] != '\0') {
      std::cerr << "enable_flag: " << requested_capability.enable_flag << "\n";
    }
    if (status == RNS8_UNSUPPORTED_BACKEND && evidence_only_accelerator_backend(backend)) {
      std::cerr << "requested accelerator is evidence-only; enable flags fail fast until a real exact "
                   "correctness backend exists\n";
    } else if (status == RNS8_UNSUPPORTED_BACKEND) {
      std::cerr << "backend is not available for this context; RNS8_BACKEND_AUTO does not route across "
                   "bounded, exact-wide, or wrap64 semantic backends\n";
    }
    return 1;
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
    std::cerr << "rns8_get_backend_capability_info(" << backend_name(info.backend) << "): "
              << rns8_status_string(status) << "\n";
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
        ctx,
        info.backend,
        snapshot,
        autotune_key,
        runtime,
        &autotune_plan_packing,
        &autotune_plan_packing_available,
        &autotune_plan_lowering,
        &autotune_plan_lowering_available);
  }
  const rns8_plan_packing_info* plan_packing =
      autotune_plan_packing_available ? &autotune_plan_packing : nullptr;
  const rns8::detail::PlanLoweringDescription* plan_lowering =
      autotune_plan_lowering_available ? &autotune_plan_lowering : nullptr;

  if (json) {
    print_json(
        info,
        selected_capability,
        inspect_autotune ? &snapshot : nullptr,
        autotune_key,
        runtime,
        plan_packing,
        plan_lowering,
        show_autotune_cache,
        selector_shadow);
  } else {
    print_text(
        info,
        selected_capability,
        inspect_autotune ? &snapshot : nullptr,
        autotune_key,
        runtime,
        plan_packing,
        plan_lowering,
        show_autotune_cache,
        selector_shadow);
  }
  rns8_destroy_context(ctx);
  return 0;
}
