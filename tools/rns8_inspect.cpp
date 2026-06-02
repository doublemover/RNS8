#include <cstdlib>
#include <iostream>
#include <string>

#include "core/autotune_cache.hpp"
#include "rns8/rns8.h"

namespace {

bool parse_backend(const std::string& value, rns8_backend_kind& out) {
  if (value == "auto") {
    out = RNS8_BACKEND_AUTO;
    return true;
  }
  if (value == "cpu" || value == "cpu-reference") {
    out = RNS8_BACKEND_CPU_REFERENCE;
    return true;
  }
  if (value == "hip-direct") {
    out = RNS8_BACKEND_HIP_DIRECT;
    return true;
  }
  if (value == "wrap64-byte-limb") {
    out = RNS8_BACKEND_WRAP64_BYTE_LIMB;
    return true;
  }
  if (value == "hipblaslt") {
    out = RNS8_BACKEND_HIPBLASLT;
    return true;
  }
  if (value == "ck") {
    out = RNS8_BACKEND_CK;
    return true;
  }
  if (value == "rocwmma" || value == "wmma") {
    out = RNS8_BACKEND_WMMA;
    return true;
  }
  return false;
}

bool evidence_only_accelerator_backend(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_HIPBLASLT || backend == RNS8_BACKEND_CK || backend == RNS8_BACKEND_WMMA;
}

void print_usage(std::ostream& out) {
  out << "usage: rns8-inspect [--backend auto|cpu-reference|hip-direct|wrap64-byte-limb|hipblaslt|ck|rocwmma]"
      << " [--device N] [--json] [--autotune-key KEY] [--show-autotune-cache]\n";
}

const char* backend_name(rns8_backend_kind backend) {
  switch (backend) {
    case RNS8_BACKEND_AUTO:
      return "auto";
    case RNS8_BACKEND_CPU_REFERENCE:
      return "cpu-reference";
    case RNS8_BACKEND_HIP_DIRECT:
      return "hip-direct";
    case RNS8_BACKEND_HIPBLASLT:
      return "hipblaslt";
    case RNS8_BACKEND_CK:
      return "ck";
    case RNS8_BACKEND_WMMA:
      return "wmma";
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
      return "wrap64-byte-limb";
  }
  return "unknown";
}

std::string json_escape(const char* input) {
  std::string escaped;
  if (!input) {
    return escaped;
  }
  for (const unsigned char ch : std::string(input)) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped.push_back(static_cast<char>(ch));
        break;
    }
  }
  return escaped;
}

std::string json_escape(const std::string& input) {
  return json_escape(input.c_str());
}

void print_capability_text(const rns8_backend_capability_info& capability) {
  std::cout << "capability_status: " << capability.status << "\n";
  std::cout << "selected_kernel:   " << capability.selected_kernel << "\n";
  std::cout << "accelerator:       " << capability.is_accelerator << "\n";
  std::cout << "correctness:       " << capability.is_correctness_backend << "\n";
  std::cout << "matrix_engine:     " << capability.is_matrix_engine_backend << "\n";
  std::cout << "compiled_kernel:   " << capability.compiled_kernel_available << "\n";
  std::cout << "exact_validated:   " << capability.exact_differential_validated << "\n";
  std::cout << "perf_validated:    " << capability.performance_validated << "\n";
  std::cout << "library:           " << capability.library_name << "\n";
  std::cout << "enable_flag:       " << capability.enable_flag << "\n";
  std::cout << "epilogue_mode:     " << capability.epilogue_mode << "\n";
  std::cout << "workspace_mode:    " << capability.workspace_mode << "\n";
  std::cout << "isa_evidence:      " << capability.isa_evidence << "\n";
  std::cout << "capability_detail: " << capability.detail << "\n";
}

void print_capability_json(const rns8_backend_capability_info& capability, bool trailing_comma) {
  std::cout << "  \"capability\": {\n";
  std::cout << "    \"backend\": \"" << backend_name(capability.backend) << "\",\n";
  std::cout << "    \"backend_name\": \"" << json_escape(capability.backend_name) << "\",\n";
  std::cout << "    \"status\": \"" << json_escape(capability.status) << "\",\n";
  std::cout << "    \"detail\": \"" << json_escape(capability.detail) << "\",\n";
  std::cout << "    \"selected_kernel\": \"" << json_escape(capability.selected_kernel) << "\",\n";
  std::cout << "    \"accelerator_backend\": " << (capability.is_accelerator ? "true" : "false") << ",\n";
  std::cout << "    \"correctness_backend\": " << (capability.is_correctness_backend ? "true" : "false") << ",\n";
  std::cout << "    \"matrix_engine_backend\": " << (capability.is_matrix_engine_backend ? "true" : "false") << ",\n";
  std::cout << "    \"requires_feature_detection\": "
            << (capability.requires_feature_detection ? "true" : "false") << ",\n";
  std::cout << "    \"enable_flag_fail_fast\": "
            << (capability.enable_flag_fail_fast ? "true" : "false") << ",\n";
  std::cout << "    \"candidate_evidence_only\": "
            << (capability.candidate_evidence_only ? "true" : "false") << ",\n";
  std::cout << "    \"compiled_kernel_available\": "
            << (capability.compiled_kernel_available ? "true" : "false") << ",\n";
  std::cout << "    \"exact_differential_validated\": "
            << (capability.exact_differential_validated ? "true" : "false") << ",\n";
  std::cout << "    \"performance_validated\": " << (capability.performance_validated ? "true" : "false") << ",\n";
  std::cout << "    \"library_name\": \"" << json_escape(capability.library_name) << "\",\n";
  std::cout << "    \"library_version\": \"" << json_escape(capability.library_version) << "\",\n";
  std::cout << "    \"enable_flag\": \"" << json_escape(capability.enable_flag) << "\",\n";
  std::cout << "    \"epilogue_mode\": \"" << json_escape(capability.epilogue_mode) << "\",\n";
  std::cout << "    \"workspace_mode\": \"" << json_escape(capability.workspace_mode) << "\",\n";
  std::cout << "    \"isa_evidence\": \"" << json_escape(capability.isa_evidence) << "\"\n";
  std::cout << "  }" << (trailing_comma ? "," : "") << "\n";
}

void print_autotune_text(
    const rns8::detail::AutotuneCacheSnapshot& snapshot,
    const std::string& autotune_key,
    const std::string& selected_backend,
    bool show_entries) {
  std::cout << "autotune_cache_path:   " << snapshot.path.string() << "\n";
  std::cout << "autotune_cache_loaded: " << (snapshot.loaded ? 1 : 0) << "\n";
  std::cout << "autotune_cache_exists: " << (snapshot.exists ? 1 : 0) << "\n";
  std::cout << "autotune_entry_count:  " << snapshot.entries.size() << "\n";
  if (!snapshot.error.empty()) {
    std::cout << "autotune_cache_error:  " << snapshot.error << "\n";
  }
  if (!autotune_key.empty()) {
    const auto* hit = rns8::detail::find_exact_autotune_entry(snapshot, autotune_key);
    std::cout << "autotune_key:          " << autotune_key << "\n";
    std::cout << "autotune_exact_hit:    " << (hit ? 1 : 0) << "\n";
    std::cout << "selection_rationale:   "
              << rns8::detail::autotune_selection_rationale(snapshot, autotune_key, selected_backend) << "\n";
    if (hit) {
      std::cout << "autotune_backend:      " << hit->selected_backend << "\n";
      std::cout << "autotune_kernel:       " << hit->selected_kernel << "\n";
      std::cout << "autotune_median_e2e:   " << hit->measured_median_end_to_end_us << "\n";
    }
  }
  if (show_entries) {
    for (const auto& entry : snapshot.entries) {
      std::cout << "autotune_entry:        " << entry.selected_backend << " " << entry.selected_kernel
                << " " << entry.key << "\n";
    }
  }
}

void print_autotune_json(
    const rns8::detail::AutotuneCacheSnapshot& snapshot,
    const std::string& autotune_key,
    const std::string& selected_backend,
    bool show_entries) {
  const auto* hit = rns8::detail::find_exact_autotune_entry(snapshot, autotune_key);
  std::cout << "  \"autotune_cache\": {\n";
  std::cout << "    \"path\": \"" << json_escape(snapshot.path.string()) << "\",\n";
  std::cout << "    \"loaded\": " << (snapshot.loaded ? "true" : "false") << ",\n";
  std::cout << "    \"exists\": " << (snapshot.exists ? "true" : "false") << ",\n";
  std::cout << "    \"schema_version\": " << snapshot.schema_version << ",\n";
  std::cout << "    \"entry_count\": " << snapshot.entries.size() << ",\n";
  std::cout << "    \"error\": \"" << json_escape(snapshot.error) << "\",\n";
  std::cout << "    \"queried_key\": \"" << json_escape(autotune_key) << "\",\n";
  std::cout << "    \"exact_hit\": " << (hit ? "true" : "false") << ",\n";
  std::cout << "    \"selection_rationale\": \""
            << json_escape(rns8::detail::autotune_selection_rationale(snapshot, autotune_key, selected_backend))
            << "\"";
  if (hit) {
    std::cout << ",\n";
    std::cout << "    \"entry\": {\n";
    std::cout << "      \"selected_backend\": \"" << json_escape(hit->selected_backend) << "\",\n";
    std::cout << "      \"selected_kernel\": \"" << json_escape(hit->selected_kernel) << "\",\n";
    std::cout << "      \"target_id\": \"" << json_escape(hit->target_id) << "\",\n";
    std::cout << "      \"validation_status\": \"" << json_escape(hit->validation_status) << "\",\n";
    std::cout << "      \"performance_validated\": " << (hit->performance_validated ? "true" : "false") << ",\n";
    std::cout << "      \"measured_median_end_to_end_us\": " << hit->measured_median_end_to_end_us << "\n";
    std::cout << "    }";
  }
  if (show_entries) {
    std::cout << ",\n";
    std::cout << "    \"entries\": [";
    for (std::size_t i = 0; i < snapshot.entries.size(); ++i) {
      const auto& entry = snapshot.entries[i];
      std::cout << (i == 0 ? "\n" : ",\n");
      std::cout << "      {\"key\": \"" << json_escape(entry.key) << "\", \"selected_backend\": \""
                << json_escape(entry.selected_backend) << "\", \"selected_kernel\": \""
                << json_escape(entry.selected_kernel) << "\"}";
    }
    if (!snapshot.entries.empty()) {
      std::cout << "\n    ";
    }
    std::cout << "]";
  }
  std::cout << "\n";
  std::cout << "  }\n";
}

void print_text(
    const rns8_device_info& info,
    const rns8_backend_capability_info& capability,
    const rns8::detail::AutotuneCacheSnapshot* snapshot,
    const std::string& autotune_key,
    bool show_autotune_cache) {
  std::cout << "RNS8 inspect\n";
  std::cout << "backend:       " << backend_name(info.backend) << "\n";
  std::cout << "device_id:     " << info.device_id << "\n";
  std::cout << "name:          " << info.name << "\n";
  std::cout << "gcn_arch:      " << info.gcn_arch << "\n";
  std::cout << "hip_available: " << info.hip_available << "\n";
  std::cout << "hip_runtime:   " << info.hip_runtime_version << "\n";
  std::cout << "hip_driver:    " << info.hip_driver_version << "\n";
  std::cout << "global_mem:    " << info.global_mem_bytes << "\n";
  std::cout << "detail:        " << info.detail << "\n";
  print_capability_text(capability);
  if (snapshot) {
    print_autotune_text(*snapshot, autotune_key, backend_name(info.backend), show_autotune_cache);
  }
}

void print_json(
    const rns8_device_info& info,
    const rns8_backend_capability_info& capability,
    const rns8::detail::AutotuneCacheSnapshot* snapshot,
    const std::string& autotune_key,
    bool show_autotune_cache) {
  std::cout << "{\n";
  std::cout << "  \"backend\": \"" << backend_name(info.backend) << "\",\n";
  std::cout << "  \"device_id\": " << info.device_id << ",\n";
  std::cout << "  \"name\": \"" << json_escape(info.name) << "\",\n";
  std::cout << "  \"gcn_arch\": \"" << json_escape(info.gcn_arch) << "\",\n";
  std::cout << "  \"hip_available\": " << info.hip_available << ",\n";
  std::cout << "  \"hip_runtime_version\": " << info.hip_runtime_version << ",\n";
  std::cout << "  \"hip_driver_version\": " << info.hip_driver_version << ",\n";
  std::cout << "  \"global_mem_bytes\": " << info.global_mem_bytes << ",\n";
  std::cout << "  \"detail\": \"" << json_escape(info.detail) << "\",\n";
  print_capability_json(capability, snapshot != nullptr);
  if (snapshot) {
    print_autotune_json(*snapshot, autotune_key, backend_name(info.backend), show_autotune_cache);
  }
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  bool json = false;
  bool show_autotune_cache = false;
  int device_id = -1;
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;
  std::string autotune_key;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--json") {
      json = true;
    } else if (arg == "--show-autotune-cache") {
      show_autotune_cache = true;
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
  rns8_destroy_context(ctx);
  if (status != RNS8_SUCCESS) {
    std::cerr << "rns8_get_device_info: " << rns8_status_string(status) << "\n";
    return 1;
  }

  rns8_backend_capability_info selected_capability{};
  selected_capability.struct_size = sizeof(selected_capability);
  selected_capability.abi_version = RNS8_ABI_VERSION;
  status = rns8_get_backend_capability_info(info.backend, &selected_capability);
  if (status != RNS8_SUCCESS) {
    std::cerr << "rns8_get_backend_capability_info(" << backend_name(info.backend) << "): "
              << rns8_status_string(status) << "\n";
    return 1;
  }

  rns8::detail::AutotuneCacheSnapshot snapshot{};
  const bool inspect_autotune = show_autotune_cache || !autotune_key.empty();
  if (inspect_autotune) {
    snapshot = rns8::detail::read_autotune_cache();
  }

  if (json) {
    print_json(info, selected_capability, inspect_autotune ? &snapshot : nullptr, autotune_key, show_autotune_cache);
  } else {
    print_text(info, selected_capability, inspect_autotune ? &snapshot : nullptr, autotune_key, show_autotune_cache);
  }
  return 0;
}
