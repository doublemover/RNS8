#include <cstdlib>
#include <iostream>
#include <string>

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
      << " [--device N] [--json]\n";
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

void print_capability_json(const rns8_backend_capability_info& capability) {
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
  std::cout << "  }\n";
}

void print_text(const rns8_device_info& info, const rns8_backend_capability_info& capability) {
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
}

void print_json(const rns8_device_info& info, const rns8_backend_capability_info& capability) {
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
  print_capability_json(capability);
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  bool json = false;
  int device_id = -1;
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--json") {
      json = true;
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

  if (json) {
    print_json(info, selected_capability);
  } else {
    print_text(info, selected_capability);
  }
  return 0;
}
