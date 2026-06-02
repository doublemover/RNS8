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

void print_text(const rns8_device_info& info) {
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
}

void print_json(const rns8_device_info& info) {
  std::cout << "{\n";
  std::cout << "  \"backend\": \"" << backend_name(info.backend) << "\",\n";
  std::cout << "  \"device_id\": " << info.device_id << ",\n";
  std::cout << "  \"name\": \"" << info.name << "\",\n";
  std::cout << "  \"gcn_arch\": \"" << info.gcn_arch << "\",\n";
  std::cout << "  \"hip_available\": " << info.hip_available << ",\n";
  std::cout << "  \"hip_runtime_version\": " << info.hip_runtime_version << ",\n";
  std::cout << "  \"hip_driver_version\": " << info.hip_driver_version << ",\n";
  std::cout << "  \"global_mem_bytes\": " << info.global_mem_bytes << ",\n";
  std::cout << "  \"detail\": \"" << info.detail << "\"\n";
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

  rns8_context* ctx = nullptr;
  rns8_status status = rns8_create_context(device_id, &options, &ctx);
  if (status != RNS8_SUCCESS) {
    std::cerr << "rns8_create_context(" << backend_name(backend) << "): " << rns8_status_string(status) << "\n";
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

  if (json) {
    print_json(info);
  } else {
    print_text(info);
  }
  return 0;
}
