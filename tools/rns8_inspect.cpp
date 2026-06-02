#include <cstdlib>
#include <iostream>
#include <string>

#include "rns8/rns8.h"

namespace {

rns8_backend_kind parse_backend(const std::string& value) {
  if (value == "auto") return RNS8_BACKEND_AUTO;
  if (value == "cpu") return RNS8_BACKEND_CPU_REFERENCE;
  if (value == "hip-direct") return RNS8_BACKEND_HIP_DIRECT;
  if (value == "wrap64-byte-limb") return RNS8_BACKEND_WRAP64_BYTE_LIMB;
  return RNS8_BACKEND_AUTO;
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
      backend = parse_backend(argv[++i]);
    } else if (arg == "--device" && i + 1 < argc) {
      device_id = std::atoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "usage: rns8-inspect [--backend cpu|hip-direct|wrap64-byte-limb|auto] [--device N] [--json]\n";
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
    std::cerr << "rns8_create_context: " << rns8_status_string(status) << "\n";
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
