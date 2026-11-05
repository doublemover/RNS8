#pragma once

#include <cstring>
#include <ostream>

#include "rns8/public_abi.h"

namespace rns8::tools {

// This path runs before dependency probes, GPU initialization and workload
// dispatch. The caller hashes the actual executable and loaded library files.
inline int emit_build_identity(int argc, char** argv, std::ostream& output) {
  if (argc != 2 || std::strcmp(argv[1], "--build-identity") != 0) return -1;
  rns8_public_abi_info library{};
  library.struct_size = sizeof(library);
  library.abi_version = RNS8_ABI_VERSION;
  if (rns8_get_public_abi_info(&library) != RNS8_SUCCESS ||
      !library.declaration_sha256) return 2;
  output << "{\"schema\":\"rns8_compiled_binary_identity_v1\",\"build_manifest\":";
#ifdef RNS8_BUILD_MANIFEST_JSON
  output << RNS8_BUILD_MANIFEST_JSON;
#else
  output << "null";
#endif
  output << ",\"library_build_manifest\":"
         << (library.build_manifest_json ? library.build_manifest_json : "null")
         << ",\"library_abi_declaration_sha256\":\"" << library.declaration_sha256 << "\"}\n";
#ifdef RNS8_BUILD_MANIFEST_JSON
  return output && library.build_manifest_json ? 0 : 2;
#else
  return 2;
#endif
}

}  // namespace rns8::tools
