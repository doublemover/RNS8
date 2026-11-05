#include "rns8/public_abi.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "public_abi_inventory.inc"

#define RNS8_ABI_STRINGIFY_INNER(value) #value
#define RNS8_ABI_STRINGIFY(value) RNS8_ABI_STRINGIFY_INNER(value)
namespace {
#if defined(__clang__)
constexpr const char* compiler_id = "clang " __clang_version__;
#elif defined(_MSC_FULL_VER)
constexpr const char* compiler_id = "msvc " RNS8_ABI_STRINGIFY(_MSC_FULL_VER);
#elif defined(__GNUC__)
constexpr const char* compiler_id = "gcc " __VERSION__;
#else
constexpr const char* compiler_id = "unknown";
#endif

#if defined(_WIN32) && defined(_DLL) && defined(_DEBUG)
constexpr const char* runtime_model = "msvc-dynamic-crt-debug";
#elif defined(_WIN32) && defined(_DLL)
constexpr const char* runtime_model = "msvc-dynamic-crt-release";
#elif defined(_WIN32) && defined(_MT) && defined(_DEBUG)
constexpr const char* runtime_model = "msvc-static-crt-debug";
#elif defined(_WIN32) && defined(_MT)
constexpr const char* runtime_model = "msvc-static-crt-release";
#elif defined(__GLIBC__)
constexpr const char* runtime_model = "glibc";
#else
constexpr const char* runtime_model = "system-c-runtime";
#endif
} // namespace

rns8_status rns8_get_public_abi_info(rns8_public_abi_info* out) {
  if (!out)
    return RNS8_INVALID_ARGUMENT;
  // Size admission reads only the documented mandatory prefix, including when
  // a C caller supplies a genuinely shorter allocation.
  uint64_t supplied_size = 0;
  std::memcpy(&supplied_size, out, sizeof(supplied_size));
  if (supplied_size != sizeof(*out) || out->abi_version != RNS8_ABI_VERSION)
    return RNS8_INVALID_ARGUMENT;
  rns8_public_abi_info result{};
  result.struct_size = sizeof(result);
  result.abi_version = RNS8_ABI_VERSION;
  result.metric_schema_version = RNS8_PUBLIC_ABI_METRICS_VERSION;
  result.metric_count = sizeof(rns8_compiled_abi_metrics) / sizeof(rns8_compiled_abi_metrics[0]);
  result.metrics = rns8_compiled_abi_metrics;
  result.declaration_sha256 = rns8_public_declaration_sha256;
  result.compiler_id = compiler_id;
  result.runtime_model = runtime_model;
#ifdef RNS8_BUILD_MANIFEST_JSON
  result.build_manifest_json = RNS8_BUILD_MANIFEST_JSON;
#endif
  *out = result;
  return RNS8_SUCCESS;
}
