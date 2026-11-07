#pragma once

#include "rns8/semantics.h"
#include <array>
#include <cstddef>

namespace rns8::detail {

enum class SourceTargetPolicy {
  resolve_first,
  host_cpu,
  hip_target,
  feature_detected_hip,
  feature_detected_cpu
};
struct SourceSemantic {
  rns8_semantics semantics;
  const char* name;
};
struct SourceGemmBackend {
  rns8_backend_kind backend;
  const char* name;
  SourceTargetPolicy target_policy;
  std::array<bool, 7> semantics;
  uint32_t finite_modulus_min, finite_modulus_max;
};

#define RNS8_GEMM_SOURCE_CONTRACT(version) inline constexpr uint32_t source_gemm_contract_version = version;
#define RNS8_GEMM_SOURCE_SEMANTIC(...)
#define RNS8_GEMM_SOURCE_BACKEND(...)
#include "source_gemm_contract.inc"
#undef RNS8_GEMM_SOURCE_CONTRACT
#define RNS8_GEMM_SOURCE_CONTRACT(...)
#undef RNS8_GEMM_SOURCE_SEMANTIC
#define RNS8_GEMM_SOURCE_SEMANTIC(value, name) {value, name},
inline constexpr SourceSemantic source_gemm_semantics[] = {
#include "source_gemm_contract.inc"
};
#undef RNS8_GEMM_SOURCE_SEMANTIC
#define RNS8_GEMM_SOURCE_SEMANTIC(...)
#undef RNS8_GEMM_SOURCE_BACKEND
#define RNS8_GEMM_SOURCE_BACKEND(value, name, target, i64, u64, wide_s, wide_u, wrap, ring, field, \
                                 lo, hi) \
  {value, name, SourceTargetPolicy::target, {i64, u64, wide_s, wide_u, wrap, ring, field}, lo, hi},
inline constexpr SourceGemmBackend source_gemm_backends[] = {
#include "source_gemm_contract.inc"
};
#undef RNS8_GEMM_SOURCE_CONTRACT
#undef RNS8_GEMM_SOURCE_SEMANTIC
#undef RNS8_GEMM_SOURCE_BACKEND

static_assert(sizeof(source_gemm_semantics) / sizeof(source_gemm_semantics[0]) == 7,
              "source semantic flag inventory");
inline constexpr size_t source_gemm_backend_count =
    sizeof(source_gemm_backends) / sizeof(source_gemm_backends[0]);
constexpr bool source_gemm_inventory_indexed() noexcept {
  for (size_t i = 0; i < source_gemm_backend_count; ++i)
    if (static_cast<uint32_t>(source_gemm_backends[i].backend) != i ||
        ((source_gemm_backends[i].semantics[5] || source_gemm_backends[i].semantics[6])
             ? (source_gemm_backends[i].finite_modulus_min != 2 ||
                source_gemm_backends[i].finite_modulus_max != 256)
             : (source_gemm_backends[i].finite_modulus_min != 0 ||
                source_gemm_backends[i].finite_modulus_max != 0)))
      return false;
  for (size_t i = 0; i < 7; ++i)
    if (static_cast<uint32_t>(source_gemm_semantics[i].semantics) != i + 1)
      return false;
  return true;
}
static_assert(source_gemm_inventory_indexed(), "source contract follows the public enum slots");
constexpr const SourceGemmBackend* source_gemm_backend(rns8_backend_kind backend) noexcept {
  const auto index = static_cast<uint32_t>(backend);
  return index < source_gemm_backend_count ? &source_gemm_backends[index] : nullptr;
}
constexpr bool source_gemm_supports_semantics(rns8_backend_kind backend,
                                              rns8_semantics semantics) noexcept {
  const auto* row = source_gemm_backend(backend);
  const auto index = static_cast<uint32_t>(semantics);
  return row && index >= 1 && index <= row->semantics.size() && row->semantics[index - 1];
}

} // namespace rns8::detail
