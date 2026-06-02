#include "core/internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

uint32_t rns8_default_modulus_count(void) {
  return RNS8_DEFAULT_MODULUS_COUNT;
}

uint16_t rns8_default_modulus(uint32_t index) {
  if (index >= RNS8_DEFAULT_MODULUS_COUNT) {
    return 0;
  }
  return rns8::detail::kDefaultModuli[index];
}

double rns8_prefix_range_bits(uint32_t prefix) {
  if (prefix == 0 || prefix > RNS8_DEFAULT_MODULUS_COUNT) {
    return 0.0;
  }
  long double bits = 0.0L;
  for (uint32_t i = 0; i < prefix; ++i) {
    bits += std::log2(static_cast<long double>(rns8::detail::kDefaultModuli[i]));
  }
  return static_cast<double>(bits);
}

rns8_status rns8_validate_default_moduli(void) {
  return rns8::detail::default_moduli_pairwise_coprime() ? RNS8_SUCCESS : RNS8_INTERNAL_ERROR;
}

namespace rns8::detail {

bool valid_abi(uint64_t struct_size, uint32_t abi_version, std::size_t expected_size) {
  return abi_version == RNS8_ABI_VERSION && struct_size >= expected_size;
}

void copy_c_string(char* dst, std::size_t dst_size, const std::string& src) {
  if (dst_size == 0) {
    return;
  }
  const std::size_t count = std::min(dst_size - 1, src.size());
  std::memcpy(dst, src.data(), count);
  dst[count] = '\0';
}

void fill_cpu_device_info(rns8_device_info& info) {
  info.backend = RNS8_BACKEND_CPU_REFERENCE;
  info.device_id = -1;
  info.hip_available = 0;
  info.hip_runtime_version = 0;
  info.hip_driver_version = 0;
  info.global_mem_bytes = 0;
  copy_c_string(info.name, sizeof(info.name), "CPU reference");
  copy_c_string(info.gcn_arch, sizeof(info.gcn_arch), "none");
  copy_c_string(info.detail, sizeof(info.detail), "portable scalar CPU reference backend");
}

bool default_moduli_pairwise_coprime() {
  for (uint32_t i = 0; i < RNS8_DEFAULT_MODULUS_COUNT; ++i) {
    for (uint32_t j = i + 1; j < RNS8_DEFAULT_MODULUS_COUNT; ++j) {
      if (std::gcd(kDefaultModuli[i], kDefaultModuli[j]) != 1) {
        return false;
      }
    }
  }
  return true;
}

cpp_int modulus_product(uint32_t prefix) {
  cpp_int product = 1;
  for (uint32_t i = 0; i < prefix; ++i) {
    product *= kDefaultModuli[i];
  }
  return product;
}

uint32_t default_prefix_for_semantics(rns8_semantics semantics) {
  switch (semantics) {
    case RNS8_BOUNDED_I64:
    case RNS8_BOUNDED_U64:
      return RNS8_DEFAULT_BOUNDED_PREFIX;
    case RNS8_FINITE_RING_U8:
    case RNS8_FINITE_FIELD_U8:
      return 1;
    case RNS8_EXACT_WIDE_SIGNED:
    case RNS8_EXACT_WIDE_UNSIGNED:
    case RNS8_WRAP_U64_MOD_2_64:
      return RNS8_DEFAULT_BOUNDED_PREFIX;
  }
  return 0;
}

rns8_status validate_bound_contract(
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint64_t bound,
    uint32_t prefix) {
  if (prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }

  const cpp_int product = modulus_product(prefix);
  switch (semantics) {
    case RNS8_BOUNDED_I64: {
      if (bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS) {
        return bound_kind == RNS8_BOUND_NONE ? RNS8_INVALID_ARGUMENT : RNS8_UNSUPPORTED_BACKEND;
      }
      if (bound > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return RNS8_INVALID_ARGUMENT;
      }
      const cpp_int required = cpp_int(2) * cpp_int(bound);
      return product > required ? RNS8_SUCCESS : RNS8_RANGE_ERROR;
    }
    case RNS8_BOUNDED_U64: {
      if (bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED) {
        return bound_kind == RNS8_BOUND_NONE ? RNS8_INVALID_ARGUMENT : RNS8_UNSUPPORTED_BACKEND;
      }
      return product > cpp_int(bound) ? RNS8_SUCCESS : RNS8_RANGE_ERROR;
    }
    case RNS8_EXACT_WIDE_SIGNED:
    case RNS8_EXACT_WIDE_UNSIGNED:
    case RNS8_WRAP_U64_MOD_2_64:
    case RNS8_FINITE_RING_U8:
    case RNS8_FINITE_FIELD_U8:
      return RNS8_UNSUPPORTED_BACKEND;
  }
  return RNS8_INVALID_ARGUMENT;
}

rns8_status validate_gemm_desc(const rns8_gemm_desc& desc, uint32_t prefix) {
  if (!valid_abi(desc.struct_size, desc.abi_version, sizeof(desc))) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (desc.m <= 0 || desc.n <= 0 || desc.k <= 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (desc.tile_m != 0 && (desc.tile_m < 64 || desc.tile_m > 512)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (desc.tile_n != 0 && (desc.tile_n < 64 || desc.tile_n > 512)) {
    return RNS8_INVALID_ARGUMENT;
  }
  return validate_bound_contract(desc.semantics, desc.bound_kind, desc.bound, prefix);
}

rns8_status validate_matrix_desc(const rns8_matrix_desc& desc, uint32_t prefix) {
  if (!valid_abi(desc.struct_size, desc.abi_version, sizeof(desc))) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (desc.rows <= 0 || desc.cols <= 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (desc.logical_layout != RNS8_LAYOUT_ROW_MAJOR) {
    return RNS8_UNSUPPORTED_BACKEND;
  }
  if (desc.logical_ld != 0 && desc.logical_ld < desc.cols) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  switch (desc.semantics) {
    case RNS8_BOUNDED_I64:
      return desc.bound_kind == RNS8_BOUND_GLOBAL_MAX_ABS ? RNS8_SUCCESS : RNS8_INVALID_ARGUMENT;
    case RNS8_BOUNDED_U64:
      return desc.bound_kind == RNS8_BOUND_GLOBAL_MAX_UNSIGNED ? RNS8_SUCCESS : RNS8_INVALID_ARGUMENT;
    default:
      return RNS8_UNSUPPORTED_BACKEND;
  }
}

}  // namespace rns8::detail
