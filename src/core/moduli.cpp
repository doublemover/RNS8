#include "core/internal.hpp"

#include <algorithm>
#include <boost/multiprecision/integer.hpp>
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

void fill_wrap64_device_info(rns8_device_info& info) {
  info.backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  info.device_id = -1;
  info.hip_available = 0;
  info.hip_runtime_version = 0;
  info.hip_driver_version = 0;
  info.global_mem_bytes = 0;
  copy_c_string(info.name, sizeof(info.name), "CPU wrap64 byte-limb reference");
  copy_c_string(info.gcn_arch, sizeof(info.gcn_arch), "none");
  copy_c_string(info.detail, sizeof(info.detail), "strict mod 2^64 byte-limb CPU reference backend");
}

bool valid_tile_size(uint32_t value) {
  return value == 0 || ((value >= 64 && value <= 512) && (value & (value - 1u)) == 0);
}

bool known_semantics(rns8_semantics semantics) {
  switch (semantics) {
    case RNS8_BOUNDED_I64:
    case RNS8_BOUNDED_U64:
    case RNS8_EXACT_WIDE_SIGNED:
    case RNS8_EXACT_WIDE_UNSIGNED:
    case RNS8_WRAP_U64_MOD_2_64:
    case RNS8_FINITE_RING_U8:
    case RNS8_FINITE_FIELD_U8:
      return true;
  }
  return false;
}

bool known_bound_kind(rns8_bound_kind bound_kind) {
  switch (bound_kind) {
    case RNS8_BOUND_NONE:
    case RNS8_BOUND_GLOBAL_MAX_ABS:
    case RNS8_BOUND_GLOBAL_MAX_UNSIGNED:
    case RNS8_BOUND_PER_TILE_MAX_ABS:
    case RNS8_BOUND_PER_TILE_MAX_UNSIGNED:
    case RNS8_BOUND_INPUT_RANGE_AND_K:
      return true;
  }
  return false;
}

bool known_layout(rns8_layout layout) {
  switch (layout) {
    case RNS8_LAYOUT_ROW_MAJOR:
    case RNS8_LAYOUT_COLUMN_MAJOR:
      return true;
  }
  return false;
}

bool valid_finite_ring_modulus(uint32_t modulus) {
  return modulus >= 2 && modulus <= 256;
}

bool valid_finite_field_modulus(uint32_t modulus) {
  if (modulus < 2 || modulus > 251) {
    return false;
  }
  for (uint32_t divisor = 2; divisor * divisor <= modulus; ++divisor) {
    if (modulus % divisor == 0) {
      return false;
    }
  }
  return true;
}

bool valid_finite_modulus_for_semantics(rns8_semantics semantics, uint32_t modulus) {
  if (semantics == RNS8_FINITE_FIELD_U8) {
    return valid_finite_field_modulus(modulus);
  }
  if (semantics == RNS8_FINITE_RING_U8) {
    return valid_finite_ring_modulus(modulus);
  }
  return false;
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

uint32_t bit_length(const cpp_int& value) {
  if (value <= 0) {
    return 0;
  }
  const auto width = boost::multiprecision::msb(value) + 1;
  return width > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                      : static_cast<uint32_t>(width);
}

uint32_t required_prefix_for_range(const cpp_int& range) {
  cpp_int product = 1;
  for (uint32_t prefix = 1; prefix <= RNS8_MAX_SUPPORTED_PREFIX; ++prefix) {
    product *= kDefaultModuli[prefix - 1];
    if (product > range) {
      return prefix;
    }
  }
  return 0;
}

uint32_t default_prefix_for_semantics(rns8_semantics semantics) {
  switch (semantics) {
    case RNS8_BOUNDED_I64:
    case RNS8_BOUNDED_U64:
      return RNS8_DEFAULT_BOUNDED_PREFIX;
    case RNS8_EXACT_WIDE_SIGNED:
    case RNS8_EXACT_WIDE_UNSIGNED:
      return RNS8_MAX_SUPPORTED_PREFIX;
    case RNS8_WRAP_U64_MOD_2_64:
    case RNS8_FINITE_RING_U8:
    case RNS8_FINITE_FIELD_U8:
      return 0;
  }
  return 0;
}

bool input_range_output_bound(const rns8_gemm_desc& desc, uint64_t& output_bound) {
  output_bound = 0;
  if (desc.bound_kind != RNS8_BOUND_INPUT_RANGE_AND_K ||
      (desc.semantics != RNS8_BOUNDED_I64 && desc.semantics != RNS8_BOUNDED_U64) ||
      desc.k <= 0) {
    return false;
  }
  if (desc.semantics == RNS8_BOUNDED_I64) {
    constexpr uint64_t max_signed_magnitude = uint64_t{1} << 63u;
    if (desc.lhs_bound > max_signed_magnitude || desc.rhs_bound > max_signed_magnitude) {
      return false;
    }
  }
  const cpp_int product =
      cpp_int(desc.k) * cpp_int(desc.lhs_bound) * cpp_int(desc.rhs_bound);
  const cpp_int max_output =
      desc.semantics == RNS8_BOUNDED_I64
          ? cpp_int(uint64_t{1} << 63u)
          : cpp_int(std::numeric_limits<uint64_t>::max());
  if (product < 0 || product > max_output) {
    return false;
  }
  output_bound = product.convert_to<uint64_t>();
  return true;
}

rns8_status validate_bound_contract(
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint64_t bound,
    uint32_t prefix) {
  if (!known_semantics(semantics) || !known_bound_kind(bound_kind)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }

  const cpp_int product = modulus_product(prefix);
  switch (semantics) {
    case RNS8_BOUNDED_I64: {
      if (bound_kind == RNS8_BOUND_INPUT_RANGE_AND_K) {
        return RNS8_UNSUPPORTED_BACKEND;
      }
      if (bound_kind == RNS8_BOUND_PER_TILE_MAX_ABS) {
        return bound == 0 ? RNS8_SUCCESS : RNS8_INVALID_ARGUMENT;
      }
      if (bound_kind == RNS8_BOUND_NONE || bound_kind == RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
          bound_kind == RNS8_BOUND_PER_TILE_MAX_UNSIGNED) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS) {
        return RNS8_UNSUPPORTED_BACKEND;
      }
      constexpr uint64_t max_signed_magnitude = uint64_t{1} << 63u;
      if (bound > max_signed_magnitude) {
        return RNS8_INVALID_ARGUMENT;
      }
      const cpp_int required = cpp_int(2) * cpp_int(bound);
      return product > required ? RNS8_SUCCESS : RNS8_RANGE_ERROR;
    }
    case RNS8_BOUNDED_U64: {
      if (bound_kind == RNS8_BOUND_INPUT_RANGE_AND_K) {
        return RNS8_UNSUPPORTED_BACKEND;
      }
      if (bound_kind == RNS8_BOUND_PER_TILE_MAX_UNSIGNED) {
        return bound == 0 ? RNS8_SUCCESS : RNS8_INVALID_ARGUMENT;
      }
      if (bound_kind == RNS8_BOUND_NONE || bound_kind == RNS8_BOUND_GLOBAL_MAX_ABS ||
          bound_kind == RNS8_BOUND_PER_TILE_MAX_ABS) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED) {
        return RNS8_UNSUPPORTED_BACKEND;
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
  constexpr uint32_t allowed_flags = RNS8_PLAN_FORCE_FIXED_PREFIX | RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
  if ((desc.flags & ~allowed_flags) != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const bool fixed_prefix_requested = (desc.flags & RNS8_PLAN_FORCE_FIXED_PREFIX) != 0;
  const bool proven_zero_tile_skips_requested = (desc.flags & RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS) != 0;
  if (!valid_tile_size(desc.tile_m)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!valid_tile_size(desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const bool per_tile_bounds =
      desc.bound_kind == RNS8_BOUND_PER_TILE_MAX_ABS || desc.bound_kind == RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  if (!per_tile_bounds && (desc.tile_bounds || desc.tile_bounds_count != 0)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (proven_zero_tile_skips_requested && !per_tile_bounds) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!known_semantics(desc.semantics) || !known_bound_kind(desc.bound_kind)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (desc.semantics != RNS8_FINITE_RING_U8 && desc.semantics != RNS8_FINITE_FIELD_U8 &&
      desc.finite_modulus != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (desc.bound_kind != RNS8_BOUND_INPUT_RANGE_AND_K && (desc.lhs_bound != 0 || desc.rhs_bound != 0)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (desc.semantics == RNS8_EXACT_WIDE_SIGNED || desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
    if (desc.bound_kind != RNS8_BOUND_NONE || desc.bound != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
      return RNS8_INVALID_ARGUMENT;
    }
    const cpp_int product = modulus_product(prefix);
    const cpp_int required =
        cpp_int(desc.k) * (cpp_int(1) << (desc.semantics == RNS8_EXACT_WIDE_SIGNED ? 127 : 128));
    return product > required ? RNS8_SUCCESS : RNS8_RANGE_ERROR;
  }
  if (desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    if (desc.bound_kind != RNS8_BOUND_NONE || desc.bound != 0 || prefix != 0 || fixed_prefix_requested) {
      return RNS8_INVALID_ARGUMENT;
    }
    return RNS8_SUCCESS;
  }
  if (desc.semantics == RNS8_FINITE_RING_U8 || desc.semantics == RNS8_FINITE_FIELD_U8) {
    if (desc.bound_kind != RNS8_BOUND_NONE || desc.bound != 0 || desc.max_prefix != 0 || prefix != 0 ||
        fixed_prefix_requested || !valid_finite_modulus_for_semantics(desc.semantics, desc.finite_modulus)) {
      return RNS8_INVALID_ARGUMENT;
    }
    return RNS8_SUCCESS;
  }
  if (desc.bound_kind == RNS8_BOUND_INPUT_RANGE_AND_K) {
    if (desc.bound != 0 || desc.tile_bounds != nullptr || desc.tile_bounds_count != 0 ||
        (desc.semantics != RNS8_BOUNDED_I64 && desc.semantics != RNS8_BOUNDED_U64)) {
      return RNS8_INVALID_ARGUMENT;
    }
    uint64_t output_bound = 0;
    if (!input_range_output_bound(desc, output_bound)) {
      return RNS8_RANGE_ERROR;
    }
    const cpp_int product = modulus_product(prefix);
    const cpp_int required = desc.semantics == RNS8_BOUNDED_I64
                                 ? cpp_int(2) * cpp_int(output_bound)
                                 : cpp_int(output_bound);
    return product > required ? RNS8_SUCCESS : RNS8_RANGE_ERROR;
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
  if (desc.flags != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!known_semantics(desc.semantics) || !known_bound_kind(desc.bound_kind) || !known_layout(desc.logical_layout)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (desc.logical_ld != 0 && desc.logical_ld < desc.cols) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!valid_tile_size(desc.tile_m) || !valid_tile_size(desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const bool column_major = desc.logical_layout == RNS8_LAYOUT_COLUMN_MAJOR;
  switch (desc.semantics) {
    case RNS8_BOUNDED_I64:
      if (prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS && desc.bound_kind != RNS8_BOUND_PER_TILE_MAX_ABS) {
        return RNS8_INVALID_ARGUMENT;
      }
      return column_major ? RNS8_UNSUPPORTED_BACKEND : RNS8_SUCCESS;
    case RNS8_BOUNDED_U64:
      if (prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED && desc.bound_kind != RNS8_BOUND_PER_TILE_MAX_UNSIGNED) {
        return RNS8_INVALID_ARGUMENT;
      }
      return column_major ? RNS8_UNSUPPORTED_BACKEND : RNS8_SUCCESS;
    case RNS8_EXACT_WIDE_SIGNED:
    case RNS8_EXACT_WIDE_UNSIGNED:
      if (prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (desc.bound_kind != RNS8_BOUND_NONE) {
        return RNS8_INVALID_ARGUMENT;
      }
      return column_major ? RNS8_UNSUPPORTED_BACKEND : RNS8_SUCCESS;
    case RNS8_WRAP_U64_MOD_2_64:
      if (desc.bound_kind != RNS8_BOUND_NONE || prefix != 0) {
        return RNS8_INVALID_ARGUMENT;
      }
      return column_major ? RNS8_UNSUPPORTED_BACKEND : RNS8_SUCCESS;
    case RNS8_FINITE_RING_U8:
    case RNS8_FINITE_FIELD_U8:
      if (prefix != 0 || desc.max_prefix != 0 || desc.bound_kind != RNS8_BOUND_NONE) {
        return RNS8_INVALID_ARGUMENT;
      }
      return column_major ? RNS8_UNSUPPORTED_BACKEND : RNS8_SUCCESS;
  }
  return RNS8_INVALID_ARGUMENT;
}

}  // namespace rns8::detail
