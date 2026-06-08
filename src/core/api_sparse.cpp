#include "core/api_internal.hpp"

#include <cstdint>
#include <limits>

namespace {

bool sparse_contract_valid(const rns8_sparse_matrix_desc& desc, uint64_t& group_count, uint64_t& value_count) {
  if (!rns8::detail::valid_abi(desc.struct_size, desc.abi_version, sizeof(desc)) || desc.flags != 0 ||
      desc.reserved0 != 0 || desc.contract != RNS8_SPARSE_A_4_TO_2_STRUCTURED_K ||
      desc.sparse_operand != RNS8_SPARSE_OPERAND_A ||
      desc.index_layout != RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1 ||
      (desc.value_signedness != RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8 &&
       desc.value_signedness != RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8) ||
      desc.rows <= 0 || desc.expanded_k <= 0 || desc.group_size != 4 || desc.nonzeros_per_group != 2 ||
      (desc.expanded_k % 4) != 0) {
    return false;
  }
  const auto rows = static_cast<uint64_t>(desc.rows);
  const auto groups_per_row = static_cast<uint64_t>(desc.expanded_k / 4);
  if (groups_per_row != 0 && rows > std::numeric_limits<uint64_t>::max() / groups_per_row) {
    return false;
  }
  group_count = rows * groups_per_row;
  if (group_count > std::numeric_limits<uint64_t>::max() / 2u) {
    return false;
  }
  value_count = group_count * 2u;
  return true;
}

bool dense_ld_valid(const rns8_sparse_matrix_desc& desc, int64_t dense_ld) {
  return dense_ld >= desc.expanded_k;
}

}  // namespace

rns8_status rns8_sparse_a_4_to_2_layout_counts(
    const rns8_sparse_matrix_desc* desc,
    uint64_t* group_count,
    uint64_t* packed_value_count) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!desc || !group_count || !packed_value_count) {
      return RNS8_INVALID_ARGUMENT;
    }
    uint64_t groups = 0;
    uint64_t values = 0;
    if (!sparse_contract_valid(*desc, groups, values)) {
      return RNS8_INVALID_ARGUMENT;
    }
    *group_count = groups;
    *packed_value_count = values;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_pack_sparse_a_4_to_2_u8(
    const rns8_sparse_matrix_desc* desc,
    const uint8_t* dense_a,
    int64_t dense_ld,
    uint8_t* packed_values,
    uint8_t* packed_indices) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!desc || !dense_a || !packed_values || !packed_indices || !dense_ld_valid(*desc, dense_ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    uint64_t group_count = 0;
    uint64_t packed_value_count = 0;
    if (!sparse_contract_valid(*desc, group_count, packed_value_count)) {
      return RNS8_INVALID_ARGUMENT;
    }
    (void)packed_value_count;
    const auto rows = static_cast<uint64_t>(desc->rows);
    const auto groups_per_row = static_cast<uint64_t>(desc->expanded_k / 4);
    const auto dense_stride = static_cast<uint64_t>(dense_ld);
    for (uint64_t row = 0; row < rows; ++row) {
      for (uint64_t group = 0; group < groups_per_row; ++group) {
        uint8_t indices[2] = {};
        uint8_t values[2] = {};
        uint32_t nonzero_count = 0;
        const uint64_t dense_base = row * dense_stride + group * 4u;
        for (uint8_t lane = 0; lane < 4u; ++lane) {
          const uint8_t value = dense_a[dense_base + lane];
          if (value == 0) {
            continue;
          }
          if (nonzero_count >= 2u) {
            return RNS8_INVALID_ARGUMENT;
          }
          indices[nonzero_count] = lane;
          values[nonzero_count] = value;
          ++nonzero_count;
        }
        if (nonzero_count != 2u || indices[0] >= indices[1]) {
          return RNS8_INVALID_ARGUMENT;
        }
        const uint64_t packed_group = row * groups_per_row + group;
        packed_values[packed_group * 2u] = values[0];
        packed_values[packed_group * 2u + 1u] = values[1];
        packed_indices[packed_group] = static_cast<uint8_t>(indices[0] | (indices[1] << 2u));
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_expand_sparse_a_4_to_2_u8(
    const rns8_sparse_matrix_desc* desc,
    const uint8_t* packed_values,
    const uint8_t* packed_indices,
    uint8_t* dense_a,
    int64_t dense_ld) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!desc || !packed_values || !packed_indices || !dense_a || !dense_ld_valid(*desc, dense_ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    uint64_t group_count = 0;
    uint64_t packed_value_count = 0;
    if (!sparse_contract_valid(*desc, group_count, packed_value_count)) {
      return RNS8_INVALID_ARGUMENT;
    }
    (void)group_count;
    (void)packed_value_count;
    const auto rows = static_cast<uint64_t>(desc->rows);
    const auto groups_per_row = static_cast<uint64_t>(desc->expanded_k / 4);
    const auto dense_stride = static_cast<uint64_t>(dense_ld);
    for (uint64_t row = 0; row < rows; ++row) {
      for (uint64_t group = 0; group < groups_per_row; ++group) {
        const uint64_t packed_group = row * groups_per_row + group;
        const uint8_t encoded = packed_indices[packed_group];
        const uint8_t idx0 = static_cast<uint8_t>(encoded & 0x3u);
        const uint8_t idx1 = static_cast<uint8_t>((encoded >> 2u) & 0x3u);
        if (idx0 >= idx1 || packed_values[packed_group * 2u] == 0 ||
            packed_values[packed_group * 2u + 1u] == 0 || (encoded & 0xf0u) != 0) {
          return RNS8_INVALID_ARGUMENT;
        }
        const uint64_t dense_base = row * dense_stride + group * 4u;
        for (uint8_t lane = 0; lane < 4u; ++lane) {
          dense_a[dense_base + lane] = 0;
        }
        dense_a[dense_base + idx0] = packed_values[packed_group * 2u];
        dense_a[dense_base + idx1] = packed_values[packed_group * 2u + 1u];
      }
    }
    return RNS8_SUCCESS;
  });
}
