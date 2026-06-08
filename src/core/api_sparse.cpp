#include "core/api_internal.hpp"

#include <atomic>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace {

using rns8::detail::api::context_accepts_backend;
using rns8::detail::api::finite_matrix_storage_matches;
using rns8::detail::api::matrix_backend_compatible_with_plan;
using rns8::detail::api::matrix_descriptor_matches;
using rns8::detail::api::rns_matrix_storage_matches;
using rns8::detail::api::rns_residue_state_current_for_backend;
using rns8::detail::api::rns_storage_prefix_for_plan;
using rns8::detail::api::storage_bound_kind_for_plan;

constexpr const char* kSparseLayoutVersion = "sparse_a_4_to_2_canonical_2bit_k_groups_v1";

std::atomic<uint64_t> g_next_sparse_matrix_instance_id{1};

struct sparse_layout_counts {
  uint32_t plane_count = 0;
  uint64_t groups_per_plane = 0;
  uint64_t group_count = 0;
  uint64_t packed_value_count = 0;
};

bool sparse_semantic_identity_valid(const rns8_sparse_matrix_desc& desc, uint32_t& plane_count) {
  plane_count = 0;
  if (rns8::detail::api::uses_rns_storage(desc.semantics)) {
    if (desc.finite_modulus != 0 || desc.max_prefix == 0 || desc.max_prefix > RNS8_MAX_SUPPORTED_PREFIX ||
        desc.value_signedness != RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8) {
      return false;
    }
    switch (desc.semantics) {
      case RNS8_BOUNDED_I64:
      case RNS8_BOUNDED_U64:
        if (desc.bound_kind == RNS8_BOUND_NONE) {
          return false;
        }
        break;
      case RNS8_EXACT_WIDE_SIGNED:
      case RNS8_EXACT_WIDE_UNSIGNED:
        if (desc.bound_kind != RNS8_BOUND_NONE) {
          return false;
        }
        break;
      default:
        return false;
    }
    plane_count = desc.max_prefix;
    return true;
  }
  if (rns8::detail::api::uses_finite_storage(desc.semantics)) {
    if (desc.bound_kind != RNS8_BOUND_NONE || desc.max_prefix != 0 ||
        desc.value_signedness != RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8 ||
        !rns8::detail::valid_finite_modulus_for_semantics(desc.semantics, desc.finite_modulus)) {
      return false;
    }
    plane_count = 1;
    return true;
  }
  return false;
}

bool sparse_contract_valid(const rns8_sparse_matrix_desc& desc, sparse_layout_counts& counts) {
  uint32_t plane_count = 0;
  if (!rns8::detail::valid_abi(desc.struct_size, desc.abi_version, sizeof(desc)) || desc.flags != 0 ||
      desc.reserved0 != 0 || desc.contract != RNS8_SPARSE_A_4_TO_2_STRUCTURED_K ||
      desc.sparse_operand != RNS8_SPARSE_OPERAND_A ||
      desc.index_layout != RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1 ||
      !sparse_semantic_identity_valid(desc, plane_count) || desc.rows <= 0 || desc.expanded_k <= 0 ||
      desc.group_size != 4 || desc.nonzeros_per_group != 2 || (desc.expanded_k % 4) != 0) {
    return false;
  }
  const auto rows = static_cast<uint64_t>(desc.rows);
  const auto groups_per_row = static_cast<uint64_t>(desc.expanded_k / 4);
  if (groups_per_row != 0 && rows > std::numeric_limits<uint64_t>::max() / groups_per_row) {
    return false;
  }
  const uint64_t groups_per_plane = rows * groups_per_row;
  if (groups_per_plane != 0 &&
      static_cast<uint64_t>(plane_count) > std::numeric_limits<uint64_t>::max() / groups_per_plane) {
    return false;
  }
  const uint64_t group_count = groups_per_plane * static_cast<uint64_t>(plane_count);
  if (group_count > std::numeric_limits<uint64_t>::max() / 2u) {
    return false;
  }
  counts.plane_count = plane_count;
  counts.groups_per_plane = groups_per_plane;
  counts.group_count = group_count;
  counts.packed_value_count = group_count * 2u;
  return true;
}

bool dense_ld_valid(const rns8_sparse_matrix_desc& desc, int64_t dense_ld) {
  return dense_ld >= desc.expanded_k;
}

bool dense_plane_byte_count(const rns8_sparse_matrix_desc& desc, uint32_t plane_count, std::size_t& bytes) {
  const auto rows = static_cast<uint64_t>(desc.rows);
  const auto cols = static_cast<uint64_t>(desc.expanded_k);
  if (rows != 0 && cols > std::numeric_limits<uint64_t>::max() / rows) {
    return false;
  }
  const uint64_t cells_per_plane = rows * cols;
  if (cells_per_plane != 0 &&
      static_cast<uint64_t>(plane_count) > std::numeric_limits<uint64_t>::max() / cells_per_plane) {
    return false;
  }
  const uint64_t total_cells = cells_per_plane * static_cast<uint64_t>(plane_count);
  if (total_cells > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  bytes = static_cast<std::size_t>(total_cells);
  return true;
}

uint64_t assign_sparse_matrix_instance_id() {
  return g_next_sparse_matrix_instance_id.fetch_add(1, std::memory_order_relaxed);
}

const char* sparse_signedness_name(rns8_sparse_value_signedness signedness) {
  switch (signedness) {
    case RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8:
      return "signed_i8";
    case RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8:
      return "unsigned_u8";
    default:
      return "unspecified";
  }
}

std::string build_sparse_cache_key(const rns8_sparse_matrix& matrix) {
  const auto& desc = matrix.desc;
  std::string key = "backend=" + std::string(rns8::detail::api::backend_name(matrix.backend));
  key += ";semantics=" + std::string(rns8::detail::api::semantics_name_for_key(desc.semantics));
  key += ";contract=sparse_a_4_to_2_structured_k_v1";
  key += ";operand=a";
  key += ";index_layout=canonical_2bit_k_groups_v1";
  key += ";signedness=" + std::string(sparse_signedness_name(desc.value_signedness));
  key += ";rows=" + std::to_string(desc.rows);
  key += ";expanded_k=" + std::to_string(desc.expanded_k);
  key += ";group_size=" + std::to_string(desc.group_size);
  key += ";nonzeros_per_group=" + std::to_string(desc.nonzeros_per_group);
  key += ";max_prefix=" + std::to_string(desc.max_prefix);
  key += ";finite_modulus=" + std::to_string(desc.finite_modulus);
  key += ";source_version=" + std::to_string(matrix.source_version);
  key += ";layout=" + std::string(kSparseLayoutVersion);
  return key;
}

rns8_matrix make_sparse_expanded_rns_matrix(const rns8_sparse_matrix& sparse, const rns8_plan& plan) {
  rns8_matrix dense{};
  const uint32_t storage_prefix = rns8::detail::api::rns_storage_prefix_for_plan(plan);
  dense.desc = rns8::detail::api::make_matrix_desc(
      plan.desc.m,
      plan.desc.k,
      plan.desc.semantics,
      rns8::detail::api::storage_bound_kind_for_plan(plan),
      storage_prefix,
      plan.desc.tile_m,
      plan.desc.tile_n);
  dense.backend = RNS8_BACKEND_CPU_REFERENCE;
  dense.matrix_instance_id = sparse.matrix_instance_id;
  dense.prefix = sparse.desc.max_prefix;
  dense.source_version = sparse.source_version;
  dense.host_residues_current = true;
  dense.device_residues_current = false;
  dense.residues.resize(
      static_cast<std::size_t>(dense.prefix) * static_cast<std::size_t>(dense.desc.rows) *
      static_cast<std::size_t>(dense.desc.cols));

  std::vector<uint8_t> expanded(dense.residues.size(), 0);
  rns8_sparse_matrix_desc expansion_desc = sparse.desc;
  expansion_desc.max_prefix = storage_prefix;
  const rns8_status status = rns8_expand_sparse_a_4_to_2_u8(
      &expansion_desc,
      sparse.packed_values.data(),
      sparse.packed_indices.data(),
      expanded.data(),
      sparse.desc.expanded_k);
  if (status != RNS8_SUCCESS) {
    dense.residues.clear();
    return dense;
  }
  for (std::size_t i = 0; i < dense.residues.size(); ++i) {
    dense.residues[i] = static_cast<int8_t>(expanded[i]);
  }
  return dense;
}

rns8_matrix make_sparse_expanded_finite_matrix(const rns8_sparse_matrix& sparse, const rns8_plan& plan, uint16_t modulus) {
  rns8_matrix dense{};
  dense.desc = rns8::detail::api::make_matrix_desc(
      plan.desc.m,
      plan.desc.k,
      plan.desc.semantics,
      RNS8_BOUND_NONE,
      0,
      plan.desc.tile_m,
      plan.desc.tile_n);
  dense.backend = RNS8_BACKEND_CPU_REFERENCE;
  dense.matrix_instance_id = sparse.matrix_instance_id;
  dense.prefix = 0;
  dense.finite_modulus = modulus;
  dense.source_version = sparse.source_version;
  dense.host_residues_current = true;
  dense.device_residues_current = false;
  dense.residues.resize(static_cast<std::size_t>(dense.desc.rows) * static_cast<std::size_t>(dense.desc.cols));

  std::vector<uint8_t> expanded(static_cast<std::size_t>(dense.desc.rows) * static_cast<std::size_t>(dense.desc.cols), 0);
  const rns8_status status = rns8_expand_sparse_a_4_to_2_u8(
      &sparse.desc, sparse.packed_values.data(), sparse.packed_indices.data(), expanded.data(), sparse.desc.expanded_k);
  if (status != RNS8_SUCCESS) {
    dense.residues.clear();
    return dense;
  }
  rns8::detail::pack_finite_u8_matrix(dense, expanded.data(), dense.desc.cols, modulus);
  return dense;
}

bool sparse_device_backend_supported(rns8_backend_kind backend) {
#if defined(RNS8_ENABLE_AMDGPU_BUILTINS) && RNS8_ENABLE_AMDGPU_BUILTINS && \
    defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  return backend == RNS8_BACKEND_AMDGPU_BUILTINS;
#else
  (void)backend;
  return false;
#endif
}

rns8_status free_sparse_device_storage(rns8_sparse_matrix& matrix) {
  rns8_status status = RNS8_SUCCESS;
  if (matrix.hip_packed_values) {
    status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_packed_values);
    matrix.hip_packed_values = nullptr;
    matrix.hip_packed_value_bytes = 0;
  }
  if (matrix.hip_packed_indices) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_packed_indices);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_packed_indices = nullptr;
    matrix.hip_packed_index_bytes = 0;
  }
  matrix.device_current = false;
  return status;
}

rns8_status ensure_sparse_device_storage(rns8_context& ctx, rns8_sparse_matrix& matrix) {
  if (!sparse_device_backend_supported(matrix.backend) || matrix.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  const std::size_t value_bytes = matrix.packed_values.size() * sizeof(uint8_t);
  const std::size_t index_bytes = matrix.packed_indices.size() * sizeof(uint8_t);
  if (value_bytes == 0 || index_bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if ((matrix.hip_packed_values && matrix.hip_packed_value_bytes != value_bytes) ||
      (matrix.hip_packed_indices && matrix.hip_packed_index_bytes != index_bytes)) {
    const rns8_status free_status = free_sparse_device_storage(matrix);
    if (free_status != RNS8_SUCCESS) {
      return free_status;
    }
  }
  if (!matrix.hip_packed_values) {
    rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, value_bytes, &matrix.hip_packed_values);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    matrix.hip_packed_value_bytes = value_bytes;
  }
  if (!matrix.hip_packed_indices) {
    rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, index_bytes, &matrix.hip_packed_indices);
    if (status != RNS8_SUCCESS) {
      (void)free_sparse_device_storage(matrix);
      return status;
    }
    matrix.hip_packed_index_bytes = index_bytes;
  }
  return RNS8_SUCCESS;
}

rns8_status upload_sparse_device_storage(rns8_context& ctx, rns8_sparse_matrix& matrix) {
  if (!matrix.host_current) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_status status = ensure_sparse_device_storage(ctx, matrix);
  if (status != RNS8_SUCCESS) {
    matrix.device_current = false;
    return status;
  }
  status = rns8::detail::hip_direct_copy_host_to_device_labeled(
      ctx.device_id,
      "sparse_a_values_h2d",
      matrix.hip_packed_values,
      matrix.packed_values.data(),
      matrix.packed_values.size() * sizeof(uint8_t));
  if (status == RNS8_SUCCESS) {
    status = rns8::detail::hip_direct_copy_host_to_device_labeled(
        ctx.device_id,
        "sparse_a_indices_h2d",
        matrix.hip_packed_indices,
        matrix.packed_indices.data(),
        matrix.packed_indices.size() * sizeof(uint8_t));
  }
  matrix.device_current = status == RNS8_SUCCESS;
  return status;
}

bool sparse_rns_matches_plan(const rns8_sparse_matrix& sparse, const rns8_plan& plan) {
  return rns8::detail::api::uses_rns_storage(plan.desc.semantics) && sparse.desc.semantics == plan.desc.semantics &&
         sparse.desc.bound_kind == rns8::detail::api::storage_bound_kind_for_plan(plan) &&
         sparse.desc.rows == plan.desc.m && sparse.desc.expanded_k == plan.desc.k &&
         sparse.desc.max_prefix >= rns8::detail::api::rns_storage_prefix_for_plan(plan) &&
         sparse.desc.finite_modulus == 0 && sparse.value_plane_count == sparse.desc.max_prefix &&
         sparse.value_plane_count != 0;
}

bool sparse_finite_matches_plan(const rns8_sparse_matrix& sparse, const rns8_plan& plan, uint16_t modulus) {
  return rns8::detail::api::uses_finite_storage(plan.desc.semantics) && sparse.desc.semantics == plan.desc.semantics &&
         sparse.desc.bound_kind == RNS8_BOUND_NONE && sparse.desc.rows == plan.desc.m &&
         sparse.desc.expanded_k == plan.desc.k && sparse.desc.max_prefix == 0 &&
         sparse.desc.finite_modulus == modulus && sparse.value_plane_count == 1;
}

rns8_status validate_sparse_rns_device_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_sparse_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C) {
  if (plan.backend != RNS8_BACKEND_AMDGPU_BUILTINS || A.backend != RNS8_BACKEND_AMDGPU_BUILTINS ||
      !sparse_device_backend_supported(plan.backend)) {
    return RNS8_UNSUPPORTED_BACKEND;
  }
  if (!context_accepts_backend(ctx, plan.backend) || !sparse_rns_matches_plan(A, plan) || !A.device_current ||
      A.hip_device_id != ctx.device_id || !A.hip_packed_values || !A.hip_packed_indices ||
      !matrix_backend_compatible_with_plan(ctx, B, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend) ||
      B.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  const uint32_t storage_prefix = rns_storage_prefix_for_plan(plan);
  const rns8_bound_kind storage_bound_kind = storage_bound_kind_for_plan(plan);
  if (!matrix_descriptor_matches(
          B, plan.desc.semantics, storage_bound_kind, plan.desc.k, plan.desc.n, storage_prefix, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          C, plan.desc.semantics, storage_bound_kind, plan.desc.m, plan.desc.n, storage_prefix, plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!rns_matrix_storage_matches(B, plan.backend, plan.desc.k, plan.desc.n, storage_prefix) ||
      !rns_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n, storage_prefix) ||
      !rns_residue_state_current_for_backend(B, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status validate_sparse_finite_device_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_sparse_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C) {
  if (plan.backend != RNS8_BACKEND_AMDGPU_BUILTINS || A.backend != RNS8_BACKEND_AMDGPU_BUILTINS ||
      !sparse_device_backend_supported(plan.backend)) {
    return RNS8_UNSUPPORTED_BACKEND;
  }
  if (!context_accepts_backend(ctx, plan.backend) || !sparse_finite_matches_plan(A, plan, modulus) ||
      !A.device_current || A.hip_device_id != ctx.device_id || !A.hip_packed_values || !A.hip_packed_indices ||
      !matrix_backend_compatible_with_plan(ctx, B, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend) ||
      B.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(
          B, plan.desc.semantics, RNS8_BOUND_NONE, plan.desc.k, plan.desc.n, 0, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          C, plan.desc.semantics, RNS8_BOUND_NONE, plan.desc.m, plan.desc.n, 0, plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!finite_matrix_storage_matches(B, plan.backend, plan.desc.k, plan.desc.n) ||
      !finite_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n) ||
      !rns_residue_state_current_for_backend(B, plan.backend) || B.finite_modulus != modulus) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
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
    sparse_layout_counts counts{};
    if (!sparse_contract_valid(*desc, counts)) {
      return RNS8_INVALID_ARGUMENT;
    }
    *group_count = counts.group_count;
    *packed_value_count = counts.packed_value_count;
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
    sparse_layout_counts counts{};
    if (!sparse_contract_valid(*desc, counts)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const auto rows = static_cast<uint64_t>(desc->rows);
    const auto groups_per_row = static_cast<uint64_t>(desc->expanded_k / 4);
    const auto dense_stride = static_cast<uint64_t>(dense_ld);
    for (uint32_t plane = 0; plane < counts.plane_count; ++plane) {
      const uint64_t dense_plane_base = static_cast<uint64_t>(plane) * rows * dense_stride;
      const uint64_t packed_plane_base = static_cast<uint64_t>(plane) * counts.groups_per_plane;
      for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t group = 0; group < groups_per_row; ++group) {
          uint8_t indices[2] = {};
          uint8_t values[2] = {};
          uint32_t nonzero_count = 0;
          const uint64_t dense_base = dense_plane_base + row * dense_stride + group * 4u;
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
          const uint64_t packed_group = packed_plane_base + row * groups_per_row + group;
          packed_values[packed_group * 2u] = values[0];
          packed_values[packed_group * 2u + 1u] = values[1];
          packed_indices[packed_group] = static_cast<uint8_t>(indices[0] | (indices[1] << 2u));
        }
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
    sparse_layout_counts counts{};
    if (!sparse_contract_valid(*desc, counts)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const auto rows = static_cast<uint64_t>(desc->rows);
    const auto groups_per_row = static_cast<uint64_t>(desc->expanded_k / 4);
    const auto dense_stride = static_cast<uint64_t>(dense_ld);
    for (uint32_t plane = 0; plane < counts.plane_count; ++plane) {
      const uint64_t dense_plane_base = static_cast<uint64_t>(plane) * rows * dense_stride;
      const uint64_t packed_plane_base = static_cast<uint64_t>(plane) * counts.groups_per_plane;
      for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t group = 0; group < groups_per_row; ++group) {
          const uint64_t packed_group = packed_plane_base + row * groups_per_row + group;
          const uint8_t encoded = packed_indices[packed_group];
          const uint8_t idx0 = static_cast<uint8_t>(encoded & 0x3u);
          const uint8_t idx1 = static_cast<uint8_t>((encoded >> 2u) & 0x3u);
          if (idx0 >= idx1 || packed_values[packed_group * 2u] == 0 ||
              packed_values[packed_group * 2u + 1u] == 0 || (encoded & 0xf0u) != 0) {
            return RNS8_INVALID_ARGUMENT;
          }
          const uint64_t dense_base = dense_plane_base + row * dense_stride + group * 4u;
          for (uint8_t lane = 0; lane < 4u; ++lane) {
            dense_a[dense_base + lane] = 0;
          }
          dense_a[dense_base + idx0] = packed_values[packed_group * 2u];
          dense_a[dense_base + idx1] = packed_values[packed_group * 2u + 1u];
        }
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_create_sparse_matrix(
    rns8_context* ctx,
    const rns8_sparse_matrix_desc* desc,
    rns8_sparse_matrix** out) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    sparse_layout_counts counts{};
    if (!sparse_contract_valid(*desc, counts)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::api::backend_supports_semantics(ctx->backend, desc->semantics)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (ctx->backend != RNS8_BACKEND_CPU_REFERENCE && !sparse_device_backend_supported(ctx->backend)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    auto* matrix = new (std::nothrow) rns8_sparse_matrix();
    if (!matrix) {
      return RNS8_INTERNAL_ERROR;
    }
    matrix->desc = *desc;
    matrix->backend = ctx->backend;
    matrix->matrix_instance_id = assign_sparse_matrix_instance_id();
    matrix->value_plane_count = counts.plane_count;
    matrix->group_count = counts.group_count;
    matrix->packed_value_count = counts.packed_value_count;
    matrix->packed_values.assign(static_cast<std::size_t>(counts.packed_value_count), 0);
    matrix->packed_indices.assign(static_cast<std::size_t>(counts.group_count), 0);
    matrix->hip_device_id = ctx->device_id;
    matrix->cache_key = build_sparse_cache_key(*matrix);
    *out = matrix;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_sparse_matrix(rns8_sparse_matrix* matrix) {
  if (matrix) {
    const rns8_status status = free_sparse_device_storage(*matrix);
    delete matrix;
    return status;
  }
  delete matrix;
  return RNS8_SUCCESS;
}

rns8_status rns8_get_sparse_matrix_storage_info(
    const rns8_sparse_matrix* matrix,
    rns8_sparse_matrix_storage_info* out) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!matrix || !out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = matrix->backend;
    out->semantics = matrix->desc.semantics;
    out->bound_kind = matrix->desc.bound_kind;
    out->contract = matrix->desc.contract;
    out->sparse_operand = matrix->desc.sparse_operand;
    out->index_layout = matrix->desc.index_layout;
    out->value_signedness = matrix->desc.value_signedness;
    out->rows = matrix->desc.rows;
    out->expanded_k = matrix->desc.expanded_k;
    out->group_size = matrix->desc.group_size;
    out->nonzeros_per_group = matrix->desc.nonzeros_per_group;
    out->max_prefix = matrix->desc.max_prefix;
    out->finite_modulus = matrix->desc.finite_modulus;
    out->matrix_instance_id = matrix->matrix_instance_id;
    out->source_version = matrix->source_version;
    out->group_count = matrix->group_count;
    out->packed_value_count = matrix->packed_value_count;
    out->host_packed_value_bytes = matrix->packed_values.size();
    out->host_packed_index_bytes = matrix->packed_indices.size();
    out->device_packed_value_bytes = matrix->hip_packed_value_bytes;
    out->device_packed_index_bytes = matrix->hip_packed_index_bytes;
    out->host_current = matrix->host_current ? 1u : 0u;
    out->device_current = matrix->device_current ? 1u : 0u;
    out->hip_device_id = matrix->hip_device_id;
    rns8::detail::copy_c_string(out->layout_version, sizeof(out->layout_version), kSparseLayoutVersion);
    rns8::detail::copy_c_string(out->cache_key, sizeof(out->cache_key), matrix->cache_key);
    rns8::detail::copy_c_string(
        out->detail,
        sizeof(out->detail),
        "resident_sparse_a_explicit_4_to_2_contract_cpu_reference_anchor_v1");
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_pack_sparse_a_4_to_2_matrix_u8(
    rns8_context* ctx,
    rns8_sparse_matrix* matrix,
    const uint8_t* dense_a_planes,
    int64_t dense_ld,
    uint64_t source_version) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!ctx || !matrix || !dense_a_planes || ctx->backend != matrix->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status status = rns8_pack_sparse_a_4_to_2_u8(
        &matrix->desc,
        dense_a_planes,
        dense_ld,
        matrix->packed_values.data(),
        matrix->packed_indices.data());
    if (status != RNS8_SUCCESS) {
      return status;
    }
    matrix->source_version = source_version;
    matrix->host_current = true;
    matrix->device_current = false;
    if (sparse_device_backend_supported(matrix->backend)) {
      const rns8_status upload_status = upload_sparse_device_storage(*ctx, *matrix);
      if (upload_status != RNS8_SUCCESS) {
        return upload_status;
      }
    }
    matrix->cache_key = build_sparse_cache_key(*matrix);
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_expand_sparse_a_4_to_2_matrix_u8(
    const rns8_sparse_matrix* matrix,
    uint8_t* dense_a_planes,
    int64_t dense_ld) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!matrix || !dense_a_planes || !matrix->host_current) {
      return RNS8_INVALID_ARGUMENT;
    }
    return rns8_expand_sparse_a_4_to_2_u8(
        &matrix->desc,
        matrix->packed_values.data(),
        matrix->packed_indices.data(),
        dense_a_planes,
        dense_ld);
  });
}

rns8_status rns8_gemm_rns_sparse_a(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_sparse_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = rns8::detail::api::validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    if (!sparse_rns_matches_plan(*A, *plan)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->backend == RNS8_BACKEND_CPU_REFERENCE && A->backend == RNS8_BACKEND_CPU_REFERENCE) {
      if (!A->host_current) {
        return RNS8_INVALID_ARGUMENT;
      }
      rns8_matrix dense_a = make_sparse_expanded_rns_matrix(*A, *plan);
      if (dense_a.residues.empty()) {
        return RNS8_INVALID_ARGUMENT;
      }
      const rns8_status operand_status =
          rns8::detail::api::validate_rns_gemm_operands(*ctx, *plan, dense_a, *B, *C);
      if (operand_status != RNS8_SUCCESS) {
        return operand_status;
      }
      const rns8_status status = rns8::detail::cpu_gemm_rns(*plan, dense_a, *B, *C);
      if (status == RNS8_SUCCESS) {
        rns8::detail::api::mark_output_host_residues_current(*C);
        C->source_version = rns8::detail::api::gemm_output_source_version_values(A->source_version, B->source_version);
      }
      return status;
    }
#if defined(RNS8_ENABLE_AMDGPU_BUILTINS) && RNS8_ENABLE_AMDGPU_BUILTINS && \
    defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
    if (plan->backend == RNS8_BACKEND_AMDGPU_BUILTINS && A->backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
      if (!plan->tile_schedule.empty() || plan->schedule_adaptive_prefix_active) {
        return RNS8_UNSUPPORTED_BACKEND;
      }
      const rns8_status operand_status = validate_sparse_rns_device_operands(*ctx, *plan, *A, *B, *C);
      if (operand_status != RNS8_SUCCESS) {
        return operand_status;
      }
      const rns8_status status = rns8::detail::amdgpu_builtins_gemm_rns_sparse_a_device(
          ctx->device_id,
          A->hip_packed_values,
          A->hip_packed_indices,
          B->hip_residues,
          C->hip_residues,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          B->desc.cols,
          C->desc.cols,
          plan->prefix);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      rns8::detail::api::mark_output_device_residues_current(*C);
      C->source_version = rns8::detail::api::gemm_output_source_version_values(A->source_version, B->source_version);
      return RNS8_SUCCESS;
    }
#endif
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_gemm_finite_u8_sparse_a(
    rns8_context* ctx,
    const rns8_plan* plan,
    uint16_t modulus,
    const rns8_sparse_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = rns8::detail::api::validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    if (!sparse_finite_matches_plan(*A, *plan, modulus)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->backend == RNS8_BACKEND_CPU_REFERENCE && A->backend == RNS8_BACKEND_CPU_REFERENCE) {
      if (!A->host_current) {
        return RNS8_INVALID_ARGUMENT;
      }
      rns8_matrix dense_a = make_sparse_expanded_finite_matrix(*A, *plan, modulus);
      if (dense_a.residues.empty()) {
        return RNS8_INVALID_ARGUMENT;
      }
      const rns8_status operand_status =
          rns8::detail::api::validate_finite_gemm_operands(*ctx, *plan, modulus, dense_a, *B, *C);
      if (operand_status != RNS8_SUCCESS) {
        return operand_status;
      }
      const rns8_status status = rns8::detail::cpu_gemm_finite_u8(*plan, modulus, dense_a, *B, *C);
      if (status == RNS8_SUCCESS) {
        rns8::detail::api::mark_output_host_residues_current(*C);
        C->finite_modulus = modulus;
        C->source_version = rns8::detail::api::gemm_output_source_version_values(A->source_version, B->source_version);
      }
      return status;
    }
#if defined(RNS8_ENABLE_AMDGPU_BUILTINS) && RNS8_ENABLE_AMDGPU_BUILTINS && \
    defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
    if (plan->backend == RNS8_BACKEND_AMDGPU_BUILTINS && A->backend == RNS8_BACKEND_AMDGPU_BUILTINS) {
      if (!plan->tile_schedule.empty() || plan->schedule_adaptive_prefix_active) {
        return RNS8_UNSUPPORTED_BACKEND;
      }
      const rns8_status operand_status = validate_sparse_finite_device_operands(*ctx, *plan, modulus, *A, *B, *C);
      if (operand_status != RNS8_SUCCESS) {
        return operand_status;
      }
      const rns8_status status = rns8::detail::amdgpu_builtins_gemm_finite_u8_sparse_a_device(
          ctx->device_id,
          A->hip_packed_values,
          A->hip_packed_indices,
          B->hip_residues,
          C->hip_residues,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          B->desc.cols,
          C->desc.cols,
          modulus);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      rns8::detail::api::mark_output_device_residues_current(*C);
      C->finite_modulus = modulus;
      C->source_version = rns8::detail::api::gemm_output_source_version_values(A->source_version, B->source_version);
      return RNS8_SUCCESS;
    }
#endif
    return RNS8_UNSUPPORTED_BACKEND;
  });
}
