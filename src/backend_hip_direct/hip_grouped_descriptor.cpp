#include "backend_hip_direct/hip_backend.hpp"

#include "core/api_internal.hpp"
#include "core/internal.hpp"

#include <cstdint>
#include <limits>

namespace rns8::detail {
namespace {

bool grouped_semantics_supported(rns8_semantics semantics) {
  return semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64 ||
         semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED ||
         semantics == RNS8_FINITE_FIELD_U8 || semantics == RNS8_FINITE_RING_U8;
}

bool grouped_source_versions_available(uint64_t first_source_version, uint32_t task_count) {
  return first_source_version != 0 &&
         static_cast<uint64_t>(task_count) <=
             std::numeric_limits<uint64_t>::max() - first_source_version + UINT64_C(1);
}

bool matrix_matches_descriptor(
    const rns8_matrix* matrix,
    rns8_semantics semantics,
    int64_t rows,
    int64_t cols,
    uint32_t prefix) {
  return matrix && matrix->backend == RNS8_BACKEND_HIP_DIRECT && matrix->desc.rows == rows &&
         matrix->desc.cols == cols && matrix->desc.logical_ld == cols &&
         matrix->desc.logical_layout == RNS8_LAYOUT_ROW_MAJOR && matrix->desc.semantics == semantics &&
         matrix->prefix == prefix && matrix->hip_residues && matrix->hip_device_id >= 0;
}

bool workspace_matches_plan(const rns8_plan& plan, const rns8_workspace& workspace) {
  return workspace.backend == RNS8_BACKEND_HIP_DIRECT && workspace.backend == plan.backend &&
         workspace.semantics == plan.desc.semantics && workspace.bound_kind == plan.desc.bound_kind &&
         workspace.m == plan.desc.m && workspace.n == plan.desc.n && workspace.k == plan.desc.k &&
         workspace.bound == plan.desc.bound && workspace.finite_modulus == plan.desc.finite_modulus &&
         workspace.tile_m == plan.desc.tile_m && workspace.tile_n == plan.desc.tile_n &&
         workspace.prefix == plan.prefix && workspace.schedule_tile_rows == plan.schedule_tile_rows &&
         workspace.schedule_tile_cols == plan.schedule_tile_cols &&
         workspace.schedule_tile_count == plan.schedule_tile_count &&
         workspace.schedule_min_required_prefix == plan.schedule_min_required_prefix &&
         workspace.schedule_max_required_prefix == plan.schedule_max_required_prefix &&
         workspace.schedule_min_selected_prefix == plan.schedule_min_selected_prefix &&
         workspace.schedule_max_selected_prefix == plan.schedule_max_selected_prefix &&
         workspace.schedule_prefix_group_count == plan.schedule_prefix_group_count &&
         workspace.schedule_range_bit_length == plan.schedule_range_bit_length &&
         workspace.schedule_adaptive_prefix_active == plan.schedule_adaptive_prefix_active &&
         workspace.schedule_adaptive_skip_active == plan.schedule_adaptive_skip_active &&
         workspace.schedule_flags == plan.schedule_flags &&
         workspace.zero_a_row_count == plan.zero_a_row_count &&
         workspace.zero_b_col_count == plan.zero_b_col_count &&
         workspace.zero_row_col_product_count == plan.zero_row_col_product_count &&
         workspace.schedule_fingerprint == api::plan_workspace_fingerprint(plan) &&
         workspace.backend_workspace_required_bytes == plan.backend_workspace_required_bytes &&
         workspace.backend_selected_kernel == plan.backend_selected_kernel &&
         workspace.backend_library == plan.backend_library &&
         workspace.backend_library_version == plan.backend_library_version &&
         workspace.backend_capability_status == plan.backend_capability_status &&
         workspace.backend_epilogue_mode == plan.backend_epilogue_mode &&
         workspace.backend_workspace_mode == plan.backend_workspace_mode &&
         workspace.backend_isa_evidence == plan.backend_isa_evidence &&
         workspace.backend_target_id == plan.backend_target_id &&
         workspace.backend_autotune_key == plan.backend_autotune_key &&
         workspace.backend_performance_validated == plan.backend_performance_validated;
}

bool task_triplet_owns_distinct_objects(const hip_direct_grouped_gemm_task& task) {
  return task.a != task.b && task.a != task.c && task.b != task.c;
}

bool task_reuses_prior_objects(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    uint32_t task_index) {
  const hip_direct_grouped_gemm_task& task = descriptor.tasks[task_index];
  for (uint32_t prior_index = 0; prior_index < task_index; ++prior_index) {
    const hip_direct_grouped_gemm_task& prior = descriptor.tasks[prior_index];
    if (task.a == prior.a || task.b == prior.b || task.c == prior.c ||
        task.workspace == prior.workspace) {
      return true;
    }
  }
  return false;
}

}  // namespace

rns8_status hip_direct_validate_grouped_gemm_descriptor_setup(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    int* out_device_id) {
  if (!descriptor.plan || !descriptor.tasks || descriptor.task_count <= 1 ||
      !grouped_semantics_supported(descriptor.semantics) ||
      descriptor.m <= 0 || descriptor.n <= 0 || descriptor.k <= 0 ||
      descriptor.plan->backend != RNS8_BACKEND_HIP_DIRECT ||
      descriptor.plan->desc.semantics != descriptor.semantics ||
      descriptor.plan->desc.m != descriptor.m || descriptor.plan->desc.n != descriptor.n ||
      descriptor.plan->desc.k != descriptor.k || descriptor.plan->prefix != descriptor.prefix) {
    return RNS8_INVALID_ARGUMENT;
  }

  int device_id = -1;
  for (uint32_t index = 0; index < descriptor.task_count; ++index) {
    const hip_direct_grouped_gemm_task& task = descriptor.tasks[index];
    if (!task.a || !task.b || !task.c || !task.workspace ||
        !task_triplet_owns_distinct_objects(task) ||
        task_reuses_prior_objects(descriptor, index) ||
        !matrix_matches_descriptor(task.a, descriptor.semantics, descriptor.m, descriptor.k, descriptor.prefix) ||
        !matrix_matches_descriptor(task.b, descriptor.semantics, descriptor.k, descriptor.n, descriptor.prefix) ||
        !matrix_matches_descriptor(task.c, descriptor.semantics, descriptor.m, descriptor.n, descriptor.prefix) ||
        !workspace_matches_plan(*descriptor.plan, *task.workspace)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (task.a->hip_device_id != task.b->hip_device_id || task.a->hip_device_id != task.c->hip_device_id ||
        task.workspace->hip_device_id != task.a->hip_device_id) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (index == 0) {
      device_id = task.a->hip_device_id;
    } else if (task.a->hip_device_id != device_id) {
      return RNS8_INVALID_ARGUMENT;
    }
  }

  if (out_device_id) {
    *out_device_id = device_id;
  }
  return RNS8_SUCCESS;
}

rns8_status hip_direct_validate_grouped_gemm_descriptor_after_pack(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    uint64_t first_source_version) {
  rns8_status status = hip_direct_validate_grouped_gemm_descriptor_setup(descriptor, nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (!grouped_source_versions_available(first_source_version, descriptor.task_count)) {
    return RNS8_INVALID_ARGUMENT;
  }

  for (uint32_t index = 0; index < descriptor.task_count; ++index) {
    const uint64_t expected_source_version = first_source_version + static_cast<uint64_t>(index);
    const hip_direct_grouped_gemm_task& task = descriptor.tasks[index];
    if (!task.a->device_residues_current || !task.b->device_residues_current ||
        task.a->source_version != expected_source_version || task.b->source_version != expected_source_version) {
      return RNS8_INVALID_ARGUMENT;
    }
  }
  return RNS8_SUCCESS;
}

rns8_status hip_direct_validate_grouped_gemm_descriptor_after_gemm(
    const hip_direct_grouped_gemm_descriptor& descriptor) {
  rns8_status status = hip_direct_validate_grouped_gemm_descriptor_setup(descriptor, nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }

  const bool bounded_output =
      descriptor.semantics == RNS8_BOUNDED_I64 || descriptor.semantics == RNS8_BOUNDED_U64;
  for (uint32_t index = 0; index < descriptor.task_count; ++index) {
    const hip_direct_grouped_gemm_task& task = descriptor.tasks[index];
    if (!task.c->device_residues_current) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (bounded_output) {
      const uint64_t expected_source_version =
          api::gemm_output_source_version_values(task.a->source_version, task.b->source_version);
      if (task.c->source_version != expected_source_version) {
        return RNS8_INVALID_ARGUMENT;
      }
    }
  }
  return RNS8_SUCCESS;
}

}  // namespace rns8::detail
