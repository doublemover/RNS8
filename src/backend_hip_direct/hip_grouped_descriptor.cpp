#include "backend_hip_direct/hip_backend.hpp"

#include "core/api_internal.hpp"
#include "core/internal.hpp"

#include <cstdint>
#include <limits>
#include <utility>

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

void keep_first_grouped_error(rns8_status& first, rns8_status next) {
  if (first == RNS8_SUCCESS && next != RNS8_SUCCESS) {
    first = next;
  }
}

rns8_status checked_grouped_pointer_table_bytes(uint32_t task_count, std::size_t* out_bytes) {
  if (!out_bytes ||
      static_cast<std::size_t>(task_count) >
          std::numeric_limits<std::size_t>::max() / sizeof(const void*)) {
    return RNS8_INVALID_ARGUMENT;
  }
  *out_bytes = static_cast<std::size_t>(task_count) * sizeof(const void*);
  return RNS8_SUCCESS;
}

rns8_status checked_grouped_elements(
    uint32_t task_count,
    int64_t rows,
    int64_t cols,
    std::size_t lanes_per_cell,
    std::size_t* out_elements) {
  if (!out_elements || rows <= 0 || cols <= 0 || lanes_per_cell == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const auto row_count = static_cast<std::size_t>(rows);
  const auto col_count = static_cast<std::size_t>(cols);
  const auto task_count_size = static_cast<std::size_t>(task_count);
  if (row_count > std::numeric_limits<std::size_t>::max() / col_count) {
    return RNS8_INVALID_ARGUMENT;
  }
  std::size_t elements = row_count * col_count;
  if (elements > std::numeric_limits<std::size_t>::max() / lanes_per_cell) {
    return RNS8_INVALID_ARGUMENT;
  }
  elements *= lanes_per_cell;
  if (task_count_size != 0 && elements > std::numeric_limits<std::size_t>::max() / task_count_size) {
    return RNS8_INVALID_ARGUMENT;
  }
  *out_elements = task_count_size * elements;
  return RNS8_SUCCESS;
}

rns8_status checked_grouped_bytes(
    uint32_t task_count,
    int64_t rows,
    int64_t cols,
    std::size_t lanes_per_cell,
    std::size_t cell_bytes,
    std::size_t* out_elements,
    std::size_t* out_bytes) {
  if (!out_bytes || cell_bytes == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  std::size_t elements = 0;
  rns8_status status = checked_grouped_elements(task_count, rows, cols, lanes_per_cell, &elements);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (elements > std::numeric_limits<std::size_t>::max() / cell_bytes) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (out_elements) {
    *out_elements = elements;
  }
  *out_bytes = elements * cell_bytes;
  return RNS8_SUCCESS;
}

rns8_status checked_grouped_total_extent_i64(
    uint32_t task_count,
    int64_t rows,
    int64_t cols,
    std::size_t lanes_per_cell,
    int64_t* out_extent) {
  std::size_t elements = 0;
  rns8_status status = checked_grouped_elements(task_count, rows, cols, lanes_per_cell, &elements);
  if (status != RNS8_SUCCESS || elements > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    return RNS8_INVALID_ARGUMENT;
  }
  *out_extent = static_cast<int64_t>(elements);
  return RNS8_SUCCESS;
}

rns8_status validate_grouped_resources_for_descriptor(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    const hip_direct_grouped_device_resources& resources,
    int* out_device_id = nullptr) {
  int descriptor_device_id = -1;
  rns8_status status = hip_direct_validate_grouped_gemm_descriptor_setup(descriptor, &descriptor_device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (resources.device_id != descriptor_device_id ||
      resources.a_matrices.size() != descriptor.task_count ||
      resources.b_matrices.size() != descriptor.task_count ||
      resources.c_matrices.size() != descriptor.task_count ||
      !resources.a_residue_ptrs.ptr || !resources.b_residue_ptrs.ptr || !resources.c_residue_ptrs.ptr) {
    return RNS8_INVALID_ARGUMENT;
  }
  for (uint32_t index = 0; index < descriptor.task_count; ++index) {
    if (resources.a_matrices[index] != descriptor.tasks[index].a ||
        resources.b_matrices[index] != descriptor.tasks[index].b ||
        resources.c_matrices[index] != descriptor.tasks[index].c) {
      return RNS8_INVALID_ARGUMENT;
    }
  }
  if (out_device_id) {
    *out_device_id = descriptor_device_id;
  }
  return RNS8_SUCCESS;
}

}  // namespace

rns8_status hip_direct_build_same_shape_grouped_bucket_plan(
    const rns8_plan* plan,
    const hip_direct_grouped_gemm_task* tasks,
    uint32_t task_count,
    rns8_semantics semantics,
    int64_t m,
    int64_t n,
    int64_t k,
    uint32_t prefix,
    hip_direct_grouped_gemm_bucket_plan* out) {
  if (!out) {
    return RNS8_INVALID_ARGUMENT;
  }
  hip_direct_grouped_gemm_descriptor descriptor{};
  descriptor.plan = plan;
  descriptor.tasks = tasks;
  descriptor.task_count = task_count;
  descriptor.semantics = semantics;
  descriptor.m = m;
  descriptor.n = n;
  descriptor.k = k;
  descriptor.prefix = prefix;

  rns8_status status = hip_direct_validate_grouped_gemm_descriptor_setup(descriptor);
  if (status != RNS8_SUCCESS) {
    return status;
  }

  hip_direct_grouped_gemm_bucket bucket{};
  bucket.descriptor = descriptor;
  bucket.task_offset = 0;
  bucket.task_count = task_count;

  hip_direct_grouped_gemm_bucket_plan next{};
  next.plan = plan;
  next.tasks = tasks;
  next.task_count = task_count;
  next.bucket_count = 1;
  next.semantics = semantics;
  next.m = m;
  next.n = n;
  next.k = k;
  next.prefix = prefix;
  next.same_shape_required = true;
  next.buckets.push_back(bucket);

  *out = std::move(next);
  return RNS8_SUCCESS;
}

const hip_direct_grouped_gemm_descriptor* hip_direct_single_bucket_descriptor(
    const hip_direct_grouped_gemm_bucket_plan& bucket_plan) {
  if (bucket_plan.bucket_count != 1 || bucket_plan.buckets.size() != 1 ||
      bucket_plan.buckets.front().task_offset != 0 ||
      bucket_plan.buckets.front().task_count != bucket_plan.task_count) {
    return nullptr;
  }
  return &bucket_plan.buckets.front().descriptor;
}

hip_direct_grouped_device_buffer::hip_direct_grouped_device_buffer(
    hip_direct_grouped_device_buffer&& other) noexcept {
  move_from(other);
}

hip_direct_grouped_device_buffer& hip_direct_grouped_device_buffer::operator=(
    hip_direct_grouped_device_buffer&& other) noexcept {
  if (this != &other) {
    (void)reset();
    move_from(other);
  }
  return *this;
}

hip_direct_grouped_device_buffer::~hip_direct_grouped_device_buffer() {
  (void)reset();
}

rns8_status hip_direct_grouped_device_buffer::allocate(
    int requested_device_id,
    std::size_t requested_bytes) {
  rns8_status status = reset();
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (requested_device_id < 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  device_id = requested_device_id;
  bytes = requested_bytes;
  if (requested_bytes == 0) {
    return RNS8_SUCCESS;
  }
  void* allocated = nullptr;
  status = hip_direct_allocate(requested_device_id, requested_bytes, &allocated);
  if (status != RNS8_SUCCESS) {
    device_id = -1;
    bytes = 0;
    return status;
  }
  ptr = allocated;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_grouped_device_buffer::reset() noexcept {
  rns8_status status = RNS8_SUCCESS;
  if (ptr) {
    status = hip_direct_free(device_id, ptr);
  }
  device_id = -1;
  ptr = nullptr;
  bytes = 0;
  return status;
}

void hip_direct_grouped_device_buffer::move_from(
    hip_direct_grouped_device_buffer& other) noexcept {
  device_id = other.device_id;
  ptr = other.ptr;
  bytes = other.bytes;
  other.device_id = -1;
  other.ptr = nullptr;
  other.bytes = 0;
}

rns8_status hip_direct_grouped_device_resources::reset() noexcept {
  rns8_status first_status = RNS8_SUCCESS;
  keep_first_grouped_error(first_status, c_residue_ptrs.reset());
  keep_first_grouped_error(first_status, b_residue_ptrs.reset());
  keep_first_grouped_error(first_status, a_residue_ptrs.reset());
  keep_first_grouped_error(first_status, status.reset());
  keep_first_grouped_error(first_status, c_slab.reset());
  keep_first_grouped_error(first_status, b_slab.reset());
  keep_first_grouped_error(first_status, a_slab.reset());
  c_matrices.clear();
  b_matrices.clear();
  a_matrices.clear();
  device_id = -1;
  return first_status;
}

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

rns8_status hip_direct_allocate_grouped_task_device_resources(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    std::size_t a_slab_bytes,
    std::size_t b_slab_bytes,
    std::size_t c_slab_bytes,
    std::size_t status_bytes,
    hip_direct_grouped_device_resources* out) {
  if (!out) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  rns8_status status = hip_direct_validate_grouped_gemm_descriptor_setup(descriptor, &device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  std::size_t pointer_table_bytes = 0;
  status = checked_grouped_pointer_table_bytes(descriptor.task_count, &pointer_table_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }

  status = out->reset();
  if (status != RNS8_SUCCESS) {
    return status;
  }
  out->device_id = device_id;

  auto allocate_or_reset = [&](hip_direct_grouped_device_buffer& buffer,
                               std::size_t bytes) -> rns8_status {
    const rns8_status allocate_status = buffer.allocate(device_id, bytes);
    if (allocate_status != RNS8_SUCCESS) {
      (void)out->reset();
    }
    return allocate_status;
  };

  status = allocate_or_reset(out->a_slab, a_slab_bytes);
  if (status != RNS8_SUCCESS) return status;
  status = allocate_or_reset(out->b_slab, b_slab_bytes);
  if (status != RNS8_SUCCESS) return status;
  status = allocate_or_reset(out->c_slab, c_slab_bytes);
  if (status != RNS8_SUCCESS) return status;
  status = allocate_or_reset(out->status, status_bytes);
  if (status != RNS8_SUCCESS) return status;
  status = allocate_or_reset(out->a_residue_ptrs, pointer_table_bytes);
  if (status != RNS8_SUCCESS) return status;
  status = allocate_or_reset(out->b_residue_ptrs, pointer_table_bytes);
  if (status != RNS8_SUCCESS) return status;
  return allocate_or_reset(out->c_residue_ptrs, pointer_table_bytes);
}

rns8_status hip_direct_prepare_grouped_task_residue_pointers(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    int* out_device_id) {
  int descriptor_device_id = -1;
  rns8_status status = hip_direct_validate_grouped_gemm_descriptor_setup(descriptor, &descriptor_device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (resources.device_id != descriptor_device_id || !resources.a_residue_ptrs.ptr ||
      !resources.b_residue_ptrs.ptr || !resources.c_residue_ptrs.ptr) {
    return RNS8_INVALID_ARGUMENT;
  }

  resources.a_matrices.clear();
  resources.b_matrices.clear();
  resources.c_matrices.clear();
  resources.a_matrices.reserve(descriptor.task_count);
  resources.b_matrices.reserve(descriptor.task_count);
  resources.c_matrices.reserve(descriptor.task_count);
  for (uint32_t index = 0; index < descriptor.task_count; ++index) {
    resources.a_matrices.push_back(descriptor.tasks[index].a);
    resources.b_matrices.push_back(descriptor.tasks[index].b);
    resources.c_matrices.push_back(descriptor.tasks[index].c);
  }

  auto prepare_table = [&](const std::vector<rns8_matrix*>& matrices,
                           hip_direct_grouped_device_buffer& pointer_table) -> rns8_status {
    int pointer_device_id = -1;
    uint32_t pointer_prefix = 0;
    const rns8_status prepare_status = hip_direct_prepare_grouped_matrix_residue_pointers(
        matrices.data(),
        static_cast<uint32_t>(matrices.size()),
        descriptor.semantics,
        pointer_table.ptr,
        pointer_table.bytes,
        &pointer_device_id,
        &pointer_prefix);
    if (prepare_status != RNS8_SUCCESS) {
      return prepare_status;
    }
    if (pointer_device_id != descriptor_device_id || pointer_prefix != descriptor.prefix) {
      return RNS8_INVALID_ARGUMENT;
    }
    return RNS8_SUCCESS;
  };

  status = prepare_table(resources.a_matrices, resources.a_residue_ptrs);
  if (status != RNS8_SUCCESS) return status;
  status = prepare_table(resources.b_matrices, resources.b_residue_ptrs);
  if (status != RNS8_SUCCESS) return status;
  status = prepare_table(resources.c_matrices, resources.c_residue_ptrs);
  if (status != RNS8_SUCCESS) return status;

  if (out_device_id) {
    *out_device_id = descriptor_device_id;
  }
  return RNS8_SUCCESS;
}

rns8_status hip_direct_pack_grouped_i64_task_inputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    const int64_t* a_slab,
    const int64_t* b_slab,
    int64_t lda,
    int64_t ldb,
    uint64_t first_source_version) {
  if (!a_slab || !b_slab ||
      (descriptor.semantics != RNS8_BOUNDED_I64 && descriptor.semantics != RNS8_EXACT_WIDE_SIGNED)) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  rns8_status status = validate_grouped_resources_for_descriptor(descriptor, resources, &device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = hip_direct_pack_i64_grouped_matrices_device(
      device_id,
      a_slab,
      resources.a_slab.ptr,
      resources.a_slab.bytes,
      resources.a_matrices.data(),
      descriptor.task_count,
      descriptor.semantics,
      resources.a_residue_ptrs.ptr,
      descriptor.m,
      descriptor.k,
      lda,
      descriptor.prefix,
      first_source_version);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  return hip_direct_pack_i64_grouped_matrices_device(
      device_id,
      b_slab,
      resources.b_slab.ptr,
      resources.b_slab.bytes,
      resources.b_matrices.data(),
      descriptor.task_count,
      descriptor.semantics,
      resources.b_residue_ptrs.ptr,
      descriptor.k,
      descriptor.n,
      ldb,
      descriptor.prefix,
      first_source_version);
}

rns8_status hip_direct_pack_grouped_u64_task_inputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    const uint64_t* a_slab,
    const uint64_t* b_slab,
    int64_t lda,
    int64_t ldb,
    uint64_t first_source_version) {
  if (!a_slab || !b_slab ||
      (descriptor.semantics != RNS8_BOUNDED_U64 && descriptor.semantics != RNS8_EXACT_WIDE_UNSIGNED)) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  rns8_status status = validate_grouped_resources_for_descriptor(descriptor, resources, &device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = hip_direct_pack_u64_grouped_matrices_device(
      device_id,
      a_slab,
      resources.a_slab.ptr,
      resources.a_slab.bytes,
      resources.a_matrices.data(),
      descriptor.task_count,
      descriptor.semantics,
      resources.a_residue_ptrs.ptr,
      descriptor.m,
      descriptor.k,
      lda,
      descriptor.prefix,
      first_source_version);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  return hip_direct_pack_u64_grouped_matrices_device(
      device_id,
      b_slab,
      resources.b_slab.ptr,
      resources.b_slab.bytes,
      resources.b_matrices.data(),
      descriptor.task_count,
      descriptor.semantics,
      resources.b_residue_ptrs.ptr,
      descriptor.k,
      descriptor.n,
      ldb,
      descriptor.prefix,
      first_source_version);
}

rns8_status hip_direct_pack_grouped_finite_u8_task_inputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    const uint8_t* a_slab,
    const uint8_t* b_slab,
    int64_t lda,
    int64_t ldb,
    uint16_t modulus,
    uint64_t first_source_version) {
  if (!a_slab || !b_slab ||
      (descriptor.semantics != RNS8_FINITE_RING_U8 && descriptor.semantics != RNS8_FINITE_FIELD_U8)) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  rns8_status status = validate_grouped_resources_for_descriptor(descriptor, resources, &device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = hip_direct_pack_finite_u8_grouped_matrices_device(
      device_id,
      a_slab,
      resources.a_slab.ptr,
      resources.a_slab.bytes,
      resources.a_matrices.data(),
      descriptor.task_count,
      descriptor.semantics,
      resources.a_residue_ptrs.ptr,
      descriptor.m,
      descriptor.k,
      lda,
      modulus,
      first_source_version);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  return hip_direct_pack_finite_u8_grouped_matrices_device(
      device_id,
      b_slab,
      resources.b_slab.ptr,
      resources.b_slab.bytes,
      resources.b_matrices.data(),
      descriptor.task_count,
      descriptor.semantics,
      resources.b_residue_ptrs.ptr,
      descriptor.k,
      descriptor.n,
      ldb,
      modulus,
      first_source_version);
}

rns8_status hip_direct_gemm_grouped_rns_task_outputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources) {
  if (descriptor.semantics != RNS8_BOUNDED_I64 && descriptor.semantics != RNS8_BOUNDED_U64 &&
      descriptor.semantics != RNS8_EXACT_WIDE_SIGNED &&
      descriptor.semantics != RNS8_EXACT_WIDE_UNSIGNED) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  rns8_status status = validate_grouped_resources_for_descriptor(descriptor, resources, &device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (descriptor.semantics == RNS8_EXACT_WIDE_SIGNED ||
      descriptor.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
    return hip_direct_gemm_rns_grouped_exact_wide_matrices_device(
        device_id,
        resources.a_matrices.data(),
        resources.b_matrices.data(),
        resources.c_matrices.data(),
        descriptor.task_count,
        descriptor.semantics,
        resources.a_residue_ptrs.ptr,
        resources.b_residue_ptrs.ptr,
        resources.c_residue_ptrs.ptr,
        descriptor.m,
        descriptor.n,
        descriptor.k,
        descriptor.prefix);
  }
  return hip_direct_gemm_rns_grouped_matrices_device(
      device_id,
      resources.a_matrices.data(),
      resources.b_matrices.data(),
      resources.c_matrices.data(),
      descriptor.task_count,
      descriptor.semantics,
      resources.a_residue_ptrs.ptr,
      resources.b_residue_ptrs.ptr,
      resources.c_residue_ptrs.ptr,
      descriptor.m,
      descriptor.n,
      descriptor.k,
      descriptor.prefix);
}

rns8_status hip_direct_gemm_grouped_finite_u8_task_outputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint16_t modulus) {
  if (descriptor.semantics != RNS8_FINITE_RING_U8 && descriptor.semantics != RNS8_FINITE_FIELD_U8) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  rns8_status status = validate_grouped_resources_for_descriptor(descriptor, resources, &device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  return hip_direct_gemm_finite_u8_grouped_matrices_device(
      device_id,
      resources.a_matrices.data(),
      resources.b_matrices.data(),
      resources.c_matrices.data(),
      descriptor.task_count,
      descriptor.semantics,
      resources.a_residue_ptrs.ptr,
      resources.b_residue_ptrs.ptr,
      resources.c_residue_ptrs.ptr,
      descriptor.m,
      descriptor.n,
      descriptor.k,
      modulus);
}

rns8_status hip_direct_export_grouped_i64_task_outputs_to_host(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint64_t bound,
    int64_t* dst,
    std::size_t dst_elements) {
  if (descriptor.semantics != RNS8_BOUNDED_I64 || !dst || !resources.c_slab.ptr ||
      !resources.status.ptr) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  rns8_status status = validate_grouped_resources_for_descriptor(descriptor, resources, &device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  std::size_t required_elements = 0;
  std::size_t required_bytes = 0;
  status = checked_grouped_bytes(
      descriptor.task_count,
      descriptor.m,
      descriptor.n,
      1,
      sizeof(int64_t),
      &required_elements,
      &required_bytes);
  if (status != RNS8_SUCCESS || dst_elements < required_elements || resources.c_slab.bytes < required_bytes) {
    return RNS8_INVALID_ARGUMENT;
  }
  status = hip_direct_export_i64_grouped_matrices_to_device(
      resources.c_matrices.data(),
      descriptor.task_count,
      resources.c_residue_ptrs.ptr,
      resources.c_slab.ptr,
      resources.status.ptr,
      descriptor.m,
      descriptor.n,
      bound);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  int64_t total_extent = 0;
  status = checked_grouped_total_extent_i64(descriptor.task_count, descriptor.m, descriptor.n, 1, &total_extent);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  return hip_direct_copy_compact_matrix_device_to_host(
      device_id,
      "crt_export_d2h",
      dst,
      total_extent,
      resources.c_slab.ptr,
      1,
      total_extent,
      sizeof(int64_t),
      false);
}

rns8_status hip_direct_export_grouped_u64_task_outputs_to_host(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint64_t bound,
    uint64_t* dst,
    std::size_t dst_elements) {
  if (descriptor.semantics != RNS8_BOUNDED_U64 || !dst || !resources.c_slab.ptr ||
      !resources.status.ptr) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  rns8_status status = validate_grouped_resources_for_descriptor(descriptor, resources, &device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  std::size_t required_elements = 0;
  std::size_t required_bytes = 0;
  status = checked_grouped_bytes(
      descriptor.task_count,
      descriptor.m,
      descriptor.n,
      1,
      sizeof(uint64_t),
      &required_elements,
      &required_bytes);
  if (status != RNS8_SUCCESS || dst_elements < required_elements || resources.c_slab.bytes < required_bytes) {
    return RNS8_INVALID_ARGUMENT;
  }
  status = hip_direct_export_u64_grouped_matrices_to_device(
      resources.c_matrices.data(),
      descriptor.task_count,
      resources.c_residue_ptrs.ptr,
      resources.c_slab.ptr,
      resources.status.ptr,
      descriptor.m,
      descriptor.n,
      bound);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  int64_t total_extent = 0;
  status = checked_grouped_total_extent_i64(descriptor.task_count, descriptor.m, descriptor.n, 1, &total_extent);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  return hip_direct_copy_compact_matrix_device_to_host(
      device_id,
      "crt_export_d2h",
      dst,
      total_extent,
      resources.c_slab.ptr,
      1,
      total_extent,
      sizeof(uint64_t),
      false);
}

rns8_status hip_direct_export_grouped_finite_u8_task_outputs_to_host(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint16_t modulus,
    uint8_t* dst,
    std::size_t dst_elements) {
  if ((descriptor.semantics != RNS8_FINITE_RING_U8 && descriptor.semantics != RNS8_FINITE_FIELD_U8) ||
      !dst || !resources.c_slab.ptr) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  rns8_status status = validate_grouped_resources_for_descriptor(descriptor, resources, &device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  std::size_t required_elements = 0;
  std::size_t required_bytes = 0;
  status = checked_grouped_bytes(
      descriptor.task_count,
      descriptor.m,
      descriptor.n,
      1,
      sizeof(uint8_t),
      &required_elements,
      &required_bytes);
  if (status != RNS8_SUCCESS || dst_elements < required_elements || resources.c_slab.bytes < required_bytes) {
    return RNS8_INVALID_ARGUMENT;
  }
  status = hip_direct_export_finite_u8_grouped_matrices_to_device(
      resources.c_matrices.data(),
      descriptor.task_count,
      resources.c_residue_ptrs.ptr,
      resources.c_slab.ptr,
      descriptor.m,
      descriptor.n,
      modulus);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  int64_t total_extent = 0;
  status = checked_grouped_total_extent_i64(descriptor.task_count, descriptor.m, descriptor.n, 1, &total_extent);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  return hip_direct_copy_compact_matrix_device_to_host(
      device_id,
      "finite_export_d2h",
      dst,
      total_extent,
      resources.c_slab.ptr,
      1,
      total_extent,
      sizeof(uint8_t),
      false);
}

rns8_status hip_direct_export_grouped_exact_wide_task_outputs_to_host(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint32_t limb_count,
    bool signed_output,
    uint64_t* dst,
    std::size_t dst_limb_elements) {
  if (!dst || !resources.c_slab.ptr || limb_count == 0 ||
      (signed_output && descriptor.semantics != RNS8_EXACT_WIDE_SIGNED) ||
      (!signed_output && descriptor.semantics != RNS8_EXACT_WIDE_UNSIGNED)) {
    return RNS8_INVALID_ARGUMENT;
  }
  int device_id = -1;
  rns8_status status = validate_grouped_resources_for_descriptor(descriptor, resources, &device_id);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  std::size_t required_elements = 0;
  std::size_t required_bytes = 0;
  status = checked_grouped_bytes(
      descriptor.task_count,
      descriptor.m,
      descriptor.n,
      limb_count,
      sizeof(uint64_t),
      &required_elements,
      &required_bytes);
  if (status != RNS8_SUCCESS || dst_limb_elements < required_elements ||
      resources.c_slab.bytes < required_bytes) {
    return RNS8_INVALID_ARGUMENT;
  }
  status = signed_output
               ? hip_direct_export_exact_wide_signed_grouped_matrix_limbs_to_device(
                     resources.c_matrices.data(),
                     descriptor.task_count,
                     resources.c_residue_ptrs.ptr,
                     resources.c_slab.ptr,
                     descriptor.m,
                     descriptor.n,
                     limb_count)
               : hip_direct_export_exact_wide_unsigned_grouped_matrix_limbs_to_device(
                     resources.c_matrices.data(),
                     descriptor.task_count,
                     resources.c_residue_ptrs.ptr,
                     resources.c_slab.ptr,
                     descriptor.m,
                     descriptor.n,
                     limb_count);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  int64_t total_extent = 0;
  status = checked_grouped_total_extent_i64(
      descriptor.task_count, descriptor.m, descriptor.n, limb_count, &total_extent);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  return hip_direct_copy_compact_matrix_device_to_host(
      device_id,
      "exact_wide_export_d2h",
      dst,
      total_extent,
      resources.c_slab.ptr,
      1,
      total_extent,
      sizeof(uint64_t),
      false);
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
