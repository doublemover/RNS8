#include "core/api_internal.hpp"

using namespace rns8::detail::api;

namespace {

struct ExactCpuLift {
  uint32_t source_prefix = 0;
  uint32_t target_prefix = 0;
  std::size_t cells = 0;
  std::size_t source_count = 0;
  std::size_t staged_count = 0;
  boost::multiprecision::cpp_int product;
  std::vector<int8_t> suffix;

  void select(uint32_t target) {
    target_prefix = target;
    staged_count = target > source_prefix ? cells * (target - source_prefix) : 0;
  }

  const int8_t* plane(const rns8_matrix& matrix, uint32_t p) const {
    return p < source_prefix ? matrix.residues.data() + cells * p
                             : suffix.data() + cells * (p - source_prefix);
  }
};

rns8_status exact_cpu_storage(const rns8_context& ctx, const rns8_matrix& matrix, uint32_t target,
                              bool require_current, std::size_t& cells) {
  if (ctx.backend != matrix.backend) return RNS8_INVALID_ARGUMENT;
  if (ctx.backend != RNS8_BACKEND_CPU_REFERENCE) return RNS8_UNSUPPORTED_BACKEND;
  if (!rns8::detail::exact_wide_semantics(matrix.desc.semantics) || target == 0 ||
      target > RNS8_MAX_SUPPORTED_PREFIX || target > matrix.prefix ||
      matrix.desc.max_prefix != matrix.prefix ||
      rns8::detail::validate_matrix_desc(matrix.desc, matrix.prefix) != RNS8_SUCCESS) {
    return RNS8_INVALID_ARGUMENT;
  }
  std::size_t capacity = 0;
  if (!matrix_cell_count(matrix.desc.rows, matrix.desc.cols, cells) ||
      !rns_residue_count(matrix.desc.rows, matrix.desc.cols, matrix.prefix, capacity)) {
    return RNS8_RANGE_ERROR;
  }
  if (!rns_matrix_storage_matches(matrix, ctx.backend, matrix.desc.rows, matrix.desc.cols, matrix.prefix) ||
      (require_current && !rns_residue_state_current_for_backend(matrix, ctx.backend))) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status prepare_exact_cpu_lift(const rns8_context& ctx, const rns8_matrix& matrix, uint32_t target,
                                   ExactCpuLift& lift) {
  const auto status = exact_cpu_storage(ctx, matrix, target, true, lift.cells);
  if (status != RNS8_SUCCESS) return status;
  lift.source_prefix = matrix.exact_range_prefix;
  if (lift.source_prefix == 0 || lift.source_prefix > matrix.prefix) return RNS8_RANGE_ERROR;
  lift.product = rns8::detail::modulus_product(lift.source_prefix);
  // A larger target range cannot recover a prior CRT alias or an unknown value.
  if (!rns8::detail::exact_range_fits(matrix.desc.semantics, matrix.exact_max_magnitude, lift.product)) {
    return RNS8_RANGE_ERROR;
  }
  lift.source_count = lift.cells * lift.source_prefix;
  lift.select(target);
  return RNS8_SUCCESS;
}

rns8_status stage_exact_cpu_lift(const rns8_matrix& matrix, ExactCpuLift& lift) {
  if (lift.staged_count == 0) return RNS8_SUCCESS;
  if (lift.staged_count > lift.suffix.max_size()) return RNS8_RANGE_ERROR;
  lift.suffix.resize(lift.staged_count);
  std::vector<int8_t> cell_residues(lift.source_prefix);
  const bool signed_value = matrix.desc.semantics == RNS8_EXACT_WIDE_SIGNED;
  const boost::multiprecision::cpp_int threshold = (lift.product + 1) / 2;
  for (std::size_t cell = 0; cell < lift.cells; ++cell) {
    for (uint32_t p = 0; p < lift.source_prefix; ++p) {
      const int value = matrix.residues[lift.cells * p + cell];
      const int modulus = rns8::detail::kDefaultModuli[p];
      if (value < -(modulus / 2) || value > (modulus - 1) / 2) return RNS8_INVALID_ARGUMENT;
      cell_residues[p] = static_cast<int8_t>(value);
    }
    auto value = rns8::detail::reconstruct_canonical(cell_residues, lift.source_prefix);
    if (signed_value && value >= threshold) value -= lift.product;
    if (boost::multiprecision::abs(value) > matrix.exact_max_magnitude) return RNS8_RANGE_ERROR;
    for (uint32_t p = lift.source_prefix; p < lift.target_prefix; ++p) {
      lift.suffix[lift.cells * (p - lift.source_prefix) + cell] =
          rns8::detail::centered_residue(value, rns8::detail::kDefaultModuli[p]);
    }
  }
  return RNS8_SUCCESS;
}

void commit_exact_cpu_lift(rns8_matrix& matrix, const ExactCpuLift& lift) noexcept {
  if (lift.staged_count == 0) return;
  std::copy(lift.suffix.begin(), lift.suffix.end(), matrix.residues.begin() + lift.source_count);
  matrix.exact_range_prefix = lift.target_prefix;
}

bool add_staged_payload(std::size_t bytes, uint64_t& total) {
  if (bytes > UINT64_MAX - total) return false;
  total += bytes;
  return true;
}

}  // namespace

rns8_status rns8_lift_exact_wide_cpu(rns8_context* ctx, rns8_matrix* matrix, uint32_t target_prefix,
                                     uint64_t max_staged_residue_bytes) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !matrix) return RNS8_INVALID_ARGUMENT;
    ExactCpuLift lift;
    auto status = prepare_exact_cpu_lift(*ctx, *matrix, target_prefix, lift);
    if (status != RNS8_SUCCESS) return status;
    if (lift.staged_count > max_staged_residue_bytes) return RNS8_WORKSPACE_TOO_SMALL;
    status = stage_exact_cpu_lift(*matrix, lift);
    if (status != RNS8_SUCCESS) return status;
    commit_exact_cpu_lift(*matrix, lift);
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_gemm_exact_wide_cpu_auto(rns8_context* ctx, const rns8_plan* plan, rns8_matrix* A,
                                          rns8_matrix* B, rns8_matrix* C, rns8_workspace* workspace,
                                          uint64_t max_staged_residue_bytes, uint32_t* out_selected_prefix) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace || !out_selected_prefix || A == C || B == C) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != plan->backend) return RNS8_INVALID_ARGUMENT;
    if (plan->backend != RNS8_BACKEND_CPU_REFERENCE) return RNS8_UNSUPPORTED_BACKEND;
    if (!rns8::detail::exact_wide_semantics(plan->desc.semantics)) return RNS8_INVALID_ARGUMENT;
    auto status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (status != RNS8_SUCCESS) return status;
    if (!matrix_descriptor_matches(*A, plan->desc.semantics, RNS8_BOUND_NONE, plan->desc.m, plan->desc.k,
                                   plan->prefix, plan->desc.tile_m, plan->desc.tile_n) ||
        !matrix_descriptor_matches(*B, plan->desc.semantics, RNS8_BOUND_NONE, plan->desc.k, plan->desc.n,
                                   plan->prefix, plan->desc.tile_m, plan->desc.tile_n) ||
        !matrix_descriptor_matches(*C, plan->desc.semantics, RNS8_BOUND_NONE, plan->desc.m, plan->desc.n,
                                   plan->prefix, plan->desc.tile_m, plan->desc.tile_n)) {
      return RNS8_INVALID_ARGUMENT;
    }
    ExactCpuLift a_lift, b_lift;
    status = prepare_exact_cpu_lift(*ctx, *A, plan->prefix, a_lift);
    if (status != RNS8_SUCCESS) return status;
    if (A != B) {
      status = prepare_exact_cpu_lift(*ctx, *B, plan->prefix, b_lift);
      if (status != RNS8_SUCCESS) return status;
    }
    std::size_t c_cells = 0;
    status = exact_cpu_storage(*ctx, *C, plan->prefix, false, c_cells);
    if (status != RNS8_SUCCESS) return status;

    boost::multiprecision::cpp_int output_bound;
    rns8_exact_matrix_ranges output_ranges;
    rns8::detail::exact_gemm_bound(*plan, *A, *B, output_bound, &output_ranges);
    const uint32_t ceiling = std::min({plan->desc.max_prefix, A->prefix, B->prefix, C->prefix});
    uint32_t selected = plan->prefix;
    for (; selected <= ceiling; ++selected) {
      if (rns8::detail::exact_range_fits(plan->desc.semantics, output_bound,
                                         rns8::detail::modulus_product(selected)))
        break;
    }
    if (selected > ceiling) return RNS8_RANGE_ERROR;
    a_lift.select(selected);
    if (A != B) b_lift.select(selected);
    // Each product fits its already validated matrix capacity; the combined
    // resident transaction budget is checked independently before allocation.
    const std::size_t output_count = c_cells * selected;
    uint64_t staged_bytes = 0;
    if (!add_staged_payload(a_lift.staged_count, staged_bytes) ||
        (A != B && !add_staged_payload(b_lift.staged_count, staged_bytes)) ||
        !add_staged_payload(output_count, staged_bytes))
      return RNS8_RANGE_ERROR;
    if (staged_bytes > max_staged_residue_bytes) return RNS8_WORKSPACE_TOO_SMALL;
    status = stage_exact_cpu_lift(*A, a_lift);
    if (status != RNS8_SUCCESS) return status;
    if (A != B) {
      status = stage_exact_cpu_lift(*B, b_lift);
      if (status != RNS8_SUCCESS) return status;
    }
    std::vector<int8_t> output;
    if (output_count > output.max_size()) return RNS8_RANGE_ERROR;
    output.resize(output_count);
    const auto& b_stage = A == B ? a_lift : b_lift;
    for (uint32_t p = 0; p < selected; ++p) {
      // Serial execution keeps row-accumulator allocations inside guard_api,
      // even in libraries whose ordinary CPU GEMM enables OpenMP.
      rns8::detail::ring_gemm_modulus(a_lift.plane(*A, p), b_stage.plane(*B, p), output.data() + c_cells * p,
                                      plan->desc.m, plan->desc.n, plan->desc.k, A->desc.cols, B->desc.cols,
                                      C->desc.cols, rns8::detail::kDefaultModuli[p], false);
    }
    const uint64_t output_version = gemm_output_source_version(*A, *B);
    // All allocation, validation and arithmetic precede the first resident
    // write. No staging buffer is swapped into a matrix's storage allocation.
    commit_exact_cpu_lift(*A, a_lift);
    if (A != B) commit_exact_cpu_lift(*B, b_lift);
    std::copy(output.begin(), output.end(), C->residues.begin());
    mark_output_host_residues_current(*C);
    C->source_version = output_version;
    C->exact_max_magnitude.swap(output_bound);
    C->exact_axis_ranges.rows.swap(output_ranges.rows);
    C->exact_axis_ranges.cols.swap(output_ranges.cols);
    C->exact_range_prefix = selected;
    *out_selected_prefix = selected;
    return RNS8_SUCCESS;
  });
}
