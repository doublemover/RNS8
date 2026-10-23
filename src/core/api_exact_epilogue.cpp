#include "backend_common/address_domain.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include "core/api_internal.hpp"

using namespace rns8::detail::api;
namespace rns8::detail::api {
rns8_status validate_exact_axpby_operands(
    const rns8_context &ctx, const rns8_plan &plan, const rns8_matrix &A,
    const rns8_matrix &B, const rns8_matrix &D, const rns8_matrix &C,
    int64_t alpha, int64_t beta, bool require_current_inputs) {
  if (!exact_wide_semantics(plan.desc.semantics) || &D == &C ||
      (plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED &&
       (alpha < 0 || beta < 0)))
    return RNS8_INVALID_ARGUMENT;
  if (plan.backend != RNS8_BACKEND_CPU_REFERENCE &&
      plan.backend != RNS8_BACKEND_HIP_DIRECT)
    return RNS8_UNSUPPORTED_BACKEND;
  auto status =
      validate_rns_gemm_operands(ctx, plan, A, B, C, require_current_inputs);
  if (status != RNS8_SUCCESS)
    return status;
  if (D.in_flight_writer)
    return RNS8_BUSY;
  const auto prefix = rns_storage_prefix_for_plan(plan);
  if (!matrix_backend_compatible_with_plan(ctx, D, plan.backend) ||
      !matrix_descriptor_matches(D, plan.desc.semantics,
                                 storage_bound_kind_for_plan(plan), plan.desc.m,
                                 plan.desc.n, prefix, plan.desc.tile_m,
                                 plan.desc.tile_n) ||
      !rns_matrix_storage_matches(D, plan.backend, plan.desc.m, plan.desc.n,
                                  prefix) ||
      (require_current_inputs &&
       !rns_residue_state_current_for_backend(D, plan.backend)) ||
      (plan.backend == RNS8_BACKEND_HIP_DIRECT &&
       D.hip_device_id != ctx.device_id))
    return RNS8_INVALID_ARGUMENT;
  if (C.generation_exhausted)
    return RNS8_RANGE_ERROR;
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT &&
      !qualified_i32_gemm_address(plan.desc.m, plan.desc.n, plan.desc.k,
                                  A.desc.cols, B.desc.cols, C.desc.cols,
                                  plan.prefix))
    return RNS8_UNSUPPORTED_BACKEND;
  return RNS8_SUCCESS;
}
} // namespace rns8::detail::api

rns8_status rns8_gemm_exact_axpby(rns8_context *ctx, const rns8_plan *plan,
                                  const rns8_matrix *A, const rns8_matrix *B,
                                  const rns8_matrix *D, rns8_matrix *C,
                                  int64_t alpha, int64_t beta,
                                  rns8_workspace *workspace) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if (!ctx || !plan || !A || !B || !D || !C || !workspace || A == C ||
        B == C || D == C)
      return RNS8_INVALID_ARGUMENT;
    auto status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (status != RNS8_SUCCESS)
      return status;
    status =
        validate_exact_axpby_operands(*ctx, *plan, *A, *B, *D, *C, alpha, beta);
    if (status != RNS8_SUCCESS)
      return status;
    cpp_int bound;
    status = exact_axpby_range(*plan, *A, *B, *D, alpha, beta, bound);
    if (status != RNS8_SUCCESS)
      return status;
    // All allocation-prone proof work finishes before entering the write
    // region.
    MatrixWriteGuard write(*C);
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      status = hip_direct_gemm_rns_axpby_device(
          ctx->device_id, A->hip_residues, B->hip_residues, D->hip_residues,
          C->hip_residues, plan->desc.m, plan->desc.n, plan->desc.k,
          A->desc.cols, B->desc.cols, D->desc.cols, C->desc.cols, plan->prefix,
          alpha, beta);
      if (status != RNS8_SUCCESS)
        return status;
      mark_output_device_residues_current(*C);
    } else {
      const auto m = plan->desc.m, n = plan->desc.n, k = plan->desc.k;
      for (uint32_t p = 0; p < plan->prefix; ++p) {
        const int64_t modulus = kDefaultModuli[p];
        const int64_t am = alpha % modulus, bm = beta % modulus;
        for (int64_t row = 0; row < m; ++row)
          for (int64_t col = 0; col < n; ++col) {
            int64_t sum = 0;
            for (int64_t q = 0; q < k; ++q)
              sum =
                  (sum + int64_t{A->residues[(int64_t{p} * m + row) * k + q]} *
                             B->residues[(int64_t{p} * k + q) * n + col]) %
                  modulus;
            int64_t result =
                (am * sum +
                 bm * D->residues[(int64_t{p} * m + row) * n + col]) %
                modulus;
            if (result < 0)
              result += modulus;
            if (result >= (modulus + 1) / 2)
              result -= modulus;
            C->residues[(int64_t{p} * m + row) * n + col] =
                static_cast<int8_t>(result);
          }
      }
      mark_output_host_residues_current(*C);
    }
    commit_exact_range(*C, *plan, bound);
    write.committed = true;
    return RNS8_SUCCESS;
  });
}
