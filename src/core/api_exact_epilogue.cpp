#include "backend_common/address_domain.hpp"
#include "backend_common/residue_epilogue.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include "core/api_internal.hpp"
#include "backend_common/complex_residue_plan.hpp"

using namespace rns8::detail::api;
#include "core/api_complex_residue.inc"
#include "core/api_block_sparse.inc"
#include "core/api_encoded_b.inc"
#include "core/api_int4.inc"

rns8_status rns8_gemm_complex_exact_signed(
    rns8_context* ctx, const rns8_plan* plan,
    const rns8_matrix* Ar, const rns8_matrix* Ai,
    const rns8_matrix* Br, const rns8_matrix* Bi,
    rns8_matrix* Cr, rns8_matrix* Ci, rns8_complex_product_method method,
    uint64_t max_scratch_bytes, rns8_workspace* workspace) {
  using rns8::detail::ComplexProductMethod;
  if (method != RNS8_COMPLEX_THREE_REAL_PRODUCTS && method != RNS8_COMPLEX_FOUR_REAL_PRODUCTS)
    return RNS8_INVALID_ARGUMENT;
  return complex_exact_signed_gemm(ctx, plan, {Ar, Ai}, {Br, Bi}, {Cr, Ci},
      method == RNS8_COMPLEX_THREE_REAL_PRODUCTS ? ComplexProductMethod::three_real
                                               : ComplexProductMethod::four_real,
      max_scratch_bytes, workspace);
}

rns8_status rns8_gemm_narrow_exact(
    rns8_context* ctx, const rns8_plan* plan,
    const rns8_matrix* A, const rns8_matrix* B, rns8_matrix* C,
    rns8_narrow_product_method method, uint64_t max_scratch_bytes,
    rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if (!ctx || !plan || !A || !B || !C || !workspace || A==C || B==C ||
        (uint32_t(method)>3))
      return RNS8_INVALID_ARGUMENT;
    if (plan->backend!=RNS8_BACKEND_HIP_DIRECT || !exact_wide_semantics(plan->desc.semantics) ||
        (!(uint32_t(method)&1) && plan->desc.semantics!=RNS8_EXACT_WIDE_SIGNED))
      return RNS8_UNSUPPORTED_BACKEND;
    rns8_operation_query query{};query.struct_size=sizeof(query);query.abi_version=RNS8_ABI_VERSION;
    query.operation=RNS8_OP_NARROW_NATIVE;query.execution_modes=RNS8_EXEC_SYNC;query.problem=plan->desc;query.algorithm=method;
    query.proof_flags=RNS8_QUERY_NATIVE_RETAINED|RNS8_QUERY_NARROW_INPUT_PROOF;query.lda=A->desc.cols;query.ldb=B->desc.cols;query.ldc=C->desc.cols;query.scratch_budget=max_scratch_bytes;
    const auto provider=operation_support(*ctx,query,plan->backend);if(provider.status!=RNS8_SUCCESS)return provider.status;
    auto status=validate_plan_context_workspace(*ctx,*plan,*workspace);
    if (status!=RNS8_SUCCESS) return status;
    status=validate_rns_gemm_operands(*ctx,*plan,*A,*B,*C,true);
    if (status!=RNS8_SUCCESS) return status;
    if (C->generation_exhausted) return RNS8_RANGE_ERROR;
    const bool unsigned_inputs=plan->desc.semantics==RNS8_EXACT_WIDE_UNSIGNED;
    const auto support=query_narrow_rns_support(plan->desc.m,plan->desc.n,plan->desc.k,
        A->desc.cols,B->desc.cols,plan->prefix,bool(uint32_t(method)&1),max_scratch_bytes,method>=RNS8_NARROW_SCALAR_FUSED_RNS);
    if (support.status!=RNS8_SUCCESS) return support.status;
    const auto native=[&](const rns8_matrix& input) -> const void* {
      if (!(input.desc.flags&RNS8_MATRIX_RETAIN_NATIVE_INPUT) || !input.device_native_current ||
          input.generation_exhausted || input.hip_device_id!=ctx->device_id) return nullptr;
      const uint64_t bytes=uint64_t(input.desc.rows)*uint64_t(input.desc.cols)*sizeof(uint64_t);
      return unsigned_inputs ? (input.hip_native_u64_bytes==bytes ? input.hip_native_u64 : nullptr)
                             : (input.hip_native_i64_bytes==bytes ? input.hip_native_i64 : nullptr);
    };
    const void* a=native(*A);
    const void* b=native(*B);
    if (!a || !b) return RNS8_INVALID_ARGUMENT;
    cpp_int bound;
    rns8_exact_matrix_ranges ranges;
    status=exact_gemm_range(*plan,*A,*B,bound,&ranges);
    if (status!=RNS8_SUCCESS) return status;
    bool commit_started=false;size_t lease_token=0;
    status=begin_synchronous_lease(*ctx,*A,*B,*C,*workspace,lease_token);if(status!=RNS8_SUCCESS)return status;
    struct LeaseGuard {rns8_context& ctx;size_t token;bool& written;bool active=true;
      ~LeaseGuard(){if(active)finish_synchronous_lease(ctx,token,RNS8_INTERNAL_ERROR,written);}} lease{*ctx,lease_token,commit_started};
    status=hip_direct_gemm_narrow_rns_device(ctx->device_id,a,b,C->hip_residues,
        plan->desc.m,plan->desc.n,plan->desc.k,A->desc.cols,B->desc.cols,plan->prefix,
        unsigned_inputs,bool(uint32_t(method)&1),max_scratch_bytes,commit_started,method>=RNS8_NARROW_SCALAR_FUSED_RNS);
    finish_synchronous_lease(*ctx,lease_token,status,commit_started);lease.active=false;
    if (status!=RNS8_SUCCESS) return status;
    mark_output_device_residues_current(*C, plan);
    commit_exact_range(*C,*plan,bound,&ranges);
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_gemm_adaptive_exact(
    rns8_context* ctx, const rns8_plan* plan,
    const rns8_matrix* A, const rns8_matrix* B, rns8_matrix* C,
    rns8_adaptive_bound_method method, uint32_t panel_k, uint32_t group,
    uint32_t compact_workers, uint64_t max_scratch_bytes, rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if (!ctx || !plan || !A || !B || !C || !workspace || A == C || B == C ||
        !exact_wide_semantics(plan->desc.semantics) ||
        (method != RNS8_ADAPTIVE_TILE_MAX && method != RNS8_ADAPTIVE_AXIS_L1 &&
         method != RNS8_ADAPTIVE_PANEL_L1) || (group != 1 && group != 2 && group != 4) ||
        compact_workers > 65535 ||
        (method == RNS8_ADAPTIVE_PANEL_L1 && (!panel_k || panel_k > 65536)))
      return RNS8_INVALID_ARGUMENT;
    if (plan->backend != RNS8_BACKEND_HIP_DIRECT) return RNS8_UNSUPPORTED_BACKEND;
    auto status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (status != RNS8_SUCCESS) return status;
    status = validate_rns_gemm_operands(*ctx, *plan, *A, *B, *C, true);
    if (status != RNS8_SUCCESS) return status;
    if (C->generation_exhausted) return RNS8_RANGE_ERROR;
    const bool unsigned_inputs = plan->desc.semantics == RNS8_EXACT_WIDE_UNSIGNED;
    const auto native = [&](const rns8_matrix& matrix) -> const void* {
      if (!(matrix.desc.flags & RNS8_MATRIX_RETAIN_NATIVE_INPUT) ||
          !matrix.device_native_current || matrix.generation_exhausted ||
          matrix.hip_device_id != ctx->device_id) return nullptr;
      const uint64_t bytes = uint64_t(matrix.desc.rows) * uint64_t(matrix.desc.cols) * sizeof(uint64_t);
      if (unsigned_inputs)
        return matrix.hip_native_u64_bytes == bytes ? matrix.hip_native_u64 : nullptr;
      return matrix.hip_native_i64_bytes == bytes ? matrix.hip_native_i64 : nullptr;
    };
    const void* a = native(*A);
    const void* b = native(*B);
    if (!a || !b) return RNS8_INVALID_ARGUMENT;
    cpp_int bound;
    rns8_exact_matrix_ranges ranges;
    status = exact_gemm_range(*plan, *A, *B, bound, &ranges);
    if (status != RNS8_SUCCESS) return status;
    // Synchronous APIs require external thread exclusion. Existing operand and
    // workspace validation exclude outstanding async readers/writers. All proof
    // allocation precedes the staged device operation; failed preflight preserves C.
    bool commit_started = false;
    struct CommitGuard {
      rns8_matrix& output;
      bool& started;
      bool published = false;
      ~CommitGuard() { if (started && !published) invalidate_output_currentness(output); }
    } write{*C, commit_started};
    const auto discovery = method == RNS8_ADAPTIVE_TILE_MAX ? NativeBoundMethod::tile_max
        : method == RNS8_ADAPTIVE_AXIS_L1 ? NativeBoundMethod::axis_l1 : NativeBoundMethod::panel_l1;
    status = hip_direct_gemm_adaptive_native_device(ctx->device_id, a, b,
        static_cast<int8_t*>(C->hip_residues), plan->desc.m, plan->desc.n, plan->desc.k,
        A->desc.cols, B->desc.cols, C->desc.cols, unsigned_inputs, plan->prefix, group,
        max_scratch_bytes, commit_started, discovery, panel_k, compact_workers);
    if (status != RNS8_SUCCESS) return status;
    mark_output_device_residues_current(*C, plan);
    commit_exact_range(*C, *plan, bound, &ranges);
    write.published = true;
    return RNS8_SUCCESS;
  });
}

namespace rns8::detail::api {
rns8_status validate_exact_axpby_operands(
    const rns8_context &ctx, const rns8_plan &plan, const rns8_matrix &A,
    const rns8_matrix &B, const rns8_matrix &D, const rns8_matrix &C,
    int64_t alpha, int64_t beta, bool require_current_inputs, bool allow_pending_input_writers) {
  if (!exact_wide_semantics(plan.desc.semantics) || &D == &C ||
      (plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED &&
       (alpha < 0 || beta < 0)))
    return RNS8_INVALID_ARGUMENT;
  if (plan.backend != RNS8_BACKEND_CPU_REFERENCE &&
      plan.backend != RNS8_BACKEND_HIP_DIRECT)
    return RNS8_UNSUPPORTED_BACKEND;
  auto status =
      validate_rns_gemm_operands(ctx, plan, A, B, C, require_current_inputs, allow_pending_input_writers);
  if (status != RNS8_SUCCESS)
    return status;
  if (D.in_flight_writer && !allow_pending_input_writers)
    return RNS8_BUSY;
  const auto prefix = rns_storage_prefix_for_plan(plan);
  if (!matrix_backend_compatible_with_plan(ctx, D, plan.backend, allow_pending_input_writers) ||
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
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT)
    return query_axpby_support(plan.desc.m,plan.desc.n,plan.desc.k,
        A.desc.cols,B.desc.cols,D.desc.cols,C.desc.cols,plan.prefix).status;
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
      mark_output_device_residues_current(*C, plan);
    } else {
      const auto m = plan->desc.m, n = plan->desc.n, k = plan->desc.k;
      for (uint32_t p = 0; p < plan->prefix; ++p) {
        const int64_t modulus = kDefaultModuli[p];
        const auto coefficients=lower_residue_axpby(alpha,beta,uint32_t(modulus));
        for (int64_t row = 0; row < m; ++row)
          for (int64_t col = 0; col < n; ++col) {
            int64_t sum = 0;
            if(coefficients.alpha!=0)for (int64_t q = 0; q < k; ++q)
              sum =
                  (sum + int64_t{A->residues[(int64_t{p} * m + row) * k + q]} *
                             B->residues[(int64_t{p} * k + q) * n + col]) %
                  modulus;
            const int8_t product=finite_u8::reduce_to_centered_i32(int32_t(sum),uint32_t(modulus),coefficients.reciprocal);
            const int8_t addend=coefficients.beta?D->residues[(int64_t{p}*m+row)*n+col]:int8_t{0};
            C->residues[(int64_t{p}*m+row)*n+col]=centered_residue_axpby_reduced(product,addend,coefficients);
          }
      }
      mark_output_host_residues_current(*C, plan);
    }
    commit_exact_range(*C, *plan, bound);
    write.committed = true;
    return RNS8_SUCCESS;
  });
}
