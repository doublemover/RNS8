#include "core/api_internal.hpp"
#include "backend_common/operation_support.hpp"
#include "core/hip_resources.hpp"
#include "backend_common/finite_verify.hpp"
#include "backend_hip_direct/finite_verify_backend.hpp"
#include <cmath>

rns8_status rns8_verify_finite_product(rns8_context* ctx, const rns8_matrix* A,
    const rns8_matrix* B, const rns8_matrix* C,
    const rns8_finite_verify_options* options, rns8_finite_verify_report* report) {
  using namespace rns8::detail;
  using namespace rns8::detail::api;
  return guard_api([&]() -> rns8_status {
    if (!ctx || !A || !B || !C || !options || !report ||
        options->struct_size != sizeof(*options) || options->abi_version != RNS8_ABI_VERSION ||
        report->struct_size != sizeof(*report) || report->abi_version != RNS8_ABI_VERSION ||
        options->repetition_count < 1 || options->repetition_count > 1048576 ||
        options->sampler != RNS8_VERIFY_SPLITMIX64_BINARY_V1 ||
        (options->flags & ~RNS8_VERIFY_CHECK_GENERATIONS)) return RNS8_INVALID_ARGUMENT;
    if (ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    const auto backend = A->backend;
    if ((backend != RNS8_BACKEND_CPU_REFERENCE && backend != RNS8_BACKEND_HIP_DIRECT) ||
        !uses_finite_storage(A->desc.semantics)) return RNS8_UNSUPPORTED_BACKEND;
    if (!context_accepts_backend(*ctx, backend)) return RNS8_INVALID_ARGUMENT;
    const bool gpu = backend == RNS8_BACKEND_HIP_DIRECT;
    const uint32_t q = A->finite_modulus;
    if (!valid_finite_modulus_for_semantics(A->desc.semantics, static_cast<uint16_t>(q)))
      return RNS8_INVALID_ARGUMENT;
    const rns8_matrix* matrices[] = {A,B,C};
    rns8_finite_verify_report completed{};
    completed.struct_size = sizeof(completed); completed.abi_version = RNS8_ABI_VERSION;
    completed.semantics = A->desc.semantics; completed.modulus = q;
    completed.sampler = options->sampler; completed.random_seed = options->random_seed;
    for (unsigned i = 0; i < 3; ++i) {
      const auto& x = *matrices[i];
      if (x.in_flight_writer) return RNS8_BUSY;
      if (x.generation_exhausted || x.backend != backend ||
          !matrix_backend_compatible_with_plan(*ctx, x, backend) ||
          (gpu && x.hip_device_id != ctx->device_id) ||
          x.desc.semantics != A->desc.semantics || x.finite_modulus != q ||
          x.desc.rows <= 0 || x.desc.cols <= 0 ||
          !matrix_descriptor_matches(x, A->desc.semantics, RNS8_BOUND_NONE,
              x.desc.rows, x.desc.cols, 0, 0, 0) ||
          !finite_matrix_storage_matches(x, backend, x.desc.rows, x.desc.cols) ||
          !rns_residue_state_current_for_backend(x, backend) ||
          ((options->flags & RNS8_VERIFY_CHECK_GENERATIONS) &&
           options->expected_generations[i] != x.content_generation)) return RNS8_INVALID_ARGUMENT;
      completed.matrix_instance_ids[i] = x.matrix_instance_id;
      completed.generations[i] = x.content_generation;
    }
    if (A->desc.cols != B->desc.rows || A->desc.rows != C->desc.rows ||
        B->desc.cols != C->desc.cols) return RNS8_INVALID_ARGUMENT;
    const uint64_t m = uint64_t(C->desc.rows), n = uint64_t(C->desc.cols), k = uint64_t(A->desc.cols);
    const auto support=query_finite_verify_support(C->desc.rows,C->desc.cols,A->desc.cols,q,
        gpu,options->max_workspace_bytes);
    if(support.status!=RNS8_SUCCESS) return support.status;
    completed.workspace_bytes=support.scratch_bytes;
    std::vector<uint8_t> challenge(static_cast<size_t>(n));
    std::vector<uint32_t> projection;
    hip_direct_device_buffer dr, dbr, flag;
    if (gpu) {
      auto status = dr.allocate(ctx->device_id, static_cast<size_t>(n));
      if (status != RNS8_SUCCESS) return status;
      status = dbr.allocate(ctx->device_id, static_cast<size_t>(k*4));
      if (status != RNS8_SUCCESS) return status;
      status = flag.allocate(ctx->device_id, 4);
      if (status != RNS8_SUCCESS) return status;
    } else projection.resize(static_cast<size_t>(k));
    uint64_t state = options->random_seed;
    for (uint32_t trial = 0; trial < options->repetition_count; ++trial) {
      for (auto& bit : challenge) bit = finite_verify::next_bit(state);
      uint32_t flags = 0;
      if (gpu) {
        auto status = hip_direct_copy_host_to_device(ctx->device_id, dr.get(), challenge.data(), challenge.size());
        if (status != RNS8_SUCCESS) return status;
        status = hip_finite_verify_trial(ctx->device_id, A->hip_residues, B->hip_residues,
            C->hip_residues, m, n, k, q, dr.get(), dbr.get(), flag.get(), flags);
        if (status != RNS8_SUCCESS) return status;
      } else flags = finite_verify::cpu_trial(A->residues.data(), B->residues.data(), C->residues.data(),
          m,n,k,q,challenge.data(),projection.data());
      if (flags & 2) return RNS8_INVALID_ARGUMENT;
      completed.completed_trials = trial+1;
      completed.ideal_failure_bound_exponent = trial+1;
      completed.ideal_failure_probability = std::ldexp(1.0, -static_cast<int>(std::min(trial+1, uint32_t(1022))));
      if (flags & 1) { *report = completed; return RNS8_VERIFICATION_FAILED; }
    }
    completed.passed = 1;
    *report = completed;
    return RNS8_SUCCESS;
  });
}
