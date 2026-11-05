#include "core/gram_contract.hpp"
#include "backend_common/gram_plan.hpp"
using namespace rns8::detail;
using namespace rns8::detail::api;
rns8_status rns8_gram_residues(rns8_context* ctx, const rns8_matrix* x, rns8_matrix* c,
                               rns8_gram_orientation orientation, rns8_gram_method method,
                               uint64_t budget) {
  return guard_api([&]() -> rns8_status {
    GramProblem p;
    cpp_int bound;
    const auto admitted = validate_gram_contract(ctx, x, c, orientation, method, p, bound);
    if (admitted != RNS8_SUCCESS)
      return admitted;
    const bool hip = x->backend == RNS8_BACKEND_HIP_DIRECT;
    const bool finite = uses_finite_storage(x->desc.semantics);
    const auto support = query_gram_support(p, budget, hip, bool(uint32_t(method) & 1));
    if (support.status != RNS8_SUCCESS)
      return support.status;
    bool started = false;
    size_t token = 0;
    const auto reserved = begin_synchronous_lease(*ctx, *x, *x, *c, nullptr, token);
    if (reserved != RNS8_SUCCESS)
      return reserved;
    SynchronousLease write(*ctx, token);
    write.track_write(started);
    return write.finish([&]() -> rns8_status {
      const auto dispatched = synchronous_dispatch_checkpoint(*ctx, "gram", &started);
      if (dispatched != RNS8_SUCCESS)
        return dispatched;
      if (hip) {
        const auto status = hip_direct_gram_device(ctx->device_id, x->hip_residues, c->hip_residues,
                                                   p, x->matrix_instance_id, x->content_generation,
                                                   bool(uint32_t(method) & 1), budget, started,
                                                   bool(uint32_t(method) & 2));
        if (status != RNS8_SUCCESS)
          return status;
        mark_output_device_residues_current(*c);
      } else {
        const auto jobs = gram_tiles(support);
        started = true;
        gram_residues_cpu(x->residues.data(), c->residues.data(), p, jobs, kDefaultModuli,
                          bool(uint32_t(method) & 2));
        mark_output_host_residues_current(*c);
      }
      c->finite_modulus = finite ? x->finite_modulus : 0;
      c->exact_axis_ranges.rows.clear();
      c->exact_axis_ranges.cols.clear();
      if (!finite) {
        c->materialized_rns_prefix = c->prefix;
        c->exact_max_magnitude.swap(bound);
        c->exact_range_prefix = c->prefix;
      }
      return RNS8_SUCCESS;
    }());
  });
}
