#include "core/gram_contract.hpp"
#include <mutex>
using namespace rns8::detail;
using namespace rns8::detail::api;
struct rns8_prepared_gram {
  rns8_context* ctx=nullptr;
  const rns8_matrix* x=nullptr;
  rns8_matrix* c=nullptr;
  GramProblem problem;
  cpp_int bound;
  std::vector<GramTile> jobs;
  void* device_jobs=nullptr;
  uint64_t device_job_bytes=0;
  bool wmma=false,jobs_ready=false,active=false,commit_started=false,quarantined=false;
  std::mutex mutex;
};
namespace {
void release_gram_reservation(rns8_prepared_gram& owner) noexcept {
  if(!owner.active) return;
  --owner.x->in_flight_readers;owner.c->in_flight_writer=false;owner.active=false;
}
void invalidate_gram_commit(rns8_prepared_gram& owner) noexcept {
  if(owner.active && owner.commit_started) {
    clear_residue_current(*owner.c);clear_byte_limb_current(*owner.c);clear_native_current(*owner.c);
  }
}
struct GramReservation {
  rns8_prepared_gram& owner;bool completed=false;
  ~GramReservation() noexcept {
    if(completed) return;
    invalidate_gram_commit(owner);
    if(owner.x->backend==RNS8_BACKEND_HIP_DIRECT) {
      try {
        if(hip_direct_synchronize(owner.ctx->device_id)!=RNS8_SUCCESS) {
          owner.quarantined=true;owner.ctx->hip_failure_poisoned=true;return;
        }
      } catch(...) {owner.quarantined=true;owner.ctx->hip_failure_poisoned=true;return;}
    }
    release_gram_reservation(owner);
  }
  void finish() noexcept {release_gram_reservation(owner);completed=true;}
};
}
rns8_status rns8_prepare_gram(rns8_context* ctx,const rns8_matrix* x,rns8_matrix* c,
    rns8_gram_orientation orientation,rns8_gram_method method,uint64_t budget,rns8_prepared_gram** out) {
  return guard_api([&]() -> rns8_status {
    if(!out) return RNS8_INVALID_ARGUMENT;
    auto owner=std::make_unique<rns8_prepared_gram>();
    const auto status=validate_gram_contract(ctx,x,c,orientation,method,owner->problem,owner->bound);
    if(status!=RNS8_SUCCESS) return status;
    const bool hip=x->backend==RNS8_BACKEND_HIP_DIRECT;
    const auto support=query_gram_support(owner->problem,budget,hip,method==RNS8_GRAM_WMMA);
    if(support.status!=RNS8_SUCCESS) return support.status;
    if(ctx->prepared_references==UINT64_MAX || x->prepared_references==UINT64_MAX || c->prepared_references==UINT64_MAX)
      return RNS8_RANGE_ERROR;
    owner->ctx=ctx;owner->x=x;owner->c=c;owner->wmma=method==RNS8_GRAM_WMMA;
    owner->jobs=gram_tiles(support);
    if(hip) {
      owner->device_job_bytes=support.job_bytes+(owner->wmma?4:0);
      const auto allocated=hip_direct_allocate(ctx->device_id,static_cast<size_t>(owner->device_job_bytes),&owner->device_jobs);
      if(allocated!=RNS8_SUCCESS) return allocated;
    }
    // Allocation is last; upload/probe happens under retained execution ownership.
    ++ctx->prepared_references;++x->prepared_references;++c->prepared_references;
    *out=owner.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_refresh_prepared_gram(rns8_prepared_gram* owner) {
  return guard_api([&]() -> rns8_status {
    if(!owner) return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(owner->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    if(owner->quarantined || owner->ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    GramProblem replacement;cpp_int bound;
    const auto status=validate_gram_contract(owner->ctx,owner->x,owner->c,
        static_cast<rns8_gram_orientation>(owner->problem.orientation),owner->wmma?RNS8_GRAM_WMMA:RNS8_GRAM_SCALAR,
        replacement,bound);
    if(status!=RNS8_SUCCESS) return status;
    if(bound>owner->bound || replacement.modulus!=owner->problem.modulus) return RNS8_RANGE_ERROR;
    // Public matrix descriptors are immutable; refresh changes only generations.
    owner->problem.source_instance=replacement.source_instance;
    owner->problem.source_generation=replacement.source_generation;
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_execute_prepared_gram(rns8_prepared_gram* owner) {
  return guard_api([&]() -> rns8_status {
    if(!owner) return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(owner->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    if(owner->quarantined || owner->ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    const auto* x=owner->x;auto* c=owner->c;
    if(x->in_flight_writer || c->in_flight_writer || c->in_flight_readers) return RNS8_BUSY;
    if(x->generation_exhausted || c->generation_exhausted || x->in_flight_readers==UINT64_MAX) return RNS8_RANGE_ERROR;
    if(x->matrix_instance_id!=owner->problem.source_instance || x->content_generation!=owner->problem.source_generation ||
       !rns_residue_state_current_for_backend(*x,x->backend)) return RNS8_INVALID_ARGUMENT;
    cpp_int bound=owner->bound;
    ++x->in_flight_readers;c->in_flight_writer=true;owner->active=true;owner->commit_started=false;
    GramReservation reservation{*owner};
    if(x->backend==RNS8_BACKEND_HIP_DIRECT) {
      const auto status=hip_direct_gram_prepared_device(owner->ctx->device_id,x->hip_residues,c->hip_residues,owner->problem,
          owner->jobs.data(),owner->device_jobs,owner->device_job_bytes,owner->wmma,owner->jobs_ready,owner->commit_started);
      if(status!=RNS8_SUCCESS) return status;
      mark_output_device_residues_current(*c);
    } else {
      owner->commit_started=true;
      gram_residues_cpu(x->residues.data(),c->residues.data(),owner->problem,owner->jobs,kDefaultModuli);
      mark_output_host_residues_current(*c);
    }
    c->finite_modulus=owner->problem.modulus;
    c->exact_axis_ranges.rows.clear();c->exact_axis_ranges.cols.clear();
    if(!uses_finite_storage(c->desc.semantics)) {c->exact_max_magnitude.swap(bound);c->exact_range_prefix=c->prefix;}
    reservation.finish();return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_prepared_gram(rns8_prepared_gram* owner) {
  return guard_api([&]() -> rns8_status {
    if(!owner) return RNS8_SUCCESS;
    std::unique_lock<std::mutex> lock(owner->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    if(owner->quarantined || owner->active) {
      if(owner->x->backend==RNS8_BACKEND_HIP_DIRECT && hip_direct_synchronize(owner->ctx->device_id)!=RNS8_SUCCESS)
        return RNS8_BACKEND_FAILURE;
      invalidate_gram_commit(*owner);release_gram_reservation(*owner);owner->quarantined=false;
    }
    if(owner->device_jobs) {
      const auto status=hip_direct_free(owner->ctx->device_id,owner->device_jobs);
      if(status!=RNS8_SUCCESS) return status;owner->device_jobs=nullptr;
    }
    --owner->ctx->prepared_references;--owner->x->prepared_references;--owner->c->prepared_references;
    lock.unlock();delete owner;return RNS8_SUCCESS;
  });
}
