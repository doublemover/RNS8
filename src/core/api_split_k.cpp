#include "core/api_internal.hpp"
#include "backend_common/split_k_plan.hpp"
#include <mutex>
using namespace rns8::detail;
using namespace rns8::detail::api;
struct rns8_split_k {
  rns8_context* ctx=nullptr;
  const rns8_matrix *a=nullptr,*b=nullptr;
  uint64_t a_instance=0,b_instance=0,a_generation=0,b_generation=0;
  cpp_int bound;
  rns8_split_k_resources resources{};
  void* scratch=nullptr;
  rns8_matrix* active_output=nullptr;
  bool quarantined=false,commit_started=false;
  std::mutex mutex;
};
namespace {
void release_split_readers(rns8_split_k& owner) noexcept {
  if(!owner.active_output) return;
  --owner.a->in_flight_readers;
  if(owner.b!=owner.a) --owner.b->in_flight_readers;
  owner.active_output->in_flight_writer=false;
  --owner.active_output->prepared_references;owner.active_output=nullptr;
}
void invalidate_split_output(rns8_split_k& owner) noexcept {
  if(owner.active_output && owner.commit_started) {
    clear_residue_current(*owner.active_output);clear_byte_limb_current(*owner.active_output);
    clear_native_current(*owner.active_output);
  }
}
struct SplitReservation {
  rns8_split_k& owner;bool finished=false;
  ~SplitReservation() noexcept {
    if(finished) return;
    invalidate_split_output(owner);
    try {
      if(hip_direct_synchronize(owner.ctx->device_id)==RNS8_SUCCESS) {
        release_split_readers(owner);return;
      }
    } catch(...) {}
    owner.quarantined=true;owner.ctx->hip_failure_poisoned=true;
  }
  void complete() noexcept {release_split_readers(owner);finished=true;}
};
bool split_input_matches(const rns8_context& ctx,const rns8_matrix& a) {
  return a.backend==RNS8_BACKEND_HIP_DIRECT && a.hip_device_id==ctx.device_id &&
      exact_wide_semantics(a.desc.semantics) && !a.generation_exhausted &&
      rns_residue_state_current_for_backend(a,a.backend) &&
      rns_matrix_storage_matches(a,a.backend,a.desc.rows,a.desc.cols,a.prefix);
}
}
rns8_status rns8_query_split_k(int64_t m,int64_t n,int64_t k,uint32_t prefix,uint32_t fragments,
    uint64_t budget,rns8_split_k_resources* out) {
  if(!out || !valid_abi(out->struct_size,out->abi_version,sizeof(*out))) return RNS8_INVALID_ARGUMENT;
  return query_split_k_resources(m,n,k,k,n,n,prefix,fragments,budget,*out);
}
rns8_status rns8_create_split_k(rns8_context* ctx,const rns8_matrix* a,const rns8_matrix* b,
    uint32_t fragments,uint64_t budget,rns8_split_k** out) {
  return guard_api([&]() -> rns8_status {
    if(!ctx || !a || !b || !out) return RNS8_INVALID_ARGUMENT;
    if(ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    if(ctx->backend!=RNS8_BACKEND_HIP_DIRECT || !hip_direct_compiled()) return RNS8_UNSUPPORTED_BACKEND;
    if(a->in_flight_writer || b->in_flight_writer) return RNS8_BUSY;
    if(!split_input_matches(*ctx,*a) || !split_input_matches(*ctx,*b) ||
       a->desc.semantics!=b->desc.semantics || a->prefix!=b->prefix || a->desc.cols!=b->desc.rows)
      return RNS8_INVALID_ARGUMENT;
    for(const auto* input:{a,b}) {
      if(input->exact_range_prefix<input->prefix || input->exact_max_magnitude<0 ||
         !exact_range_fits(input->desc.semantics,input->exact_max_magnitude,modulus_product(input->prefix)))
        return RNS8_RANGE_ERROR;
    }
    auto owner=std::make_unique<rns8_split_k>();
    owner->bound=cpp_int(a->desc.cols)*a->exact_max_magnitude*b->exact_max_magnitude;
    if(!exact_range_fits(a->desc.semantics,owner->bound,modulus_product(a->prefix))) return RNS8_RANGE_ERROR;
    const auto status=query_split_k_resources(a->desc.rows,b->desc.cols,a->desc.cols,
        a->desc.cols,b->desc.cols,b->desc.cols,a->prefix,fragments,budget,owner->resources);
    if(status!=RNS8_SUCCESS) return status;
    if(ctx->prepared_references==UINT64_MAX || a->prepared_references==UINT64_MAX || b->prepared_references==UINT64_MAX)
      return RNS8_RANGE_ERROR;
    owner->ctx=ctx;owner->a=a;owner->b=b;
    owner->a_instance=a->matrix_instance_id;owner->b_instance=b->matrix_instance_id;
    owner->a_generation=a->content_generation;owner->b_generation=b->content_generation;
    const auto allocated=hip_direct_allocate(ctx->device_id,static_cast<size_t>(owner->resources.scratch_bytes),&owner->scratch);
    if(allocated!=RNS8_SUCCESS) return allocated;
    // Nothing after allocation can throw; lifetime holds are unique per handle.
    ++ctx->prepared_references;++a->prepared_references;if(b!=a) ++b->prepared_references;
    *out=owner.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_execute_split_k(rns8_split_k* owner,rns8_matrix* output) {
  return guard_api([&]() -> rns8_status {
    if(!owner || !output) return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(owner->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    if(owner->quarantined || owner->ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    const auto* a=owner->a;const auto* b=owner->b;
    if(output==a || output==b) return RNS8_INVALID_ARGUMENT;
    if(output->in_flight_writer || output->in_flight_readers || a->in_flight_writer || b->in_flight_writer) return RNS8_BUSY;
    if(!split_input_matches(*owner->ctx,*a) || !split_input_matches(*owner->ctx,*b) ||
       a->matrix_instance_id!=owner->a_instance || b->matrix_instance_id!=owner->b_instance ||
       a->content_generation!=owner->a_generation || b->content_generation!=owner->b_generation ||
       output->backend!=RNS8_BACKEND_HIP_DIRECT || output->hip_device_id!=owner->ctx->device_id ||
       output->desc.semantics!=a->desc.semantics || output->prefix!=a->prefix ||
       output->desc.rows!=a->desc.rows || output->desc.cols!=b->desc.cols ||
       !rns_matrix_storage_matches(*output,output->backend,a->desc.rows,b->desc.cols,a->prefix)) return RNS8_INVALID_ARGUMENT;
    if(output->generation_exhausted || output->prepared_references==UINT64_MAX ||
       a->in_flight_readers==UINT64_MAX || b->in_flight_readers==UINT64_MAX) return RNS8_RANGE_ERROR;
    cpp_int bound=owner->bound;
    ++a->in_flight_readers;if(b!=a) ++b->in_flight_readers;
    ++output->prepared_references;output->in_flight_writer=true;
    owner->active_output=output;owner->commit_started=false;SplitReservation reservation{*owner};
    const auto status=hip_direct_gemm_rns_split_k_workspace(owner->ctx->device_id,a->hip_residues,b->hip_residues,
        output->hip_residues,a->desc.rows,b->desc.cols,a->desc.cols,a->desc.cols,b->desc.cols,b->desc.cols,
        a->prefix,owner->resources.fragments,owner->scratch,owner->resources.scratch_bytes,owner->commit_started);
    if(status!=RNS8_SUCCESS) return status;
    mark_output_device_residues_current(*output);output->exact_max_magnitude.swap(bound);
    output->exact_range_prefix=output->prefix;output->exact_axis_ranges.rows.clear();output->exact_axis_ranges.cols.clear();
    reservation.complete();return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_split_k(rns8_split_k* owner) {
  return guard_api([&]() -> rns8_status {
    if(!owner) return RNS8_SUCCESS;
    std::unique_lock<std::mutex> lock(owner->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    if(owner->quarantined || owner->active_output) {
      if(hip_direct_synchronize(owner->ctx->device_id)!=RNS8_SUCCESS) return RNS8_BACKEND_FAILURE;
      invalidate_split_output(*owner);release_split_readers(*owner);owner->quarantined=false;
    }
    if(owner->scratch) {
      const auto status=hip_direct_free(owner->ctx->device_id,owner->scratch);
      if(status!=RNS8_SUCCESS) return status;owner->scratch=nullptr;
    }
    --owner->ctx->prepared_references;--owner->a->prepared_references;
    if(owner->b!=owner->a) --owner->b->prepared_references;
    lock.unlock();delete owner;return RNS8_SUCCESS;
  });
}
