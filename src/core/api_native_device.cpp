#include "core/api_internal.hpp"
#include "core/prepared_transfer.hpp"
#include "core/resident_native_import.hpp"
#include "backend_common/native_device_contract.hpp"
#include <mutex>
#include <utility>
using namespace rns8::detail;
using namespace rns8::detail::api;
struct rns8_native_device_lease {
  uint64_t prepared_references=0;
  bool active=false,write_window=false,readiness_pending=false;
  void* producer_stream=nullptr;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hipEvent_t readiness_event=nullptr;
#endif
  rns8_native_device_lease* active_next=nullptr;
  rns8_native_device_lease* registered_next=nullptr;
  bool registered=false;
  rns8_context* ctx=nullptr;
  rns8_native_device_desc desc{};
  rns8_native_device_resources resources{};
  void* view=nullptr;
  void* scratch=nullptr;
  void* snapshot=nullptr;
  int64_t view_ld=0;
  bool owner_acquired=false,context_reference=false;
  rns8_native_device_lease* failed_cleanup_next=nullptr;
  rns8_matrix* pinned=nullptr;
  bool writer=false,quarantined=false;
  std::mutex mutex;
};
namespace {
bool native_semantics(rns8_semantics semantics,bool signed_values) {
  return signed_values?(semantics==RNS8_BOUNDED_I64 || semantics==RNS8_EXACT_WIDE_SIGNED):
                       (semantics==RNS8_BOUNDED_U64 || semantics==RNS8_EXACT_WIDE_UNSIGNED);
}
void release_native_matrix(rns8_native_device_lease& lease) {
  if(!lease.pinned) return;
  if(lease.writer) lease.pinned->in_flight_writer=false;
  else --lease.pinned->in_flight_readers;
  --lease.pinned->prepared_references;lease.pinned=nullptr;
}
rns8_status native_failure(rns8_native_device_lease& lease,rns8_status original) {
  if(hip_direct_synchronize(lease.ctx->device_id)!=RNS8_SUCCESS) {
    lease.quarantined=true;lease.ctx->hip_failure_poisoned=true;
  } else release_native_matrix(lease);
  return original;
}
void unlink_native_active(rns8_native_device_lease& lease) noexcept {
  auto** cursor=&lease.ctx->active_native_operations;while(*cursor && *cursor!=&lease)cursor=&(*cursor)->active_next;
  if(*cursor==&lease)*cursor=lease.active_next;lease.active_next=nullptr;
}
rns8_status native_active_alias_contract(const rns8_native_device_lease& lease) {
  const auto bytes=lease.snapshot?lease.resources.compact_native_bytes:lease.resources.view_span_bytes;
  for(auto* other=lease.ctx->active_native_operations;other;other=other->active_next) {
    if(other==&lease){if(lease.readiness_pending && !lease.active && !lease.write_window)continue;return RNS8_BUSY;}
    if((lease.desc.access==RNS8_NATIVE_DEVICE_WRITE || other->desc.access==RNS8_NATIVE_DEVICE_WRITE || other->write_window || other->readiness_pending) &&
        !nonoverlapping_byte_ranges(lease.view,bytes,other->view,other->snapshot?other->resources.compact_native_bytes:other->resources.view_span_bytes))return RNS8_BUSY;
  }
  return RNS8_SUCCESS;
}
rns8_status resolve_native_readiness(rns8_native_device_lease& lease,bool wait) {
  if(!lease.readiness_pending)return RNS8_SUCCESS;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_scoped_device selected(lease.ctx->device_id);if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
  const auto code=wait?hipEventSynchronize(lease.readiness_event):hipEventQuery(lease.readiness_event);
  if(code==hipErrorNotReady)return RNS8_BUSY;
  if(code!=hipSuccess){lease.quarantined=true;lease.ctx->hip_failure_poisoned=true;return RNS8_BACKEND_FAILURE;}
  if(!lease.active)unlink_native_active(lease);
  lease.readiness_pending=false;lease.producer_stream=nullptr;return RNS8_SUCCESS;
#else
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status native_matrix_contract(const rns8_context* ctx,const rns8_matrix* matrix,const rns8_native_device_lease* lease,bool writing) {
  if(!ctx || !matrix || !lease || lease->ctx!=ctx) return RNS8_INVALID_ARGUMENT;
  if(lease->active || lease->write_window)return RNS8_BUSY;
  const auto external=native_active_alias_contract(*lease);if(external!=RNS8_SUCCESS)return external;
  if(ctx->hip_failure_poisoned || lease->quarantined) return RNS8_BACKEND_FAILURE;
  if(ctx->backend!=RNS8_BACKEND_HIP_DIRECT || matrix->backend!=ctx->backend || matrix->hip_device_id!=ctx->device_id ||
     matrix->desc.rows!=lease->desc.rows || matrix->desc.cols!=lease->desc.cols || matrix->generation_exhausted ||
     !native_semantics(matrix->desc.semantics,native_device_signed_type(lease->desc.element_type)) ||
     !rns_matrix_storage_matches(*matrix,ctx->backend,matrix->desc.rows,matrix->desc.cols,matrix->prefix)) return RNS8_INVALID_ARGUMENT;
  if(native_device_limb_type(lease->desc.element_type) &&
      (writing || !exact_wide_semantics(matrix->desc.semantics))) return RNS8_INVALID_ARGUMENT;
  if(matrix->in_flight_writer || (writing && matrix->in_flight_readers)) return RNS8_BUSY;
  if(matrix->prepared_references==UINT64_MAX || (!writing && matrix->in_flight_readers==UINT64_MAX)) return RNS8_RANGE_ERROR;
  if(!nonoverlapping_byte_ranges(lease->view,lease->snapshot?lease->resources.compact_native_bytes:lease->resources.view_span_bytes,matrix->hip_residues,matrix->hip_residue_bytes)) return RNS8_INVALID_ARGUMENT;
  for(const auto storage:{std::pair<const void*,uint64_t>{matrix->hip_native_i64,matrix->hip_native_i64_bytes},
                         std::pair<const void*,uint64_t>{matrix->hip_native_u64,matrix->hip_native_u64_bytes}})
    if((storage.first || storage.second) &&
       !nonoverlapping_byte_ranges(lease->view,lease->snapshot?lease->resources.compact_native_bytes:lease->resources.view_span_bytes,storage.first,storage.second))
      return RNS8_INVALID_ARGUMENT;
  return RNS8_SUCCESS;
}
void pin_native_matrix(rns8_native_device_lease& lease,rns8_matrix& matrix,bool writer) {
  lease.pinned=&matrix;lease.writer=writer;++matrix.prepared_references;
  if(writer) matrix.in_flight_writer=true;else ++matrix.in_flight_readers;
}
// Owns an already-admitted reservation. Generic MatrixWriteGuard is unsuitable:
// it checks writability again, including our own writer, during destruction.
struct NativeMatrixReservation {
  rns8_native_device_lease& lease;
  rns8_matrix& matrix;
  bool mutating=false,finished=false;
  NativeMatrixReservation(rns8_native_device_lease& owner,rns8_matrix& value,bool writer,bool writes)
      :lease(owner),matrix(value),mutating(writes) {
    if(mutating) invalidate_output_currentness(matrix); // before our writer exists
    pin_native_matrix(lease,matrix,writer);
  }
  NativeMatrixReservation(const NativeMatrixReservation&)=delete;
  ~NativeMatrixReservation() noexcept {
    if(finished) return;
    if(mutating) {
      clear_residue_current(matrix);clear_byte_limb_current(matrix);clear_native_current(matrix);
    }
    if(!lease.pinned || lease.quarantined) return;
    // Even an unexpected host exception cannot prove device completion.
    try { (void)native_failure(lease,RNS8_INTERNAL_ERROR); }
    catch(...) {lease.quarantined=true;lease.ctx->hip_failure_poisoned=true;}
  }
  void complete() noexcept {release_native_matrix(lease);finished=true;}
};
}
rns8_status rns8_query_native_device(const rns8_native_device_desc* desc,rns8_native_device_resources* out) {
  return guard_api([&]() -> rns8_status {
    if(!desc || !out || !valid_abi(out->struct_size,out->abi_version,sizeof(*out))) return RNS8_INVALID_ARGUMENT;
    rns8_native_device_resources result{};const auto status=query_native_device_contract(*desc,result);
    if(status==RNS8_SUCCESS) *out=result;return status;
  });
}
rns8_status rns8_create_native_device_lease(rns8_context* ctx,const rns8_native_device_desc* desc,rns8_native_device_lease** out) {
  return guard_api([&]() -> rns8_status {
    if(!ctx || !desc || !out) return RNS8_INVALID_ARGUMENT;
    if(ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    if(ctx->backend!=RNS8_BACKEND_HIP_DIRECT || !hip_direct_compiled()) return RNS8_UNSUPPORTED_BACKEND;
    const auto discard=[](rns8_native_device_lease* lease) noexcept {
      if(!lease) return;
      auto* context=lease->ctx;
      if(!context) {delete lease;return;} // query failed before any ownership transfer
      if(rns8_destroy_native_device_lease(lease)==RNS8_SUCCESS) return;
      context->hip_failure_poisoned=true;
      if(lease->context_reference) {--context->prepared_references;lease->context_reference=false;}
      lease->failed_cleanup_next=context->failed_native_cleanup;context->failed_native_cleanup=lease;
    };
    std::unique_ptr<rns8_native_device_lease,decltype(discard)> lease(new rns8_native_device_lease,discard);
    auto status=query_native_device_contract(*desc,lease->resources);if(status!=RNS8_SUCCESS) return status;
    if(desc->device_id!=ctx->device_id) return RNS8_INVALID_ARGUMENT;
    if(ctx->prepared_references==UINT64_MAX) return RNS8_RANGE_ERROR;
    lease->ctx=ctx;lease->desc=*desc;lease->view_ld=desc->ld;
    ++ctx->prepared_references;lease->context_reference=true;
    lease->view=static_cast<uint8_t*>(desc->allocation_base)+desc->byte_offset;
    // Multiple immutable read views may coexist. A borrowed writer excludes
    // overlapping leases for its entire owner lifetime, not just one kernel.
    for(auto* other=ctx->native_leases;other;other=other->registered_next) {
      if(nonoverlapping_byte_ranges(lease->view,lease->resources.view_span_bytes,other->view,
          other->snapshot?other->resources.compact_native_bytes:other->resources.view_span_bytes))continue;
      if(desc->access==RNS8_NATIVE_DEVICE_WRITE || other->desc.access==RNS8_NATIVE_DEVICE_WRITE)return RNS8_INVALID_ARGUMENT;
      if(other->write_window || other->readiness_pending)return RNS8_BUSY;
    }

    status=desc->acquire(desc->owner);if(status!=RNS8_SUCCESS) return status;
    lease->owner_acquired=true;
    status=hip_direct_validate_native_allocation(ctx->device_id,desc->allocation_base,desc->allocation_bytes);
    if(status!=RNS8_SUCCESS) return status;
    if(desc->policy==RNS8_NATIVE_DEVICE_COPY_OWNED) {
      status=hip_direct_allocate(ctx->device_id,lease->resources.snapshot_bytes,&lease->snapshot);
      if(status!=RNS8_SUCCESS) return status;
      // Keep both ownership domains live until a successful drain, including
      // exceptions or copy failure before the lease can be returned publicly.
      lease->quarantined=true;
      status=hip_direct_copy_native_view(ctx->device_id,lease->view,lease->snapshot,desc->rows,desc->cols,desc->ld);
      if(status!=RNS8_SUCCESS) return status;
      lease->quarantined=false;lease->view=lease->snapshot;lease->view_ld=desc->cols;
      desc->release(desc->owner);lease->owner_acquired=false;
    }
    lease->registered_next=ctx->native_leases;ctx->native_leases=lease.get();lease->registered=true;
    *out=lease.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_native_device_lease(rns8_native_device_lease* lease) {
  return guard_api([&]() -> rns8_status {
    if(!lease) return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(lease->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    if(lease->prepared_references || lease->active || lease->write_window)return RNS8_BUSY;
    const auto ready=resolve_native_readiness(*lease,true);if(ready!=RNS8_SUCCESS && !lease->quarantined)return ready;
    if(lease->quarantined || lease->pinned) {
      if(hip_direct_synchronize(lease->ctx->device_id)!=RNS8_SUCCESS) return RNS8_BACKEND_FAILURE;
      unlink_native_active(*lease);lease->quarantined=false;lease->readiness_pending=false;lease->producer_stream=nullptr;release_native_matrix(*lease);
    }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(lease->readiness_event){hip_scoped_device selected(lease->ctx->device_id);if(selected.status()!=hipSuccess || hipEventDestroy(lease->readiness_event)!=hipSuccess)return RNS8_BACKEND_FAILURE;lease->readiness_event=nullptr;}
#endif
    if(lease->scratch) {
      const auto status=hip_direct_free(lease->ctx->device_id,lease->scratch);
      if(status!=RNS8_SUCCESS) return status;lease->scratch=nullptr;
    }
    if(lease->snapshot) {
      const auto status=hip_direct_free(lease->ctx->device_id,lease->snapshot);
      if(status!=RNS8_SUCCESS) return status;lease->snapshot=nullptr;
    }
    if(lease->registered){auto** cursor=&lease->ctx->native_leases;while(*cursor && *cursor!=lease)cursor=&(*cursor)->registered_next;
      if(*cursor==lease)*cursor=lease->registered_next;lease->registered_next=nullptr;lease->registered=false;}
    if(lease->owner_acquired) {lease->desc.release(lease->desc.owner);lease->owner_acquired=false;}
    if(lease->context_reference) --lease->ctx->prepared_references;
    lock.unlock();delete lease;return RNS8_SUCCESS;
  });
}
static rns8_status import_native_device_impl(rns8_context* ctx,rns8_matrix* matrix,rns8_native_device_lease* lease,bool verified,uint64_t budget) {
  return guard_api([&]() -> rns8_status {
    if(!lease) return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(lease->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    auto status=resolve_native_readiness(*lease,true);if(status!=RNS8_SUCCESS)return status;
    status=native_matrix_contract(ctx,matrix,lease,true);if(status!=RNS8_SUCCESS) return status;
    if(lease->desc.access!=RNS8_NATIVE_DEVICE_READ) return RNS8_INVALID_ARGUMENT;
    const bool retain=(matrix->desc.flags & RNS8_MATRIX_RETAIN_NATIVE_INPUT)!=0;
    const bool signed_values=native_device_signed_type(lease->desc.element_type);
    boost::multiprecision::cpp_int bound=signed_values?(boost::multiprecision::cpp_int(1)<<63):boost::multiprecision::cpp_int(UINT64_MAX);
    if(verified) {
      if(budget<lease->resources.verified_import_scratch_bytes) return RNS8_WORKSPACE_TOO_SMALL;
      if(!lease->scratch) {
        status=hip_direct_allocate(ctx->device_id,lease->resources.verified_import_scratch_bytes,&lease->scratch);
        if(status!=RNS8_SUCCESS) return status;
      }
      uint64_t magnitude=0;NativeMatrixReservation scan(*lease,*matrix,true,false);
      status=hip_direct_native_view_bound(ctx->device_id,lease->view,lease->scratch,
          matrix->desc.rows,matrix->desc.cols,lease->view_ld,signed_values,magnitude);
      if(status!=RNS8_SUCCESS) return native_failure(*lease,status);
      scan.complete();bound=magnitude;
    }
    const auto semantics=signed_values?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
    if(!exact_range_fits(semantics,bound,modulus_product(matrix->prefix))) return RNS8_RANGE_ERROR;
    void* retained=nullptr;
    if(retain) {
      auto& storage=signed_values?matrix->hip_native_i64:matrix->hip_native_u64;
      auto& bytes=signed_values?matrix->hip_native_i64_bytes:matrix->hip_native_u64_bytes;
      const auto expected=lease->resources.compact_native_bytes;
      if((storage || bytes) && (!storage || bytes!=expected)) return RNS8_INVALID_ARGUMENT;
      if(!storage) {
        // No memset: the upcoming logical-row copy writes the complete compact
        // allocation. Matrix ownership survives any subsequent failed drain.
        status=hip_direct_allocate(ctx->device_id,static_cast<size_t>(expected),&storage);
        if(status!=RNS8_SUCCESS) return status;
        bytes=static_cast<size_t>(expected);
      }
      retained=storage;
    }
    // No cache lookup by pointer or source-version: every import reads this lease.
    NativeMatrixReservation write(*lease,*matrix,true,true);
    status=hip_direct_import_native_view(ctx->device_id,lease->view,matrix->hip_residues,
        matrix->desc.rows,matrix->desc.cols,lease->view_ld,matrix->prefix,signed_values,retained);
    if(status!=RNS8_SUCCESS) return native_failure(*lease,status);
    mark_device_residues_current(*matrix);clear_native_current(*matrix);
    if(retain) mark_device_native_current(*matrix);
    matrix->exact_max_magnitude.swap(bound);matrix->exact_range_prefix=matrix->prefix;
    matrix->exact_axis_ranges.rows.clear();matrix->exact_axis_ranges.cols.clear();
    // External generation is not a host-pack source-version cache key.
    publish_matrix_value(*matrix,false,0);write.complete();
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_export_native_device(rns8_context* ctx,const rns8_matrix* matrix,rns8_native_device_lease* lease,uint64_t budget) {
  return guard_api([&]() -> rns8_status {
    if(!lease) return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(lease->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    auto status=resolve_native_readiness(*lease,true);if(status!=RNS8_SUCCESS)return status;
    status=native_matrix_contract(ctx,matrix,lease,false);if(status!=RNS8_SUCCESS) return status;
    if(lease->desc.access!=RNS8_NATIVE_DEVICE_WRITE || !matrix->device_residues_current) return RNS8_INVALID_ARGUMENT;
    if(!materialized_rns_prefix_covers(*matrix,matrix->prefix)) return RNS8_RANGE_ERROR;
    const bool signed_values=native_device_signed_type(lease->desc.element_type);
    const bool limbs=native_device_limb_type(lease->desc.element_type);
    boost::multiprecision::cpp_int limit;
    if(limbs) limit=(boost::multiprecision::cpp_int(1)<<
        (64*lease->desc.limb_count-(signed_values?1:0)))-1;
    else if(signed_values) limit=boost::multiprecision::cpp_int(1)<<63;
    else limit=UINT64_MAX;
    if(!matrix->exact_range_prefix || matrix->exact_range_prefix!=matrix->prefix || matrix->exact_max_magnitude<0 ||
       matrix->exact_max_magnitude>limit || !exact_range_fits(signed_values?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED,
           matrix->exact_max_magnitude,modulus_product(matrix->exact_range_prefix))) return RNS8_RANGE_ERROR;
    if(budget<lease->resources.export_scratch_bytes) return RNS8_WORKSPACE_TOO_SMALL;
    if(!lease->scratch) {
      status=hip_direct_allocate(ctx->device_id,lease->resources.export_scratch_bytes,&lease->scratch);
      if(status!=RNS8_SUCCESS) return status;
    }
    const auto bound=limbs?uint64_t{0}:matrix->exact_max_magnitude.convert_to<uint64_t>();
    NativeMatrixReservation read(*lease,*const_cast<rns8_matrix*>(matrix),false,false);bool external_started=false;
    status=hip_direct_export_native_view(ctx->device_id,matrix->hip_residues,lease->view,lease->scratch,
        matrix->desc.rows,matrix->desc.cols,lease->view_ld,matrix->prefix,bound,signed_values,external_started,lease->desc.limb_count);
    if(status!=RNS8_SUCCESS) return native_failure(*lease,status);
    read.complete();return RNS8_SUCCESS;
  });
}

rns8_status rns8_import_native_device(rns8_context* ctx,rns8_matrix* matrix,rns8_native_device_lease* lease) {
  return import_native_device_impl(ctx,matrix,lease,false,0);
}
rns8_status rns8_import_native_device_verified(rns8_context* ctx,rns8_matrix* matrix,rns8_native_device_lease* lease,uint64_t budget) {
  return import_native_device_impl(ctx,matrix,lease,true,budget);
}

rns8_status rns8::detail::api::drain_failed_native_cleanup(rns8_context& ctx) {
  while(ctx.failed_native_cleanup) {
    auto* lease=ctx.failed_native_cleanup;auto* next=lease->failed_cleanup_next;
    const auto status=rns8_destroy_native_device_lease(lease);
    if(status!=RNS8_SUCCESS) return status;
    ctx.failed_native_cleanup=next;
  }
  return RNS8_SUCCESS;
}

#include "api_prepared_native_transfer.inc"

namespace rns8::detail::api {
namespace {
class GraphNativeImport final:public ResidentNativeImport {
 public:
  rns8_context* ctx=nullptr;rns8_matrix* matrix=nullptr;rns8_native_device_lease* lease=nullptr;
  uint64_t generation=0,bound_native=0;uint32_t prefix=0;
  bool verify=false,signed_values=false,held=false;
  void* allocation=nullptr;void* native=nullptr;uint64_t native_bytes=0;
  ~GraphNativeImport() override {if(held)--lease->prepared_references;}
  rns8_status validate(bool check_generation,boost::multiprecision::cpp_int& bound) const override {
    const auto status=native_matrix_contract(ctx,matrix,lease,true);if(status!=RNS8_SUCCESS)return status;
    if(lease->pinned || lease->desc.access!=RNS8_NATIVE_DEVICE_READ || matrix->hip_residues!=allocation ||
        (check_generation && generation!=lease->desc.generation))return RNS8_INVALID_ARGUMENT;
    if(matrix->content_generation==UINT64_MAX)return RNS8_RANGE_ERROR;
    if(native && native!=(signed_values?matrix->hip_native_i64:matrix->hip_native_u64))return RNS8_INVALID_ARGUMENT;
    bound=bound_native;return RNS8_SUCCESS;
  }
  rns8_status prepare_retained_native() override {
    if(!(matrix->desc.flags&RNS8_MATRIX_RETAIN_NATIVE_INPUT))return RNS8_SUCCESS;
    auto& pointer=signed_values?matrix->hip_native_i64:matrix->hip_native_u64;
    auto& bytes=signed_values?matrix->hip_native_i64_bytes:matrix->hip_native_u64_bytes;
    if((pointer || bytes) && (!pointer || bytes!=native_bytes))return RNS8_INVALID_ARGUMENT;
    if(!pointer) {const auto status=hip_direct_allocate(ctx->device_id,native_bytes,&pointer);if(status!=RNS8_SUCCESS)return status;bytes=native_bytes;}
    native=pointer;return RNS8_SUCCESS;
  }
  rns8_status before_enqueue(void* stream) override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(lease->readiness_pending && hipStreamWaitEvent(static_cast<hipStream_t>(stream),lease->readiness_event,0)!=hipSuccess)return RNS8_BACKEND_FAILURE;
#else
    (void)stream;
#endif
    return RNS8_SUCCESS;
  }
  rns8_status enqueue(void* scratch,uint64_t bytes,void* pinned_status,void* stream) override {
    return hip_direct_enqueue_native_import(ctx->device_id,lease->view,allocation,matrix->desc.rows,matrix->desc.cols,
        lease->view_ld,prefix,signed_values,verify,bound_native,native,scratch,bytes,pinned_status,stream);
  }
  void reserve_external() noexcept override {
    unlink_native_active(*lease);lease->active=true;
    lease->active_next=ctx->active_native_operations;ctx->active_native_operations=lease;
  }
  void release_external(bool success) noexcept override {
    unlink_native_active(*lease);lease->active=false;
    if(success){lease->readiness_pending=false;lease->producer_stream=nullptr;}
    else if(lease->readiness_pending){lease->active_next=ctx->active_native_operations;ctx->active_native_operations=lease;}
  }
  void commit_refresh() noexcept override {generation=lease->desc.generation;}
  const void* owner_key() const noexcept override {return lease;}
  const void* source() const noexcept override {return lease->view;}
  uint64_t source_bytes() const noexcept override {return lease->snapshot?lease->resources.compact_native_bytes:lease->resources.view_span_bytes;}
  void* retained_native() const noexcept override {return native;}
  uint64_t retained_native_bytes() const noexcept override {return native?native_bytes:0;}
  uint64_t required_native_bytes() const noexcept override {return native_bytes;}
};
}
rns8_status create_resident_native_import(rns8_context* ctx,const rns8_resident_node& node,
    std::unique_ptr<ResidentNativeImport>& out) {
  if(!ctx || !node.c || !node.native_source || node.verify_native_bound>1 ||
      (!node.verify_native_bound && node.native_max_magnitude))return RNS8_INVALID_ARGUMENT;
  auto* lease=node.native_source;auto* matrix=node.c;
  if(!exact_wide_semantics(matrix->desc.semantics))return RNS8_UNSUPPORTED_BACKEND;
  const auto status=native_matrix_contract(ctx,matrix,lease,true);if(status!=RNS8_SUCCESS)return status;
  if(lease->desc.access!=RNS8_NATIVE_DEVICE_READ || lease->pinned || native_device_limb_type(lease->desc.element_type) ||
      !node.target_prefix || node.target_prefix>matrix->prefix)return RNS8_INVALID_ARGUMENT;
  if(lease->prepared_references==UINT64_MAX)return RNS8_RANGE_ERROR;
  auto owner=std::make_unique<GraphNativeImport>();owner->ctx=ctx;owner->matrix=matrix;owner->lease=lease;
  owner->generation=lease->desc.generation;owner->prefix=node.target_prefix;owner->verify=node.verify_native_bound!=0;
  owner->signed_values=native_device_signed_type(lease->desc.element_type);owner->allocation=matrix->hip_residues;
  owner->bound_native=owner->verify?node.native_max_magnitude:owner->signed_values?(UINT64_C(1)<<63):UINT64_MAX;
  if(owner->signed_values && owner->bound_native>(UINT64_C(1)<<63))return RNS8_INVALID_ARGUMENT;
  if(!exact_range_fits(matrix->desc.semantics,owner->bound_native,modulus_product(owner->prefix)))return RNS8_RANGE_ERROR;
  if(matrix->desc.flags&RNS8_MATRIX_RETAIN_NATIVE_INPUT)owner->native_bytes=uint64_t(matrix->desc.rows)*matrix->desc.cols*8;
  rns8_operation_query query{};query.struct_size=sizeof(query);query.abi_version=RNS8_ABI_VERSION;
  query.operation=RNS8_OP_NATIVE_IMPORT;query.execution_modes=RNS8_EXEC_SYNC|RNS8_EXEC_BORROWED_STREAM|RNS8_EXEC_OWNED_GRAPH;
  query.proof_flags=RNS8_QUERY_CURRENT_RANGE_PROOF|(owner->native_bytes?RNS8_QUERY_NATIVE_RETAINED:0);
  query.problem.struct_size=sizeof(query.problem);query.problem.abi_version=RNS8_ABI_VERSION;
  query.problem.semantics=matrix->desc.semantics;query.problem.bound_kind=RNS8_BOUND_NONE;
  query.problem.requested_backend=RNS8_BACKEND_HIP_DIRECT;query.problem.m=matrix->desc.rows;query.problem.n=matrix->desc.cols;
  query.problem.k=1;query.problem.max_prefix=owner->prefix;query.lda=lease->view_ld;query.ldc=matrix->desc.cols;
  query.algorithm=owner->verify;query.scratch_budget=node.max_transfer_storage_bytes;
  const auto support=operation_support(*ctx,query,RNS8_BACKEND_HIP_DIRECT);if(support.status!=RNS8_SUCCESS)return support.status;
  ++lease->prepared_references;owner->held=true;out=std::move(owner);return RNS8_SUCCESS;
}
}
