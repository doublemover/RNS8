#include "core/api_internal.hpp"
#include "backend_common/native_device_contract.hpp"
#include <mutex>
#include <utility>
using namespace rns8::detail;
using namespace rns8::detail::api;
struct rns8_native_device_lease {
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
rns8_status native_matrix_contract(const rns8_context* ctx,const rns8_matrix* matrix,const rns8_native_device_lease* lease,bool writing) {
  if(!ctx || !matrix || !lease || lease->ctx!=ctx) return RNS8_INVALID_ARGUMENT;
  if(ctx->hip_failure_poisoned || lease->quarantined) return RNS8_BACKEND_FAILURE;
  if(ctx->backend!=RNS8_BACKEND_HIP_DIRECT || matrix->backend!=ctx->backend || matrix->hip_device_id!=ctx->device_id ||
     matrix->desc.rows!=lease->desc.rows || matrix->desc.cols!=lease->desc.cols || matrix->generation_exhausted ||
     !native_semantics(matrix->desc.semantics,lease->desc.element_type==RNS8_NATIVE_DEVICE_I64) ||
     !rns_matrix_storage_matches(*matrix,ctx->backend,matrix->desc.rows,matrix->desc.cols,matrix->prefix)) return RNS8_INVALID_ARGUMENT;
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
    *out=lease.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_native_device_lease(rns8_native_device_lease* lease) {
  return guard_api([&]() -> rns8_status {
    if(!lease) return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(lease->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    if(lease->quarantined || lease->pinned) {
      if(hip_direct_synchronize(lease->ctx->device_id)!=RNS8_SUCCESS) return RNS8_BACKEND_FAILURE;
      lease->quarantined=false;release_native_matrix(*lease);
    }
    if(lease->scratch) {
      const auto status=hip_direct_free(lease->ctx->device_id,lease->scratch);
      if(status!=RNS8_SUCCESS) return status;lease->scratch=nullptr;
    }
    if(lease->snapshot) {
      const auto status=hip_direct_free(lease->ctx->device_id,lease->snapshot);
      if(status!=RNS8_SUCCESS) return status;lease->snapshot=nullptr;
    }
    if(lease->owner_acquired) {lease->desc.release(lease->desc.owner);lease->owner_acquired=false;}
    if(lease->context_reference) --lease->ctx->prepared_references;
    lock.unlock();delete lease;return RNS8_SUCCESS;
  });
}
static rns8_status import_native_device_impl(rns8_context* ctx,rns8_matrix* matrix,rns8_native_device_lease* lease,bool verified,uint64_t budget) {
  return guard_api([&]() -> rns8_status {
    if(!lease) return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(lease->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    auto status=native_matrix_contract(ctx,matrix,lease,true);if(status!=RNS8_SUCCESS) return status;
    if(lease->desc.access!=RNS8_NATIVE_DEVICE_READ) return RNS8_INVALID_ARGUMENT;
    const bool retain=(matrix->desc.flags & RNS8_MATRIX_RETAIN_NATIVE_INPUT)!=0;
    const bool signed_values=lease->desc.element_type==RNS8_NATIVE_DEVICE_I64;
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
    auto status=native_matrix_contract(ctx,matrix,lease,false);if(status!=RNS8_SUCCESS) return status;
    if(lease->desc.access!=RNS8_NATIVE_DEVICE_WRITE || !matrix->device_residues_current) return RNS8_INVALID_ARGUMENT;
    if(!materialized_rns_prefix_covers(*matrix,matrix->prefix)) return RNS8_RANGE_ERROR;
    const bool signed_values=lease->desc.element_type==RNS8_NATIVE_DEVICE_I64;
    const boost::multiprecision::cpp_int limit=signed_values?(boost::multiprecision::cpp_int(1)<<63):boost::multiprecision::cpp_int(UINT64_MAX);
    if(!matrix->exact_range_prefix || matrix->exact_range_prefix!=matrix->prefix || matrix->exact_max_magnitude<0 ||
       matrix->exact_max_magnitude>limit || !exact_range_fits(signed_values?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED,
           matrix->exact_max_magnitude,modulus_product(matrix->exact_range_prefix))) return RNS8_RANGE_ERROR;
    if(budget<lease->resources.export_scratch_bytes) return RNS8_WORKSPACE_TOO_SMALL;
    if(!lease->scratch) {
      status=hip_direct_allocate(ctx->device_id,lease->resources.export_scratch_bytes,&lease->scratch);
      if(status!=RNS8_SUCCESS) return status;
    }
    const auto bound=matrix->exact_max_magnitude.convert_to<uint64_t>();
    NativeMatrixReservation read(*lease,*const_cast<rns8_matrix*>(matrix),false,false);bool external_started=false;
    status=hip_direct_export_native_view(ctx->device_id,matrix->hip_residues,lease->view,lease->scratch,
        matrix->desc.rows,matrix->desc.cols,lease->view_ld,matrix->prefix,bound,signed_values,external_started);
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
