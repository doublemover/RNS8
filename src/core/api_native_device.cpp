#include "core/api_internal.hpp"
#include "backend_common/native_device_contract.hpp"
#include <mutex>
using namespace rns8::detail;
using namespace rns8::detail::api;
struct rns8_native_device_lease {
  rns8_context* ctx=nullptr;
  rns8_native_device_desc desc{};
  rns8_native_device_resources resources{};
  void* view=nullptr;
  void* scratch=nullptr;
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
  if(!nonoverlapping_byte_ranges(lease->view,lease->resources.view_span_bytes,matrix->hip_residues,matrix->hip_residue_bytes)) return RNS8_INVALID_ARGUMENT;
  return RNS8_SUCCESS;
}
void pin_native_matrix(rns8_native_device_lease& lease,rns8_matrix& matrix,bool writer) {
  lease.pinned=&matrix;lease.writer=writer;++matrix.prepared_references;
  if(writer) matrix.in_flight_writer=true;else ++matrix.in_flight_readers;
}
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
    auto lease=std::make_unique<rns8_native_device_lease>();
    auto status=query_native_device_contract(*desc,lease->resources);if(status!=RNS8_SUCCESS) return status;
    if(desc->device_id!=ctx->device_id) return RNS8_INVALID_ARGUMENT;
    if(ctx->prepared_references==UINT64_MAX) return RNS8_RANGE_ERROR;
    lease->ctx=ctx;lease->desc=*desc;
    lease->view=static_cast<uint8_t*>(desc->allocation_base)+desc->byte_offset;
    status=desc->acquire(desc->owner);if(status!=RNS8_SUCCESS) return status;
    status=hip_direct_validate_native_allocation(ctx->device_id,desc->allocation_base,desc->allocation_bytes);
    if(status!=RNS8_SUCCESS) {desc->release(desc->owner);return status;}
    ++ctx->prepared_references;*out=lease.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_native_device_lease(rns8_native_device_lease* lease) {
  return guard_api([&]() -> rns8_status {
    if(!lease) return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(lease->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    if(lease->quarantined) {
      if(hip_direct_synchronize(lease->ctx->device_id)!=RNS8_SUCCESS) return RNS8_BACKEND_FAILURE;
      lease->quarantined=false;release_native_matrix(*lease);
    }
    if(lease->scratch) {
      const auto status=hip_direct_free(lease->ctx->device_id,lease->scratch);
      if(status!=RNS8_SUCCESS) return status;lease->scratch=nullptr;
    }
    lease->desc.release(lease->desc.owner);--lease->ctx->prepared_references;
    lock.unlock();delete lease;return RNS8_SUCCESS;
  });
}
static rns8_status import_native_device_impl(rns8_context* ctx,rns8_matrix* matrix,rns8_native_device_lease* lease,bool verified,uint64_t budget) {
  return guard_api([&]() -> rns8_status {
    if(!lease) return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(lease->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    auto status=native_matrix_contract(ctx,matrix,lease,true);if(status!=RNS8_SUCCESS) return status;
    if(lease->desc.access!=RNS8_NATIVE_DEVICE_READ) return RNS8_INVALID_ARGUMENT;
    if(matrix->desc.flags & RNS8_MATRIX_RETAIN_NATIVE_INPUT) return RNS8_UNSUPPORTED_BACKEND;
    const bool signed_values=lease->desc.element_type==RNS8_NATIVE_DEVICE_I64;
    boost::multiprecision::cpp_int bound=signed_values?(boost::multiprecision::cpp_int(1)<<63):boost::multiprecision::cpp_int(UINT64_MAX);
    if(verified) {
      if(budget<lease->resources.verified_import_scratch_bytes) return RNS8_WORKSPACE_TOO_SMALL;
      if(!lease->scratch) {
        status=hip_direct_allocate(ctx->device_id,lease->resources.verified_import_scratch_bytes,&lease->scratch);
        if(status!=RNS8_SUCCESS) return status;
      }
      uint64_t magnitude=0;pin_native_matrix(*lease,*matrix,true);
      status=hip_direct_native_view_bound(ctx->device_id,lease->view,lease->scratch,
          matrix->desc.rows,matrix->desc.cols,lease->desc.ld,signed_values,magnitude);
      if(status!=RNS8_SUCCESS) return native_failure(*lease,status);
      release_native_matrix(*lease);bound=magnitude;
    }
    const auto semantics=signed_values?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
    if(!exact_range_fits(semantics,bound,modulus_product(matrix->prefix))) return RNS8_RANGE_ERROR;
    // No cache lookup by pointer or source-version: every import reads this lease.
    pin_native_matrix(*lease,*matrix,true);MatrixWriteGuard write(*matrix);
    status=hip_direct_import_native_view(ctx->device_id,lease->view,matrix->hip_residues,
        matrix->desc.rows,matrix->desc.cols,lease->desc.ld,matrix->prefix,signed_values);
    if(status!=RNS8_SUCCESS) return native_failure(*lease,status);
    mark_device_residues_current(*matrix);clear_native_current(*matrix);
    matrix->exact_max_magnitude.swap(bound);matrix->exact_range_prefix=matrix->prefix;
    matrix->exact_axis_ranges.rows.clear();matrix->exact_axis_ranges.cols.clear();
    // External generation is not a host-pack source-version cache key.
    publish_matrix_value(*matrix,false,0);write.committed=true;release_native_matrix(*lease);
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_export_native_device(rns8_context* ctx,const rns8_matrix* matrix,rns8_native_device_lease* lease,uint64_t budget) {
  return guard_api([&]() -> rns8_status {
    if(!lease) return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(lease->mutex,std::try_to_lock);if(!lock.owns_lock()) return RNS8_BUSY;
    auto status=native_matrix_contract(ctx,matrix,lease,false);if(status!=RNS8_SUCCESS) return status;
    if(lease->desc.access!=RNS8_NATIVE_DEVICE_WRITE || !matrix->device_residues_current) return RNS8_INVALID_ARGUMENT;
    const bool signed_values=lease->desc.element_type==RNS8_NATIVE_DEVICE_I64;
    const boost::multiprecision::cpp_int limit=signed_values?(boost::multiprecision::cpp_int(1)<<63):boost::multiprecision::cpp_int(UINT64_MAX);
    if(!matrix->exact_range_prefix || matrix->exact_range_prefix>matrix->prefix || matrix->exact_max_magnitude<0 ||
       matrix->exact_max_magnitude>limit || !exact_range_fits(signed_values?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED,
           matrix->exact_max_magnitude,modulus_product(matrix->exact_range_prefix))) return RNS8_RANGE_ERROR;
    if(budget<lease->resources.export_scratch_bytes) return RNS8_WORKSPACE_TOO_SMALL;
    if(!lease->scratch) {
      status=hip_direct_allocate(ctx->device_id,lease->resources.export_scratch_bytes,&lease->scratch);
      if(status!=RNS8_SUCCESS) return status;
    }
    const auto bound=matrix->exact_max_magnitude.convert_to<uint64_t>();
    pin_native_matrix(*lease,*const_cast<rns8_matrix*>(matrix),false);bool external_started=false;
    status=hip_direct_export_native_view(ctx->device_id,matrix->hip_residues,lease->view,lease->scratch,
        matrix->desc.rows,matrix->desc.cols,lease->desc.ld,matrix->prefix,bound,signed_values,external_started);
    if(status!=RNS8_SUCCESS) return native_failure(*lease,status);
    release_native_matrix(*lease);return RNS8_SUCCESS;
  });
}

rns8_status rns8_import_native_device(rns8_context* ctx,rns8_matrix* matrix,rns8_native_device_lease* lease) {
  return import_native_device_impl(ctx,matrix,lease,false,0);
}
rns8_status rns8_import_native_device_verified(rns8_context* ctx,rns8_matrix* matrix,rns8_native_device_lease* lease,uint64_t budget) {
  return import_native_device_impl(ctx,matrix,lease,true,budget);
}
