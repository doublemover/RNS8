#include "core/api_internal.hpp"
#include "core/prepared_transfer.hpp"
#include "backend_common/pinned_host_contract.hpp"
#include <atomic>
#include <cstring>
using namespace rns8::detail;
using namespace rns8::detail::api;
struct rns8_pinned_host_buffer {
  rns8_context* ctx=nullptr;rns8_pinned_host_desc desc{};PinnedHostResources resources;
  uint64_t instance=0,generation=0,prepared_references=0,host_readers=0,device_readers=0;
  bool current=false,write_window=false,device_writer=false;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_pinned_host_buffer storage;
  void* data() const noexcept{return storage.get();}
#else
  void* data() const noexcept{return nullptr;}
#endif
};
namespace {
uint64_t host_buffer_instance() noexcept {
  static std::atomic<uint64_t> next{1};auto value=next.load();
  while(value!=UINT64_MAX)if(next.compare_exchange_weak(value,value+1))return value;return 0;
}
bool host_buffer_writable(const rns8_pinned_host_buffer& b) noexcept {
  return !b.write_window && !b.device_writer && !b.host_readers && !b.device_readers;
}
class PreparedHostTransfer final:public PreparedTransfer {
 public:
  rns8_pinned_host_buffer* buffer=nullptr;bool retained=false,importing=false,signed_values=false,verify=false,published=false;
  uint32_t prefix=0,limbs=0;uint64_t external_generation=0,published_generation=0,bound_native=0,retained_bytes=0;
  void* matrix_allocation=nullptr;void* retained_native=nullptr;boost::multiprecision::cpp_int envelope;
  hip_direct_device_buffer native,conversion;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_pinned_host_buffer status;hipStream_t owned_stream=nullptr;
#endif
  ~PreparedHostTransfer() override {(void)close();if(retained && !native.pending() && !conversion.pending()){--buffer->prepared_references;retained=false;}}
  bool writes_matrix() const noexcept override{return importing;}
  bool reads_matrix() const noexcept override{return !importing;}
  bool produces_resident() const noexcept override{return importing;}
  uint32_t output_prefix() const noexcept override{return prefix;}
  const char* operation_name() const noexcept override{return importing?"pinned_host_import":"pinned_host_export";}
  ExternalAccess external_access() const noexcept override {
    return buffer?ExternalAccess{buffer->data(),buffer->resources.pinned,!importing,importing?nullptr:buffer}:ExternalAccess{};
  }
  void* sync_stream() const noexcept override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    return owned_stream;
#else
    return nullptr;
#endif
  }
  rns8_status value(const boost::multiprecision::cpp_int* pending,uint32_t pending_prefix,boost::multiprecision::cpp_int& bound,bool refresh) {
    if(!buffer || buffer->ctx!=ctx || ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
    if(buffer->write_window || buffer->device_writer || (!importing && (buffer->host_readers || buffer->device_readers)))return RNS8_BUSY;
    if(!refresh && buffer->generation!=external_generation)return RNS8_INVALID_ARGUMENT;
    if(matrix->generation_exhausted || (importing && (matrix->content_generation==UINT64_MAX || matrix->prepared_in_flight_writers==UINT64_MAX)) ||
       (!importing && buffer->generation==UINT64_MAX) || (importing && buffer->device_readers==UINT64_MAX))return RNS8_RANGE_ERROR;
    if(matrix->hip_residues!=matrix_allocation || matrix->backend!=RNS8_BACKEND_HIP_DIRECT || matrix->hip_device_id!=ctx->device_id ||
       !exact_wide_semantics(matrix->desc.semantics) || native_device_signed_type(buffer->desc.element_type)!=(matrix->desc.semantics==RNS8_EXACT_WIDE_SIGNED))return RNS8_INVALID_ARGUMENT;
    if(importing){if(pending || !buffer->current || !buffer->generation)return RNS8_INVALID_ARGUMENT;
      if(matrix->in_flight_writer || matrix->in_flight_readers)return RNS8_BUSY;
      if(retained_native && retained_native!=(signed_values?matrix->hip_native_i64:matrix->hip_native_u64))return RNS8_INVALID_ARGUMENT;
      bound=envelope;return RNS8_SUCCESS;}
    if(matrix->in_flight_writer && !pending)return RNS8_BUSY;
    if(matrix->in_flight_readers==UINT64_MAX)return RNS8_RANGE_ERROR;
    if(pending && (!matrix->in_flight_writer || !matrix->prepared_in_flight_writers))return RNS8_INVALID_ARGUMENT;
    const auto active=pending?pending_prefix:matrix->exact_range_prefix;const auto& magnitude=pending?*pending:matrix->exact_max_magnitude;
    if((!pending && !matrix->device_residues_current) || active<prefix || magnitude<0 || magnitude>envelope ||
       !exact_range_fits(matrix->desc.semantics,magnitude,modulus_product(prefix)))return RNS8_RANGE_ERROR;
    bound=magnitude;return RNS8_SUCCESS;
  }
  rns8_status validate(const boost::multiprecision::cpp_int* pending,uint32_t pending_prefix,boost::multiprecision::cpp_int& bound) override {return value(pending,pending_prefix,bound,false);}
  rns8_status refresh() override {boost::multiprecision::cpp_int bound;return value(nullptr,0,bound,true);}
  void commit_refresh() noexcept override {external_generation=buffer->generation;published=false;}
  rns8_status enqueue(void* raw) override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    const auto stream=static_cast<hipStream_t>(raw);const uint64_t width=uint64_t(buffer->desc.cols)*buffer->resources.element_bytes;
    if(importing){if(hipMemcpy2DAsync(native.get(),width,buffer->data(),uint64_t(buffer->desc.ld)*8,width,buffer->desc.rows,hipMemcpyHostToDevice,stream)!=hipSuccess)return RNS8_BACKEND_FAILURE;
      return hip_direct_enqueue_native_import(ctx->device_id,native.get(),matrix_allocation,matrix->desc.rows,matrix->desc.cols,matrix->desc.cols,
          prefix,signed_values,verify,bound_native,retained_native,conversion.get(),conversion.bytes(),status.get(),raw);}
    auto result=hip_direct_enqueue_native_export(ctx->device_id,matrix_allocation,native.get(),matrix->desc.rows,matrix->desc.cols,matrix->desc.cols,
        prefix,signed_values,limbs,bound_native,conversion.get(),conversion.bytes(),status.get(),raw);
    if(result!=RNS8_SUCCESS)return result;
    return hipMemcpy2DAsync(buffer->data(),uint64_t(buffer->desc.ld)*buffer->resources.element_bytes,native.get(),width,width,buffer->desc.rows,hipMemcpyDeviceToHost,stream)==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
    (void)raw;return RNS8_UNSUPPORTED_BACKEND;
#endif
  }
  void reserve() noexcept override {
    published=false;native.mark_pending();conversion.mark_pending();
    if(importing){++buffer->device_readers;clear_residue_current(*matrix);clear_native_current(*matrix);clear_byte_limb_current(*matrix);++matrix->prepared_in_flight_writers;matrix->in_flight_writer=true;}
    else{buffer->device_writer=true;buffer->current=false;++matrix->in_flight_readers;}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    buffer->storage.mark_pending();status.mark_pending();
    *static_cast<int*>(status.get())=-1;
#endif
  }
  rns8_status finish(rns8_status operation,boost::multiprecision::cpp_int& bound) noexcept override {
    native.mark_completed();conversion.mark_completed();
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    status.mark_completed();
    if(operation==RNS8_SUCCESS){const auto flag=*static_cast<int*>(status.get());operation=flag==int(RNS8_SUCCESS)?RNS8_SUCCESS:flag==int(RNS8_RANGE_ERROR)?RNS8_RANGE_ERROR:RNS8_BACKEND_FAILURE;}
#endif
    if(importing){--buffer->device_readers;--matrix->prepared_in_flight_writers;matrix->in_flight_writer=false;
      if(operation==RNS8_SUCCESS){mark_output_device_residues_current(*matrix);matrix->exact_max_magnitude.swap(bound);matrix->exact_range_prefix=prefix;matrix->materialized_rns_prefix=prefix;
        matrix->exact_axis_ranges.rows.clear();matrix->exact_axis_ranges.cols.clear();if(retained_native)mark_device_native_current(*matrix);}
      matrix->in_flight_writer=matrix->prepared_in_flight_writers!=0;published_generation=matrix->content_generation;
    }else{--matrix->in_flight_readers;buffer->device_writer=false;
      if(operation==RNS8_SUCCESS){++buffer->generation;buffer->current=true;external_generation=buffer->generation;published_generation=buffer->generation;}}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(!buffer->device_readers && !buffer->device_writer)buffer->storage.mark_completed();
#endif
    published=operation==RNS8_SUCCESS;return operation;
  }
  bool output_current() const noexcept override {
    if(!published)return false;
    return importing?matrix->content_generation==published_generation && matrix->device_residues_current:
      buffer->current && !buffer->device_writer && !buffer->write_window && buffer->generation==published_generation;
  }
  void describe(rns8_prepared_transfer_info& i) const noexcept override {
    i.operation=importing?RNS8_OP_HOST_IMPORT:RNS8_OP_HOST_EXPORT;i.source_prefix=importing?0:prefix;i.target_prefix=importing?prefix:0;
    i.external_owner_identity=buffer?buffer->instance:0;i.external_generation=external_generation;
    i.retained_device_bytes=native.bytes()+conversion.bytes();i.retained_pinned_host_bytes=buffer?buffer->resources.pinned+4:4;i.retained_native_matrix_bytes=retained_bytes;
    i.verified_input_bound=verify;i.limb_count=limbs;i.external_result_current=!importing && output_current();
  }
  rns8_status close() override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(owned_stream){hip_scoped_device selected(ctx->device_id);if(selected.status()!=hipSuccess || hipStreamDestroy(owned_stream)!=hipSuccess)return RNS8_BACKEND_FAILURE;owned_stream=nullptr;}
    if(status.reset()!=hipSuccess)return RNS8_BACKEND_FAILURE;
#endif
    auto result=native.reset();if(result!=RNS8_SUCCESS)return result;result=conversion.reset();if(result!=RNS8_SUCCESS)return result;
    if(retained){--buffer->prepared_references;retained=false;}return RNS8_SUCCESS;
  }
};
}
rns8_status rns8_query_pinned_host_bytes(const rns8_pinned_host_desc* desc,uint64_t* out) {
  if(!desc || !out)return RNS8_INVALID_ARGUMENT;PinnedHostResources r;const auto status=query_pinned_host_contract(*desc,r);if(status==RNS8_SUCCESS)*out=r.pinned;return status;
}
rns8_status rns8_create_pinned_host_buffer_hip(rns8_context* ctx,const rns8_pinned_host_desc* desc,uint64_t budget,rns8_pinned_host_buffer** out) {
  return guard_api([&]()->rns8_status{if(!ctx || !desc || !out)return RNS8_INVALID_ARGUMENT;*out=nullptr;PinnedHostResources resources;
    auto result=query_pinned_host_contract(*desc,resources);if(result!=RNS8_SUCCESS)return result;if(resources.pinned>budget)return RNS8_WORKSPACE_TOO_SMALL;
    if(ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;if(ctx->backend!=RNS8_BACKEND_HIP_DIRECT || !direct_hip_compiled())return RNS8_UNSUPPORTED_BACKEND;
    if(ctx->prepared_references==UINT64_MAX)return RNS8_RANGE_ERROR;
    auto buffer=std::make_unique<rns8_pinned_host_buffer>();buffer->ctx=ctx;buffer->desc=*desc;buffer->resources=resources;buffer->instance=host_buffer_instance();if(!buffer->instance)return RNS8_RANGE_ERROR;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(buffer->storage.ensure(ctx->device_id,resources.pinned)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    std::memset(buffer->data(),0,resources.pinned);++ctx->prepared_references;*out=buffer.release();return RNS8_SUCCESS;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
rns8_status rns8_destroy_pinned_host_buffer(rns8_pinned_host_buffer* b) {
  if(!b)return RNS8_SUCCESS;if(b->prepared_references || !host_buffer_writable(*b))return RNS8_BUSY;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(b->storage.reset()!=hipSuccess)return RNS8_BACKEND_FAILURE;
#endif
  --b->ctx->prepared_references;delete b;return RNS8_SUCCESS;
}
rns8_status rns8_get_pinned_host_info(const rns8_pinned_host_buffer* b,rns8_pinned_host_info* out) {
  if(!b || !out || !valid_abi(out->struct_size,out->abi_version,sizeof(*out)))return RNS8_INVALID_ARGUMENT;
  *out={};out->struct_size=sizeof(*out);out->abi_version=RNS8_ABI_VERSION;out->instance=b->instance;out->generation=b->generation;out->pinned_bytes=b->resources.pinned;
  out->current=b->current && !b->write_window && !b->device_writer;out->write_window=b->write_window;out->device_writer=b->device_writer;
  out->host_readers=b->host_readers;out->device_readers=b->device_readers;out->prepared_references=b->prepared_references;return RNS8_SUCCESS;
}
rns8_status rns8_begin_pinned_host_write(rns8_pinned_host_buffer* b,void** data,uint64_t* bytes) {
  if(!b || !data || !bytes)return RNS8_INVALID_ARGUMENT;if(b->ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
  if(!host_buffer_writable(*b))return RNS8_BUSY;if(b->generation==UINT64_MAX)return RNS8_RANGE_ERROR;
  b->write_window=true;b->current=false;*data=b->data();*bytes=b->resources.pinned;return RNS8_SUCCESS;
}
rns8_status rns8_end_pinned_host_write(rns8_pinned_host_buffer* b,uint64_t generation) {
  if(!b || !b->write_window || generation<=b->generation)return RNS8_INVALID_ARGUMENT;
  b->generation=generation;b->write_window=false;b->current=true;return RNS8_SUCCESS;
}
rns8_status rns8_begin_pinned_host_read(rns8_pinned_host_buffer* b,const void** data,uint64_t* bytes) {
  if(!b || !data || !bytes)return RNS8_INVALID_ARGUMENT;if(b->write_window || b->device_writer)return RNS8_BUSY;
  if(!b->current)return RNS8_INVALID_ARGUMENT;if(b->host_readers==UINT64_MAX)return RNS8_RANGE_ERROR;
  ++b->host_readers;*data=b->data();*bytes=b->resources.pinned;return RNS8_SUCCESS;
}
rns8_status rns8_end_pinned_host_read(rns8_pinned_host_buffer* b) {
  if(!b || !b->host_readers)return RNS8_INVALID_ARGUMENT;--b->host_readers;return RNS8_SUCCESS;
}
rns8_status rns8_prepare_host_transfer_hip(rns8_context* ctx,rns8_matrix* matrix,rns8_pinned_host_buffer* buffer,
    const rns8_native_transfer_options* options,rns8_prepared_execution** out) {
  return guard_api([&]()->rns8_status{
    if(!ctx || !matrix || !buffer || !options || !out || !valid_abi(options->struct_size,options->abi_version,sizeof(*options)) || options->flags || options->import_values>1 || options->verify_input_bound>1)return RNS8_INVALID_ARGUMENT;
    *out=nullptr;if(ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
    if(ctx->backend!=RNS8_BACKEND_HIP_DIRECT || !direct_hip_compiled())return RNS8_UNSUPPORTED_BACKEND;
    const bool importing=options->import_values,sign=native_device_signed_type(buffer->desc.element_type);
    if(buffer->ctx!=ctx || matrix->backend!=ctx->backend || matrix->hip_device_id!=ctx->device_id || !exact_wide_semantics(matrix->desc.semantics) ||
       sign!=(matrix->desc.semantics==RNS8_EXACT_WIDE_SIGNED) || matrix->desc.rows!=buffer->desc.rows || matrix->desc.cols!=buffer->desc.cols ||
       buffer->desc.access!=(importing?RNS8_NATIVE_DEVICE_READ:RNS8_NATIVE_DEVICE_WRITE) ||
       (!options->verify_input_bound && options->max_input_magnitude) || (!importing && (options->verify_input_bound || options->max_input_magnitude)) ||
       (importing && native_device_limb_type(buffer->desc.element_type)))return RNS8_INVALID_ARGUMENT;
    if(buffer->prepared_references==UINT64_MAX)return RNS8_RANGE_ERROR;
    if(!rns_matrix_storage_matches(*matrix,ctx->backend,matrix->desc.rows,matrix->desc.cols,matrix->prefix))return RNS8_INVALID_ARGUMENT;
    auto owner=std::make_unique<PreparedHostTransfer>();owner->ctx=ctx;owner->matrix=matrix;owner->buffer=buffer;owner->importing=importing;owner->signed_values=sign;owner->verify=options->verify_input_bound;
    owner->limbs=buffer->desc.limb_count;owner->external_generation=buffer->generation;owner->matrix_allocation=matrix->hip_residues;owner->prefix=importing?matrix->prefix:matrix->exact_range_prefix;
    if(!owner->prefix || owner->prefix>matrix->prefix)return RNS8_RANGE_ERROR;
    if(importing){owner->bound_native=owner->verify?options->max_input_magnitude:sign?(UINT64_C(1)<<63):UINT64_MAX;
      if(sign && owner->bound_native>(UINT64_C(1)<<63))return RNS8_INVALID_ARGUMENT;owner->envelope=owner->bound_native;
    }else{if(!matrix->device_residues_current)return RNS8_INVALID_ARGUMENT;owner->envelope=matrix->exact_max_magnitude;
      const boost::multiprecision::cpp_int limit=owner->limbs?(boost::multiprecision::cpp_int(1)<<(64*owner->limbs-(sign?1:0)))-1:
        sign?boost::multiprecision::cpp_int(1)<<63:boost::multiprecision::cpp_int(UINT64_MAX);
      if(owner->envelope<0 || owner->envelope>limit)return RNS8_RANGE_ERROR;owner->bound_native=owner->limbs?0:owner->envelope.convert_to<uint64_t>();}
    if(!exact_range_fits(matrix->desc.semantics,owner->envelope,modulus_product(owner->prefix)))return RNS8_RANGE_ERROR;
    const bool retain=importing && (matrix->desc.flags&RNS8_MATRIX_RETAIN_NATIVE_INPUT);uint64_t device=0,pinned=0,total=0;
    if(!pinned_host_transfer_bytes(buffer->resources,owner->prefix,importing,retain,device,pinned,total))return RNS8_RANGE_ERROR;
    if(total>options->max_total_scratch_bytes)return RNS8_WORKSPACE_TOO_SMALL;
    rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;q.operation=importing?RNS8_OP_HOST_IMPORT:RNS8_OP_HOST_EXPORT;
    q.execution_modes=RNS8_EXEC_SYNC|RNS8_EXEC_BORROWED_STREAM|RNS8_EXEC_OWNED_GRAPH;q.proof_flags=RNS8_QUERY_CURRENT_RANGE_PROOF|(retain?RNS8_QUERY_NATIVE_RETAINED:0);
    q.problem.struct_size=sizeof(q.problem);q.problem.abi_version=RNS8_ABI_VERSION;q.problem.semantics=matrix->desc.semantics;q.problem.bound_kind=RNS8_BOUND_NONE;
    q.problem.requested_backend=RNS8_BACKEND_HIP_DIRECT;q.problem.m=matrix->desc.rows;q.problem.n=matrix->desc.cols;q.problem.k=1;q.problem.max_prefix=owner->prefix;
    q.lda=importing?buffer->desc.ld:matrix->desc.cols;q.ldc=importing?matrix->desc.cols:buffer->desc.ld;q.algorithm=owner->verify;q.output_limbs=owner->limbs;q.scratch_budget=options->max_total_scratch_bytes;
    const auto support=operation_support(*ctx,q,RNS8_BACKEND_HIP_DIRECT);if(support.status!=RNS8_SUCCESS)return support.status;
    boost::multiprecision::cpp_int bound;auto valid=owner->validate(nullptr,0,bound);if(valid!=RNS8_SUCCESS)return valid;
    if(retain){const uint64_t bytes=uint64_t(matrix->desc.rows)*matrix->desc.cols*8;auto& pointer=sign?matrix->hip_native_i64:matrix->hip_native_u64;auto& capacity=sign?matrix->hip_native_i64_bytes:matrix->hip_native_u64_bytes;
      if((pointer || capacity) && (!pointer || capacity!=bytes))return RNS8_INVALID_ARGUMENT;
      if(!pointer){valid=hip_direct_allocate(ctx->device_id,bytes,&pointer);if(valid!=RNS8_SUCCESS)return valid;capacity=bytes;}owner->retained_native=pointer;owner->retained_bytes=bytes;}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    hip_scoped_device selected(ctx->device_id);if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    valid=owner->native.allocate(ctx->device_id,buffer->resources.compact);if(valid!=RNS8_SUCCESS)return valid;
    valid=owner->conversion.allocate(ctx->device_id,device-buffer->resources.compact);if(valid!=RNS8_SUCCESS)return valid;
    if(owner->status.ensure(ctx->device_id,4)!=hipSuccess || hipStreamCreateWithFlags(&owner->owned_stream,hipStreamNonBlocking)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    ++buffer->prepared_references;owner->retained=true;return create_transfer_prepared(std::move(owner),out);
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
