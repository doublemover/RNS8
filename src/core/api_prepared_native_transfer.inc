namespace {
class PreparedNativeTransfer final:public PreparedTransfer {
 public:
  rns8_native_device_lease* lease=nullptr;bool retained_lease=false,importing=false,verify=false,signed_values=false,published=false;
  uint32_t prefix=0,limbs=0;uint64_t bound_native=0,external_generation=0,published_generation=0;
  void* matrix_allocation=nullptr;void* retained_native=nullptr;uint64_t retained_bytes=0;
  boost::multiprecision::cpp_int envelope;rns8::detail::hip_direct_device_buffer scratch;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_pinned_host_buffer host_status;hipStream_t owned_stream=nullptr;
#endif
  ~PreparedNativeTransfer() override {(void)close();}
  bool writes_matrix() const noexcept override{return importing;}
  bool reads_matrix() const noexcept override{return !importing;}
  bool produces_resident() const noexcept override{return importing;}
  uint32_t output_prefix() const noexcept override{return prefix;}
  const char* operation_name() const noexcept override{return importing?"native_device_import":"native_device_export";}
  ExternalAccess external_access() const noexcept override{return lease?ExternalAccess{lease->view,lease->snapshot?lease->resources.compact_native_bytes:lease->resources.view_span_bytes,!importing,lease}:ExternalAccess{};}
  void* sync_stream() const noexcept override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    return owned_stream;
#else
    return nullptr;
#endif
  }
  rns8_status validate_value(const boost::multiprecision::cpp_int* pending,uint32_t pending_prefix,
      boost::multiprecision::cpp_int& bound,bool refresh_generation) {
    if(!lease || lease->ctx!=ctx || ctx->hip_failure_poisoned || lease->quarantined)return RNS8_BACKEND_FAILURE;
    if(lease->active || lease->write_window || lease->pinned)return RNS8_BUSY;
    const auto aliases=native_active_alias_contract(*lease);if(aliases!=RNS8_SUCCESS)return aliases;
    if(!refresh_generation && lease->desc.generation!=external_generation)return RNS8_INVALID_ARGUMENT;
    if(matrix->generation_exhausted || (importing && matrix->content_generation==UINT64_MAX))return RNS8_RANGE_ERROR;
    if(matrix->hip_residues!=matrix_allocation || matrix->backend!=RNS8_BACKEND_HIP_DIRECT || matrix->hip_device_id!=ctx->device_id ||
        !exact_wide_semantics(matrix->desc.semantics) || !native_semantics(matrix->desc.semantics,signed_values))return RNS8_INVALID_ARGUMENT;
    if(importing) {
      if(pending)return RNS8_INVALID_ARGUMENT;
      if(matrix->in_flight_writer || matrix->in_flight_readers)return RNS8_BUSY;
      if(retained_native && retained_native!=(signed_values?matrix->hip_native_i64:matrix->hip_native_u64))return RNS8_INVALID_ARGUMENT;
      bound=envelope;return RNS8_SUCCESS;
    }
    if(matrix->in_flight_writer && !pending)return RNS8_BUSY;
    if(pending && (!matrix->in_flight_writer || !matrix->prepared_in_flight_writers))return RNS8_INVALID_ARGUMENT;
    const auto current_prefix=pending?pending_prefix:matrix->exact_range_prefix;
    const auto& current_bound=pending?*pending:matrix->exact_max_magnitude;
    if((!pending && !matrix->device_residues_current) || current_prefix<prefix || current_bound<0 || current_bound>envelope ||
        !exact_range_fits(matrix->desc.semantics,current_bound,modulus_product(prefix)))return RNS8_RANGE_ERROR;
    bound=current_bound;return RNS8_SUCCESS;
  }
  rns8_status validate(const boost::multiprecision::cpp_int* pending,uint32_t pending_prefix,
      boost::multiprecision::cpp_int& bound) override {return validate_value(pending,pending_prefix,bound,false);}
  rns8_status refresh() override {boost::multiprecision::cpp_int bound;return validate_value(nullptr,0,bound,true);}
  void commit_refresh() noexcept override {external_generation=lease->desc.generation;published=false;}
  rns8_status before_enqueue(void* stream) override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(lease->readiness_pending && hipStreamWaitEvent(static_cast<hipStream_t>(stream),lease->readiness_event,0)!=hipSuccess)return RNS8_BACKEND_FAILURE;
#else
    (void)stream;
#endif
    return RNS8_SUCCESS;
  }
  rns8_status enqueue(void* stream) override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    return importing?hip_direct_enqueue_native_import(ctx->device_id,lease->view,matrix_allocation,matrix->desc.rows,matrix->desc.cols,
        lease->view_ld,prefix,signed_values,verify,bound_native,retained_native,scratch.get(),scratch.bytes(),host_status.get(),stream):
        hip_direct_enqueue_native_export(ctx->device_id,matrix_allocation,lease->view,matrix->desc.rows,matrix->desc.cols,lease->view_ld,
        prefix,signed_values,limbs,bound_native,scratch.get(),scratch.bytes(),host_status.get(),stream);
#else
    (void)stream;return RNS8_UNSUPPORTED_BACKEND;
#endif
  }
  void reserve() noexcept override {
    published=false;
    if(importing){clear_residue_current(*matrix);clear_native_current(*matrix);clear_byte_limb_current(*matrix);
      ++matrix->prepared_in_flight_writers;matrix->in_flight_writer=true;}
    else ++matrix->in_flight_readers;
    unlink_native_active(*lease);lease->active=true;lease->active_next=ctx->active_native_operations;ctx->active_native_operations=lease;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    *static_cast<int*>(host_status.get())=-1;
#endif
  }
  rns8_status finish(rns8_status operation,boost::multiprecision::cpp_int& bound) noexcept override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(operation==RNS8_SUCCESS){const int flag=*static_cast<const int*>(host_status.get());
      operation=flag==int(RNS8_SUCCESS)?RNS8_SUCCESS:flag==int(RNS8_RANGE_ERROR)?RNS8_RANGE_ERROR:RNS8_BACKEND_FAILURE;}
#endif
    if(importing) {
      --matrix->prepared_in_flight_writers;matrix->in_flight_writer=false;
      if(operation==RNS8_SUCCESS){mark_output_device_residues_current(*matrix);matrix->exact_max_magnitude.swap(bound);matrix->exact_range_prefix=prefix;
        matrix->materialized_rns_prefix=prefix;matrix->exact_axis_ranges.rows.clear();matrix->exact_axis_ranges.cols.clear();
        if(retained_native)mark_device_native_current(*matrix);}
      matrix->in_flight_writer=matrix->prepared_in_flight_writers!=0;
    } else --matrix->in_flight_readers;
    unlink_native_active(*lease);lease->active=false;
    if(operation==RNS8_SUCCESS){lease->readiness_pending=false;lease->producer_stream=nullptr;}
    else if(lease->readiness_pending){lease->active_next=ctx->active_native_operations;ctx->active_native_operations=lease;}
    published=operation==RNS8_SUCCESS;published_generation=matrix->content_generation;return operation;
  }
  bool output_current() const noexcept override {
    return published && lease && !lease->write_window && lease->desc.generation==external_generation &&
        matrix->content_generation==published_generation && matrix->device_residues_current;
  }
  void describe(rns8_prepared_transfer_info& info) const noexcept override {
    info.operation=importing?RNS8_OP_NATIVE_IMPORT:RNS8_OP_NATIVE_EXPORT;info.source_prefix=importing?0:prefix;info.target_prefix=importing?prefix:0;
    info.external_owner_identity=lease?lease->desc.owner_identity:0;info.external_generation=external_generation;
    info.retained_device_bytes=scratch.bytes();info.retained_pinned_host_bytes=4;info.retained_native_matrix_bytes=retained_bytes;
    info.verified_input_bound=verify;info.limb_count=limbs;info.external_result_current=!importing && output_current();
  }
  rns8_status close() override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(owned_stream){hip_scoped_device selected(ctx->device_id);if(selected.status()!=hipSuccess || hipStreamDestroy(owned_stream)!=hipSuccess)return RNS8_BACKEND_FAILURE;owned_stream=nullptr;}
    if(host_status.reset()!=hipSuccess)return RNS8_BACKEND_FAILURE;
#endif
    const auto status=scratch.reset();if(status!=RNS8_SUCCESS)return status;
    if(retained_lease){--lease->prepared_references;retained_lease=false;}return RNS8_SUCCESS;
  }
};
}
rns8_status rns8_begin_native_device_write(rns8_native_device_lease* lease) {
  return guard_api([&]() -> rns8_status {
    if(!lease || lease->desc.policy!=RNS8_NATIVE_DEVICE_BORROW)return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(lease->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    if(lease->ctx->hip_failure_poisoned || lease->quarantined)return RNS8_BACKEND_FAILURE;
    if(lease->active || lease->pinned || lease->write_window)return RNS8_BUSY;
    const auto ready=resolve_native_readiness(*lease,false);if(ready!=RNS8_SUCCESS)return ready;
    // An idle alias also binds the old external generation. Require a unique
    // mutable lease rather than silently invalidating another owner's proof.
    for(auto* other=lease->ctx->native_leases;other;other=other->registered_next)
      if(other!=lease && !nonoverlapping_byte_ranges(lease->view,lease->resources.view_span_bytes,other->view,
          other->snapshot?other->resources.compact_native_bytes:other->resources.view_span_bytes))return RNS8_BUSY;
    // A write window conflicts even with another read-only view of this owner.
    for(auto* other=lease->ctx->active_native_operations;other;other=other->active_next)
      if(!nonoverlapping_byte_ranges(lease->view,lease->resources.view_span_bytes,other->view,other->snapshot?other->resources.compact_native_bytes:other->resources.view_span_bytes))return RNS8_BUSY;
    lease->write_window=true;lease->desc.producer_complete=0;
    lease->active_next=lease->ctx->active_native_operations;lease->ctx->active_native_operations=lease;return RNS8_SUCCESS;
  });
}
rns8_status rns8_end_native_device_write(rns8_native_device_lease* lease,uint64_t generation) {
  return guard_api([&]() -> rns8_status {
    if(!lease)return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(lease->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    if(!lease->write_window || generation<=lease->desc.generation)return RNS8_INVALID_ARGUMENT;
    auto** cursor=&lease->ctx->active_native_operations;while(*cursor && *cursor!=lease)cursor=&(*cursor)->active_next;
    if(*cursor==lease)*cursor=lease->active_next;lease->active_next=nullptr;
    lease->desc.generation=generation;lease->desc.producer_complete=1;lease->write_window=false;return RNS8_SUCCESS;
  });
}
rns8_status rns8_prepare_native_transfer_hip(rns8_context* ctx,rns8_matrix* matrix,rns8_native_device_lease* lease,
    const rns8_native_transfer_options* options,rns8_prepared_execution** out) {
  return guard_api([&]() -> rns8_status {
    if(!ctx || !matrix || !lease || !options || !out || !valid_abi(options->struct_size,options->abi_version,sizeof(*options)) ||
        options->import_values>1 || options->verify_input_bound>1 || options->flags ||
        (!options->verify_input_bound && options->max_input_magnitude) || (!options->import_values && options->verify_input_bound))return RNS8_INVALID_ARGUMENT;
    *out=nullptr;std::unique_lock<std::mutex> lock(lease->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    if(!exact_wide_semantics(matrix->desc.semantics))return RNS8_UNSUPPORTED_BACKEND;
    auto status=native_matrix_contract(ctx,matrix,lease,options->import_values!=0);if(status!=RNS8_SUCCESS)return status;
    if(lease->desc.access!=(options->import_values?RNS8_NATIVE_DEVICE_READ:RNS8_NATIVE_DEVICE_WRITE) || lease->prepared_references==UINT64_MAX)return RNS8_INVALID_ARGUMENT;
    auto owner=std::make_unique<PreparedNativeTransfer>();owner->ctx=ctx;owner->matrix=matrix;owner->lease=lease;
    owner->importing=options->import_values!=0;owner->verify=options->verify_input_bound!=0;owner->signed_values=native_device_signed_type(lease->desc.element_type);
    owner->limbs=lease->desc.limb_count;owner->external_generation=lease->desc.generation;owner->matrix_allocation=matrix->hip_residues;
    owner->prefix=owner->importing?matrix->prefix:matrix->exact_range_prefix;
    if(!owner->prefix || owner->prefix>matrix->prefix)return RNS8_RANGE_ERROR;
    if(owner->importing){owner->bound_native=owner->verify?options->max_input_magnitude:owner->signed_values?(UINT64_C(1)<<63):UINT64_MAX;
      if(owner->signed_values && owner->bound_native>(UINT64_C(1)<<63))return RNS8_INVALID_ARGUMENT;owner->envelope=owner->bound_native;}
    else {if(!matrix->device_residues_current)return RNS8_INVALID_ARGUMENT;owner->envelope=matrix->exact_max_magnitude;
      const boost::multiprecision::cpp_int limit=owner->limbs?(boost::multiprecision::cpp_int(1)<<(64*owner->limbs-(owner->signed_values?1:0)))-1:
          owner->signed_values?boost::multiprecision::cpp_int(1)<<63:boost::multiprecision::cpp_int(UINT64_MAX);
      if(owner->envelope<0 || owner->envelope>limit)return RNS8_RANGE_ERROR;
      owner->bound_native=owner->limbs?0:owner->envelope.convert_to<uint64_t>();}
    if(!exact_range_fits(matrix->desc.semantics,owner->envelope,modulus_product(owner->prefix)))return RNS8_RANGE_ERROR;
    const uint64_t cells=uint64_t(matrix->desc.rows)*matrix->desc.cols;
    const bool retain=owner->importing && (matrix->desc.flags&RNS8_MATRIX_RETAIN_NATIVE_INPUT);
    const uint64_t scratch=owner->importing?cells*owner->prefix+16:lease->resources.export_scratch_bytes;
    rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;q.operation=owner->importing?RNS8_OP_NATIVE_IMPORT:RNS8_OP_NATIVE_EXPORT;
    q.execution_modes=RNS8_EXEC_SYNC|RNS8_EXEC_BORROWED_STREAM|RNS8_EXEC_OWNED_GRAPH;q.proof_flags=RNS8_QUERY_CURRENT_RANGE_PROOF|(retain?RNS8_QUERY_NATIVE_RETAINED:0);
    q.problem.struct_size=sizeof(q.problem);q.problem.abi_version=RNS8_ABI_VERSION;q.problem.semantics=matrix->desc.semantics;q.problem.bound_kind=RNS8_BOUND_NONE;q.problem.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    q.problem.m=matrix->desc.rows;q.problem.n=matrix->desc.cols;q.problem.k=1;q.problem.max_prefix=owner->prefix;
    q.lda=owner->importing?lease->view_ld:matrix->desc.cols;q.ldc=owner->importing?matrix->desc.cols:lease->view_ld;q.output_limbs=owner->limbs;q.algorithm=owner->verify;
    q.scratch_budget=options->max_total_scratch_bytes;const auto support=operation_support(*ctx,q,RNS8_BACKEND_HIP_DIRECT);if(support.status!=RNS8_SUCCESS)return support.status;
    if(retain){auto& pointer=owner->signed_values?matrix->hip_native_i64:matrix->hip_native_u64;auto& bytes=owner->signed_values?matrix->hip_native_i64_bytes:matrix->hip_native_u64_bytes;
      if((pointer || bytes) && (!pointer || bytes!=cells*8))return RNS8_INVALID_ARGUMENT;
      if(!pointer){status=hip_direct_allocate(ctx->device_id,cells*8,&pointer);if(status!=RNS8_SUCCESS)return status;bytes=cells*8;}
      owner->retained_native=pointer;owner->retained_bytes=bytes;}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    hip_scoped_device selected(ctx->device_id);if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    status=owner->scratch.allocate(ctx->device_id,scratch);if(status!=RNS8_SUCCESS)return status;
    if(owner->host_status.ensure(ctx->device_id,4)!=hipSuccess || hipStreamCreateWithFlags(&owner->owned_stream,hipStreamNonBlocking)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    ++lease->prepared_references;owner->retained_lease=true;
    return create_transfer_prepared(std::move(owner),out);
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}

rns8_status rns8_end_native_device_write_hip(rns8_native_device_lease* lease,uint64_t generation,void* producer_stream) {
  return guard_api([&]() -> rns8_status {
    if(!lease || !producer_stream)return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(lease->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    if(!lease->write_window || generation<=lease->desc.generation)return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    hip_scoped_device selected(lease->ctx->device_id);if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    const auto stream=static_cast<hipStream_t>(producer_stream);hipDevice_t device=-1;hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    if(stream==hipStreamLegacy || stream==hipStreamPerThread || hipStreamGetDevice(stream,&device)!=hipSuccess || device!=lease->ctx->device_id)return RNS8_INVALID_ARGUMENT;
    if(hipStreamIsCapturing(stream,&capture)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    if(capture!=hipStreamCaptureStatusNone)return RNS8_UNSUPPORTED_BACKEND;
    if(!lease->readiness_event && hipEventCreateWithFlags(&lease->readiness_event,hipEventDisableTiming)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    const auto code=hipEventRecord(lease->readiness_event,stream);
    lease->write_window=false;lease->producer_stream=producer_stream;
    if(code!=hipSuccess){lease->quarantined=true;lease->ctx->hip_failure_poisoned=true;return RNS8_BACKEND_FAILURE;}
    lease->readiness_pending=true;lease->desc.generation=generation;lease->desc.producer_complete=1;return RNS8_SUCCESS;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
