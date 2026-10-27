namespace {
class PreparedBasisConversion final:public PreparedTransfer {
 public:
  const rns8_matrix* source=nullptr;
  std::shared_ptr<const ExperimentalBasis> from,to;
  uint32_t source_prefix=0,target_prefix=0;
  uint64_t cells=0,published_generation=0;
  const void* input_allocation=nullptr;void* output_allocation=nullptr;
  cpp_int envelope;BasisMagnitude magnitude{};hip_direct_device_buffer scratch;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_pinned_host_buffer host_status;hipStream_t stream=nullptr;
#endif
  ~PreparedBasisConversion() override {(void)close();}
  const rns8_matrix* input_a() const noexcept override{return source;}
  const rns8_matrix* input_b() const noexcept override{return source;}
  bool reads_matrix() const noexcept override{return true;}
  bool writes_matrix() const noexcept override{return true;}
  bool produces_resident() const noexcept override{return true;}
  uint32_t output_prefix() const noexcept override{return target_prefix;}
  const char* operation_name() const noexcept override{return "ordered_basis_integer_conversion";}
  void* sync_stream() const noexcept override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    return stream;
#else
    return nullptr;
#endif
  }
  rns8_status validate(const cpp_int* pending,uint32_t pending_prefix,cpp_int& bound) override {
    if(ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
    if(source==matrix || source->hip_residues!=input_allocation || matrix->hip_residues!=output_allocation ||
        stored_basis(*source)!=from || stored_basis(*matrix)!=to || source->desc.semantics!=matrix->desc.semantics ||
        !exact_wide_semantics(matrix->desc.semantics))return RNS8_INVALID_ARGUMENT;
    for(const auto* value:{source,static_cast<const rns8_matrix*>(matrix)}) {
      if(!basis_storage_matches(*value,RNS8_BACKEND_HIP_DIRECT) || value->hip_device_id!=ctx->device_id)return RNS8_INVALID_ARGUMENT;
      if(value->generation_exhausted)return RNS8_RANGE_ERROR;
    }
    if(matrix->in_flight_writer || matrix->in_flight_readers || (!pending && source->in_flight_writer))return RNS8_BUSY;
    if(source->in_flight_readers==UINT64_MAX || matrix->prepared_in_flight_writers==UINT64_MAX ||
        matrix->content_generation==UINT64_MAX)return RNS8_RANGE_ERROR;
    if(pending && (!source->in_flight_writer || !source->prepared_in_flight_writers))return RNS8_INVALID_ARGUMENT;
    const auto prefix=pending?pending_prefix:source->exact_range_prefix;
    const auto& value=pending?*pending:source->exact_max_magnitude;
    if((!pending && !source->device_residues_current) || prefix<source_prefix || value<0 || value>envelope ||
        !exact_range_fits(source->desc.semantics,value,from->products[source_prefix]) ||
        !exact_range_fits(matrix->desc.semantics,value,to->products[target_prefix]))return RNS8_RANGE_ERROR;
    if(!nonoverlapping_byte_ranges(input_allocation,source->hip_residue_bytes,output_allocation,matrix->hip_residue_bytes))return RNS8_INVALID_ARGUMENT;
    bound=value;return RNS8_SUCCESS;
  }
  rns8_status enqueue(void* target) override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    return hip_direct_enqueue_basis_conversion(ctx->device_id,input_allocation,output_allocation,cells,from->packet,to->packet,
        source_prefix,target_prefix,matrix->desc.semantics==RNS8_EXACT_WIDE_SIGNED,magnitude,scratch.get(),scratch.bytes(),host_status.get(),target);
#else
    (void)target;return RNS8_UNSUPPORTED_BACKEND;
#endif
  }
  void reserve() noexcept override {
    ++const_cast<rns8_matrix*>(source)->in_flight_readers;
    clear_residue_current(*matrix);clear_native_current(*matrix);clear_byte_limb_current(*matrix);
    ++matrix->prepared_in_flight_writers;matrix->in_flight_writer=true;published_generation=0;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    *static_cast<int*>(host_status.get())=-1;
#endif
  }
  rns8_status finish(rns8_status operation,cpp_int& bound) noexcept override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(operation==RNS8_SUCCESS) {
      const auto flag=*static_cast<const int*>(host_status.get());
      operation=flag==0?RNS8_SUCCESS:flag==int(RNS8_RANGE_ERROR)?RNS8_RANGE_ERROR:RNS8_BACKEND_FAILURE;
    }
#endif
    --const_cast<rns8_matrix*>(source)->in_flight_readers;
    --matrix->prepared_in_flight_writers;matrix->in_flight_writer=matrix->prepared_in_flight_writers!=0;
    if(operation==RNS8_SUCCESS) {
      mark_output_device_residues_current(*matrix);matrix->exact_max_magnitude.swap(bound);
      matrix->exact_range_prefix=target_prefix;matrix->materialized_rns_prefix=target_prefix;
      matrix->exact_axis_ranges.rows.clear();matrix->exact_axis_ranges.cols.clear();published_generation=matrix->content_generation;
    }
    return operation;
  }
  bool output_current() const noexcept override {
    return published_generation && matrix->content_generation==published_generation && matrix->device_residues_current;
  }
  void describe(rns8_prepared_transfer_info& info) const noexcept override {
    info.operation=RNS8_OP_BASE_EXTEND;info.source_prefix=source_prefix;info.target_prefix=target_prefix;
    info.retained_device_bytes=scratch.bytes();info.retained_pinned_host_bytes=4;
  }
  rns8_status close() override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(stream){hip_scoped_device selected(ctx->device_id);if(selected.status()!=hipSuccess || hipStreamDestroy(stream)!=hipSuccess)return RNS8_BACKEND_FAILURE;stream=nullptr;}
    if(host_status.reset()!=hipSuccess)return RNS8_BACKEND_FAILURE;
#endif
    return scratch.reset();
  }
};
}
rns8_status rns8_prepare_basis_conversion_hip(rns8_basis_plan* plan,const rns8_matrix* source,rns8_matrix* destination,
    uint64_t budget,rns8_prepared_execution** output) {
  return guard_api([&]() -> rns8_status {
    if(!output)return RNS8_INVALID_ARGUMENT;*output=nullptr;
    if(!plan || !source || !destination || source==destination)return RNS8_INVALID_ARGUMENT;
    if(plan->desc.backend!=RNS8_BACKEND_HIP_DIRECT || !direct_hip_compiled())return RNS8_UNSUPPORTED_BACKEND;
    std::unique_lock<std::mutex> lock(plan->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    auto status=basis_plan_ready(*plan);if(status!=RNS8_SUCCESS)return status;
    status=basis_matrix_contract(*plan,*source,true,false);if(status!=RNS8_SUCCESS)return status;
    status=basis_matrix_contract(*plan,*destination,false);if(status!=RNS8_SUCCESS)return status;
    if(source->desc.rows!=destination->desc.rows || source->desc.cols!=destination->desc.cols)return RNS8_INVALID_ARGUMENT;
    auto owner=std::make_unique<PreparedBasisConversion>();owner->ctx=plan->ctx;owner->matrix=destination;owner->source=source;
    owner->from=stored_basis(*source);owner->to=stored_basis(*destination);owner->source_prefix=source->exact_range_prefix;
    owner->target_prefix=destination->prefix;owner->cells=uint64_t(source->desc.rows)*source->desc.cols;
    owner->input_allocation=source->hip_residues;owner->output_allocation=destination->hip_residues;
    owner->envelope=source->exact_max_magnitude;basis_words(owner->envelope,owner->magnitude.words);
    cpp_int bound;status=owner->validate(nullptr,0,bound);if(status!=RNS8_SUCCESS)return status;
    if(owner->cells>INT32_MAX || owner->cells>(UINT64_MAX-8)/owner->target_prefix)return RNS8_RANGE_ERROR;
    const auto device_bytes=owner->cells*owner->target_prefix+4;
    if(device_bytes+4>budget)return RNS8_WORKSPACE_TOO_SMALL;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    hip_scoped_device selected(plan->ctx->device_id);if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    status=owner->scratch.allocate(plan->ctx->device_id,device_bytes);if(status!=RNS8_SUCCESS)return status;
    if(owner->host_status.ensure(plan->ctx->device_id,4)!=hipSuccess ||
        hipStreamCreateWithFlags(&owner->stream,hipStreamNonBlocking)!=hipSuccess)return RNS8_BACKEND_FAILURE;
#endif
    return create_transfer_prepared(std::move(owner),output);
  });
}
