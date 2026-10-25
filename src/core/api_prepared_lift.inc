namespace {
class PreparedExactLift final:public PreparedTransfer {
 public:
  uint32_t source=0,target=0;bool mixed=false;int64_t cells=0;
  void* allocation=nullptr;uint64_t bytes=0;boost::multiprecision::cpp_int envelope;
  uint64_t bound_limbs[3]{};rns8::detail::hip_direct_device_buffer scratch;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8::detail::hip_pinned_host_buffer host_status;hipStream_t owned_stream=nullptr;
#endif
  ~PreparedExactLift() override {(void)close();}
  bool writes_matrix() const noexcept override{return true;}
  bool reads_matrix() const noexcept override{return true;}
  bool produces_resident() const noexcept override{return true;}
  uint32_t output_prefix() const noexcept override{return target;}
  const char* operation_name() const noexcept override{return "exact_base_extension";}
  void* sync_stream() const noexcept override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    return owned_stream;
#else
    return nullptr;
#endif
  }
  rns8_status validate(const boost::multiprecision::cpp_int* pending,uint32_t pending_prefix,
      boost::multiprecision::cpp_int& output_bound) override {
    using namespace rns8::detail;
    if(ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
    if(ctx->backend!=RNS8_BACKEND_HIP_DIRECT || matrix->backend!=ctx->backend || matrix->hip_device_id!=ctx->device_id)
      return RNS8_UNSUPPORTED_BACKEND;
    if(matrix->generation_exhausted || matrix->prepared_in_flight_writers==UINT64_MAX)return RNS8_RANGE_ERROR;
    if(matrix->in_flight_readers || (matrix->in_flight_writer && !pending))return RNS8_BUSY;
    if(pending && (!matrix->in_flight_writer || !matrix->prepared_in_flight_writers))return RNS8_INVALID_ARGUMENT;
    if(matrix->hip_residues!=allocation || matrix->hip_residue_bytes!=bytes || matrix->materialized_rns_schedule ||
        !exact_wide_semantics(matrix->desc.semantics) || target>matrix->prefix)return RNS8_INVALID_ARGUMENT;
    const auto prefix=pending?pending_prefix:matrix->exact_range_prefix;
    const auto& bound=pending?*pending:matrix->exact_max_magnitude;
    if((!pending && !matrix->device_residues_current) || prefix<source || bound<0 || bound>envelope ||
        !exact_range_fits(matrix->desc.semantics,bound,modulus_product(source)))return RNS8_RANGE_ERROR;
    output_bound=bound;return RNS8_SUCCESS;
  }
  rns8_status enqueue(void* stream) override {
    if(target==source)return RNS8_SUCCESS;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    return rns8::detail::hip_direct_enqueue_exact_lift(ctx->device_id,allocation,cells,source,target,
        matrix->desc.semantics==RNS8_EXACT_WIDE_SIGNED,bound_limbs,scratch.get(),scratch.bytes(),host_status.get(),mixed,stream);
#else
    (void)stream;return RNS8_UNSUPPORTED_BACKEND;
#endif
  }
  void reserve() noexcept override {
    ++matrix->prepared_in_flight_writers;matrix->in_flight_writer=true;scratch.mark_pending();
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    host_status.mark_pending();*static_cast<int*>(host_status.get())=target==source?0:-1;
#endif
  }
  rns8_status finish(rns8_status operation,boost::multiprecision::cpp_int&) noexcept override {
    scratch.mark_completed();
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    host_status.mark_completed();
    if(operation==RNS8_SUCCESS){const int flag=*static_cast<const int*>(host_status.get());
      operation=flag==0?RNS8_SUCCESS:flag==1?RNS8_INVALID_ARGUMENT:flag==2?RNS8_RANGE_ERROR:RNS8_BACKEND_FAILURE;}
#endif
    if(operation==RNS8_SUCCESS){matrix->exact_range_prefix=std::max(matrix->exact_range_prefix,target);
      matrix->materialized_rns_prefix=std::max(matrix->materialized_rns_prefix,target);
      if(target>source)matrix->host_residues_current=false;}
    // Only the suffix was written. Failure/cancellation leaves the previously
    // proven source prefix and its logical generation intact.
    --matrix->prepared_in_flight_writers;matrix->in_flight_writer=matrix->prepared_in_flight_writers!=0;
    return operation;
  }
  bool output_current() const noexcept override{return matrix->device_residues_current && matrix->exact_range_prefix>=target;}
  void describe(rns8_prepared_transfer_info& info) const noexcept override {
    info.operation=RNS8_OP_BASE_EXTEND;info.source_prefix=source;info.target_prefix=target;
    info.retained_device_bytes=scratch.bytes();info.retained_pinned_host_bytes=4;
  }
  rns8_status close() override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(owned_stream){rns8::detail::hip_scoped_device selected(ctx->device_id);
      if(selected.status()!=hipSuccess || hipStreamDestroy(owned_stream)!=hipSuccess)return RNS8_BACKEND_FAILURE;owned_stream=nullptr;}
    if(host_status.reset()!=hipSuccess)return RNS8_BACKEND_FAILURE;
#endif
    return scratch.reset();
  }
};
}
rns8_status rns8_prepare_exact_lift_hip(rns8_context* ctx,rns8_matrix* matrix,
    const rns8_exact_lift_options* options,rns8_prepared_execution** out) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!ctx || !matrix || !options || !out || !valid_abi(options->struct_size,options->abi_version,sizeof(*options)) ||
        options->algorithm>1 || options->flags || !options->target_prefix || options->target_prefix>matrix->prefix)return RNS8_INVALID_ARGUMENT;
    *out=nullptr;if(ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
    if(ctx->backend!=RNS8_BACKEND_HIP_DIRECT || !direct_hip_compiled())return RNS8_UNSUPPORTED_BACKEND;
    require_matrix_writable(*matrix);
    if(!exact_wide_semantics(matrix->desc.semantics) || !matrix->device_residues_current || !matrix->exact_range_prefix ||
        matrix->exact_range_prefix>matrix->prefix || matrix->hip_device_id!=ctx->device_id ||
        !rns_matrix_storage_matches(*matrix,ctx->backend,matrix->desc.rows,matrix->desc.cols,matrix->prefix))return RNS8_INVALID_ARGUMENT;
    if(!exact_range_fits(matrix->desc.semantics,matrix->exact_max_magnitude,modulus_product(matrix->exact_range_prefix)))return RNS8_RANGE_ERROR;
    auto owner=std::make_unique<PreparedExactLift>();owner->ctx=ctx;owner->matrix=matrix;owner->source=matrix->exact_range_prefix;
    owner->target=std::max(owner->source,options->target_prefix);owner->mixed=options->algorithm==1;
    std::size_t cells=0;if(!matrix_cell_count(matrix->desc.rows,matrix->desc.cols,cells) || cells>INT64_MAX)return RNS8_RANGE_ERROR;
    owner->cells=int64_t(cells);owner->envelope=matrix->exact_max_magnitude;owner->allocation=matrix->hip_residues;owner->bytes=matrix->hip_residue_bytes;
    auto bound=owner->envelope;for(auto& limb:owner->bound_limbs){limb=static_cast<uint64_t>(bound&UINT64_MAX);bound>>=64;}if(bound!=0)return RNS8_RANGE_ERROR;
    rns8_operation_query query{};query.struct_size=sizeof(query);query.abi_version=RNS8_ABI_VERSION;
    query.operation=RNS8_OP_BASE_EXTEND;query.execution_modes=RNS8_EXEC_SYNC|RNS8_EXEC_BORROWED_STREAM|RNS8_EXEC_OWNED_GRAPH;
    query.problem.struct_size=sizeof(query.problem);query.problem.abi_version=RNS8_ABI_VERSION;
    query.problem.semantics=matrix->desc.semantics;query.problem.bound_kind=RNS8_BOUND_NONE;query.problem.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    query.problem.m=matrix->desc.rows;query.problem.n=matrix->desc.cols;query.problem.k=1;query.problem.max_prefix=owner->target;
    query.proof_flags=RNS8_QUERY_CURRENT_RANGE_PROOF;
    query.source_prefix=owner->source;query.algorithm=options->algorithm;query.scratch_budget=options->max_total_scratch_bytes;
    const auto provider=operation_support(*ctx,query,RNS8_BACKEND_HIP_DIRECT);if(provider.status!=RNS8_SUCCESS)return provider.status;
    const uint64_t needed=owner->target==owner->source?0:query_exact_lift_support(owner->cells,owner->source,owner->target,UINT64_MAX).scratch_bytes;
    if(owner->target>owner->source){const auto support=query_exact_lift_support(owner->cells,owner->source,owner->target,UINT64_MAX);if(support.status!=RNS8_SUCCESS)return support.status;}
    if(needed>UINT64_MAX-4 || needed+4>options->max_total_scratch_bytes)return RNS8_WORKSPACE_TOO_SMALL;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    hip_scoped_device selected(ctx->device_id);if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    if(needed){const auto status=owner->scratch.allocate(ctx->device_id,static_cast<size_t>(needed));if(status!=RNS8_SUCCESS)return status;}
    if(owner->host_status.ensure(ctx->device_id,4)!=hipSuccess || hipStreamCreateWithFlags(&owner->owned_stream,hipStreamNonBlocking)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    return create_transfer_prepared(std::move(owner),out);
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
