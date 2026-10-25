#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_global_bound(const NativeTileBound*,uint64_t,NativeTileBound*,uint32_t,bool);
#endif
rns8_status hip_direct_gemm_global_minimum_native_device(int device_id,const void* a,const void* b,int8_t* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,bool unsigned_inputs,
    uint32_t capacity,uint32_t group,uint64_t budget,bool& commit_started,NativeGlobalMinimumReceipt& receipt) {
  commit_started=false;receipt={};
  if(!a || !b || !c)return RNS8_INVALID_ARGUMENT;
  const auto support=query_native_global_minimum_support(m,n,k,lda,ldb,ldc,capacity,group,budget);
  if(support.status!=RNS8_SUCCESS)return support.status;
  if(!nonoverlapping_byte_ranges(c,uint64_t(m)*ldc*capacity,a,uint64_t(m)*lda*8) ||
      !nonoverlapping_byte_ranges(c,uint64_t(m)*ldc*capacity,b,uint64_t(k)*ldb*8))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using Timer=std::chrono::steady_clock;const auto begin=Timer::now();
  const auto elapsed=[](auto start){return std::chrono::duration<double,std::nano>(Timer::now()-start).count();};
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS)return status;
  hip_direct_device_buffer stage,records,metadata,global,suffix;
  for(const auto& allocation:{std::pair<hip_direct_device_buffer*,uint64_t>{&stage,support.discovery.stage_bytes},
      {&records,support.discovery.bound_bytes},{&metadata,support.discovery.metadata_bytes},{&global,sizeof(NativeTileBound)}})
    if((status=allocation.first->allocate(device_id,size_t(allocation.second)))!=RNS8_SUCCESS)return status;
  auto* bounds=static_cast<NativeTileBound*>(records.get());auto* indices=static_cast<uint32_t*>(metadata.get());
  records.mark_pending();metadata.mark_pending();
  status=hip_direct_discover_native_tile_bounds(device_id,a,b,m,n,k,lda,ldb,unsigned_inputs,capacity,bounds,indices,
      indices+support.discovery.tiles,support.discovery.discovery_bytes,NativeBoundMethod::axis_l1,256);
  if(status!=RNS8_SUCCESS) {
    if(status==RNS8_RANGE_ERROR){records.mark_completed();metadata.mark_completed();}
    return status;
  }
  records.mark_completed();metadata.mark_completed();
  records.mark_pending();global.mark_pending();
  const auto reduced=timed_hip_operation("native_global_bound_reduce",[&] {
    const auto launched=static_cast<hipError_t>(rns8_hip_native_global_bound(bounds,support.discovery.tiles,
        static_cast<NativeTileBound*>(global.get()),capacity,!unsigned_inputs));
    return launched==hipSuccess?hipDeviceSynchronize():launched;
  });
  if(reduced!=hipSuccess)return RNS8_BACKEND_FAILURE;records.mark_completed();global.mark_completed();
  global.mark_pending();
  status=hip_direct_copy_device_to_host(device_id,&receipt.bound,global.get(),sizeof(receipt.bound));if(status!=RNS8_SUCCESS)return status;
  global.mark_completed();
  auto checked=receipt.bound;select_native_bound_prefix(checked,!unsigned_inputs,capacity,garner::kConstants.products);
  if(checked.required_prefix!=receipt.bound.required_prefix || checked.required_prefix>capacity)return RNS8_RANGE_ERROR;
  receipt.required_prefix=checked.required_prefix;receipt.compute_prefix=std::max(1u,checked.required_prefix);
  // Only the small host certificate survives discovery. Retiring its device
  // buffers before production avoids charging two disjoint scratch lifetimes.
  for(auto* allocation:{&records,&metadata,&global})if((status=allocation->reset())!=RNS8_SUCCESS)return status;
  receipt.proof_readback_bytes=sizeof(NativeTileBound)+sizeof(uint32_t);receipt.discovery_ns=elapsed(begin);
  const auto product_start=Timer::now();stage.mark_pending();
  status=hip_direct_gemm_native_wmma_grouped_device(device_id,a,b,static_cast<int8_t*>(stage.get()),m,n,k,lda,ldb,n,
      receipt.compute_prefix,group,unsigned_inputs);
  if(status!=RNS8_SUCCESS){if(status==RNS8_UNSUPPORTED_BACKEND)stage.mark_completed();return status;}stage.mark_completed();receipt.conversion_product_ns=elapsed(product_start);
  const auto completion_start=Timer::now();
  if(receipt.compute_prefix<capacity) {
    const auto lift=query_exact_lift_support(m*n,receipt.compute_prefix,capacity,UINT64_MAX);
    if(lift.status!=RNS8_SUCCESS)return lift.status;
    if((status=suffix.allocate(device_id,size_t(lift.scratch_bytes)))!=RNS8_SUCCESS)return status;
    bool suffix_started=false;stage.mark_pending();suffix.mark_pending();
    status=hip_direct_lift_exact_suffix_workspace(device_id,stage.get(),m*n,receipt.compute_prefix,capacity,!unsigned_inputs,
        receipt.bound.magnitude,suffix.get(),suffix.bytes(),suffix_started);
    if(status!=RNS8_SUCCESS)return status;stage.mark_completed();suffix.mark_completed();
  }
  receipt.completion_ns=elapsed(completion_start);
  const auto commit_start=Timer::now();commit_started=true;stage.mark_pending();
  const auto committed=timed_hip_operation("native_global_completed_commit",[&] {
    const auto launched=static_cast<hipError_t>(rns8_hip_strassen_commit(static_cast<const int8_t*>(stage.get()),c,int(m),int(n),int(ldc),capacity));
    return launched==hipSuccess?hipDeviceSynchronize():launched;
  });
  if(committed!=hipSuccess)return RNS8_BACKEND_FAILURE;stage.mark_completed();receipt.commit_ns=elapsed(commit_start);
  const auto cleanup_start=Timer::now();
  for(auto* allocation:{&stage,&records,&metadata,&global,&suffix})if((status=allocation->reset())!=RNS8_SUCCESS)return status;
  receipt.cleanup_ns=elapsed(cleanup_start);return RNS8_SUCCESS;
#else
  (void)device_id;(void)unsigned_inputs;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
