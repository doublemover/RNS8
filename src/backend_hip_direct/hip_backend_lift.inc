rns8_status hip_direct_lift_exact_suffix_workspace(
    int device_id,void* residues,int64_t cells,uint32_t source_prefix,uint32_t target_prefix,
    bool signed_value,const uint64_t* bound,void* scratch,uint64_t scratch_bytes,
    bool& commit_started,bool mixed_radix) {
  commit_started=false;
  if(!bound)return RNS8_INVALID_ARGUMENT;
  ExactLiftWorkspaceView view;
  const auto admitted=exact_lift_workspace_view(scratch,scratch_bytes,residues,cells,source_prefix,target_prefix,view);
  if(admitted!=RNS8_SUCCESS)return admitted;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto selected=set_hip_device(device_id);if(selected!=RNS8_SUCCESS)return selected;
  // Every replay resets and checks proof status. Retained capacity is not a
  // retained proof, and a prior failure must never poison the next status word.
  int host_status=0;
  if(hipMemset(view.status,0,sizeof(int))!=hipSuccess)return RNS8_BACKEND_FAILURE;
  const auto error=timed_hip_operation(mixed_radix?"exact_lift_mixed_radix_suffix_kernel":"exact_lift_suffix_kernel",[&](){
    const int code=rns8_hip_direct_lift_suffix_device(static_cast<const int8_t*>(residues),view.suffix,
        static_cast<int>(cells),static_cast<int>(source_prefix),static_cast<int>(target_prefix),signed_value,
        bound[0],bound[1],bound[2],view.status,mixed_radix,nullptr);
    return code==static_cast<int>(hipSuccess)?hipDeviceSynchronize():static_cast<hipError_t>(code);
  });
  if(error!=hipSuccess || tracked_sync_copy(&host_status,view.status,sizeof(int),hipMemcpyDeviceToHost)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  if(host_status)return host_status==2?RNS8_RANGE_ERROR:RNS8_INVALID_ARGUMENT;
  commit_started=true;
  auto* destination=static_cast<int8_t*>(residues)+static_cast<std::size_t>(cells)*source_prefix;
  return hip_direct_copy_device_to_device(device_id,destination,view.suffix,static_cast<std::size_t>(view.suffix_bytes));
#else
  (void)device_id;(void)signed_value;(void)mixed_radix;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_lift_exact_suffix(
    int device_id,void* residues,int64_t cells,uint32_t source_prefix,uint32_t target_prefix,
    bool signed_value,const uint64_t* bound,bool& commit_started,bool mixed_radix) {
  commit_started=false;if(!residues || !bound)return RNS8_INVALID_ARGUMENT;
  const auto support=query_exact_lift_support(cells,source_prefix,target_prefix,UINT64_MAX);
  if(support.status!=RNS8_SUCCESS)return support.status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_direct_device_buffer scratch;
  const auto allocated=scratch.allocate(device_id,static_cast<std::size_t>(support.scratch_bytes));
  if(allocated!=RNS8_SUCCESS)return allocated;
  return hip_direct_lift_exact_suffix_workspace(device_id,residues,cells,source_prefix,target_prefix,
      signed_value,bound,scratch.get(),support.scratch_bytes,commit_started,mixed_radix);
#else
  (void)device_id;(void)signed_value;(void)mixed_radix;return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_enqueue_exact_lift(int device_id,void* residues,int64_t cells,
    uint32_t source_prefix,uint32_t target_prefix,bool signed_value,const uint64_t* bound,
    void* scratch,uint64_t scratch_bytes,void* pinned_status,bool mixed_radix,void* stream) {
  if(!bound || !pinned_status || !stream)return RNS8_INVALID_ARGUMENT;
  ExactLiftWorkspaceView view;
  const auto admitted=exact_lift_workspace_view(scratch,scratch_bytes,residues,cells,source_prefix,target_prefix,view);
  if(admitted!=RNS8_SUCCESS)return admitted;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device_id)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  auto st=static_cast<hipStream_t>(stream);
  if(hipMemsetAsync(view.status,0,sizeof(int),st)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  const auto launched=static_cast<hipError_t>(rns8_hip_direct_lift_suffix_device(static_cast<const int8_t*>(residues),view.suffix,
      int(cells),int(source_prefix),int(target_prefix),signed_value,bound[0],bound[1],bound[2],view.status,mixed_radix,stream));
  if(launched!=hipSuccess)return RNS8_BACKEND_FAILURE;
  // Separate grid boundary makes the complete proof word visible before any
  // suffix publication. Invalid input never copies a partially staged suffix.
  auto* destination=static_cast<int8_t*>(residues)+uint64_t(cells)*source_prefix;
  if(rns8_hip_guarded_copy_bytes(view.suffix,destination,1,view.suffix_bytes,view.suffix_bytes,view.suffix_bytes,view.status,stream)!=int(hipSuccess))
    return RNS8_BACKEND_FAILURE;
  return hipMemcpyAsync(pinned_status,view.status,sizeof(int),hipMemcpyDeviceToHost,st)==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)signed_value;(void)mixed_radix;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
