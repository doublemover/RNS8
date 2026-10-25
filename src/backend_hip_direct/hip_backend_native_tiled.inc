#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_conversion(const void*,const void*,int8_t*,int,int,int,int,int,int,int,int,int,int,int,int,bool);
#endif
static rns8_status validate_native_tiled_task(const NativeWmmaTask& t,const NativeTiledConfig& config) {
  const auto resources=query_native_tiled_resources(config.depth,config.buffers,config.lds_budget,
      config.conversion,t.group,config.waves);
  if(resources.status!=RNS8_SUCCESS)return resources.status;
  if(!t.a || !t.b || !t.c || t.m<=0 || t.n<=0 || t.k<=0 || t.lda<t.k || t.ldb<t.n || t.ldc<t.n ||
      !t.prefix || t.prefix>20 || t.unsigned_inputs>1 || t.encoded_b || t.encoded_k_words)return RNS8_INVALID_ARGUMENT;
  if(!qualified_i32_matrix_address(t.m,t.k,t.lda,1,8) || !qualified_i32_matrix_address(t.k,t.n,t.ldb,1,8) ||
      !qualified_i32_matrix_address(t.m,t.n,t.ldc,t.prefix))return RNS8_UNSUPPORTED_BACKEND;
  AddressDomain a,b,c;
  if(!checked_host_plane_address(t.m,t.k,t.lda,1,8,a) || !checked_host_plane_address(t.k,t.n,t.ldb,1,8,b) ||
      !checked_host_plane_address(t.m,t.n,t.ldc,t.prefix,1,c))return RNS8_INVALID_ARGUMENT;
  if(!nonoverlapping_byte_ranges(t.c,c.byte_extent,t.a,a.byte_extent) ||
      !nonoverlapping_byte_ranges(t.c,c.byte_extent,t.b,b.byte_extent))return RNS8_INVALID_ARGUMENT;
  return RNS8_SUCCESS;
}
rns8_status NativeTiledPrepared::prepare(int device_id,const NativeWmmaTask& task,const NativeTiledConfig& config) {
  std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
  if(!lock.owns_lock())return RNS8_BUSY;
  auto status=validate_native_tiled_task(task,config);
  if(status!=RNS8_SUCCESS)return status;
  if(device_id<0)return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if((status=set_hip_device(device_id))!=RNS8_SUCCESS)return status;
  hipDeviceProp_t properties{};
  if(hipGetDeviceProperties(&properties,device_id)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  const auto resources=query_native_tiled_resources(config.depth,config.buffers,properties.sharedMemPerBlock,
      config.conversion,task.group,config.waves);
  if(resources.status!=RNS8_SUCCESS)return resources.status;
  hip_direct_device_buffer capability;
  if((status=capability.allocate(device_id,sizeof(int)))!=RNS8_SUCCESS)return status;
  const auto error=timed_hip_operation("native_conversion_prepare_capability",[&]() {
    const auto launch=static_cast<hipError_t>(rns8_hip_native_wmma_capability(static_cast<int*>(capability.get())));
    // Drain even on launch failure before local ownership can retire.
    const auto completion=hipDeviceSynchronize();
    return launch==hipSuccess ? completion : launch;
  });
  if(error!=hipSuccess)return RNS8_BACKEND_FAILURE;
  int supported=0;
  if((status=hip_direct_copy_device_to_host(device_id,&supported,capability.get(),sizeof(int)))!=RNS8_SUCCESS)return status;
  if(!supported)return RNS8_UNSUPPORTED_BACKEND;
  capability_lifetime_=std::move(capability);
  task_=task;config_=config;device_id_=device_id;ready_=true;poisoned_=false;
  return RNS8_SUCCESS;
#else
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status NativeTiledPrepared::execute(bool& outputs_may_be_written) {
  outputs_may_be_written=false;
  std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
  if(!lock.owns_lock())return RNS8_BUSY;
  if(!ready_ || poisoned_)return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id_);
  if(status!=RNS8_SUCCESS)return status;
  const char* label=config_.conversion==NativeConversion::cooperative ? "native_cooperative_encode_once_gemm" :
      config_.conversion==NativeConversion::encoded_lds ? "native_encoded_lds_once_gemm" :
      config_.waves==1 ? "native_prepared_one_wave_gemm" : config_.buffers==0 ? "native_tiled_register_gemm" :
      config_.buffers==1 ? "native_tiled_one_buffer_gemm" : "native_tiled_two_buffer_gemm";
  outputs_may_be_written=true;
  const auto error=timed_hip_operation(label,[&]() {
    const auto& t=task_;const auto& cfg=config_;
    const auto launch=static_cast<hipError_t>(rns8_hip_native_conversion(t.a,t.b,t.c,t.m,t.n,t.k,t.lda,t.ldb,t.ldc,
        t.prefix,t.group,cfg.depth,cfg.buffers,cfg.waves,static_cast<int>(cfg.conversion),t.unsigned_inputs!=0));
    const auto completion=hipDeviceSynchronize();
    return launch==hipSuccess ? completion : launch;
  });
  if(error!=hipSuccess){poisoned_=true;return RNS8_BACKEND_FAILURE;}
  return RNS8_SUCCESS;
#else
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_gemm_native_conversion_device(int device_id,const NativeWmmaTask& task,
    const NativeTiledConfig& config,bool& outputs_may_be_written) {
  outputs_may_be_written=false;
  NativeTiledPrepared prepared;
  const auto status=prepared.prepare(device_id,task,config);
  return status==RNS8_SUCCESS ? prepared.execute(outputs_may_be_written) : status;
}
rns8_status hip_direct_gemm_native_tiled_device(
    int device_id,const void* a,const void* b,int8_t* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint32_t group,bool unsigned_inputs,
    uint32_t depth,uint32_t buffers,uint64_t max_lds_bytes,bool& commit_started) {
  return hip_direct_gemm_native_conversion_device(device_id,
      NativeWmmaTask{a,b,c,m,n,k,lda,ldb,ldc,prefix,group,unsigned_inputs?1u:0u},
      NativeTiledConfig{depth,buffers,4,NativeConversion::replicated,max_lds_bytes},commit_started);
}
