#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_gram(const int8_t*,int8_t*,const GramTile*,int,int,int,int,int,int,int,int,int);
#endif
rns8_status hip_direct_gram_device(int device_id,const void* x,void* c,const GramProblem& p,
    uint64_t source_instance,uint64_t source_generation,bool wmma,uint64_t budget,bool& commit_started) {
  commit_started=false;
  const auto support=query_gram_support(p,budget,true,wmma);
  if(support.status!=RNS8_SUCCESS) return support.status;
  if(p.source_instance!=source_instance || p.source_generation!=source_generation ||
     !nonoverlapping_byte_ranges(x,support.source_bytes,c,support.output_bytes)) return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  const auto jobs=gram_tiles(support);
  hip_direct_device_buffer device_jobs;
  status=device_jobs.allocate(device_id,support.job_bytes+(wmma?4:0));if(status!=RNS8_SUCCESS) return status;
  status=hip_direct_copy_host_to_device(device_id,device_jobs.get(),jobs.data(),support.job_bytes);
  if(status!=RNS8_SUCCESS) return status;
  if(wmma) {
    auto* flag=reinterpret_cast<int*>(static_cast<uint8_t*>(device_jobs.get())+support.job_bytes);
    const auto launch=static_cast<hipError_t>(rns8_hip_native_wmma_capability(flag));
    const auto drain=hipDeviceSynchronize();
    if(launch!=hipSuccess || drain!=hipSuccess) return RNS8_BACKEND_FAILURE;
    int supported=0;status=hip_direct_copy_device_to_host(device_id,&supported,flag,4);
    if(status!=RNS8_SUCCESS) return status;
    if(!supported) return RNS8_UNSUPPORTED_BACKEND;
  }
  commit_started=true;
  const auto error=timed_hip_operation(wmma?"gram_triangular_wmma":"gram_triangular_scalar",[&]() {
    const auto launch=static_cast<hipError_t>(rns8_hip_gram(static_cast<const int8_t*>(x),static_cast<int8_t*>(c),
      static_cast<const GramTile*>(device_jobs.get()),int(support.tile_count),int(p.rows),int(p.cols),
      int(p.source_ld),int(p.output_ld),int(p.prefix),int(p.modulus),
      p.orientation==GramOrientation::transpose_times_self,wmma));
    const auto drain=hipDeviceSynchronize();return launch==hipSuccess?drain:launch;
  });
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gram_prepared_device(int device_id,const void* x,void* c,const GramProblem& p,
    const GramTile* host_jobs,void* device_jobs,uint64_t device_job_bytes,bool wmma,
    bool& jobs_ready,bool& commit_started) {
  commit_started=false;const auto support=query_gram_support(p,UINT64_MAX,true,wmma);
  if(support.status!=RNS8_SUCCESS) return support.status;
  const uint64_t required=support.job_bytes+(wmma?4:0);
  if(!host_jobs || !device_jobs || device_job_bytes<required) return RNS8_INVALID_ARGUMENT;
  if(!nonoverlapping_byte_ranges(x,support.source_bytes,c,support.output_bytes) ||
     !nonoverlapping_byte_ranges(device_jobs,required,x,support.source_bytes) ||
     !nonoverlapping_byte_ranges(device_jobs,required,c,support.output_bytes)) return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  if(!jobs_ready) {
    status=hip_direct_copy_host_to_device(device_id,device_jobs,host_jobs,support.job_bytes);
    if(status!=RNS8_SUCCESS) return status;
    if(wmma) {
      auto* flag=reinterpret_cast<int*>(static_cast<uint8_t*>(device_jobs)+support.job_bytes);
      const auto launch=static_cast<hipError_t>(rns8_hip_native_wmma_capability(flag));
      const auto drain=hipDeviceSynchronize();if(launch!=hipSuccess || drain!=hipSuccess) return RNS8_BACKEND_FAILURE;
      int supported=0;status=hip_direct_copy_device_to_host(device_id,&supported,flag,4);
      if(status!=RNS8_SUCCESS) return status;if(!supported) return RNS8_UNSUPPORTED_BACKEND;
    }
    jobs_ready=true;
  }
  commit_started=true;bool completed=false;
  const auto error=timed_hip_operation(wmma?"gram_prepared_wmma":"gram_prepared_scalar",[&]() {
    const auto launch=static_cast<hipError_t>(rns8_hip_gram(static_cast<const int8_t*>(x),static_cast<int8_t*>(c),
        static_cast<const GramTile*>(device_jobs),int(support.tile_count),int(p.rows),int(p.cols),
        int(p.source_ld),int(p.output_ld),int(p.prefix),int(p.modulus),
        p.orientation==GramOrientation::transpose_times_self,wmma));
    const auto drain=hipDeviceSynchronize();completed=drain==hipSuccess;return launch==hipSuccess?drain:launch;
  });
  if(!completed) completed=hipDeviceSynchronize()==hipSuccess;
  return error==hipSuccess && completed?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)jobs_ready;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
