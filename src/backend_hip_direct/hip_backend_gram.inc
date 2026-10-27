#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_gram_native(const void*,void*,const GramTile*,int,int,int,int,int,int,int,int,int);
extern "C" int rns8_hip_gram_transpose(const void*,void*,int,int,int);
extern "C" int rns8_hip_gram_export_once(const void*,void*,const GramTile*,int,int,int,int,int*);
extern "C" int rns8_hip_gram(const int8_t*,int8_t*,const GramTile*,int,int,int,int,int,int,int,int,int,int);
#endif
rns8_status hip_direct_gram_device(int device_id,const void* x,void* c,const GramProblem& p,
    uint64_t source_instance,uint64_t source_generation,bool wmma,uint64_t budget,bool& commit_started,bool separate_mirror) {
  commit_started=false;
  const auto support=query_gram_support(p,budget,true,wmma);
  if(support.status!=RNS8_SUCCESS) return support.status;
  if(p.source_instance!=source_instance || p.source_generation!=source_generation ||
     !nonoverlapping_byte_ranges(x,support.source_bytes,c,support.output_bytes)) return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  const auto jobs=gram_tiles(support);
  hip_probe_buffer device_jobs;
  status=device_jobs.allocate(device_id,support.job_bytes+(wmma?4:0));if(status!=RNS8_SUCCESS) return status;
  device_jobs.mark_pending();
  status=hip_direct_copy_host_to_device(device_id,device_jobs.get(),jobs.data(),support.job_bytes);
  if(status!=RNS8_SUCCESS){if(hipDeviceSynchronize()==hipSuccess)device_jobs.mark_completed();return status;}
  device_jobs.mark_completed();
  if(wmma) {
    auto* flag=reinterpret_cast<int*>(static_cast<uint8_t*>(device_jobs.get())+support.job_bytes);
    device_jobs.mark_pending();
    const auto launch=static_cast<hipError_t>(rns8_hip_native_wmma_capability(flag));
    const auto drain=hipDeviceSynchronize();if(drain==hipSuccess)device_jobs.mark_completed();
    if(launch!=hipSuccess || drain!=hipSuccess) return RNS8_BACKEND_FAILURE;
    int supported=0;status=hip_direct_copy_device_to_host(device_id,&supported,flag,4);
    if(status!=RNS8_SUCCESS) return status;
    if(!supported) return RNS8_UNSUPPORTED_BACKEND;
  }
  commit_started=true;device_jobs.mark_pending();
  const auto error=timed_hip_operation(wmma?"gram_triangular_wmma":"gram_triangular_scalar",[&]() {
    const auto launch=static_cast<hipError_t>(rns8_hip_gram(static_cast<const int8_t*>(x),static_cast<int8_t*>(c),
      static_cast<const GramTile*>(device_jobs.get()),int(support.tile_count),int(p.rows),int(p.cols),
      int(p.source_ld),int(p.output_ld),int(p.prefix),int(p.modulus),
      p.orientation==GramOrientation::transpose_times_self,wmma,separate_mirror));
    const auto drain=hipDeviceSynchronize();if(drain==hipSuccess)device_jobs.mark_completed();return launch==hipSuccess?drain:launch;
  });
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gram_prepared_device(int device_id,const void* x,void* c,const GramProblem& p,
    const GramTile* host_jobs,void* device_jobs,uint64_t device_job_bytes,bool wmma,
    bool& jobs_ready,bool& commit_started,bool native_i64,bool separate_mirror) {
  commit_started=false;const auto support=query_gram_support(p,UINT64_MAX,true,wmma);
  if(support.status!=RNS8_SUCCESS) return support.status;
  if(native_i64 && (p.modulus || !qualified_i32_matrix_address(p.rows,p.cols,p.source_ld,1,8)))return RNS8_INVALID_ARGUMENT;
  const uint64_t source_bytes=native_i64?uint64_t(p.rows)*p.source_ld*8:support.source_bytes;
  const uint64_t required=support.job_bytes+(wmma?4:0);
  if(!host_jobs || !device_jobs || device_job_bytes<required) return RNS8_INVALID_ARGUMENT;
  if(!nonoverlapping_byte_ranges(x,source_bytes,c,support.output_bytes) ||
     !nonoverlapping_byte_ranges(device_jobs,required,x,source_bytes) ||
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
  const auto error=timed_hip_operation(native_i64?(wmma?"gram_native_wmma":"gram_native_scalar"):(wmma?"gram_prepared_wmma":"gram_prepared_scalar"),[&]() {
    const auto launch=static_cast<hipError_t>(native_i64?rns8_hip_gram_native(x,c,static_cast<const GramTile*>(device_jobs),int(support.tile_count),int(p.rows),int(p.cols),int(p.source_ld),int(p.output_ld),int(p.prefix),p.orientation==GramOrientation::transpose_times_self,wmma,separate_mirror):rns8_hip_gram(static_cast<const int8_t*>(x),static_cast<int8_t*>(c),
        static_cast<const GramTile*>(device_jobs),int(support.tile_count),int(p.rows),int(p.cols),
        int(p.source_ld),int(p.output_ld),int(p.prefix),int(p.modulus),
        p.orientation==GramOrientation::transpose_times_self,wmma,separate_mirror));
    const auto drain=hipDeviceSynchronize();completed=drain==hipSuccess;return launch==hipSuccess?drain:launch;
  });
  if(!completed) completed=hipDeviceSynchronize()==hipSuccess;
  return error==hipSuccess && completed?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)jobs_ready;return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gram_transpose(int device_id,const void* source,void* target,const GramProblem& p){
  if(!source || !target || !qualified_i32_matrix_address(p.rows,p.cols,p.cols,p.prefix) ||
     !nonoverlapping_byte_ranges(source,uint64_t(p.rows)*p.cols*p.prefix,target,uint64_t(p.rows)*p.cols*p.prefix))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS)return status;
  const auto error=timed_hip_operation("gram_orientation_adapter",[&](){const auto launch=static_cast<hipError_t>(rns8_hip_gram_transpose(source,target,int(p.rows),int(p.cols),int(p.prefix)));const auto drain=hipDeviceSynchronize();return launch==hipSuccess?drain:launch;});return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_gram_export_once(int device_id,const void* residues,void* staged,int* status,
    const void* jobs,int64_t order,uint32_t prefix,uint32_t limbs,uint64_t count){
  if(!residues || !staged || !status || !jobs || !prefix || prefix>20 || !limbs || limbs>32 || !count || count>INT32_MAX ||
     !qualified_i32_matrix_address(order,order,order,prefix) || !qualified_i32_matrix_address(order,order,order,1,uint64_t(limbs)*8))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto selected=set_hip_device(device_id);if(selected!=RNS8_SUCCESS)return selected;
  const auto error=timed_hip_operation("gram_reconstruct_once_mirror",[&](){
    auto issued=hipMemsetAsync(status,0,4,nullptr);
    if(issued==hipSuccess)issued=static_cast<hipError_t>(rns8_hip_gram_export_once(residues,staged,static_cast<const GramTile*>(jobs),int(count),int(order),int(prefix),int(limbs),status));
    const auto drain=hipDeviceSynchronize();return issued==hipSuccess?drain:issued;});
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_gram_five_transform(const int8_t*,int8_t*,int8_t*,GramFivePacket,int);
extern "C" int rns8_hip_gram_five_finish(const int8_t*,const int8_t*,int8_t*,GramFivePacket);
#endif
rns8_status hip_direct_gram_five_product(int device,const void* source,void* output,const GramFiveResources& r,
    void* scratch,uint64_t bytes,bool matrix,bool& commit_started) {
  commit_started=false;
  if(r.status!=RNS8_SUCCESS)return r.status;
  const auto& packet=r.packet;
  const uint64_t source_bytes=uint64_t(packet.rows)*packet.ld*packet.prefix;
  if(!source || !output || !scratch || bytes<r.total ||
      !nonoverlapping_byte_ranges(source,source_bytes,output,r.output) ||
      !nonoverlapping_byte_ranges(source,source_bytes,scratch,r.total) ||
      !nonoverlapping_byte_ranges(output,r.output,scratch,r.total))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device);if(status!=RNS8_SUCCESS)return status;
  if(matrix && !amdgpu_builtins_compiled())return RNS8_UNSUPPORTED_BACKEND;
  const auto& p=r.packet;auto* left=static_cast<int8_t*>(scratch);auto* right=left+r.panel;
  auto* products=left+r.products_offset;auto* stage=left+r.output_offset;
  for(int term=0;term<5;++term) {
    const auto transform=timed_hip_operation("gram_five_skew_transform",[&]() {
      const auto issued=static_cast<hipError_t>(rns8_hip_gram_five_transform(static_cast<const int8_t*>(source),left,right,p,term));
      const auto drained=hipDeviceSynchronize();return issued==hipSuccess?drained:issued;
    });
    if(transform!=hipSuccess)return RNS8_BACKEND_FAILURE;
    auto* destination=products+uint64_t(term)*r.product;
    status=matrix?(p.prefix==1?amdgpu_builtins_gemm_finite_u8_device(device,left,right,destination,p.h,p.h,p.hk,p.hk,p.h,p.h,p.moduli[0]):
        amdgpu_builtins_gemm_rns_device(device,left,right,destination,p.h,p.h,p.hk,p.hk,p.h,p.h,p.prefix)):
        hip_direct_basis_gemm(device,left,right,destination,p.h,p.h,p.hk,p.moduli,p.prefix);
    if(status!=RNS8_SUCCESS)return status;
  }
  const auto finish=timed_hip_operation("gram_five_recombine_and_classical_channels",[&]() {
    const auto issued=static_cast<hipError_t>(rns8_hip_gram_five_finish(static_cast<const int8_t*>(source),products,stage,p));
    const auto drained=hipDeviceSynchronize();return issued==hipSuccess?drained:issued;
  });
  if(finish!=hipSuccess)return RNS8_BACKEND_FAILURE;
  commit_started=true;return hip_direct_copy_device_to_device(device,output,stage,r.output);
#else
  (void)device;(void)matrix;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
