#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_tiled(const void*,const void*,int8_t*,int,int,int,int,int,int,int,int,int,int,bool);
#endif
rns8_status hip_direct_gemm_native_tiled_device(
    int device_id,const void* a,const void* b,int8_t* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint32_t group,bool unsigned_inputs,
    uint32_t depth,uint32_t buffers,uint64_t max_lds_bytes,bool& commit_started) {
  commit_started=false;
  const auto resources=query_native_tiled_resources(depth,buffers,max_lds_bytes);
  if (resources.status!=RNS8_SUCCESS) return resources.status;
  if (!a || !b || !c || m<=0 || n<=0 || k<=0 || lda<k || ldb<n || ldc<n ||
      !prefix || prefix>20 || (group!=1 && group!=2 && group!=4)) return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_matrix_address(m,k,lda,1,8) || !qualified_i32_matrix_address(k,n,ldb,1,8) ||
      !qualified_i32_matrix_address(m,n,ldc,prefix)) return RNS8_UNSUPPORTED_BACKEND;
  if (!nonoverlapping_byte_ranges(c,uint64_t(m)*ldc*prefix,a,uint64_t(m)*lda*8) ||
      !nonoverlapping_byte_ranges(c,uint64_t(m)*ldc*prefix,b,uint64_t(k)*ldb*8)) return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);
  if (status!=RNS8_SUCCESS) return status;
  hip_direct_device_buffer capability;
  if ((status=capability.allocate(device_id,sizeof(int)))!=RNS8_SUCCESS) return status;
  const auto probe=timed_hip_operation("native_tiled_device_capability",[&]() {
    auto error=static_cast<hipError_t>(rns8_hip_native_wmma_capability(static_cast<int*>(capability.get())));
    return error==hipSuccess ? hipDeviceSynchronize() : error;
  });
  if (probe!=hipSuccess) return RNS8_BACKEND_FAILURE;
  int supported=0;
  status=hip_direct_copy_device_to_host(device_id,&supported,capability.get(),sizeof(supported));
  if (status!=RNS8_SUCCESS) return status;
  if (!supported) return RNS8_UNSUPPORTED_BACKEND;
  commit_started=true;
  const auto error=timed_hip_operation(buffers==0 ? "native_tiled_register_gemm" :
      buffers==1 ? "native_tiled_one_buffer_gemm" : "native_tiled_two_buffer_gemm",[&]() {
    auto launch=static_cast<hipError_t>(rns8_hip_native_tiled(a,b,c,m,n,k,lda,ldb,ldc,prefix,group,depth,buffers,unsigned_inputs));
    return launch==hipSuccess ? hipDeviceSynchronize() : launch;
  });
  return error==hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)unsigned_inputs;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
