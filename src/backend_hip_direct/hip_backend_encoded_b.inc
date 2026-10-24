#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_encode_native_b(const void*,void*,int,int,int,int,uint64_t,bool);
extern "C" int rns8_hip_native_encoded_b_wmma(const void*,const void*,void*,int,int,int,int,int,int,uint64_t,int,bool);
#endif
rns8_status hip_prepare_encoded_b(int device,const void* source,void* storage,
    int64_t k,int64_t n,int64_t ldb,uint32_t prefix,bool unsigned_inputs,uint64_t bytes) {
  if(!source || !storage || reinterpret_cast<uintptr_t>(storage)%4) return RNS8_INVALID_ARGUMENT;
  const auto layout=encoded_b_layout(k,n,ldb,prefix,bytes);
  if(layout.status!=RNS8_SUCCESS) return layout.status;
  if(!nonoverlapping_byte_ranges(storage,layout.total_bytes,source,uint64_t(k)*uint64_t(ldb)*8))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device); if(status!=RNS8_SUCCESS) return status;
  auto* capability=reinterpret_cast<int*>(static_cast<int8_t*>(storage)+layout.payload_bytes);
  const auto probe=timed_hip_operation("encoded_b_capability",[&]() {
    const auto error=static_cast<hipError_t>(rns8_hip_native_wmma_capability(capability));
    return error==hipSuccess?hipDeviceSynchronize():error;
  });
  if(probe!=hipSuccess) return RNS8_BACKEND_FAILURE;
  int supported=0;
  status=hip_direct_copy_device_to_host(device,&supported,capability,sizeof(supported));
  if(status!=RNS8_SUCCESS) return status;
  if(!supported) return RNS8_UNSUPPORTED_BACKEND;
  const auto result=timed_hip_operation("encode_native_b_k_contiguous",[&]() {
    const auto error=static_cast<hipError_t>(rns8_hip_encode_native_b(source,storage,int(k),int(n),int(ldb),
        int(prefix),layout.k_words,unsigned_inputs));
    return error==hipSuccess?hipDeviceSynchronize():error;
  });
  return result==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device; (void)unsigned_inputs;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_gemm_encoded_b(int device,const void* a,const void* encoded,void* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldc,uint32_t prefix,uint32_t group,
    bool unsigned_inputs,bool& commit_started) {
  commit_started=false;
  if(!a || !encoded || !c || m<=0 || lda<k || ldc<n || reinterpret_cast<uintptr_t>(encoded)%4 ||
      (group!=1 && group!=2 && group!=4)) return RNS8_INVALID_ARGUMENT;
  const auto layout=encoded_b_layout(k,n,n,prefix,UINT64_MAX);
  if(layout.status!=RNS8_SUCCESS) return layout.status;
  if(!qualified_i32_matrix_address(m,k,lda,1,8) || !qualified_i32_matrix_address(m,n,ldc,prefix))
    return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t bytes=uint64_t(m)*uint64_t(ldc)*prefix;
  if(!nonoverlapping_byte_ranges(c,bytes,a,uint64_t(m)*uint64_t(lda)*8) ||
      !nonoverlapping_byte_ranges(c,bytes,encoded,layout.total_bytes)) return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device); if(status!=RNS8_SUCCESS) return status;
  const auto result=timed_hip_operation("native_encoded_b_wmma_gemm",[&]() {
    commit_started=true;
    const auto error=static_cast<hipError_t>(rns8_hip_native_encoded_b_wmma(a,encoded,c,int(m),int(n),int(k),
        int(lda),int(ldc),int(prefix),layout.k_words,int(group),unsigned_inputs));
    return error==hipSuccess?hipDeviceSynchronize():error;
  });
  return result==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device; (void)unsigned_inputs;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
