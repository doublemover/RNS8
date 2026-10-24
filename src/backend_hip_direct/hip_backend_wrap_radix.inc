#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_wrap_radix_wmma(const uint8_t*,const uint8_t*,uint8_t*,int,int,int,int);
#endif
rns8_status hip_direct_wrap_radix_wmma_device(int device_id,const void* a,const void* b,void* c,
    int64_t m,int64_t n,int64_t k,bool signed_digits,uint64_t max_scratch_bytes,bool& commit_started) {
  commit_started=false;
  if(!a || !b || !c || m<=0 || n<=0 || k<=0) return RNS8_INVALID_ARGUMENT;
  if(!qualified_i32_matrix_address(m,k,k,1,8) || !qualified_i32_matrix_address(k,n,n,1,8) ||
     !qualified_i32_matrix_address(m,n,n,1,8)) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t ab=uint64_t(m)*k*8,bb=uint64_t(k)*n*8,cb=uint64_t(m)*n*8;
  if(!nonoverlapping_byte_ranges(c,cb,a,ab) || !nonoverlapping_byte_ranges(c,cb,b,bb))
    return RNS8_INVALID_ARGUMENT;
  if(max_scratch_bytes<sizeof(int)) return RNS8_WORKSPACE_TOO_SMALL;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id); if(status!=RNS8_SUCCESS) return status;
  hip_direct_device_buffer capability;
  status=capability.allocate(device_id,sizeof(int)); if(status!=RNS8_SUCCESS) return status;
  const auto launched=static_cast<hipError_t>(rns8_hip_native_wmma_capability(static_cast<int*>(capability.get())));
  const auto completed=hipDeviceSynchronize();
  if(launched!=hipSuccess || completed!=hipSuccess) return RNS8_BACKEND_FAILURE;
  int supported=0;
  status=hip_direct_copy_device_to_host(device_id,&supported,capability.get(),sizeof(supported));
  if(status!=RNS8_SUCCESS) return status;
  if(!supported) return RNS8_UNSUPPORTED_BACKEND;
  commit_started=true;
  const auto result=timed_hip_operation(signed_digits?"wrap64_signed_radix_wmma":"wrap64_native_u8_wmma",[&]() {
    const auto launch=static_cast<hipError_t>(rns8_hip_wrap_radix_wmma(
      static_cast<const uint8_t*>(a),static_cast<const uint8_t*>(b),static_cast<uint8_t*>(c),
      int(m),int(n),int(k),signed_digits?1:0));
    const auto drain=hipDeviceSynchronize();
    return launch==hipSuccess?drain:launch;
  });
  return result==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)signed_digits;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
