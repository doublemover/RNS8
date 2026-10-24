#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_fixed_conversion_capability(int*);
extern "C" int rns8_hip_fixed_conversion(const void*,void*,int,int,int);
#endif
rns8_status hip_direct_fixed_conversion_device(int device_id,const void* source,void* destination,
    int64_t cells,uint32_t prefix,uint32_t mode,uint64_t budget,bool& commit_started) {
  commit_started=false;
  if(!source || !destination || cells<=0 || !prefix || prefix>20 || mode>2) return RNS8_INVALID_ARGUMENT;
  const auto support=query_canonical_crt_support(cells,prefix);
  if(support.status!=RNS8_SUCCESS) return support.status;
  const uint64_t source_bytes=uint64_t(cells)*(mode==2?prefix:8),destination_bytes=uint64_t(cells)*(mode==2?24:prefix);
  if((mode!=2 && reinterpret_cast<uintptr_t>(source)%8) ||
     (mode==2 && reinterpret_cast<uintptr_t>(destination)%8) ||
     !nonoverlapping_byte_ranges(source,source_bytes,destination,destination_bytes)) return RNS8_INVALID_ARGUMENT;
  if(budget<4) return RNS8_WORKSPACE_TOO_SMALL;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  hip_direct_device_buffer flag;status=flag.allocate(device_id,4);if(status!=RNS8_SUCCESS) return status;
  const auto launched=static_cast<hipError_t>(rns8_hip_fixed_conversion_capability(static_cast<int*>(flag.get())));
  const auto completed=hipDeviceSynchronize();if(launched!=hipSuccess || completed!=hipSuccess) return RNS8_BACKEND_FAILURE;
  int supported=0;status=hip_direct_copy_device_to_host(device_id,&supported,flag.get(),4);
  if(status!=RNS8_SUCCESS) return status;if(!supported) return RNS8_UNSUPPORTED_BACKEND;
  commit_started=true;
  const auto error=timed_hip_operation(mode==2?"fixed_dot4_crt":"fixed_dot4_import",[&]() {
    const auto launch=static_cast<hipError_t>(rns8_hip_fixed_conversion(source,destination,int(cells),prefix,mode));
    const auto drain=hipDeviceSynchronize();return launch==hipSuccess?drain:launch;
  });
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
