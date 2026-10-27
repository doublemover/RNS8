#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_tdm_capability(int*);
extern "C" int rns8_hip_native_tdm(const uint64_t*,const uint64_t*,int8_t*,NativeTdmProblem,unsigned*,void*);
extern "C" int rns8_hip_native_tdm_litmus(const uint64_t*,uint64_t*,unsigned,unsigned,unsigned,unsigned,unsigned*);
#endif
rns8_status hip_direct_prepare_native_tdm(int device,const NativeTdmProblem& p,void* probe,uint32_t& flags) {
  flags=0;if(!probe || query_native_tdm(p).status!=RNS8_SUCCESS)return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device);if(status!=RNS8_SUCCESS)return status;
  const auto launch=hipError_t(rns8_hip_native_tdm_capability(static_cast<int*>(probe)));
  const auto drained=hipDeviceSynchronize();if(launch!=hipSuccess || drained!=hipSuccess)return RNS8_BACKEND_FAILURE;
  if(hipMemcpy(&flags,probe,4,hipMemcpyDeviceToHost)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  if((p.mode==1 && !(flags&1)) || (p.mode==2 && (flags&3)!=3) || (p.compute && !(flags&4)))return RNS8_UNSUPPORTED_BACKEND;
  return RNS8_SUCCESS;
#else
  (void)device;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_native_tdm_litmus(int device,const void* input,void* output,void* status,
    uint32_t rows,uint32_t cols,uint32_t ld,uint32_t cluster,uint32_t prepared_flags) {
  AddressDomain span;
  if(!rows || rows>16 || !cols || cols>32 || (cluster!=1 && cluster!=2 && cluster!=4) ||
      !checked_host_plane_address(rows,cols,ld,1,8,span) || !input || !output || !status ||
      reinterpret_cast<uintptr_t>(input)%8 || reinterpret_cast<uintptr_t>(output)%8 || reinterpret_cast<uintptr_t>(status)%4 ||
      !nonoverlapping_byte_ranges(input,span.byte_extent,output,uint64_t(cluster)*512*8) ||
      !nonoverlapping_byte_ranges(status,4,input,span.byte_extent) || !nonoverlapping_byte_ranges(status,4,output,uint64_t(cluster)*512*8))return RNS8_INVALID_ARGUMENT;
  if(!(prepared_flags&1) || (cluster>1 && !(prepared_flags&2)))return RNS8_UNSUPPORTED_BACKEND;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto result=set_hip_device(device);if(result!=RNS8_SUCCESS)return result;
  auto error=hipMemset(status,0,4);if(error==hipSuccess)error=hipError_t(rns8_hip_native_tdm_litmus(
      static_cast<const uint64_t*>(input),static_cast<uint64_t*>(output),rows,cols,ld,cluster,static_cast<unsigned*>(status)));
  const auto drained=hipDeviceSynchronize();if(error!=hipSuccess || drained!=hipSuccess)return RNS8_BACKEND_FAILURE;
  unsigned failure=0;if(hipMemcpy(&failure,status,4,hipMemcpyDeviceToHost)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  return failure?RNS8_RANGE_ERROR:RNS8_SUCCESS;
#else
  (void)device;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_native_tdm_admitted(int device,const void* a,const void* b,void* c,void* status,
    const NativeTdmProblem& p,uint32_t prepared_flags) {
  const auto resources=query_native_tdm(p);
  if(resources.status!=RNS8_SUCCESS || !a || !b || !c || !status ||
      reinterpret_cast<uintptr_t>(a)%8 || reinterpret_cast<uintptr_t>(b)%8 || reinterpret_cast<uintptr_t>(status)%4 ||
      !nonoverlapping_byte_ranges(a,resources.native_a_bytes,c,resources.output_bytes) ||
      !nonoverlapping_byte_ranges(b,resources.native_b_bytes,c,resources.output_bytes) ||
      !nonoverlapping_byte_ranges(status,4,a,resources.native_a_bytes) || !nonoverlapping_byte_ranges(status,4,b,resources.native_b_bytes) ||
      !nonoverlapping_byte_ranges(status,4,c,resources.output_bytes))return RNS8_INVALID_ARGUMENT;
  if((p.mode==1 && !(prepared_flags&1)) || (p.mode==2 && (prepared_flags&3)!=3) || (p.compute && !(prepared_flags&4)))return RNS8_UNSUPPORTED_BACKEND;
  if(p.mode && (reinterpret_cast<uintptr_t>(a)>=(uint64_t{1}<<57) || reinterpret_cast<uintptr_t>(b)>=(uint64_t{1}<<57) ||
      resources.native_a_bytes>(uint64_t{1}<<57)-reinterpret_cast<uintptr_t>(a) ||
      resources.native_b_bytes>(uint64_t{1}<<57)-reinterpret_cast<uintptr_t>(b)))return RNS8_UNSUPPORTED_BACKEND;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto result=set_hip_device(device);if(result!=RNS8_SUCCESS)return result;
  auto error=hipMemset(status,0,4);if(error!=hipSuccess)return RNS8_BACKEND_FAILURE;
  error=timed_hip_operation(p.mode==0?"native_raw_lds_control":p.mode==1?"native_two_stage_tdm":"native_two_stage_tdm_cluster_multicast",[&] {
    const auto launch=hipError_t(rns8_hip_native_tdm(static_cast<const uint64_t*>(a),static_cast<const uint64_t*>(b),
        static_cast<int8_t*>(c),p,static_cast<unsigned*>(status),nullptr));
    const auto drained=hipDeviceSynchronize();return launch==hipSuccess?drained:launch;
  });
  if(error!=hipSuccess)return RNS8_BACKEND_FAILURE;
  unsigned failure=0;if(hipMemcpy(&failure,status,4,hipMemcpyDeviceToHost)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  return failure?RNS8_RANGE_ERROR:RNS8_SUCCESS;
#else
  (void)device;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
