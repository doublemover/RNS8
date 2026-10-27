#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_wmma_capability(int*);
extern "C" int rns8_hip_instruction_roof(const int8_t*,const int8_t*,int32_t*,int,int,int,bool);
#endif
rns8_status hip_direct_instruction_roof(int device_id,const void* a,const void* b,void* output,
    void* capability,const InstructionRoofProblem& p) {
  const auto resources=query_instruction_roof(p);
  if(resources.status!=RNS8_SUCCESS) return resources.status;
  if(!nonoverlapping_byte_ranges(a,resources.input_bytes,output,resources.output_bytes) ||
     !nonoverlapping_byte_ranges(b,resources.input_bytes,output,resources.output_bytes) ||
     !nonoverlapping_byte_ranges(capability,4,output,resources.output_bytes) ||
     !nonoverlapping_byte_ranges(capability,4,a,resources.input_bytes) ||
     !nonoverlapping_byte_ranges(capability,4,b,resources.input_bytes) ||
     reinterpret_cast<uintptr_t>(output)%alignof(int32_t) ||
     reinterpret_cast<uintptr_t>(capability)%alignof(int32_t)) return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  const auto probe=static_cast<hipError_t>(rns8_hip_native_wmma_capability(static_cast<int*>(capability)));
  const auto drain=hipDeviceSynchronize();
  if(probe!=hipSuccess || drain!=hipSuccess) return RNS8_BACKEND_FAILURE;
  int supported=0;status=hip_direct_copy_device_to_host(device_id,&supported,capability,4);
  if(status!=RNS8_SUCCESS) return status;if(!supported) return RNS8_UNSUPPORTED_BACKEND;
  bool completed=false;
  const auto error=timed_hip_operation(p.reduce_each_iteration?"instruction_roof_wmma_reduce251":"instruction_roof_wmma_raw",[&]() {
    const auto launch=static_cast<hipError_t>(rns8_hip_instruction_roof(static_cast<const int8_t*>(a),
        static_cast<const int8_t*>(b),static_cast<int32_t*>(output),p.chains,p.iterations,p.waves,p.reduce_each_iteration));
    const auto done=hipDeviceSynchronize();completed=done==hipSuccess;return launch==hipSuccess?done:launch;
  });
  if(!completed) completed=hipDeviceSynchronize()==hipSuccess;
  return error==hipSuccess && completed?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
