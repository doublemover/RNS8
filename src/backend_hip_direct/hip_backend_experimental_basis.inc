#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_basis_pack(const uint64_t*,int8_t*,uint64_t,const BasisPacket*,uint32_t,bool);
extern "C" int rns8_hip_basis_gemm(const int8_t*,const int8_t*,int8_t*,int,int,int,const uint16_t*,uint32_t);
extern "C" int rns8_hip_basis_gemm_stream(const int8_t*,const int8_t*,int8_t*,int,int,int,const uint16_t*,uint32_t,void*);
extern "C" int rns8_hip_basis_export(const int8_t*,uint64_t*,uint64_t,const BasisPacket*,uint32_t,bool,uint32_t,BasisMagnitude,int*);
extern "C" int rns8_hip_basis_convert(const int8_t*,int8_t*,uint64_t,const BasisPacket*,const BasisPacket*,uint32_t,uint32_t,bool,BasisMagnitude,int*);
extern "C" int rns8_hip_basis_convert_value_stream(const int8_t*,int8_t*,uint64_t,BasisPacket,BasisPacket,uint32_t,uint32_t,bool,BasisMagnitude,int*,void*);
#endif
rns8_status hip_direct_enqueue_basis_conversion(int device,const void* source,void* output,uint64_t cells,
    const BasisPacket& from,const BasisPacket& to,uint32_t source_prefix,uint32_t target_prefix,bool sign,
    BasisMagnitude bound,void* scratch,uint64_t bytes,void* host_status,void* stream) {
  if(!source || !output || !scratch || !host_status || !stream || !cells || cells>INT32_MAX ||
      !source_prefix || source_prefix>from.count || !target_prefix || target_prefix>to.count ||
      bytes<cells*target_prefix+4)return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  const auto st=static_cast<hipStream_t>(stream);auto* flag=static_cast<int*>(scratch);
  auto* stage=static_cast<int8_t*>(scratch)+4;
  if(hipMemsetAsync(flag,0,4,st)!=hipSuccess || rns8_hip_basis_convert_value_stream(static_cast<const int8_t*>(source),
      stage,cells,from,to,source_prefix,target_prefix,sign,bound,flag,stream)!=int(hipSuccess) ||
      rns8_hip_guarded_copy_bytes(stage,output,1,cells*target_prefix,cells*target_prefix,cells*target_prefix,flag,stream)!=int(hipSuccess))return RNS8_BACKEND_FAILURE;
  return hipMemcpyAsync(host_status,flag,4,hipMemcpyDeviceToHost,st)==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device;(void)sign;(void)bound;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_enqueue_basis_gemm(int device,const void* a,const void* b,void* c,
    int64_t m,int64_t n,int64_t k,const uint16_t* moduli,uint32_t prefix,void* stream) {
  if(!stream || !moduli || !prefix || prefix>20 || !qualified_i32_gemm_address(m,n,k,k,n,n,prefix))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  return static_cast<hipError_t>(rns8_hip_basis_gemm_stream(static_cast<const int8_t*>(a),static_cast<const int8_t*>(b),
      static_cast<int8_t*>(c),int(m),int(n),int(k),moduli,prefix,stream))==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device;(void)a;(void)b;(void)c;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_basis_pack(int device,const void* native,void* residues,uint64_t cells,const void* packet,uint32_t prefix,bool sign) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  const auto error=timed_hip_operation("experimental_basis_native_pack",[&]() {
    const auto issued=static_cast<hipError_t>(rns8_hip_basis_pack(static_cast<const uint64_t*>(native),static_cast<int8_t*>(residues),cells,static_cast<const BasisPacket*>(packet),prefix,sign));
    return issued==hipSuccess?hipDeviceSynchronize():issued;
  });return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device;(void)native;(void)residues;(void)cells;(void)packet;(void)prefix;(void)sign;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_basis_gemm(int device,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,const uint16_t* moduli,uint32_t prefix) {
  if(!moduli || !prefix || prefix>20 || !qualified_i32_gemm_address(m,n,k,k,n,n,prefix))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  const auto error=timed_hip_operation("experimental_basis_residue_gemm",[&]() {
    const auto issued=static_cast<hipError_t>(rns8_hip_basis_gemm(static_cast<const int8_t*>(a),static_cast<const int8_t*>(b),static_cast<int8_t*>(c),int(m),int(n),int(k),moduli,prefix));
    return issued==hipSuccess?hipDeviceSynchronize():issued;
  });return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device;(void)a;(void)b;(void)c;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_basis_export(int device,const void* residues,void* output,uint64_t cells,const void* packet,uint32_t prefix,bool sign,uint32_t limbs,BasisMagnitude bound,void* flag,int* host_status) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(!flag || !host_status)return RNS8_INVALID_ARGUMENT;
  if(set_hip_device(device)!=RNS8_SUCCESS || hipMemset(flag,0,4)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  const auto error=timed_hip_operation("experimental_basis_checked_crt",[&]() {
    const auto issued=static_cast<hipError_t>(rns8_hip_basis_export(static_cast<const int8_t*>(residues),static_cast<uint64_t*>(output),cells,static_cast<const BasisPacket*>(packet),prefix,sign,limbs,bound,static_cast<int*>(flag)));
    return issued==hipSuccess?hipDeviceSynchronize():issued;
  });
  if(error!=hipSuccess || tracked_sync_copy(host_status,flag,4,hipMemcpyDeviceToHost)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  return *host_status==0?RNS8_SUCCESS:*host_status==int(RNS8_INVALID_ARGUMENT)?RNS8_INVALID_ARGUMENT:RNS8_RANGE_ERROR;
#else
  (void)device;(void)residues;(void)output;(void)cells;(void)packet;(void)prefix;(void)sign;(void)limbs;(void)bound;(void)flag;(void)host_status;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_basis_convert(int device,const void* source,void* output,uint64_t cells,const void* from,const void* to,uint32_t source_prefix,uint32_t target_prefix,bool sign,BasisMagnitude bound,void* flag,int* host_status) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(!flag || !host_status)return RNS8_INVALID_ARGUMENT;
  if(set_hip_device(device)!=RNS8_SUCCESS || hipMemset(flag,0,4)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  const auto error=timed_hip_operation("experimental_basis_exact_conversion",[&]() {
    const auto issued=static_cast<hipError_t>(rns8_hip_basis_convert(static_cast<const int8_t*>(source),static_cast<int8_t*>(output),cells,static_cast<const BasisPacket*>(from),static_cast<const BasisPacket*>(to),source_prefix,target_prefix,sign,bound,static_cast<int*>(flag)));
    return issued==hipSuccess?hipDeviceSynchronize():issued;
  });
  if(error!=hipSuccess || tracked_sync_copy(host_status,flag,4,hipMemcpyDeviceToHost)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  return *host_status==0?RNS8_SUCCESS:*host_status==int(RNS8_INVALID_ARGUMENT)?RNS8_INVALID_ARGUMENT:RNS8_RANGE_ERROR;
#else
  (void)device;(void)source;(void)output;(void)cells;(void)from;(void)to;(void)source_prefix;(void)target_prefix;(void)sign;(void)bound;(void)flag;(void)host_status;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
