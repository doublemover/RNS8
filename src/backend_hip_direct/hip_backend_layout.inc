#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_convert_residue_planes(const int8_t*,ResidueLayout,int8_t*,ResidueLayout,uint32_t,uint32_t);
extern "C" int rns8_hip_gemm_residue_layout(const int8_t*,ResidueLayout,const int8_t*,ResidueLayout,
    int8_t*,ResidueLayout,uint32_t,ResidueLayoutModuli);
extern "C" int rns8_hip_convert_residue_planes_on_stream(const int8_t*,ResidueLayout,int8_t*,ResidueLayout,uint32_t,uint32_t,void*,const int*);
extern "C" int rns8_hip_gemm_residue_layout_on_stream(const int8_t*,ResidueLayout,const int8_t*,ResidueLayout,int8_t*,ResidueLayout,uint32_t,ResidueLayoutModuli,void*,const int*);
extern "C" int rns8_hip_first_failure_on_stream(const int*,int*,void*);
#endif

namespace {
bool layout_guard_disjoint(const int* guard,const void* storage,uint64_t bytes) {
  return !guard || (reinterpret_cast<uintptr_t>(guard)%alignof(int)==0 &&
      nonoverlapping_byte_ranges(guard,sizeof(int),storage,bytes));
}
rns8_status admit_layout_conversion(const void* src,const ResidueLayout& source,void* dst,
    const ResidueLayout& destination,uint32_t first,uint32_t end,const int* guard=nullptr) {
  if(!compatible_residue_planes(source,destination,first,end) ||
      !nonoverlapping_byte_ranges(src,source.bytes,dst,destination.bytes) ||
      !layout_guard_disjoint(guard,src,source.bytes) || !layout_guard_disjoint(guard,dst,destination.bytes))return RNS8_INVALID_ARGUMENT;
  return source.bytes>INT32_MAX || destination.bytes>INT32_MAX?RNS8_UNSUPPORTED_BACKEND:RNS8_SUCCESS;
}
rns8_status admit_layout_product(const void* a,const ResidueLayout& al,const void* b,const ResidueLayout& bl,
    void* c,const ResidueLayout& cl,uint32_t prefix,const int* guard=nullptr) {
  if(!compatible_layout_gemm(al,bl,cl,prefix) || !nonoverlapping_byte_ranges(a,al.bytes,c,cl.bytes) ||
      !nonoverlapping_byte_ranges(b,bl.bytes,c,cl.bytes) || !layout_guard_disjoint(guard,a,al.bytes) ||
      !layout_guard_disjoint(guard,b,bl.bytes) || !layout_guard_disjoint(guard,c,cl.bytes))return RNS8_INVALID_ARGUMENT;
  return al.bytes>INT32_MAX || bl.bytes>INT32_MAX || cl.bytes>INT32_MAX ||
      (al.rows+15)/16>65535 || (bl.cols+15)/16>INT32_MAX?RNS8_UNSUPPORTED_BACKEND:RNS8_SUCCESS;
}
}

rns8_status hip_direct_convert_residue_layout(
    int device_id, const void* src, const ResidueLayout& source, void* dst, const ResidueLayout& destination) {
  if (!compatible_residue_layouts(source, destination) ||
      !nonoverlapping_byte_ranges(src, source.bytes, dst, destination.bytes)) return RNS8_INVALID_ARGUMENT;
  return hip_direct_convert_residue_planes(device_id,src,source,dst,destination,0,source.prefix);
}

rns8_status hip_direct_convert_residue_planes(int device_id,const void* src,
    const ResidueLayout& source,void* dst,const ResidueLayout& destination,uint32_t first,uint32_t end) {
  const auto admitted=admit_layout_conversion(src,source,dst,destination,first,end);
  if(admitted!=RNS8_SUCCESS)return admitted;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  const auto error = timed_hip_operation("residue_layout_convert_kernel", [&]() {
    const auto launched = static_cast<hipError_t>(rns8_hip_convert_residue_planes(
        static_cast<const int8_t*>(src),source,static_cast<int8_t*>(dst),destination,first,end));
    return launched == hipSuccess ? hipDeviceSynchronize() : launched;
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_residue_layout(int device_id,const void* a,const ResidueLayout& al,
    const void* b,const ResidueLayout& bl,void* c,const ResidueLayout& cl,uint32_t prefix) {
  const auto admitted=admit_layout_product(a,al,b,bl,c,cl,prefix);
  if(admitted!=RNS8_SUCCESS)return admitted;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS)return status;
  const auto moduli=residue_layout_moduli(al.basis);
  const auto error=timed_hip_operation("residue_layout_gemm_blocked_k65536_v2",[&]() {
    const auto issued=static_cast<hipError_t>(rns8_hip_gemm_residue_layout(
        static_cast<const int8_t*>(a),al,static_cast<const int8_t*>(b),bl,static_cast<int8_t*>(c),cl,prefix,moduli));
    return issued==hipSuccess?hipDeviceSynchronize():issued;
  });
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_enqueue_residue_planes(int device,const void* src,const ResidueLayout& source,
    void* dst,const ResidueLayout& destination,uint32_t first,uint32_t end,void* stream,const int* guard) {
  if(!stream)return RNS8_INVALID_ARGUMENT;
  const auto admitted=admit_layout_conversion(src,source,dst,destination,first,end,guard);
  if(admitted!=RNS8_SUCCESS)return admitted;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  return rns8_hip_convert_residue_planes_on_stream(static_cast<const int8_t*>(src),source,static_cast<int8_t*>(dst),
      destination,first,end,stream,guard)==int(hipSuccess)?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_enqueue_residue_layout_gemm(int device,const void* a,const ResidueLayout& al,
    const void* b,const ResidueLayout& bl,void* c,const ResidueLayout& cl,uint32_t prefix,void* stream,const int* guard) {
  if(!stream)return RNS8_INVALID_ARGUMENT;
  const auto admitted=admit_layout_product(a,al,b,bl,c,cl,prefix,guard);
  if(admitted!=RNS8_SUCCESS)return admitted;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  return rns8_hip_gemm_residue_layout_on_stream(static_cast<const int8_t*>(a),al,static_cast<const int8_t*>(b),bl,
      static_cast<int8_t*>(c),cl,prefix,residue_layout_moduli(al.basis),stream,guard)==int(hipSuccess)?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_enqueue_first_failure(int device,const int* stage,int* sticky,void* stream) {
  if(!stream || !stage || !sticky || reinterpret_cast<uintptr_t>(stage)%alignof(int) ||
      reinterpret_cast<uintptr_t>(sticky)%alignof(int) || !nonoverlapping_byte_ranges(stage,sizeof(int),sticky,sizeof(int)))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  return rns8_hip_first_failure_on_stream(stage,sticky,stream)==int(hipSuccess)?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
