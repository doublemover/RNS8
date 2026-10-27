#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_convert_residue_planes(const int8_t*,ResidueLayout,int8_t*,ResidueLayout,uint32_t,uint32_t);
extern "C" int rns8_hip_gemm_residue_layout(const int8_t*,ResidueLayout,const int8_t*,ResidueLayout,
    int8_t*,ResidueLayout,uint32_t,ResidueLayoutModuli);
#endif

rns8_status hip_direct_convert_residue_layout(
    int device_id, const void* src, const ResidueLayout& source, void* dst, const ResidueLayout& destination) {
  if (!compatible_residue_layouts(source, destination) ||
      !nonoverlapping_byte_ranges(src, source.bytes, dst, destination.bytes)) return RNS8_INVALID_ARGUMENT;
  return hip_direct_convert_residue_planes(device_id,src,source,dst,destination,0,source.prefix);
}

rns8_status hip_direct_convert_residue_planes(int device_id,const void* src,
    const ResidueLayout& source,void* dst,const ResidueLayout& destination,uint32_t first,uint32_t end) {
  if(!compatible_residue_planes(source,destination,first,end) ||
      !nonoverlapping_byte_ranges(src,source.bytes,dst,destination.bytes))return RNS8_INVALID_ARGUMENT;
  if (source.bytes > INT32_MAX || destination.bytes > INT32_MAX) return RNS8_UNSUPPORTED_BACKEND;
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
  if(!compatible_layout_gemm(al,bl,cl,prefix) ||
      !nonoverlapping_byte_ranges(a,al.bytes,c,cl.bytes) ||
      !nonoverlapping_byte_ranges(b,bl.bytes,c,cl.bytes))return RNS8_INVALID_ARGUMENT;
  // Bound launch dimensions independently of each descriptor's element extent.
  if(al.bytes>INT32_MAX || bl.bytes>INT32_MAX || cl.bytes>INT32_MAX ||
      (al.rows+15)/16>65535 || (bl.cols+15)/16>INT32_MAX)return RNS8_UNSUPPORTED_BACKEND;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS)return status;
  ResidueLayoutModuli moduli;
  std::copy_n(al.basis.moduli.data(),20,moduli.values);
  const auto error=timed_hip_operation("residue_layout_gemm_kernel",[&]() {
    const auto issued=static_cast<hipError_t>(rns8_hip_gemm_residue_layout(
        static_cast<const int8_t*>(a),al,static_cast<const int8_t*>(b),bl,static_cast<int8_t*>(c),cl,prefix,moduli));
    return issued==hipSuccess?hipDeviceSynchronize():issued;
  });
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
