// Enqueue-only panel primitives. The public owner admits bounds/storage and
// retains its three streams, pinned rings, events and arena through completion.
rns8_status hip_direct_enqueue_panel_pack(int device,const void* source,void* residues,
    int64_t rows,int64_t cols,uint32_t prefix,bool sign,void* stream) {
  if(!source || !residues || !stream || !prefix || prefix>20 ||
      !qualified_i32_matrix_address(rows,cols,cols,1,8) ||
      !qualified_i32_matrix_address(rows,cols,cols,prefix))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  const auto code=sign?rns8_hip_direct_pack_i64_device_on_stream(static_cast<const int64_t*>(source),static_cast<int8_t*>(residues),
      int(rows),int(cols),int(cols),int(prefix),stream):
      rns8_hip_direct_pack_u64_device_on_stream(static_cast<const uint64_t*>(source),static_cast<int8_t*>(residues),
      int(rows),int(cols),int(cols),int(prefix),stream);
  return code==int(hipSuccess)?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device;(void)sign;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_enqueue_panel_gemm(int device,const void* a,const void* b,const void* previous,void* output,
    int64_t m,int64_t n,int64_t k,uint32_t prefix,void* stream) {
  if(!a || !b || !previous || !output || !stream)return RNS8_INVALID_ARGUMENT;
  const auto support=query_axpby_support(m,n,k,k,n,n,n,prefix);if(support.status!=RNS8_SUCCESS)return support.status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  const auto code=rns8_hip_gemm_rns_axpby(static_cast<const int8_t*>(a),static_cast<const int8_t*>(b),
      static_cast<const int8_t*>(previous),static_cast<int8_t*>(output),int(m),int(n),int(k),int(k),int(n),int(n),int(n),
      int(prefix),1,1,stream);
  return code==int(hipSuccess)?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_enqueue_panel_export(int device,const void* residues,void* output,void* flag,
    int64_t rows,int64_t cols,uint32_t prefix,uint64_t bound,bool sign,void* stream) {
  if(!residues || !output || !flag || !stream || !prefix || prefix>20 ||
      !qualified_i32_matrix_address(rows,cols,cols,prefix) ||
      !qualified_i32_matrix_address(rows,cols,cols,1,8))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device)!=RNS8_SUCCESS || hipMemsetAsync(flag,0,4,static_cast<hipStream_t>(stream))!=hipSuccess)
    return RNS8_BACKEND_FAILURE;
  const auto code=sign?rns8_hip_direct_export_i64_device_on_stream(static_cast<const int8_t*>(residues),static_cast<int64_t*>(output),
      int(rows),int(cols),int(prefix),bound,static_cast<int*>(flag),stream):
      rns8_hip_direct_export_u64_device_on_stream(static_cast<const int8_t*>(residues),static_cast<uint64_t*>(output),
      int(rows),int(cols),int(prefix),bound,static_cast<int*>(flag),stream);
  return code==int(hipSuccess)?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device;(void)bound;(void)sign;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
