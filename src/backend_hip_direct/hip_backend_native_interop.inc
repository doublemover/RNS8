rns8_status hip_direct_copy_native_view(int device_id,const void* source,void* destination,
    int64_t rows,int64_t cols,int64_t ld) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  const auto error=timed_hip_operation("interop_lease_snapshot_d2d",[&]() {
    const auto issued=tracked_sync_copy_2d(destination,size_t(cols)*8,source,size_t(ld)*8,
        size_t(cols)*8,size_t(rows),hipMemcpyDeviceToDevice);
    const auto drain=hipDeviceSynchronize();return issued==hipSuccess?drain:issued;
  });
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)source;(void)destination;(void)rows;(void)cols;(void)ld;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_validate_native_allocation(int device_id,const void* base,uint64_t bytes) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  hipPointerAttribute_t attributes{};
  if(hipPointerGetAttributes(&attributes,base)!=hipSuccess) return RNS8_INVALID_ARGUMENT;
  if(attributes.type!=hipMemoryTypeDevice || attributes.isManaged || attributes.device!=device_id)
    return RNS8_UNSUPPORTED_BACKEND;
  hipDeviceptr_t actual=nullptr;size_t available=0;
  if(hipMemGetAddressRange(&actual,&available,reinterpret_cast<hipDeviceptr_t>(const_cast<void*>(base)))!=hipSuccess)
    return RNS8_UNSUPPORTED_BACKEND;
  if(actual!=base || bytes>available) return RNS8_INVALID_ARGUMENT;
  return RNS8_SUCCESS;
#else
  (void)device_id;(void)base;(void)bytes;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_import_native_view(int device_id,const void* source,void* residues,
    int64_t rows,int64_t cols,int64_t ld,uint32_t prefix,bool signed_values,void* retained_native) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  if(retained_native) {
    const auto copy=timed_hip_operation("interop_native_retain_d2d",[&]() {
      const auto issued=tracked_sync_copy_2d(retained_native,size_t(cols)*8,source,size_t(ld)*8,
          size_t(cols)*8,size_t(rows),hipMemcpyDeviceToDevice);
      const auto drain=hipDeviceSynchronize();return issued==hipSuccess?drain:issued;
    });
    if(copy!=hipSuccess) return RNS8_BACKEND_FAILURE;
    source=retained_native;ld=cols;
  }
  const auto error=timed_hip_operation("interop_native_import",[&]() {
    const int code=signed_values?rns8_hip_direct_pack_i64_device(static_cast<const int64_t*>(source),static_cast<int8_t*>(residues),int(rows),int(cols),int(ld),int(prefix)):
        rns8_hip_direct_pack_u64_device(static_cast<const uint64_t*>(source),static_cast<int8_t*>(residues),int(rows),int(cols),int(ld),int(prefix));
    const auto drain=hipDeviceSynchronize();return code==int(hipSuccess)?drain:static_cast<hipError_t>(code);
  });return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)source;(void)residues;(void)rows;(void)cols;(void)ld;(void)prefix;(void)signed_values;(void)retained_native;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_export_native_view(int device_id,const void* residues,void* destination,void* scratch,
    int64_t rows,int64_t cols,int64_t ld,uint32_t prefix,uint64_t bound,bool signed_values,bool& external_write_started,uint32_t limb_count) {
  external_write_started=false;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  const uint64_t element_bytes=8*uint64_t(limb_count?limb_count:1);
  auto* flag=reinterpret_cast<int*>(static_cast<uint8_t*>(scratch)+uint64_t(rows)*cols*element_bytes);
  auto err=hipMemset(flag,0,4);if(err!=hipSuccess) return RNS8_BACKEND_FAILURE;
  err=timed_hip_operation("interop_native_reconstruct",[&]() {
    const int code=limb_count ? (signed_values ?
        rns8_hip_direct_export_exact_wide_signed_limbs_device(static_cast<const int8_t*>(residues),static_cast<uint64_t*>(scratch),int(rows),int(cols),int(prefix),int(limb_count),flag) :
        rns8_hip_direct_export_exact_wide_unsigned_limbs_device(static_cast<const int8_t*>(residues),static_cast<uint64_t*>(scratch),int(rows),int(cols),int(prefix),int(limb_count),flag)) :
        signed_values?rns8_hip_direct_export_i64_device(static_cast<const int8_t*>(residues),static_cast<int64_t*>(scratch),int(rows),int(cols),int(prefix),bound,flag):
        rns8_hip_direct_export_u64_device(static_cast<const int8_t*>(residues),static_cast<uint64_t*>(scratch),int(rows),int(cols),int(prefix),bound,flag);
    const auto drain=hipDeviceSynchronize();return code==int(hipSuccess)?drain:static_cast<hipError_t>(code);
  });if(err!=hipSuccess) return RNS8_BACKEND_FAILURE;
  int result=0;err=tracked_sync_copy(&result,flag,4,hipMemcpyDeviceToHost);if(err!=hipSuccess) return RNS8_BACKEND_FAILURE;
  if(result!=int(RNS8_SUCCESS)) return static_cast<rns8_status>(result);
  external_write_started=true;
  err=timed_hip_operation("interop_native_commit_d2d",[&]() {
    const auto issued=tracked_sync_copy_2d(destination,size_t(ld)*element_bytes,scratch,size_t(cols)*element_bytes,size_t(cols)*element_bytes,size_t(rows),hipMemcpyDeviceToDevice);
    const auto drain=hipDeviceSynchronize();return issued==hipSuccess?drain:issued;
  });return err==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)residues;(void)destination;(void)scratch;(void)rows;(void)cols;(void)ld;(void)prefix;(void)bound;(void)signed_values;(void)limb_count;return RNS8_UNSUPPORTED_BACKEND;
#endif
}

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_abs_max(const void*,int,int,int,void*,bool);
#endif
rns8_status hip_direct_native_view_bound(int device_id,const void* source,void* scratch,
    int64_t rows,int64_t cols,int64_t ld,bool signed_values,uint64_t& bound) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  if(hipMemset(scratch,0,8)!=hipSuccess) return RNS8_BACKEND_FAILURE;
  const auto error=timed_hip_operation("interop_native_bound_scan",[&]() {
    const auto launch=static_cast<hipError_t>(rns8_hip_native_abs_max(source,int(rows),int(cols),int(ld),scratch,signed_values));
    const auto drain=hipDeviceSynchronize();return launch==hipSuccess?drain:launch;
  });if(error!=hipSuccess) return RNS8_BACKEND_FAILURE;
  uint64_t result=0;status=hip_direct_copy_device_to_host(device_id,&result,scratch,8);
  if(status==RNS8_SUCCESS) bound=result;return status;
#else
  (void)device_id;(void)source;(void)scratch;(void)rows;(void)cols;(void)ld;(void)signed_values;(void)bound;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_direct_pack_i64_device_on_stream(const int64_t*,int8_t*,int,int,int,int,void*);
extern "C" int rns8_hip_direct_pack_u64_device_on_stream(const uint64_t*,int8_t*,int,int,int,int,void*);
extern "C" int rns8_hip_direct_export_i64_device_on_stream(const int8_t*,int64_t*,int,int,int,uint64_t,int*,void*);
extern "C" int rns8_hip_direct_export_u64_device_on_stream(const int8_t*,uint64_t*,int,int,int,uint64_t,int*,void*);
extern "C" int rns8_hip_direct_export_exact_wide_signed_limbs_device_on_stream(const int8_t*,uint64_t*,int,int,int,int,int*,void*);
extern "C" int rns8_hip_direct_export_exact_wide_unsigned_limbs_device_on_stream(const int8_t*,uint64_t*,int,int,int,int,int*,void*);
extern "C" int rns8_hip_native_abs_max_on_stream(const void*,int,int,int,void*,bool,void*);
extern "C" int rns8_hip_native_bound_status(const void*,uint64_t,int*,void*);
extern "C" int rns8_hip_basis_pack_view_stream(const uint64_t*,int8_t*,uint64_t,uint64_t,uint64_t,BasisPacket,uint32_t,bool,void*);
extern "C" int rns8_hip_basis_export_view_stream(const int8_t*,uint64_t*,uint64_t,BasisPacket,uint32_t,bool,uint32_t,BasisMagnitude,int*,void*);
#endif
rns8_status hip_direct_enqueue_native_import(int device_id,const void* source,void* residues,
    int64_t rows,int64_t cols,int64_t ld,uint32_t prefix,bool signed_values,bool verify,uint64_t bound,
    void* retained,void* scratch,uint64_t scratch_bytes,void* pinned_status,void* stream,const BasisPacket* basis) {
  if(!source || !residues || !scratch || !pinned_status || !stream || !prefix || prefix>20 ||
      !qualified_i32_matrix_address(rows,cols,ld,1,8) || !qualified_i32_matrix_address(rows,cols,cols,prefix) ||
      scratch_bytes<uint64_t(rows)*cols*prefix+16)return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device_id)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  const auto st=static_cast<hipStream_t>(stream);auto* flag=static_cast<int*>(scratch);
  auto* maximum=static_cast<uint8_t*>(scratch)+8;auto* stage=static_cast<int8_t*>(scratch)+16;
  if(hipMemsetAsync(scratch,0,16,st)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  if(verify && (rns8_hip_native_abs_max_on_stream(source,int(rows),int(cols),int(ld),maximum,signed_values,stream)!=int(hipSuccess) ||
      rns8_hip_native_bound_status(maximum,bound,flag,stream)!=int(hipSuccess)))return RNS8_BACKEND_FAILURE;
  const auto packed=basis?rns8_hip_basis_pack_view_stream(static_cast<const uint64_t*>(source),stage,uint64_t(rows)*cols,cols,ld,*basis,prefix,signed_values,stream):
      signed_values?rns8_hip_direct_pack_i64_device_on_stream(static_cast<const int64_t*>(source),stage,int(rows),int(cols),int(ld),int(prefix),stream):
      rns8_hip_direct_pack_u64_device_on_stream(static_cast<const uint64_t*>(source),stage,int(rows),int(cols),int(ld),int(prefix),stream);
  if(packed!=int(hipSuccess))return RNS8_BACKEND_FAILURE;
  const uint64_t payload=uint64_t(rows)*cols*prefix;
  if(rns8_hip_guarded_copy_bytes(stage,residues,1,payload,payload,payload,flag,stream)!=int(hipSuccess))return RNS8_BACKEND_FAILURE;
  if(retained && rns8_hip_guarded_copy_bytes(source,retained,rows,uint64_t(cols)*8,uint64_t(ld)*8,uint64_t(cols)*8,flag,stream)!=int(hipSuccess))return RNS8_BACKEND_FAILURE;
  return hipMemcpyAsync(pinned_status,flag,4,hipMemcpyDeviceToHost,st)==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)signed_values;(void)verify;(void)bound;(void)retained;(void)basis;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_enqueue_native_export(int device_id,const void* residues,void* destination,
    int64_t rows,int64_t cols,int64_t ld,uint32_t prefix,bool signed_values,uint32_t limbs,uint64_t bound,
    void* scratch,uint64_t scratch_bytes,void* pinned_status,void* stream,const BasisPacket* basis,BasisMagnitude magnitude) {
  const uint64_t width=8*uint64_t(limbs?limbs:1);
  if(!residues || !destination || !scratch || !pinned_status || !stream || !prefix || prefix>20 || limbs>32 ||
      !qualified_i32_matrix_address(rows,cols,ld,1,width) || !qualified_i32_matrix_address(rows,cols,cols,prefix) ||
      scratch_bytes<uint64_t(rows)*cols*width+4)return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device_id)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  const auto st=static_cast<hipStream_t>(stream);auto* flag=reinterpret_cast<int*>(static_cast<uint8_t*>(scratch)+uint64_t(rows)*cols*width);
  if(hipMemsetAsync(flag,0,4,st)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  const auto* input=static_cast<const int8_t*>(residues);auto* output=static_cast<uint64_t*>(scratch);
  const int code=basis?rns8_hip_basis_export_view_stream(input,output,uint64_t(rows)*cols,*basis,prefix,signed_values,limbs?limbs:1,magnitude,flag,stream):
      limbs?(signed_values?rns8_hip_direct_export_exact_wide_signed_limbs_device_on_stream(input,output,int(rows),int(cols),int(prefix),int(limbs),flag,stream):
      rns8_hip_direct_export_exact_wide_unsigned_limbs_device_on_stream(input,output,int(rows),int(cols),int(prefix),int(limbs),flag,stream)):
      signed_values?rns8_hip_direct_export_i64_device_on_stream(input,reinterpret_cast<int64_t*>(output),int(rows),int(cols),int(prefix),bound,flag,stream):
      rns8_hip_direct_export_u64_device_on_stream(input,output,int(rows),int(cols),int(prefix),bound,flag,stream);
  if(code!=int(hipSuccess))return RNS8_BACKEND_FAILURE;
  if(rns8_hip_guarded_copy_bytes(scratch,destination,rows,uint64_t(cols)*width,uint64_t(cols)*width,uint64_t(ld)*width,flag,stream)!=int(hipSuccess))return RNS8_BACKEND_FAILURE;
  return hipMemcpyAsync(pinned_status,flag,4,hipMemcpyDeviceToHost,st)==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)signed_values;(void)bound;(void)basis;(void)magnitude;return RNS8_UNSUPPORTED_BACKEND;
#endif
}

// Graph export deliberately stops at owned pinned storage. Publication into
// caller memory is a host transaction after all node status words are observed.
rns8_status hip_direct_enqueue_limb_export_to_host(int device_id,const void* residues,
    int64_t rows,int64_t cols,uint32_t prefix,bool signed_values,uint32_t limbs,
    void* scratch,uint64_t scratch_bytes,void* pinned_storage,uint64_t pinned_bytes,void* stream) {
  const auto layout=resident_export_layout(rows,cols,limbs);
  if(layout.status!=RNS8_SUCCESS)return layout.status;
  if(!residues || !scratch || !pinned_storage || !stream || !prefix || prefix>20 ||
      scratch_bytes<layout.device_bytes || pinned_bytes<layout.pinned_bytes ||
      reinterpret_cast<uintptr_t>(scratch)%8 || reinterpret_cast<uintptr_t>(pinned_storage)%8 ||
      !qualified_i32_matrix_address(rows,cols,cols,prefix))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device_id)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  const auto st=static_cast<hipStream_t>(stream);
  auto* flag=reinterpret_cast<int*>(static_cast<uint8_t*>(scratch)+layout.payload_bytes);
  if(hipMemsetAsync(flag,0,4,st)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  const int code=signed_values?rns8_hip_direct_export_exact_wide_signed_limbs_device_on_stream(
      static_cast<const int8_t*>(residues),static_cast<uint64_t*>(scratch),int(rows),int(cols),int(prefix),int(limbs),flag,stream):
      rns8_hip_direct_export_exact_wide_unsigned_limbs_device_on_stream(
      static_cast<const int8_t*>(residues),static_cast<uint64_t*>(scratch),int(rows),int(cols),int(prefix),int(limbs),flag,stream);
  if(code!=int(hipSuccess))return RNS8_BACKEND_FAILURE;
  if(hipMemcpyAsync(pinned_storage,flag,4,hipMemcpyDeviceToHost,st)!=hipSuccess ||
      hipMemcpyAsync(static_cast<uint8_t*>(pinned_storage)+8,scratch,layout.payload_bytes,hipMemcpyDeviceToHost,st)!=hipSuccess)
    return RNS8_BACKEND_FAILURE;
  return RNS8_SUCCESS;
#else
  (void)device_id;(void)signed_values;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
