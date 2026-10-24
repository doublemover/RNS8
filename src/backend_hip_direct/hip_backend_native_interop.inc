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
    int64_t rows,int64_t cols,int64_t ld,uint32_t prefix,uint64_t bound,bool signed_values,bool& external_write_started) {
  external_write_started=false;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  auto* flag=reinterpret_cast<int*>(static_cast<uint8_t*>(scratch)+uint64_t(rows)*cols*8);
  auto err=hipMemset(flag,0,4);if(err!=hipSuccess) return RNS8_BACKEND_FAILURE;
  err=timed_hip_operation("interop_native_reconstruct",[&]() {
    const int code=signed_values?rns8_hip_direct_export_i64_device(static_cast<const int8_t*>(residues),static_cast<int64_t*>(scratch),int(rows),int(cols),int(prefix),bound,flag):
        rns8_hip_direct_export_u64_device(static_cast<const int8_t*>(residues),static_cast<uint64_t*>(scratch),int(rows),int(cols),int(prefix),bound,flag);
    const auto drain=hipDeviceSynchronize();return code==int(hipSuccess)?drain:static_cast<hipError_t>(code);
  });if(err!=hipSuccess) return RNS8_BACKEND_FAILURE;
  int result=0;err=tracked_sync_copy(&result,flag,4,hipMemcpyDeviceToHost);if(err!=hipSuccess) return RNS8_BACKEND_FAILURE;
  if(result!=int(RNS8_SUCCESS)) return static_cast<rns8_status>(result);
  external_write_started=true;
  err=timed_hip_operation("interop_native_commit_d2d",[&]() {
    const auto issued=tracked_sync_copy_2d(destination,size_t(ld)*8,scratch,size_t(cols)*8,size_t(cols)*8,size_t(rows),hipMemcpyDeviceToDevice);
    const auto drain=hipDeviceSynchronize();return issued==hipSuccess?drain:issued;
  });return err==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)residues;(void)destination;(void)scratch;(void)rows;(void)cols;(void)ld;(void)prefix;(void)bound;(void)signed_values;return RNS8_UNSUPPORTED_BACKEND;
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
