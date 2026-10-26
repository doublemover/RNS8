#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_export_canonical_group_crt(const int8_t*,uint64_t*,uint64_t*,int,
    const group_crt::Plan*,uint32_t,uint32_t);
#endif
rns8_status hip_direct_export_canonical_group_crt_device(int device_id,const void* residues,
    void* limbs,int64_t cells,uint32_t prefix,bool split,void* scratch,std::size_t scratch_bytes,uint32_t order) {
  if(!residues || !limbs)return RNS8_INVALID_ARGUMENT;
  const auto support=query_group_crt_support(cells,prefix,split,scratch_bytes);
  if(support.status!=RNS8_SUCCESS)return support.status;
  const uint32_t groups=(prefix+7)/8;
  if(order>=2*groups || (!split && (order || scratch || scratch_bytes)))return RNS8_INVALID_ARGUMENT;
  if(split && !scratch)return RNS8_WORKSPACE_TOO_SMALL;
  const auto source_bytes=uint64_t(cells)*prefix;
  if(!nonoverlapping_byte_ranges(residues,source_bytes,limbs,support.payload_bytes) ||
      (split && (!nonoverlapping_byte_ranges(residues,source_bytes,scratch,support.scratch_bytes) ||
                 !nonoverlapping_byte_ranges(limbs,support.payload_bytes,scratch,support.scratch_bytes))))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  try {
    const auto& plan=group_crt::plan(prefix);
    const auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS)return status;
    auto error=timed_hip_operation(split?"group_crt_producer_kernel":"group_crt_local_kernel",[&] {
      return static_cast<hipError_t>(rns8_hip_export_canonical_group_crt(
          static_cast<const int8_t*>(residues),static_cast<uint64_t*>(limbs),
          static_cast<uint64_t*>(scratch),static_cast<int>(cells),&plan,order,split?1:0));
    });
    if(split && error==hipSuccess)error=timed_hip_operation("group_crt_root_merge_kernel",[&] {
      return static_cast<hipError_t>(rns8_hip_export_canonical_group_crt(
          nullptr,static_cast<uint64_t*>(limbs),static_cast<uint64_t*>(scratch),static_cast<int>(cells),&plan,order,2));
    });
    // Also drain any issued producer if the subsequent launch is rejected.
    // Publication/ownership remains with the caller of this raw research path.
    const auto drain=hipDeviceSynchronize();if(error==hipSuccess)error=drain;
    return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
  } catch(const std::exception&) {return RNS8_INTERNAL_ERROR;}
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
