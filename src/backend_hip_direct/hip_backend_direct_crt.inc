#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_export_canonical_direct_crt(const int8_t*, uint64_t*, int, int);
#endif

rns8_status hip_direct_export_canonical_direct_crt_device(
    int device_id, const void* residues, void* limbs, int64_t cells, uint32_t prefix) {
  if (!residues || !limbs || cells <= 0 || !prefix || prefix > RNS8_MAX_SUPPORTED_PREFIX) return RNS8_INVALID_ARGUMENT;
  const auto support = query_canonical_crt_support(cells,prefix);
  if (support.status != RNS8_SUCCESS) return support.status;
  if (!nonoverlapping_byte_ranges(residues, static_cast<uint64_t>(cells) * prefix,
                                 limbs, static_cast<uint64_t>(cells) * 24)) return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  const auto error = timed_hip_operation("direct_crt_canonical_export_kernel", [&]() {
    const auto launched = static_cast<hipError_t>(rns8_hip_export_canonical_direct_crt(
        static_cast<const int8_t*>(residues), static_cast<uint64_t*>(limbs), static_cast<int>(cells), prefix));
    return launched == hipSuccess ? hipDeviceSynchronize() : launched;
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_export_quarter_crt(const int8_t*,uint64_t*,int,int,int,int,int);
#endif
rns8_status hip_direct_export_quarter_crt_device(int device_id,const void* residues,void* output,
    int64_t cells,uint32_t prefix,uint32_t digits,uint32_t limbs,uint32_t mode) {
  if(!residues || !output || cells<=0 || cells>INT32_MAX-255 || !prefix || prefix>20 || !digits || digits>12 ||
      !limbs || limbs>32 || mode>2 || (mode==2 && digits!=12) ||
      !nonoverlapping_byte_ranges(residues,uint64_t(cells)*prefix,output,uint64_t(cells)*limbs*8))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS)return status;
  const auto error=timed_hip_operation("quarter_crt_fma_delayed16_export_kernel",[&](){
    const auto launched=static_cast<hipError_t>(rns8_hip_export_quarter_crt(static_cast<const int8_t*>(residues),static_cast<uint64_t*>(output),
      static_cast<int>(cells),prefix,digits,limbs,mode));return launched==hipSuccess?hipDeviceSynchronize():launched;});
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
