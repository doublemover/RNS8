#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_checked_crt_decode(const uint64_t*,uint64_t*,int,const checked_crt::Decode*,uint32_t*);
#endif
rns8_status hip_direct_decode_crt_roots_device(int device_id,const void* roots,void* output,
    int64_t cells,const checked_crt::Decode& contract,void* status_word) {
  if(!roots || !output || !status_word || !contract.limbs || contract.limbs>32 || contract.mode>2)
    return RNS8_INVALID_ARGUMENT;
  if(cells<=0 || cells>INT32_MAX-255)return RNS8_RANGE_ERROR;
  const auto count=uint64_t(cells);
  if(!nonoverlapping_byte_ranges(roots,count*24,output,count*contract.limbs*8) ||
      !nonoverlapping_byte_ranges(roots,count*24,status_word,4) ||
      !nonoverlapping_byte_ranges(output,count*contract.limbs*8,status_word,4))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto selected=set_hip_device(device_id);if(selected!=RNS8_SUCCESS)return selected;
  const auto error=timed_hip_operation("checked_crt_decode_kernel",[&] {
    return static_cast<hipError_t>(rns8_hip_checked_crt_decode(static_cast<const uint64_t*>(roots),
        static_cast<uint64_t*>(output),int(cells),&contract,static_cast<uint32_t*>(status_word)));
  });
  // The prepared owner drains and retains all buffers/read leases on failure.
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
