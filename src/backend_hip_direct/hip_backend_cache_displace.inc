#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_cache_displace(void*,uint64_t,uint32_t);
#endif
rns8_status hip_direct_cache_displace(int device_id,void* storage,uint64_t bytes,uint32_t passes){
  if(!storage || reinterpret_cast<uintptr_t>(storage)%64 || bytes<64 || bytes%64 || bytes>(UINT64_C(1)<<32) || !passes || passes>16)return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS)return status;
  const auto issued=static_cast<hipError_t>(rns8_hip_cache_displace(storage,bytes,passes));const auto drained=hipDeviceSynchronize();
  return issued==hipSuccess && drained==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
