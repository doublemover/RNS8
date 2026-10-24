#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_finite_verify(const int8_t*, const int8_t*, const int8_t*,
    uint64_t,uint64_t,uint64_t,uint32_t,const uint8_t*,uint32_t*,uint32_t*);
#endif
rns8_status hip_finite_verify_trial(int device, const void* a, const void* b,
    const void* c, uint64_t m, uint64_t n, uint64_t k, uint32_t modulus,
    const void* r, void* br, void* flag, uint32_t& result) {
  if (!a || !b || !c || !r || !br || !flag || !m || !n || !k || modulus<2 || modulus>256)
    return RNS8_INVALID_ARGUMENT;
  if(m>INT64_MAX || n>INT64_MAX || k>INT64_MAX) return RNS8_RANGE_ERROR;
  const auto support=query_finite_verify_support(int64_t(m),int64_t(n),int64_t(k),modulus,true,UINT64_MAX);
  if(support.status!=RNS8_SUCCESS) return support.status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device);
  if (status != RNS8_SUCCESS) return status;
  status = hip_direct_zero(device, flag, 4);
  if (status != RNS8_SUCCESS) return status;
  const auto error = timed_hip_operation("finite_freivalds_projection", [&]() {
    const auto launch = static_cast<hipError_t>(rns8_hip_finite_verify(
        static_cast<const int8_t*>(a), static_cast<const int8_t*>(b), static_cast<const int8_t*>(c),
        m,n,k,modulus,static_cast<const uint8_t*>(r),static_cast<uint32_t*>(br),static_cast<uint32_t*>(flag)));
    // B*r may already be queued when the second launch fails. Always drain
    // before the caller releases shared trial scratch; retain the launch error.
    const auto completion = hipDeviceSynchronize();
    return launch == hipSuccess ? completion : launch;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  return hip_direct_copy_device_to_host(device, &result, flag, sizeof(result));
#else
  (void)device; (void)result;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
