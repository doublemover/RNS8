#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_wide_gemm(const int8_t*,const int8_t*,int8_t*,uint64_t,uint64_t,uint64_t,
                                 uint64_t,uint64_t,uint64_t,uint64_t);
#endif
rns8_status hip_direct_gemm_rns_wide_device(int device_id,const void* a,const void* b,void* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix) {
  if (!a || !b || !c) return RNS8_INVALID_ARGUMENT;
  const auto support=query_wide_gemm_support(m,n,k,lda,ldb,ldc,prefix);
  if (support.status!=RNS8_SUCCESS) return support.status;
  if (!nonoverlapping_byte_ranges(c,support.c.byte_extent,a,support.a.byte_extent) ||
      !nonoverlapping_byte_ranges(c,support.c.byte_extent,b,support.b.byte_extent)) return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto status=set_hip_device(device_id);
  if (status!=RNS8_SUCCESS) return status;
  const auto error=timed_hip_operation("rns_gemm_u64_index_kernel",[&]() {
    const auto launch=static_cast<hipError_t>(rns8_hip_wide_gemm(static_cast<const int8_t*>(a),
        static_cast<const int8_t*>(b),static_cast<int8_t*>(c),m,n,k,lda,ldb,ldc,support.logical_outputs));
    return launch==hipSuccess ? hipDeviceSynchronize() : launch;
  });
  return error==hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
