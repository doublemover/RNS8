#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_integrity_check(const void *, const void *,
                                        const int8_t *, int, int, int, int, int,
                                        int, bool, const uint64_t *,
                                        uint32_t *, void *);
#endif
static rns8_status validate_native_integrity(
    const void* a,const void* b,const void* candidate,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,bool unsigned_inputs,const uint64_t bound[3],
    const IntegrityIdentity& expected,const IntegrityIdentity& candidate_identity) {
  if (!a || !b || !candidate || !bound ||
      !integrity_identity_matches(expected, candidate_identity) || m <= 0 ||
      n <= 0 || k <= 0 || lda < k || ldb < n || bound[2] >> 32)
    return RNS8_INVALID_ARGUMENT;
  const auto prefix = expected.prefix;
  if (!qualified_i32_matrix_address(m, k, lda, 1, 8) ||
      !qualified_i32_matrix_address(k, n, ldb, 1, 8) ||
      !qualified_i32_matrix_address(m, n, n, prefix))
    return RNS8_UNSUPPORTED_BACKEND;
  NativeTileBound checked{{bound[0], bound[1], bound[2]}, 0, 0};
  select_native_bound_prefix(checked, !unsigned_inputs, prefix,
                             garner::kConstants.products);
  if (checked.required_prefix > prefix)
    return RNS8_RANGE_ERROR;
  return RNS8_SUCCESS;
}
rns8_status hip_direct_enqueue_native_product_integrity(
    int device_id,const void* a,const void* b,const void* candidate,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,bool unsigned_inputs,const uint64_t bound[3],
    const IntegrityIdentity& expected,const IntegrityIdentity& candidate_identity,
    void* device_flag,void* pinned_host_flag,void* stream) {
  if(!device_flag || !pinned_host_flag || !stream)return RNS8_INVALID_ARGUMENT;
  const auto status=validate_native_integrity(a,b,candidate,m,n,k,lda,ldb,unsigned_inputs,bound,expected,candidate_identity);
  if(status!=RNS8_SUCCESS)return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(set_hip_device(device_id)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  auto hip_stream=static_cast<hipStream_t>(stream);
  *static_cast<uint32_t*>(pinned_host_flag)=UINT32_MAX;
  if(hipMemsetAsync(device_flag,0,sizeof(uint32_t),hip_stream)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  const auto launch=static_cast<hipError_t>(rns8_hip_integrity_check(a,b,static_cast<const int8_t*>(candidate),
      int(m),int(n),int(k),int(lda),int(ldb),int(expected.prefix),unsigned_inputs,bound,
      static_cast<uint32_t*>(device_flag),stream));
  if(launch!=hipSuccess)return RNS8_BACKEND_FAILURE;
  return hipMemcpyAsync(pinned_host_flag,device_flag,sizeof(uint32_t),hipMemcpyDeviceToHost,hip_stream)==hipSuccess
      ?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_check_native_product_integrity(
    int device_id, const void *a, const void *b, const void *candidate,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb,
    bool unsigned_inputs, const uint64_t bound[3],
    const IntegrityIdentity &expected,
    const IntegrityIdentity &candidate_identity, uint32_t &failures) {
  failures = 0;
  const auto admitted = validate_native_integrity(a,b,candidate,m,n,k,lda,ldb,unsigned_inputs,bound,expected,candidate_identity);
  if(admitted!=RNS8_SUCCESS)return admitted;
  const auto prefix=expected.prefix;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  hip_direct_device_buffer flag;
  status = flag.allocate(device_id, sizeof(uint32_t));
  if (status != RNS8_SUCCESS)
    return status;
  status = hip_direct_zero(device_id, flag.get(), sizeof(uint32_t));
  if (status != RNS8_SUCCESS)
    return status;
  const auto error =
      timed_hip_operation("independent_native_product_integrity", [&]() {
        const auto launch = static_cast<hipError_t>(rns8_hip_integrity_check(
            a, b, static_cast<const int8_t *>(candidate), static_cast<int>(m),
            static_cast<int>(n), static_cast<int>(k), static_cast<int>(lda),
            static_cast<int>(ldb), prefix, unsigned_inputs, bound,
            static_cast<uint32_t *>(flag.get()), nullptr));
        return launch == hipSuccess ? hipDeviceSynchronize() : launch;
      });
  if (error != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  status = hip_direct_copy_device_to_host(device_id, &failures, flag.get(),
                                          sizeof(failures));
  if (status != RNS8_SUCCESS)
    return status;
  return failures ? RNS8_VERIFICATION_FAILED : RNS8_SUCCESS;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
