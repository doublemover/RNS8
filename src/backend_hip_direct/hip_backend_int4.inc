#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_int4_capability(int*);
extern "C" int rns8_hip_finite_int4_gemm(const int8_t*, const int8_t*, int8_t*, int, int, int, int, int, int, int);
#endif

rns8_status hip_direct_gemm_finite_int4_device(
    int device_id, const int8_t* a, const int8_t* b, int8_t* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc, uint32_t modulus) {
  if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      modulus < 2 || modulus > 16) return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_gemm_address(m, n, k, lda, ldb, ldc)) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t bytes = static_cast<uint64_t>(m) * ldc;
  if (!nonoverlapping_byte_ranges(c, bytes, a, static_cast<uint64_t>(m) * lda) ||
      !nonoverlapping_byte_ranges(c, bytes, b, static_cast<uint64_t>(k) * ldb)) return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer capability;
  status = capability.allocate(device_id, sizeof(int));
  if (status != RNS8_SUCCESS) return status;
  const auto probe = timed_hip_operation("finite_int4_device_capability", [&]() {
    auto error = static_cast<hipError_t>(rns8_hip_int4_capability(static_cast<int*>(capability.get())));
    return error == hipSuccess ? hipDeviceSynchronize() : error;
  });
  if (probe != hipSuccess) return RNS8_BACKEND_FAILURE;
  int supported = 0;
  status = hip_direct_copy_device_to_host(device_id, &supported, capability.get(), sizeof(supported));
  if (status != RNS8_SUCCESS) return status;
  if (!supported) return RNS8_UNSUPPORTED_BACKEND;
  const auto error = timed_hip_operation("finite_int4_rdna3_wmma_kernel", [&]() {
    auto launched = static_cast<hipError_t>(rns8_hip_finite_int4_gemm(
        a, b, c, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        static_cast<int>(lda), static_cast<int>(ldb), static_cast<int>(ldc), static_cast<int>(modulus)));
    return launched == hipSuccess ? hipDeviceSynchronize() : launched;
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_prepare_int4_capability(int device_id) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS)return status;
  hip_direct_device_buffer capability;status=capability.allocate(device_id,sizeof(int));if(status!=RNS8_SUCCESS)return status;
  auto error=static_cast<hipError_t>(rns8_hip_int4_capability(static_cast<int*>(capability.get())));
  if(error==hipSuccess)error=hipDeviceSynchronize();if(error!=hipSuccess)return RNS8_BACKEND_FAILURE;
  int supported=0;status=hip_direct_copy_device_to_host(device_id,&supported,capability.get(),sizeof(supported));
  if(status!=RNS8_SUCCESS)return status;return supported?RNS8_SUCCESS:RNS8_UNSUPPORTED_BACKEND;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_finite_int4_gemm_packed(const int8_t*,const uint32_t*,int8_t*,int,int,int,int,int,int);
extern "C" int rns8_hip_pack_finite_int4_b(const int8_t*,uint32_t*,int,int,int,int);
#endif
rns8_status hip_direct_pack_int4_b(int device_id,const int8_t* b,void* packed,int64_t k,int64_t n,int64_t ldb,uint32_t modulus) {
  if(!b || !packed || modulus<2 || modulus>16 || !qualified_i32_matrix_address(k,n,ldb))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS)return status;
  auto error=static_cast<hipError_t>(rns8_hip_pack_finite_int4_b(b,static_cast<uint32_t*>(packed),int(k),int(n),int(ldb),int(modulus)));
  if(error==hipSuccess)error=hipDeviceSynchronize();return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_gemm_int4_admitted(int device_id,const int8_t* a,const int8_t* b,const void* packed_b,int8_t* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,uint32_t modulus) {
  if(!a || !b || !c || modulus<2 || modulus>16 || !qualified_i32_gemm_address(m,n,k,lda,ldb,ldc))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS)return status;
  auto error=timed_hip_operation(packed_b?"finite_int4_prepacked_b_kernel":"finite_int4_onthefly_kernel",[&](){
    auto result=static_cast<hipError_t>(packed_b?rns8_hip_finite_int4_gemm_packed(a,static_cast<const uint32_t*>(packed_b),c,int(m),int(n),int(k),int(lda),int(ldc),int(modulus)):
      rns8_hip_finite_int4_gemm(a,b,c,int(m),int(n),int(k),int(lda),int(ldb),int(ldc),int(modulus)));
    return result==hipSuccess?hipDeviceSynchronize():result;});
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)packed_b;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
