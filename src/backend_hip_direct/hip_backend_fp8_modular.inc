#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_fp8_modular_capability(int*);
extern "C" int rns8_hip_mxfp_capability(int*);
extern "C" int rns8_hip_mxfp_modular(const int8_t*,const int8_t*,int8_t*,int,int,int,int,int,int,int,unsigned*,int32_t*,uint32_t,const uint8_t*,const uint8_t*);
extern "C" int rns8_hip_fp8_modular(const int8_t*,const int8_t*,int8_t*,int,int,int,int,int,int,int,unsigned*,int32_t*,bool,uint32_t);
#endif
rns8_status hip_direct_fp8_modular_prepare(int device,void* probe,uint32_t format) {
  if(!probe || format<1 || format>10)return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device);if(status!=RNS8_SUCCESS)return status;
  const auto launch=hipError_t((format>=3?rns8_hip_mxfp_capability(static_cast<int*>(probe)):rns8_hip_fp8_modular_capability(static_cast<int*>(probe))));
  const auto drained=hipDeviceSynchronize();if(launch!=hipSuccess || drained!=hipSuccess)return RNS8_BACKEND_FAILURE;
  int available=0;const auto copied=hipMemcpy(&available,probe,4,hipMemcpyDeviceToHost);
  return copied!=hipSuccess?RNS8_BACKEND_FAILURE:available==int(format>=3?1:format)?RNS8_SUCCESS:RNS8_UNSUPPORTED_BACKEND;
#else
  (void)device;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_fp8_modular_gemm(int device,const int8_t* a,const int8_t* b,int8_t* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,uint32_t modulus,void* failure,void* product_planes,bool three,uint32_t format) {
  if(!a || !b || !c || !failure || reinterpret_cast<uintptr_t>(failure)%4 || format<1 || format>2 || modulus<2 || modulus>256 || !qualified_i32_gemm_address(m,n,k,lda,ldb,ldc))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device);if(status!=RNS8_SUCCESS)return status;
  if(hipMemset(failure,0,4)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  const auto error=timed_hip_operation(product_planes?"finite_fp8_materialized_products_and_recombine":three?"finite_fp8_three_product_kernel":"finite_fp8_four_product_kernel",[&] {
    const auto launch=hipError_t(rns8_hip_fp8_modular(a,b,c,int(m),int(n),int(k),int(lda),int(ldb),int(ldc),int(modulus),static_cast<unsigned*>(failure),static_cast<int32_t*>(product_planes),three,format));
    const auto done=hipDeviceSynchronize();return launch==hipSuccess?done:launch;
  });if(error!=hipSuccess)return RNS8_BACKEND_FAILURE;
  unsigned flags=0;if(hipMemcpy(&flags,failure,4,hipMemcpyDeviceToHost)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  return flags?RNS8_VERIFICATION_FAILED:RNS8_SUCCESS;
#else
  (void)device;return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_mxfp_modular_gemm(int device,const int8_t* a,const int8_t* b,int8_t* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,uint32_t modulus,
    void* failure,void* product_planes,uint32_t format,const uint8_t* a_scales,const uint8_t* b_scales) {
  if(!a || !b || !c || !failure || !a_scales || !b_scales || reinterpret_cast<uintptr_t>(failure)%4 ||
      format<3 || format>10 || modulus<2 || modulus>256 || !qualified_i32_gemm_address(m,n,k,lda,ldb,ldc))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device);if(status!=RNS8_SUCCESS)return status;
  if(hipMemset(failure,0,4)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  const auto error=timed_hip_operation(product_planes?"finite_mxfp_materialized_scale_barriers":"finite_mxfp_fused_scale_barriers",[&] {
    const auto launch=hipError_t(rns8_hip_mxfp_modular(a,b,c,int(m),int(n),int(k),int(lda),int(ldb),int(ldc),int(modulus),
        static_cast<unsigned*>(failure),static_cast<int32_t*>(product_planes),format,a_scales,b_scales));
    const auto done=hipDeviceSynchronize();return launch==hipSuccess?done:launch;
  });if(error!=hipSuccess)return RNS8_BACKEND_FAILURE;
  unsigned flags=0;if(hipMemcpy(&flags,failure,4,hipMemcpyDeviceToHost)!=hipSuccess)return RNS8_BACKEND_FAILURE;
  return flags?RNS8_VERIFICATION_FAILED:RNS8_SUCCESS;
#else
  (void)device;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
