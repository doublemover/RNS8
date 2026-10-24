#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_rank23_slp_transform(const int8_t*,int8_t*,int,int,int,int,int,int,bool);
extern "C" int rns8_hip_rank23_slp_combine(const int8_t*,int8_t*,int,int,int,int,int);
#endif
rns8_status hip_direct_gemm_rank23_slp_device(
    int device_id,const void *a,const void *b,void *c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget,bool &commit_started) {
  commit_started = false;
  if (!a || !b || !c) return RNS8_INVALID_ARGUMENT;
  const auto resources = query_rank23_slp_resources(m,n,k,lda,ldb,ldc,prefix,budget);
  if (resources.status != RNS8_SUCCESS) return resources.status;
  if (!nonoverlapping_byte_ranges(c,resources.destination.byte_extent,a,resources.source_a.byte_extent) ||
      !nonoverlapping_byte_ranges(c,resources.destination.byte_extent,b,resources.source_b.byte_extent))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer scratch;
  status = scratch.allocate(device_id,resources.total_bytes);
  if (status != RNS8_SUCCESS) return status;
  auto *left = static_cast<int8_t*>(scratch.get());
  auto *right = left+resources.b_offset;
  auto *products = left+resources.product_offset;
  auto *stage = left+resources.output_offset;
  const int hm = static_cast<int>(resources.block_m), hn = static_cast<int>(resources.block_n),
            hk = static_cast<int>(resources.block_k);
  auto error = timed_hip_operation("rank23_slp_basis_transform_pair",[&]() {
    auto launch = static_cast<hipError_t>(rns8_hip_rank23_slp_transform(
        static_cast<const int8_t*>(a),left,static_cast<int>(m),static_cast<int>(k),
        static_cast<int>(lda),hm,hk,prefix,true));
    if (launch != hipSuccess) return launch;
    launch = static_cast<hipError_t>(rns8_hip_rank23_slp_transform(
        static_cast<const int8_t*>(b),right,static_cast<int>(k),static_cast<int>(n),
        static_cast<int>(ldb),hk,hn,prefix,false));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  for (int term = 0; term < 23; ++term) {
    status = hip_direct_gemm_rns_device(device_id,
        left+uint64_t(term)*resources.a_bytes,right+uint64_t(term)*resources.b_bytes,
        products+uint64_t(term)*resources.product_bytes,hm,hn,hk,hk,hn,hn,prefix);
    if (status != RNS8_SUCCESS) return status;
  }
  error = timed_hip_operation("rank23_slp_recombine_output_basis",[&]() {
    const auto launch = static_cast<hipError_t>(rns8_hip_rank23_slp_combine(
        products,stage,static_cast<int>(m),static_cast<int>(n),hm,hn,prefix));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  commit_started = true;
  error = timed_hip_operation("rank23_slp_commit",[&]() {
    const auto launch = static_cast<hipError_t>(rns8_hip_strassen_commit(
        stage,static_cast<int8_t*>(c),static_cast<int>(m),static_cast<int>(n),static_cast<int>(ldc),prefix));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
