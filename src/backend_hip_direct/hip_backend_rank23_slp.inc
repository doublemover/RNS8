#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_rank23_slp_transform(const int8_t*,int8_t*,int,int,int,int,int,int,bool);
extern "C" int rns8_hip_rank23_slp_combine(const int8_t*,int8_t*,int,int,int,int,int);
extern "C" int rns8_hip_rank23_oct55_transform(const int8_t*,int8_t*,int,int,int,int,int,int,bool,int);
extern "C" int rns8_hip_rank23_oct55_combine(const int8_t*,int8_t*,int,int,int,int,int);
#endif
static rns8_status hip_direct_gemm_rank23_dag_device(
    int device_id,const void *a,const void *b,void *c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget,bool &commit_started,bool october,bool recompute,void* retained_scratch,uint64_t retained_bytes) {
  commit_started = false;
  if (!a || !b || !c) return RNS8_INVALID_ARGUMENT;
  const auto resources = query_rank23_slp_resources(m,n,k,lda,ldb,ldc,prefix,budget,recompute);
  if (resources.status != RNS8_SUCCESS) return resources.status;
  if (!nonoverlapping_byte_ranges(c,resources.destination.byte_extent,a,resources.source_a.byte_extent) ||
      !nonoverlapping_byte_ranges(c,resources.destination.byte_extent,b,resources.source_b.byte_extent))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer scratch;
  if(retained_scratch && retained_bytes<resources.total_bytes)return RNS8_WORKSPACE_TOO_SMALL;
  if(!retained_scratch) {status=scratch.allocate(device_id,resources.total_bytes);if(status!=RNS8_SUCCESS)return status;}
  auto *left = static_cast<int8_t*>(retained_scratch?retained_scratch:scratch.get());
  auto *right = left+resources.b_offset;
  auto *products = left+resources.product_offset;
  auto *stage = left+resources.output_offset;
  const int hm = static_cast<int>(resources.block_m), hn = static_cast<int>(resources.block_n),
            hk = static_cast<int>(resources.block_k);
  auto transform=[&](int selected) {return timed_hip_operation(october?"rank23_oct55_transform_pair":"rank23_slp_basis_transform_pair",[&]() {
    auto launch = static_cast<hipError_t>(october?rns8_hip_rank23_oct55_transform(
        static_cast<const int8_t*>(a),left,int(m),int(k),int(lda),hm,hk,prefix,true,selected):
        rns8_hip_rank23_slp_transform(static_cast<const int8_t*>(a),left,int(m),int(k),int(lda),hm,hk,prefix,true));
    if (launch == hipSuccess) {
      launch = static_cast<hipError_t>(october?rns8_hip_rank23_oct55_transform(
          static_cast<const int8_t*>(b),right,int(k),int(n),int(ldb),hk,hn,prefix,false,selected):
          rns8_hip_rank23_slp_transform(static_cast<const int8_t*>(b),right,int(k),int(n),int(ldb),hk,hn,prefix,false));
    }
    // A may already use scratch when B's launch fails. Drain submitted work
    // before releasing the owner, preserving the original launch diagnostic.
    const auto completion = hipDeviceSynchronize();
    return launch != hipSuccess ? launch : completion;
  });};
  auto error=hipSuccess;
  if(!recompute)error=transform(-1);
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  for (int term = 0; term < 23; ++term) {
    if(recompute && transform(term)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    status = hip_direct_gemm_rns_device(device_id,
        left+uint64_t(recompute?0:term)*resources.a_bytes,right+uint64_t(recompute?0:term)*resources.b_bytes,
        products+uint64_t(term)*resources.product_bytes,hm,hn,hk,hk,hn,hn,prefix);
    if (status != RNS8_SUCCESS) return status;
  }
  error = timed_hip_operation(october?"rank23_oct55_recombine":"rank23_slp_recombine_output_basis",[&]() {
    const auto launch = static_cast<hipError_t>(october?rns8_hip_rank23_oct55_combine(
        products,stage,int(m),int(n),hm,hn,prefix):rns8_hip_rank23_slp_combine(products,stage,int(m),int(n),hm,hn,prefix));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  commit_started = true;
  error = timed_hip_operation(october?"rank23_oct55_commit":"rank23_slp_commit",[&]() {
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

rns8_status hip_direct_gemm_rank23_slp_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget,bool& commit_started,
    void* retained_scratch,uint64_t retained_bytes) {
  return hip_direct_gemm_rank23_dag_device(device_id,a,b,c,m,n,k,lda,ldb,ldc,prefix,budget,commit_started,
      false,false,retained_scratch,retained_bytes);
}
rns8_status hip_direct_gemm_rank23_oct55_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget,bool& commit_started,
    void* retained_scratch,uint64_t retained_bytes) {
  return hip_direct_gemm_rank23_dag_device(device_id,a,b,c,m,n,k,lda,ldb,ldc,prefix,budget,commit_started,
      true,false,retained_scratch,retained_bytes);
}
rns8_status hip_direct_gemm_rank23_oct55_recompute_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget,bool& commit_started,
    void* retained_scratch,uint64_t retained_bytes) {
  return hip_direct_gemm_rank23_dag_device(device_id,a,b,c,m,n,k,lda,ldb,ldc,prefix,budget,commit_started,
      true,true,retained_scratch,retained_bytes);
}
