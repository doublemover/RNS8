#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_rank23_slp_transform(const int8_t*,int8_t*,int,int,int,int,int,int,bool);
extern "C" int rns8_hip_rank23_slp_combine(const int8_t*,int8_t*,int,int,int,int,int);
extern "C" int rns8_hip_rank23_oct55_transform(const int8_t*,int8_t*,int,int,int,int,int,int,bool,int);
extern "C" int rns8_hip_rank23_oct55_combine(const int8_t*,int8_t*,int,int,int,int,int);
extern "C" int rns8_hip_rank48_slp_transform(const int8_t*,int8_t*,int,int,int,int,int,int,bool,int);
extern "C" int rns8_hip_rank48_slp_combine(const int8_t*,int8_t*,int,int,int,int,int);
#endif
static rns8_status hip_direct_gemm_rank23_dag_device(
    int device_id,const void *a,const void *b,void *c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget,bool &commit_started,bool october,bool recompute,void* retained_scratch,uint64_t retained_bytes,bool matrix_subproducts,bool rank48=false) {
  commit_started = false;
  if (!a || !b || !c) return RNS8_INVALID_ARGUMENT;
  const uint32_t product_count=rank48?48:23;
  const auto resources = query_bilinear_dag_resources(m,n,k,lda,ldb,ldc,prefix,budget,rank48?4:3,product_count,recompute);
  if (resources.status != RNS8_SUCCESS) return resources.status;
  if(matrix_subproducts && (resources.block_k>RNS8_SAFE_INT32_K_BLOCK || !amdgpu_builtins_compiled()))return RNS8_UNSUPPORTED_BACKEND;
  if (!nonoverlapping_byte_ranges(c,resources.destination.byte_extent,a,resources.source_a.byte_extent) ||
      !nonoverlapping_byte_ranges(c,resources.destination.byte_extent,b,resources.source_b.byte_extent))
    return RNS8_INVALID_ARGUMENT;
  const auto arena_status=validate_bilinear_arena(resources,retained_scratch,retained_bytes,a,b,c);
  if(arena_status!=RNS8_SUCCESS)return arena_status;
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
  auto transform=[&](int selected) {return timed_hip_operation(rank48?"rank48_slp_transform_pair":october?"rank23_oct55_transform_pair":"rank23_slp_basis_transform_pair",[&]() {
    auto launch = static_cast<hipError_t>(rank48?rns8_hip_rank48_slp_transform(
        static_cast<const int8_t*>(a),left,int(m),int(k),int(lda),hm,hk,prefix,true,selected):october?rns8_hip_rank23_oct55_transform(
        static_cast<const int8_t*>(a),left,int(m),int(k),int(lda),hm,hk,prefix,true,selected):
        rns8_hip_rank23_slp_transform(static_cast<const int8_t*>(a),left,int(m),int(k),int(lda),hm,hk,prefix,true));
    if (launch == hipSuccess) {
      launch = static_cast<hipError_t>(rank48?rns8_hip_rank48_slp_transform(
          static_cast<const int8_t*>(b),right,int(k),int(n),int(ldb),hk,hn,prefix,false,selected):october?rns8_hip_rank23_oct55_transform(
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
  if(matrix_subproducts && !recompute) {
    status=amdgpu_builtins_gemm_rns_batch_device(device_id,left,right,products,hm,hn,hk,hk,hn,hn,prefix,product_count);
    if(status!=RNS8_SUCCESS)return status;
  }
  for (int term = 0; term < int(matrix_subproducts && !recompute?0:product_count); ++term) {
    if(recompute && transform(term)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    const auto* ta=left+uint64_t(recompute?0:term)*resources.a_bytes;
    const auto* tb=right+uint64_t(recompute?0:term)*resources.b_bytes;
    auto* output=products+uint64_t(term)*resources.product_bytes;
    status=matrix_subproducts?amdgpu_builtins_gemm_rns_device(device_id,ta,tb,output,hm,hn,hk,hk,hn,hn,prefix):
        hip_direct_gemm_rns_device(device_id,ta,tb,output,hm,hn,hk,hk,hn,hn,prefix);
    if (status != RNS8_SUCCESS) return status;
  }
  error = timed_hip_operation(rank48?"rank48_slp_recombine":october?"rank23_oct55_recombine":"rank23_slp_recombine_output_basis",[&]() {
    const auto launch = static_cast<hipError_t>(rank48?rns8_hip_rank48_slp_combine(
        products,stage,int(m),int(n),hm,hn,prefix):october?rns8_hip_rank23_oct55_combine(
        products,stage,int(m),int(n),hm,hn,prefix):rns8_hip_rank23_slp_combine(products,stage,int(m),int(n),hm,hn,prefix));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  if(rank48)for(uint32_t plane=0;plane<prefix;++plane)if(!(kDefaultModuli[plane]&1)) {
    status=hip_direct_gemm_finite_u8_resident_device(device_id,
        static_cast<const int8_t*>(a)+uint64_t(plane)*m*lda,
        static_cast<const int8_t*>(b)+uint64_t(plane)*k*ldb,
        stage+uint64_t(plane)*m*n,m,n,k,lda,ldb,n,kDefaultModuli[plane]);
    if(status!=RNS8_SUCCESS)return status;
  }
  commit_started = true;
  error = timed_hip_operation(rank48?"rank48_slp_commit":october?"rank23_oct55_commit":"rank23_slp_commit",[&]() {
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
    void* retained_scratch,uint64_t retained_bytes,bool matrix_subproducts) {
  return hip_direct_gemm_rank23_dag_device(device_id,a,b,c,m,n,k,lda,ldb,ldc,prefix,budget,commit_started,
      false,false,retained_scratch,retained_bytes,matrix_subproducts);
}
rns8_status hip_direct_gemm_rank23_oct55_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget,bool& commit_started,
    void* retained_scratch,uint64_t retained_bytes,bool matrix_subproducts) {
  return hip_direct_gemm_rank23_dag_device(device_id,a,b,c,m,n,k,lda,ldb,ldc,prefix,budget,commit_started,
      true,false,retained_scratch,retained_bytes,matrix_subproducts);
}
rns8_status hip_direct_gemm_rank23_oct55_recompute_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget,bool& commit_started,
    void* retained_scratch,uint64_t retained_bytes,bool matrix_subproducts) {
  return hip_direct_gemm_rank23_dag_device(device_id,a,b,c,m,n,k,lda,ldb,ldc,prefix,budget,commit_started,
      true,true,retained_scratch,retained_bytes,matrix_subproducts);
}

rns8_status hip_direct_gemm_rank48_slp_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget,bool& commit_started,
    void* retained_scratch,uint64_t retained_bytes,bool matrix_subproducts) {
  return hip_direct_gemm_rank23_dag_device(device_id,a,b,c,m,n,k,lda,ldb,ldc,prefix,budget,commit_started,
      false,false,retained_scratch,retained_bytes,matrix_subproducts,true);
}
rns8_status hip_direct_gemm_rank48_slp_recompute_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget,bool& commit_started,
    void* retained_scratch,uint64_t retained_bytes,bool matrix_subproducts) {
  return hip_direct_gemm_rank23_dag_device(device_id,a,b,c,m,n,k,lda,ldb,ldc,prefix,budget,commit_started,
      false,true,retained_scratch,retained_bytes,matrix_subproducts,true);
}
