#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_strassen_transform(const int8_t *, int8_t *, int, int,
                                           int, int, int, int, int, bool, int);
extern "C" int rns8_hip_strassen_combine(const int8_t *, int8_t *, int, int,
                                         int, int, int, int, int);
extern "C" int rns8_hip_strassen_commit(const int8_t *, int8_t *, int, int, int,
                                        int);
#endif
static rns8_status hip_direct_gemm_bilinear_device(
    int device_id, const void *a, const void *b, void *c, int64_t m, int64_t n,
    int64_t k, int64_t lda, int64_t ldb, int64_t ldc, uint32_t prefix,
    uint64_t max_scratch_bytes, bool &commit_started, uint32_t blocks, void* retained_scratch,uint64_t retained_bytes,bool matrix_subproducts) {
  const bool rank23=blocks==3,rank48=blocks==4;
  commit_started = false;
  if (!a || !b || !c)
    return RNS8_INVALID_ARGUMENT;
  const auto resources = query_bilinear_resources(
      m, n, k, lda, ldb, ldc, prefix, blocks, max_scratch_bytes);
  if (resources.status != RNS8_SUCCESS)
    return resources.status;
  if (!nonoverlapping_byte_ranges(c, resources.destination.byte_extent, a,
                                  resources.source_a.byte_extent) ||
      !nonoverlapping_byte_ranges(c, resources.destination.byte_extent, b,
                                  resources.source_b.byte_extent))
    return RNS8_INVALID_ARGUMENT;
  const auto hm = resources.block_m, hn = resources.block_n,
             hk = resources.block_k;
  const auto cb = resources.output_bytes;
  if(matrix_subproducts && (hk>RNS8_SAFE_INT32_K_BLOCK || !amdgpu_builtins_compiled()))return RNS8_UNSUPPORTED_BACKEND;
  const auto arena_status=validate_bilinear_arena(resources,retained_scratch,retained_bytes,a,b,c);
  if(arena_status!=RNS8_SUCCESS)return arena_status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  hip_direct_device_buffer scratch;
  if(retained_scratch && retained_bytes<resources.total_bytes)return RNS8_WORKSPACE_TOO_SMALL;
  if(!retained_scratch) {status=scratch.allocate(device_id,resources.total_bytes);if(status!=RNS8_SUCCESS)return status;}
  auto *ta = static_cast<int8_t *>(retained_scratch?retained_scratch:scratch.get());
  auto *tb = ta + resources.b_offset;
  auto *product = ta + resources.product_offset;
  auto *stage = ta + resources.output_offset;
  status = hip_direct_zero(device_id, stage, cb);
  if (status != RNS8_SUCCESS)
    return status;
  for (int term = 0; term < (rank48?48:rank23?23:7); ++term) {
    auto error = timed_hip_operation(
        rank48?"rank48_dyadic_transform_pair":rank23 ? "rank23_transform_pair" : "strassen_transform_pair", [&]() {
          auto launch = static_cast<hipError_t>(rns8_hip_strassen_transform(
              static_cast<const int8_t *>(a), ta, static_cast<int>(m),
              static_cast<int>(k), static_cast<int>(lda), static_cast<int>(hm),
              static_cast<int>(hk), prefix, term, true, int(blocks)));
          if (launch != hipSuccess)
            return launch;
          launch = static_cast<hipError_t>(rns8_hip_strassen_transform(
              static_cast<const int8_t *>(b), tb, static_cast<int>(k),
              static_cast<int>(n), static_cast<int>(ldb), static_cast<int>(hk),
              static_cast<int>(hn), prefix, term, false, int(blocks)));
          // The first transform may still own scratch if the second launch fails.
          const auto completion=hipDeviceSynchronize();
          return launch == hipSuccess ? completion : launch;
        });
    if (error != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    status = matrix_subproducts?amdgpu_builtins_gemm_rns_device(device_id,ta,tb,product,hm,hn,hk,hk,hn,hn,prefix):
        hip_direct_gemm_rns_device(device_id,ta,tb,product,hm,hn,hk,hk,hn,hn,prefix);
    if (status != RNS8_SUCCESS)
      return status;
    error = timed_hip_operation(
        rank48?"rank48_dyadic_recombine":rank23 ? "rank23_recombine" : "strassen_recombine", [&]() {
          const auto launch = static_cast<hipError_t>(rns8_hip_strassen_combine(
              product, stage, static_cast<int>(m), static_cast<int>(n),
              static_cast<int>(hm), static_cast<int>(hn), prefix, term,
              int(blocks)));
          return launch == hipSuccess ? hipDeviceSynchronize() : launch;
        });
    if (error != hipSuccess)
      return RNS8_BACKEND_FAILURE;
  }
  // Inverse2 is not defined in even channels. Execute those original planes
  // classically into the same private output stage before publishing anything.
  if(rank48)for(uint32_t plane=0;plane<prefix;++plane)if(!(kDefaultModuli[plane]&1)) {
    status=hip_direct_gemm_finite_u8_resident_device(device_id,
        static_cast<const int8_t*>(a)+uint64_t(plane)*m*lda,
        static_cast<const int8_t*>(b)+uint64_t(plane)*k*ldb,
        stage+uint64_t(plane)*m*n,m,n,k,lda,ldb,n,kDefaultModuli[plane]);
    if(status!=RNS8_SUCCESS)return status;
  }
  commit_started = true;
  const auto error =
      timed_hip_operation(rank48?"rank48_mixed_basis_commit":rank23 ? "rank23_commit" : "strassen_commit", [&]() {
        const auto launch = static_cast<hipError_t>(rns8_hip_strassen_commit(
            stage, static_cast<int8_t *>(c), static_cast<int>(m),
            static_cast<int>(n), static_cast<int>(ldc), prefix));
        return launch == hipSuccess ? hipDeviceSynchronize() : launch;
      });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)rank23;(void)rank48;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_gemm_strassen_device(
    int device_id, const void *a, const void *b, void *c, int64_t m, int64_t n,
    int64_t k, int64_t lda, int64_t ldb, int64_t ldc, uint32_t prefix,
    uint64_t max_scratch_bytes, bool &commit_started,void* retained_scratch,uint64_t retained_bytes,bool matrix_subproducts) {
  return hip_direct_gemm_bilinear_device(device_id, a, b, c, m, n, k, lda, ldb,
                                         ldc, prefix, max_scratch_bytes,
                                         commit_started, 2,retained_scratch,retained_bytes,matrix_subproducts);
}
rns8_status hip_direct_gemm_rank23_device(
    int device_id, const void *a, const void *b, void *c, int64_t m, int64_t n,
    int64_t k, int64_t lda, int64_t ldb, int64_t ldc, uint32_t prefix,
    uint64_t max_scratch_bytes, bool &commit_started,void* retained_scratch,uint64_t retained_bytes,bool matrix_subproducts) {
  return hip_direct_gemm_bilinear_device(device_id, a, b, c, m, n, k, lda, ldb,
                                         ldc, prefix, max_scratch_bytes,
                                         commit_started, 3,retained_scratch,retained_bytes,matrix_subproducts);
}
rns8_status hip_direct_gemm_rank48_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,
    uint32_t prefix,uint64_t budget,bool& commit_started,void* scratch,uint64_t bytes,bool matrix_subproducts) {
  return hip_direct_gemm_bilinear_device(device_id,a,b,c,m,n,k,lda,ldb,ldc,prefix,budget,commit_started,4,scratch,bytes,matrix_subproducts);
}
