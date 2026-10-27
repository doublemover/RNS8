#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_split_k_produce(const int8_t*, const int8_t*, uint8_t*, int, int, int, int, int, int, int);
extern "C" int rns8_hip_split_k_merge(const uint8_t*, uint8_t*, int64_t, int, int);
extern "C" int rns8_hip_split_k_store(const uint8_t*, int8_t*, int, int, int, int);
extern "C" int rns8_hip_split_k_window_produce(const int8_t*,const int8_t*,uint8_t*,int,int,int,int,int,int,int,int,int);
extern "C" int rns8_hip_split_k_window_accumulate(const uint8_t*,uint8_t*,int64_t,int,bool);
#endif

rns8_status hip_direct_gemm_rns_split_k_device(
    int device_id, const void* a, const void* b, void* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    uint32_t prefix, uint32_t splits, uint64_t max_scratch_bytes) {
  if (!a || !b || !c || a == c || b == c || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || !prefix || prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      !splits || splits > 4096 || splits > static_cast<uint64_t>(k)) return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_gemm_address(m, n, k, lda, ldb, ldc, prefix)) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t output_bytes = static_cast<uint64_t>(m) * ldc * prefix;
  if (!nonoverlapping_byte_ranges(c, output_bytes, a, static_cast<uint64_t>(m) * lda * prefix) ||
      !nonoverlapping_byte_ranges(c, output_bytes, b, static_cast<uint64_t>(k) * ldb * prefix))
    return RNS8_INVALID_ARGUMENT;
  if ((static_cast<uint64_t>(k) + splits - 1) / splits > 65536) return RNS8_RANGE_ERROR;
  const uint64_t plane_cells = static_cast<uint64_t>(m) * n * prefix;
  const uint64_t merge_slots = (splits + 127) / 128;
  if (plane_cells > SIZE_MAX / (splits + merge_slots)) return RNS8_RANGE_ERROR;
  const uint64_t bytes = plane_cells * (splits + merge_slots);
  if (bytes > max_scratch_bytes) return RNS8_WORKSPACE_TOO_SMALL;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer scratch;
  status = scratch.allocate(device_id, static_cast<std::size_t>(bytes));
  if (status != RNS8_SUCCESS) return status;
  auto* first = static_cast<uint8_t*>(scratch.get());
  auto* second = first + plane_cells * splits;
  auto run = [](const char* label, auto operation) {
    return timed_hip_operation(label, [&]() {
      const auto error = static_cast<hipError_t>(operation());
      return error == hipSuccess ? hipDeviceSynchronize() : error;
    });
  };
  if (run("split_k_producer_kernel", [&]() {
        return rns8_hip_split_k_produce(static_cast<const int8_t*>(a), static_cast<const int8_t*>(b), first,
                                      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                      static_cast<int>(lda), static_cast<int>(ldb), prefix, splits);
      }) != hipSuccess) return RNS8_BACKEND_FAILURE;
  uint32_t fragments = splits;
  while (fragments > 1) {
    if (run("split_k_merge_kernel", [&]() {
          return rns8_hip_split_k_merge(first, second, m * n, prefix, fragments);
        }) != hipSuccess) return RNS8_BACKEND_FAILURE;
    std::swap(first, second);
    fragments = (fragments + 127) / 128;
  }
  return run("split_k_store_kernel", [&]() {
    return rns8_hip_split_k_store(first, static_cast<int8_t*>(c), static_cast<int>(m), static_cast<int>(n),
                                static_cast<int>(ldc), prefix);
  }) == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

// Reusable owner path: no helper-owned GPU allocation can disappear on failure.
rns8_status hip_direct_gemm_rns_split_k_workspace(
    int device_id,const void* a,const void* b,void* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,
    uint32_t prefix,uint32_t splits,void* scratch,uint64_t scratch_bytes,bool& commit_started,uint32_t fragment_window) {
  commit_started=false;rns8_split_k_resources resources{};
  const auto support=query_split_k_window_resources(m,n,k,lda,ldb,ldc,prefix,splits,
      fragment_window?fragment_window:splits,scratch_bytes,resources);
  if(support!=RNS8_SUCCESS) return support;
  if(!a || !b || !c || !scratch) return RNS8_INVALID_ARGUMENT;
  const uint64_t a_bytes=uint64_t(m)*lda*prefix,b_bytes=uint64_t(k)*ldb*prefix,c_bytes=uint64_t(m)*ldc*prefix;
  if(!nonoverlapping_byte_ranges(a,a_bytes,c,c_bytes) || !nonoverlapping_byte_ranges(b,b_bytes,c,c_bytes) ||
     !nonoverlapping_byte_ranges(scratch,resources.scratch_bytes,a,a_bytes) ||
     !nonoverlapping_byte_ranges(scratch,resources.scratch_bytes,b,b_bytes) ||
     !nonoverlapping_byte_ranges(scratch,resources.scratch_bytes,c,c_bytes)) return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  auto* first=static_cast<uint8_t*>(scratch);auto* second=first+resources.partial_bytes;
  const auto run=[](const char* label,auto operation) {
    bool completed=false;
    const auto launch=timed_hip_operation(label,[&]() {
      const auto error=static_cast<hipError_t>(operation());
      if(error!=hipSuccess) return error;
      const auto sync=hipDeviceSynchronize();completed=sync==hipSuccess;return sync;
    });
    // Drain launch/event failure unless completion was already established.
    // Successful calls need one boundary, not a redundant second device drain.
    if(!completed) completed=hipDeviceSynchronize()==hipSuccess;
    return launch==hipSuccess && completed;
  };
  if(resources.fragment_window==splits) {
    if(!run("split_k_producer_kernel",[&]() {return rns8_hip_split_k_produce(
        static_cast<const int8_t*>(a),static_cast<const int8_t*>(b),first,
        int(m),int(n),int(k),int(lda),int(ldb),prefix,splits);})) return RNS8_BACKEND_FAILURE;
    for(uint32_t fragments=splits;fragments>1;fragments=(fragments+127)/128) {
      if(!run("split_k_merge_kernel",[&]() {return rns8_hip_split_k_merge(first,second,m*n,prefix,fragments);}))
        return RNS8_BACKEND_FAILURE;
      std::swap(first,second);
    }
  } else {
    auto* const partial=first;auto* const merge=second;
    auto* const running=merge+resources.merge_bytes;
    for(uint32_t base=0;base<splits;base+=resources.fragment_window) {
      const auto count=std::min(resources.fragment_window,splits-base);
      first=partial;second=merge;
      if(!run("split_k_window_producer_kernel",[&]() {return rns8_hip_split_k_window_produce(
          static_cast<const int8_t*>(a),static_cast<const int8_t*>(b),first,
          int(m),int(n),int(k),int(lda),int(ldb),prefix,splits,base,count);})) return RNS8_BACKEND_FAILURE;
      for(uint32_t fragments=count;fragments>1;fragments=(fragments+127)/128) {
        if(!run("split_k_window_merge_kernel",[&]() {return rns8_hip_split_k_merge(first,second,m*n,prefix,fragments);}))
          return RNS8_BACKEND_FAILURE;
        std::swap(first,second);
      }
      if(!run("split_k_window_accumulate_kernel",[&]() {return rns8_hip_split_k_window_accumulate(
          first,running,m*n,prefix,base==0);})) return RNS8_BACKEND_FAILURE;
    }
    first=running;
  }
  commit_started=true;
  return run("split_k_store_kernel",[&]() {return rns8_hip_split_k_store(first,static_cast<int8_t*>(c),
      int(m),int(n),int(ldc),prefix);})?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
