#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_split_k_canonicalize(uint8_t*, int64_t, int, int);
extern "C" int rns8_hip_split_k_merge_centered(const uint8_t*,uint8_t*,int64_t,int,int);
extern "C" int rns8_hip_split_k_store_centered(const uint8_t*,int8_t*,int,int,int,int);
#endif

rns8_status hip_direct_gemm_native_wmma_split_k_device(
    int device_id, const void* a, const void* b, int8_t* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    uint32_t prefix, uint32_t group, bool unsigned_inputs, uint32_t splits,
    uint32_t workers, uint64_t max_scratch_bytes, bool& commit_started, bool separate_canonicalization_control,const rns8_locality_desc* locality,rns8_locality_info* locality_info) try {
  commit_started = false;
  if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      !prefix || prefix > 20 || (group != 1 && group != 2 && group != 4) ||
      !splits || splits > 4096 || splits > static_cast<uint64_t>(k) || !workers || workers > 1024)
    return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_matrix_address(m, k, lda, 1, 8) ||
      !qualified_i32_matrix_address(k, n, ldb, 1, 8) ||
      !qualified_i32_matrix_address(m, n, ldc, prefix)) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t output_bytes = static_cast<uint64_t>(m) * ldc * prefix;
  if (!nonoverlapping_byte_ranges(c, output_bytes, a, static_cast<uint64_t>(m) * lda * 8) ||
      !nonoverlapping_byte_ranges(c, output_bytes, b, static_cast<uint64_t>(k) * ldb * 8)) return RNS8_INVALID_ARGUMENT;
  const uint64_t plane_cells = static_cast<uint64_t>(m) * n * prefix;
  const uint64_t merge_slots = (splits + 127) / 128;
  if (plane_cells > SIZE_MAX / (splits + merge_slots)) return RNS8_RANGE_ERROR;
  const uint64_t bytes = plane_cells * (splits + merge_slots);
  if (bytes > max_scratch_bytes) return RNS8_WORKSPACE_TOO_SMALL;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_probe_buffer scratch;
  status = scratch.allocate(device_id, static_cast<std::size_t>(bytes));
  if (status != RNS8_SUCCESS) return status;
  auto* first = static_cast<uint8_t*>(scratch.get());
  auto* second = first + plane_cells * splits;
  std::vector<NativeWmmaTask> tasks;
  tasks.reserve(splits);
  for (uint32_t fragment = 0; fragment < splits; ++fragment) {
    const int64_t begin = k * fragment / splits, end = k * (fragment + 1) / splits;
    tasks.push_back({static_cast<const uint64_t*>(a) + begin, static_cast<const uint64_t*>(b) + begin * ldb,
                     reinterpret_cast<int8_t*>(first + plane_cells * fragment), m, n, end - begin,
                     lda, ldb, n, prefix, group, static_cast<uint32_t>(unsigned_inputs)});
  }
  bool scratch_may_be_written = false;
  scratch.mark_pending();
  if(locality){NativeWmmaPreparedQueue queue;status=queue.prepare(device_id,tasks,workers,max_scratch_bytes-bytes,{},locality);
    if(status==RNS8_SUCCESS){if(locality_info)*locality_info=queue.locality_info();status=queue.execute(scratch_may_be_written);}
    const auto closed=queue.close();if(status==RNS8_SUCCESS)status=closed;
  }else status = hip_direct_gemm_native_wmma_queue(device_id, tasks, workers, max_scratch_bytes - bytes, scratch_may_be_written);
  if(status!=RNS8_SUCCESS){if(hipDeviceSynchronize()==hipSuccess)scratch.mark_completed();return status;}
  scratch.mark_completed();
  auto run = [&](const char* label, auto operation) {
    return timed_hip_operation(label, [&]() {
      scratch.mark_pending();const auto error = static_cast<hipError_t>(operation());
      const auto drained=hipDeviceSynchronize();if(drained==hipSuccess)scratch.mark_completed();
      return error==hipSuccess?drained:error;
    });
  };
  if (separate_canonicalization_control && run("native_split_k_canonicalize", [&]() {
        return rns8_hip_split_k_canonicalize(first, m * n, prefix, splits);
      }) != hipSuccess) return RNS8_BACKEND_FAILURE;
  uint32_t fragments = splits;
  bool centered_input = !separate_canonicalization_control;
  while (fragments > 1) {
    if (run(centered_input ? "native_split_k_first_merge_centered" : "native_split_k_merge", [&]() {
          return centered_input ? rns8_hip_split_k_merge_centered(first, second, m * n, prefix, fragments)
                                : rns8_hip_split_k_merge(first, second, m * n, prefix, fragments);
        }) != hipSuccess) return RNS8_BACKEND_FAILURE;
    centered_input = false;
    std::swap(first, second);
    fragments = (fragments + 127) / 128;
  }
  commit_started = true;
  return run("native_split_k_store", [&]() {
    // F=1 never entered the merge tree: retain its centered format explicitly.
    return centered_input ? rns8_hip_split_k_store_centered(first,c,static_cast<int>(m),static_cast<int>(n),static_cast<int>(ldc),prefix)
                          : rns8_hip_split_k_store(first,c,static_cast<int>(m),static_cast<int>(n),static_cast<int>(ldc),prefix);
  }) == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id; (void)unsigned_inputs; (void)separate_canonicalization_control;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
  return RNS8_INTERNAL_ERROR;
}
