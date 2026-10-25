#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_adaptive_native(const void*, const void*, int8_t*,
    const NativeTileBound*, const uint32_t*, int, int, int, int, int, int, int, bool, const uint32_t*, uint32_t);
extern "C" int rns8_hip_adaptive_native_suffix(int8_t*, const NativeTileBound*, int, int, int, bool, int*);
#endif

rns8_status hip_direct_gemm_adaptive_native_device(
    int device_id, const void* a, const void* b, int8_t* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    bool unsigned_inputs, uint32_t capacity, uint32_t group,
    uint64_t max_scratch_bytes, bool& commit_started, NativeBoundMethod method, uint32_t panel_k, uint32_t compact_workers,
    const void* retained_summaries) {
  commit_started = false;
  if (!a || !b || !c) return RNS8_INVALID_ARGUMENT;
  const auto support = query_adaptive_native_support(m,n,k,lda,ldb,ldc,capacity,group,
      method,panel_k,compact_workers,max_scratch_bytes);
  if (support.status != RNS8_SUCCESS) return support.status;
  const uint64_t output_bytes = static_cast<uint64_t>(m) * ldc * capacity;
  if (!nonoverlapping_byte_ranges(c, output_bytes, a, static_cast<uint64_t>(m) * lda * 8) ||
      !nonoverlapping_byte_ranges(c, output_bytes, b, static_cast<uint64_t>(k) * ldb * 8))
    return RNS8_INVALID_ARGUMENT;
  const auto tiles = support.tiles, stage_bytes = support.stage_bytes;
  const auto bound_bytes = support.bound_bytes, metadata_bytes = support.metadata_bytes;
  const auto reserved = support.reserved_bytes;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer stage, records, metadata;
  if ((status = stage.allocate(device_id, static_cast<size_t>(stage_bytes))) != RNS8_SUCCESS ||
      (status = records.allocate(device_id, static_cast<size_t>(bound_bytes))) != RNS8_SUCCESS ||
      (status = metadata.allocate(device_id, static_cast<size_t>(metadata_bytes))) != RNS8_SUCCESS) return status;
  auto* output = static_cast<int8_t*>(stage.get());
  auto* bounds = static_cast<NativeTileBound*>(records.get());
  auto* indices = static_cast<uint32_t*>(metadata.get());
  auto* offsets = indices + tiles;
  auto* device_status = reinterpret_cast<int*>(offsets + 22);
  auto run = [](const char* label, auto operation) {
    return timed_hip_operation(label, [&]() {
      const auto error = static_cast<hipError_t>(operation());
      return error == hipSuccess ? hipDeviceSynchronize() : error;
    });
  };
  if (run("adaptive_native_capability", [&]() { return rns8_hip_native_wmma_capability(device_status); }) != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  int flag = 0;
  if ((status = hip_direct_copy_device_to_host(device_id, &flag, device_status, sizeof(flag))) != RNS8_SUCCESS) return status;
  if (!flag) return RNS8_UNSUPPORTED_BACKEND;
  status = hip_direct_discover_native_tile_bounds(device_id, a, b, m, n, k, lda, ldb,
      unsigned_inputs, capacity, bounds, indices, offsets, max_scratch_bytes - reserved, method, panel_k, retained_summaries);
  if (status != RNS8_SUCCESS) return status;
  if (hipMemset(output, 0, static_cast<size_t>(stage_bytes)) != hipSuccess ||
      hipMemset(device_status, 0, sizeof(int)) != hipSuccess) return RNS8_BACKEND_FAILURE;
  if (run("adaptive_native_wmma", [&]() {
        return rns8_hip_adaptive_native(a, b, output, bounds, indices, static_cast<int>(m), static_cast<int>(n),
            static_cast<int>(k), static_cast<int>(lda), static_cast<int>(ldb), capacity, group, unsigned_inputs, offsets, compact_workers);
      }) != hipSuccess) return RNS8_BACKEND_FAILURE;
  if (run("adaptive_native_exact_suffix", [&]() {
        return rns8_hip_adaptive_native_suffix(output, bounds, static_cast<int>(m), static_cast<int>(n),
            capacity, !unsigned_inputs, device_status);
      }) != hipSuccess) return RNS8_BACKEND_FAILURE;
  if ((status = hip_direct_copy_device_to_host(device_id, &flag, device_status, sizeof(flag))) != RNS8_SUCCESS) return status;
  if (flag) return RNS8_RANGE_ERROR;
  commit_started = true;
  return run("adaptive_native_commit", [&]() {
    return rns8_hip_strassen_commit(output, c, static_cast<int>(m), static_cast<int>(n), static_cast<int>(ldc), capacity);
  }) == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id; (void)unsigned_inputs;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
