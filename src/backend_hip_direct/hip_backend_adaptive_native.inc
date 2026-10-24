#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_adaptive_native(const void*, const void*, int8_t*,
    const NativeTileBound*, const uint32_t*, int, int, int, int, int, int, int, bool, const uint32_t*, uint32_t);
extern "C" int rns8_hip_adaptive_native_suffix(int8_t*, const NativeTileBound*, int, int, int, bool, int*);
#endif

rns8_status hip_direct_gemm_adaptive_native_device(
    int device_id, const void* a, const void* b, int8_t* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    bool unsigned_inputs, uint32_t capacity, uint32_t group,
    uint64_t max_scratch_bytes, bool& commit_started, NativeBoundMethod method, uint32_t panel_k, uint32_t compact_workers) {
  commit_started = false;
  if (compact_workers > 65535 || !a || !b || !c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      !capacity || capacity > 20 || (group != 1 && group != 2 && group != 4) ||
      (method != NativeBoundMethod::tile_max && method != NativeBoundMethod::axis_l1 && method != NativeBoundMethod::panel_l1) ||
      (method == NativeBoundMethod::panel_l1 && (!panel_k || panel_k > 65536)))
    return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_matrix_address(m, k, lda, 1, 8) ||
      !qualified_i32_matrix_address(k, n, ldb, 1, 8) ||
      !qualified_i32_matrix_address(m, n, ldc, capacity)) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t output_bytes = static_cast<uint64_t>(m) * ldc * capacity;
  if (!nonoverlapping_byte_ranges(c, output_bytes, a, static_cast<uint64_t>(m) * lda * 8) ||
      !nonoverlapping_byte_ranges(c, output_bytes, b, static_cast<uint64_t>(k) * ldb * 8))
    return RNS8_INVALID_ARGUMENT;
  // The qualified i32 domains bound all products below UINT64_MAX, including
  // panel summaries. No scratch region aliases native inputs or caller output.
  const uint64_t rows = (static_cast<uint64_t>(m) + 15) / 16;
  const uint64_t cols = (static_cast<uint64_t>(n) + 15) / 16;
  const uint64_t tiles = rows * cols;
  const uint64_t stage_bytes = static_cast<uint64_t>(m) * n * capacity;
  const uint64_t bound_bytes = tiles * sizeof(NativeTileBound);
  const uint64_t metadata_bytes = (tiles + 23) * sizeof(uint32_t);
  const uint64_t reserved = stage_bytes + bound_bytes + metadata_bytes;
  const uint64_t extent = method == NativeBoundMethod::panel_l1 ? panel_k : static_cast<uint64_t>(k);
  const uint64_t panels = (static_cast<uint64_t>(k) + extent - 1) / extent;
  if (panels > 65535) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t discovery_bytes = (rows + cols) * panels *
      (method == NativeBoundMethod::tile_max ? sizeof(uint64_t) : sizeof(NativeAxisSummary)) + 43 * sizeof(uint32_t);
  if (reserved > SIZE_MAX || discovery_bytes > SIZE_MAX - reserved) return RNS8_RANGE_ERROR;
  if (reserved > max_scratch_bytes || discovery_bytes > max_scratch_bytes - reserved)
    return RNS8_WORKSPACE_TOO_SMALL;
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
      unsigned_inputs, capacity, bounds, indices, offsets, max_scratch_bytes - reserved, method, panel_k);
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
