#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_wmma_capability(int*);
extern "C" int rns8_hip_block_sparse_gemm(const int8_t*, const uint32_t*, const uint32_t*,
                                          const int8_t*, const uint32_t*, const uint32_t*, int8_t*,
                                          int, int, int, int, int, int, uint32_t);
namespace {
rns8_status launch_block_sparse_intersection(const void* a, const BlockSparseOperand& lhs,
                                             const uint32_t* ao, const uint32_t* ai, const void* b,
                                             const BlockSparseOperand& rhs, const uint32_t* bo,
                                             const uint32_t* bi, void* c, int64_t ldc,
                                             bool* commit_started, uint32_t method,
                                             uint32_t scalar_intersection_limit) {
  const auto error = timed_hip_operation("block_sparse_intersection_gemm_kernel", [&]() {
    if (commit_started)
      *commit_started = true;
    const auto launch = static_cast<hipError_t>(rns8_hip_block_sparse_gemm(
        static_cast<const int8_t*>(a), ao, ai, static_cast<const int8_t*>(b), bo, bi,
        static_cast<int8_t*>(c), static_cast<int>(lhs.rows), static_cast<int>(rhs.cols),
        static_cast<int>(lhs.cols), static_cast<int>(ldc), lhs.prefix, static_cast<int>(method),
        scalar_intersection_limit));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}
} // namespace
#endif
namespace {
rns8_status validate_block_sparse_execution(const void* a, const BlockSparseOperand& lhs,
                                            const void* b, const BlockSparseOperand& rhs,
                                            const void* c, int64_t ldc, uint32_t method,
                                            uint32_t scalar_intersection_limit) {
  if (method > 2 || (method != 2 && scalar_intersection_limit) ||
      lhs.role != BlockSparseRole::lhs || rhs.role != BlockSparseRole::rhs ||
      lhs.cols != rhs.rows || lhs.prefix != rhs.prefix || !c || ldc < 0 ||
      uint64_t(ldc) < rhs.cols || (lhs.payload_bytes && !a) || (rhs.payload_bytes && !b))
    return RNS8_INVALID_ARGUMENT;
  if (lhs.rows > INT32_MAX || lhs.cols > INT32_MAX || rhs.cols > INT32_MAX ||
      lhs.payload_bytes > INT32_MAX || rhs.payload_bytes > INT32_MAX ||
      !qualified_i32_matrix_address(lhs.rows, rhs.cols, ldc, lhs.prefix))
    return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t c_bytes = lhs.rows * ldc * lhs.prefix;
  if ((lhs.payload_bytes && !nonoverlapping_byte_ranges(c, c_bytes, a, lhs.payload_bytes)) ||
      (rhs.payload_bytes && !nonoverlapping_byte_ranges(c, c_bytes, b, rhs.payload_bytes)))
    return RNS8_INVALID_ARGUMENT;
  return RNS8_SUCCESS;
}
} // namespace
rns8_status hip_direct_gemm_block_sparse_device(
    int device_id, const void* a, const BlockSparseOperand& lhs, uint64_t a_instance,
    uint64_t a_generation, const void* b, const BlockSparseOperand& rhs, uint64_t b_instance,
    uint64_t b_generation, void* c, int64_t ldc, uint64_t max_scratch_bytes, bool* commit_started,
    uint32_t method, uint32_t scalar_intersection_limit) {
  if (commit_started)
    *commit_started = false;
  if (method > 2 || (method != 2 && scalar_intersection_limit))
    return RNS8_INVALID_ARGUMENT;
  if (!valid_block_sparse_operand(lhs, a_instance, a_generation) ||
      !valid_block_sparse_operand(rhs, b_instance, b_generation))
    return RNS8_INVALID_ARGUMENT;
  const auto admitted =
      validate_block_sparse_execution(a, lhs, b, rhs, c, ldc, method, scalar_intersection_limit);
  if (admitted != RNS8_SUCCESS)
    return admitted;
  const uint64_t words = uint64_t{lhs.offsets.size()} + lhs.k_blocks.size() + rhs.offsets.size() +
                         rhs.k_blocks.size() + (method ? 1 : 0);
  if (words > SIZE_MAX / sizeof(uint32_t))
    return RNS8_RANGE_ERROR;
  if (words * sizeof(uint32_t) > max_scratch_bytes)
    return RNS8_WORKSPACE_TOO_SMALL;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  hip_direct_device_buffer metadata;
  status = metadata.allocate(device_id, words * sizeof(uint32_t));
  if (status != RNS8_SUCCESS)
    return status;
  auto sources = std::make_shared<std::array<std::vector<uint32_t>, 4>>();
  *sources = {lhs.offsets, lhs.k_blocks, rhs.offsets, rhs.k_blocks};
  status = metadata.retain_host_payload(sources);
  if (status != RNS8_SUCCESS)
    return status;
  metadata.mark_pending();
  auto* ao = static_cast<uint32_t*>(metadata.get());
  auto* ai = ao + lhs.offsets.size();
  auto* bo = ai + lhs.k_blocks.size();
  auto* bi = bo + rhs.offsets.size();
  uint32_t* destinations[] = {ao, ai, bo, bi};
  for (int i = 0; i < 4; ++i)
    if (!(*sources)[i].empty()) {
      status = hip_direct_copy_host_to_device(device_id, destinations[i], (*sources)[i].data(),
                                              (*sources)[i].size() * sizeof(uint32_t));
      if (status != RNS8_SUCCESS)
        return status;
    }
  if (method) {
    auto* capability = static_cast<int*>(metadata.get()) + words - 1;
    const auto launched = static_cast<hipError_t>(rns8_hip_native_wmma_capability(capability));
    const auto completed = hipDeviceSynchronize();
    if (launched != hipSuccess || completed != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    int supported = 0;
    status = hip_direct_copy_device_to_host(device_id, &supported, capability, sizeof(supported));
    if (status != RNS8_SUCCESS)
      return status;
    if (!supported) {
      metadata.mark_completed();
      return RNS8_UNSUPPORTED_BACKEND;
    }
  }
  status = launch_block_sparse_intersection(a, lhs, ao, ai, b, rhs, bo, bi, c, ldc, commit_started,
                                            method, scalar_intersection_limit);
  if (status == RNS8_SUCCESS)
    metadata.mark_completed();
  return status;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_query_block_sparse_capability(int device_id,
                                                     BlockSparseDeviceCapability& out) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8_device_info device{};
  auto status = hip_direct_probe(device_id, device);
  if (status != RNS8_SUCCESS)
    return status;
  if (!device.gcn_arch[0])
    return RNS8_BACKEND_FAILURE;
  status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  hip_direct_device_buffer flag;
  status = flag.allocate(device_id, sizeof(uint32_t));
  if (status != RNS8_SUCCESS)
    return status;
  flag.mark_pending();
  status = hip_direct_zero(device_id, flag.get(), sizeof(uint32_t));
  if (status != RNS8_SUCCESS)
    return status;
  const auto error = timed_hip_operation("block_sparse_prepare_capability", [&] {
    const auto launch =
        static_cast<hipError_t>(rns8_hip_native_wmma_capability(static_cast<int*>(flag.get())));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  if (error != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  uint32_t supported = 0;
  status = hip_direct_copy_device_to_host(device_id, &supported, flag.get(), sizeof(supported));
  if (status != RNS8_SUCCESS)
    return status;
  flag.mark_completed();
  if (supported > 1)
    return RNS8_BACKEND_FAILURE;
  BlockSparseDeviceCapability candidate;
  candidate.device_id = device_id;
  candidate.supported = supported != 0;
  std::copy_n(device.gcn_arch, candidate.target.size(), candidate.target.begin());
  candidate.provider = &hip_direct_query_block_sparse_capability;
  out = candidate;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)out;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_prepared_block_sparse_device(
    int device_id, const void* a, const BlockSparseOperand& lhs,
    const BlockSparseDeviceDirectory& lhs_directory, const void* b, const BlockSparseOperand& rhs,
    const BlockSparseDeviceDirectory& rhs_directory, const BlockSparseDeviceCapability& capability,
    void* c, int64_t ldc, bool& commit_started, uint32_t method,
    uint32_t scalar_intersection_limit) {
  commit_started = false;
  if (!block_sparse_directory_matches(lhs_directory, lhs, device_id, lhs.value_instance,
                                      lhs.value_generation) ||
      !block_sparse_directory_matches(rhs_directory, rhs, device_id, rhs.value_instance,
                                      rhs.value_generation) ||
      capability.provider != &hip_direct_query_block_sparse_capability ||
      capability.device_id != device_id || !capability.target[0])
    return RNS8_INVALID_ARGUMENT;
  const auto admitted =
      validate_block_sparse_execution(a, lhs, b, rhs, c, ldc, method, scalar_intersection_limit);
  if (admitted != RNS8_SUCCESS)
    return admitted;
  const uint64_t c_bytes = lhs.rows * ldc * lhs.prefix;
  if (!nonoverlapping_byte_ranges(c, c_bytes, lhs_directory.allocation,
                                  lhs_directory.allocation_bytes) ||
      !nonoverlapping_byte_ranges(c, c_bytes, rhs_directory.allocation,
                                  rhs_directory.allocation_bytes))
    return RNS8_INVALID_ARGUMENT;
  if (method && !capability.supported)
    return RNS8_UNSUPPORTED_BACKEND;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto selected = set_hip_device(device_id);
  if (selected != RNS8_SUCCESS)
    return selected;
  return launch_block_sparse_intersection(a, lhs, lhs_directory.offsets, lhs_directory.k_blocks, b,
                                          rhs, rhs_directory.offsets, rhs_directory.k_blocks, c,
                                          ldc, &commit_started, method, scalar_intersection_limit);
#else
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
