#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_wmma_capability(int*);
extern "C" int rns8_hip_block_sparse_gemm(const int8_t *, const uint32_t *,
                                          const uint32_t *, const int8_t *,
                                          const uint32_t *, const uint32_t *,
                                          int8_t *, int, int, int, int, int, int, uint32_t);
#endif
rns8_status hip_direct_gemm_block_sparse_device(
    int device_id, const void *a, const BlockSparseOperand &lhs,
    uint64_t a_instance, uint64_t a_generation, const void *b,
    const BlockSparseOperand &rhs, uint64_t b_instance, uint64_t b_generation,
    void *c, int64_t ldc, uint64_t max_scratch_bytes, bool* commit_started, uint32_t method, uint32_t scalar_intersection_limit) {
  if (commit_started) *commit_started=false;
  if (method>2 || (method!=2 && scalar_intersection_limit)) return RNS8_INVALID_ARGUMENT;
  if (!valid_block_sparse_operand(lhs, a_instance, a_generation) ||
      !valid_block_sparse_operand(rhs, b_instance, b_generation) ||
      lhs.role != BlockSparseRole::lhs || rhs.role != BlockSparseRole::rhs ||
      lhs.cols != rhs.rows || lhs.prefix != rhs.prefix || !c || ldc < 0 ||
      uint64_t(ldc) < rhs.cols || (lhs.payload_bytes && !a) ||
      (rhs.payload_bytes && !b))
    return RNS8_INVALID_ARGUMENT;
  if (lhs.rows > INT32_MAX || lhs.cols > INT32_MAX || rhs.cols > INT32_MAX ||
      lhs.payload_bytes > INT32_MAX || rhs.payload_bytes > INT32_MAX ||
      !qualified_i32_matrix_address(lhs.rows, rhs.cols, ldc, lhs.prefix))
    return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t c_bytes = lhs.rows * ldc * lhs.prefix;
  if ((lhs.payload_bytes &&
       !nonoverlapping_byte_ranges(c, c_bytes, a, lhs.payload_bytes)) ||
      (rhs.payload_bytes &&
       !nonoverlapping_byte_ranges(c, c_bytes, b, rhs.payload_bytes)))
    return RNS8_INVALID_ARGUMENT;
  const uint64_t words = uint64_t{lhs.offsets.size()} + lhs.k_blocks.size() +
                         rhs.offsets.size() + rhs.k_blocks.size() + (method?1:0);
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
  auto *ao = static_cast<uint32_t *>(metadata.get());
  auto *ai = ao + lhs.offsets.size();
  auto *bo = ai + lhs.k_blocks.size();
  auto *bi = bo + rhs.offsets.size();
  const std::vector<uint32_t> *arrays[] = {&lhs.offsets, &lhs.k_blocks,
                                           &rhs.offsets, &rhs.k_blocks};
  uint32_t *destinations[] = {ao, ai, bo, bi};
  for (int i = 0; i < 4; ++i)
    if (!arrays[i]->empty()) {
      status = hip_direct_copy_host_to_device(
          device_id, destinations[i], arrays[i]->data(),
          arrays[i]->size() * sizeof(uint32_t));
      if (status != RNS8_SUCCESS)
        return status;
    }
  if(method) {
    auto* capability=static_cast<int*>(metadata.get())+words-1;
    const auto launched=static_cast<hipError_t>(rns8_hip_native_wmma_capability(capability));
    const auto completed=hipDeviceSynchronize();
    if(launched!=hipSuccess || completed!=hipSuccess) return RNS8_BACKEND_FAILURE;
    int supported=0;
    status=hip_direct_copy_device_to_host(device_id,&supported,capability,sizeof(supported));
    if(status!=RNS8_SUCCESS) return status;
    if(!supported) return RNS8_UNSUPPORTED_BACKEND;
  }
  const auto error =
      timed_hip_operation("block_sparse_intersection_gemm_kernel", [&]() {
        if (commit_started) *commit_started=true;
        const auto launch = static_cast<hipError_t>(rns8_hip_block_sparse_gemm(
            static_cast<const int8_t *>(a), ao, ai,
            static_cast<const int8_t *>(b), bo, bi, static_cast<int8_t *>(c),
            static_cast<int>(lhs.rows), static_cast<int>(rhs.cols),
            static_cast<int>(lhs.cols), static_cast<int>(ldc), lhs.prefix,static_cast<int>(method),scalar_intersection_limit));
        return launch == hipSuccess ? hipDeviceSynchronize() : launch;
      });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
