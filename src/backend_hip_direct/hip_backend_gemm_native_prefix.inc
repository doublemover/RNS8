rns8_status hip_direct_gemm_i64_native_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_native || !device_c_residues || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("rns_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_i64_native_prefix9_device(
        static_cast<const int64_t*>(device_a_native),
        static_cast<const int64_t*>(device_b_native),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_native;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_i64_native_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_native || !device_c_residues || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("rns_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_i64_native_prefix9_colpair_device(
        static_cast<const int64_t*>(device_a_native),
        static_cast<const int64_t*>(device_b_native),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_native;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_u64_native_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_native || !device_c_residues || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("rns_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_u64_native_prefix9_device(
        static_cast<const uint64_t*>(device_a_native),
        static_cast<const uint64_t*>(device_b_native),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_native;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_u64_native_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_native || !device_c_residues || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("rns_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_u64_native_prefix9_colpair_device(
        static_cast<const uint64_t*>(device_a_native),
        static_cast<const uint64_t*>(device_b_native),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  return code == 0 ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_native;
  (void)device_b_native;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

