rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("bounded_native_a_reuse_b_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_i64_native_a_resident_b_prefix9_device(
        static_cast<const int64_t*>(device_a_native),
        static_cast<const int8_t*>(device_b_residues),
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
  (void)device_b_residues;
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

rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("bounded_native_a_colpair_reuse_b_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_i64_native_a_resident_b_prefix9_colpair_device(
        static_cast<const int64_t*>(device_a_native),
        static_cast<const int8_t*>(device_b_residues),
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
  (void)device_b_residues;
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

rns8_status hip_direct_gemm_i64_uniform_small_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code =
      rns8::detail::run_timed_device_code("bounded_uniform_small_native_a_reuse_b_gemm_kernel_group", [&]() {
        const int launch_status = rns8_hip_direct_ring_gemm_i64_uniform_small_native_a_resident_b_prefix9_device(
            static_cast<const int64_t*>(device_a_native),
            static_cast<const int8_t*>(device_b_residues),
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
  (void)device_b_residues;
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

rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version) {
  if (!device_a_native || !B || !C || !B->device_residues_current || !B->hip_residues || !C->hip_residues ||
      B->desc.semantics != RNS8_BOUNDED_I64 || C->desc.semantics != RNS8_BOUNDED_I64 ||
      B->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS || C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS ||
      B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_i64_native_a_resident_b_prefix9_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  mark_device_residue_output_current(C);
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_colpair_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version) {
  if (!device_a_native || !B || !C || !B->device_residues_current || !B->hip_residues || !C->hip_residues ||
      B->desc.semantics != RNS8_BOUNDED_I64 || C->desc.semantics != RNS8_BOUNDED_I64 ||
      B->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS || C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS ||
      B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_i64_native_a_resident_b_prefix9_colpair_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  mark_device_residue_output_current(C);
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_i64_uniform_small_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version) {
  if (!device_a_native || !B || !C || !B->device_residues_current || !B->hip_residues || !C->hip_residues ||
      B->desc.semantics != RNS8_BOUNDED_I64 || C->desc.semantics != RNS8_BOUNDED_I64 ||
      B->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS || C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS ||
      B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_i64_uniform_small_native_a_resident_b_prefix9_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  mark_device_residue_output_current(C);
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("bounded_native_a_reuse_b_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_u64_native_a_resident_b_prefix9_device(
        static_cast<const uint64_t*>(device_a_native),
        static_cast<const int8_t*>(device_b_residues),
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
  (void)device_b_residues;
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

rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("bounded_native_a_colpair_reuse_b_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_u64_native_a_resident_b_prefix9_colpair_device(
        static_cast<const uint64_t*>(device_a_native),
        static_cast<const int8_t*>(device_b_residues),
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
  (void)device_b_residues;
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

rns8_status hip_direct_gemm_i64_resident_a_native_b_prefix9_colpair_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_native || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("bounded_native_b_colpair_reuse_a_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_i64_resident_a_native_b_prefix9_colpair_device(
        static_cast<const int8_t*>(device_a_residues),
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
  (void)device_a_residues;
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

rns8_status hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_native || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("bounded_native_b_colpair_reuse_a_gemm_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemm_u64_resident_a_native_b_prefix9_colpair_device(
        static_cast<const int8_t*>(device_a_residues),
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
  (void)device_a_residues;
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

rns8_status hip_direct_gemm_u64_uniform_small_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_native || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code =
      rns8::detail::run_timed_device_code("bounded_uniform_small_native_a_reuse_b_gemm_kernel_group", [&]() {
        const int launch_status = rns8_hip_direct_ring_gemm_u64_uniform_small_native_a_resident_b_prefix9_device(
            static_cast<const uint64_t*>(device_a_native),
            static_cast<const int8_t*>(device_b_residues),
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
  (void)device_b_residues;
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

rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version) {
  if (!device_a_native || !B || !C || !B->device_residues_current || !B->hip_residues || !C->hip_residues ||
      B->desc.semantics != RNS8_BOUNDED_U64 || C->desc.semantics != RNS8_BOUNDED_U64 ||
      B->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_u64_native_a_resident_b_prefix9_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  mark_device_residue_output_current(C);
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version) {
  if (!device_a_native || !B || !C || !B->device_residues_current || !B->hip_residues || !C->hip_residues ||
      B->desc.semantics != RNS8_BOUNDED_U64 || C->desc.semantics != RNS8_BOUNDED_U64 ||
      B->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  mark_device_residue_output_current(C);
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_i64_resident_a_native_b_prefix9_colpair_matrix(
    int device_id,
    const rns8_matrix* A,
    const void* device_b_native,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    uint64_t source_version) {
  if (!A || !device_b_native || !C || !A->device_residues_current || !A->hip_residues || !C->hip_residues ||
      A->desc.semantics != RNS8_BOUNDED_I64 || C->desc.semantics != RNS8_BOUNDED_I64 ||
      A->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS || C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS ||
      A->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      A->desc.rows != m || A->desc.cols != k || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_i64_resident_a_native_b_prefix9_colpair_device(
      device_id,
      A->hip_residues,
      device_b_native,
      C->hip_residues,
      m,
      n,
      k,
      A->desc.logical_ld,
      ldb,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  mark_device_residue_output_current(C);
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_matrix(
    int device_id,
    const rns8_matrix* A,
    const void* device_b_native,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    uint64_t source_version) {
  if (!A || !device_b_native || !C || !A->device_residues_current || !A->hip_residues || !C->hip_residues ||
      A->desc.semantics != RNS8_BOUNDED_U64 || C->desc.semantics != RNS8_BOUNDED_U64 ||
      A->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      A->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      A->desc.rows != m || A->desc.cols != k || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_device(
      device_id,
      A->hip_residues,
      device_b_native,
      C->hip_residues,
      m,
      n,
      k,
      A->desc.logical_ld,
      ldb,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  mark_device_residue_output_current(C);
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

rns8_status hip_direct_gemm_u64_uniform_small_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version) {
  if (!device_a_native || !B || !C || !B->device_residues_current || !B->hip_residues || !C->hip_residues ||
      B->desc.semantics != RNS8_BOUNDED_U64 || C->desc.semantics != RNS8_BOUNDED_U64 ||
      B->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      C->desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED ||
      B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      B->desc.rows != k || B->desc.cols != n || C->desc.rows != m || C->desc.cols != n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status status = hip_direct_gemm_u64_uniform_small_native_a_resident_b_prefix9_device(
      device_id,
      device_a_native,
      B->hip_residues,
      C->hip_residues,
      m,
      n,
      k,
      lda,
      B->desc.logical_ld,
      C->desc.logical_ld);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  mark_device_residue_output_current(C);
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
}

