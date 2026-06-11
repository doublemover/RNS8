
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_direct_persistent_small_gemm_rns_device(
    const int8_t* a_residues, const int8_t* b_residues, int8_t* c_residues,
    int m, int n, int k, int prefix);
extern "C" int rns8_hip_direct_fused_gemm_export_i64_device(
    const int8_t* a_residues, const int8_t* b_residues, int64_t* dst,
    int m, int n, int k, int lda, int ldb, int ldc,
    int prefix, int64_t bound, int* status);
#endif


#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_direct_persistent_small_gemm_rns_device(
    const int8_t* a_residues, const int8_t* b_residues, int8_t* c_residues,
    int m, int n, int k, int prefix);
extern "C" int rns8_hip_direct_fused_gemm_export_i64_device(
    const int8_t* a_residues, const int8_t* b_residues, int64_t* dst,
    int m, int n, int k, int lda, int ldb, int ldc,
    int prefix, int64_t bound, int* status);
#endif

rns8_status hip_direct_ring_gemm_i8_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus,
    uint32_t modulus_index,
    uint32_t selected_prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_rns_modulus_launch launch{};
  launch.a = static_cast<const int8_t*>(device_a_residues);
  launch.b = static_cast<const int8_t*>(device_b_residues);
  launch.c = static_cast<int8_t*>(device_c_residues);
  launch.m = m;
  launch.n = n;
  launch.k = k;
  launch.lda = lda;
  launch.ldb = ldb;
  launch.ldc = ldc;
  launch.modulus = modulus;
  launch.modulus_index = modulus_index;
  launch.selected_prefix = selected_prefix;
  if (!checked_rns_modulus_launch(launch)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("rns_gemm_kernel", [&]() {
    const hipError_t launch_status = launch_rns_modulus_gemm(launch);
    return launch_status == hipSuccess ? hipDeviceSynchronize() : launch_status;
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)modulus;
  (void)modulus_index;
  (void)selected_prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
hipError_t launch_direct_rns_gemm_no_sync(
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix,
    void* stream = nullptr) {
  const auto* a_base = static_cast<const int8_t*>(device_a_residues);
  const auto* b_base = static_cast<const int8_t*>(device_b_residues);
  auto* c_base = static_cast<int8_t*>(device_c_residues);
  return launch_rns_grouped_prefix_gemm(a_base, b_base, c_base, m, n, k, lda, ldb, ldc, prefix, stream);
}
#endif

rns8_status hip_direct_gemm_rns_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_base = static_cast<const int8_t*>(device_a_residues);
  const auto* b_base = static_cast<const int8_t*>(device_b_residues);
  auto* c_base = static_cast<int8_t*>(device_c_residues);
  const bool use_persistent_small = (static_cast<int64_t>(m) * static_cast<int64_t>(n) <= 64) && (prefix >= 1) && (lda == k) && (ldb == n);
  hipError_t err;
  if (use_persistent_small) {
    err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
      return static_cast<hipError_t>(rns8_hip_direct_persistent_small_gemm_rns_device(
          a_base, b_base, c_base, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), static_cast<int>(prefix)));
    });
  } else {
    err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
      const hipError_t launch_status =
          launch_direct_rns_gemm_no_sync(a_base, b_base, c_base, m, n, k, lda, ldb, ldc, prefix);
      return launch_status == hipSuccess ? hipDeviceSynchronize() : launch_status;
    });
  }
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_rns_region_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t region_row_offset,
    int64_t region_col_offset,
    int64_t region_rows,
    int64_t region_cols,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || region_row_offset < 0 || region_col_offset < 0 || region_rows <= 0 ||
      region_cols <= 0 || region_row_offset > m || region_col_offset > n ||
      region_rows > m - region_row_offset || region_cols > n - region_col_offset || prefix == 0 ||
      prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max() ||
      region_row_offset > std::numeric_limits<int>::max() ||
      region_col_offset > std::numeric_limits<int>::max() ||
      region_rows > std::numeric_limits<int>::max() || region_cols > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipError_t err = timed_hip_operation("rns_gemm_region_kernel_group", [&]() {
    const int code = rns8_hip_direct_ring_gemm_i8_grouped_prefix_region_device(
        static_cast<const int8_t*>(device_a_residues),
        static_cast<const int8_t*>(device_b_residues),
        static_cast<int8_t*>(device_c_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(region_row_offset),
        static_cast<int>(region_col_offset),
        static_cast<int>(region_rows),
        static_cast<int>(region_cols),
        static_cast<int>(prefix),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    return code == static_cast<int>(hipSuccess) ? hipDeviceSynchronize() : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)region_row_offset;
  (void)region_col_offset;
  (void)region_rows;
  (void)region_cols;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_rns_grouped_matrices_device(
    int device_id,
    rns8_matrix* const* a_matrices,
    rns8_matrix* const* b_matrices,
    rns8_matrix* const* c_matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_a_residue_ptrs,
    const void* device_b_residue_ptrs,
    const void* device_c_residue_ptrs,
    int64_t m,
    int64_t n,
    int64_t k,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!a_matrices || !b_matrices || !c_matrices || task_count == 0 || !device_a_residue_ptrs ||
      !device_b_residue_ptrs || !device_c_residue_ptrs ||
      (expected_semantics != RNS8_BOUNDED_I64 && expected_semantics != RNS8_BOUNDED_U64 &&
       expected_semantics != RNS8_EXACT_WIDE_SIGNED && expected_semantics != RNS8_EXACT_WIDE_UNSIGNED) ||
      m <= 0 || n <= 0 || k <= 0 || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }

  int a_device_id = -1;
  int b_device_id = -1;
  int c_device_id = -1;
  uint32_t a_prefix = 0;
  uint32_t b_prefix = 0;
  uint32_t c_prefix = 0;
  rns8_status status = validate_grouped_rns_matrices(
      a_matrices, task_count, expected_semantics, m, k, true, &a_device_id, &a_prefix, nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = validate_grouped_rns_matrices(
      b_matrices, task_count, expected_semantics, k, n, true, &b_device_id, &b_prefix, nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = validate_grouped_rns_matrices(
      c_matrices, task_count, expected_semantics, m, n, false, &c_device_id, &c_prefix, nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (a_device_id != b_device_id || a_device_id != c_device_id || a_prefix != b_prefix || a_prefix != c_prefix ||
      a_prefix != prefix) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (device_id < 0) {
    device_id = a_device_id;
  } else if (device_id != a_device_id) {
    return RNS8_INVALID_ARGUMENT;
  }

  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_ptrs = static_cast<const int8_t* const*>(device_a_residue_ptrs);
  const auto* b_ptrs = static_cast<const int8_t* const*>(device_b_residue_ptrs);
  auto* c_ptrs = static_cast<int8_t* const*>(const_cast<void*>(device_c_residue_ptrs));
  const hipError_t err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
    const hipError_t launch_status =
        launch_rns_grouped_task_prefix_gemm(a_ptrs, b_ptrs, c_ptrs, task_count, m, n, k, k, n, n, prefix);
    return launch_status == hipSuccess ? hipDeviceSynchronize() : launch_status;
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  for (uint32_t index = 0; index < task_count; ++index) {
    rns8_matrix* matrix = c_matrices[index];
    mark_device_residue_output_current(matrix);
    if (expected_semantics == RNS8_BOUNDED_I64 || expected_semantics == RNS8_BOUNDED_U64) {
      matrix->source_version = api::gemm_output_source_version_values(
          a_matrices[index]->source_version, b_matrices[index]->source_version);
    }
  }
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)a_matrices;
  (void)b_matrices;
  (void)c_matrices;
  (void)task_count;
  (void)expected_semantics;
  (void)device_a_residue_ptrs;
  (void)device_b_residue_ptrs;
  (void)device_c_residue_ptrs;
  (void)m;
  (void)n;
  (void)k;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_rns_grouped_exact_wide_matrices_device(
    int device_id,
    rns8_matrix* const* a_matrices,
    rns8_matrix* const* b_matrices,
    rns8_matrix* const* c_matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_a_residue_ptrs,
    const void* device_b_residue_ptrs,
    const void* device_c_residue_ptrs,
    int64_t m,
    int64_t n,
    int64_t k,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!a_matrices || !b_matrices || !c_matrices || task_count == 0 || !device_a_residue_ptrs ||
      !device_b_residue_ptrs || !device_c_residue_ptrs ||
      (expected_semantics != RNS8_EXACT_WIDE_SIGNED && expected_semantics != RNS8_EXACT_WIDE_UNSIGNED) ||
      m <= 0 || n <= 0 || k <= 0 || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }

  int a_device_id = -1;
  int b_device_id = -1;
  int c_device_id = -1;
  uint32_t a_prefix = 0;
  uint32_t b_prefix = 0;
  uint32_t c_prefix = 0;
  rns8_status status = validate_grouped_rns_matrices(
      a_matrices, task_count, expected_semantics, m, k, true, &a_device_id, &a_prefix, nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = validate_grouped_rns_matrices(
      b_matrices, task_count, expected_semantics, k, n, true, &b_device_id, &b_prefix, nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = validate_grouped_rns_matrices(
      c_matrices, task_count, expected_semantics, m, n, false, &c_device_id, &c_prefix, nullptr);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  if (a_device_id != b_device_id || a_device_id != c_device_id || a_prefix != b_prefix || a_prefix != c_prefix ||
      a_prefix != prefix) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (device_id < 0) {
    device_id = a_device_id;
  } else if (device_id != a_device_id) {
    return RNS8_INVALID_ARGUMENT;
  }

  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_ptrs = static_cast<const int8_t* const*>(device_a_residue_ptrs);
  const auto* b_ptrs = static_cast<const int8_t* const*>(device_b_residue_ptrs);
  auto* c_ptrs = static_cast<int8_t* const*>(const_cast<void*>(device_c_residue_ptrs));
  const hipError_t err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
    const hipError_t launch_status =
        launch_rns_grouped_task_prefix_gemm(a_ptrs, b_ptrs, c_ptrs, task_count, m, n, k, k, n, n, prefix);
    return launch_status == hipSuccess ? hipDeviceSynchronize() : launch_status;
  });
  if (err != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  for (uint32_t index = 0; index < task_count; ++index) {
    rns8_matrix* matrix = c_matrices[index];
    mark_device_residue_output_current(matrix);
  }
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)a_matrices;
  (void)b_matrices;
  (void)c_matrices;
  (void)task_count;
  (void)expected_semantics;
  (void)device_a_residue_ptrs;
  (void)device_b_residue_ptrs;
  (void)device_c_residue_ptrs;
  (void)m;
  (void)n;
  (void)k;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_rns_matrix_launch_current_device_no_sync(
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    void* stream) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!plan || !A || !B || !C || plan->backend != RNS8_BACKEND_HIP_DIRECT || !plan->tile_schedule.empty()) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!A->device_residues_current || !B->device_residues_current || !A->hip_residues || !B->hip_residues ||
      !C->hip_residues) {
    return RNS8_INVALID_ARGUMENT;
  }
  const hipError_t launch_status = launch_direct_rns_gemm_no_sync(
      A->hip_residues,
      B->hip_residues,
      C->hip_residues,
      plan->desc.m,
      plan->desc.n,
      plan->desc.k,
      A->desc.cols,
      B->desc.cols,
      C->desc.cols,
      plan->prefix,
      stream);
  if (launch_status != hipSuccess) {
    return RNS8_BACKEND_FAILURE;
  }
  mark_device_residue_output_current(C);
  if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
    C->source_version = static_cast<uint64_t>(A->source_version + B->source_version + 1u);
  }
  return RNS8_SUCCESS;
#else
  (void)plan;
  (void)A;
  (void)B;
  (void)C;
  (void)stream;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemv_n1_rns_prefix9_matrix(
    int device_id,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t k,
    uint64_t source_version) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!A || !B || !C || !A->device_residues_current || !B->device_residues_current ||
      !A->hip_residues || !B->hip_residues || !C->hip_residues ||
      (A->desc.semantics != RNS8_BOUNDED_I64 && A->desc.semantics != RNS8_BOUNDED_U64) ||
      A->desc.semantics != B->desc.semantics || A->desc.semantics != C->desc.semantics ||
      A->desc.bound_kind != B->desc.bound_kind || A->desc.bound_kind != C->desc.bound_kind ||
      A->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || m <= 0 || k <= 0 || A->desc.rows != m ||
      A->desc.cols != k || B->desc.rows != k || B->desc.cols != 1 || C->desc.rows != m ||
      C->desc.cols != 1 || A->desc.logical_ld < k || B->desc.logical_ld < 1 ||
      C->desc.logical_ld < 1 || m > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || A->desc.logical_ld > std::numeric_limits<int>::max() ||
      B->desc.logical_ld > std::numeric_limits<int>::max() ||
      C->desc.logical_ld > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("rns_gemv_n1_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemv_n1_i8_grouped_prefix_device(
        static_cast<const int8_t*>(A->hip_residues),
        static_cast<const int8_t*>(B->hip_residues),
        static_cast<int8_t*>(C->hip_residues),
        static_cast<int>(m),
        static_cast<int>(k),
        static_cast<int>(A->desc.logical_ld),
        static_cast<int>(B->desc.logical_ld),
        static_cast<int>(C->desc.logical_ld),
        static_cast<int>(RNS8_DEFAULT_BOUNDED_PREFIX),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  if (code != 0) {
    return RNS8_BACKEND_FAILURE;
  }
  mark_device_residue_output_current(C);
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)A;
  (void)B;
  (void)C;
  (void)m;
  (void)k;
  (void)source_version;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemv_small_n_rns_prefix9_matrix(
    int device_id,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    uint64_t source_version) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!A || !B || !C || !A->device_residues_current || !B->device_residues_current ||
      !A->hip_residues || !B->hip_residues || !C->hip_residues ||
      (A->desc.semantics != RNS8_BOUNDED_I64 && A->desc.semantics != RNS8_BOUNDED_U64) ||
      A->desc.semantics != B->desc.semantics || A->desc.semantics != C->desc.semantics ||
      A->desc.bound_kind != B->desc.bound_kind || A->desc.bound_kind != C->desc.bound_kind ||
      A->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || B->prefix != RNS8_DEFAULT_BOUNDED_PREFIX ||
      C->prefix != RNS8_DEFAULT_BOUNDED_PREFIX || m <= 0 || n <= 1 || n > 8 || k <= 0 ||
      A->desc.rows != m || A->desc.cols != k || B->desc.rows != k || B->desc.cols != n ||
      C->desc.rows != m || C->desc.cols != n || A->desc.logical_ld < k || B->desc.logical_ld < n ||
      C->desc.logical_ld < n || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      A->desc.logical_ld > std::numeric_limits<int>::max() ||
      B->desc.logical_ld > std::numeric_limits<int>::max() ||
      C->desc.logical_ld > std::numeric_limits<int>::max()) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const int code = rns8::detail::run_timed_device_code("rns_gemv_small_n_kernel_group", [&]() {
    const int launch_status = rns8_hip_direct_ring_gemv_small_n_i8_grouped_prefix_device(
        static_cast<const int8_t*>(A->hip_residues),
        static_cast<const int8_t*>(B->hip_residues),
        static_cast<int8_t*>(C->hip_residues),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(A->desc.logical_ld),
        static_cast<int>(B->desc.logical_ld),
        static_cast<int>(C->desc.logical_ld),
        static_cast<int>(RNS8_DEFAULT_BOUNDED_PREFIX),
        static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
    if (launch_status != static_cast<int>(hipSuccess)) {
      return launch_status;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    return sync_status == hipSuccess ? 0 : static_cast<int>(sync_status);
  });
  if (code != 0) {
    return RNS8_BACKEND_FAILURE;
  }
  mark_device_residue_output_current(C);
  C->finite_modulus = 0;
  C->source_version = source_version;
  C->prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)A;
  (void)B;
  (void)C;
  (void)m;
  (void)n;
  (void)k;
  (void)source_version;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}


// === HIP graph full-path replay for bounded RNS GEMM ===
// Captures per-plane GEMM as a HIP graph, replays for repeated workloads.

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
namespace {
struct bounded_graph_state {
  hipGraphExec_t graph_exec = nullptr;
  bool captured = false;
  hipStream_t capture_stream = nullptr;
  int captured_prefix = 0;
  int64_t captured_m = 0;
  int64_t captured_n = 0;
  int64_t captured_k = 0;
};
thread_local bounded_graph_state g_bounded_graph;
}  // namespace

rns8_status hip_direct_bounded_graph_begin(hipStream_t stream) {
  g_bounded_graph.capture_stream = stream;
  hipError_t err = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

rns8_status hip_direct_bounded_graph_end() {
  if (!g_bounded_graph.capture_stream) return RNS8_BACKEND_FAILURE;
  hipGraph_t graph = nullptr;
  hipError_t err = hipStreamEndCapture(g_bounded_graph.capture_stream, &graph);
  if (err != hipSuccess) return RNS8_BACKEND_FAILURE;
  if (g_bounded_graph.graph_exec) {
    hipGraphExecDestroy(g_bounded_graph.graph_exec);
  }
  err = hipGraphInstantiate(&g_bounded_graph.graph_exec, graph, nullptr, nullptr, 0);
  hipGraphDestroy(graph);
  g_bounded_graph.captured = (err == hipSuccess);
  return g_bounded_graph.captured ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

rns8_status hip_direct_bounded_graph_launch() {
  if (!g_bounded_graph.captured || !g_bounded_graph.graph_exec)
    return RNS8_UNSUPPORTED_BACKEND;
  hipError_t err = hipGraphLaunch(g_bounded_graph.graph_exec, g_bounded_graph.capture_stream);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

void hip_direct_bounded_graph_reset() {
  if (g_bounded_graph.graph_exec) {
    hipGraphExecDestroy(g_bounded_graph.graph_exec);
    g_bounded_graph.graph_exec = nullptr;
  }
  g_bounded_graph.captured = false;
  g_bounded_graph.capture_stream = nullptr;
}

// Graph-safe GEMM: launches per-plane GEMM on capture_stream without sync.
// Called during graph capture to record all plane launches into the graph.
rns8_status hip_direct_gemm_rns_graph_capture_launch(
    const void* device_a_residues, const void* device_b_residues,
    void* device_c_residues,
    int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    uint32_t prefix) {
  if (!g_bounded_graph.capture_stream) return RNS8_BACKEND_FAILURE;
  const auto* a = static_cast<const int8_t*>(device_a_residues);
  const auto* b = static_cast<const int8_t*>(device_b_residues);
  auto* c = static_cast<int8_t*>(device_c_residues);
  const int safe_k = 65536;
  for (uint32_t p = 0; p < prefix; ++p) {
    const int ps_mk = static_cast<int>(p * m * k);
    const int ps_kn = static_cast<int>(p * k * n);
    const int ps_mn = static_cast<int>(p * m * n);
    int code = rns8_hip_direct_ring_gemm_i8_grouped_prefix_device_on_stream(
        a + ps_mk, b + ps_kn, c + ps_mn,
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        static_cast<int>(lda), static_cast<int>(ldb), static_cast<int>(ldc),
        1, safe_k, g_bounded_graph.capture_stream);
    if (code != static_cast<int>(hipSuccess)) return RNS8_BACKEND_FAILURE;
  }
  g_bounded_graph.captured_prefix = static_cast<int>(prefix);
  g_bounded_graph.captured_m = m;
  g_bounded_graph.captured_n = n;
  g_bounded_graph.captured_k = k;
  return RNS8_SUCCESS;
}
#endif  // RNS8_ENABLE_HIP


// === Fused GEMM+export: computes GEMM + Garner CRT in one kernel ===
// Active for prefix <= 8 where precomputed __constant__ weights accelerate
// the Garner reconstruction. Outputs final i64/i64 directly, eliminating
// the intermediate centered i8 residue store/load between GEMM and export.
rns8_status hip_direct_gemm_rns_fused_export_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    int64_t* dst,
    int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    uint32_t prefix, int64_t bound) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!device_a_residues || !device_b_residues || !dst || m <= 0 || n <= 0 || k <= 0 ||
      prefix == 0 || prefix > 8 || lda < k || ldb < n || ldc < n) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) return device_status;
  hipError_t err = timed_hip_operation("rns_gemm_kernel_group", [&]() {
    // Fused kernel: GEMM per plane + Garner CRT -> final i64 output
    const int code = rns8_hip_direct_fused_gemm_export_i64_device(
        static_cast<const int8_t*>(device_a_residues),
        static_cast<const int8_t*>(device_b_residues),
        dst,
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        static_cast<int>(lda), static_cast<int>(ldb), static_cast<int>(ldc),
        static_cast<int>(prefix), bound, nullptr);
    return static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id; (void)device_a_residues; (void)device_b_residues;
  (void)dst; (void)m; (void)n; (void)k; (void)lda; (void)ldb; (void)ldc;
  (void)prefix; (void)bound;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

