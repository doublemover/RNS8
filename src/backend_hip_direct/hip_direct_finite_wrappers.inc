template <int Modulus>
hipError_t launch_finite_ring_fixed_modulus_gemm(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block,
    dim3 grid,
    dim3 block) {
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_finite_ring_gemm_i8_i32_fixed_modulus_kernel<Modulus>),
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

template <int Modulus>
hipError_t launch_finite_ring_grouped_fixed_modulus_gemm(
    const int8_t* const* d_a_ptrs,
    const int8_t* const* d_b_ptrs,
    int8_t* const* d_c_ptrs,
    int task_count,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block,
    dim3 grid,
    dim3 block) {
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_finite_ring_gemm_i8_i32_grouped_fixed_modulus_kernel<Modulus>),
        grid,
        block,
        0,
        nullptr,
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        task_count,
        m,
        n,
        k,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

extern "C" int rns8_hip_direct_finite_ring_gemm_i8_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || modulus < 2 ||
      modulus > 256 || modulus_reciprocal != rns8_modulus_reciprocal_host(modulus) ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  if (modulus == 256) {
    return static_cast<int>(launch_finite_ring_fixed_modulus_gemm<256>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 255) {
    return static_cast<int>(launch_finite_ring_fixed_modulus_gemm<255>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 251) {
    return static_cast<int>(launch_finite_ring_fixed_modulus_gemm<251>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_i8_i32_tiled_kernel,
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        modulus,
        modulus_reciprocal,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

extern "C" int rns8_hip_direct_finite_ring_gemm_i8_grouped_device(
    const int8_t* const* d_a_ptrs,
    const int8_t* const* d_b_ptrs,
    int8_t* const* d_c_ptrs,
    int task_count,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int safe_k_block) {
  if (!d_a_ptrs || !d_b_ptrs || !d_c_ptrs || task_count <= 0 || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || modulus < 2 || modulus > 256 ||
      modulus_reciprocal != rns8_modulus_reciprocal_host(modulus) ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, task_count);
  if (modulus == 256) {
    return static_cast<int>(launch_finite_ring_grouped_fixed_modulus_gemm<256>(
        d_a_ptrs, d_b_ptrs, d_c_ptrs, task_count, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 255) {
    return static_cast<int>(launch_finite_ring_grouped_fixed_modulus_gemm<255>(
        d_a_ptrs, d_b_ptrs, d_c_ptrs, task_count, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 251) {
    return static_cast<int>(launch_finite_ring_grouped_fixed_modulus_gemm<251>(
        d_a_ptrs, d_b_ptrs, d_c_ptrs, task_count, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_finite_ring_gemm_i8_i32_grouped_modulus_kernel,
        grid,
        block,
        0,
        nullptr,
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        task_count,
        m,
        n,
        k,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        modulus,
        modulus_reciprocal,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

template <int Modulus>
hipError_t launch_finite_ring_native_fixed_modulus_gemm(
    const uint8_t* d_a,
    const uint8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block,
    dim3 grid,
    dim3 block) {
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_finite_ring_gemm_u8_native_i32_fixed_modulus_kernel<Modulus>),
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

extern "C" int rns8_hip_direct_finite_ring_gemm_u8_native_device(
    const uint8_t* d_a,
    const uint8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || modulus < 2 ||
      modulus > 256 || modulus_reciprocal != rns8_modulus_reciprocal_host(modulus) ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  if (modulus == 256) {
    return static_cast<int>(launch_finite_ring_native_fixed_modulus_gemm<256>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 255) {
    return static_cast<int>(launch_finite_ring_native_fixed_modulus_gemm<255>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 251) {
    return static_cast<int>(launch_finite_ring_native_fixed_modulus_gemm<251>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_finite_ring_gemm_u8_native_i32_kernel,
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        modulus,
        modulus_reciprocal,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

template <int Modulus>
hipError_t launch_finite_ring_native_a_fixed_modulus_gemm(
    const uint8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block,
    dim3 grid,
    dim3 block) {
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_finite_ring_gemm_u8_native_a_i8_b_i32_fixed_modulus_kernel<Modulus>),
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

extern "C" int rns8_hip_direct_finite_ring_gemm_u8_native_a_i8_b_device(
    const uint8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || modulus < 2 ||
      modulus > 256 || modulus_reciprocal != rns8_modulus_reciprocal_host(modulus) ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  if (modulus == 256) {
    return static_cast<int>(launch_finite_ring_native_a_fixed_modulus_gemm<256>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 255) {
    return static_cast<int>(launch_finite_ring_native_a_fixed_modulus_gemm<255>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  if (modulus == 251) {
    return static_cast<int>(launch_finite_ring_native_a_fixed_modulus_gemm<251>(
        d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block, grid, block));
  }
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_finite_ring_gemm_u8_native_a_i8_b_i32_kernel,
        grid,
        block,
        0,
        nullptr,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        modulus,
        modulus_reciprocal,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

int rns8_hip_direct_ring_gemm_i8_grouped_prefix_on_stream(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block,
    hipStream_t stream) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      grouped_prefix <= 0 || grouped_prefix > kRns8MaxSupportedPrefix ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / grouped_prefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(lda) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb) ||
      static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, grouped_prefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_i8_i32_grouped_prefix_kernel,
        grid,
        block,
        0,
        stream,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        grouped_prefix,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

int rns8_hip_direct_ring_gemm_i8_grouped_task_prefix_on_stream(
    const int8_t* const* d_a_ptrs,
    const int8_t* const* d_b_ptrs,
    int8_t* const* d_c_ptrs,
    int task_count,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int grouped_prefix,
    int safe_k_block,
    hipStream_t stream) {
  if (!d_a_ptrs || !d_b_ptrs || !d_c_ptrs || task_count <= 0 || m <= 0 || n <= 0 || k <= 0 || lda < k ||
      ldb < n || ldc < n || grouped_prefix <= 0 || grouped_prefix > kRns8MaxSupportedPrefix ||
      task_count > kRns8MaxInt / grouped_prefix || safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / grouped_prefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(lda) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb) ||
      static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid(
      (n + block.x - 1) / block.x,
      (m + block.y - 1) / block.y,
      static_cast<unsigned int>(task_count * grouped_prefix));
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_i8_i32_grouped_task_prefix_kernel,
        grid,
        block,
        0,
        stream,
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        task_count,
        m,
        n,
        k,
        k_offset,
        k_block,
        lda,
        ldb,
        ldc,
        grouped_prefix,
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return static_cast<int>(err);
}

