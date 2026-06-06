extern "C" int rns8_hip_direct_ring_gemm_i8_device(
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
    int modulus_index,
    int selected_prefix,
    int safe_k_block) {
  return rns8_hip_direct_ring_gemm_i8_on_stream(
      d_a,
      d_b,
      d_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      modulus,
      modulus_reciprocal,
      modulus_index,
      selected_prefix,
      safe_k_block,
      nullptr);
}

extern "C" int rns8_hip_direct_ring_gemm_i8_device_on_stream(
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
    int modulus_index,
    int selected_prefix,
    int safe_k_block,
    void* stream) {
  return rns8_hip_direct_ring_gemm_i8_on_stream(
      d_a,
      d_b,
      d_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      modulus,
      modulus_reciprocal,
      modulus_index,
      selected_prefix,
      safe_k_block,
      reinterpret_cast<hipStream_t>(stream));
}

template <typename NativeT>
hipError_t launch_native_prefix9_grouped_gemm(
    const NativeT* d_a,
    const NativeT* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_ring_gemm_native_i64_i32_grouped_prefix9_kernel<NativeT>),
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

template <typename NativeT>
hipError_t launch_native_prefix9_colpair_grouped_gemm(
    const NativeT* d_a,
    const NativeT* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN / 2, kRns8HipTileM);
  const dim3 grid(
      (n + kRns8HipTileN - 1) / kRns8HipTileN,
      (m + kRns8HipTileM - 1) / kRns8HipTileM,
      kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_ring_gemm_native_i64_i32_colpair_grouped_prefix9_kernel<NativeT>),
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

template <typename NativeT, bool UniformSmallA>
hipError_t launch_native_a_resident_b_prefix9_grouped_gemm(
    const NativeT* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_ring_gemm_native_a_i8_b_i32_grouped_prefix9_kernel<NativeT, UniformSmallA>),
        grid,
        block,
        0,
        nullptr,
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
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

template <typename NativeT, bool UniformSmallA>
hipError_t launch_native_a_resident_b_prefix9_colpair_grouped_gemm(
    const NativeT* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN / 2, kRns8HipTileM);
  const dim3 grid(
      (n + kRns8HipTileN - 1) / kRns8HipTileN,
      (m + kRns8HipTileM - 1) / kRns8HipTileM,
      kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_ring_gemm_native_a_i8_b_i32_colpair_grouped_prefix9_kernel<NativeT, UniformSmallA>),
        grid,
        block,
        0,
        nullptr,
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
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

template <typename NativeT, bool UniformSmallB>
hipError_t launch_resident_a_native_b_prefix9_colpair_grouped_gemm(
    const int8_t* d_a,
    const NativeT* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN / 2, kRns8HipTileM);
  const dim3 grid(
      (n + kRns8HipTileN - 1) / kRns8HipTileN,
      (m + kRns8HipTileM - 1) / kRns8HipTileM,
      kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rns8_ring_gemm_i8_a_native_b_i32_colpair_grouped_prefix9_kernel<NativeT, UniformSmallB>),
        grid,
        block,
        0,
        nullptr,
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
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

hipError_t launch_uniform_small_i8_ab_resident_b_prefix9_grouped_gemm(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_uniform_small_i8_ab_i32_grouped_prefix9_kernel,
        grid,
        block,
        0,
        nullptr,
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
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

hipError_t launch_uniform_small_i8_ab_resident_b_prefix9_colpair_grouped_gemm(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  const dim3 block(kRns8HipTileN / 2, kRns8HipTileM);
  const dim3 grid((n + kRns8HipTileN - 1) / kRns8HipTileN, (m + kRns8HipTileM - 1) / kRns8HipTileM, kRns8DefaultBoundedPrefix);
  hipError_t err = hipSuccess;
  int k_offset = 0;
  int accumulate = 0;
  while (k_offset < k && err == hipSuccess) {
    const int k_remaining = k - k_offset;
    const int k_block = k_remaining > safe_k_block ? safe_k_block : k_remaining;
    hipLaunchKernelGGL(
        rns8_ring_gemm_uniform_small_i8_ab_i32_colpair_grouped_prefix9_kernel,
        grid,
        block,
        0,
        nullptr,
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
        accumulate);
    err = hipGetLastError();
    k_offset += k_block;
    accumulate = 1;
  }
  return err;
}

extern "C" int rns8_hip_direct_ring_gemm_i64_native_prefix9_device(
    const int64_t* d_a,
    const int64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_prefix9_grouped_gemm(d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

extern "C" int rns8_hip_direct_ring_gemm_i64_native_prefix9_colpair_device(
    const int64_t* d_a,
    const int64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_prefix9_colpair_grouped_gemm(d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

extern "C" int rns8_hip_direct_ring_gemm_i64_native_a_resident_b_prefix9_device(
    const int64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_a_resident_b_prefix9_grouped_gemm<int64_t, false>(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

extern "C" int rns8_hip_direct_ring_gemm_i64_uniform_small_native_a_resident_b_prefix9_device(
    const int64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_a_resident_b_prefix9_grouped_gemm<int64_t, true>(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

extern "C" int rns8_hip_direct_ring_gemm_u64_native_prefix9_device(
    const uint64_t* d_a,
    const uint64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_prefix9_grouped_gemm(d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

extern "C" int rns8_hip_direct_ring_gemm_u64_native_prefix9_colpair_device(
    const uint64_t* d_a,
    const uint64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_prefix9_colpair_grouped_gemm(d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

extern "C" int rns8_hip_direct_ring_gemm_u64_native_a_resident_b_prefix9_device(
    const uint64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_a_resident_b_prefix9_grouped_gemm<uint64_t, false>(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

extern "C" int rns8_hip_direct_ring_gemm_u64_native_a_resident_b_prefix9_colpair_device(
    const uint64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_a_resident_b_prefix9_colpair_grouped_gemm<uint64_t, false>(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

extern "C" int rns8_hip_direct_ring_gemm_u64_resident_a_native_b_prefix9_colpair_device(
    const int8_t* d_a,
    const uint64_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(lda) ||
      static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_resident_a_native_b_prefix9_colpair_grouped_gemm<uint64_t, false>(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

extern "C" int rns8_hip_direct_ring_gemm_u64_uniform_small_native_a_resident_b_prefix9_device(
    const uint64_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc) ||
      static_cast<int64_t>(k) > max_grouped_plane_elements / static_cast<int64_t>(ldb)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_native_a_resident_b_prefix9_grouped_gemm<uint64_t, true>(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

extern "C" int rns8_hip_direct_ring_gemm_uniform_small_i8_ab_resident_b_prefix9_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_uniform_small_i8_ab_resident_b_prefix9_grouped_gemm(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

extern "C" int rns8_hip_direct_ring_gemm_uniform_small_i8_ab_resident_b_prefix9_colpair_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int safe_k_block) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t max_grouped_plane_elements = kRns8MaxInt64 / kRns8DefaultBoundedPrefix;
  if (static_cast<int64_t>(m) > max_grouped_plane_elements / static_cast<int64_t>(ldc)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(
      launch_uniform_small_i8_ab_resident_b_prefix9_colpair_grouped_gemm(
          d_a, d_b, d_c, m, n, k, lda, ldb, ldc, safe_k_block));
}

