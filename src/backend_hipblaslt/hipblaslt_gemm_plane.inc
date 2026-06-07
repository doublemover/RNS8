#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT

rns8_status run_reduce_block(
    int32_t* scratch,
    int8_t* residues,
    int64_t m,
    int64_t n,
    int64_t scratch_ld,
    int64_t ldc,
    uint16_t modulus,
    bool accumulate) {
  const hipError_t err = timed_hip_operation("hipblaslt_i32_to_residue_reduce", [&]() {
    const int code = rns8_hipblaslt_reduce_i32_to_centered_strided_device(
        scratch,
        residues,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(scratch_ld),
        static_cast<int>(ldc),
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        accumulate ? 1 : 0);
    return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

rns8_status gemm_one_plane(
    int device_id,
    hipblasLtHandle_t handle,
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int32_t* scratch,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus,
    const int8_t* cached_packed_a_t,
    const int8_t* cached_packed_b_t) {
  int64_t k_offset = 0;
  bool accumulate = false;
  while (k_offset < k) {
    const int64_t k_remaining = k - k_offset;
    const int64_t k_block =
        std::min<int64_t>(k_remaining, static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK));
    padded_block_shape shape;
    if (!padded_block_shape_for(m, n, k_block, shape)) {
      return RNS8_RANGE_ERROR;
    }
    const std::size_t packed_a_bytes = static_cast<std::size_t>(shape.m) * static_cast<std::size_t>(shape.k);
    const std::size_t packed_b_bytes = static_cast<std::size_t>(shape.n) * static_cast<std::size_t>(shape.k);
    if (workspace_bytes < packed_a_bytes ||
        workspace_bytes - packed_a_bytes < packed_b_bytes ||
        workspace_bytes - packed_a_bytes - packed_b_bytes < kHipblasLtBaselineWorkspaceBytes) {
      return RNS8_INVALID_ARGUMENT;
    }
    auto* workspace_bytes_base = static_cast<std::byte*>(workspace);
    auto* packed_a_t = reinterpret_cast<int8_t*>(workspace_bytes_base);
    auto* packed_b_t = reinterpret_cast<int8_t*>(workspace_bytes_base + packed_a_bytes);
    const int8_t* matmul_a_t = packed_a_t;
    const int8_t* matmul_b_t = packed_b_t;
    void* library_workspace = workspace_bytes_base + packed_a_bytes + packed_b_bytes;
    const std::size_t library_workspace_bytes = workspace_bytes - packed_a_bytes - packed_b_bytes;
    rns8_status pack_status = RNS8_SUCCESS;
    if (cached_packed_a_t) {
      if (k_offset != 0 || k_block != k) {
        return RNS8_INVALID_ARGUMENT;
      }
      matmul_a_t = cached_packed_a_t;
    } else {
      pack_status = pack_transpose_centered(
          a + k_offset,
          packed_a_t,
          static_cast<int>(m),
          static_cast<int>(k_block),
          static_cast<int>(lda),
          shape.k,
          shape.m,
          shape.k);
      if (pack_status != RNS8_SUCCESS) {
        return pack_status;
      }
    }
    if (cached_packed_b_t) {
      if (k_offset != 0 || k_block != k) {
        return RNS8_INVALID_ARGUMENT;
      }
      matmul_b_t = cached_packed_b_t;
    } else {
      pack_status = pack_transpose_centered(
          b + k_offset * ldb,
          packed_b_t,
          static_cast<int>(k_block),
          static_cast<int>(n),
          static_cast<int>(ldb),
          shape.n,
          shape.k,
          shape.n);
      if (pack_status != RNS8_SUCCESS) {
        return pack_status;
      }
    }
    const hipblasStatus_t matmul_status = run_matmul_block(
        device_id,
        handle,
        matmul_a_t,
        matmul_b_t,
        scratch,
        library_workspace,
        library_workspace_bytes,
        shape);
    const rns8_status mapped_status = status_from_hipblas(matmul_status);
    if (mapped_status != RNS8_SUCCESS) {
      return mapped_status;
    }
    const rns8_status reduce_status = run_reduce_block(scratch, c, m, n, shape.n, ldc, modulus, accumulate);
    if (reduce_status != RNS8_SUCCESS) {
      return reduce_status;
    }
    accumulate = true;
    k_offset += k_block;
  }
  return RNS8_SUCCESS;
}
#endif

}  // namespace

