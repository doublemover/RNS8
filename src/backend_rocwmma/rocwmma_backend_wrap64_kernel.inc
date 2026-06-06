__global__ void __launch_bounds__(kRns8WmmaWaveThreads) rocwmma_wrap64_byte_gemm36_candidate_kernel(
    const uint8_t* a_limbs,
    const uint8_t* b_limbs,
    uint8_t* c_limbs,
    int64_t m,
    int64_t n,
    int64_t k) {
  __shared__ int8_t a_pack[kRns8WmmaTileM * kRns8WmmaTileK];
  __shared__ int8_t b_pack[kRns8WmmaTileK * kRns8WmmaTileN];
  __shared__ int32_t tile_acc[kRns8WmmaTileM * kRns8WmmaTileN];
  __shared__ int64_t diagonal_acc[8 * kRns8WmmaTileM * kRns8WmmaTileN];

  const int lane = static_cast<int>(threadIdx.x);
  const int tile_col = static_cast<int>(blockIdx.x) * kRns8WmmaTileN;
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8WmmaTileM;
  for (int index = lane; index < 8 * kRns8WmmaTileM * kRns8WmmaTileN; index += kRns8WmmaWaveThreads) {
    diagonal_acc[index] = 0;
  }
  __syncthreads();

  for (int64_t k_base = 0; k_base < k; k_base += kRns8WmmaTileK) {
    const int valid_k = static_cast<int>(k - k_base < kRns8WmmaTileK ? k - k_base : kRns8WmmaTileK);
    for (int a_limb = 0; a_limb < 8; ++a_limb) {
      for (int b_limb = 0; b_limb + a_limb < 8; ++b_limb) {
        const int diagonal = a_limb + b_limb;
        rocwmma_wrap64_accumulate_component(
            a_limbs,
            b_limbs,
            m,
            n,
            k,
            k_base,
            valid_k,
            tile_row,
            tile_col,
            a_limb,
            b_limb,
            0,
            0,
            diagonal,
            1,
            a_pack,
            b_pack,
            tile_acc,
            diagonal_acc);
        rocwmma_wrap64_accumulate_component(
            a_limbs,
            b_limbs,
            m,
            n,
            k,
            k_base,
            valid_k,
            tile_row,
            tile_col,
            a_limb,
            b_limb,
            1,
            0,
            diagonal,
            256,
            a_pack,
            b_pack,
            tile_acc,
            diagonal_acc);
        rocwmma_wrap64_accumulate_component(
            a_limbs,
            b_limbs,
            m,
            n,
            k,
            k_base,
            valid_k,
            tile_row,
            tile_col,
            a_limb,
            b_limb,
            0,
            1,
            diagonal,
            256,
            a_pack,
            b_pack,
            tile_acc,
            diagonal_acc);
        rocwmma_wrap64_accumulate_component(
            a_limbs,
            b_limbs,
            m,
            n,
            k,
            k_base,
            valid_k,
            tile_row,
            tile_col,
            a_limb,
            b_limb,
            1,
            1,
            diagonal,
            65536,
            a_pack,
            b_pack,
            tile_acc,
            diagonal_acc);
      }
    }
  }

  for (int index = lane; index < kRns8WmmaTileM * kRns8WmmaTileN; index += kRns8WmmaWaveThreads) {
    const int local_row = index / kRns8WmmaTileN;
    const int local_col = index - local_row * kRns8WmmaTileN;
    const int64_t row = static_cast<int64_t>(tile_row + local_row);
    const int64_t col = static_cast<int64_t>(tile_col + local_col);
    if (row >= m || col >= n) {
      continue;
    }
    uint64_t carry = 0;
    uint8_t* dst = c_limbs + wrap64_compact_limb_index(row, col, n, 0);
    for (int diagonal = 0; diagonal < 8; ++diagonal) {
      const auto column = static_cast<uint64_t>(
          diagonal_acc[diagonal * kRns8WmmaTileM * kRns8WmmaTileN + index] + static_cast<int64_t>(carry));
      dst[diagonal] = static_cast<uint8_t>(column & 0xffu);
      carry = column >> 8u;
    }
  }
}

int launch_pack_and_gemm(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int8_t* a_pack,
    int8_t* b_pack,
    int rows,
    int cols,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t reciprocal,
    int safe_k_block,
    int prefix_index) {
  char pack_a_label[72];
  char pack_b_label[72];
  char matmul_label[72];
  const bool have_prefix_label = prefix_index >= 0;
  if (have_prefix_label) {
    prefix_label(pack_a_label, sizeof(pack_a_label), prefix_index, "pack_a");
    prefix_label(pack_b_label, sizeof(pack_b_label), prefix_index, "pack_b");
    prefix_label(matmul_label, sizeof(matmul_label), prefix_index, "matmul");
  }
  const char* pack_a_timing_label = have_prefix_label ? pack_a_label : "rocwmma_pack_a_kernel";
  const char* pack_b_timing_label = have_prefix_label ? pack_b_label : "rocwmma_pack_b_kernel";
  const char* matmul_timing_label = have_prefix_label ? matmul_label : "rocwmma_matmul_kernel";
  const char* pack_a_alias = have_prefix_label ? "rocwmma_pack_a_kernel" : nullptr;
  const char* pack_b_alias = have_prefix_label ? "rocwmma_pack_b_kernel" : nullptr;
  const char* matmul_alias = have_prefix_label ? "rocwmma_matmul_kernel" : nullptr;
  int64_t padded_rows64 = 0;
  int64_t padded_cols64 = 0;
  if (!round_up_aligned_i64(rows, kRns8WmmaTileM, &padded_rows64) ||
      !round_up_aligned_i64(cols, kRns8WmmaTileN, &padded_cols64)) {
    return rocwmma_range_error;
  }
  const int padded_rows = static_cast<int>(padded_rows64);
  const int padded_cols = static_cast<int>(padded_cols64);
  int accumulate = 0;
  for (int k_offset = 0; k_offset < k; k_offset += safe_k_block) {
    int valid_k_block = k - k_offset < safe_k_block ? k - k_offset : safe_k_block;
    int64_t gemm_k_block64 = valid_k_block;
    if (gemm_k_block64 < kRns8WmmaTileK) {
      gemm_k_block64 = kRns8WmmaTileK;
    }
    if (!round_up_aligned_i64(gemm_k_block64, kRns8WmmaTileK, &gemm_k_block64)) {
      return rocwmma_range_error;
    }
    const int gemm_k_block = static_cast<int>(gemm_k_block64);
    const dim3 pack_block(16, 16, 1);
    const dim3 pack_a_grid(
        static_cast<unsigned int>((gemm_k_block + pack_block.x - 1) / pack_block.x),
        static_cast<unsigned int>((padded_rows + pack_block.y - 1) / pack_block.y),
        1);
    const dim3 pack_b_grid(
        static_cast<unsigned int>((gemm_k_block + pack_block.x - 1) / pack_block.x),
        static_cast<unsigned int>((padded_cols + pack_block.y - 1) / pack_block.y),
        1);
    hipError_t err = timed_hip_launch(pack_a_timing_label, pack_a_alias, [&]() {
      pack_a_rowmajor_kernel<<<pack_a_grid, pack_block>>>(
          a, a_pack, rows, padded_rows, valid_k_block, gemm_k_block, lda, k_offset);
      return hipGetLastError();
    });
    if (err != hipSuccess) {
      return rocwmma_backend_failure;
    }
    err = timed_hip_launch(pack_b_timing_label, pack_b_alias, [&]() {
      pack_b_colmajor_kernel<<<pack_b_grid, pack_block>>>(
          b, b_pack, valid_k_block, gemm_k_block, cols, padded_cols, ldb, k_offset);
      return hipGetLastError();
    });
    if (err != hipSuccess) {
      return rocwmma_backend_failure;
    }
    const dim3 grid(
        static_cast<unsigned int>(padded_cols / kRns8WmmaTileN),
        static_cast<unsigned int>(padded_rows / kRns8WmmaTileM),
        1);
    err = timed_hip_launch(matmul_timing_label, matmul_alias, [&]() {
      rocwmma_i8_residue_gemm_kernel<<<grid, kRns8WmmaWaveThreads>>>(
          a_pack,
          b_pack,
          c,
          rows,
          cols,
          padded_rows,
          padded_cols,
          gemm_k_block,
          ldc,
          modulus,
          reciprocal,
          accumulate);
      return hipGetLastError();
    });
    if (err != hipSuccess) {
      return rocwmma_backend_failure;
    }
    accumulate = 1;
  }
  return rocwmma_success;
}

int run_rns_prefix(
    int device_id,
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int8_t* workspace,
    uint64_t workspace_bytes,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    uint32_t prefix) {
  if (prefix == 0 || prefix > kRns8MaxSupportedPrefix) {
    return rocwmma_invalid_argument;
  }
  uint64_t a_bytes = 0;
  uint64_t b_bytes = 0;
  if (!rocwmma_workspace_requirements_u64(m, n, k, &a_bytes, &b_bytes)) {
    return rocwmma_range_error;
  }
  if (!workspace || workspace_bytes < a_bytes + b_bytes) {
    return rocwmma_invalid_argument;
  }
  auto* a_pack = workspace;
  auto* b_pack = workspace + a_bytes;
  const int safe_k_block = k < kRns8SignedSafeInt32KBlock ? k : kRns8SignedSafeInt32KBlock;
  for (uint32_t modulus_index = 0; modulus_index < prefix; ++modulus_index) {
    const uint16_t modulus = kDefaultModuli[modulus_index];
    const auto plane_a = a + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * lda;
    const auto plane_b = b + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(k) * ldb;
    auto plane_c = c + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * ldc;
    const int status = launch_pack_and_gemm(
        plane_a,
        plane_b,
        plane_c,
        a_pack,
        b_pack,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        safe_k_block,
        static_cast<int>(modulus_index));
    if (status != rocwmma_success) {
      return status;
    }
  }
  return hipSetDevice(device_id) == hipSuccess && hipDeviceSynchronize() == hipSuccess ? rocwmma_success
                                                                                       : rocwmma_backend_failure;
}

int prepack_b_rns_prefix(
    int device_id,
    const int8_t* b,
    int8_t* b_prepack,
    uint64_t b_prepack_bytes,
    int k,
    int n,
    int ldb,
    uint32_t prefix) {
  if (prefix == 0 || prefix > kRns8MaxSupportedPrefix || k <= 0 || k > kRns8SignedSafeInt32KBlock) {
    return rocwmma_invalid_argument;
  }
  uint64_t a_bytes = 0;
  uint64_t b_bytes = 0;
  if (!rocwmma_workspace_requirements_u64(16, n, k, &a_bytes, &b_bytes)) {
    return rocwmma_range_error;
  }
  if (b_bytes != 0 && static_cast<uint64_t>(prefix) > UINT64_MAX / b_bytes) {
    return rocwmma_range_error;
  }
  if (!b_prepack || b_prepack_bytes < static_cast<uint64_t>(prefix) * b_bytes) {
    return rocwmma_invalid_argument;
  }
  int64_t padded_n64 = 0;
  int64_t gemm_k_block64 = k < kRns8WmmaTileK ? kRns8WmmaTileK : k;
  if (!round_up_aligned_i64(n, kRns8WmmaTileN, &padded_n64) ||
      !round_up_aligned_i64(gemm_k_block64, kRns8WmmaTileK, &gemm_k_block64)) {
    return rocwmma_range_error;
  }
  const int padded_n = static_cast<int>(padded_n64);
  const int gemm_k_block = static_cast<int>(gemm_k_block64);
  const dim3 pack_block(16, 16, 1);
  const dim3 pack_b_grid(
      static_cast<unsigned int>((gemm_k_block + pack_block.x - 1) / pack_block.x),
      static_cast<unsigned int>((padded_n + pack_block.y - 1) / pack_block.y),
      1);
  for (uint32_t modulus_index = 0; modulus_index < prefix; ++modulus_index) {
    const auto plane_b = b + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(k) * ldb;
    auto* plane_b_prepack = b_prepack + static_cast<uint64_t>(modulus_index) * b_bytes;
    pack_b_colmajor_kernel<<<pack_b_grid, pack_block>>>(
        plane_b, plane_b_prepack, k, gemm_k_block, n, padded_n, ldb, 0);
    if (hipGetLastError() != hipSuccess) {
      return rocwmma_backend_failure;
    }
  }
  return hipSetDevice(device_id) == hipSuccess && hipDeviceSynchronize() == hipSuccess ? rocwmma_success
                                                                                       : rocwmma_backend_failure;
}

int launch_pack_a_and_gemm_prepacked_b(
    const int8_t* a,
    const int8_t* b_pack,
    int8_t* c,
    int8_t* a_pack,
    int rows,
    int cols,
    int k,
    int lda,
    int ldc,
    int modulus,
    uint32_t reciprocal,
    int prefix_index) {
  char pack_a_label[80];
  char matmul_label[80];
  const bool have_prefix_label = prefix_index >= 0;
  if (have_prefix_label) {
    prefix_label(pack_a_label, sizeof(pack_a_label), prefix_index, "pack_a_prepacked_b");
    prefix_label(matmul_label, sizeof(matmul_label), prefix_index, "matmul_prepacked_b");
  }
  const char* pack_a_timing_label =
      have_prefix_label ? pack_a_label : "rocwmma_pack_a_prepacked_b_kernel";
  const char* matmul_timing_label =
      have_prefix_label ? matmul_label : "rocwmma_matmul_prepacked_b_kernel";
  const char* pack_a_alias = have_prefix_label ? "rocwmma_pack_a_prepacked_b_kernel" : nullptr;
  const char* matmul_alias = have_prefix_label ? "rocwmma_matmul_prepacked_b_kernel" : nullptr;
  if (k <= 0 || k > kRns8SignedSafeInt32KBlock) {
    return rocwmma_invalid_argument;
  }
  int64_t padded_rows64 = 0;
  int64_t padded_cols64 = 0;
  int64_t gemm_k_block64 = k < kRns8WmmaTileK ? kRns8WmmaTileK : k;
  if (!round_up_aligned_i64(rows, kRns8WmmaTileM, &padded_rows64) ||
      !round_up_aligned_i64(cols, kRns8WmmaTileN, &padded_cols64) ||
      !round_up_aligned_i64(gemm_k_block64, kRns8WmmaTileK, &gemm_k_block64)) {
    return rocwmma_range_error;
  }
  const int padded_rows = static_cast<int>(padded_rows64);
  const int padded_cols = static_cast<int>(padded_cols64);
  const int gemm_k_block = static_cast<int>(gemm_k_block64);
  const dim3 pack_block(16, 16, 1);
  const dim3 pack_a_grid(
      static_cast<unsigned int>((gemm_k_block + pack_block.x - 1) / pack_block.x),
      static_cast<unsigned int>((padded_rows + pack_block.y - 1) / pack_block.y),
      1);
  hipError_t err = timed_hip_launch(pack_a_timing_label, pack_a_alias, [&]() {
    pack_a_rowmajor_kernel<<<pack_a_grid, pack_block>>>(a, a_pack, rows, padded_rows, k, gemm_k_block, lda, 0);
    return hipGetLastError();
  });
  if (err != hipSuccess) {
    return rocwmma_backend_failure;
  }
  const dim3 grid(
      static_cast<unsigned int>(padded_cols / kRns8WmmaTileN),
      static_cast<unsigned int>(padded_rows / kRns8WmmaTileM),
      1);
  err = timed_hip_launch(matmul_timing_label, matmul_alias, [&]() {
    rocwmma_i8_residue_gemm_kernel<<<grid, kRns8WmmaWaveThreads>>>(
        a_pack,
        b_pack,
        c,
        rows,
        cols,
        padded_rows,
        padded_cols,
        gemm_k_block,
        ldc,
        modulus,
        reciprocal,
        0);
    return hipGetLastError();
  });
  return err == hipSuccess ? rocwmma_success : rocwmma_backend_failure;
}

int run_rns_prefix_prepacked_b(
    int device_id,
    const int8_t* a,
    const int8_t* b_prepack,
    int8_t* c,
    int8_t* workspace,
    uint64_t workspace_bytes,
    int m,
    int n,
    int k,
    int lda,
    int ldc,
    uint32_t prefix) {
  if (prefix == 0 || prefix > kRns8MaxSupportedPrefix || k <= 0 || k > kRns8SignedSafeInt32KBlock) {
    return rocwmma_invalid_argument;
  }
  uint64_t a_bytes = 0;
  uint64_t b_bytes = 0;
  if (!rocwmma_workspace_requirements_u64(m, n, k, &a_bytes, &b_bytes)) {
    return rocwmma_range_error;
  }
  if (!workspace || workspace_bytes < a_bytes || !b_prepack) {
    return rocwmma_invalid_argument;
  }
  auto* a_pack = workspace;
  for (uint32_t modulus_index = 0; modulus_index < prefix; ++modulus_index) {
    const uint16_t modulus = kDefaultModuli[modulus_index];
    const auto plane_a = a + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * lda;
    const auto plane_b_prepack = b_prepack + static_cast<uint64_t>(modulus_index) * b_bytes;
    auto plane_c = c + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * ldc;
    const int status = launch_pack_a_and_gemm_prepacked_b(
        plane_a,
        plane_b_prepack,
        plane_c,
        a_pack,
        m,
        n,
        k,
        lda,
        ldc,
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        static_cast<int>(modulus_index));
    if (status != rocwmma_success) {
      return status;
    }
  }
  return hipSetDevice(device_id) == hipSuccess && hipDeviceSynchronize() == hipSuccess ? rocwmma_success
                                                                                       : rocwmma_backend_failure;
}

}  // namespace

