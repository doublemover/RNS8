__global__ void pack_a_rowmajor_kernel(
    const int8_t* a,
    int8_t* packed,
    int rows,
    int padded_rows,
    int valid_k_block,
    int gemm_k_block,
    int lda,
    int k_offset,
    int modulus) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = padded_rows * gemm_k_block;
  if (index >= total) {
    return;
  }
  const int row = index / gemm_k_block;
  const int kk = index - row * gemm_k_block;
  packed[row * gemm_k_block + kk] =
      row < rows && kk < valid_k_block ? ck_input_byte_from_centered_device(a[row * lda + k_offset + kk], modulus)
                                       : 0;
}

__global__ void pack_b_colmajor_kernel(
    const int8_t* b,
    int8_t* packed,
    int valid_k_block,
    int gemm_k_block,
    int n,
    int padded_n,
    int ldb,
    int k_offset,
    int modulus) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = gemm_k_block * padded_n;
  if (index >= total) {
    return;
  }
  const int kk = index % gemm_k_block;
  const int col = index / gemm_k_block;
  packed[col * gemm_k_block + kk] =
      col < n && kk < valid_k_block ? ck_input_byte_from_centered_device(b[(k_offset + kk) * ldb + col], modulus)
                                    : 0;
}

__global__ void copy_centered_residue_kernel(
    const int8_t* src,
    int8_t* dst,
    int rows,
    int cols,
    int src_ld,
    int dst_ld) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (index >= total) {
    return;
  }
  const int row = index / cols;
  const int col = index - row * cols;
  dst[row * dst_ld + col] = src[row * src_ld + col];
}

__global__ void add_centered_residue_kernel(
    const int8_t* rhs,
    int8_t* dst,
    int rows,
    int cols,
    int rhs_ld,
    int dst_ld,
    int modulus,
    uint32_t reciprocal) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (index >= total) {
    return;
  }
  const int row = index / cols;
  const int col = index - row * cols;
  const int32_t sum = static_cast<int32_t>(dst[row * dst_ld + col]) + static_cast<int32_t>(rhs[row * rhs_ld + col]);
  dst[row * dst_ld + col] = center_i32_device(sum, modulus, reciprocal);
}

hipError_t launch_pack_a(
    const int8_t* a,
    int8_t* packed,
    int rows,
    int padded_rows,
    int valid_k_block,
    int gemm_k_block,
    int lda,
    int k_offset,
    int modulus) {
  const int total = padded_rows * gemm_k_block;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  hipLaunchKernelGGL(
      pack_a_rowmajor_kernel,
      dim3(grid),
      dim3(block),
      0,
      nullptr,
      a,
      packed,
      rows,
      padded_rows,
      valid_k_block,
      gemm_k_block,
      lda,
      k_offset,
      modulus);
  return hipGetLastError();
}

hipError_t launch_pack_b(
    const int8_t* b,
    int8_t* packed,
    int valid_k_block,
    int gemm_k_block,
    int n,
    int padded_n,
    int ldb,
    int k_offset,
    int modulus) {
  const int total = gemm_k_block * padded_n;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  hipLaunchKernelGGL(
      pack_b_colmajor_kernel,
      dim3(grid),
      dim3(block),
      0,
      nullptr,
      b,
      packed,
      valid_k_block,
      gemm_k_block,
      n,
      padded_n,
      ldb,
      k_offset,
      modulus);
  return hipGetLastError();
}

hipError_t launch_copy_centered(
    const int8_t* src,
    int8_t* dst,
    int rows,
    int cols,
    int src_ld,
    int dst_ld) {
  const int total = rows * cols;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  hipLaunchKernelGGL(copy_centered_residue_kernel, dim3(grid), dim3(block), 0, nullptr, src, dst, rows, cols, src_ld,
                     dst_ld);
  return hipGetLastError();
}

hipError_t launch_add_centered(
    const int8_t* rhs,
    int8_t* dst,
    int rows,
    int cols,
    int rhs_ld,
    int dst_ld,
    int modulus,
    uint32_t reciprocal) {
  const int total = rows * cols;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  hipLaunchKernelGGL(
      add_centered_residue_kernel, dim3(grid), dim3(block), 0, nullptr, rhs, dst, rows, cols, rhs_ld, dst_ld, modulus,
      reciprocal);
  return hipGetLastError();
}

template <typename CElementwiseOperation>
int run_ck_tile_modulus_impl(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int8_t* a_workspace,
    int8_t* b_workspace,
    int8_t* temp_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t reciprocal,
    int prefix_index,
    CElementwiseOperation epilogue) {
  char pack_a_label[64];
  char pack_b_label[64];
  char matmul_label[64];
  char copy_label[64];
  char add_label[64];
  const bool have_prefix_label = prefix_index >= 0;
  if (have_prefix_label) {
    prefix_label(pack_a_label, sizeof(pack_a_label), prefix_index, "pack_a");
    prefix_label(pack_b_label, sizeof(pack_b_label), prefix_index, "pack_b");
    prefix_label(matmul_label, sizeof(matmul_label), prefix_index, "matmul");
    prefix_label(copy_label, sizeof(copy_label), prefix_index, "copy_centered");
    prefix_label(add_label, sizeof(add_label), prefix_index, "add_centered");
  }
  const char* pack_a_timing_label = have_prefix_label ? pack_a_label : "ck_pack_a_kernel";
  const char* pack_b_timing_label = have_prefix_label ? pack_b_label : "ck_pack_b_kernel";
  const char* matmul_timing_label = have_prefix_label ? matmul_label : "ck_wmma_cshuffle_matmul";
  const char* copy_timing_label = have_prefix_label ? copy_label : "ck_copy_centered_kernel";
  const char* add_timing_label = have_prefix_label ? add_label : "ck_add_centered_kernel";
  const char* pack_a_alias = have_prefix_label ? "ck_pack_a_kernel" : nullptr;
  const char* pack_b_alias = have_prefix_label ? "ck_pack_b_kernel" : nullptr;
  const char* matmul_alias = have_prefix_label ? "ck_wmma_cshuffle_matmul" : nullptr;
  const char* copy_alias = have_prefix_label ? "ck_copy_centered_kernel" : nullptr;
  const char* add_alias = have_prefix_label ? "ck_add_centered_kernel" : nullptr;
  bool have_accumulated = false;
  int64_t padded_m_i64 = 0;
  int64_t padded_n_i64 = 0;
  if (!round_up_aligned_i64(m, kRns8CkShapeAlignment, &padded_m_i64) ||
      !round_up_aligned_i64(n, kRns8CkShapeAlignment, &padded_n_i64) ||
      !checked_i32_shape(padded_m_i64) || !checked_i32_shape(padded_n_i64) || !temp_c) {
    return ck_invalid_argument;
  }
  const int gemm_m = static_cast<int>(padded_m_i64);
  const int gemm_n = static_cast<int>(padded_n_i64);
  const bool padded_output = gemm_m != m || gemm_n != n;
  for (int64_t k_offset = 0; k_offset < k; k_offset += kRns8CanonicalSafeInt32KBlock) {
    const int valid_k_block = static_cast<int>(
        (static_cast<int64_t>(k) - k_offset) < kRns8CanonicalSafeInt32KBlock
            ? (static_cast<int64_t>(k) - k_offset)
            : kRns8CanonicalSafeInt32KBlock);
    const int gemm_k_block = valid_k_block < 64 ? 64 : valid_k_block;
    hipError_t err = timed_hip_launch(pack_a_timing_label, pack_a_alias, [&]() {
      return launch_pack_a(
          a, a_workspace, m, gemm_m, valid_k_block, gemm_k_block, lda, static_cast<int>(k_offset), modulus);
    });
    if (err != hipSuccess) {
      return ck_backend_failure;
    }
    err = timed_hip_launch(pack_b_timing_label, pack_b_alias, [&]() {
      return launch_pack_b(
          b, b_workspace, valid_k_block, gemm_k_block, n, gemm_n, ldb, static_cast<int>(k_offset), modulus);
    });
    if (err != hipSuccess) {
      return ck_backend_failure;
    }
    const bool write_workspace_output = padded_output || have_accumulated;
    int8_t* output = write_workspace_output ? temp_c : c;
    const int output_ld = write_workspace_output ? gemm_n : ldc;
    using GemmInstance = DeviceGemmInstanceFor<CElementwiseOperation>;
    auto arg = GemmInstance::MakeArgument(
        a_workspace,
        b_workspace,
        output,
        gemm_m,
        gemm_n,
        gemm_k_block,
        gemm_k_block,
        gemm_k_block,
        output_ld,
        PassThrough{},
        PassThrough{},
        epilogue);
    if (!GemmInstance::IsSupportedArgument(arg)) {
      return ck_unsupported_backend;
    }
    const int matmul_status = timed_ck_operation(matmul_timing_label, matmul_alias, [&]() {
      try {
        auto invoker = GemmInstance::MakeInvoker();
        (void)invoker.Run(arg);
      } catch (const std::exception&) {
        return ck_backend_failure;
      } catch (...) {
        return ck_backend_failure;
      }
      return ck_success;
    });
    if (matmul_status != ck_success) {
      return ck_backend_failure;
    }
    if (write_workspace_output) {
      if (have_accumulated) {
        err = timed_hip_launch(add_timing_label, add_alias, [&]() {
          return launch_add_centered(temp_c, c, m, n, gemm_n, ldc, modulus, reciprocal);
        });
      } else {
        err = timed_hip_launch(copy_timing_label, copy_alias, [&]() {
          return launch_copy_centered(temp_c, c, m, n, gemm_n, ldc);
        });
      }
      if (err != hipSuccess) {
        return ck_backend_failure;
      }
    }
    have_accumulated = true;
  }
  return ck_success;
}

template <uint32_t Modulus>
int run_ck_tile_modulus_fixed(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int8_t* a_workspace,
    int8_t* b_workspace,
    int8_t* temp_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int prefix_index) {
  return run_ck_tile_modulus_impl(
      a,
      b,
      c,
      a_workspace,
      b_workspace,
      temp_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      static_cast<int>(Modulus),
      modulus_reciprocal_u32(static_cast<uint16_t>(Modulus)),
      prefix_index,
      CenteredModuloFixed<Modulus>{});
}

int run_ck_tile_modulus(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int8_t* a_workspace,
    int8_t* b_workspace,
    int8_t* temp_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t reciprocal,
    int prefix_index) {
  (void)reciprocal;
#define RNS8_CK_FIXED_MODULUS_CASE(modulus_value)                                                        \
  case modulus_value:                                                                                    \
    return run_ck_tile_modulus_fixed<modulus_value>(                                                     \
        a, b, c, a_workspace, b_workspace, temp_c, m, n, k, lda, ldb, ldc, prefix_index)

  switch (modulus) {
    RNS8_CK_FIXED_MODULUS_CASE(256);
    RNS8_CK_FIXED_MODULUS_CASE(255);
    RNS8_CK_FIXED_MODULUS_CASE(253);
    RNS8_CK_FIXED_MODULUS_CASE(251);
    RNS8_CK_FIXED_MODULUS_CASE(247);
    RNS8_CK_FIXED_MODULUS_CASE(243);
    RNS8_CK_FIXED_MODULUS_CASE(241);
    RNS8_CK_FIXED_MODULUS_CASE(239);
    RNS8_CK_FIXED_MODULUS_CASE(233);
    RNS8_CK_FIXED_MODULUS_CASE(229);
    RNS8_CK_FIXED_MODULUS_CASE(227);
    RNS8_CK_FIXED_MODULUS_CASE(223);
    RNS8_CK_FIXED_MODULUS_CASE(217);
    RNS8_CK_FIXED_MODULUS_CASE(211);
    RNS8_CK_FIXED_MODULUS_CASE(199);
    RNS8_CK_FIXED_MODULUS_CASE(197);
    RNS8_CK_FIXED_MODULUS_CASE(193);
    RNS8_CK_FIXED_MODULUS_CASE(191);
    RNS8_CK_FIXED_MODULUS_CASE(181);
    RNS8_CK_FIXED_MODULUS_CASE(179);
    RNS8_CK_FIXED_MODULUS_CASE(173);
    RNS8_CK_FIXED_MODULUS_CASE(167);
    RNS8_CK_FIXED_MODULUS_CASE(163);
    RNS8_CK_FIXED_MODULUS_CASE(157);
    RNS8_CK_FIXED_MODULUS_CASE(151);
    RNS8_CK_FIXED_MODULUS_CASE(149);
    RNS8_CK_FIXED_MODULUS_CASE(139);
    RNS8_CK_FIXED_MODULUS_CASE(137);
    RNS8_CK_FIXED_MODULUS_CASE(131);
    RNS8_CK_FIXED_MODULUS_CASE(127);
    default:
      return ck_unsupported_backend;
  }
#undef RNS8_CK_FIXED_MODULUS_CASE
}
