extern "C" int rns8_hip_direct_pack_i64_device_on_stream(
    const int64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix,
    void* stream) {
  hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
  const int threads = 256;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t total =
      static_cast<int64_t>(prefix) * elements;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const int fixed_blocks = static_cast<int>((elements + threads - 1) / threads);
  const int fixed_pair_blocks = static_cast<int>(((elements + 1) / 2 + threads - 1) / threads);
  const bool contiguous = ld == cols;
  switch (prefix) {
#define RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(PrefixValue) \
    case PrefixValue:                                  \
      if (contiguous) {                                \
        hipLaunchKernelGGL(                            \
            HIP_KERNEL_NAME(rns8_pack_i64_fixed_prefix_contiguous_pair_kernel<PrefixValue>), \
            dim3(fixed_pair_blocks),                   \
            dim3(threads),                             \
            0,                                         \
            hip_stream,                                \
            d_src,                                     \
            d_residues,                                \
            rows,                                      \
            cols);                                     \
      } else {                                         \
        hipLaunchKernelGGL(                            \
            HIP_KERNEL_NAME(rns8_pack_i64_fixed_prefix_kernel<PrefixValue>), \
            dim3(fixed_blocks),                        \
            dim3(threads),                             \
            0,                                         \
            hip_stream,                                \
            d_src,                                     \
            d_residues,                                \
            rows,                                      \
            cols,                                      \
            ld);                                       \
      }                                                \
      break
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(1);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(2);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(3);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(4);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(5);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(6);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(7);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(8);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(kRns8DefaultBoundedPrefix);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(18);
    RNS8_LAUNCH_PACK_I64_FIXED_PREFIX(kRns8MaxSupportedPrefix);
#undef RNS8_LAUNCH_PACK_I64_FIXED_PREFIX
    default:
      if (contiguous) {
        hipLaunchKernelGGL(
            rns8_pack_i64_contiguous_kernel,
            dim3(blocks),
            dim3(threads),
            0,
            hip_stream,
            d_src,
            d_residues,
            rows,
            cols,
            prefix);
      } else {
        hipLaunchKernelGGL(
            rns8_pack_i64_kernel,
            dim3(blocks),
            dim3(threads),
            0,
            hip_stream,
            d_src,
            d_residues,
            rows,
            cols,
            ld,
            prefix);
      }
      break;
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_pack_i64_device(
    const int64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
  return rns8_hip_direct_pack_i64_device_on_stream(d_src, d_residues, rows, cols, ld, prefix, nullptr);
}

extern "C" int rns8_hip_direct_pack_u64_device_on_stream(
    const uint64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix,
    void* stream) {
  hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
  const int threads = 256;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t total =
      static_cast<int64_t>(prefix) * elements;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const int fixed_blocks = static_cast<int>((elements + threads - 1) / threads);
  const int fixed_pair_blocks = static_cast<int>(((elements + 1) / 2 + threads - 1) / threads);
  const bool contiguous = ld == cols;
  switch (prefix) {
#define RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(PrefixValue) \
    case PrefixValue:                                  \
      if (contiguous) {                                \
        hipLaunchKernelGGL(                            \
            HIP_KERNEL_NAME(rns8_pack_u64_fixed_prefix_contiguous_pair_kernel<PrefixValue>), \
            dim3(fixed_pair_blocks),                   \
            dim3(threads),                             \
            0,                                         \
            hip_stream,                                \
            d_src,                                     \
            d_residues,                                \
            rows,                                      \
            cols);                                     \
      } else {                                         \
        hipLaunchKernelGGL(                            \
            HIP_KERNEL_NAME(rns8_pack_u64_fixed_prefix_kernel<PrefixValue>), \
            dim3(fixed_blocks),                        \
            dim3(threads),                             \
            0,                                         \
            hip_stream,                                \
            d_src,                                     \
            d_residues,                                \
            rows,                                      \
            cols,                                      \
            ld);                                       \
      }                                                \
      break
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(1);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(2);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(3);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(4);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(5);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(6);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(7);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(8);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(kRns8DefaultBoundedPrefix);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(18);
    RNS8_LAUNCH_PACK_U64_FIXED_PREFIX(kRns8MaxSupportedPrefix);
#undef RNS8_LAUNCH_PACK_U64_FIXED_PREFIX
    default:
      if (contiguous) {
        hipLaunchKernelGGL(
            rns8_pack_u64_contiguous_kernel,
            dim3(blocks),
            dim3(threads),
            0,
            hip_stream,
            d_src,
            d_residues,
            rows,
            cols,
            prefix);
      } else {
        hipLaunchKernelGGL(
            rns8_pack_u64_kernel,
            dim3(blocks),
            dim3(threads),
            0,
            hip_stream,
            d_src,
            d_residues,
            rows,
            cols,
            ld,
            prefix);
      }
      break;
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_pack_u64_device(
    const uint64_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
  return rns8_hip_direct_pack_u64_device_on_stream(d_src, d_residues, rows, cols, ld, prefix, nullptr);
}

extern "C" int rns8_hip_direct_pack_i64_grouped_device(
    const int64_t* d_src,
    int8_t* const* d_residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld,
    int prefix) {
  if (!d_src || !d_residue_ptrs || task_count <= 0 || src_task_stride <= 0 ||
      rows <= 0 || cols <= 0 || ld < cols || prefix <= 0 || prefix > kRns8MaxSupportedPrefix) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t per_task_total = static_cast<int64_t>(prefix) * elements;
  const int blocks = static_cast<int>((per_task_total + threads - 1) / threads);
  const int fixed_blocks = static_cast<int>((elements + threads - 1) / threads);
  const int fixed_pair_blocks = static_cast<int>(((elements + 1) / 2 + threads - 1) / threads);
  const bool contiguous = ld == cols;
  switch (prefix) {
#define RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(PrefixValue) \
    case PrefixValue:                                          \
      if (contiguous) {                                        \
        hipLaunchKernelGGL(                                    \
            HIP_KERNEL_NAME(rns8_pack_i64_grouped_fixed_prefix_contiguous_pair_kernel<PrefixValue>), \
            dim3(fixed_pair_blocks, task_count),               \
            dim3(threads),                                     \
            0,                                                 \
            nullptr,                                           \
            d_src,                                             \
            d_residue_ptrs,                                    \
            task_count,                                        \
            src_task_stride,                                   \
            rows,                                              \
            cols);                                             \
      } else {                                                 \
        hipLaunchKernelGGL(                                    \
            HIP_KERNEL_NAME(rns8_pack_i64_grouped_fixed_prefix_kernel<PrefixValue>), \
            dim3(fixed_blocks, task_count),                    \
            dim3(threads),                                     \
            0,                                                 \
            nullptr,                                           \
            d_src,                                             \
            d_residue_ptrs,                                    \
            task_count,                                        \
            src_task_stride,                                   \
            rows,                                              \
            cols,                                              \
            ld);                                               \
      }                                                        \
      break
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(1);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(2);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(3);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(4);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(5);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(6);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(7);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(8);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(kRns8DefaultBoundedPrefix);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(18);
    RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX(kRns8MaxSupportedPrefix);
#undef RNS8_LAUNCH_GROUPED_PACK_I64_FIXED_PREFIX
    default:
      if (contiguous) {
        hipLaunchKernelGGL(
            rns8_pack_i64_grouped_contiguous_kernel,
            dim3(blocks, task_count),
            dim3(threads),
            0,
            nullptr,
            d_src,
            d_residue_ptrs,
            task_count,
            src_task_stride,
            rows,
            cols,
            prefix);
      } else {
        hipLaunchKernelGGL(
            rns8_pack_i64_grouped_kernel,
            dim3(blocks, task_count),
            dim3(threads),
            0,
            nullptr,
            d_src,
            d_residue_ptrs,
            task_count,
            src_task_stride,
            rows,
            cols,
            ld,
            prefix);
      }
      break;
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_pack_u64_grouped_device(
    const uint64_t* d_src,
    int8_t* const* d_residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld,
    int prefix) {
  if (!d_src || !d_residue_ptrs || task_count <= 0 || src_task_stride <= 0 ||
      rows <= 0 || cols <= 0 || ld < cols || prefix <= 0 || prefix > kRns8MaxSupportedPrefix) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t per_task_total = static_cast<int64_t>(prefix) * elements;
  const int blocks = static_cast<int>((per_task_total + threads - 1) / threads);
  const int fixed_blocks = static_cast<int>((elements + threads - 1) / threads);
  const int fixed_pair_blocks = static_cast<int>(((elements + 1) / 2 + threads - 1) / threads);
  const bool contiguous = ld == cols;
  switch (prefix) {
#define RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(PrefixValue) \
    case PrefixValue:                                          \
      if (contiguous) {                                        \
        hipLaunchKernelGGL(                                    \
            HIP_KERNEL_NAME(rns8_pack_u64_grouped_fixed_prefix_contiguous_pair_kernel<PrefixValue>), \
            dim3(fixed_pair_blocks, task_count),               \
            dim3(threads),                                     \
            0,                                                 \
            nullptr,                                           \
            d_src,                                             \
            d_residue_ptrs,                                    \
            task_count,                                        \
            src_task_stride,                                   \
            rows,                                              \
            cols);                                             \
      } else {                                                 \
        hipLaunchKernelGGL(                                    \
            HIP_KERNEL_NAME(rns8_pack_u64_grouped_fixed_prefix_kernel<PrefixValue>), \
            dim3(fixed_blocks, task_count),                    \
            dim3(threads),                                     \
            0,                                                 \
            nullptr,                                           \
            d_src,                                             \
            d_residue_ptrs,                                    \
            task_count,                                        \
            src_task_stride,                                   \
            rows,                                              \
            cols,                                              \
            ld);                                               \
      }                                                        \
      break
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(1);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(2);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(3);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(4);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(5);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(6);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(7);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(8);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(kRns8DefaultBoundedPrefix);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(18);
    RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX(kRns8MaxSupportedPrefix);
#undef RNS8_LAUNCH_GROUPED_PACK_U64_FIXED_PREFIX
    default:
      if (contiguous) {
        hipLaunchKernelGGL(
            rns8_pack_u64_grouped_contiguous_kernel,
            dim3(blocks, task_count),
            dim3(threads),
            0,
            nullptr,
            d_src,
            d_residue_ptrs,
            task_count,
            src_task_stride,
            rows,
            cols,
            prefix);
      } else {
        hipLaunchKernelGGL(
            rns8_pack_u64_grouped_kernel,
            dim3(blocks, task_count),
            dim3(threads),
            0,
            nullptr,
            d_src,
            d_residue_ptrs,
            task_count,
            src_task_stride,
            rows,
            cols,
            ld,
            prefix);
      }
      break;
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_pack_u8_modulus_device_on_stream(
    const uint8_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int modulus,
    void* stream) {
  if (!d_src || !d_residues || rows <= 0 || cols <= 0 || ld < cols || modulus < 2 || modulus > 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
  const int threads = 256;
  const int64_t total = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const bool contiguous = ld == cols;
  if (modulus == 256) {
#define RNS8_LAUNCH_PACK_U8_FIXED_MODULUS(ModulusValue) \
    if (contiguous) {                                    \
      hipLaunchKernelGGL(                                \
          HIP_KERNEL_NAME(rns8_pack_u8_fixed_modulus_contiguous_kernel<ModulusValue>), \
          dim3(blocks),                                  \
          dim3(threads),                                 \
          0,                                             \
          hip_stream,                                    \
          d_src,                                         \
          d_residues,                                    \
          rows,                                          \
          cols);                                         \
    } else {                                             \
      hipLaunchKernelGGL(                                \
          HIP_KERNEL_NAME(rns8_pack_u8_fixed_modulus_kernel<ModulusValue>), \
          dim3(blocks),                                  \
          dim3(threads),                                 \
          0,                                             \
          hip_stream,                                    \
          d_src,                                         \
          d_residues,                                    \
          rows,                                          \
          cols,                                          \
          ld);                                           \
    }
    RNS8_LAUNCH_PACK_U8_FIXED_MODULUS(256);
    return static_cast<int>(hipGetLastError());
  }
  if (modulus == 255) {
    RNS8_LAUNCH_PACK_U8_FIXED_MODULUS(255);
    return static_cast<int>(hipGetLastError());
  }
  if (modulus == 251) {
    RNS8_LAUNCH_PACK_U8_FIXED_MODULUS(251);
    return static_cast<int>(hipGetLastError());
  }
#undef RNS8_LAUNCH_PACK_U8_FIXED_MODULUS
  if (contiguous) {
    hipLaunchKernelGGL(
        rns8_pack_u8_modulus_contiguous_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        hip_stream,
        d_src,
        d_residues,
        rows,
        cols,
        modulus);
  } else {
    hipLaunchKernelGGL(
        rns8_pack_u8_modulus_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        hip_stream,
        d_src,
        d_residues,
        rows,
        cols,
        ld,
        modulus);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_direct_pack_u8_modulus_device(
    const uint8_t* d_src,
    int8_t* d_residues,
    int rows,
    int cols,
    int ld,
    int modulus) {
  return rns8_hip_direct_pack_u8_modulus_device_on_stream(d_src, d_residues, rows, cols, ld, modulus, nullptr);
}

extern "C" int rns8_hip_direct_pack_u8_grouped_modulus_device(
    const uint8_t* d_src,
    int8_t* const* d_residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld,
    int modulus) {
  if (!d_src || !d_residue_ptrs || task_count <= 0 || rows <= 0 || cols <= 0 || ld < cols ||
      src_task_stride < static_cast<int64_t>(rows) * static_cast<int64_t>(ld) ||
      modulus < 2 || modulus > 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t total = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const bool contiguous = ld == cols;
  if (modulus == 256) {
#define RNS8_LAUNCH_GROUPED_PACK_U8_FIXED_MODULUS(ModulusValue) \
    if (contiguous) {                                           \
      hipLaunchKernelGGL(                                       \
          HIP_KERNEL_NAME(rns8_pack_u8_grouped_fixed_modulus_contiguous_kernel<ModulusValue>), \
          dim3(blocks, task_count),                             \
          dim3(threads),                                        \
          0,                                                    \
          nullptr,                                              \
          d_src,                                                \
          d_residue_ptrs,                                       \
          task_count,                                           \
          src_task_stride,                                      \
          rows,                                                 \
          cols);                                                \
    } else {                                                    \
      hipLaunchKernelGGL(                                       \
          HIP_KERNEL_NAME(rns8_pack_u8_grouped_fixed_modulus_kernel<ModulusValue>), \
          dim3(blocks, task_count),                             \
          dim3(threads),                                        \
          0,                                                    \
          nullptr,                                              \
          d_src,                                                \
          d_residue_ptrs,                                       \
          task_count,                                           \
          src_task_stride,                                      \
          rows,                                                 \
          cols,                                                 \
          ld);                                                  \
    }
    RNS8_LAUNCH_GROUPED_PACK_U8_FIXED_MODULUS(256);
    return static_cast<int>(hipGetLastError());
  }
  if (modulus == 255) {
    RNS8_LAUNCH_GROUPED_PACK_U8_FIXED_MODULUS(255);
    return static_cast<int>(hipGetLastError());
  }
  if (modulus == 251) {
    RNS8_LAUNCH_GROUPED_PACK_U8_FIXED_MODULUS(251);
    return static_cast<int>(hipGetLastError());
  }
#undef RNS8_LAUNCH_GROUPED_PACK_U8_FIXED_MODULUS
  if (contiguous) {
    hipLaunchKernelGGL(
        rns8_pack_u8_grouped_modulus_contiguous_kernel,
        dim3(blocks, task_count),
        dim3(threads),
        0,
        nullptr,
        d_src,
        d_residue_ptrs,
        task_count,
        src_task_stride,
        rows,
        cols,
        modulus);
  } else {
    hipLaunchKernelGGL(
        rns8_pack_u8_grouped_modulus_kernel,
        dim3(blocks, task_count),
        dim3(threads),
        0,
        nullptr,
        d_src,
        d_residue_ptrs,
        task_count,
        src_task_stride,
        rows,
        cols,
        ld,
        modulus);
  }
  return static_cast<int>(hipGetLastError());
}

int rns8_hip_direct_ring_gemm_i8_on_stream(
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
    hipStream_t stream) {
  if (!d_a || !d_b || !d_c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || modulus <= 0 ||
      modulus_index < 0 || selected_prefix <= 0 || modulus_index >= selected_prefix ||
      selected_prefix > kRns8DefaultModulusCount ||
      modulus != rns8_default_moduli_host[modulus_index] ||
      modulus_reciprocal != rns8_modulus_reciprocal_host(modulus) ||
      safe_k_block <= 0 || safe_k_block > kRns8SafeInt32KBlock) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(kRns8HipTileN, kRns8HipTileM);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
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
        stream,
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

