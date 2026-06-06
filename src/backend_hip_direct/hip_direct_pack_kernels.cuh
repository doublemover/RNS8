__global__ void rns8_pack_i64_kernel(
    const int64_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t total = elements * static_cast<int64_t>(prefix);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int modulus_index = static_cast<int>(idx / elements);
  const int64_t element = idx - static_cast<int64_t>(modulus_index) * elements;
  const int row = static_cast<int>(element / cols);
  const int col = static_cast<int>(element - static_cast<int64_t>(row) * cols);
  const int modulus = rns8_default_moduli_device[modulus_index];
  residues[idx] = rns8_center_i64_device(src[static_cast<int64_t>(row) * ld + col], modulus);
}

__global__ void rns8_pack_u64_kernel(
    const uint64_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t total = elements * static_cast<int64_t>(prefix);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int modulus_index = static_cast<int>(idx / elements);
  const int64_t element = idx - static_cast<int64_t>(modulus_index) * elements;
  const int row = static_cast<int>(element / cols);
  const int col = static_cast<int>(element - static_cast<int64_t>(row) * cols);
  const int modulus = rns8_default_moduli_device[modulus_index];
  residues[idx] = rns8_center_u64_device(src[static_cast<int64_t>(row) * ld + col], modulus);
}

template <int Prefix>
__global__ void rns8_pack_i64_fixed_prefix_kernel(
    const int64_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= elements) {
    return;
  }
  const int row = static_cast<int>(cell / cols);
  const int col = static_cast<int>(cell - static_cast<int64_t>(row) * cols);
  const int64_t value = src[static_cast<int64_t>(row) * ld + col];
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    residues[static_cast<int64_t>(modulus_index) * elements + cell] =
        rns8_center_i64_default_modulus_fixed_device(value, modulus_index, modulus);
  }
}

template <int Prefix>
__global__ void rns8_pack_u64_fixed_prefix_kernel(
    const uint64_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= elements) {
    return;
  }
  const int row = static_cast<int>(cell / cols);
  const int col = static_cast<int>(cell - static_cast<int64_t>(row) * cols);
  const uint64_t value = src[static_cast<int64_t>(row) * ld + col];
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    residues[static_cast<int64_t>(modulus_index) * elements + cell] =
        rns8_center_u64_default_modulus_fixed_device(value, modulus_index, modulus);
  }
}

template <int Prefix>
__global__ void rns8_pack_i64_grouped_fixed_prefix_kernel(
    const int64_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= elements) {
    return;
  }
  const int row = static_cast<int>(cell / cols);
  const int col = static_cast<int>(cell - static_cast<int64_t>(row) * cols);
  const int64_t value =
      src[static_cast<int64_t>(task) * src_task_stride + static_cast<int64_t>(row) * ld + col];
  int8_t* residues = residue_ptrs[task];
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    residues[static_cast<int64_t>(modulus_index) * elements + cell] =
        rns8_center_i64_default_modulus_fixed_device(value, modulus_index, modulus);
  }
}

template <int Prefix>
__global__ void rns8_pack_u64_grouped_fixed_prefix_kernel(
    const uint64_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= elements) {
    return;
  }
  const int row = static_cast<int>(cell / cols);
  const int col = static_cast<int>(cell - static_cast<int64_t>(row) * cols);
  const uint64_t value =
      src[static_cast<int64_t>(task) * src_task_stride + static_cast<int64_t>(row) * ld + col];
  int8_t* residues = residue_ptrs[task];
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    residues[static_cast<int64_t>(modulus_index) * elements + cell] =
        rns8_center_u64_default_modulus_fixed_device(value, modulus_index, modulus);
  }
}

__global__ void rns8_pack_i64_grouped_kernel(
    const int64_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t per_task_total = elements * static_cast<int64_t>(prefix);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int task = blockIdx.y;
  if (task >= task_count || idx >= per_task_total) {
    return;
  }
  const int modulus_index = static_cast<int>(idx / elements);
  const int64_t element = idx - static_cast<int64_t>(modulus_index) * elements;
  const int row = static_cast<int>(element / cols);
  const int col = static_cast<int>(element - static_cast<int64_t>(row) * cols);
  const int modulus = rns8_default_moduli_device[modulus_index];
  const int64_t value =
      src[static_cast<int64_t>(task) * src_task_stride + static_cast<int64_t>(row) * ld + col];
  int8_t* residues = residue_ptrs[task];
  residues[idx] = rns8_center_i64_device(value, modulus);
}

__global__ void rns8_pack_u64_grouped_kernel(
    const uint64_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t per_task_total = elements * static_cast<int64_t>(prefix);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int task = blockIdx.y;
  if (task >= task_count || idx >= per_task_total) {
    return;
  }
  const int modulus_index = static_cast<int>(idx / elements);
  const int64_t element = idx - static_cast<int64_t>(modulus_index) * elements;
  const int row = static_cast<int>(element / cols);
  const int col = static_cast<int>(element - static_cast<int64_t>(row) * cols);
  const int modulus = rns8_default_moduli_device[modulus_index];
  const uint64_t value =
      src[static_cast<int64_t>(task) * src_task_stride + static_cast<int64_t>(row) * ld + col];
  int8_t* residues = residue_ptrs[task];
  residues[idx] = rns8_center_u64_device(value, modulus);
}

__global__ void rns8_pack_u8_modulus_kernel(
    const uint8_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld,
    int modulus) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  const int row = static_cast<int>(idx / cols);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  residues[idx] = rns8_center_u8_device(src[static_cast<int64_t>(row) * ld + col], modulus);
}

__global__ void rns8_pack_u8_grouped_modulus_kernel(
    const uint8_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld,
    int modulus) {
  const int task = blockIdx.y;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  if (task >= task_count || idx >= elements) {
    return;
  }
  const int row = static_cast<int>(idx / cols);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  const uint8_t value =
      src[static_cast<int64_t>(task) * src_task_stride + static_cast<int64_t>(row) * ld + col];
  residue_ptrs[task][idx] = rns8_center_u8_device(value, modulus);
}

template <int Modulus>
__global__ void rns8_pack_u8_fixed_modulus_kernel(
    const uint8_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int ld) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  const int row = static_cast<int>(idx / cols);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  residues[idx] = rns8_center_u8_fixed_modulus_device<Modulus>(src[static_cast<int64_t>(row) * ld + col]);
}

template <int Modulus>
__global__ void rns8_pack_u8_grouped_fixed_modulus_kernel(
    const uint8_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int ld) {
  const int task = blockIdx.y;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  if (task >= task_count || idx >= elements) {
    return;
  }
  const int row = static_cast<int>(idx / cols);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  const uint8_t value =
      src[static_cast<int64_t>(task) * src_task_stride + static_cast<int64_t>(row) * ld + col];
  residue_ptrs[task][idx] = rns8_center_u8_fixed_modulus_device<Modulus>(value);
}

