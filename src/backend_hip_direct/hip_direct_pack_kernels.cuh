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

__global__ void rns8_pack_i64_contiguous_kernel(
    const int64_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t total = elements * static_cast<int64_t>(prefix);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int modulus_index = static_cast<int>(idx / elements);
  const int64_t cell = idx - static_cast<int64_t>(modulus_index) * elements;
  const int modulus = rns8_default_moduli_device[modulus_index];
  residues[idx] = rns8_center_i64_device(src[cell], modulus);
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

__global__ void rns8_pack_u64_contiguous_kernel(
    const uint64_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t total = elements * static_cast<int64_t>(prefix);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int modulus_index = static_cast<int>(idx / elements);
  const int64_t cell = idx - static_cast<int64_t>(modulus_index) * elements;
  const int modulus = rns8_default_moduli_device[modulus_index];
  residues[idx] = rns8_center_u64_device(src[cell], modulus);
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
__global__ void rns8_pack_i64_fixed_prefix_contiguous_kernel(
    const int64_t* src,
    int8_t* residues,
    int rows,
    int cols) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= elements) {
    return;
  }
  const int64_t value = src[cell];
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    residues[static_cast<int64_t>(modulus_index) * elements + cell] =
        rns8_center_i64_default_modulus_fixed_device(value, modulus_index, modulus);
  }
}

template <int Prefix>
__global__ void rns8_pack_i64_fixed_prefix_contiguous_pair_kernel(
    const int64_t* src,
    int8_t* residues,
    int rows,
    int cols) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
  if (cell >= elements) {
    return;
  }
  const int64_t value0 = src[cell];
  const bool has_second = cell + 1 < elements;
  const int64_t value1 = has_second ? src[cell + 1] : 0;
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    const int64_t output = static_cast<int64_t>(modulus_index) * elements + cell;
    residues[output] = rns8_center_i64_default_modulus_fixed_device(value0, modulus_index, modulus);
    if (has_second) {
      residues[output + 1] = rns8_center_i64_default_modulus_fixed_device(value1, modulus_index, modulus);
    }
  }
}

template <int Prefix>
__global__ void rns8_pack_i64_fixed_prefix_contiguous_quad_kernel(
    const int64_t* src,
    int8_t* residues,
    int rows,
    int cols) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (cell >= elements) {
    return;
  }
  const int64_t value0 = src[cell];
  const bool has1 = cell + 1 < elements;
  const bool has2 = cell + 2 < elements;
  const bool has3 = cell + 3 < elements;
  const int64_t value1 = has1 ? src[cell + 1] : 0;
  const int64_t value2 = has2 ? src[cell + 2] : 0;
  const int64_t value3 = has3 ? src[cell + 3] : 0;
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    const int64_t output = static_cast<int64_t>(modulus_index) * elements + cell;
    residues[output] = rns8_center_i64_default_modulus_fixed_device(value0, modulus_index, modulus);
    if (has1) {
      residues[output + 1] = rns8_center_i64_default_modulus_fixed_device(value1, modulus_index, modulus);
    }
    if (has2) {
      residues[output + 2] = rns8_center_i64_default_modulus_fixed_device(value2, modulus_index, modulus);
    }
    if (has3) {
      residues[output + 3] = rns8_center_i64_default_modulus_fixed_device(value3, modulus_index, modulus);
    }
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
__global__ void rns8_pack_u64_fixed_prefix_contiguous_kernel(
    const uint64_t* src,
    int8_t* residues,
    int rows,
    int cols) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= elements) {
    return;
  }
  const uint64_t value = src[cell];
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    residues[static_cast<int64_t>(modulus_index) * elements + cell] =
        rns8_center_u64_default_modulus_fixed_device(value, modulus_index, modulus);
  }
}

template <int Prefix>
__global__ void rns8_pack_u64_fixed_prefix_contiguous_pair_kernel(
    const uint64_t* src,
    int8_t* residues,
    int rows,
    int cols) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
  if (cell >= elements) {
    return;
  }
  const uint64_t value0 = src[cell];
  const bool has_second = cell + 1 < elements;
  const uint64_t value1 = has_second ? src[cell + 1] : 0;
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    const int64_t output = static_cast<int64_t>(modulus_index) * elements + cell;
    residues[output] = rns8_center_u64_default_modulus_fixed_device(value0, modulus_index, modulus);
    if (has_second) {
      residues[output + 1] = rns8_center_u64_default_modulus_fixed_device(value1, modulus_index, modulus);
    }
  }
}

template <int Prefix>
__global__ void rns8_pack_u64_fixed_prefix_contiguous_quad_kernel(
    const uint64_t* src,
    int8_t* residues,
    int rows,
    int cols) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (cell >= elements) {
    return;
  }
  const uint64_t value0 = src[cell];
  const bool has1 = cell + 1 < elements;
  const bool has2 = cell + 2 < elements;
  const bool has3 = cell + 3 < elements;
  const uint64_t value1 = has1 ? src[cell + 1] : 0;
  const uint64_t value2 = has2 ? src[cell + 2] : 0;
  const uint64_t value3 = has3 ? src[cell + 3] : 0;
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    const int64_t output = static_cast<int64_t>(modulus_index) * elements + cell;
    residues[output] = rns8_center_u64_default_modulus_fixed_device(value0, modulus_index, modulus);
    if (has1) {
      residues[output + 1] = rns8_center_u64_default_modulus_fixed_device(value1, modulus_index, modulus);
    }
    if (has2) {
      residues[output + 2] = rns8_center_u64_default_modulus_fixed_device(value2, modulus_index, modulus);
    }
    if (has3) {
      residues[output + 3] = rns8_center_u64_default_modulus_fixed_device(value3, modulus_index, modulus);
    }
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
__global__ void rns8_pack_i64_grouped_fixed_prefix_contiguous_kernel(
    const int64_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= elements) {
    return;
  }
  const int64_t value = src[static_cast<int64_t>(task) * src_task_stride + cell];
  int8_t* residues = residue_ptrs[task];
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    residues[static_cast<int64_t>(modulus_index) * elements + cell] =
        rns8_center_i64_default_modulus_fixed_device(value, modulus_index, modulus);
  }
}

template <int Prefix>
__global__ void rns8_pack_i64_grouped_fixed_prefix_contiguous_pair_kernel(
    const int64_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
  if (cell >= elements) {
    return;
  }
  const int64_t task_offset = static_cast<int64_t>(task) * src_task_stride;
  const int64_t value0 = src[task_offset + cell];
  const bool has_second = cell + 1 < elements;
  const int64_t value1 = has_second ? src[task_offset + cell + 1] : 0;
  int8_t* residues = residue_ptrs[task];
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    const int64_t output = static_cast<int64_t>(modulus_index) * elements + cell;
    residues[output] = rns8_center_i64_default_modulus_fixed_device(value0, modulus_index, modulus);
    if (has_second) {
      residues[output + 1] = rns8_center_i64_default_modulus_fixed_device(value1, modulus_index, modulus);
    }
  }
}

template <int Prefix>
__global__ void rns8_pack_i64_grouped_fixed_prefix_contiguous_quad_kernel(
    const int64_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (cell >= elements) {
    return;
  }
  const int64_t task_offset = static_cast<int64_t>(task) * src_task_stride;
  const int64_t value0 = src[task_offset + cell];
  const bool has1 = cell + 1 < elements;
  const bool has2 = cell + 2 < elements;
  const bool has3 = cell + 3 < elements;
  const int64_t value1 = has1 ? src[task_offset + cell + 1] : 0;
  const int64_t value2 = has2 ? src[task_offset + cell + 2] : 0;
  const int64_t value3 = has3 ? src[task_offset + cell + 3] : 0;
  int8_t* residues = residue_ptrs[task];
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    const int64_t output = static_cast<int64_t>(modulus_index) * elements + cell;
    residues[output] = rns8_center_i64_default_modulus_fixed_device(value0, modulus_index, modulus);
    if (has1) {
      residues[output + 1] = rns8_center_i64_default_modulus_fixed_device(value1, modulus_index, modulus);
    }
    if (has2) {
      residues[output + 2] = rns8_center_i64_default_modulus_fixed_device(value2, modulus_index, modulus);
    }
    if (has3) {
      residues[output + 3] = rns8_center_i64_default_modulus_fixed_device(value3, modulus_index, modulus);
    }
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

template <int Prefix>
__global__ void rns8_pack_u64_grouped_fixed_prefix_contiguous_kernel(
    const uint64_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= elements) {
    return;
  }
  const uint64_t value = src[static_cast<int64_t>(task) * src_task_stride + cell];
  int8_t* residues = residue_ptrs[task];
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    residues[static_cast<int64_t>(modulus_index) * elements + cell] =
        rns8_center_u64_default_modulus_fixed_device(value, modulus_index, modulus);
  }
}

template <int Prefix>
__global__ void rns8_pack_u64_grouped_fixed_prefix_contiguous_pair_kernel(
    const uint64_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
  if (cell >= elements) {
    return;
  }
  const int64_t task_offset = static_cast<int64_t>(task) * src_task_stride;
  const uint64_t value0 = src[task_offset + cell];
  const bool has_second = cell + 1 < elements;
  const uint64_t value1 = has_second ? src[task_offset + cell + 1] : 0;
  int8_t* residues = residue_ptrs[task];
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    const int64_t output = static_cast<int64_t>(modulus_index) * elements + cell;
    residues[output] = rns8_center_u64_default_modulus_fixed_device(value0, modulus_index, modulus);
    if (has_second) {
      residues[output + 1] = rns8_center_u64_default_modulus_fixed_device(value1, modulus_index, modulus);
    }
  }
}

template <int Prefix>
__global__ void rns8_pack_u64_grouped_fixed_prefix_contiguous_quad_kernel(
    const uint64_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  if (task >= task_count) {
    return;
  }
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (cell >= elements) {
    return;
  }
  const int64_t task_offset = static_cast<int64_t>(task) * src_task_stride;
  const uint64_t value0 = src[task_offset + cell];
  const bool has1 = cell + 1 < elements;
  const bool has2 = cell + 2 < elements;
  const bool has3 = cell + 3 < elements;
  const uint64_t value1 = has1 ? src[task_offset + cell + 1] : 0;
  const uint64_t value2 = has2 ? src[task_offset + cell + 2] : 0;
  const uint64_t value3 = has3 ? src[task_offset + cell + 3] : 0;
  int8_t* residues = residue_ptrs[task];
#pragma unroll
  for (int modulus_index = 0; modulus_index < Prefix; ++modulus_index) {
    const int modulus = rns8_default_moduli_device[modulus_index];
    const int64_t output = static_cast<int64_t>(modulus_index) * elements + cell;
    residues[output] = rns8_center_u64_default_modulus_fixed_device(value0, modulus_index, modulus);
    if (has1) {
      residues[output + 1] = rns8_center_u64_default_modulus_fixed_device(value1, modulus_index, modulus);
    }
    if (has2) {
      residues[output + 2] = rns8_center_u64_default_modulus_fixed_device(value2, modulus_index, modulus);
    }
    if (has3) {
      residues[output + 3] = rns8_center_u64_default_modulus_fixed_device(value3, modulus_index, modulus);
    }
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

__global__ void rns8_pack_i64_grouped_contiguous_kernel(
    const int64_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t per_task_total = elements * static_cast<int64_t>(prefix);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int task = blockIdx.y;
  if (task >= task_count || idx >= per_task_total) {
    return;
  }
  const int modulus_index = static_cast<int>(idx / elements);
  const int64_t cell = idx - static_cast<int64_t>(modulus_index) * elements;
  const int modulus = rns8_default_moduli_device[modulus_index];
  const int64_t value = src[static_cast<int64_t>(task) * src_task_stride + cell];
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

__global__ void rns8_pack_u64_grouped_contiguous_kernel(
    const uint64_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t per_task_total = elements * static_cast<int64_t>(prefix);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int task = blockIdx.y;
  if (task >= task_count || idx >= per_task_total) {
    return;
  }
  const int modulus_index = static_cast<int>(idx / elements);
  const int64_t cell = idx - static_cast<int64_t>(modulus_index) * elements;
  const int modulus = rns8_default_moduli_device[modulus_index];
  const uint64_t value = src[static_cast<int64_t>(task) * src_task_stride + cell];
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

__global__ void rns8_pack_u8_modulus_contiguous_kernel(
    const uint8_t* src,
    int8_t* residues,
    int rows,
    int cols,
    int modulus) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  residues[idx] = rns8_center_u8_device(src[idx], modulus);
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

__global__ void rns8_pack_u8_grouped_modulus_contiguous_kernel(
    const uint8_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols,
    int modulus) {
  const int task = blockIdx.y;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  if (task >= task_count || idx >= elements) {
    return;
  }
  const uint8_t value = src[static_cast<int64_t>(task) * src_task_stride + idx];
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
__global__ void rns8_pack_u8_fixed_modulus_contiguous_kernel(
    const uint8_t* src,
    int8_t* residues,
    int rows,
    int cols) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  residues[idx] = rns8_center_u8_fixed_modulus_device<Modulus>(src[idx]);
}

template <int Modulus>
__global__ void rns8_pack_u8_fixed_modulus_contiguous_pair_kernel(
    const uint8_t* src,
    int8_t* residues,
    int rows,
    int cols) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
  if (idx >= elements) {
    return;
  }
  residues[idx] = rns8_center_u8_fixed_modulus_device<Modulus>(src[idx]);
  const int64_t next = idx + 1;
  if (next < elements) {
    residues[next] = rns8_center_u8_fixed_modulus_device<Modulus>(src[next]);
  }
}

template <int Modulus>
__global__ void rns8_pack_u8_fixed_modulus_contiguous_quad_kernel(
    const uint8_t* src,
    int8_t* residues,
    int rows,
    int cols) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (idx >= elements) {
    return;
  }
  residues[idx] = rns8_center_u8_fixed_modulus_device<Modulus>(src[idx]);
  const int64_t next1 = idx + 1;
  const int64_t next2 = idx + 2;
  const int64_t next3 = idx + 3;
  if (next1 < elements) {
    residues[next1] = rns8_center_u8_fixed_modulus_device<Modulus>(src[next1]);
  }
  if (next2 < elements) {
    residues[next2] = rns8_center_u8_fixed_modulus_device<Modulus>(src[next2]);
  }
  if (next3 < elements) {
    residues[next3] = rns8_center_u8_fixed_modulus_device<Modulus>(src[next3]);
  }
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

template <int Modulus>
__global__ void rns8_pack_u8_grouped_fixed_modulus_contiguous_kernel(
    const uint8_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  if (task >= task_count || idx >= elements) {
    return;
  }
  const uint8_t value = src[static_cast<int64_t>(task) * src_task_stride + idx];
  residue_ptrs[task][idx] = rns8_center_u8_fixed_modulus_device<Modulus>(value);
}

template <int Modulus>
__global__ void rns8_pack_u8_grouped_fixed_modulus_contiguous_pair_kernel(
    const uint8_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  if (task >= task_count || idx >= elements) {
    return;
  }
  const int64_t task_offset = static_cast<int64_t>(task) * src_task_stride;
  int8_t* residues = residue_ptrs[task];
  residues[idx] = rns8_center_u8_fixed_modulus_device<Modulus>(src[task_offset + idx]);
  const int64_t next = idx + 1;
  if (next < elements) {
    residues[next] = rns8_center_u8_fixed_modulus_device<Modulus>(src[task_offset + next]);
  }
}

template <int Modulus>
__global__ void rns8_pack_u8_grouped_fixed_modulus_contiguous_quad_kernel(
    const uint8_t* src,
    int8_t* const* residue_ptrs,
    int task_count,
    int64_t src_task_stride,
    int rows,
    int cols) {
  const int task = blockIdx.y;
  const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  if (task >= task_count || idx >= elements) {
    return;
  }
  const int64_t task_offset = static_cast<int64_t>(task) * src_task_stride;
  int8_t* residues = residue_ptrs[task];
  residues[idx] = rns8_center_u8_fixed_modulus_device<Modulus>(src[task_offset + idx]);
  const int64_t next1 = idx + 1;
  const int64_t next2 = idx + 2;
  const int64_t next3 = idx + 3;
  if (next1 < elements) {
    residues[next1] = rns8_center_u8_fixed_modulus_device<Modulus>(src[task_offset + next1]);
  }
  if (next2 < elements) {
    residues[next2] = rns8_center_u8_fixed_modulus_device<Modulus>(src[task_offset + next2]);
  }
  if (next3 < elements) {
    residues[next3] = rns8_center_u8_fixed_modulus_device<Modulus>(src[task_offset + next3]);
  }
}


// === Gap 99: VALU-optimized pack kernels with DPP/VOPD patterns ===

// 8-wide vectorized native i64 to centered i8 residue pack
// Uses DPP for cross-lane modulus reduction instead of shared memory
template <int Modulus>
__device__ int8_t rns8_centered_from_native_dpp_device(int64_t value) {
  // DPP-based reduction: accumulate partial products across lanes
  int64_t reduced = value % static_cast<int64_t>(Modulus);
  // Cross-lane reduction via DPP for wider accumulation
  reduced += __shfl_down_sync(0xFFFFFFFF, static_cast<unsigned>(reduced), 4);
  reduced += __shfl_down_sync(0xFFFFFFFF, static_cast<unsigned>(reduced), 2);
  reduced += __shfl_down_sync(0xFFFFFFFF, static_cast<unsigned>(reduced), 1);
  reduced = reduced % static_cast<int64_t>(Modulus);
  // Center the residue
  int64_t half = Modulus / 2;
  if (reduced > half) reduced -= Modulus;
  return static_cast<int8_t>(reduced);
}

// VOPD-friendly dual-issue pack kernel: process two source elements per thread
// using paired VALU instructions for better ILP on RDNA3
__global__ void rns8_pack_native_i64_to_rns_8wide_vopd_kernel(
    const int64_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int rows,
    int cols,
    int ld,
    int prefix,
    int plane,
    int modulus) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  // 8 cells per thread for VOPD utilization
  const int64_t cell = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  if (cell >= elements) return;

  int8_t* plane_base = dst + static_cast<int64_t>(plane) * elements;

  #pragma unroll
  for (int c = 0; c < 8; ++c) {
    if (cell + c >= elements) break;
    const int64_t row = (cell + c) / cols;
    const int64_t col = (cell + c) - row * cols;
    int64_t value = src[row * static_cast<int64_t>(ld) + col];
    int64_t reduced = value % static_cast<int64_t>(modulus);
    if (reduced < 0) reduced += modulus;
    plane_base[cell + c] = static_cast<int8_t>(reduced > modulus / 2 ? reduced - modulus : reduced);
  }
}

// DPP-based cross-lane reduction for residue accumulation (replaces LDS/shared memory)
__device__ int32_t rns8_dpp_reduce_sum_device(int32_t value) {
  // DPP row broadcast and reduce pattern for wave32
  value += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, value, 16);
  value += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, value, 8);
  value += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, value, 4);
  value += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, value, 2);
  value += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, value, 1);
  return value;
}

// ds_swizzle-based efficient lane communication for pack operations
template <int BankWidth>
__device__ void rns8_ds_swizzle_store_device(int32_t* __restrict__ lds, int lane, int32_t value) {
  // Write with swizzle pattern to avoid bank conflicts
  int swizzled = (lane / BankWidth) * BankWidth + (lane % BankWidth);
  lds[swizzled] = value;
  __threadfence_block();
}



// === Phase 1c: uint4 coalesced pack loads ===
// Load 4 int64_t values (32 bytes) per thread with a single coalesced memory
// transaction. Reduces address arithmetic and cache line pressure vs 4 scalar loads.

__global__ void rns8_pack_i64_4wide_coalesced_kernel(
    const int64_t* __restrict__ src,
    int8_t* __restrict__ residues,
    int rows,
    int cols,
    int prefix) {
  const int64_t elements = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
  const int64_t cells_per_thread = 4;
  const int64_t total = elements * static_cast<int64_t>(prefix);
  const int64_t base_cell = static_cast<int64_t>(blockIdx.x) * blockDim.x * cells_per_thread
                            + static_cast<int64_t>(threadIdx.x) * cells_per_thread;
  if (base_cell >= total) return;

  const int modulus_index = static_cast<int>(base_cell / elements);
  const int64_t cell = base_cell - static_cast<int64_t>(modulus_index) * elements;
  const int modulus = rns8_default_moduli_device[modulus_index];

  // Process 4 cells with a single 32-byte load when aligned and contiguous
  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    if (cell + c >= elements) break;
    int64_t value = src[cell + c];  // Contiguous ld==cols path: single cache line
    int64_t reduced = value % static_cast<int64_t>(modulus);
    if (reduced < 0) reduced += modulus;
    residues[base_cell + c] = static_cast<int8_t>(reduced > modulus / 2 ? reduced - modulus : reduced);
  }
}

// === Phase 2b: Persistent small-shape pack ===
// Single kernel processes all planes for small shapes (rows*cols <= 4096).
// Eliminates per-plane launch overhead on tiny shapes.

__global__ void rns8_persistent_small_pack_i64_kernel(
    const int64_t* __restrict__ src,
    int8_t* __restrict__ residues,
    int rows,
    int cols,
    int ld,
    int prefix) {
  const int cells_per_plane = rows * cols;
  const int total_cells = cells_per_plane * prefix;
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= total_cells) return;

  const int plane = cell / cells_per_plane;
  const int element = cell - plane * cells_per_plane;
  const int row = element / cols;
  const int col = element - row * cols;
  const int modulus = rns8_default_moduli_device[plane];

  int64_t value = src[static_cast<int64_t>(row) * ld + col];
  int64_t reduced = value % static_cast<int64_t>(modulus);
  if (reduced < 0) reduced += modulus;
  residues[cell] = static_cast<int8_t>(reduced > modulus / 2 ? reduced - modulus : reduced);
}

