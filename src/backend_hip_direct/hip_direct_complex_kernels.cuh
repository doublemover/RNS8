__global__ void rns8_complex_sum_kernel(const int8_t *real, const int8_t *imag, int8_t *sum,
                                        int rows, int cols, int ldr, int ldi,
                                        uint32_t finite_modulus) {
  const int64_t cell = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= int64_t(rows) * cols)
    return;
  const int64_t row = cell / cols, col = cell % cols, plane = blockIdx.y;
  const uint32_t modulus = finite_modulus ? finite_modulus : rns8_default_moduli_device[plane];
  sum[plane * rows * cols + cell] =
      rns8::detail::centered_residue_axpby(real[(plane * rows + row) * ldr + col],
                                           imag[(plane * rows + row) * ldi + col], 1, 1, modulus);
}
__global__ void rns8_complex_combine_kernel(const int8_t *products, int8_t *real, int8_t *imag,
                                            int rows, int cols, int prefix, uint32_t finite_modulus,
                                            bool three_real) {
  const int64_t cell = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= int64_t(rows) * cols)
    return;
  const int64_t offset = int64_t(blockIdx.y) * rows * cols + cell;
  const int64_t stride = int64_t(prefix) * rows * cols;
  const uint32_t modulus = finite_modulus ? finite_modulus : rns8_default_moduli_device[blockIdx.y];
  const uint32_t reciprocal = rns8::detail::finite_u8::modulus_reciprocal_u32(modulus);
  const int32_t p0 = products[offset], p1 = products[stride + offset],
                p2 = products[2 * stride + offset];
  // Each product is already centered after safe K slabs. These expressions
  // lie in [-383,383], so recombination never adds unrestricted INT32 dots.
  real[offset] = rns8::detail::finite_u8::reduce_to_centered_i32(p0 - p1, modulus, reciprocal);
  imag[offset] = rns8::detail::finite_u8::reduce_to_centered_i32(
      three_real ? p2 - p0 - p1 : p2 + int32_t(products[3 * stride + offset]), modulus, reciprocal);
}
__global__ void rns8_complex_commit_kernel(const int8_t *real, const int8_t *imag,
                                           int8_t *output_real, int8_t *output_imag, int rows,
                                           int cols, int ldr, int ldi) {
  const int64_t cell = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= int64_t(rows) * cols)
    return;
  const int64_t row = cell / cols, col = cell % cols, plane = blockIdx.y;
  output_real[(plane * rows + row) * ldr + col] = real[plane * rows * cols + cell];
  output_imag[(plane * rows + row) * ldi + col] = imag[plane * rows * cols + cell];
}
