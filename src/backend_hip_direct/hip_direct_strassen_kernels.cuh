__device__ __constant__ rns8::detail::strassen::Program rns8_strassen_program =
    rns8::detail::strassen::program();
__device__ int8_t rns8_strassen_center(int value, int modulus) {
  value %= modulus;
  if (value < 0)
    value += modulus;
  if (value >= (modulus + 1) / 2)
    value -= modulus;
  return static_cast<int8_t>(value);
}
__global__ void rns8_strassen_transform_kernel(const int8_t *source,
                                               int8_t *output, int rows,
                                               int cols, int ld, int half_rows,
                                               int half_cols, int product,
                                               bool lhs) {
  const int64_t cell =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = static_cast<int64_t>(half_rows) * half_cols;
  if (cell >= count)
    return;
  const int plane = blockIdx.y, modulus = rns8_default_moduli_device[plane];
  const auto &term = rns8_strassen_program.products[product];
  int value = 0;
  for (int block = 0; block < 4; ++block) {
    const int coefficient = lhs ? term.a[block] : term.b[block];
    const int64_t row = cell / half_cols + (block / 2) * int64_t{half_rows};
    const int64_t col = cell % half_cols + (block % 2) * int64_t{half_cols};
    if (coefficient && row < rows && col < cols)
      value += coefficient *
               source[(static_cast<int64_t>(plane) * rows + row) * ld + col];
  }
  // Each transform uses at most two centered bytes; reduce before INT8 GEMM.
  output[static_cast<int64_t>(plane) * count + cell] =
      rns8_strassen_center(value, modulus);
}
__global__ void rns8_strassen_combine_kernel(const int8_t *product,
                                             int8_t *stage, int m, int n,
                                             int hm, int hn,
                                             int product_index) {
  const int64_t cell =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= static_cast<int64_t>(m) * n)
    return;
  const int64_t row = cell / n, col = cell % n;
  const int quadrant = (row / hm) * 2 + col / hn, plane = blockIdx.y;
  const int coefficient =
      rns8_strassen_program.products[product_index].c[quadrant];
  const int64_t output = static_cast<int64_t>(plane) * m * n + cell;
  const int64_t input =
      (static_cast<int64_t>(plane) * hm + row % hm) * hn + col % hn;
  // Reduction after every product bounds recombination by two centered bytes.
  stage[output] =
      rns8_strassen_center(stage[output] + coefficient * product[input],
                           rns8_default_moduli_device[plane]);
}
__global__ void rns8_strassen_commit_kernel(const int8_t *stage, int8_t *output,
                                            int m, int n, int ldc) {
  const int64_t cell =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= static_cast<int64_t>(m) * n)
    return;
  output[(static_cast<int64_t>(blockIdx.y) * m + cell / n) * ldc + cell % n] =
      stage[static_cast<int64_t>(blockIdx.y) * m * n + cell];
}
