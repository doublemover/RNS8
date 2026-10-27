__device__ __constant__ rns8::detail::strassen::Program rns8_strassen_program =
    rns8::detail::strassen::program();
__device__ __constant__ rns8::detail::rank23::Program rns8_rank23_program =
    rns8::detail::rank23::program();
__device__ __constant__ rns8::detail::rank48::Program rns8_rank48_program =
    rns8::detail::rank48::program();
template <int Blocks> __device__ const auto &rns8_bilinear_term(int product) {
  if constexpr (Blocks == 2)
    return rns8_strassen_program.products[product];
  else if constexpr(Blocks==3)
    return rns8_rank23_program.products[product];
  else
    return rns8_rank48_program.products[product];
}
__device__ int8_t rns8_strassen_center(int value, int modulus) {
  value %= modulus;
  if (value < 0)
    value += modulus;
  if (value >= (modulus + 1) / 2)
    value -= modulus;
  return static_cast<int8_t>(value);
}
template <int Blocks>
__global__ void
rns8_bilinear_transform_kernel(const int8_t *source, int8_t *output, int rows,
                               int cols, int ld, int half_rows, int half_cols,
                               int product, bool lhs) {
  const int64_t cell =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = static_cast<int64_t>(half_rows) * half_cols;
  if (cell >= count)
    return;
  const int plane = blockIdx.y, modulus = rns8_default_moduli_device[plane];
  if constexpr(Blocks==4)if(!(modulus&1)){output[static_cast<int64_t>(plane)*count+cell]=0;return;}
  const auto &term = rns8_bilinear_term<Blocks>(product);
  int value = 0;
  for (int block = 0; block < Blocks * Blocks; ++block) {
    const int coefficient = lhs ? term.a[block] : term.b[block];
    const int64_t row =
        cell / half_cols + (block / Blocks) * int64_t{half_rows};
    const int64_t col =
        cell % half_cols + (block % Blocks) * int64_t{half_cols};
    if (coefficient && row < rows && col < cols)
      value += coefficient *
               source[(static_cast<int64_t>(plane) * rows + row) * ld + col];
  }
  // Rank48 stores twice each dyadic coefficient. Sixteen terms with magnitude
  // <=4 and byte inputs bound |sum|<=8192; scaling by inverse2 stays in INT32.
  if constexpr(Blocks==4)value*=((modulus+1)/2);
  output[static_cast<int64_t>(plane) * count + cell] =
      rns8_strassen_center(value, modulus);
}
template <int Blocks>
__global__ void
rns8_bilinear_combine_kernel(const int8_t *product, int8_t *stage, int m, int n,
                             int hm, int hn, int product_index) {
  const int64_t cell =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= static_cast<int64_t>(m) * n)
    return;
  const int64_t row = cell / n, col = cell % n;
  const int quadrant = (row / hm) * Blocks + col / hn, plane = blockIdx.y;
  if constexpr(Blocks==4)if(!(rns8_default_moduli_device[plane]&1))return;
  const int coefficient = rns8_bilinear_term<Blocks>(product_index).c[quadrant];
  // A zero coefficient leaves this staged block unchanged. Avoid product and
  // stage traffic, and avoid division/reduction for these inactive outputs.
  if (!coefficient)
    return;
  const int64_t output = static_cast<int64_t>(plane) * m * n + cell;
  const int64_t input =
      (static_cast<int64_t>(plane) * hm + row % hm) * hn + col % hn;
  // Reduction after every product bounds recombination by two centered bytes.
  const int scaled=Blocks==4?coefficient*((rns8_default_moduli_device[plane]+1)/2):coefficient;
  stage[output] =
      rns8_strassen_center(stage[output] + scaled * product[input],
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
