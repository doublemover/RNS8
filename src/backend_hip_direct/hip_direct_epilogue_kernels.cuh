// Small/skinny scalar baseline, not a matrix-engine claim. Each output has one
// owner; partial dot residues stay in registers and the final expression stores
// once. There is no intermediate AB matrix or read-modify-write of old C.
__global__ void rns8_gemm_rns_axpby_kernel(
    const int8_t* a, const int8_t* b, const int8_t* d, int8_t* c,
    int m, int n, int k, int lda, int ldb, int ldd, int ldc, int64_t alpha, int64_t beta) {
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= static_cast<int64_t>(m) * n) return;
  const int plane = blockIdx.y;
  const int64_t row = cell / n, col = cell % n;
  const uint32_t modulus = rns8_default_moduli_device[plane];
  const int8_t product = rns8::detail::centered_residue_dot(
      a + static_cast<int64_t>(plane) * m * lda + row * lda,
      b + static_cast<int64_t>(plane) * k * ldb + col, k, ldb, modulus);
  c[static_cast<int64_t>(plane) * m * ldc + row * ldc + col] = rns8::detail::centered_residue_axpby(
      product, d[static_cast<int64_t>(plane) * m * ldd + row * ldd + col], alpha, beta, modulus);
}

__global__ void rns8_axpby_residues_kernel(
    const int8_t* x, const int8_t* d, int8_t* c,
    int m, int n, int ldx, int ldd, int ldc, int64_t alpha, int64_t beta) {
  const int64_t cell = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (cell >= int64_t{m} * n) return;
  const int64_t row = cell / n, col = cell % n, plane = blockIdx.y;
  c[(plane * m + row) * ldc + col] = rns8::detail::centered_residue_axpby(
      x[(plane * m + row) * ldx + col], d[(plane * m + row) * ldd + col],
      alpha, beta, rns8_default_moduli_device[plane]);
}
