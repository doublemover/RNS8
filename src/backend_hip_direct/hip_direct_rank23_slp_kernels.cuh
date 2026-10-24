__device__ __constant__ rns8::detail::rank23::slp::Plan rns8_rank23_slp_u =
    rns8::detail::rank23::slp::u;
__device__ __constant__ rns8::detail::rank23::slp::Plan rns8_rank23_slp_v =
    rns8::detail::rank23::slp::v;
__device__ __constant__ rns8::detail::rank23::slp::Plan rns8_rank23_slp_w =
    rns8::detail::rank23::slp::w;

__device__ void rns8_rank23_slp_evaluate(
    const rns8::detail::rank23::slp::Plan &plan, int8_t *slots, int modulus) {
  for (int i = 0; i < plan.count; ++i) {
    const auto op = plan.steps[i];
    const int left = slots[op.left], right = slots[op.right];
    slots[op.target] = rns8_strassen_center(left + (op.subtract ? -right : right), modulus);
  }
}
template<bool Lhs>
__global__ void rns8_rank23_slp_transform_kernel(
    const int8_t *source, int8_t *output, int rows, int cols, int ld,
    int block_rows, int block_cols, int prefix) {
  const int64_t cell = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t cells = int64_t(block_rows)*block_cols;
  if (cell >= cells) return;
  const int plane = blockIdx.y, modulus = rns8_default_moduli_device[plane];
  const auto &plan = Lhs ? rns8_rank23_slp_u : rns8_rank23_slp_v;
  int8_t slots[rns8::detail::rank23::slp::capacity]{};
  for (int block = 0; block < 9; ++block) {
    const int64_t row = cell/block_cols + int64_t(block/3)*block_rows;
    const int64_t col = cell%block_cols + int64_t(block%3)*block_cols;
    if (row < rows && col < cols)
      slots[block] = rns8_strassen_center(source[(int64_t(plane)*rows+row)*ld+col],modulus);
  }
  rns8_rank23_slp_evaluate(plan,slots,modulus);
  for (int product = 0; product < 23; ++product)
    output[(int64_t(product)*prefix+plane)*cells+cell] = slots[plan.results[product]];
}
__global__ void rns8_rank23_slp_combine_kernel(
    const int8_t *products, int8_t *stage, int m, int n, int hm, int hn, int prefix) {
  const int64_t cell = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t cells = int64_t(hm)*hn;
  if (cell >= cells) return;
  const int plane = blockIdx.y, modulus = rns8_default_moduli_device[plane];
  int8_t slots[rns8::detail::rank23::slp::capacity]{};
  for (int product = 0; product < 23; ++product)
    slots[product] = products[(int64_t(product)*prefix+plane)*cells+cell];
  rns8_rank23_slp_evaluate(rns8_rank23_slp_w,slots,modulus);
  for (int block = 0; block < 9; ++block) {
    const int64_t row = cell/hn+int64_t(block/3)*hm;
    const int64_t col = cell%hn+int64_t(block%3)*hn;
    if (row < m && col < n)
      stage[(int64_t(plane)*m+row)*n+col] = slots[rns8_rank23_slp_w.results[block]];
  }
}
