__global__ void rns8_export_canonical_balanced_crt_kernel(
    const int8_t *residues, uint64_t *limbs, int cells,
    rns8::detail::balanced_crt::Plan plan) {
  const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cell >= cells)
    return;
  using namespace rns8::detail::balanced_crt;
  Wide values[20];
  for (uint32_t p = 0; p < plan.prefix; ++p)
    values[p].limb[0] = rns8_canonical_from_centered_device(
        residues[static_cast<int64_t>(p) * cells + cell],
        rns8_default_moduli_device[p]);
  for (uint32_t i = 0; i < plan.count; ++i) {
    const auto &node = plan.merges[i];
    values[node.left] = merge(values[node.left], values[node.right], node);
  }
  for (int i = 0; i < 3; ++i)
    limbs[static_cast<int64_t>(cell) * 3 + i] = values[0].limb[i];
}
