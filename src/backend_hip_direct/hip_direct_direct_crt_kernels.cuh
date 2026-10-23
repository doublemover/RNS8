__device__ __constant__ rns8::detail::direct_crt::Constants rns8_direct_crt_table = rns8::detail::direct_crt::make_constants();

__global__ void rns8_export_canonical_direct_crt_kernel(const int8_t* residues, uint64_t* limbs, int cells, int prefix) {
  const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cell >= cells) return;
  const auto* product_limbs = rns8_direct_crt_table.products[prefix];
  const rns8_u192_device product{product_limbs[0], product_limbs[1], product_limbs[2]};
  rns8_u192_device sum{};
  for (int p = 0; p < prefix; ++p) {
    const uint32_t modulus = rns8_default_moduli_device[p];
    const int8_t stored = residues[static_cast<int64_t>(p) * cells + cell];
    const uint32_t residue = rns8_canonical_from_centered_device(stored, modulus);
    const uint32_t coefficient = residue * rns8_direct_crt_table.inverses[prefix][p] % modulus;
    const auto* weight = rns8_direct_crt_table.weights[prefix][p];
    sum = rns8_u192_add_device(sum, rns8_u192_mul_u32_device({weight[0], weight[1], weight[2]}, coefficient));
  }
  // Each term is below P, so at most prefix-1 subtractions are needed.
  while (rns8_u192_compare_device(sum, product) >= 0) sum = rns8_u192_sub_device(sum, product);
  limbs[static_cast<int64_t>(cell) * 3] = sum.limb0;
  limbs[static_cast<int64_t>(cell) * 3 + 1] = sum.limb1;
  limbs[static_cast<int64_t>(cell) * 3 + 2] = sum.limb2;
}
