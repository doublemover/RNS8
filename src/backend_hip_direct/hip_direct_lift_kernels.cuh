__global__ void rns8_exact_lift_suffix_kernel(
    const int8_t* residues, int8_t* suffix, int elements, int source_prefix,
    int target_prefix, bool signed_value, rns8_u192_device bound, int* status) {
  const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cell >= elements) return;
  for (int p = 0; p < source_prefix; ++p) {
    const int modulus = rns8_default_moduli_device[p];
    const int value = residues[static_cast<int64_t>(p) * elements + cell];
    if (value < -(modulus / 2) || value > (modulus - 1) / 2) {
      atomicMax(status, 1);
      return;
    }
  }
  rns8_u192_device value{}, product{};
  rns8_reconstruct_canonical_wide_device(residues, cell, elements, source_prefix, &value, &product);
  const bool negative = signed_value && rns8_u192_centered_is_negative_device(value, product);
  const auto magnitude = negative ? rns8_u192_sub_device(product, value) : value;
  if (rns8_u192_compare_device(magnitude, bound) > 0) {
    atomicMax(status, 2);
    return;
  }
  for (int p = source_prefix; p < target_prefix; ++p) {
    const uint32_t modulus = rns8_default_moduli_device[p];
    uint32_t residue = rns8_u192_mod_u32_device(magnitude, modulus);
    if (negative && residue) residue = modulus - residue;
    const int centered = residue >= (modulus + 1) / 2 ? static_cast<int>(residue) - static_cast<int>(modulus) : static_cast<int>(residue);
    suffix[static_cast<int64_t>(p - source_prefix) * elements + cell] = static_cast<int8_t>(centered);
  }
}
