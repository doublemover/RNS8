template <typename NativeT, int Group>
__global__ __launch_bounds__(32) void rns8_adaptive_native_kernel(
    const NativeT* a, const NativeT* b, int8_t* c,
    const rns8::detail::NativeTileBound* bounds, const uint32_t* indices,
    int m, int n, int k, int lda, int ldb) {
  const uint32_t tile = indices[blockIdx.x];
  const int prefix = bounds[tile].required_prefix;
  const int first = blockIdx.y * Group;
  if (first >= prefix) return;
  const uint32_t cols = (static_cast<uint64_t>(n) + 15) / 16;
  rns8_native_grouped_wmma_tile<NativeT, Group>(a, b, c, m, n, k, lda, ldb, n,
      prefix, static_cast<int64_t>(tile / cols) * 16,
      static_cast<int64_t>(tile % cols) * 16, first);
}

// Every tile has a strict exact range proof from the current native operands.
// Complete the common basis before exposing it to an ordinary residue consumer.
__global__ void rns8_adaptive_native_suffix_kernel(
    int8_t* c, const rns8::detail::NativeTileBound* bounds,
    int m, int n, int capacity, bool signed_value, int* status) {
  const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int elements = m * n;
  if (cell >= elements) return;
  const uint32_t cols = (static_cast<uint64_t>(n) + 15) / 16;
  const auto proof = bounds[(cell / n / 16) * cols + cell % n / 16];
  const int prefix = proof.required_prefix;
  if (!prefix) return; // The pre-zeroed full tile is exact, including its suffix.
  rns8_u192_device value{}, product{};
  uint32_t digits[20]{};
  rns8_reconstruct_canonical_wide_device(c, cell, elements, prefix, &value, &product, digits);
  const bool negative = signed_value && rns8_u192_centered_is_negative_device(value, product);
  const auto magnitude = negative ? rns8_u192_sub_device(product, value) : value;
  const rns8_u192_device bound{proof.magnitude[0], proof.magnitude[1], proof.magnitude[2]};
  if (rns8_u192_compare_device(magnitude, bound) > 0) {
    atomicMax(status, 1);
    return;
  }
  for (int p = prefix; p < capacity; ++p) {
    const uint32_t modulus = rns8_default_moduli_device[p];
    const uint32_t residue = rns8::detail::mixed_radix_lift_residue(
        digits, rns8_default_moduli_device, prefix, modulus, negative);
    const int centered = residue >= (modulus + 1) / 2
        ? static_cast<int>(residue) - static_cast<int>(modulus) : static_cast<int>(residue);
    c[static_cast<int64_t>(p) * elements + cell] = static_cast<int8_t>(centered);
  }
}
