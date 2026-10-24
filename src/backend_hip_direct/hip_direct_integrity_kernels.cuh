__device__ uint32_t rns8_integrity_native_mod(int64_t value, uint32_t q) {
  const int64_t r = value % static_cast<int64_t>(q);
  return static_cast<uint32_t>(r < 0 ? r + q : r);
}
__device__ uint32_t rns8_integrity_native_mod(uint64_t value, uint32_t q) {
  return static_cast<uint32_t>(value % q);
}
template <typename T>
__global__ void
rns8_integrity_check_kernel(const T *a, const T *b, const int8_t *candidate,
                            int m, int n, int k, int lda, int ldb, int prefix,
                            bool signed_value, rns8_u192_device bound,
                            uint32_t *failures) {
  const int64_t cell =
                    static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                cells = static_cast<int64_t>(m) * n;
  if (cell >= cells)
    return;
  for (int p = 0; p < prefix; ++p) {
    const int stored = candidate[static_cast<int64_t>(p) * cells + cell],
              modulus = rns8_default_moduli_device[p];
    if (stored < -(modulus / 2) || stored > (modulus - 1) / 2) {
      atomicOr(failures, 1u);
      return;
    }
  }
  rns8_u192_device value{}, product{};
  rns8_reconstruct_canonical_wide_device(candidate, cell, cells, prefix, &value,
                                         &product);
  const bool negative =
      signed_value && rns8_u192_centered_is_negative_device(value, product);
  const auto magnitude =
      negative ? rns8_u192_sub_device(product, value) : value;
  if (rns8_u192_compare_device(magnitude, bound) > 0) {
    atomicOr(failures, 2u);
    return;
  }
  const uint32_t primes[2] = {257, 263};
  uint32_t checks[2]{};
  for (int inner = 0; inner < k; ++inner) {
    const auto av = a[(cell / n) * lda + inner],
               bv = b[static_cast<int64_t>(inner) * ldb + cell % n];
    for (int p = 0; p < 2; ++p) {
      const auto q = primes[p];
      // Each independent native product plus carried check is <=262^2+262.
      checks[p] = (checks[p] + rns8_integrity_native_mod(av, q) *
                                   rns8_integrity_native_mod(bv, q)) %
                  q;
    }
  }
  for (int p = 0; p < 2; ++p) {
    auto residue = rns8_u192_mod_u32_device(magnitude, primes[p]);
    if (negative && residue)
      residue = primes[p] - residue;
    if (residue != checks[p])
      atomicOr(failures, 4u << p);
  }
}
