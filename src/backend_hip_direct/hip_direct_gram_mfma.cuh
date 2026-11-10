#include "backend_common/mfma_i8_layout.hpp"

#if defined(__gfx942__) && __has_builtin(__builtin_amdgcn_mfma_i32_16x16x32_i8)
#define RNS8_GRAM_MFMA_GFX942_BODY 1
#else
#define RNS8_GRAM_MFMA_GFX942_BODY 0
#endif

__global__ void rns8_gram_matrix_capability_kernel(int* result, int kind) {
  int supported = 0;
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) &&                      \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
  if (kind == 1)
    supported = 1;
#endif
#if RNS8_GRAM_MFMA_GFX942_BODY
  if (kind == 2)
    supported = 1;
#endif
  *result = supported;
}
extern "C" int rns8_hip_gram_matrix_capability(int* result, int kind) {
  if (!result || kind < 1 || kind > 2)
    return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(rns8_gram_matrix_capability_kernel, dim3(1), dim3(1), 0, nullptr, result,
                     kind);
  return int(hipGetLastError());
}

template <bool Native>
__global__ __launch_bounds__(64) void rns8_gram_mfma_gfx942_kernel(
    const void* source, int8_t* output, const rns8::detail::GramTile* jobs, int rows, int cols,
    int ld, int ldc, int finite_modulus, bool transposed, bool separate_mirror) {
#if RNS8_GRAM_MFMA_GFX942_BODY
  using Layout = rns8::detail::MfmaI8Layout<32>;
  using Accumulator = int __attribute__((ext_vector_type(4)));
  const unsigned lane = threadIdx.x;
  const auto job = jobs[blockIdx.x];
  const int64_t order = transposed ? cols : rows, k = transposed ? rows : cols, plane = blockIdx.y,
                row_base = int64_t(job.row) * 16, col_base = int64_t(job.col) * 16;
  const int modulus = finite_modulus ? finite_modulus : rns8_default_moduli_device[plane];
  const auto reducer = rns8::detail::finite_u8::native64_reducer(uint16_t(modulus));
  const auto load = [&](int64_t row, int64_t q) -> int8_t {
    if (row >= order || q >= k)
      return 0;
    if constexpr (Native) {
      const auto value =
          static_cast<const int64_t*>(source)[transposed ? q * ld + row : row * ld + q];
      return int8_t(rns8_native_wmma_center(value, reducer));
    } else {
      return rns8_gram_load(static_cast<const int8_t*>(source), plane, row, q, rows, cols, ld,
                            transposed);
    }
  };
  Accumulator acc{};
  for (int64_t base = 0; base < k; base += Layout::k) {
    uint64_t a = 0, b = 0;
    for (unsigned byte = 0; byte < Layout::input_bytes_per_lane; ++byte) {
      const auto ac = Layout::a(lane, byte), bc = Layout::b(lane, byte);
      const auto av = load(row_base + ac.row, base + ac.col);
      // In Gram, the two physical MFMA roles map to the same raw coordinates
      // on a diagonal job. Reuse the loaded/reduced byte there.
      const auto bv = job.row == job.col ? av : load(col_base + bc.col, base + bc.row);
      a |= uint64_t(uint8_t(av)) << (8 * byte);
      b |= uint64_t(uint8_t(bv)) << (8 * byte);
    }
    acc = __builtin_amdgcn_mfma_i32_16x16x32_i8(__builtin_bit_cast(long long, a),
                                                __builtin_bit_cast(long long, b), acc, 0, 0, 0);
    // One instruction contributes at most32*128^2; reduced carry is <q.
    // Reducing each slab bounds every INT32 partial independently of final CRT.
    for (unsigned reg = 0; reg < Layout::accumulators_per_lane; ++reg)
      acc[reg] %= modulus;
  }
  for (unsigned reg = 0; reg < Layout::accumulators_per_lane; ++reg) {
    const auto cell = Layout::c(lane, reg);
    const int64_t row = row_base + cell.row, col = col_base + cell.col;
    if (row >= order || col >= order || row > col)
      continue;
    int value = acc[reg];
    if (value < 0)
      value += modulus;
    if (value >= (modulus + 1) / 2)
      value -= modulus;
    output[(plane * order + row) * ldc + col] = int8_t(value);
    if (row != col && !separate_mirror)
      output[(plane * order + col) * ldc + row] = int8_t(value);
  }
#else
  (void)source;
  (void)output;
  (void)jobs;
  (void)rows;
  (void)cols;
  (void)ld;
  (void)ldc;
  (void)finite_modulus;
  (void)transposed;
  (void)separate_mirror;
#endif
}
