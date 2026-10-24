extern "C" int rns8_hip_complex_sum(const int8_t *real, const int8_t *imag, int8_t *sum, int rows,
                                    int cols, int ldr, int ldi, int prefix,
                                    uint32_t finite_modulus) {
  hipLaunchKernelGGL(rns8_complex_sum_kernel, dim3((uint64_t(rows) * cols + 255) / 256, prefix),
                     dim3(256), 0, nullptr, real, imag, sum, rows, cols, ldr, ldi, finite_modulus);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_complex_combine(const int8_t *products, int8_t *real, int8_t *imag,
                                        int rows, int cols, int prefix, uint32_t finite_modulus,
                                        bool three_real) {
  hipLaunchKernelGGL(rns8_complex_combine_kernel, dim3((uint64_t(rows) * cols + 255) / 256, prefix),
                     dim3(256), 0, nullptr, products, real, imag, rows, cols, prefix,
                     finite_modulus, three_real);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_complex_commit(const int8_t *real, const int8_t *imag, int8_t *output_real,
                                       int8_t *output_imag, int rows, int cols, int ldr, int ldi,
                                       int prefix) {
  hipLaunchKernelGGL(rns8_complex_commit_kernel, dim3((uint64_t(rows) * cols + 255) / 256, prefix),
                     dim3(256), 0, nullptr, real, imag, output_real, output_imag, rows, cols, ldr,
                     ldi);
  return static_cast<int>(hipGetLastError());
}
