extern "C" int rns8_hip_split_k_produce(
    const int8_t* a, const int8_t* b, uint8_t* scratch,
    int m, int n, int k, int lda, int ldb, int prefix, int splits) {
  const auto blocks = static_cast<unsigned>((static_cast<int64_t>(m) * n + 255) / 256);
  hipLaunchKernelGGL(rns8_split_k_produce_kernel, dim3(blocks, prefix, splits), dim3(256), 0, nullptr,
                    a, b, scratch, m, n, k, lda, ldb, prefix, splits);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_split_k_merge(
    const uint8_t* input, uint8_t* output, int64_t cells, int prefix, int fragments) {
  const auto blocks = static_cast<unsigned>((cells + 255) / 256);
  hipLaunchKernelGGL(rns8_split_k_merge_kernel, dim3(blocks, prefix, (fragments + 127) / 128), dim3(256), 0, nullptr,
                    input, output, cells, prefix, fragments);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_split_k_store(const uint8_t* input, int8_t* output, int m, int n, int ldc, int prefix) {
  const auto blocks = static_cast<unsigned>((static_cast<int64_t>(m) * n + 255) / 256);
  hipLaunchKernelGGL(rns8_split_k_store_kernel, dim3(blocks, prefix), dim3(256), 0, nullptr,
                    input, output, m, n, ldc);
  return static_cast<int>(hipGetLastError());
}
