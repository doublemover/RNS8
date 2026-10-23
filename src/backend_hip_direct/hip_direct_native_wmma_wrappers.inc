extern "C" int rns8_hip_native_wmma_capability(int* supported) {
  hipLaunchKernelGGL(rns8_native_wmma_capability_kernel, dim3(1), dim3(1), 0, nullptr, supported);
  return static_cast<int>(hipGetLastError());
}
template <typename NativeT, int Group>
int rns8_launch_native_wmma_group(
    const void* a, const void* b, int8_t* c, int m, int n, int k,
    int lda, int ldb, int ldc, int prefix) {
  hipLaunchKernelGGL((rns8_native_grouped_wmma_kernel<NativeT, Group>),
      dim3((static_cast<uint64_t>(n) + 15) / 16, (static_cast<uint64_t>(m) + 15) / 16,
           (prefix + Group - 1) / Group), dim3(32), 0, nullptr,
      static_cast<const NativeT*>(a), static_cast<const NativeT*>(b), c, m, n, k, lda, ldb, ldc, prefix);
  return static_cast<int>(hipGetLastError());
}
template <typename NativeT>
int rns8_select_native_wmma_group(
    const void* a, const void* b, int8_t* c, int m, int n, int k,
    int lda, int ldb, int ldc, int prefix, int group) {
  switch (group) {
    case 1: return rns8_launch_native_wmma_group<NativeT, 1>(a, b, c, m, n, k, lda, ldb, ldc, prefix);
    case 2: return rns8_launch_native_wmma_group<NativeT, 2>(a, b, c, m, n, k, lda, ldb, ldc, prefix);
    case 4: return rns8_launch_native_wmma_group<NativeT, 4>(a, b, c, m, n, k, lda, ldb, ldc, prefix);
    default: return static_cast<int>(hipErrorInvalidValue);
  }
}
extern "C" int rns8_hip_native_wmma_grouped(
    const void* a, const void* b, int8_t* c, int m, int n, int k,
    int lda, int ldb, int ldc, int prefix, int group, bool unsigned_inputs) {
  return unsigned_inputs
      ? rns8_select_native_wmma_group<uint64_t>(a, b, c, m, n, k, lda, ldb, ldc, prefix, group)
      : rns8_select_native_wmma_group<int64_t>(a, b, c, m, n, k, lda, ldb, ldc, prefix, group);
}
