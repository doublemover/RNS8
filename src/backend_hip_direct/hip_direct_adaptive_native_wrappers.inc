template <typename T, int Group>
int rns8_launch_adaptive_native(const void* a, const void* b, int8_t* c,
    const rns8::detail::NativeTileBound* bounds, const uint32_t* indices,
    int m, int n, int k, int lda, int ldb, int capacity, const uint32_t* offsets, uint32_t workers) {
  if (workers) {
    hipLaunchKernelGGL((rns8_adaptive_native_compact_kernel<T, Group>),
        dim3(workers), dim3(32), 0, nullptr, static_cast<const T*>(a),
        static_cast<const T*>(b), c, indices, offsets, m, n, k, lda, ldb, capacity);
    return static_cast<int>(hipGetLastError());
  }
  const uint64_t tiles = ((static_cast<uint64_t>(m) + 15) / 16) *
                         ((static_cast<uint64_t>(n) + 15) / 16);
  hipLaunchKernelGGL((rns8_adaptive_native_kernel<T, Group>),
      dim3(tiles, (capacity + Group - 1) / Group), dim3(32), 0, nullptr,
      static_cast<const T*>(a), static_cast<const T*>(b), c, bounds, indices, m, n, k, lda, ldb);
  return static_cast<int>(hipGetLastError());
}
template <typename T>
int rns8_select_adaptive_native(const void* a, const void* b, int8_t* c,
    const rns8::detail::NativeTileBound* bounds, const uint32_t* indices,
    int m, int n, int k, int lda, int ldb, int capacity, int group, const uint32_t* offsets, uint32_t workers) {
  if (group == 1) return rns8_launch_adaptive_native<T, 1>(a,b,c,bounds,indices,m,n,k,lda,ldb,capacity,offsets,workers);
  if (group == 2) return rns8_launch_adaptive_native<T, 2>(a,b,c,bounds,indices,m,n,k,lda,ldb,capacity,offsets,workers);
  if (group == 4) return rns8_launch_adaptive_native<T, 4>(a,b,c,bounds,indices,m,n,k,lda,ldb,capacity,offsets,workers);
  return static_cast<int>(hipErrorInvalidValue);
}
extern "C" int rns8_hip_adaptive_native(const void* a, const void* b, int8_t* c,
    const rns8::detail::NativeTileBound* bounds, const uint32_t* indices,
    int m, int n, int k, int lda, int ldb, int capacity, int group, bool unsigned_inputs,
    const uint32_t* offsets, uint32_t workers) {
  return unsigned_inputs
      ? rns8_select_adaptive_native<uint64_t>(a,b,c,bounds,indices,m,n,k,lda,ldb,capacity,group,offsets,workers)
      : rns8_select_adaptive_native<int64_t>(a,b,c,bounds,indices,m,n,k,lda,ldb,capacity,group,offsets,workers);
}
extern "C" int rns8_hip_adaptive_native_suffix(int8_t* c,
    const rns8::detail::NativeTileBound* bounds, int m, int n, int capacity,
    bool signed_value, int* status) {
  hipLaunchKernelGGL(rns8_adaptive_native_suffix_kernel,
      dim3((static_cast<uint64_t>(m) * n + 255) / 256), dim3(256), 0, nullptr,
      c, bounds, m, n, capacity, signed_value, status);
  return static_cast<int>(hipGetLastError());
}
