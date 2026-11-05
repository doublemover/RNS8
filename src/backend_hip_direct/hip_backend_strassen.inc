#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_strassen_transform(const int8_t*, int8_t*, int, int, int, int, int, int,
                                           int, bool, int, IndexedModulusPlanes, void*);
extern "C" int rns8_hip_strassen_combine(const int8_t*, int8_t*, int, int, int, int, int, int, int,
                                         IndexedModulusPlanes, void*);
extern "C" int rns8_hip_strassen_commit(const int8_t*, int8_t*, int, int, int, int, void*);
// One terminal observation also covers exceptions from diagnostic hooks. Local
// scratch remains pending if completion cannot be established; the shared drain
// queue owns its eventual reclamation. Borrowed arenas stay under the API lease.
class BilinearCompletion {
public:
  BilinearCompletion(hipStream_t stream, hip_direct_device_buffer& local) noexcept
      : stream_(stream), local_(local) {
    local_.mark_pending();
  }
  ~BilinearCompletion() {
    if (active_)
      (void)finish(RNS8_INTERNAL_ERROR);
  }
  rns8_status finish(rns8_status operation) noexcept {
    if (!active_)
      return operation;
    active_ = false;
    const auto completion = hipStreamSynchronize(stream_);
    if (completion == hipSuccess)
      local_.mark_completed();
    return operation != RNS8_SUCCESS  ? operation
           : completion == hipSuccess ? RNS8_SUCCESS
                                      : RNS8_BACKEND_FAILURE;
  }

private:
  hipStream_t stream_;
  hip_direct_device_buffer& local_;
  bool active_ = true;
};
#endif
static rns8_status indexed_bilinear_products(int device, const int8_t* a, const int8_t* b,
                                             int8_t* c, int64_t m, int64_t n, int64_t k,
                                             const IndexedModulusPlanes& planes, uint32_t batches,
                                             bool matrix, void* stream) {
  if (matrix)
    return amdgpu_builtins_gemm_indexed_batch_device(device, a, b, c, m, n, k, k, n, n, planes,
                                                     batches, stream);
  for (uint32_t batch = 0; batch < batches; ++batch) {
    const auto status =
        hip_direct_enqueue_basis_gemm(device, a + uint64_t(batch) * planes.eligible_count * m * k,
                                      b + uint64_t(batch) * planes.eligible_count * k * n,
                                      c + uint64_t(batch) * planes.eligible_count * m * n, m, n, k,
                                      planes.moduli, planes.eligible_count, stream);
    if (status != RNS8_SUCCESS)
      return status;
  }
  return RNS8_SUCCESS;
}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
static rns8_status bilinear_products(int device, const int8_t* a, const int8_t* b, int8_t* c,
                                     int64_t m, int64_t n, int64_t k,
                                     const IndexedModulusPlanes& planes, uint32_t batches,
                                     bool matrix, bool indexed, hipStream_t stream) {
  if (matrix)
    return indexed ? amdgpu_builtins_gemm_indexed_batch_device(device, a, b, c, m, n, k, k, n, n,
                                                               planes, batches, stream)
                   : amdgpu_builtins_gemm_rns_batch_device(device, a, b, c, m, n, k, k, n, n,
                                                           planes.logical_count, batches, stream);
  rns8_status status = RNS8_SUCCESS;
  const auto error = timed_hip_operation(
      "bilinear_ordered_products_v3",
      [&]() {
        if (indexed)
          status =
              indexed_bilinear_products(device, a, b, c, m, n, k, planes, batches, matrix, stream);
        else
          for (uint32_t batch = 0; batch < batches; ++batch) {
            const auto launch =
                launch_direct_rns_gemm_no_sync(a + uint64_t(batch) * planes.logical_count * m * k,
                                               b + uint64_t(batch) * planes.logical_count * k * n,
                                               c + uint64_t(batch) * planes.logical_count * m * n,
                                               m, n, k, k, n, n, planes.logical_count, stream);
            if (launch != hipSuccess)
              return launch;
          }
        return status == RNS8_SUCCESS ? hipSuccess : hipErrorUnknown;
      },
      stream);
  return status != RNS8_SUCCESS ? status
         : error == hipSuccess  ? RNS8_SUCCESS
                                : RNS8_BACKEND_FAILURE;
}
static rns8_status bilinear_classical_fallback(const void* a, const void* b, int8_t* stage,
                                               int64_t m, int64_t n, int64_t k, int64_t lda,
                                               int64_t ldb, const IndexedModulusPlanes& planes,
                                               hipStream_t stream) {
  const auto error = timed_hip_operation(
      "bilinear_ordered_classical_fallback_v3",
      [&]() {
        for (uint32_t index = 0; index < planes.fallback_count; ++index) {
          const auto plane = planes.fallback[index];
          const auto q = kDefaultModuli[plane];
          const auto launch =
              static_cast<hipError_t>(rns8_hip_direct_finite_ring_gemm_i8_device_on_stream(
                  static_cast<const int8_t*>(a) + uint64_t(plane) * m * lda,
                  static_cast<const int8_t*>(b) + uint64_t(plane) * k * ldb,
                  stage + uint64_t(plane) * m * n, int(m), int(n), int(k), int(lda), int(ldb),
                  int(n), q, modulus_reciprocal_u32(q), RNS8_SAFE_INT32_K_BLOCK, stream));
          if (launch != hipSuccess)
            return launch;
        }
        return hipSuccess;
      },
      stream);
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}
#endif
static rns8_status hip_direct_gemm_bilinear_device(int device_id, const void* a, const void* b,
                                                   void* c, int64_t m, int64_t n, int64_t k,
                                                   int64_t lda, int64_t ldb, int64_t ldc,
                                                   uint32_t prefix, uint64_t max_scratch_bytes,
                                                   bool& commit_started, uint32_t blocks,
                                                   void* retained_scratch, uint64_t retained_bytes,
                                                   bool matrix_subproducts, void* stream) {
  const bool rank23 = blocks == 3, rank48 = blocks == 4;
  commit_started = false;
  if (!a || !b || !c)
    return RNS8_INVALID_ARGUMENT;
  const auto resources =
      query_bilinear_resources(m, n, k, lda, ldb, ldc, prefix, blocks, max_scratch_bytes);
  if (resources.status != RNS8_SUCCESS)
    return resources.status;
  if (!nonoverlapping_byte_ranges(c, resources.destination.byte_extent, a,
                                  resources.source_a.byte_extent) ||
      !nonoverlapping_byte_ranges(c, resources.destination.byte_extent, b,
                                  resources.source_b.byte_extent))
    return RNS8_INVALID_ARGUMENT;
  const auto hm = resources.block_m, hn = resources.block_n, hk = resources.block_k;
  const auto cb = resources.output_bytes;
  if (matrix_subproducts && (hk > RNS8_SAFE_INT32_K_BLOCK || !amdgpu_builtins_compiled()))
    return RNS8_UNSUPPORTED_BACKEND;
  const auto arena_status =
      validate_bilinear_arena(resources, retained_scratch, retained_bytes, a, b, c);
  if (arena_status != RNS8_SUCCESS)
    return arena_status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  hip_direct_device_buffer scratch;
  if (retained_scratch && retained_bytes < resources.total_bytes)
    return RNS8_WORKSPACE_TOO_SMALL;
  if (!retained_scratch) {
    status = scratch.allocate(device_id, resources.total_bytes);
    if (status != RNS8_SUCCESS)
      return status;
  }
  auto* ta = static_cast<int8_t*>(retained_scratch ? retained_scratch : scratch.get());
  auto* tb = ta + resources.b_offset;
  auto* product = ta + resources.product_offset;
  auto* stage = ta + resources.output_offset;
  hip_unique_stream local_stream;
  if (!stream) {
    if (local_stream.create_non_blocking() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    stream = local_stream.get();
  }
  const auto ordered = static_cast<hipStream_t>(stream);
  BilinearCompletion completion(ordered, scratch);
  return completion.finish([&]() -> rns8_status {
    if (hipMemsetAsync(stage, 0, size_t(cb), ordered) != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    for (int term = 0; term < (rank48 ? 48 : rank23 ? 23 : 7); ++term) {
      auto error = timed_hip_operation(
          rank48   ? "rank48_dyadic_transform_pair"
          : rank23 ? "rank23_transform_pair"
                   : "strassen_transform_pair",
          [&]() {
            auto launch = static_cast<hipError_t>(rns8_hip_strassen_transform(
                static_cast<const int8_t*>(a), ta, static_cast<int>(m), static_cast<int>(k),
                static_cast<int>(lda), static_cast<int>(hm), static_cast<int>(hk), prefix, term,
                true, int(blocks), resources.planes, stream));
            if (launch != hipSuccess)
              return launch;
            launch = static_cast<hipError_t>(rns8_hip_strassen_transform(
                static_cast<const int8_t*>(b), tb, static_cast<int>(k), static_cast<int>(n),
                static_cast<int>(ldb), static_cast<int>(hk), static_cast<int>(hn), prefix, term,
                false, int(blocks), resources.planes, stream));
            return launch;
          },
          ordered);
      if (error != hipSuccess)
        return RNS8_BACKEND_FAILURE;
      status = bilinear_products(device_id, ta, tb, product, hm, hn, hk, resources.planes, 1,
                                 matrix_subproducts, rank48, ordered);
      if (status != RNS8_SUCCESS)
        return status;
      error = timed_hip_operation(
          rank48   ? "rank48_dyadic_recombine"
          : rank23 ? "rank23_recombine"
                   : "strassen_recombine",
          [&]() {
            const auto launch = static_cast<hipError_t>(rns8_hip_strassen_combine(
                product, stage, static_cast<int>(m), static_cast<int>(n), static_cast<int>(hm),
                static_cast<int>(hn), prefix, term, int(blocks), resources.planes, stream));
            return launch;
          },
          ordered);
      if (error != hipSuccess)
        return RNS8_BACKEND_FAILURE;
    }
    // Inverse2 is not defined in even channels. Execute those original planes
    // classically into the same private output stage before publishing anything.
    if (rank48) {
      status =
          bilinear_classical_fallback(a, b, stage, m, n, k, lda, ldb, resources.planes, ordered);
      if (status != RNS8_SUCCESS)
        return status;
    }
    commit_started = true;
    const auto error = timed_hip_operation(
        rank48   ? "rank48_mixed_basis_commit"
        : rank23 ? "rank23_commit"
                 : "strassen_commit",
        [&]() {
          const auto launch = static_cast<hipError_t>(
              rns8_hip_strassen_commit(stage, static_cast<int8_t*>(c), static_cast<int>(m),
                                       static_cast<int>(n), static_cast<int>(ldc), prefix, stream));
          return launch;
        },
        ordered);
    return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
  }());
#else
  (void)device_id;
  (void)rank23;
  (void)rank48;
  (void)stream;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_gemm_strassen_device(int device_id, const void* a, const void* b, void* c,
                                            int64_t m, int64_t n, int64_t k, int64_t lda,
                                            int64_t ldb, int64_t ldc, uint32_t prefix,
                                            uint64_t max_scratch_bytes, bool& commit_started,
                                            void* retained_scratch, uint64_t retained_bytes,
                                            bool matrix_subproducts, void* stream) {
  return hip_direct_gemm_bilinear_device(device_id, a, b, c, m, n, k, lda, ldb, ldc, prefix,
                                         max_scratch_bytes, commit_started, 2, retained_scratch,
                                         retained_bytes, matrix_subproducts, stream);
}
rns8_status hip_direct_gemm_rank23_device(int device_id, const void* a, const void* b, void* c,
                                          int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb,
                                          int64_t ldc, uint32_t prefix, uint64_t max_scratch_bytes,
                                          bool& commit_started, void* retained_scratch,
                                          uint64_t retained_bytes, bool matrix_subproducts,
                                          void* stream) {
  return hip_direct_gemm_bilinear_device(device_id, a, b, c, m, n, k, lda, ldb, ldc, prefix,
                                         max_scratch_bytes, commit_started, 3, retained_scratch,
                                         retained_bytes, matrix_subproducts, stream);
}
rns8_status hip_direct_gemm_rank48_device(int device_id, const void* a, const void* b, void* c,
                                          int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb,
                                          int64_t ldc, uint32_t prefix, uint64_t budget,
                                          bool& commit_started, void* scratch, uint64_t bytes,
                                          bool matrix_subproducts, void* stream) {
  return hip_direct_gemm_bilinear_device(device_id, a, b, c, m, n, k, lda, ldb, ldc, prefix, budget,
                                         commit_started, 4, scratch, bytes, matrix_subproducts,
                                         stream);
}
