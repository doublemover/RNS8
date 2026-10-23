#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_integrity_check(const void *, const void *,
                                        const int8_t *, int, int, int, int, int,
                                        int, bool, const uint64_t *,
                                        uint32_t *);
#endif
rns8_status hip_direct_check_native_product_integrity(
    int device_id, const void *a, const void *b, const void *candidate,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb,
    bool unsigned_inputs, const uint64_t bound[3],
    const IntegrityIdentity &expected,
    const IntegrityIdentity &candidate_identity, uint32_t &failures) {
  failures = 0;
  if (!a || !b || !candidate || !bound ||
      !integrity_identity_matches(expected, candidate_identity) || m <= 0 ||
      n <= 0 || k <= 0 || lda < k || ldb < n || bound[2] >> 32)
    return RNS8_INVALID_ARGUMENT;
  const auto prefix = expected.prefix;
  if (!qualified_i32_matrix_address(m, k, lda, 1, 8) ||
      !qualified_i32_matrix_address(k, n, ldb, 1, 8) ||
      !qualified_i32_matrix_address(m, n, n, prefix))
    return RNS8_UNSUPPORTED_BACKEND;
  NativeTileBound checked{{bound[0], bound[1], bound[2]}, 0, 0};
  select_native_bound_prefix(checked, !unsigned_inputs, prefix,
                             garner::kConstants.products);
  if (checked.required_prefix > prefix)
    return RNS8_RANGE_ERROR;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  hip_direct_device_buffer flag;
  status = flag.allocate(device_id, sizeof(uint32_t));
  if (status != RNS8_SUCCESS)
    return status;
  status = hip_direct_zero(device_id, flag.get(), sizeof(uint32_t));
  if (status != RNS8_SUCCESS)
    return status;
  const auto error =
      timed_hip_operation("independent_native_product_integrity", [&]() {
        const auto launch = static_cast<hipError_t>(rns8_hip_integrity_check(
            a, b, static_cast<const int8_t *>(candidate), static_cast<int>(m),
            static_cast<int>(n), static_cast<int>(k), static_cast<int>(lda),
            static_cast<int>(ldb), prefix, unsigned_inputs, bound,
            static_cast<uint32_t *>(flag.get())));
        return launch == hipSuccess ? hipDeviceSynchronize() : launch;
      });
  if (error != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  status = hip_direct_copy_device_to_host(device_id, &failures, flag.get(),
                                          sizeof(failures));
  if (status != RNS8_SUCCESS)
    return status;
  return failures ? RNS8_VERIFICATION_FAILED : RNS8_SUCCESS;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
