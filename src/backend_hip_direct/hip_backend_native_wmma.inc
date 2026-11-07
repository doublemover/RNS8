#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_wmma_capability(int*);
extern "C" int rns8_hip_native_wmma_grouped(const void*, const void*, int8_t*, int, int, int, int,
                                            int, int, int, int, bool);
#endif

rns8_status hip_direct_query_native_wmma_capability(int device_id,
                                                    NativeWmmaDeviceCapability& out) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8_device_info device{};
  auto status = hip_direct_probe(device_id, device);
  if (status != RNS8_SUCCESS)
    return status;
  if (!device.gcn_arch[0])
    return RNS8_BACKEND_FAILURE;
  status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  hip_direct_device_buffer flag;
  status = flag.allocate(device_id, sizeof(uint32_t));
  if (status != RNS8_SUCCESS)
    return status;
  flag.mark_pending();
  status = hip_direct_zero(device_id, flag.get(), sizeof(uint32_t));
  if (status != RNS8_SUCCESS)
    return status;
  const auto error = timed_hip_operation("native_wmma_prepare_capability", [&] {
    const auto launch =
        static_cast<hipError_t>(rns8_hip_native_wmma_capability(static_cast<int*>(flag.get())));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  if (error != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  uint32_t supported = 0;
  status = hip_direct_copy_device_to_host(device_id, &supported, flag.get(), sizeof(supported));
  if (status != RNS8_SUCCESS)
    return status;
  flag.mark_completed();
  if (supported > 1)
    return RNS8_BACKEND_FAILURE;
  NativeWmmaDeviceCapability candidate;
  candidate.device_id = device_id;
  candidate.supported = supported != 0;
  std::copy_n(device.gcn_arch, candidate.target.size(), candidate.target.begin());
  candidate.provider = &hip_direct_query_native_wmma_capability;
  out = candidate;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)out;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_native_wmma_grouped_device(int device_id, const void* a, const void* b,
                                                       int8_t* c, int64_t m, int64_t n, int64_t k,
                                                       int64_t lda, int64_t ldb, int64_t ldc,
                                                       uint32_t prefix, uint32_t group,
                                                       bool unsigned_inputs) {
  if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || !prefix ||
      prefix > RNS8_MAX_SUPPORTED_PREFIX || (group != 1 && group != 2 && group != 4))
    return RNS8_INVALID_ARGUMENT;
  if (!qualified_i32_matrix_address(m, k, lda, 1, 8) ||
      !qualified_i32_matrix_address(k, n, ldb, 1, 8) ||
      !qualified_i32_matrix_address(m, n, ldc, prefix))
    return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t bytes = static_cast<uint64_t>(m) * ldc * prefix;
  if (!nonoverlapping_byte_ranges(c, bytes, a, static_cast<uint64_t>(m) * lda * 8) ||
      !nonoverlapping_byte_ranges(c, bytes, b, static_cast<uint64_t>(k) * ldb * 8))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  hip_direct_device_buffer capability;
  status = capability.allocate(device_id, sizeof(int));
  if (status != RNS8_SUCCESS)
    return status;
  capability.mark_pending();
  const auto probe = timed_hip_operation("native_wmma_device_capability", [&]() {
    auto error = static_cast<hipError_t>(
        rns8_hip_native_wmma_capability(static_cast<int*>(capability.get())));
    return error == hipSuccess ? hipDeviceSynchronize() : error;
  });
  if (probe != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  int supported = 0;
  status =
      hip_direct_copy_device_to_host(device_id, &supported, capability.get(), sizeof(supported));
  if (status != RNS8_SUCCESS)
    return status;
  capability.mark_completed();
  if (!supported)
    return RNS8_UNSUPPORTED_BACKEND;
  const auto error = timed_hip_operation("native_wmma_grouped_conversion_gemm", [&]() {
    auto launched = static_cast<hipError_t>(rns8_hip_native_wmma_grouped(
        a, b, c, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        static_cast<int>(lda), static_cast<int>(ldb), static_cast<int>(ldc), prefix, group,
        unsigned_inputs));
    return launched == hipSuccess ? hipDeviceSynchronize() : launched;
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)unsigned_inputs;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

#include "hip_backend_encoded_b.inc"
