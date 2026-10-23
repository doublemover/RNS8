template <typename NativeT>
bool check_native_wmma_groups(bool unsigned_inputs, int m, int n, int k, uint32_t prefix,
                              bool generated = false, uint32_t splits = 0) {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  const int lda = k + 1, ldb = n + 2, ldc = n + 3;
  const NativeT lo = std::numeric_limits<NativeT>::min(), hi = std::numeric_limits<NativeT>::max();
  const NativeT values[] = {lo, hi, NativeT{0}, NativeT{1}, NativeT{127}, NativeT{128}, NativeT{255}};
  std::vector<NativeT> a(m * lda, hi), b(k * ldb, hi);
  for (int row = 0; row < m; ++row)
    for (int q = 0; q < k; ++q) a[row * lda + q] = values[(row * 3 + q) % 7];
  for (int q = 0; q < k; ++q)
    for (int col = 0; col < n; ++col) b[q * ldb + col] = values[(q * 5 + col) % 7];
  // The extra plane catches accidental execution of a partial group's suffix.
  std::vector<int8_t> expected((prefix + 1) * m * ldc, 99), actual(expected.size(), 99);
  constexpr int moduli[] = {256,255,253,251,247,239,233,229,227,223,217,211,199,197,193,191,181,179,173,167};
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < n; ++col) {
      cpp_int sum = 0;
      for (int q = 0; q < k; ++q) sum += cpp_int(a[row * lda + q]) * b[q * ldb + col];
      for (uint32_t p = 0; p < prefix; ++p) {
        int r = (sum % moduli[p]).convert_to<int>();
        if (r < 0) r += moduli[p];
        if (r >= (moduli[p] + 1) / 2) r -= moduli[p];
        expected[(p * m + row) * ldc + col] = static_cast<int8_t>(r);
      }
    }
  void *da = nullptr, *db = nullptr, *dc = nullptr;
  REQUIRE(hip_direct_allocate(0, a.size() * sizeof(NativeT), &da) == RNS8_SUCCESS);
  REQUIRE(hip_direct_allocate(0, b.size() * sizeof(NativeT), &db) == RNS8_SUCCESS);
  REQUIRE(hip_direct_allocate(0, actual.size(), &dc) == RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0, da, a.data(), a.size() * sizeof(NativeT)) == RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0, db, b.data(), b.size() * sizeof(NativeT)) == RNS8_SUCCESS);
  if (generated) {
    // Independent unfused conversion/GEMM route with the identical native
    // semantics, basis and padded output policy.
    hip_direct_device_buffer ra, rb;
    REQUIRE(ra.allocate(0, std::size_t{prefix} * m * k) == RNS8_SUCCESS);
    REQUIRE(rb.allocate(0, std::size_t{prefix} * k * n) == RNS8_SUCCESS);
    void* upload = nullptr;
    std::size_t upload_bytes = 0;
    const auto pack = [&](const NativeT* source, void* destination, int rows, int cols, int ld) {
      if constexpr (std::is_same_v<NativeT, uint64_t>)
        return hip_direct_pack_u64_device(0, source, &upload, &upload_bytes, destination, rows, cols, ld, prefix);
      else
        return hip_direct_pack_i64_device(0, source, &upload, &upload_bytes, destination, rows, cols, ld, prefix);
    };
    REQUIRE(pack(a.data(), ra.get(), m, k, lda) == RNS8_SUCCESS);
    REQUIRE(pack(b.data(), rb.get(), k, n, ldb) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, dc, actual.data(), actual.size()) == RNS8_SUCCESS);
    REQUIRE(hip_direct_gemm_rns_device(0, ra.get(), rb.get(), dc, m, n, k, k, n, ldc, prefix) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_device_to_host(0, actual.data(), dc, actual.size()) == RNS8_SUCCESS);
    REQUIRE(actual == expected);
    REQUIRE(hip_direct_free(0, upload) == RNS8_SUCCESS);
  }
  bool supported = true;
  for (uint32_t group : {1u, 2u, 4u}) {
    std::fill(actual.begin(), actual.end(), 99);
    REQUIRE(hip_direct_copy_host_to_device(0, dc, actual.data(), actual.size()) == RNS8_SUCCESS);
    auto status = RNS8_UNSUPPORTED_BACKEND;
    if (splits) {
      bool commit = true;
      REQUIRE(hip_direct_gemm_native_wmma_split_k_device(
          0, da, db, static_cast<int8_t*>(dc), m, n, k, lda, ldb, ldc, prefix, group,
          unsigned_inputs, splits, 32, 0, commit) == RNS8_WORKSPACE_TOO_SMALL);
      REQUIRE_FALSE(commit);
      status = hip_direct_gemm_native_wmma_split_k_device(
          0, da, db, static_cast<int8_t*>(dc), m, n, k, lda, ldb, ldc, prefix, group,
          unsigned_inputs, splits, 32, UINT64_MAX, commit);
      REQUIRE(commit == (status == RNS8_SUCCESS));
    } else {
      (void)generated;
#if defined(RNS8_EXPERIMENTAL_GENERATED_NATIVE_WMMA)
    if (generated) {
      for (const auto& variant : rns8::generated::native_variants) {
        if (variant.group == group && variant.unsigned_inputs == unsigned_inputs) {
          INFO("generated variant " << variant.id);
          REQUIRE(variant.launch(da, db, static_cast<int8_t*>(dc), m, n, k, lda, ldb, ldc, prefix) == 0);
          status = RNS8_SUCCESS;
        }
      }
      REQUIRE(status == RNS8_SUCCESS);
    } else
#endif
      status = hip_direct_gemm_native_wmma_grouped_device(
          0, da, db, static_cast<int8_t*>(dc), m, n, k, lda, ldb, ldc, prefix, group, unsigned_inputs);
    }
    REQUIRE(hip_direct_copy_device_to_host(0, actual.data(), dc, actual.size()) == RNS8_SUCCESS);
    if (status == RNS8_UNSUPPORTED_BACKEND) {
      REQUIRE(std::all_of(actual.begin(), actual.end(), [](int8_t x) { return x == 99; }));
      supported = false;
      break;
    }
    REQUIRE(status == RNS8_SUCCESS);
    REQUIRE(actual == expected);
#if defined(RNS8_EXPERIMENTAL_GENERATED_NATIVE_WMMA)
    if (generated) {
      // Reverse the contraction axis in both operands. This moves values
      // across WMMA tiles and slab boundaries while preserving exact AB.
      auto reversed_a = a, reversed_b = b;
      for (int row = 0; row < m; ++row)
        for (int q = 0; q < k; ++q) reversed_a[row * lda + q] = a[row * lda + k - 1 - q];
      for (int q = 0; q < k; ++q)
        for (int col = 0; col < n; ++col) reversed_b[q * ldb + col] = b[(k - 1 - q) * ldb + col];
      REQUIRE(hip_direct_copy_host_to_device(0, da, reversed_a.data(), reversed_a.size() * sizeof(NativeT)) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, db, reversed_b.data(), reversed_b.size() * sizeof(NativeT)) == RNS8_SUCCESS);
      std::fill(actual.begin(), actual.end(), 99);
      REQUIRE(hip_direct_copy_host_to_device(0, dc, actual.data(), actual.size()) == RNS8_SUCCESS);
      for (const auto& variant : rns8::generated::native_variants)
        if (variant.group == group && variant.unsigned_inputs == unsigned_inputs) {
          INFO("reverse K variant " << variant.id);
          REQUIRE(variant.launch(da, db, static_cast<int8_t*>(dc), m, n, k, lda, ldb, ldc, prefix) == 0);
        }
      REQUIRE(hip_direct_copy_device_to_host(0, actual.data(), dc, actual.size()) == RNS8_SUCCESS);
      REQUIRE(actual == expected);
      REQUIRE(hip_direct_copy_host_to_device(0, da, a.data(), a.size() * sizeof(NativeT)) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, db, b.data(), b.size() * sizeof(NativeT)) == RNS8_SUCCESS);
    }
#endif
  }
  REQUIRE(hip_direct_gemm_native_wmma_grouped_device(
      0, da, db, static_cast<int8_t*>(dc), m, n, k, lda, ldb, ldc, prefix, 3, unsigned_inputs) == RNS8_INVALID_ARGUMENT);
  REQUIRE(hip_direct_gemm_native_wmma_grouped_device(
      0, da, db, static_cast<int8_t*>(da), m, n, k, lda, ldb, ldc, prefix, 2, unsigned_inputs) == RNS8_INVALID_ARGUMENT);
  CHECK(hip_direct_free(0, dc) == RNS8_SUCCESS);
  CHECK(hip_direct_free(0, db) == RNS8_SUCCESS);
  CHECK(hip_direct_free(0, da) == RNS8_SUCCESS);
  return supported;
}

TEST_CASE("persistent native WMMA split K canonicalizes odd moduli before hierarchical merges") {
  if (!hip_available()) SKIP("no HIP device for native WMMA split K");
  for (const uint32_t splits : {1u, 3u, 129u})
    for (const int k : {257, 65537})
      for (const uint32_t prefix : {3u, 20u}) {
        const int m = k < 1024 ? 17 : 1, n = k < 1024 ? 19 : 1;
        if (!check_native_wmma_groups<int64_t>(false, m, n, k, prefix, false, splits))
          SKIP("compiled target lacks native WMMA");
        REQUIRE(check_native_wmma_groups<uint64_t>(true, m, n, k, prefix, false, splits));
      }
}

#if defined(RNS8_EXPERIMENTAL_GENERATED_NATIVE_WMMA)
TEST_CASE("generated native WMMA variants match Boost for every declared group and signedness") {
  if (!hip_available()) SKIP("no HIP device for generated native variants");
  for (const int k : {17, 65535, 65536, 65537})
    for (const uint32_t prefix : {1u, 3u, 9u, 20u}) {
      const int m = k < 32 ? 19 : 1, n = k < 32 ? 21 : 1;
      REQUIRE(check_native_wmma_groups<int64_t>(false, m, n, k, prefix, true));
      REQUIRE(check_native_wmma_groups<uint64_t>(true, m, n, k, prefix, true));
    }
}
#endif

TEST_CASE("native grouped WMMA preserves signedness planes tails and safe K slabs") {
  if (!hip_available()) SKIP("no HIP device for grouped native WMMA differential");
  for (const int k : {1, 15, 16, 17, 65535, 65536, 65537})
    for (const uint32_t prefix : {1u, 3u, 9u, 20u}) {
      const int m = k < 32 ? 19 : 1, n = k < 32 ? 21 : 1;
      if (!check_native_wmma_groups<int64_t>(false, m, n, k, prefix))
        SKIP("compiled device target lacks native grouped RDNA3 WMMA");
      REQUIRE(check_native_wmma_groups<uint64_t>(true, m, n, k, prefix));
    }
}
