TEST_CASE("direct HIP block sparse intersections match exact dense products") {
  if (!hip_available())
    SKIP("no HIP device for block-sparse differential");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  auto run = [](int m, int n, int k, int mask_a, int mask_b, uint32_t prefix) {
    const int lda = k + 1, ldb = n + 2, ldc = n + 1;
    std::vector<int64_t> a(m * lda, 99), b(k * ldb, 99);
    for (int row = 0; row < m; ++row)
      for (int q = 0; q < k; ++q)
        a[row * lda + q] =
            (mask_a & (1 << (q / 64))) ? ((row + q) % 2 ? INT64_MIN : 256) : 0;
    for (int q = 0; q < k; ++q)
      for (int col = 0; col < n; ++col)
        b[q * ldb + col] =
            (mask_b & (1 << (q / 64))) ? ((q + col) % 2 ? INT64_MAX : -3) : 0;
    BlockSparseOperand lhs, rhs;
    std::vector<int8_t> pa, pb;
    REQUIRE(pack_block_sparse_i64(a.data(), m, k, lda, prefix,
                                  BlockSparseRole::lhs, 11, 2, UINT64_MAX, lhs,
                                  pa) == RNS8_SUCCESS);
    REQUIRE(pack_block_sparse_i64(b.data(), k, n, ldb, prefix,
                                  BlockSparseRole::rhs, 12, 3, UINT64_MAX, rhs,
                                  pb) == RNS8_SUCCESS);
    std::vector<int8_t> c(prefix * m * ldc, 99), expected = c;
    for (int row = 0; row < m; ++row)
      for (int col = 0; col < n; ++col) {
        cpp_int sum = 0;
        for (int q = 0; q < k; ++q)
          sum += cpp_int(a[row * lda + q]) * b[q * ldb + col];
        for (uint32_t p = 0; p < prefix; ++p)
          expected[(p * m + row) * ldc + col] =
              centered_residue(sum, kDefaultModuli[p]);
      }
    void *da = nullptr, *db = nullptr, *dc = nullptr;
    REQUIRE(hip_direct_allocate(0, std::max<std::size_t>(1, pa.size()), &da) ==
            RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, std::max<std::size_t>(1, pb.size()), &db) ==
            RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, c.size(), &dc) == RNS8_SUCCESS);
    if (!pa.empty())
      REQUIRE(hip_direct_copy_host_to_device(0, da, pa.data(), pa.size()) ==
              RNS8_SUCCESS);
    if (!pb.empty())
      REQUIRE(hip_direct_copy_host_to_device(0, db, pb.data(), pb.size()) ==
              RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, dc, c.data(), c.size()) ==
            RNS8_SUCCESS);
    REQUIRE(hip_direct_gemm_block_sparse_device(0, da, lhs, 11, 2, db, rhs, 12,
                                                3, dc, ldc,
                                                4096) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_device_to_host(0, c.data(), dc, c.size()) ==
            RNS8_SUCCESS);
    REQUIRE(c == expected);
    REQUIRE(hip_direct_gemm_block_sparse_device(0, da, lhs, 11, 9, db, rhs, 12,
                                                3, dc, ldc,
                                                4096) == RNS8_INVALID_ARGUMENT);
    REQUIRE(hip_direct_gemm_block_sparse_device(0, da, lhs, 11, 2, db, rhs, 12,
                                                3, dc, ldc,
                                                0) == RNS8_WORKSPACE_TOO_SMALL);
    CHECK(hip_direct_free(0, dc) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, db) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, da) == RNS8_SUCCESS);
  };
  for (uint32_t prefix : {1u, 9u, 20u}) {
    for (int ma = 0; ma < 4; ++ma)
      for (int mb = 0; mb < 4; ++mb)
        run(1, 1, 65, ma, mb, prefix);
    run(17, 19, 129, 5, 6, prefix);
    run(17, 19, 129, 7, 7, prefix);
  }
}
