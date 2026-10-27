template <typename NativeT>
void check_native_bound_discovery(bool unsigned_inputs, rns8::detail::NativeBoundMethod method) {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr int m = 35, n = 33, k = 17, lda = 19, ldb = 36;
  constexpr int tr = 3, tc = 3, tiles = tr * tc;
  std::vector<NativeT> a(m * lda, std::numeric_limits<NativeT>::max());
  std::vector<NativeT> b(k * ldb, std::numeric_limits<NativeT>::max());
  for (int row = 0; row < m; ++row)
    for (int q = 0; q < k; ++q)
      a[row * lda + q] = row < 16 ? NativeT{0} : row < 32 ? (q < 7 ? NativeT{2} : NativeT{0}) : std::numeric_limits<NativeT>::max();
  for (int q = 0; q < k; ++q)
    for (int col = 0; col < n; ++col)
      b[q * ldb + col] = col < 16 ? NativeT{0} : col < 32 ? (q >= 7 ? NativeT{3} : NativeT{0}) : std::numeric_limits<NativeT>::max();
  if (!unsigned_inputs) a[34 * lda + 1] = std::numeric_limits<NativeT>::min();
  void *da = nullptr, *db = nullptr, *dr = nullptr, *di = nullptr, *dof = nullptr;
  REQUIRE(hip_direct_allocate(0, a.size() * sizeof(NativeT), &da) == RNS8_SUCCESS);
  REQUIRE(hip_direct_allocate(0, b.size() * sizeof(NativeT), &db) == RNS8_SUCCESS);
  REQUIRE(hip_direct_allocate(0, (tiles + 1) * sizeof(NativeTileBound), &dr) == RNS8_SUCCESS);
  REQUIRE(hip_direct_allocate(0, (tiles + 1) * sizeof(uint32_t), &di) == RNS8_SUCCESS);
  REQUIRE(hip_direct_allocate(0, 23 * sizeof(uint32_t), &dof) == RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0, db, b.data(), b.size() * sizeof(NativeT)) == RNS8_SUCCESS);
  for (int mutation = 0; mutation < 2; ++mutation) {
    // Recompute after native storage changes; old raw records cannot certify it.
    if (mutation) a[0] = 7;
    REQUIRE(hip_direct_copy_host_to_device(0, da, a.data(), a.size() * sizeof(NativeT)) == RNS8_SUCCESS);
    std::vector<NativeTileBound> records(tiles + 1);
    records.back().reserved = 12345;
    std::vector<uint32_t> indices(tiles + 1, UINT32_MAX), offsets(23, UINT32_MAX);
    REQUIRE(hip_direct_copy_host_to_device(0, dr, records.data(), records.size() * sizeof(NativeTileBound)) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, di, indices.data(), indices.size() * sizeof(uint32_t)) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, dof, offsets.data(), offsets.size() * sizeof(uint32_t)) == RNS8_SUCCESS);
    REQUIRE(hip_direct_discover_native_tile_bounds(0, da, db, m, n, k, lda, ldb, unsigned_inputs, 20,
        static_cast<NativeTileBound*>(dr), static_cast<uint32_t*>(di), static_cast<uint32_t*>(dof), 4096, method, 7) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_device_to_host(0, records.data(), dr, records.size() * sizeof(NativeTileBound)) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_device_to_host(0, indices.data(), di, indices.size() * sizeof(uint32_t)) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_device_to_host(0, offsets.data(), dof, offsets.size() * sizeof(uint32_t)) == RNS8_SUCCESS);
    REQUIRE(records.back().reserved == 12345);
    REQUIRE(indices.back() == UINT32_MAX);
    REQUIRE(offsets.back() == UINT32_MAX);
    REQUIRE(offsets[0] == 0);
    REQUIRE(offsets[21] == tiles);
    std::vector<bool> seen(tiles);
    constexpr int moduli[] = {256,255,253,251,247,239,233,229,227,223,217,211,199,197,193,191,181,179,173,167};
    for (int tile = 0; tile < tiles; ++tile) {
      cpp_int exact = 0;
      const int extent = method == NativeBoundMethod::panel_l1 ? 7 : k;
      for (int begin = 0; begin < k; begin += extent) {
      const int end = std::min(k, begin + extent);
      cpp_int max_a = 0, max_b = 0, sum_a = 0, sum_b = 0;
      for (int row = tile / tc * 16; row < std::min(m, (tile / tc + 1) * 16); ++row) {
        cpp_int row_sum = 0;
        for (int q = begin; q < end; ++q) {
          cpp_int x = boost::multiprecision::abs(cpp_int(a[row * lda + q]));
          if (x > max_a) max_a = x;
          row_sum += x;
        }
        if (row_sum > sum_a) sum_a = row_sum;
      }
      for (int col = tile % tc * 16; col < std::min(n, (tile % tc + 1) * 16); ++col) {
        cpp_int col_sum = 0;
        for (int q = begin; q < end; ++q) {
          cpp_int x = boost::multiprecision::abs(cpp_int(b[q * ldb + col]));
          if (x > max_b) max_b = x;
          col_sum += x;
        }
        if (col_sum > sum_b) sum_b = col_sum;
      }
      cpp_int panel_bound = max_a * max_b * (end - begin);
      if (method != NativeBoundMethod::tile_max) {
        const cpp_int left = max_a * sum_b, right = max_b * sum_a;
        if (left < panel_bound) panel_bound = left;
        if (right < panel_bound) panel_bound = right;
      }
      exact += panel_bound;
      }
      if (method == NativeBoundMethod::panel_l1) {
        if (tile == 4) REQUIRE(exact == 0); // Disjoint nonzero K panels.
        for (int row = tile / tc * 16; row < std::min(m, (tile / tc + 1) * 16); ++row)
          for (int col = tile % tc * 16; col < std::min(n, (tile % tc + 1) * 16); ++col) {
            cpp_int product = 0;
            for (int q = 0; q < k; ++q) product += cpp_int(a[row * lda + q]) * b[q * ldb + col];
            REQUIRE(boost::multiprecision::abs(product) <= exact);
          }
      }
      const auto& record = records[tile];
      REQUIRE(cpp_int(record.magnitude[0]) + (cpp_int(record.magnitude[1]) << 64) +
          (cpp_int(record.magnitude[2]) << 128) == exact);
      cpp_int product = 1;
      uint32_t required = exact == 0 ? 0 : 21;
      for (uint32_t p = 0; p < 20 && required == 21; ++p) {
        product *= moduli[p];
        if (product > exact * (unsigned_inputs ? 1 : 2)) required = p + 1;
      }
      REQUIRE(record.required_prefix == required);
    }
    for (uint32_t p = 0; p <= 20; ++p) {
      REQUIRE(offsets[p] <= offsets[p + 1]);
      for (uint32_t slot = offsets[p]; slot < offsets[p + 1]; ++slot) {
        REQUIRE(indices[slot] < tiles);
        REQUIRE_FALSE(seen[indices[slot]]);
        seen[indices[slot]] = true;
        REQUIRE(records[indices[slot]].required_prefix == p);
      }
    }
    REQUIRE(std::all_of(seen.begin(), seen.end(), [](bool x) { return x; }));
  }
  REQUIRE(hip_direct_discover_native_tile_bounds(0, da, db, m, n, k, lda, ldb, unsigned_inputs, 1,
      static_cast<NativeTileBound*>(dr), static_cast<uint32_t*>(di), static_cast<uint32_t*>(dof), 4096) == RNS8_RANGE_ERROR);
  REQUIRE(hip_direct_discover_native_tile_bounds(0, da, db, m, n, k, lda, ldb, unsigned_inputs, 20,
      static_cast<NativeTileBound*>(dr), static_cast<uint32_t*>(di), static_cast<uint32_t*>(dof), 0) == RNS8_WORKSPACE_TOO_SMALL);
  for (uint32_t invalid_panel : {0u, 65537u})
    REQUIRE(hip_direct_discover_native_tile_bounds(0, da, db, m, n, k, lda, ldb, unsigned_inputs, 20,
        static_cast<NativeTileBound*>(dr), static_cast<uint32_t*>(di), static_cast<uint32_t*>(dof), 4096,
        NativeBoundMethod::panel_l1, invalid_panel) == RNS8_INVALID_ARGUMENT);
  REQUIRE(hip_direct_discover_native_tile_bounds(0, da, db, m, n, k, lda, ldb, unsigned_inputs, 20,
      static_cast<NativeTileBound*>(dr), static_cast<uint32_t*>(di), static_cast<uint32_t*>(dof), 0,
      NativeBoundMethod::panel_l1, 7) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(hip_direct_free(0, dof) == RNS8_SUCCESS);
  CHECK(hip_direct_free(0, di) == RNS8_SUCCESS);
  CHECK(hip_direct_free(0, dr) == RNS8_SUCCESS);
  CHECK(hip_direct_free(0, db) == RNS8_SUCCESS);
  CHECK(hip_direct_free(0, da) == RNS8_SUCCESS);
}
TEST_CASE("device native bound discovery proves prefixes and complete compact buckets") {
  if (!hip_available()) SKIP("no HIP device for native bound discovery");
  for (auto method : {rns8::detail::NativeBoundMethod::tile_max, rns8::detail::NativeBoundMethod::axis_l1,
                      rns8::detail::NativeBoundMethod::panel_l1}) {
    check_native_bound_discovery<int64_t>(false, method);
    check_native_bound_discovery<uint64_t>(true, method);
  }
}
