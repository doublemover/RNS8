TEST_CASE("shared tiled residue dataflow preserves native resident region tails and slab carries",
          "[hip][residue-tile]") {
  using namespace rns8::detail;
  if (!hip_available())
    SKIP("no HIP device for shared tiled residue differential");
  enum class Route { resident, native_ab, native_a, region };
  for (int64_t k : {65, 65537})
    for (uint16_t q : {247, 251, 256}) {
      const int64_t m = k > 1000 ? 1 : 17, n = k > 1000 ? 17 : 19, lda = k + 3, ldb = n + 5,
                    ldc = n + 7;
      std::vector<uint8_t> raw_a(std::size_t(m * lda), 0xa5), raw_b(std::size_t(k * ldb), 0x5a);
      std::vector<int8_t> centered_a(raw_a.size()), centered_b(raw_b.size());
      auto center = [q](boost::multiprecision::cpp_int value) {
        value %= q;
        if (value < 0)
          value += q;
        if (value >= (q + 1) / 2)
          value -= q;
        return value.convert_to<int8_t>();
      };
      for (std::size_t index = 0; index < raw_a.size(); ++index) {
        raw_a[index] = uint8_t((index * 37 + 17) & 255);
        centered_a[index] = center(raw_a[index]);
      }
      for (std::size_t index = 0; index < raw_b.size(); ++index) {
        raw_b[index] = uint8_t((index * 23 + 41) & 255);
        centered_b[index] = center(raw_b[index]);
      }
      const int64_t rr = m > 1 ? 1 : 0, rc = m > 1 ? 3 : 1, rrows = m > 1 ? m - 2 : m,
                    rcols = m > 1 ? n - 4 : n - 2;
      std::vector<int8_t> exact(std::size_t(m * ldc), 91);
      for (int64_t row = 0; row < m; ++row)
        for (int64_t col = 0; col < n; ++col) {
          boost::multiprecision::cpp_int sum = 0;
          for (int64_t x = 0; x < k; ++x)
            sum += boost::multiprecision::cpp_int(raw_a[std::size_t(row * lda + x)]) *
                   raw_b[std::size_t(x * ldb + col)];
          exact[std::size_t(row * ldc + col)] = center(sum);
        }
      for (auto route : {Route::resident, Route::native_ab, Route::native_a, Route::region}) {
        INFO("route=" << int(route) << " q=" << q << " K=" << k);
        std::vector<int8_t> actual(std::size_t(m * ldc), 91), expected = exact;
        if (route == Route::region)
          for (int64_t row = 0; row < m; ++row)
            for (int64_t col = 0; col < n; ++col)
              if (row < rr || row >= rr + rrows || col < rc || col >= rc + rcols)
                expected[std::size_t(row * ldc + col)] = 91;
        hip_direct_device_buffer a, b, c;
        REQUIRE(a.allocate(0, raw_a.size()) == RNS8_SUCCESS);
        REQUIRE(b.allocate(0, raw_b.size()) == RNS8_SUCCESS);
        REQUIRE(c.allocate(0, actual.size()) == RNS8_SUCCESS);
        struct Drain {
          hip_direct_device_buffer& a;
          hip_direct_device_buffer& b;
          hip_direct_device_buffer& c;
          bool pending = true;
          ~Drain() {
            if (pending) {
              if (hip_direct_synchronize(0) != RNS8_SUCCESS)
                std::abort();
              a.mark_completed();
              b.mark_completed();
              c.mark_completed();
            }
          }
        } drain{a, b, c};
        a.mark_pending();
        b.mark_pending();
        c.mark_pending();
        const void* source_a = (route == Route::native_ab || route == Route::native_a)
                                   ? static_cast<const void*>(raw_a.data())
                                   : centered_a.data();
        const void* source_b =
            route == Route::native_ab ? static_cast<const void*>(raw_b.data()) : centered_b.data();
        REQUIRE(hip_direct_copy_host_to_device(0, a.get(), source_a, raw_a.size()) == RNS8_SUCCESS);
        REQUIRE(hip_direct_copy_host_to_device(0, b.get(), source_b, raw_b.size()) == RNS8_SUCCESS);
        REQUIRE(hip_direct_copy_host_to_device(0, c.get(), actual.data(), actual.size()) ==
                RNS8_SUCCESS);
        rns8_status status = RNS8_INTERNAL_ERROR;
        switch (route) {
        case Route::resident:
          status = hip_direct_gemm_finite_u8_resident_device(0, a.get(), b.get(), c.get(), m, n, k,
                                                             lda, ldb, ldc, q);
          break;
        case Route::native_ab:
          status = hip_direct_gemm_finite_u8_native_device(0, a.get(), b.get(), c.get(), m, n, k,
                                                           lda, ldb, ldc, q);
          break;
        case Route::native_a:
          status = hip_direct_gemm_finite_u8_native_a_resident_b_device(
              0, a.get(), b.get(), c.get(), m, n, k, lda, ldb, ldc, q);
          break;
        case Route::region:
          status = hip_direct_gemm_finite_u8_region_device(0, a.get(), b.get(), c.get(), m, n, k,
                                                           lda, ldb, ldc, rr, rc, rrows, rcols, q);
          break;
        }
        REQUIRE(status == RNS8_SUCCESS);
        REQUIRE(hip_direct_copy_device_to_host(0, actual.data(), c.get(), actual.size()) ==
                RNS8_SUCCESS);
        a.mark_completed();
        b.mark_completed();
        c.mark_completed();
        drain.pending = false;
        CHECK(actual == expected);
        REQUIRE(c.reset() == RNS8_SUCCESS);
        REQUIRE(b.reset() == RNS8_SUCCESS);
        REQUIRE(a.reset() == RNS8_SUCCESS);
      }
    }
}
