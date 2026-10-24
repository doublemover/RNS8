#include "test_hip_direct_native_byte_cases.inc"
TEST_CASE("direct HIP native narrow admission proves partial sums from actual device inputs") {
  if (!hip_available()) SKIP("no HIP device for native narrow differential");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  auto run = [](const std::vector<int64_t>& av, const std::vector<int64_t>& bv,
                int64_t m, int64_t n, int64_t k, rns8_status expected_status) {
    const int64_t lda = k + 1, ldb = n + 1, ldc = n + 2;
    std::vector<int64_t> a(m * lda, INT64_MIN), b(k * ldb, INT64_MIN), c(m * ldc, 99);
    for (int64_t row = 0; row < m; ++row)
      std::copy_n(av.data() + row * k, k, a.data() + row * lda);
    for (int64_t row = 0; row < k; ++row)
      std::copy_n(bv.data() + row * n, n, b.data() + row * ldb);
    void *da = nullptr, *db = nullptr, *dc = nullptr;
    REQUIRE(hip_direct_allocate(0, a.size() * sizeof(int64_t), &da) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, b.size() * sizeof(int64_t), &db) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, c.size() * sizeof(int64_t), &dc) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, da, a.data(), a.size() * sizeof(int64_t)) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, db, b.data(), b.size() * sizeof(int64_t)) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, dc, c.data(), c.size() * sizeof(int64_t)) == RNS8_SUCCESS);
    uint64_t bound = UINT64_MAX;
    REQUIRE(hip_direct_gemm_i64_narrow_checked_device(0, static_cast<const int64_t*>(da), static_cast<const int64_t*>(db),
                                                      static_cast<int64_t*>(dc), m, n, k, lda, ldb, ldc, &bound) == expected_status);
    REQUIRE(hip_direct_copy_device_to_host(0, c.data(), dc, c.size() * sizeof(int64_t)) == RNS8_SUCCESS);
    if (expected_status == RNS8_SUCCESS) {
      REQUIRE(bound <= static_cast<uint64_t>(INT64_MAX));
      for (int64_t row = 0; row < m; ++row)
        for (int64_t col = 0; col < n; ++col) {
          cpp_int expected = 0;
          for (int64_t q = 0; q < k; ++q) expected += cpp_int(av[row * k + q]) * bv[q * n + col];
          REQUIRE(cpp_int(c[row * ldc + col]) == expected);
          REQUIRE(boost::multiprecision::abs(expected) <= bound);
        }
      for (int64_t row = 0; row < m; ++row)
        for (int64_t col = n; col < ldc; ++col) REQUIRE(c[row * ldc + col] == 99);
    } else {
      REQUIRE(bound == UINT64_MAX);
      REQUIRE(std::all_of(c.begin(), c.end(), [](int64_t value) { return value == 99; }));
    }
    CHECK(hip_direct_free(0, dc) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, db) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, da) == RNS8_SUCCESS);
  };
  run({INT64_MAX}, {1}, 1, 1, 1, RNS8_SUCCESS);
  run({INT64_MIN}, {0}, 1, 1, 1, RNS8_SUCCESS);
  run({INT64_MIN}, {-1}, 1, 1, 1, RNS8_RANGE_ERROR);
  run({INT64_MAX, INT64_MAX}, {1, -1}, 1, 1, 2, RNS8_RANGE_ERROR); // final zero is insufficient
  for (const int64_t k : {1, 7, 65537}) {
    constexpr int64_t m = 3, n = 5;
    std::vector<int64_t> a(m * k), b(k * n);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<int64_t>(i % 23) - 11;
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<int64_t>(i % 19) - 9;
    run(a, b, m, n, k, RNS8_SUCCESS);
  }
}

template<class T> void check_public_narrow_owner(rns8_narrow_product_method method) {
  using boost::multiprecision::cpp_int;
  constexpr bool u=std::is_unsigned_v<T>;
  constexpr int side=17,host_ld=19;
  const auto semantics=u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  auto desc=exact_signed_desc(side,side,side,RNS8_BACKEND_HIP_DIRECT);
  desc.semantics=semantics; desc.max_prefix=20; desc.flags|=RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* plan=nullptr;
  rns8_workspace* ws=nullptr;
  REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx,plan,&ws)==RNS8_SUCCESS);
  auto md=matrix_desc(side,side,semantics,RNS8_BOUND_NONE);
  md.max_prefix=20; md.flags=RNS8_MATRIX_RETAIN_NATIVE_INPUT;
  rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
  REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx,&md,&b)==RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);
  const auto cleanup=[&]() {
    rns8_destroy_matrix(c); rns8_destroy_matrix(b); rns8_destroy_matrix(a);
    rns8_destroy_workspace(ws); rns8_destroy_plan(plan); rns8_destroy_context(ctx);
  };
  std::vector<T> av(side*host_ld,T(77)),bv(side*host_ld,T(77));
  for(int r=0;r<side;++r) for(int q=0;q<side;++q) {
    av[r*host_ld+q]=u?T((r+q)%2?255:0):T((r+q)%2?-128:127);
    bv[r*host_ld+q]=u?T((r+q)%3?251:1):T((r+q)%3?-127:126);
  }
  const auto pack=[&](rns8_matrix* matrix,const std::vector<T>& data) {
    if constexpr(u) return rns8_pack_u64(ctx,matrix,data.data(),host_ld,0);
    else return rns8_pack_i64(ctx,matrix,data.data(),host_ld,0);
  };
  const auto run=[&](uint64_t budget=UINT64_MAX) {
    return rns8_gemm_narrow_exact(ctx,plan,a,b,c,method,budget,ws);
  };
  REQUIRE(pack(a,av)==RNS8_SUCCESS); REQUIRE(pack(b,bv)==RNS8_SUCCESS); REQUIRE(pack(c,av)==RNS8_SUCCESS);
  auto generation=c->content_generation;
  CHECK(run(0)==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(c->content_generation==generation);
  CHECK(c->device_native_current);
  c->in_flight_readers=1;
  CHECK(run()==RNS8_BUSY);
  c->in_flight_readers=0;
  CHECK(c->content_generation==generation);
  CHECK(rns8_gemm_narrow_exact(ctx,plan,a,b,a,method,UINT64_MAX,ws)==RNS8_INVALID_ARGUMENT);
  for(int mutation=0;mutation<2;++mutation) {
    if(mutation) { av[0]=3; REQUIRE(pack(a,av)==RNS8_SUCCESS); }
    const auto status=run();
    if(status==RNS8_UNSUPPORTED_BACKEND) { cleanup(); SKIP("compiled device lacks requested narrow method"); }
    REQUIRE(status==RNS8_SUCCESS);
    CHECK_FALSE(c->device_native_current);
    std::vector<uint64_t> limbs(side*side*3);
    if constexpr(u) REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx,plan,c,limbs.data(),side,3)==RNS8_SUCCESS);
    else REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c,limbs.data(),side,3)==RNS8_SUCCESS);
    for(int r=0;r<side;++r) for(int col=0;col<side;++col) {
      cpp_int expected=0;
      for(int q=0;q<side;++q) expected+=cpp_int(av[r*host_ld+q])*bv[q*host_ld+col];
      const int i=(r*side+col)*3;
      cpp_int actual=cpp_int(limbs[i])+(cpp_int(limbs[i+1])<<64)+(cpp_int(limbs[i+2])<<128);
      if constexpr(!u) if(limbs[i+2]>>63) actual-=cpp_int(1)<<192;
      CHECK(actual==expected);
    }
  }
  // Admission must use the current packed generation and reject invalid values.
  av[0]=method==RNS8_NARROW_BYTE_WMMA?T(256):std::numeric_limits<T>::max();
  REQUIRE(pack(a,av)==RNS8_SUCCESS);
  generation=c->content_generation;
  CHECK(run()==RNS8_RANGE_ERROR);
  CHECK(c->content_generation==generation);
  CHECK(c->device_residues_current);
  av[0]=3; REQUIRE(pack(a,av)==RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx,plan,b,b,a,ws)==RNS8_SUCCESS);
  CHECK_FALSE(a->device_native_current);
  CHECK(run()==RNS8_INVALID_ARGUMENT);
  CHECK(c->content_generation==generation);
  cleanup();
}
TEST_CASE("public narrow products preserve exact ownership and reject changed domains") {
  if(!hip_available()) SKIP("no HIP device for public narrow ownership");
  SECTION("signed scalar") { check_public_narrow_owner<int64_t>(RNS8_NARROW_SCALAR_I64); }
  SECTION("signed byte WMMA") { check_public_narrow_owner<int64_t>(RNS8_NARROW_BYTE_WMMA); }
  SECTION("unsigned byte WMMA") { check_public_narrow_owner<uint64_t>(RNS8_NARROW_BYTE_WMMA); }
}
