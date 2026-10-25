#include "backend_common/rank23_slp_resources.hpp"
TEST_CASE(
    "one-level HIP Strassen coefficient rank23 and actual SLP match classical residues with odd tails") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  if (!hip_available())
    SKIP("no HIP device for Strassen");
  for (int route : {0,1,2}) for (uint32_t prefix : {1u, 3u, 9u, 20u})
    for (int k : {1, 2, 3, 17, route ? 196609 : 131073}) {
      const bool rank23 = route != 0;
      const auto multiply = route == 2 ? hip_direct_gemm_rank23_slp_device :
                            rank23 ? hip_direct_gemm_rank23_device : hip_direct_gemm_strassen_device;
      // The 3x3 case exercises individual rank-23 output blocks, including
      // skipped zero-coefficient contributions and a padded K block.
      const int m = k > 100 ? 1 : k == 2 ? 3 : 5, n = k > 100 ? 1 : k == 2 ? 3 : 7, lda = k + 1,
                ldb = n + 2, ldc = n + 3;
      std::vector<int8_t> a(prefix * m * lda, 99), b(prefix * k * ldb, 99),
          expected(prefix * m * ldc, 77), observed(expected.size(), 77);
      for (uint32_t p = 0; p < prefix; ++p) {
        const int modulus = kDefaultModuli[p];
        for (int row = 0; row < m; ++row)
          for (int q = 0; q < k; ++q)
            a[(p * m + row) * lda + q] =
                centered_residue(cpp_int((row + q) % 2 ? -128 : 127), modulus);
        for (int q = 0; q < k; ++q)
          for (int col = 0; col < n; ++col)
            b[(p * k + q) * ldb + col] =
                centered_residue(cpp_int((q + col) % 3 ? -127 : 128), modulus);
        for (int row = 0; row < m; ++row)
          for (int col = 0; col < n; ++col) {
            cpp_int sum = 0;
            for (int q = 0; q < k; ++q)
              sum += cpp_int(a[(p * m + row) * lda + q]) *
                     b[(p * k + q) * ldb + col];
            expected[(p * m + row) * ldc + col] =
                centered_residue(sum, modulus);
          }
      }
      void *da = nullptr, *db = nullptr, *dc = nullptr;
      REQUIRE(hip_direct_allocate(0, a.size(), &da) == RNS8_SUCCESS);
      REQUIRE(hip_direct_allocate(0, b.size(), &db) == RNS8_SUCCESS);
      REQUIRE(hip_direct_allocate(0, observed.size(), &dc) == RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, da, a.data(), a.size()) ==
              RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, db, b.data(), b.size()) ==
              RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0, dc, observed.data(),
                                             observed.size()) == RNS8_SUCCESS);
      bool committed = true;
      REQUIRE(multiply(0, da, db, dc, m, n, k, lda, ldb,
                                              ldc, prefix, 0, committed,nullptr,0) ==
              RNS8_WORKSPACE_TOO_SMALL);
      REQUIRE_FALSE(committed);
      REQUIRE(hip_direct_copy_device_to_host(0, observed.data(), dc,
                                             observed.size()) == RNS8_SUCCESS);
      REQUIRE(std::all_of(observed.begin(), observed.end(),
                          [](int8_t v) { return v == 77; }));
      if (route == 2) {
        const auto resources=query_rank23_slp_resources(m,n,k,lda,ldb,ldc,prefix,UINT64_MAX);
        REQUIRE(resources.status==RNS8_SUCCESS);
        REQUIRE(multiply(0,da,db,dc,m,n,k,lda,ldb,ldc,prefix,resources.total_bytes-1,
                         committed,nullptr,0)==RNS8_WORKSPACE_TOO_SMALL);
        REQUIRE_FALSE(committed);
        REQUIRE(multiply(0,da,db,da,m,n,k,lda,ldb,ldc,prefix,UINT64_MAX,
                         committed,nullptr,0)==RNS8_INVALID_ARGUMENT);
        REQUIRE_FALSE(committed);
      }
      REQUIRE(multiply(0, da, db, dc, m, n, k, lda, ldb,
                                              ldc, prefix, UINT64_MAX,
                                              committed,nullptr,0) == RNS8_SUCCESS);
      REQUIRE(committed);
      REQUIRE(hip_direct_copy_device_to_host(0, observed.data(), dc,
                                             observed.size()) == RNS8_SUCCESS);
      REQUIRE(observed == expected);
      CHECK(hip_direct_free(0, dc) == RNS8_SUCCESS);
      CHECK(hip_direct_free(0, db) == RNS8_SUCCESS);
      CHECK(hip_direct_free(0, da) == RNS8_SUCCESS);
    }
}

template<class T> void check_public_strassen_owner(rns8_bilinear_method method=RNS8_BILINEAR_STRASSEN) {
  using boost::multiprecision::cpp_int;
  constexpr bool is_unsigned = std::is_unsigned_v<T>;
  const auto semantics = is_unsigned ? RNS8_EXACT_WIDE_UNSIGNED : RNS8_EXACT_WIDE_SIGNED;
  auto* ctx = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto descriptor = exact_signed_desc(17, 17, 17, RNS8_BACKEND_HIP_DIRECT);
  descriptor.semantics = semantics;
  descriptor.max_prefix = 20;
  descriptor.flags |= RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_plan(ctx, &descriptor, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  auto md = matrix_desc(17, 17, semantics, RNS8_BOUND_NONE);
  md.max_prefix = 20;
  md.flags = 0;
  rns8_matrix *a=nullptr, *b=nullptr, *c=nullptr;
  REQUIRE(rns8_create_matrix(ctx, &md, &a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &md, &b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &md, &c) == RNS8_SUCCESS);
  std::vector<T> av(17*19), bv(17*19);
  for (int r=0; r<17; ++r) for (int q=0; q<17; ++q) {
    av[r*19+q] = r<16 ? T(2) : std::numeric_limits<T>::max();
    bv[r*19+q] = q<16 ? T(3) : std::numeric_limits<T>::max();
  }
  if constexpr (!std::is_unsigned_v<T>) av[16*19+1] = std::numeric_limits<T>::min();
  const auto pack = [&](rns8_matrix* matrix, const std::vector<T>& data) {
    if constexpr (std::is_unsigned_v<T>) return rns8_pack_u64(ctx, matrix, data.data(), 19, 0);
    else return rns8_pack_i64(ctx, matrix, data.data(), 19, 0);
  };
  double overhead=-1;
  const auto run = [&](int level) {
    if(method==RNS8_BILINEAR_STRASSEN || level!=1)
      return rns8_strassen_gemm_research(ctx,plan,a,b,c,workspace,level,&overhead);
    return rns8_gemm_bilinear_exact(ctx,plan,a,b,c,method,UINT64_MAX,workspace,&overhead);
  };
  REQUIRE(pack(a,av) == RNS8_SUCCESS);
  REQUIRE(pack(b,bv) == RNS8_SUCCESS);
  REQUIRE(pack(c,av) == RNS8_SUCCESS);
  const auto initial_generation=c->content_generation;
  CHECK(rns8_gemm_bilinear_exact(ctx,plan,a,b,c,method,0,workspace,&overhead)==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(c->content_generation==initial_generation);CHECK(overhead==-1);
  CHECK(rns8_gemm_bilinear_exact(ctx,plan,a,b,c,static_cast<rns8_bilinear_method>(999),UINT64_MAX,workspace,&overhead)==RNS8_INVALID_ARGUMENT);
  CHECK(run(2)==RNS8_UNSUPPORTED_BACKEND);
  CHECK(overhead==-1);
  c->in_flight_readers=1;
  CHECK(run(1)==RNS8_BUSY);
  c->in_flight_readers=0;
  CHECK(c->content_generation==initial_generation);
  CHECK(overhead==-1);
  for (int mutation=0; mutation<2; ++mutation) {
    if (mutation) {
      av[0] = std::numeric_limits<T>::max();
      REQUIRE(pack(a,av) == RNS8_SUCCESS);
    }
    {
      REQUIRE(run(1)==RNS8_SUCCESS);
      CHECK(overhead>0);
      std::vector<uint64_t> limbs(17*17*3);
      if constexpr (std::is_unsigned_v<T>)
        REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx,plan,c,limbs.data(),17,3) == RNS8_SUCCESS);
      else
        REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c,limbs.data(),17,3) == RNS8_SUCCESS);
      for (int r=0; r<17; ++r) for (int col=0; col<17; ++col) {
        cpp_int expected=0;
        for (int q=0; q<17; ++q) expected += cpp_int(av[r*19+q])*bv[q*19+col];
        const int at=(r*17+col)*3;
        cpp_int actual = cpp_int(limbs[at])+(cpp_int(limbs[at+1])<<64)+(cpp_int(limbs[at+2])<<128);
        if constexpr (!std::is_unsigned_v<T>) if (limbs[at+2]>>63) actual -= cpp_int(1)<<192;
        REQUIRE(actual == expected);
      }
    }
  }
  rns8_destroy_matrix(c); rns8_destroy_matrix(b); rns8_destroy_matrix(a);
  rns8_destroy_workspace(workspace); rns8_destroy_plan(plan); rns8_destroy_context(ctx);
}
TEST_CASE("public Strassen owns signed and unsigned exact exports and preserves rejected outputs") {
  if (!hip_available()) SKIP("no HIP device for public Strassen");
  check_public_strassen_owner<int64_t>();
  check_public_strassen_owner<uint64_t>();
}

TEST_CASE("public rank23 methods retain exact proof and output transaction ownership") {
  if(!hip_available()) SKIP("no HIP device for public rank23 methods");
  for(auto method:{RNS8_BILINEAR_RANK23_COEFFICIENTS,RNS8_BILINEAR_RANK23_SLP,RNS8_BILINEAR_RANK23_OCTOBER55,RNS8_BILINEAR_RANK23_OCTOBER55_RECOMPUTE}) {
    check_public_strassen_owner<int64_t>(method);
    check_public_strassen_owner<uint64_t>(method);
  }
}
