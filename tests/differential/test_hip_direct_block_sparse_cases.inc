TEST_CASE("direct HIP block sparse intersections match exact dense products") {
  if (!hip_available())
    SKIP("no HIP device for block-sparse differential");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  auto* capability_ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  const std::string target=capability_ctx->device_info.gcn_arch;
  const bool expected_wmma=target.find("gfx1100")==0 || target.find("gfx1101")==0 || target.find("gfx1102")==0;
  REQUIRE(rns8_destroy_context(capability_ctx)==RNS8_SUCCESS);
  auto run = [expected_wmma](int m, int n, int k, int mask_a, int mask_b, uint32_t prefix) {
    const int lda = k + 1, ldb = n + 2, ldc = n + 1;
    std::vector<int64_t> a(m * lda, 99), b(k * ldb, 99);
    for (int row = 0; row < m; ++row)
      for (int q = 0; q < k; ++q)
        a[row * lda + q] =
            (mask_a & (1 << ((q / 64)%3))) ? ((row + q) % 2 ? INT64_MIN : 256) : 0;
    for (int q = 0; q < k; ++q)
      for (int col = 0; col < n; ++col)
        b[q * ldb + col] =
            (mask_b & (1 << ((q / 64)%3))) ? ((q + col) % 2 ? INT64_MAX : -3) : 0;
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
    for(uint32_t method:{0u,1u,2u}) for(uint32_t threshold:{0u,1u,UINT32_MAX}) {
      if(method!=2 && threshold) continue;
      std::fill(c.begin(),c.end(),99);
      REQUIRE(hip_direct_copy_host_to_device(0,dc,c.data(),c.size())==RNS8_SUCCESS);
      bool started=false;
      const auto status=hip_direct_gemm_block_sparse_device(0,da,lhs,11,2,db,rhs,12,3,dc,ldc,
          UINT64_MAX,&started,method,threshold);
      if(status==RNS8_UNSUPPORTED_BACKEND && method && !expected_wmma) {
        CHECK_FALSE(started);
        REQUIRE(hip_direct_copy_device_to_host(0,c.data(),dc,c.size())==RNS8_SUCCESS);
        CHECK(std::all_of(c.begin(),c.end(),[](int8_t v){return v==99;}));
        continue;
      }
      REQUIRE(status==RNS8_SUCCESS);CHECK(started);
      REQUIRE(hip_direct_copy_device_to_host(0,c.data(),dc,c.size())==RNS8_SUCCESS);
      REQUIRE(c==expected);
    }
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
  for(int k:{65535,65536,65537}) run(1,1,k,7,7,20);
}

template<class T> void check_public_block_sparse_owner(rns8_block_sparse_method method=RNS8_BLOCK_SPARSE_SCALAR) {
  using boost::multiprecision::cpp_int;
  constexpr bool u=std::is_unsigned_v<T>;
  constexpr int m=17,n=19,k=65;
  const auto semantics=u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  auto desc=exact_signed_desc(m,n,k,RNS8_BACKEND_HIP_DIRECT);
  desc.semantics=semantics; desc.max_prefix=20; desc.flags|=RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* plan=nullptr; rns8_workspace* ws=nullptr;
  REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx,plan,&ws)==RNS8_SUCCESS);
  rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
  const auto create=[&](int rows,int cols,rns8_matrix** out) {
    auto md=matrix_desc(rows,cols,semantics,RNS8_BOUND_NONE);
    md.max_prefix=20; md.flags=RNS8_MATRIX_RETAIN_NATIVE_INPUT;
    REQUIRE(rns8_create_matrix(ctx,&md,out)==RNS8_SUCCESS);
  };
  create(m,k,&a); create(k,n,&b); create(m,n,&c);
  std::vector<T> av(m*k,T(0)),bv(k*n,T(0)),cv(m*n,T(1));
  av[0]=std::numeric_limits<T>::max(); av[16*k+64]=T(3);
  bv[0]=T(5); bv[64*n+18]=std::numeric_limits<T>::max();
  if constexpr(!u) av[1]=std::numeric_limits<T>::min();
  bv[n]=T(7);
  const auto pack=[&](rns8_matrix* matrix,const std::vector<T>& data,int ld) {
    if constexpr(u) return rns8_pack_u64(ctx,matrix,data.data(),ld,0);
    else return rns8_pack_i64(ctx,matrix,data.data(),ld,0);
  };
  REQUIRE(pack(a,av,k)==RNS8_SUCCESS); REQUIRE(pack(b,bv,n)==RNS8_SUCCESS); REQUIRE(pack(c,cv,n)==RNS8_SUCCESS);
  rns8_block_sparse_operand *sa=nullptr,*sb=nullptr;
  CHECK(rns8_prepare_block_sparse_operand(ctx,a,RNS8_BLOCK_SPARSE_LHS,0,&sa)==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(sa==nullptr);
  REQUIRE(rns8_prepare_block_sparse_operand(ctx,a,RNS8_BLOCK_SPARSE_LHS,UINT64_MAX,&sa)==RNS8_SUCCESS);
  REQUIRE(rns8_prepare_block_sparse_operand(ctx,b,RNS8_BLOCK_SPARSE_RHS,UINT64_MAX,&sb)==RNS8_SUCCESS);
  CHECK(rns8_destroy_matrix(a)==RNS8_BUSY);
  const auto run=[&](uint64_t budget=UINT64_MAX) {
    return rns8_gemm_block_sparse_exact_with_method(ctx,plan,sa,sb,c,method,method==RNS8_BLOCK_SPARSE_HYBRID?1u:0u,budget,ws);
  };
  auto generation=c->content_generation;
  CHECK(run(0)==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(c->content_generation==generation);
  CHECK(c->device_native_current);
  c->in_flight_readers=1;
  CHECK(run()==RNS8_BUSY);
  c->in_flight_readers=0;
  REQUIRE(run()==RNS8_SUCCESS);
  std::vector<uint64_t> limbs(m*n*3);
  if constexpr(u) REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx,plan,c,limbs.data(),n,3)==RNS8_SUCCESS);
  else REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c,limbs.data(),n,3)==RNS8_SUCCESS);
  for(int row=0;row<m;++row) for(int col=0;col<n;++col) {
    cpp_int expected=0;
    for(int q=0;q<k;++q) expected+=cpp_int(av[row*k+q])*bv[q*n+col];
    const int i=(row*n+col)*3;
    cpp_int actual=cpp_int(limbs[i])+(cpp_int(limbs[i+1])<<64)+(cpp_int(limbs[i+2])<<128);
    if constexpr(!u) if(limbs[i+2]>>63) actual-=cpp_int(1)<<192;
    CHECK(actual==expected);
  }
  generation=c->content_generation;
  std::fill(av.begin(),av.end(),T(0));
  REQUIRE(pack(a,av,k)==RNS8_SUCCESS);
  CHECK(run()==RNS8_INVALID_ARGUMENT);
  CHECK(c->content_generation==generation);
  REQUIRE(rns8_destroy_block_sparse_operand(sa)==RNS8_SUCCESS);
  REQUIRE(rns8_prepare_block_sparse_operand(ctx,a,RNS8_BLOCK_SPARSE_LHS,UINT64_MAX,&sa)==RNS8_SUCCESS);
  REQUIRE(run()==RNS8_SUCCESS); // empty payload is a real exact-zero operand
  if constexpr(u) REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx,plan,c,limbs.data(),n,3)==RNS8_SUCCESS);
  else REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c,limbs.data(),n,3)==RNS8_SUCCESS);
  for(auto word:limbs) CHECK(word==0);
  REQUIRE(rns8_destroy_block_sparse_operand(sb)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_block_sparse_operand(sa)==RNS8_SUCCESS);
  for(auto* matrix:{a,b,c}) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(ws)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
TEST_CASE("public block sparse handles retain sources and reject stale generations") {
  if(!hip_available()) SKIP("no HIP device for public block sparse ownership");
  SECTION("signed") { check_public_block_sparse_owner<int64_t>(); }
  SECTION("unsigned") { check_public_block_sparse_owner<uint64_t>(); }
}

TEST_CASE("public block sparse matrix-engine and hybrid owners preserve exact semantics") {
  if(!hip_available()) SKIP("no HIP device for block-sparse WMMA");
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  const std::string target=ctx->device_info.gcn_arch;
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  if(target.find("gfx1100")!=0 && target.find("gfx1101")!=0 && target.find("gfx1102")!=0)
    SKIP("block-sparse WMMA target not in compiled RDNA3 implementation");
  for(auto method:{RNS8_BLOCK_SPARSE_WMMA,RNS8_BLOCK_SPARSE_HYBRID}) {
    check_public_block_sparse_owner<int64_t>(method);
    check_public_block_sparse_owner<uint64_t>(method);
  }
}
