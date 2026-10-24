template <typename T>
void check_adaptive_native_product() {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr int m=35, n=33, k=17, lda=19, ldb=36, ldc=37, capacity=20;
  constexpr bool unsigned_inputs = std::is_unsigned_v<T>;
  constexpr int moduli[] = {256,255,253,251,247,239,233,229,227,223,217,211,199,197,193,191,181,179,173,167};
  std::vector<T> a(m*lda, 0), b(k*ldb, 0);
  for (int row=16; row<m; ++row)
    for (int q=0; q<k; ++q)
      a[row*lda+q] = row<32 ? T{2} : std::numeric_limits<T>::max();
  for (int q=0; q<k; ++q)
    for (int col=16; col<n; ++col)
      b[q*ldb+col] = col<32 ? T{3} : std::numeric_limits<T>::max();
  if constexpr (!unsigned_inputs) a[34*lda+1] = std::numeric_limits<T>::min();
  hip_direct_device_buffer da, db, dc, drhs, dchain;
  REQUIRE(da.allocate(0,a.size()*sizeof(T)) == RNS8_SUCCESS);
  REQUIRE(db.allocate(0,b.size()*sizeof(T)) == RNS8_SUCCESS);
  REQUIRE(dc.allocate(0,capacity*m*ldc) == RNS8_SUCCESS);
  REQUIRE(drhs.allocate(0,capacity*n*2) == RNS8_SUCCESS);
  REQUIRE(dchain.allocate(0,capacity*m*3) == RNS8_SUCCESS);
  std::vector<int8_t> rhs(capacity*n*2);
  for (int p=0; p<capacity; ++p)
    for (int q=0; q<n; ++q) {
      rhs[(p*n+q)*2]=1;
      rhs[(p*n+q)*2+1]=q%2 ? -1 : 1;
    }
  REQUIRE(hip_direct_copy_host_to_device(0,drhs.get(),rhs.data(),rhs.size()) == RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,db.get(),b.data(),b.size()*sizeof(T)) == RNS8_SUCCESS);
  auto centered = [](const cpp_int& x,int modulus) {
    int r = (x%modulus).convert_to<int>();
    if (r<0) r+=modulus;
    if (r>=(modulus+1)/2) r-=modulus;
    return static_cast<int8_t>(r);
  };
  for (int mutation=0; mutation<2; ++mutation) {
    if (mutation) a[0]=std::numeric_limits<T>::max();
    REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*sizeof(T)) == RNS8_SUCCESS);
    std::vector<cpp_int> exact(m*n);
    for (int row=0; row<m; ++row)
      for (int col=0; col<n; ++col)
        for (int q=0; q<k; ++q) exact[row*n+col]+=cpp_int(a[row*lda+q])*b[q*ldb+col];
    std::vector<int8_t> expected(capacity*m*ldc,99), chain_expected(capacity*m*3,99);
    for (int p=0; p<capacity; ++p)
      for (int row=0; row<m; ++row) {
        cpp_int sum=0, alternating=0;
        for (int col=0; col<n; ++col) {
          expected[(p*m+row)*ldc+col]=centered(exact[row*n+col],moduli[p]);
          sum+=exact[row*n+col];
          alternating+=exact[row*n+col]*(col%2 ? -1 : 1);
        }
        chain_expected[(p*m+row)*3]=centered(sum,moduli[p]);
        chain_expected[(p*m+row)*3+1]=centered(alternating,moduli[p]);
      }
    for (auto method : {NativeBoundMethod::tile_max,NativeBoundMethod::axis_l1,NativeBoundMethod::panel_l1})
      for (uint32_t group : {1u,2u,4u})
        for (uint32_t workers : {0u, 3u}) {
        std::vector<int8_t> actual(expected.size(),99), chain(chain_expected.size(),99);
        REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),actual.data(),actual.size()) == RNS8_SUCCESS);
        bool committed=true;
        const auto result=hip_direct_gemm_adaptive_native_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),
            m,n,k,lda,ldb,ldc,unsigned_inputs,capacity,group,1<<20,committed,method,7,workers);
        if (result==RNS8_UNSUPPORTED_BACKEND) {
          REQUIRE_FALSE(committed);
          SKIP("compiled target lacks native RDNA3 WMMA");
        }
        REQUIRE(result==RNS8_SUCCESS);
        REQUIRE(committed);
        REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size()) == RNS8_SUCCESS);
        REQUIRE(actual==expected);
        // The next contraction crosses 16-column precision boundaries.
        REQUIRE(hip_direct_copy_host_to_device(0,dchain.get(),chain.data(),chain.size()) == RNS8_SUCCESS);
        REQUIRE(hip_direct_gemm_rns_device(0,dc.get(),drhs.get(),dchain.get(),m,2,n,ldc,2,3,capacity)==RNS8_SUCCESS);
        REQUIRE(hip_direct_copy_device_to_host(0,chain.data(),dchain.get(),chain.size()) == RNS8_SUCCESS);
        REQUIRE(chain==chain_expected);
        committed = true;
        REQUIRE(hip_direct_gemm_adaptive_native_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),
            m,n,k,lda,ldb,ldc,unsigned_inputs,capacity,group,1<<20,committed,method,7,65536)
            == RNS8_INVALID_ARGUMENT);
        REQUIRE_FALSE(committed);
        for (bool insufficient_capacity : {false,true}) {
          committed=true;
          REQUIRE(hip_direct_gemm_adaptive_native_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),
              m,n,k,lda,ldb,ldc,unsigned_inputs,insufficient_capacity?1:capacity,group,
              insufficient_capacity?1<<20:0,committed,method,7,workers)==
              (insufficient_capacity?RNS8_RANGE_ERROR:RNS8_WORKSPACE_TOO_SMALL));
          REQUIRE_FALSE(committed);
          REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size()) == RNS8_SUCCESS);
          REQUIRE(actual==expected);
        }
      }
  }
}

TEST_CASE("adaptive native products complete every residue plane before chaining") {
  if (!hip_available()) SKIP("no HIP device for adaptive native product");
  check_adaptive_native_product<int64_t>();
  check_adaptive_native_product<uint64_t>();
}

template<class T> void check_public_adaptive_owner() {
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
  md.flags = RNS8_MATRIX_RETAIN_NATIVE_INPUT;
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
    const auto before=rns8::detail::hip_direct_allocation_counters_snapshot();
    rns8_status status;
    if constexpr (std::is_unsigned_v<T>) status=rns8_pack_u64(ctx, matrix, data.data(), 19, 0);
    else status=rns8_pack_i64(ctx, matrix, data.data(), 19, 0);
    const auto after=rns8::detail::hip_direct_allocation_counters_snapshot();
    if(status==RNS8_SUCCESS) {
      CHECK(after.h2d_calls-before.h2d_calls==1);
      CHECK(after.h2d_bytes-before.h2d_bytes==17*17*sizeof(T));
    }
    return status;
  };
  const auto run = [&](rns8_matrix* lhs, uint64_t budget, uint32_t workers) {
    return rns8_gemm_adaptive_exact(ctx, plan, lhs, b, c, RNS8_ADAPTIVE_AXIS_L1,
                                    256, 2, workers, budget, workspace);
  };
  REQUIRE(pack(a,av) == RNS8_SUCCESS);
  REQUIRE(pack(b,bv) == RNS8_SUCCESS);
  REQUIRE(pack(c,av) == RNS8_SUCCESS);
  // Both padded and compact retained packs use one native allocation and no
  // second full input upload staging allocation.
  const auto check_single_upload_owner = [&](rns8_matrix* matrix) {
    CHECK(matrix->residues.empty());
    CHECK(matrix->residues.capacity() == 0);
    CHECK_FALSE(matrix->host_residues_current);
    CHECK(matrix->hip_upload_buffer == nullptr);
    CHECK(matrix->hip_upload_bytes == 0);
    CHECK(matrix->device_native_current);
    CHECK(matrix->device_residues_current);
    if constexpr (std::is_unsigned_v<T>) {
      CHECK(matrix->hip_native_u64 != nullptr);
      CHECK(matrix->hip_native_u64_bytes == 17*17*sizeof(T));
      CHECK(matrix->hip_native_i64 == nullptr);
    } else {
      CHECK(matrix->hip_native_i64 != nullptr);
      CHECK(matrix->hip_native_i64_bytes == 17*17*sizeof(T));
      CHECK(matrix->hip_native_u64 == nullptr);
    }
  };
  check_single_upload_owner(a);
  check_single_upload_owner(b);
  check_single_upload_owner(c);
  const auto initial_generation = c->content_generation;
  CHECK(run(a,0,3) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(c->content_generation == initial_generation);
  CHECK(c->device_native_current);
  CHECK(run(a,UINT64_MAX,65536) == RNS8_INVALID_ARGUMENT);
  for (int mutation=0; mutation<2; ++mutation) {
    if (mutation) {
      av[0] = std::numeric_limits<T>::max();
      REQUIRE(pack(a,av) == RNS8_SUCCESS);
    }
    for (uint32_t workers : {0u,3u}) {
      const auto status = run(a,UINT64_MAX,workers);
      if (status == RNS8_UNSUPPORTED_BACKEND) {
        rns8_destroy_matrix(c); rns8_destroy_matrix(b); rns8_destroy_matrix(a);
        rns8_destroy_workspace(workspace); rns8_destroy_plan(plan); rns8_destroy_context(ctx);
        SKIP("compiled target lacks adaptive native WMMA");
      }
      REQUIRE(status == RNS8_SUCCESS);
      CHECK_FALSE(c->device_native_current);
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
  // An ordinary resident product replaces A and must retire its native materialization.
  REQUIRE(rns8_gemm_rns(ctx,plan,b,b,a,workspace) == RNS8_SUCCESS);
  CHECK_FALSE(a->device_native_current);
  const auto generation = c->content_generation;
  CHECK(run(a,UINT64_MAX,3) == RNS8_INVALID_ARGUMENT);
  CHECK(c->content_generation == generation);
  std::vector<T> compact(17*17);
  for (int row=0; row<17; ++row)
    std::copy_n(av.data()+row*19,17,compact.data()+row*17);
  const auto compact_before=rns8::detail::hip_direct_allocation_counters_snapshot();
  if constexpr (std::is_unsigned_v<T>)
    REQUIRE(rns8_pack_u64(ctx,a,compact.data(),17,0) == RNS8_SUCCESS);
  else
    REQUIRE(rns8_pack_i64(ctx,a,compact.data(),17,0) == RNS8_SUCCESS);
  const auto compact_after=rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(compact_after.h2d_calls-compact_before.h2d_calls==1);
  CHECK(compact_after.h2d_bytes-compact_before.h2d_bytes==17*17*sizeof(T));
  check_single_upload_owner(a);
  REQUIRE(run(a,UINT64_MAX,3) == RNS8_SUCCESS);
  rns8_matrix_allocation_info before{};
  before.struct_size=sizeof(before); before.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_get_matrix_allocation_info(c,&before)==RNS8_SUCCESS);
  CHECK(before.host_payload_capacity_bytes==0);
  CHECK(before.device_export_staging_bytes>0);
  const auto value_generation=c->content_generation;
  auto* retained_native=c->hip_native_i64 ? c->hip_native_i64 : c->hip_native_u64;
  c->prepared_references=1;
  CHECK(rns8_trim_matrix_staging(c)==RNS8_BUSY);
  c->prepared_references=0;
  REQUIRE(rns8_trim_matrix_staging(c)==RNS8_SUCCESS);
  rns8_matrix_allocation_info after=before;
  REQUIRE(rns8_get_matrix_allocation_info(c,&after)==RNS8_SUCCESS);
  CHECK(after.device_export_staging_bytes==0);
  CHECK(after.device_upload_staging_bytes==0);
  CHECK(after.device_status_and_schedule_bytes==0);
  CHECK(after.device_payload_capacity_bytes==before.device_payload_capacity_bytes);
  CHECK(c->content_generation==value_generation);
  CHECK((c->hip_native_i64 ? c->hip_native_i64 : c->hip_native_u64)==retained_native);
  // Rebuilding export staging must preserve the complete exact value.
  std::vector<uint64_t> rebuilt(17*17*3);
  if constexpr (std::is_unsigned_v<T>)
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx,plan,c,rebuilt.data(),17,3)==RNS8_SUCCESS);
  else REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c,rebuilt.data(),17,3)==RNS8_SUCCESS);
  for (int r=0;r<17;++r) for (int col=0;col<17;++col) {
    cpp_int expected=0;
    for (int q=0;q<17;++q) expected+=cpp_int(av[r*19+q])*bv[q*19+col];
    const int at=(r*17+col)*3;
    cpp_int actual=cpp_int(rebuilt[at])+(cpp_int(rebuilt[at+1])<<64)+(cpp_int(rebuilt[at+2])<<128);
    if constexpr (!std::is_unsigned_v<T>) if (rebuilt[at+2]>>63) actual-=cpp_int(1)<<192;
    REQUIRE(actual==expected);
  }
  rns8_destroy_matrix(c); rns8_destroy_matrix(b); rns8_destroy_matrix(a);
  rns8_destroy_workspace(workspace); rns8_destroy_plan(plan); rns8_destroy_context(ctx);
}
TEST_CASE("public adaptive exact products own native inputs and retire stale copies") {
  if (!hip_available()) SKIP("no HIP device for public adaptive ownership");
  check_public_adaptive_owner<int64_t>();
  check_public_adaptive_owner<uint64_t>();
}

TEST_CASE("direct HIP matrices allocate only the requested device payload") {
  if (!hip_available()) SKIP("no HIP device for storage admission");
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  for (auto semantics : {RNS8_BOUNDED_I64,RNS8_BOUNDED_U64,
       RNS8_EXACT_WIDE_SIGNED,RNS8_EXACT_WIDE_UNSIGNED,
       RNS8_FINITE_RING_U8,RNS8_FINITE_FIELD_U8,RNS8_WRAP_U64_MOD_2_64}) {
    const bool bounded=semantics==RNS8_BOUNDED_I64 || semantics==RNS8_BOUNDED_U64;
    auto md=matrix_desc(2,3,semantics,bounded?RNS8_BOUND_GLOBAL_MAX_ABS:RNS8_BOUND_NONE);
    rns8_matrix* matrix=nullptr;
    REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
    CHECK(matrix->residues.capacity()==0);
    CHECK(matrix->byte_limbs.capacity()==0);
    CHECK_FALSE(matrix->host_residues_current);
    CHECK_FALSE(matrix->device_residues_current);
    CHECK_FALSE(matrix->host_byte_limbs_current);
    CHECK_FALSE(matrix->device_byte_limbs_current);
    rns8_matrix_storage_info info{};
    info.struct_size=sizeof(info); info.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_get_matrix_storage_info(matrix,&info)==RNS8_SUCCESS);
    CHECK(info.host_residue_bytes==0);
    CHECK(info.host_byte_limb_bytes==0);
    const int64_t signed_values[]={INT64_MIN,0,INT64_MAX,1,-1,2};
    const uint64_t unsigned_values[]={UINT64_MAX,0,1,2,3,4};
    const uint8_t finite_values[]={0,1,249,2,3,4};
    if (semantics==RNS8_FINITE_RING_U8 || semantics==RNS8_FINITE_FIELD_U8)
      REQUIRE(rns8_pack_finite_u8(ctx,matrix,semantics==RNS8_FINITE_RING_U8?250:251,
                                 finite_values,3,0)==RNS8_SUCCESS);
    else if (semantics==RNS8_BOUNDED_I64 || semantics==RNS8_EXACT_WIDE_SIGNED)
      REQUIRE(rns8_pack_i64(ctx,matrix,signed_values,3,0)==RNS8_SUCCESS);
    else REQUIRE(rns8_pack_u64(ctx,matrix,unsigned_values,3,0)==RNS8_SUCCESS);
    CHECK(matrix->residues.capacity()==0);
    CHECK(matrix->byte_limbs.capacity()==0);
    REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
