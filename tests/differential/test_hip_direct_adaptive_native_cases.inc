#include "backend_hip_direct/native_global_minimum.hpp"
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

template<class T> void check_public_adaptive_owner(bool cached=false,
    rns8_adaptive_bound_method method=RNS8_ADAPTIVE_AXIS_L1) {
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
    const auto operation=cached ? rns8_gemm_adaptive_exact_cached : rns8_gemm_adaptive_exact;
    return operation(ctx,plan,lhs,b,c,method,16,2,workers,budget,workspace);
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
      if (cached) {
        rns8_adaptive_summary_info info{};
        info.struct_size=sizeof(info); info.abi_version=RNS8_ABI_VERSION;
        REQUIRE(rns8_get_adaptive_summary_info(workspace,&info)==RNS8_SUCCESS);
        CHECK(info.a_instance==a->matrix_instance_id);
        CHECK(info.a_generation==a->content_generation);
        CHECK(info.b_instance==b->matrix_instance_id);
        CHECK(info.b_generation==b->content_generation);
        CHECK(info.a_scans==uint64_t(mutation+1));
        CHECK(info.b_scans==1);
        CHECK(info.a_reuses==uint64_t(mutation+(workers ? 1 : 0)));
        CHECK(info.b_reuses==uint64_t(2*mutation+(workers ? 1 : 0)));
        const uint64_t bytes=4*(method==RNS8_ADAPTIVE_TILE_MAX ? sizeof(uint64_t) : sizeof(rns8::detail::NativeAxisSummary))*
            (method==RNS8_ADAPTIVE_PANEL_L1 ? 2 : 1);
        CHECK(info.retained_bytes==bytes);
        const auto generation=c->content_generation;
        CHECK(run(a,bytes-1,3)==RNS8_WORKSPACE_TOO_SMALL);
        CHECK(c->content_generation==generation);
      }
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
  if (cached) {
    const auto info = [&]() {
      rns8_adaptive_summary_info value{};
      value.struct_size=sizeof(value); value.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_get_adaptive_summary_info(workspace,&value)==RNS8_SUCCESS);
      return value;
    };
    const auto before=info();
    bv[0]=T(7);
    REQUIRE(pack(b,bv)==RNS8_SUCCESS);
    REQUIRE(run(a,UINT64_MAX,3)==RNS8_SUCCESS);
    const auto after=info();
    CHECK(after.a_scans==before.a_scans);
    CHECK(after.b_scans==before.b_scans+1);
    CHECK(after.a_reuses==before.a_reuses+1);
    CHECK(after.b_generation==b->content_generation);
    // Policy changes invalidate both roles even if the native generations did
    // not change; their payload has a different arithmetic interpretation.
    const auto alternate=method==RNS8_ADAPTIVE_TILE_MAX ? RNS8_ADAPTIVE_AXIS_L1 : RNS8_ADAPTIVE_TILE_MAX;
    REQUIRE(rns8_gemm_adaptive_exact_cached(ctx,plan,a,b,c,alternate,16,2,3,UINT64_MAX,workspace)==RNS8_SUCCESS);
    const auto switched=info();
    CHECK(switched.a_scans==after.a_scans+1);
    CHECK(switched.b_scans==after.b_scans+1);
    REQUIRE(run(a,UINT64_MAX,3)==RNS8_SUCCESS);
    rns8_matrix* replacement=nullptr;
    REQUIRE(rns8_create_matrix(ctx,&md,&replacement)==RNS8_SUCCESS);
    REQUIRE(pack(replacement,av)==RNS8_SUCCESS);
    const auto previous=info();
    REQUIRE(run(replacement,UINT64_MAX,3)==RNS8_SUCCESS);
    const auto replaced=info();
    CHECK(replaced.a_instance==replacement->matrix_instance_id);
    CHECK(replaced.a_scans==previous.a_scans+1);
    CHECK(replaced.b_scans==previous.b_scans);
    REQUIRE(rns8_destroy_matrix(replacement)==RNS8_SUCCESS);
  }
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

TEST_CASE("public adaptive summaries reuse each unchanged axis and refresh mutated generations", "[hip][native][bounds][summary]") {
  if (!hip_available()) SKIP("no HIP device for retained adaptive summaries");
  for (const auto method : {RNS8_ADAPTIVE_TILE_MAX,RNS8_ADAPTIVE_AXIS_L1,RNS8_ADAPTIVE_PANEL_L1}) {
    check_public_adaptive_owner<int64_t>(true,method);
    check_public_adaptive_owner<uint64_t>(true,method);
  }
}

TEST_CASE("direct HIP matrices allocate only the requested device payload") {
  if (!hip_available()) SKIP("no HIP device for storage admission");
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  for (auto semantics : {RNS8_BOUNDED_I64,RNS8_BOUNDED_U64,
       RNS8_EXACT_WIDE_SIGNED,RNS8_EXACT_WIDE_UNSIGNED,
       RNS8_FINITE_RING_U8,RNS8_FINITE_FIELD_U8,RNS8_WRAP_U64_MOD_2_64}) {
    const bool bounded=semantics==RNS8_BOUNDED_I64 || semantics==RNS8_BOUNDED_U64;
    auto md=matrix_desc(2,3,semantics,bounded?(semantics==RNS8_BOUNDED_I64?RNS8_BOUND_GLOBAL_MAX_ABS:RNS8_BOUND_GLOBAL_MAX_UNSIGNED):RNS8_BOUND_NONE);
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

#include "../../benchmarks/native_dataset.hpp"
TEST_CASE("adaptive grid and compact complete versioned heterogeneous datasets", "[hip][native][bounds]") {
  if(!hip_available())SKIP("no HIP device for versioned adaptive datasets");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr uint32_t m=17,n=19,k=17,ldc=21;
  for(bool u:{false,true})for(const std::string profile:{"full-width-v1","uniform-small-v2","mixed-tiles-v2","zero-tiles-v2","localized-outlier-v2"}) {
    std::vector<uint64_t> a(m*k),b(k*n);std::mt19937_64 rng(3135);
    rns8::bench::fill_native_dataset(a,b,m,n,k,u,profile,3135,rng);
    const auto model=rns8::bench::model_native_dataset(a,b,m,n,k,u,1,profile);
    CHECK(model.fixed_capacity_group_work_items==80);
    auto native=[&](uint64_t bits)->cpp_int {return !u && (bits>>63)?cpp_int(bits)-(cpp_int(1)<<64):cpp_int(bits);};
    std::vector<int8_t> expected(20*m*ldc,99),actual(expected.size());
    for(uint32_t row=0;row<m;++row)for(uint32_t col=0;col<n;++col) {
      cpp_int value=0;for(uint32_t q=0;q<k;++q)value+=native(a[row*k+q])*native(b[q*n+col]);
      for(uint32_t p=0;p<20;++p) {
        const auto modulus=garner::kConstants.moduli[p];int residue=(value%modulus).convert_to<int>();
        if(residue<0)residue+=modulus;if(residue>=int((modulus+1)/2))residue-=modulus;
        expected[(p*m+row)*ldc+col]=static_cast<int8_t>(residue);
      }
    }
    hip_direct_device_buffer da,db,dc;
    REQUIRE(da.allocate(0,a.size()*8)==RNS8_SUCCESS);REQUIRE(db.allocate(0,b.size()*8)==RNS8_SUCCESS);
    REQUIRE(dc.allocate(0,expected.size())==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*8)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,db.get(),b.data(),b.size()*8)==RNS8_SUCCESS);
    for(uint32_t group:{1u,4u})for(uint32_t workers:{0u,3u}) {
      std::fill(actual.begin(),actual.end(),99);
      REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),actual.data(),actual.size())==RNS8_SUCCESS);
      bool committed=false;
      const auto status=hip_direct_gemm_adaptive_native_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),
        m,n,k,k,n,ldc,u,20,group,1<<20,committed,NativeBoundMethod::axis_l1,256,workers);
      if(status==RNS8_UNSUPPORTED_BACKEND)SKIP("compiled target lacks native WMMA");
      REQUIRE(status==RNS8_SUCCESS);REQUIRE(committed);
      REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size())==RNS8_SUCCESS);
      REQUIRE(actual==expected);
    }
  }
}

template<class T> void check_global_minimum_completed_native() {
  using namespace rns8::detail;using boost::multiprecision::cpp_int;
  constexpr int m=17,n=19,k=17,lda=19,ldb=22,ldc=23,capacity=20;
  constexpr bool uns=std::is_unsigned_v<T>;
  std::vector<T> a(m*lda),b(k*ldb);
  hip_direct_device_buffer da,db,dc;
  REQUIRE(da.allocate(0,a.size()*8)==RNS8_SUCCESS);REQUIRE(db.allocate(0,b.size()*8)==RNS8_SUCCESS);
  REQUIRE(dc.allocate(0,capacity*m*ldc)==RNS8_SUCCESS);
  auto mag=[](T value){cpp_int x=value;return x<0?cpp_int(-x):x;};
  for(unsigned mutation=0;mutation<3;++mutation) {
    for(int r=0;r<m;++r)for(int q=0;q<k;++q)a[r*lda+q]=mutation==2?T(0):mutation==1?std::numeric_limits<T>::max():T(r<16?2:7);
    for(int q=0;q<k;++q)for(int c=0;c<n;++c)b[q*ldb+c]=mutation==2?T(0):mutation==1?std::numeric_limits<T>::max():T(c<16?3:5);
    if constexpr(!uns)if(mutation==1)a[16*lda]=std::numeric_limits<T>::min();
    REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*8)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,db.get(),b.data(),b.size()*8)==RNS8_SUCCESS);
    cpp_int bound=0;
    for(int tr=0;tr<m;tr+=16)for(int tc=0;tc<n;tc+=16) {
      cpp_int am=0,al=0,bm=0,bl=0;
      for(int r=tr;r<std::min(m,tr+16);++r){cpp_int sum=0;for(int q=0;q<k;++q){auto v=mag(a[r*lda+q]);sum+=v;am=std::max(am,v);}al=std::max(al,sum);}
      for(int c=tc;c<std::min(n,tc+16);++c){cpp_int sum=0;for(int q=0;q<k;++q){auto v=mag(b[q*ldb+c]);sum+=v;bm=std::max(bm,v);}bl=std::max(bl,sum);}
      cpp_int tile=am*bm*k;tile=std::min(tile,cpp_int(am*bl));tile=std::min(tile,cpp_int(bm*al));bound=std::max(bound,tile);
    }
    uint32_t required=0;cpp_int product=1,threshold=bound*(uns?1:2);
    if(bound!=0)do{product*=rns8_default_modulus(required++);}while(product<=threshold);
    std::vector<int8_t> expected(capacity*m*ldc,99);
    for(int r=0;r<m;++r)for(int c=0;c<n;++c) {
      cpp_int exact=0;for(int q=0;q<k;++q)exact+=cpp_int(a[r*lda+q])*b[q*ldb+c];
      for(int plane=0;plane<capacity;++plane){const auto mod=rns8_default_modulus(plane);int v=(exact%mod).template convert_to<int>();if(v<0)v+=mod;if(v>=(mod+1)/2)v-=mod;expected[(plane*m+r)*ldc+c]=int8_t(v);}
    }
    for(uint32_t group:{1u,2u,4u}) {
      std::vector<int8_t> actual(expected.size(),99);REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),actual.data(),actual.size())==RNS8_SUCCESS);
      const auto support=query_native_global_minimum_support(m,n,k,lda,ldb,ldc,capacity,group,UINT64_MAX);
      auto* ledger=hip_direct_allocation_ledger(0);REQUIRE(ledger!=nullptr);
      const auto baseline=ledger->snapshot();REQUIRE(baseline.complete);uint64_t observation_epoch=0;REQUIRE(ledger->reset_peaks(observation_epoch));
      bool written=true;NativeGlobalMinimumReceipt receipt;
      const auto status=hip_direct_gemm_global_minimum_native_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),m,n,k,lda,ldb,ldc,uns,capacity,group,1<<20,written,receipt);
      if(status==RNS8_UNSUPPORTED_BACKEND){CHECK_FALSE(written);SKIP("global minimum control needs compiled RDNA3 native WMMA");}
      REQUIRE(status==RNS8_SUCCESS);CHECK(written);CHECK(receipt.required_prefix==required);CHECK(receipt.compute_prefix==std::max(1u,required));CHECK(receipt.proof_readback_bytes==36);
      const auto observed=ledger->snapshot();REQUIRE(observed.complete);
      CHECK(observed.domains[0].live_bytes==baseline.domains[0].live_bytes);
      CHECK(observed.domains[0].peak_bytes-baseline.domains[0].live_bytes<=support.peak_scratch_bytes);
      const cpp_int observed_bound=cpp_int(receipt.bound.magnitude[0])+(cpp_int(receipt.bound.magnitude[1])<<64)+(cpp_int(receipt.bound.magnitude[2])<<128);
      CHECK(observed_bound==bound);
      REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size())==RNS8_SUCCESS);REQUIRE(actual==expected);
      written=true;CHECK(hip_direct_gemm_global_minimum_native_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),m,n,k,lda,ldb,ldc,uns,capacity,group,support.peak_scratch_bytes-1,written,receipt)==RNS8_WORKSPACE_TOO_SMALL);CHECK_FALSE(written);
      if(required>1){written=true;CHECK(hip_direct_gemm_global_minimum_native_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),m,n,k,lda,ldb,ldc,uns,1,group,1<<20,written,receipt)==RNS8_RANGE_ERROR);CHECK_FALSE(written);}
      REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size())==RNS8_SUCCESS);CHECK(actual==expected);
      written=true;CHECK(hip_direct_gemm_global_minimum_native_device(0,da.get(),db.get(),static_cast<int8_t*>(da.get()),m,n,k,lda,ldb,ldc,uns,capacity,group,1<<20,written,receipt)==RNS8_INVALID_ARGUMENT);CHECK_FALSE(written);
    }
  }
}
TEST_CASE("global minimum device proof completes full output and refreshes after native mutation") {
  if(!hip_available())SKIP("no HIP device for global minimum native control");
  check_global_minimum_completed_native<int64_t>();check_global_minimum_completed_native<uint64_t>();
}

TEST_CASE("global minimum reduction includes a unique maximum beyond its first 256 certificates") {
  using namespace rns8::detail;using boost::multiprecision::cpp_int;
  if(!hip_available())SKIP("no HIP device for global minimum native control");
  constexpr int m=257,n=257,k=1,ldc=259,capacity=20;
  std::vector<uint64_t> a(m,1),b(n,1);a.back()=b.back()=UINT64_MAX;
  hip_direct_device_buffer da,db,dc;
  REQUIRE(da.allocate(0,a.size()*8)==RNS8_SUCCESS);REQUIRE(db.allocate(0,b.size()*8)==RNS8_SUCCESS);
  REQUIRE(dc.allocate(0,capacity*m*ldc)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*8)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,db.get(),b.data(),b.size()*8)==RNS8_SUCCESS);
  std::vector<int8_t> actual(capacity*m*ldc,99),expected=actual;
  REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),actual.data(),actual.size())==RNS8_SUCCESS);
  const cpp_int maximum=cpp_int(UINT64_MAX)*UINT64_MAX;
  uint32_t required=0;cpp_int product=1;do{product*=rns8_default_modulus(required++);}while(product<=maximum);
  for(int plane=0;plane<capacity;++plane) {
    const auto modulus=rns8_default_modulus(plane);
    auto residue=[&](const cpp_int& value){int v=(value%modulus).convert_to<int>();if(v>=(modulus+1)/2)v-=modulus;return int8_t(v);};
    const int8_t one=residue(cpp_int(1)),edge=residue(cpp_int(UINT64_MAX)),corner=residue(maximum);
    for(int r=0;r<m;++r)for(int c=0;c<n;++c)expected[(plane*m+r)*ldc+c]=r==m-1?(c==n-1?corner:edge):(c==n-1?edge:one);
  }
  bool written=true;NativeGlobalMinimumReceipt receipt;
  const auto status=hip_direct_gemm_global_minimum_native_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),m,n,k,1,n,ldc,true,capacity,4,1<<24,written,receipt);
  if(status==RNS8_UNSUPPORTED_BACKEND){CHECK_FALSE(written);SKIP("global minimum control needs compiled RDNA3 native WMMA");}
  REQUIRE(status==RNS8_SUCCESS);CHECK(written);CHECK(receipt.required_prefix==required);
  const cpp_int observed=cpp_int(receipt.bound.magnitude[0])+(cpp_int(receipt.bound.magnitude[1])<<64)+(cpp_int(receipt.bound.magnitude[2])<<128);
  CHECK(observed==maximum);
  REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size())==RNS8_SUCCESS);CHECK(actual==expected);
}
