#include "backend_hip_direct/encoded_b_backend.hpp"
template <typename NativeT>
bool check_native_wmma_groups(bool unsigned_inputs, int m, int n, int k, uint32_t prefix,
                              bool generated = false, uint32_t splits = 0,
                              uint32_t tiled_depth = 0, uint32_t tiled_buffers = 0, bool encoded = false) {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  const int lda = k + 1, ldb = n + 2, ldc = n + 3;
  const NativeT lo = std::numeric_limits<NativeT>::min(), hi = std::numeric_limits<NativeT>::max();
  const NativeT values[] = {lo, hi, NativeT{0}, NativeT{1}, NativeT{127}, NativeT{128}, NativeT{255},
      NativeT{UINT32_MAX}, NativeT{uint64_t{1} << 32}, NativeT{(uint64_t{1} << 32) + 1},
      NativeT{0x123456789abcdefULL}, static_cast<NativeT>(lo + 1), static_cast<NativeT>(hi - 1)};
  constexpr auto value_count = sizeof(values) / sizeof(values[0]);
  std::vector<NativeT> a(m * lda, hi), b(k * ldb, hi);
  for (int row = 0; row < m; ++row)
    for (int q = 0; q < k; ++q) a[row * lda + q] = values[(row * 3 + q) % value_count];
  for (int q = 0; q < k; ++q)
    for (int col = 0; col < n; ++col) b[q * ldb + col] = values[(q * 5 + col) % value_count];
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
  hip_direct_device_buffer encoded_storage;
  auto encoded_status=RNS8_SUCCESS;
  if(encoded) {
    const auto layout=encoded_b_layout(k,n,ldb,prefix,UINT64_MAX);
    REQUIRE(layout.status==RNS8_SUCCESS);
    REQUIRE(encoded_storage.allocate(0,layout.total_bytes)==RNS8_SUCCESS);
    REQUIRE(hip_prepare_encoded_b(0,db,encoded_storage.get(),k,n,ldb,prefix,unsigned_inputs,layout.total_bytes-1)
        ==RNS8_WORKSPACE_TOO_SMALL);
    encoded_status=hip_prepare_encoded_b(0,db,encoded_storage.get(),k,n,ldb,prefix,unsigned_inputs,layout.total_bytes);
  }
  bool supported = true;
  for (uint32_t group : {1u, 2u, 4u}) {
    std::fill(actual.begin(), actual.end(), 99);
    REQUIRE(hip_direct_copy_host_to_device(0, dc, actual.data(), actual.size()) == RNS8_SUCCESS);
    auto status = RNS8_UNSUPPORTED_BACKEND;
    if(encoded) {
      bool started=false;
      status=encoded_status==RNS8_SUCCESS?hip_gemm_encoded_b(0,da,encoded_storage.get(),dc,
          m,n,k,lda,ldc,prefix,group,unsigned_inputs,started):encoded_status;
      CHECK(started==(status==RNS8_SUCCESS));
    } else if (tiled_depth) {
      bool commit = true;
      if (tiled_buffers) {
        REQUIRE(hip_direct_gemm_native_tiled_device(0, da, db, static_cast<int8_t*>(dc),
            m,n,k,lda,ldb,ldc,prefix,group,unsigned_inputs,tiled_depth,tiled_buffers,0,commit)
            == RNS8_WORKSPACE_TOO_SMALL);
        REQUIRE_FALSE(commit);
        REQUIRE(hip_direct_copy_device_to_host(0, actual.data(), dc, actual.size()) == RNS8_SUCCESS);
        REQUIRE(std::all_of(actual.begin(), actual.end(), [](int8_t x) { return x == 99; }));
      }
      status = hip_direct_gemm_native_tiled_device(0, da, db, static_cast<int8_t*>(dc),
          m,n,k,lda,ldb,ldc,prefix,group,unsigned_inputs,tiled_depth,tiled_buffers,UINT64_MAX,commit);
      REQUIRE(commit == (status == RNS8_SUCCESS));
    } else if (splits) {
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

TEST_CASE("executable RDNA3 fragment maps preserve complete tiles and masked edges") {
  if (!hip_available()) SKIP("no HIP device for executable fragment maps");
  for (const int extent : {1, 16, 17}) {
    if (!check_native_wmma_groups<int64_t>(false, extent, extent, 17, 20))
      SKIP("compiled device target lacks RDNA3 fragment instruction");
    REQUIRE(check_native_wmma_groups<uint64_t>(true, extent, extent, 17, 20));
  }
}

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

TEST_CASE("multiwave native tiles preserve full-width residues across storage families") {
  if (!hip_available()) SKIP("no HIP device for multiwave native tiles");
  for (uint32_t buffers : {0u,1u,2u})
    for (uint32_t depth : {16u,32u}) {
      if (!buffers && depth==32) continue;
      for (uint32_t prefix=1; prefix<=20; ++prefix) {
        INFO("buffers=" << buffers << " depth=" << depth << " prefix=" << prefix);
        if (!check_native_wmma_groups<int64_t>(false,35,33,33,prefix,false,0,depth,buffers))
          SKIP("compiled target lacks multiwave RDNA3 WMMA");
        REQUIRE(check_native_wmma_groups<uint64_t>(true,35,33,33,prefix,false,0,depth,buffers));
      }
      for (int k : {65535,65536,65537}) {
        REQUIRE(check_native_wmma_groups<int64_t>(false,1,1,k,20,false,0,depth,buffers));
        REQUIRE(check_native_wmma_groups<uint64_t>(true,1,1,k,20,false,0,depth,buffers));
      }
    }
}

TEST_CASE("encoded B WMMA preserves prefix tails padding and K-slab boundaries") {
  if(!hip_available()) SKIP("no HIP device for encoded B");
  for(uint32_t prefix=1;prefix<=20;++prefix) {
    if(!check_native_wmma_groups<int64_t>(false,17,19,33,prefix,false,0,0,0,true))
      SKIP("compiled target lacks encoded B WMMA");
    REQUIRE(check_native_wmma_groups<uint64_t>(true,17,19,33,prefix,false,0,0,0,true));
  }
  for(int k:{65535,65536,65537}) {
    REQUIRE(check_native_wmma_groups<int64_t>(false,1,1,k,20,false,0,0,0,true));
    REQUIRE(check_native_wmma_groups<uint64_t>(true,1,1,k,20,false,0,0,0,true));
  }
}

template<class T> void check_public_encoded_b_owner() {
  using boost::multiprecision::cpp_int;
  constexpr bool u=std::is_unsigned_v<T>;
  constexpr int n=17;
  const auto semantics=u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  auto desc=exact_signed_desc(n,n,n,RNS8_BACKEND_HIP_DIRECT);
  desc.semantics=semantics; desc.max_prefix=20; desc.flags|=RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* plan=nullptr; rns8_workspace* ws=nullptr;
  REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx,plan,&ws)==RNS8_SUCCESS);
  auto md=matrix_desc(n,n,semantics,RNS8_BOUND_NONE);
  md.max_prefix=20; md.flags=RNS8_MATRIX_RETAIN_NATIVE_INPUT;
  rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr,*c2=nullptr;
  for(auto** matrix:{&a,&b,&c,&c2}) REQUIRE(rns8_create_matrix(ctx,&md,matrix)==RNS8_SUCCESS);
  std::vector<T> av(n*n,0),bv(n*n);
  for(int row=0;row<n;++row) {
    av[row*n+row]=2;
    for(int col=0;col<n;++col) bv[row*n+col]=(row+col)%2?std::numeric_limits<T>::max():std::numeric_limits<T>::min();
  }
  const auto pack=[&](rns8_matrix* matrix,const std::vector<T>& values) {
    if constexpr(u) return rns8_pack_u64(ctx,matrix,values.data(),n,0);
    else return rns8_pack_i64(ctx,matrix,values.data(),n,0);
  };
  REQUIRE(pack(a,av)==RNS8_SUCCESS); REQUIRE(pack(b,bv)==RNS8_SUCCESS); REQUIRE(pack(c,av)==RNS8_SUCCESS);
  rns8_encoded_b* cache=nullptr;
  const auto cleanup=[&]() {
    rns8_destroy_encoded_b(cache);
    for(auto* matrix:{a,b,c,c2}) rns8_destroy_matrix(matrix);
    rns8_destroy_workspace(ws); rns8_destroy_plan(plan); rns8_destroy_context(ctx);
  };
  CHECK(rns8_prepare_encoded_b(ctx,b,0,&cache)==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(cache==nullptr);
  const auto prepared=rns8_prepare_encoded_b(ctx,b,UINT64_MAX,&cache);
  if(prepared==RNS8_UNSUPPORTED_BACKEND) { cleanup(); SKIP("compiled target lacks encoded B WMMA"); }
  REQUIRE(prepared==RNS8_SUCCESS);
  CHECK(rns8_destroy_matrix(b)==RNS8_BUSY);
  const auto run=[&](uint32_t group) { return rns8_gemm_encoded_b_exact(ctx,plan,a,cache,c,group,ws); };
  const auto generation=c->content_generation;
  CHECK(run(3)==RNS8_INVALID_ARGUMENT);
  c->in_flight_readers=1; CHECK(run(1)==RNS8_BUSY); c->in_flight_readers=0;
  CHECK(c->content_generation==generation);
  for(uint32_t group:{1u,2u,4u}) {
    REQUIRE(run(group)==RNS8_SUCCESS);
    std::vector<uint64_t> limbs(n*n*3);
    if constexpr(u) REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx,plan,c,limbs.data(),n,3)==RNS8_SUCCESS);
    else REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c,limbs.data(),n,3)==RNS8_SUCCESS);
    for(int cell=0;cell<n*n;++cell) {
      cpp_int actual=cpp_int(limbs[cell*3])+(cpp_int(limbs[cell*3+1])<<64)+(cpp_int(limbs[cell*3+2])<<128);
      if constexpr(!u) if(limbs[cell*3+2]>>63) actual-=cpp_int(1)<<192;
      CHECK(actual==cpp_int(2)*bv[cell]);
    }
  }
  REQUIRE(pack(c,av)==RNS8_SUCCESS);
  REQUIRE(pack(c2,av)==RNS8_SUCCESS);
  rns8_encoded_b_task batch[2]{};
  for(auto& task:batch) {
    task.struct_size=sizeof(task); task.abi_version=RNS8_ABI_VERSION;
    task.plan=plan; task.a=a; task.b=cache; task.workspace=ws;
  }
  batch[0].group=1; batch[0].c=c; batch[1].group=4; batch[1].c=c2;
  const auto before0=c->content_generation,before1=c2->content_generation;
  CHECK(rns8_gemm_encoded_b_grouped(ctx,batch,2,3,0)==RNS8_WORKSPACE_TOO_SMALL);
  batch[1].a=c; // cross-task logical native/RNS alias despite disjoint byte allocations
  CHECK(rns8_gemm_encoded_b_grouped(ctx,batch,2,3,UINT64_MAX)==RNS8_INVALID_ARGUMENT);
  batch[1].a=a;
  CHECK(c->content_generation==before0); CHECK(c2->content_generation==before1);
  REQUIRE(rns8_gemm_encoded_b_grouped(ctx,batch,2,3,UINT64_MAX)==RNS8_SUCCESS);
  std::vector<uint64_t> left(n*n*3),right(n*n*3);
  if constexpr(u) {
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx,plan,c,left.data(),n,3)==RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx,plan,c2,right.data(),n,3)==RNS8_SUCCESS);
  } else {
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c,left.data(),n,3)==RNS8_SUCCESS);
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c2,right.data(),n,3)==RNS8_SUCCESS);
  }
  CHECK(left==right);
  bv[0]=3; REQUIRE(pack(b,bv)==RNS8_SUCCESS);
  const auto changed=c->content_generation;
  CHECK(run(1)==RNS8_INVALID_ARGUMENT);
  const auto c2_generation=c2->content_generation;
  CHECK(rns8_gemm_encoded_b_grouped(ctx,batch,2,3,UINT64_MAX)==RNS8_INVALID_ARGUMENT);
  CHECK(c2->content_generation==c2_generation);
  CHECK(c->content_generation==changed);
  REQUIRE(rns8_destroy_encoded_b(cache)==RNS8_SUCCESS); cache=nullptr;
  REQUIRE(rns8_prepare_encoded_b(ctx,b,UINT64_MAX,&cache)==RNS8_SUCCESS);
  REQUIRE(run(1)==RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx,plan,b,b,a,ws)==RNS8_SUCCESS);
  CHECK_FALSE(a->device_native_current);
  const auto preserved=c->content_generation;
  CHECK(run(1)==RNS8_INVALID_ARGUMENT);
  CHECK(c->content_generation==preserved);
  cleanup();
}
TEST_CASE("public encoded B binds native generations and exact output ownership") {
  if(!hip_available()) SKIP("no HIP device for public encoded B");
  SECTION("signed") { check_public_encoded_b_owner<int64_t>(); }
  SECTION("unsigned") { check_public_encoded_b_owner<uint64_t>(); }
}
