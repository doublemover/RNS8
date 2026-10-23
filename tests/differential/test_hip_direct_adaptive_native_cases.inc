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
      for (uint32_t group : {1u,2u,4u}) {
        std::vector<int8_t> actual(expected.size(),99), chain(chain_expected.size(),99);
        REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),actual.data(),actual.size()) == RNS8_SUCCESS);
        bool committed=true;
        const auto result=hip_direct_gemm_adaptive_native_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),
            m,n,k,lda,ldb,ldc,unsigned_inputs,capacity,group,1<<20,committed,method,7);
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
        for (bool insufficient_capacity : {false,true}) {
          committed=true;
          REQUIRE(hip_direct_gemm_adaptive_native_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),
              m,n,k,lda,ldb,ldc,unsigned_inputs,insufficient_capacity?1:capacity,group,
              insufficient_capacity?1<<20:0,committed,method,7)==
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
