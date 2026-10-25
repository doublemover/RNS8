TEST_CASE("u64-index residue control matches Boost across prefixes tails and K slabs") {
  if (!hip_available()) SKIP("no HIP device for wide-index residue control");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (int k : {33,65535,65536,65537}) {
    const int m=k==33?17:1,n=k==33?19:1;
    const int lda=k+2,ldb=n+3,ldc=n+4;
    for (uint32_t prefix=1; prefix<=20; ++prefix) {
      if (k!=33 && prefix!=20) continue;
      std::vector<int8_t> a(prefix*m*lda,91),b(prefix*k*ldb,92);
      std::vector<int8_t> expected((prefix+1)*m*ldc,93),actual(expected);
      for (uint32_t p=0; p<prefix; ++p) {
        const int modulus=rns8_default_modulus(p);
        for (int row=0; row<m; ++row) for (int q=0; q<k; ++q)
          a[(p*m+row)*lda+q]=static_cast<int8_t>((row+q)%2 ? -128 : 127);
        for (int q=0; q<k; ++q) for (int col=0; col<n; ++col)
          b[(p*k+q)*ldb+col]=static_cast<int8_t>((q+col)%3 ? -128 : 127);
        for (int row=0; row<m; ++row) for (int col=0; col<n; ++col) {
          cpp_int sum=0;
          for (int q=0; q<k; ++q) sum+=int(a[(p*m+row)*lda+q])*int(b[(p*k+q)*ldb+col]);
          int value=(sum%modulus).convert_to<int>();
          if (value<0) value+=modulus;
          if (value>=(modulus+1)/2) value-=modulus;
          expected[(p*m+row)*ldc+col]=static_cast<int8_t>(value);
        }
      }
      hip_direct_device_buffer da,db,dc;
      REQUIRE(da.allocate(0,a.size())==RNS8_SUCCESS);
      REQUIRE(db.allocate(0,b.size())==RNS8_SUCCESS);
      REQUIRE(dc.allocate(0,actual.size())==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size())==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,db.get(),b.data(),b.size())==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),actual.data(),actual.size())==RNS8_SUCCESS);
      CHECK(hip_direct_gemm_rns_wide_device(0,da.get(),db.get(),da.get(),m,n,k,lda,ldb,ldc,prefix)==RNS8_INVALID_ARGUMENT);
      REQUIRE(hip_direct_gemm_rns_wide_device(0,da.get(),db.get(),dc.get(),m,n,k,lda,ldb,ldc,prefix)==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size())==RNS8_SUCCESS);
      REQUIRE(actual==expected);
    }
  }
  // Huge unused final pitch is valid without a huge allocation; all accessed
  // addresses remain zero here. This checks argument width, not >2GiB traffic.
  int8_t a=7,b=-9,c=0;
  hip_direct_device_buffer da,db,dc;
  REQUIRE(da.allocate(0,1)==RNS8_SUCCESS);
  REQUIRE(db.allocate(0,1)==RNS8_SUCCESS);
  REQUIRE(dc.allocate(0,1)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,da.get(),&a,1)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,db.get(),&b,1)==RNS8_SUCCESS);
  REQUIRE(hip_direct_gemm_rns_wide_device(0,da.get(),db.get(),dc.get(),1,1,1,INT64_MAX,INT64_MAX,INT64_MAX,1)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_device_to_host(0,&c,dc.get(),1)==RNS8_SUCCESS);
  CHECK(c==-63);
}

TEST_CASE("finite address-width controls match Boost and preserve rows outside each window") {
  if(!hip_available())SKIP("no HIP device for finite index-width controls");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for(int k:{33,65535,65536,65537})for(int modulus:{2,127,253,256}) {
    const int m=k==33?17:3,n=k==33?19:2,lda=k+2,ldb=n+3,ldc=n+4;
    std::vector<int8_t> a(m*lda,91),b(k*ldb,92),expected(m*ldc,93),actual(expected);
    for(int row=0;row<m;++row)for(int q=0;q<k;++q)a[row*lda+q]=int8_t((row+3*q)%modulus-modulus/2);
    for(int q=0;q<k;++q)for(int col=0;col<n;++col)b[q*ldb+col]=int8_t((q+5*col)%modulus-modulus/2);
    for(int row=1;row<m-1;++row)for(int col=0;col<n;++col) {
      cpp_int sum=0;for(int q=0;q<k;++q)sum+=int(a[row*lda+q])*int(b[q*ldb+col]);
      int value=(sum%modulus).convert_to<int>();if(value<0)value+=modulus;if(value>=(modulus+1)/2)value-=modulus;
      expected[row*ldc+col]=int8_t(value);
    }
    hip_direct_device_buffer da,db,dc;
    REQUIRE(da.allocate(0,a.size())==RNS8_SUCCESS);REQUIRE(db.allocate(0,b.size())==RNS8_SUCCESS);REQUIRE(dc.allocate(0,actual.size())==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size())==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,db.get(),b.data(),b.size())==RNS8_SUCCESS);
    for(auto width:{FiniteGemmAddressWidth::signed32,FiniteGemmAddressWidth::unsigned64}) {
      std::fill(actual.begin(),actual.end(),93);
      REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),actual.data(),actual.size())==RNS8_SUCCESS);
      REQUIRE(hip_direct_gemm_finite_indexed_window_device(0,da.get(),db.get(),da.get(),m,n,k,lda,ldb,ldc,uint16_t(modulus),1,m-2,width)==RNS8_INVALID_ARGUMENT);
      da.mark_pending();db.mark_pending();dc.mark_pending();
      const auto status=hip_direct_gemm_finite_indexed_window_device(0,da.get(),db.get(),dc.get(),m,n,k,lda,ldb,ldc,uint16_t(modulus),1,m-2,width);
      if(status==RNS8_SUCCESS){da.mark_completed();db.mark_completed();dc.mark_completed();}
      REQUIRE(status==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size())==RNS8_SUCCESS);
      REQUIRE(actual==expected);
    }
  }
}
