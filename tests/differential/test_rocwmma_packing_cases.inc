TEST_CASE("rocWMMA packing controls preserve exact channel products and call-local reuse") {
  using namespace rns8::detail;
  if(!rocwmma_available())SKIP("rocWMMA backend unavailable");
  struct Restore {RocwmmaPackingPolicy value=rocwmma_packing_policy();~Restore(){(void)rocwmma_set_packing_policy(value);}} restore;
  constexpr uint32_t prefix=2;
  for(int64_t k:{16,17,65535,65536,65537})for(int padding:{0,1}) {
    constexpr int64_t m=16,n=32;const int64_t lda=k+padding,ldb=n+3,ldc=n+2;
    std::vector<int8_t> a(prefix*m*lda,83),b(prefix*k*ldb,83),result(prefix*m*ldc,83);
    for(unsigned p=0;p<prefix;++p) {
      for(int64_t r=0;r<m;++r)for(int64_t q=0;q<k;++q)a[(p*m+r)*lda+q]=p?int8_t(127):int8_t(-128);
      for(int64_t q=0;q<k;++q)for(int64_t c=0;c<n;++c)b[(p*k+q)*ldb+c]=p?int8_t(-127):int8_t(-128);
    }
    HipBuffer da(a.size()),db(b.size()),dc(result.size());
    REQUIRE(hipMemcpy(da.get(),a.data(),a.size(),hipMemcpyHostToDevice)==hipSuccess);
    REQUIRE(hipMemcpy(db.get(),b.data(),b.size(),hipMemcpyHostToDevice)==hipSuccess);
    size_t a_bytes=0,b_bytes=0,bytes=0;
    REQUIRE(rocwmma_workspace_requirements(m,n,k,a_bytes,b_bytes,bytes));REQUIRE(bytes>0);HipBuffer workspace(bytes);
    HipBuffer prepacked_b(prefix*b_bytes);
    if(k<=65536)REQUIRE(rocwmma_prepack_b_rns_device(0,db.get(),prepacked_b.get(),prefix*b_bytes,k,n,ldb,prefix)==RNS8_SUCCESS);
    std::array<rns8_plan_tile_schedule_entry,2> entries{};
    for(unsigned i=0;i<2;++i) {
      entries[i].struct_size=sizeof(entries[i]);entries[i].abi_version=RNS8_ABI_VERSION;
      entries[i].row_extent=m;entries[i].col_extent=16;entries[i].col_offset=i*16;entries[i].selected_prefix=prefix;
    }
    for(auto policy:{RocwmmaPackingPolicy::packed,RocwmmaPackingPolicy::direct_a,
        RocwmmaPackingPolicy::reuse,RocwmmaPackingPolicy::direct_a_reuse}) {
      REQUIRE(rocwmma_set_packing_policy(policy));rocwmma_reset_packing_stats();
      REQUIRE(hipMemcpy(dc.get(),result.data(),result.size(),hipMemcpyHostToDevice)==hipSuccess);
      REQUIRE(rocwmma_gemm_rns_tiled_device(0,da.get(),db.get(),dc.get(),workspace.get(),bytes,
          m,n,k,lda,ldb,ldc,entries.data(),entries.size())==RNS8_SUCCESS);
      REQUIRE(hipMemcpy(result.data(),dc.get(),result.size(),hipMemcpyDeviceToHost)==hipSuccess);
      for(unsigned p=0;p<prefix;++p)for(int64_t r=0;r<m;++r)for(int64_t c=0;c<ldc;++c) {
        const boost::multiprecision::cpp_int expected=boost::multiprecision::cpp_int(k)*(p?127:-128)*(p?-127:-128);
        if(c<n)CHECK(result[(p*m+r)*ldc+c]==centered_residue(expected,kDefaultModuli[p]));
        else CHECK(result[(p*m+r)*ldc+c]==83);
      }
      const auto work=rocwmma_packing_stats();
      const bool direct=(uint32_t(policy)&1) && !padding && !(k%16);
      CHECK((work.direct_a_blocks>0)==direct);
      if(!(uint32_t(policy)&2))CHECK(work.a_panel_hits+work.b_panel_hits==0);
      if((uint32_t(policy)&2) && k<=65536 && !direct)CHECK(work.a_panel_hits==prefix);
      if(k<=65536) {
        const auto expected=result;
        REQUIRE(rocwmma_gemm_rns_prepacked_b_device(0,da.get(),prepacked_b.get(),dc.get(),workspace.get(),bytes,
            m,n,k,lda,ldc,prefix)==RNS8_SUCCESS);
        REQUIRE(hipMemcpy(result.data(),dc.get(),result.size(),hipMemcpyDeviceToHost)==hipSuccess);
        CHECK(result==expected);
      }
      // Mutate A between calls: no panel key is allowed to survive a call.
      a[0]=int8_t(a[0]+1);REQUIRE(hipMemcpy(da.get(),a.data(),a.size(),hipMemcpyHostToDevice)==hipSuccess);
      rocwmma_reset_packing_stats();
      REQUIRE(rocwmma_gemm_rns_device(0,da.get(),db.get(),dc.get(),workspace.get(),bytes,m,n,k,lda,ldb,ldc,prefix)==RNS8_SUCCESS);
      REQUIRE(hipMemcpy(result.data(),dc.get(),result.size(),hipMemcpyDeviceToHost)==hipSuccess);
      for(int64_t c=0;c<n;++c)
        CHECK(result[c]==centered_residue(boost::multiprecision::cpp_int(k)*16384+int(a[0]+128)*-128,256));
      a[0]=-128;REQUIRE(hipMemcpy(da.get(),a.data(),a.size(),hipMemcpyHostToDevice)==hipSuccess);
    }
  }
}
