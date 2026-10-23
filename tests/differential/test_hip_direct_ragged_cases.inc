TEST_CASE("HIP ragged exact extension feeds a GEMM crossing source tile boundaries") {
  if (!hip_available()) SKIP("no HIP device for ragged exact materialization");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (bool signed_values : {false,true}) {
    auto f=make_ragged_fixture(signed_values);
    ResidueLayout target;
    REQUIRE(make_residue_layout(ResidueLayoutKind::modulus_major,3,5,7,20,0,0,target));
    std::vector<int8_t> expected(target.bytes,99), actual=expected;
    REQUIRE(materialize_ragged_residues_host(f.bytes.data(),f.layout,expected.data(),target,71,9,target.bytes)==RNS8_SUCCESS);
    void *source=nullptr, *output=nullptr, *db=nullptr, *dc=nullptr;
    REQUIRE(hip_direct_allocate(0,f.bytes.size(),&source)==RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0,target.bytes,&output)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,source,f.bytes.data(),f.bytes.size())==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,output,actual.data(),actual.size())==RNS8_SUCCESS);
    bool commit=false;
    REQUIRE(hip_direct_materialize_ragged_residues(0,source,f.layout,output,target,71,9,4096,commit)==RNS8_SUCCESS);
    REQUIRE(commit);
    REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),output,actual.size())==RNS8_SUCCESS);
    REQUIRE(actual==expected);
    REQUIRE(hip_direct_materialize_ragged_residues(0,source,f.layout,output,target,71,10,4096,commit)==RNS8_INVALID_ARGUMENT);
    REQUIRE_FALSE(commit);
    auto bad=f.layout; bad.tiles[0].magnitude_bound[0]=0;
    REQUIRE(hip_direct_materialize_ragged_residues(0,source,bad,output,target,71,9,4096,commit)==RNS8_RANGE_ERROR);
    REQUIRE_FALSE(commit);
    REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),output,actual.size())==RNS8_SUCCESS);
    REQUIRE(actual==expected);
    std::vector<int8_t> b(20*5*2), c(20*3*2);
    for(uint32_t p=0;p<20;++p) for(int q=0;q<5;++q) for(int col=0;col<2;++col)
      b[(p*5+q)*2+col]=centered_residue(cpp_int(q+col+1),kDefaultModuli[p]);
    REQUIRE(hip_direct_allocate(0,b.size(),&db)==RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0,c.size(),&dc)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,db,b.data(),b.size())==RNS8_SUCCESS);
    REQUIRE(hip_direct_gemm_rns_device(0,output,db,dc,3,2,5,7,2,2,20)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_device_to_host(0,c.data(),dc,c.size())==RNS8_SUCCESS);
    for(int row=0;row<3;++row) for(int col=0;col<2;++col) {
      cpp_int sum=0;
      for(int q=0;q<5;++q) sum+=f.values[row*5+q]*(q+col+1);
      for(uint32_t p=0;p<20;++p) REQUIRE(c[(p*3+row)*2+col]==centered_residue(sum,kDefaultModuli[p]));
    }
    CHECK(hip_direct_free(0,dc)==RNS8_SUCCESS);
    CHECK(hip_direct_free(0,db)==RNS8_SUCCESS);
    CHECK(hip_direct_free(0,output)==RNS8_SUCCESS);
    CHECK(hip_direct_free(0,source)==RNS8_SUCCESS);
  }
}
