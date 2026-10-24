TEST_CASE("fixed dot4 import and canonical CRT match Boost for every prefix") {
  if(!hip_available()) SKIP("no HIP device for fixed conversion");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for(int cells:{1,17,257}) {
    std::vector<uint64_t> values(cells);
    const uint64_t edge[]={0,1,127,128,255,256,UINT64_MAX,UINT64_C(1)<<63,
      UINT64_C(0x8080808080808080),UINT64_C(0x7f8081ff007fff80)};
    for(int cell=0;cell<cells;++cell) values[cell]=edge[cell%10];
    hip_direct_device_buffer source,residues,limbs;
    REQUIRE(source.allocate(0,values.size()*8)==RNS8_SUCCESS);
    REQUIRE(residues.allocate(0,cells*20+16)==RNS8_SUCCESS);
    REQUIRE(limbs.allocate(0,cells*24+16)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,source.get(),values.data(),values.size()*8)==RNS8_SUCCESS);
    for(unsigned prefix=1;prefix<=20;++prefix) for(unsigned mode:{0u,1u}) {
      std::vector<int8_t> packed(cells*prefix+16,99);
      std::vector<uint64_t> output(cells*3+2,UINT64_C(0xabcdef));
      REQUIRE(hip_direct_copy_host_to_device(0,residues.get(),packed.data(),packed.size())==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,limbs.get(),output.data(),output.size()*8)==RNS8_SUCCESS);
      bool started=true;
      CHECK(hip_direct_fixed_conversion_device(0,source.get(),residues.get(),cells,prefix,mode,3,started)
            ==RNS8_WORKSPACE_TOO_SMALL);CHECK_FALSE(started);
      CHECK(hip_direct_fixed_conversion_device(0,source.get(),source.get(),cells,prefix,mode,4,started)
            ==RNS8_INVALID_ARGUMENT);CHECK_FALSE(started);
      const auto status=hip_direct_fixed_conversion_device(0,source.get(),residues.get(),cells,prefix,mode,4,started);
      if(status==RNS8_UNSUPPORTED_BACKEND) {CHECK_FALSE(started);SKIP("target lacks admitted dot4 conversion");}
      REQUIRE(status==RNS8_SUCCESS);REQUIRE(started);
      REQUIRE(hip_direct_copy_device_to_host(0,packed.data(),residues.get(),packed.size())==RNS8_SUCCESS);
      REQUIRE(hip_direct_fixed_conversion_device(0,residues.get(),limbs.get(),cells,prefix,2,4,started)==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_device_to_host(0,output.data(),limbs.get(),output.size()*8)==RNS8_SUCCESS);
      cpp_int product=1;for(unsigned p=0;p<prefix;++p) product*=kDefaultModuli[p];
      for(int cell=0;cell<cells;++cell) {
        cpp_int value=values[cell];if(mode==0 && (values[cell]>>63)) value-=cpp_int(1)<<64;
        for(unsigned p=0;p<prefix;++p) REQUIRE(packed[p*cells+cell]==centered_residue(value,kDefaultModuli[p]));
        cpp_int expected=value%product;if(expected<0) expected+=product;
        const cpp_int actual=cpp_int(output[cell*3])+(cpp_int(output[cell*3+1])<<64)+(cpp_int(output[cell*3+2])<<128);
        REQUIRE(actual==expected);
      }
      for(size_t i=cells*prefix;i<packed.size();++i) REQUIRE(packed[i]==99);
      REQUIRE(output[cells*3]==UINT64_C(0xabcdef));REQUIRE(output[cells*3+1]==UINT64_C(0xabcdef));
    }
  }
}
