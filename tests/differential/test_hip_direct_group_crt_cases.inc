TEST_CASE("direct HIP group CRT preserves complete roots and stable shuffled subgroup slots") {
  if(!hip_available())SKIP("no HIP device for group CRT differential");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr size_t cells=257;
  std::mt19937_64 random(3939);
  for(uint32_t prefix=1;prefix<=20;++prefix) {
    CAPTURE(prefix);
    const auto& plan=group_crt::plan(prefix);const cpp_int product=modulus_product(prefix);
    std::vector<cpp_int> values(cells);
    for(size_t cell=0;cell<cells;++cell)
      values[cell]=(cpp_int(random())+(cpp_int(random())<<64)+(cpp_int(random())<<128))%product;
    values[0]=0;values[1]=1;values[2]=product-1;values[3]=product/2;values[4]=product/2+1;
    std::vector<int8_t> residues(cells*prefix),observed_source(residues.size());
    for(uint32_t p=0;p<prefix;++p)for(size_t cell=0;cell<cells;++cell)
      residues[p*cells+cell]=centered_residue(values[cell],kDefaultModuli[p]);
    const auto bytes=query_group_crt_support(cells,prefix,true,UINT64_MAX).scratch_bytes;
    std::vector<uint64_t> output(cells*3+4,UINT64_MAX),partials(size_t(bytes/8)+4,UINT64_MAX);
    hip_direct_device_buffer input,destination,scratch;
    REQUIRE(input.allocate(0,residues.size())==RNS8_SUCCESS);
    REQUIRE(destination.allocate(0,output.size()*8)==RNS8_SUCCESS);
    REQUIRE(scratch.allocate(0,partials.size()*8)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,input.get(),residues.data(),residues.size())==RNS8_SUCCESS);
    for(uint32_t mode=0;mode<3;++mode) {
      const bool split=mode!=0;const auto order=mode==2?plan.group_count:0u;
      std::fill(output.begin(),output.end(),UINT64_MAX);std::fill(partials.begin(),partials.end(),UINT64_MAX);
      REQUIRE(hip_direct_copy_host_to_device(0,destination.get(),output.data(),output.size()*8)==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,scratch.get(),partials.data(),partials.size()*8)==RNS8_SUCCESS);
      REQUIRE(hip_direct_export_canonical_group_crt_device(0,input.get(),destination.get(),cells,prefix,
          split,split?scratch.get():nullptr,split?size_t(bytes):0,order)==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_device_to_host(0,output.data(),destination.get(),output.size()*8)==RNS8_SUCCESS);
      for(size_t cell=0;cell<cells;++cell) {
        const cpp_int root=cpp_int(output[cell*3])+(cpp_int(output[cell*3+1])<<64)+(cpp_int(output[cell*3+2])<<128);
        REQUIRE(root==values[cell]);
      }
      for(size_t i=cells*3;i<output.size();++i)CHECK(output[i]==UINT64_MAX);
      if(split) {
        REQUIRE(hip_direct_copy_device_to_host(0,partials.data(),scratch.get(),partials.size()*8)==RNS8_SUCCESS);
        for(uint32_t g=0;g<plan.group_count;++g)for(size_t cell=0;cell<cells;++cell)
          REQUIRE(cpp_int(partials[g*cells+cell])==values[cell]%plan.groups[g].product);
        for(size_t i=size_t(bytes/8);i<partials.size();++i)CHECK(partials[i]==UINT64_MAX);
      }
      REQUIRE(hip_direct_copy_device_to_host(0,observed_source.data(),input.get(),residues.size())==RNS8_SUCCESS);
      CHECK(observed_source==residues);
    }
  }
}

TEST_CASE("group CRT static rejections preserve prior destination bytes") {
  if(!hip_available())SKIP("no HIP device for group CRT rejection differential");
  using namespace rns8::detail;
  constexpr int64_t cells=17;constexpr uint32_t prefix=20;
  const auto bytes=query_group_crt_support(cells,prefix,true,UINT64_MAX).scratch_bytes;
  hip_direct_device_buffer input,output,scratch;
  std::vector<uint8_t> prior(size_t(cells)*24,0xa5),observed(prior.size());
  REQUIRE(input.allocate(0,size_t(cells)*prefix)==RNS8_SUCCESS);
  REQUIRE(output.allocate(0,prior.size())==RNS8_SUCCESS);
  REQUIRE(scratch.allocate(0,size_t(bytes))==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,output.get(),prior.data(),prior.size())==RNS8_SUCCESS);
  CHECK(hip_direct_export_canonical_group_crt_device(0,input.get(),output.get(),cells,prefix,true,scratch.get(),size_t(bytes-1))==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(hip_direct_export_canonical_group_crt_device(0,input.get(),output.get(),cells,prefix,true,nullptr,size_t(bytes))==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(hip_direct_export_canonical_group_crt_device(0,input.get(),output.get(),cells,prefix,true,output.get(),size_t(bytes))==RNS8_INVALID_ARGUMENT);
  CHECK(hip_direct_export_canonical_group_crt_device(0,input.get(),output.get(),cells,prefix,false,scratch.get(),size_t(bytes))==RNS8_INVALID_ARGUMENT);
  CHECK(hip_direct_export_canonical_group_crt_device(0,input.get(),output.get(),cells,prefix,true,scratch.get(),size_t(bytes),6)==RNS8_INVALID_ARGUMENT);
  CHECK(hip_direct_export_canonical_group_crt_device(0,input.get(),output.get(),cells,0,false,nullptr,0)==RNS8_INVALID_ARGUMENT);
  REQUIRE(hip_direct_copy_device_to_host(0,observed.data(),output.get(),observed.size())==RNS8_SUCCESS);
  CHECK(observed==prior);
}
