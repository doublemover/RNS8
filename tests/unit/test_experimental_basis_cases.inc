TEST_CASE("experimental ordered bases reject noncoprime descriptors and retain owned identities") {
  rns8_basis_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.representation=1;
  rns8_basis* basis=nullptr;
  for(const std::vector<uint16_t>& moduli:std::vector<std::vector<uint16_t>>{{},{1},{257},{6,9},{251,251}}) {
    d.moduli=moduli.data();d.count=uint32_t(moduli.size());REQUIRE(rns8_create_basis(&d,&basis)==RNS8_INVALID_ARGUMENT);REQUIRE(basis==nullptr);
  }
  auto* ctx=create_cpu_context();auto ordered=experimental_basis_fixture::moduli();
  auto* owner=experimental_basis_fixture::create(ordered);
  auto* matrix=experimental_basis_fixture::matrix(ctx,owner,RNS8_EXACT_WIDE_SIGNED,1,1);
  const auto data=matrix->experimental_basis;REQUIRE(data!=nullptr);
  REQUIRE(rns8_destroy_basis(owner)==RNS8_SUCCESS);
  using boost::multiprecision::cpp_int;
  for(uint32_t prefix=1;prefix<=20;++prefix) {
    const auto& product=data->products[prefix];
    for(bool sign:{false,true}) {
      std::vector<cpp_int> values{0,1,product-1,product/2};
      if(sign)values={-(product/2),-1,0,(product-1)/2};
      for(const auto& expected:values) {
        std::vector<int8_t> residues(prefix);
        for(uint32_t p=0;p<prefix;++p)residues[p]=rns8::detail::centered_residue(expected,data->packet.moduli[p]);
        uint64_t words[3]{};REQUIRE(rns8::detail::basis_reconstruct_words(residues.data(),1,prefix,data->packet,sign,words));
        cpp_int actual=cpp_int(words[0])+(cpp_int(words[1])<<64)+(cpp_int(words[2])<<128);
        if(sign && words[2]>>63)actual-=cpp_int(1)<<192;
        REQUIRE(actual==expected);
      }
    }
  }
  const int8_t invalid=-128;uint64_t rejected[3]{};
  CHECK_FALSE(rns8::detail::basis_reconstruct_words(&invalid,1,1,data->packet,true,rejected));
  REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
TEST_CASE("experimental basis CPU routes preserve full-width products conversion and slab edges") {
  auto* ctx=create_cpu_context();
  for(bool sign:{false,true})experimental_basis_fixture::roundtrip(ctx,sign);
  experimental_basis_fixture::slab_edges(ctx);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
