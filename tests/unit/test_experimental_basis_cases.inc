TEST_CASE("experimental basis typed inventory shares concrete owner resources and full identity") {
  auto* ctx=create_cpu_context();auto ordered=experimental_basis_fixture::moduli(),reordered=ordered;
  std::reverse(reordered.begin(),reordered.end());auto* basis=experimental_basis_fixture::create(ordered);
  auto* other=experimental_basis_fixture::create(reordered);
  rns8_basis_plan_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.backend=RNS8_BACKEND_CPU_REFERENCE;
  d.semantics=RNS8_EXACT_WIDE_SIGNED;d.m=3;d.n=2;d.k=5;d.prefix=20;d.max_host_bytes=d.max_device_bytes=UINT64_MAX;
  rns8_operation_support support{};support.struct_size=sizeof(support);support.abi_version=RNS8_ABI_VERSION;
  rns8_basis_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
  std::string first_identity;
  for(auto operation:{RNS8_OP_GEMM,RNS8_OP_PACK,RNS8_OP_EXPORT,RNS8_OP_BASE_EXTEND,RNS8_OP_EXPERIMENTAL_BASIS}) {
    REQUIRE(rns8_query_basis_operation(ctx,basis,&d,operation,RNS8_EXEC_SYNC,&support,&info)==RNS8_SUCCESS);
    CHECK(support.source_implemented==1);CHECK(support.compiled_inventory==1);CHECK(support.requires_binding==1);
    CHECK(support.correctness_qualified==0);CHECK(support.performance_qualified==0);
    CHECK(support.supported_execution_modes==RNS8_EXEC_SYNC);CHECK(support.required_pinned_host_bytes==0);
    REQUIRE(std::equal(ordered.begin(),ordered.end(),info.moduli));
    rns8_basis_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_query_basis_plan(basis,&d,&resources)==RNS8_SUCCESS);
    CHECK(resources.required_device_bytes==support.required_device_bytes);CHECK(resources.required_host_bytes==support.required_host_bytes);
    rns8_basis_plan* plan=nullptr;REQUIRE(rns8_create_basis_plan(ctx,basis,&d,&plan)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_basis_plan(plan)==RNS8_SUCCESS);
    if(operation==RNS8_OP_GEMM)first_identity=support.provider_identity_sha256;
  }
  REQUIRE(rns8_query_basis_operation(ctx,other,&d,RNS8_OP_GEMM,RNS8_EXEC_SYNC,&support,&info)==RNS8_SUCCESS);
  CHECK(first_identity!=support.provider_identity_sha256);
  CHECK(rns8_query_basis_operation(ctx,basis,&d,RNS8_OP_GEMM,RNS8_EXEC_OWNED_GRAPH,&support,&info)==RNS8_UNSUPPORTED_BACKEND);
  CHECK(support.reason_code==RNS8_SUPPORT_EXECUTION_MODE);
  CHECK(rns8_query_basis_operation(ctx,basis,&d,RNS8_OP_QUARTER_CRT,RNS8_EXEC_SYNC,&support,&info)==RNS8_UNSUPPORTED_BACKEND);
  CHECK(support.reason_code==RNS8_SUPPORT_OPERATION);
  REQUIRE(rns8_query_basis_operation(ctx,basis,&d,RNS8_OP_GEMM,RNS8_EXEC_SYNC,&support,&info)==RNS8_SUCCESS);
  d.max_host_bytes=support.required_host_bytes-1;
  CHECK(rns8_query_basis_operation(ctx,basis,&d,RNS8_OP_GEMM,RNS8_EXEC_SYNC,&support,&info)==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(support.reason_code==RNS8_SUPPORT_BUDGET);CHECK(support.required_host_bytes==d.max_host_bytes+1);
  REQUIRE(rns8_destroy_basis(other)==RNS8_SUCCESS);REQUIRE(rns8_destroy_basis(basis)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

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
        for(uint32_t limbs:{1u,2u,3u,4u,32u}) {
          std::array<uint64_t,32> exported;exported.fill(99);
          const cpp_int limit=cpp_int(1)<<(64*limbs-(sign?1:0));
          const bool fits=sign?(expected>=-limit && expected<limit):(expected>=0 && expected<limit);
          REQUIRE(rns8::detail::basis_export_words(words,sign,limbs,exported.data())==fits);
          if(!fits){CHECK(std::all_of(exported.begin(),exported.end(),[](uint64_t v){return v==99;}));continue;}
          cpp_int reference=expected;if(reference<0)reference+=cpp_int(1)<<(64*limbs);
          for(uint32_t limb=0;limb<limbs;++limb){CHECK(exported[limb]==static_cast<uint64_t>(reference&UINT64_MAX));reference>>=64;}
          for(uint32_t limb=limbs;limb<32;++limb)CHECK(exported[limb]==99);
        }
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
TEST_CASE("runtime basis selection binds prices to one exact workload and charges complete reuse") {
  auto* ctx=create_cpu_context();auto order=experimental_basis_fixture::moduli();
  auto* first=experimental_basis_fixture::create(order);std::reverse(order.begin(),order.end());auto* second=experimental_basis_fixture::create(order);
  rns8_basis_candidate candidates[2]{};
  for(unsigned i=0;i<2;++i){auto& c=candidates[i];c.basis=i?second:first;
    auto& d=c.plan;d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.backend=RNS8_BACKEND_CPU_REFERENCE;
    d.semantics=RNS8_EXACT_WIDE_SIGNED;d.m=2;d.n=3;d.k=5;d.prefix=20;d.max_device_bytes=d.max_host_bytes=UINT64_MAX;
    rns8_basis_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_query_basis_plan(c.basis,&d,&resources)==RNS8_SUCCESS);
    rns8_basis_info basis{};basis.struct_size=sizeof(basis);basis.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_get_basis_info(c.basis,20,&basis)==RNS8_SUCCESS);
    auto& p=c.cost;p.struct_size=sizeof(p);p.abi_version=RNS8_ABI_VERSION;p.kind=2;
    REQUIRE(rns8_get_pipeline_build_identity(p.build_identity,sizeof(p.build_identity))==RNS8_SUCCESS);
    std::fill_n(p.cohort_identity,64,'a');std::strcpy(p.target_id,"cpu");std::copy_n(basis.moduli,20,p.ordered_moduli);
    p.count=p.prefix=20;p.backend=d.backend;p.semantics=d.semantics;p.m=d.m;p.n=d.n;p.k=d.k;
    p.device_bytes=resources.required_device_bytes;p.host_bytes=resources.required_host_bytes;
    p.preparation_ns=i?10:1000;p.pack_ns=3;p.gemm_ns=i?30:20;p.export_ns=7;
  }
  rns8_basis_selection_desc request{};request.struct_size=sizeof(request);request.abi_version=RNS8_ABI_VERSION;
  request.semantics=RNS8_EXACT_WIDE_SIGNED;request.m=2;request.n=3;request.k=5;request.reuse_count=10;request.magnitude_bound[0]=5000;
  std::fill_n(request.cohort_identity,64,'a');
  rns8_basis_selection selection{};selection.struct_size=sizeof(selection);selection.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_select_basis_plan(ctx,&request,candidates,2,&selection)==RNS8_SUCCESS);
  CHECK(selection.selected_index==1);CHECK(selection.projected_ns==320);CHECK(selection.eligible_candidates==2);
  request.reuse_count=1000;
  rns8_basis_plan* plan=nullptr;REQUIRE(rns8_create_selected_basis_plan(ctx,&request,candidates,2,&selection,&plan)==RNS8_SUCCESS);
  CHECK(selection.selected_index==0);CHECK(selection.projected_ns==21010);
  REQUIRE(rns8_destroy_basis_plan(plan)==RNS8_SUCCESS);
  const auto saved=selection;
  candidates[0].cost.ordered_moduli[0]^=1;candidates[1].cost.gemm_ns=UINT64_MAX;
  CHECK(rns8_select_basis_plan(ctx,&request,candidates,2,&selection)==RNS8_UNSUPPORTED_BACKEND);
  CHECK(std::memcmp(&selection,&saved,sizeof(saved))==0);
  candidates[0].cost.ordered_moduli[0]^=1;candidates[1].cost.gemm_ns=30;
  request.magnitude_bound[2]=UINT64_MAX;
  CHECK(rns8_select_basis_plan(ctx,&request,candidates,2,&selection)==RNS8_UNSUPPORTED_BACKEND);
  request.magnitude_bound[2]=0;request.reuse_count=UINT64_MAX;
  CHECK(rns8_select_basis_plan(ctx,&request,candidates,2,&selection)==RNS8_UNSUPPORTED_BACKEND);
  request.reuse_count=1;request.n=4;
  CHECK(rns8_select_basis_plan(ctx,&request,candidates,2,&selection)==RNS8_UNSUPPORTED_BACKEND);
  REQUIRE(rns8_destroy_basis(first)==RNS8_SUCCESS);REQUIRE(rns8_destroy_basis(second)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
