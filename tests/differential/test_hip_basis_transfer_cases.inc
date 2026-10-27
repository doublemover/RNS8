TEST_CASE("ordered basis native transfers and exact conversion compose real stream and graph owners") {
  if(!hip_available())SKIP("HIP device required for ordered basis transfers");
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  for(bool sign:{false,true}) {
    auto order=experimental_basis_fixture::moduli();std::reverse(order.begin(),order.end());
    auto* basis=experimental_basis_fixture::create(order);
    const auto semantics=sign?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
    auto* source=experimental_basis_fixture::matrix(ctx,basis,semantics,1,3);
    auto* destination=experimental_basis_fixture::matrix(ctx,nullptr,semantics,1,3);
    auto* target_plan=experimental_basis_fixture::plan(ctx,nullptr,semantics,1,3,1,20);
    std::array<uint64_t,5> input{99,UINT64_MAX,UINT64_C(1)<<63,17,99};
    std::array<uint64_t,16> output;output.fill(99);
    hip_direct_device_buffer native_input,native_output;
    REQUIRE(native_input.allocate(ctx->device_id,sizeof(input))==RNS8_SUCCESS);
    REQUIRE(native_output.allocate(ctx->device_id,sizeof(output))==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(ctx->device_id,native_input.get(),input.data(),sizeof(input))==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(ctx->device_id,native_output.get(),output.data(),sizeof(output))==RNS8_SUCCESS);
    unsigned acquired=0,released=0;std::array<unsigned*,2> counters{&acquired,&released};
    auto descriptor=[&](bool exporting) {
      rns8_native_device_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
      d.element_type=exporting?(sign?RNS8_NATIVE_DEVICE_SIGNED_LIMBS:RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS):
          (sign?RNS8_NATIVE_DEVICE_I64:RNS8_NATIVE_DEVICE_U64);
      d.access=exporting?RNS8_NATIVE_DEVICE_WRITE:RNS8_NATIVE_DEVICE_READ;
      d.allocation_base=exporting?native_output.get():native_input.get();d.allocation_bytes=exporting?sizeof(output):sizeof(input);
      d.byte_offset=8;d.rows=1;d.cols=3;d.ld=4;d.device_id=ctx->device_id;d.producer_complete=1;
      d.policy=RNS8_NATIVE_DEVICE_BORROW;d.owner_identity=exporting?202:101;d.generation=1;d.owner=&counters;
      d.acquire=[](void* p){++*(*static_cast<std::array<unsigned*,2>*>(p))[0];return RNS8_SUCCESS;};
      d.release=[](void* p){++*(*static_cast<std::array<unsigned*,2>*>(p))[1];};d.limb_count=exporting?3:0;return d;
    };
    rns8_native_device_lease *input_lease=nullptr,*output_lease=nullptr;
    auto id=descriptor(false),od=descriptor(true);
    REQUIRE(rns8_create_native_device_lease(ctx,&id,&input_lease)==RNS8_SUCCESS);
    REQUIRE(rns8_create_native_device_lease(ctx,&od,&output_lease)==RNS8_SUCCESS);
    rns8_native_transfer_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
    options.import_values=1;options.max_total_scratch_bytes=4096;
    rns8_prepared_execution *importer=nullptr,*converter=nullptr,*exporter=nullptr;
    REQUIRE(rns8_prepare_native_transfer_hip(ctx,source,input_lease,&options,&importer)==RNS8_SUCCESS);
    hipStream_t first=nullptr,second=nullptr,third=nullptr;
    REQUIRE(hipStreamCreateWithFlags(&first,hipStreamNonBlocking)==hipSuccess);
    REQUIRE(hipStreamCreateWithFlags(&second,hipStreamNonBlocking)==hipSuccess);
    REQUIRE(hipStreamCreateWithFlags(&third,hipStreamNonBlocking)==hipSuccess);
    auto wait=[&](rns8_prepared_execution* owner,rns8_completion_token token) {
      rns8_completion_info result{};result.struct_size=sizeof(result);result.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_wait_prepared_hip(owner,token,&result)==RNS8_SUCCESS);
      REQUIRE(result.operation_status==RNS8_SUCCESS);REQUIRE(result.output_is_current==1);
    };
    rns8_completion_token it{},ct{},et{};
    REQUIRE(rns8_submit_prepared_hip(importer,first,&it)==RNS8_SUCCESS);wait(importer,it);
    CHECK(rns8_prepare_basis_conversion_hip(target_plan,source,destination,67,&converter)==RNS8_WORKSPACE_TOO_SMALL);
    REQUIRE(converter==nullptr);
    REQUIRE(rns8_prepare_basis_conversion_hip(target_plan,source,destination,68,&converter)==RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip(converter,second,&ct)==RNS8_SUCCESS);wait(converter,ct);
    options.import_values=0;
    REQUIRE(rns8_prepare_native_transfer_hip(ctx,destination,output_lease,&options,&exporter)==RNS8_SUCCESS);
    for(auto* owner:{importer,converter,exporter})REQUIRE(rns8_capture_prepared_hip_graph(owner)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_basis_plan(target_plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_basis(basis)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_native_device_lease(input_lease)==RNS8_BUSY);
    REQUIRE(rns8_submit_prepared_hip_graph(importer,first,&it)==RNS8_SUCCESS);
    REQUIRE(rns8_bind_prepared_dependency(converter,importer,it)==RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip_graph_pipeline(converter,second,&ct)==RNS8_SUCCESS);
    REQUIRE(rns8_bind_prepared_dependency(exporter,converter,ct)==RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip_graph_pipeline(exporter,third,&et)==RNS8_SUCCESS);wait(exporter,et);
    REQUIRE(hip_direct_copy_device_to_host(ctx->device_id,output.data(),native_output.get(),sizeof(output))==RNS8_SUCCESS);
    for(unsigned cell=0;cell<3;++cell) {
      boost::multiprecision::cpp_int expected=input[cell+1];
      if(sign && input[cell+1]>>63)expected-=boost::multiprecision::cpp_int(1)<<64;
      if(expected<0)expected+=boost::multiprecision::cpp_int(1)<<192;
      for(unsigned limb=0;limb<3;++limb){CHECK(output[1+3*cell+limb]==static_cast<uint64_t>(expected&UINT64_MAX));expected>>=64;}
    }
    CHECK(output.front()==99);for(unsigned i=10;i<output.size();++i)CHECK(output[i]==99);
    REQUIRE(rns8_destroy_prepared_execution(exporter)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_prepared_execution(converter)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_prepared_execution(importer)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_native_device_lease(output_lease)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_native_device_lease(input_lease)==RNS8_SUCCESS);CHECK(acquired==2);CHECK(released==2);
    REQUIRE(hipStreamDestroy(third)==hipSuccess);REQUIRE(hipStreamDestroy(second)==hipSuccess);REQUIRE(hipStreamDestroy(first)==hipSuccess);
    REQUIRE(rns8_destroy_matrix(destination)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(source)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#endif
}
TEST_CASE("ordered runtime bases execute actual library and builtin planes including small composite moduli") {
  if(!hip_available())SKIP("HIP device required for basis providers");
  using boost::multiprecision::cpp_int;
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  for(uint16_t first_modulus:{3,9}) {
    std::array<uint16_t,20> order{9,16,25,251,241,239,233,229,227,223,211,199,197,193,191,181,179,173,167,163};
    order[0]=first_modulus;
    auto* basis=experimental_basis_fixture::create(order);
    for(bool sign:{false,true})for(auto backend:{RNS8_BACKEND_CK,RNS8_BACKEND_ROCWMMA,RNS8_BACKEND_HIPBLASLT,RNS8_BACKEND_AMDGPU_BUILTINS}) {
      CAPTURE(first_modulus,sign,backend);
      rns8_backend_capability_info capability{};capability.struct_size=sizeof(capability);capability.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_get_backend_capability_info(backend,&capability)==RNS8_SUCCESS);
      rns8_basis_plan_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.backend=backend;
      d.semantics=sign?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;d.m=2;d.n=3;d.k=5;d.prefix=20;
      d.max_host_bytes=d.max_device_bytes=UINT64_MAX;rns8_basis_plan* plan=nullptr;
      if(!capability.compiled_kernel_available){CHECK(rns8_create_basis_plan(ctx,basis,&d,&plan)==RNS8_UNSUPPORTED_BACKEND);continue;}
      REQUIRE(rns8_create_basis_plan(ctx,basis,&d,&plan)==RNS8_SUCCESS);
      auto* a=experimental_basis_fixture::matrix(ctx,basis,d.semantics,2,5);
      auto* b=experimental_basis_fixture::matrix(ctx,basis,d.semantics,5,3);
      auto* c=experimental_basis_fixture::matrix(ctx,basis,d.semantics,2,3);
      std::vector<uint64_t> av(14,99),bv(25,99),out(24,99);
      for(int row=0;row<2;++row)for(int k=0;k<5;++k)av[row*7+k]=(row+k)%2?UINT64_MAX:(UINT64_C(1)<<63);
      for(int k=0;k<5;++k)for(int col=0;col<3;++col)bv[k*5+col]=(col+k)%2?UINT64_MAX:17;
      REQUIRE((sign?rns8_pack_basis_i64(plan,a,reinterpret_cast<const int64_t*>(av.data()),7,0):rns8_pack_basis_u64(plan,a,av.data(),7,0))==RNS8_SUCCESS);
      REQUIRE((sign?rns8_pack_basis_i64(plan,b,reinterpret_cast<const int64_t*>(bv.data()),5,0):rns8_pack_basis_u64(plan,b,bv.data(),5,0))==RNS8_SUCCESS);
      REQUIRE(rns8_gemm_basis(plan,a,b,c)==RNS8_SUCCESS);
      REQUIRE(rns8_export_basis_limbs(plan,c,out.data(),4,3)==RNS8_SUCCESS);
      for(int row=0;row<2;++row)for(int col=0;col<3;++col){cpp_int expected=0;
        for(int k=0;k<5;++k)expected+=(sign?cpp_int(int64_t(av[row*7+k])):cpp_int(av[row*7+k]))*(sign?cpp_int(int64_t(bv[k*5+col])):cpp_int(bv[k*5+col]));
        if(expected<0)expected+=cpp_int(1)<<192;
        for(int limb=0;limb<3;++limb){CHECK(out[(row*4+col)*3+limb]==uint64_t(expected&UINT64_MAX));expected>>=64;}
      }
      for(int row=0;row<2;++row)for(int limb=0;limb<3;++limb)CHECK(out[(row*4+3)*3+limb]==99);
      REQUIRE(rns8_destroy_basis_plan(plan)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);
    }
    REQUIRE(rns8_destroy_basis(basis)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
