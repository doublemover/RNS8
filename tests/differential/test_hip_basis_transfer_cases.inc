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
