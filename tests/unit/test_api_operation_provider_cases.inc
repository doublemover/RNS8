TEST_CASE("distributed discovery separates written HIP source from unbound topology", "[api][provider][distributed]") {
  auto* ctx=create_cpu_context();
  rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
  q.operation=RNS8_OP_MULTI_DEVICE;q.execution_modes=RNS8_EXEC_SYNC|RNS8_EXEC_MULTI_DEVICE;
  q.problem=bounded_gemm_desc(RNS8_BOUNDED_I64,17,19,65);q.problem.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  q.input_domain=4;q.output_domain=5;q.output_limbs=3;q.device_count=2;q.scratch_budget=UINT64_MAX;
  rns8_operation_support support{};uint32_t count=0;
  REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);REQUIRE(count==1);
  CHECK(support.source_implemented==1);CHECK(support.compiled_inventory==uint32_t(rns8::detail::hip_direct_compiled()));
  CHECK(support.status==RNS8_UNSUPPORTED_BACKEND);CHECK(support.reason_code==RNS8_SUPPORT_TOPOLOGY);
  CHECK(support.requires_binding==1);CHECK(support.resource_status==3);CHECK(support.runtime_observed==0);
  CHECK(support.correctness_qualified==0);CHECK(support.performance_qualified==0);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
TEST_CASE("operation discovery shares CPU plan admission and rejects representation/mode mismatches", "[api][provider]") {
  auto* ctx=create_cpu_context();
  rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;q.operation=RNS8_OP_GEMM;q.execution_modes=RNS8_EXEC_SYNC;q.scratch_budget=UINT64_MAX;
  for(auto semantics:{RNS8_BOUNDED_I64,RNS8_BOUNDED_U64,RNS8_EXACT_WIDE_SIGNED,RNS8_EXACT_WIDE_UNSIGNED,RNS8_FINITE_RING_U8,RNS8_FINITE_FIELD_U8}) {
    q.problem=semantics==RNS8_BOUNDED_I64 || semantics==RNS8_BOUNDED_U64?bounded_gemm_desc(semantics,17,19,65537):
      semantics==RNS8_FINITE_RING_U8 || semantics==RNS8_FINITE_FIELD_U8?finite_gemm_desc(semantics,17,19,65537,251):exact_desc(semantics,17,19,65537);
    rns8_operation_support support{};uint32_t count=0;
    REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);REQUIRE(count==1);
    rns8_plan* plan=nullptr;const auto status=rns8_create_plan(ctx,&q.problem,&plan);CHECK(status==support.status);
    CHECK_FALSE(support.correctness_qualified);CHECK_FALSE(support.performance_qualified);CHECK(support.requires_binding);
    REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  }
  q.problem=exact_desc(RNS8_EXACT_WIDE_SIGNED,16,16,16);q.problem.requested_backend=RNS8_BACKEND_AUTO;
  uint32_t required=0;REQUIRE(rns8_query_operation_support(ctx,&q,nullptr,0,&required)==RNS8_SUCCESS);CHECK(required==9);
  rns8_operation_support first[2]{};REQUIRE(rns8_query_operation_support(ctx,&q,first,2,&required)==RNS8_SUCCESS);CHECK(required==9);
  q.problem.requested_backend=RNS8_BACKEND_CPU_REFERENCE;
  q.execution_modes=RNS8_EXEC_CALLER_CAPTURE;REQUIRE(rns8_query_operation_support(ctx,&q,first,1,&required)==RNS8_SUCCESS);
  CHECK(first[0].reason_code==RNS8_SUPPORT_EXECUTION_MODE);
  q.execution_modes=RNS8_EXEC_SYNC;q.input_domain=4;REQUIRE(rns8_query_operation_support(ctx,&q,first,1,&required)==RNS8_SUCCESS);CHECK(first[0].status==RNS8_UNSUPPORTED_BACKEND);
  q.input_domain=0;q.operation=RNS8_OP_EXACT_EPILOGUE;q.problem=bounded_gemm_desc(RNS8_BOUNDED_I64,16,16,16);
  REQUIRE(rns8_query_operation_support(ctx,&q,first,1,&required)==RNS8_SUCCESS);CHECK(first[0].status==RNS8_UNSUPPORTED_BACKEND);
  q.problem=exact_desc(RNS8_EXACT_WIDE_UNSIGNED,16,16,16);q.alpha=-1;
  REQUIRE(rns8_query_operation_support(ctx,&q,first,1,&required)==RNS8_SUCCESS);CHECK(first[0].status==RNS8_INVALID_ARGUMENT);
  q.operation=RNS8_OP_STRASSEN;q.alpha=0;REQUIRE(rns8_query_operation_support(ctx,&q,first,1,&required)==RNS8_SUCCESS);CHECK(first[0].status==RNS8_UNSUPPORTED_BACKEND);
  q.operation=RNS8_OP_GEMM;q.lda=INT64_MAX;REQUIRE(rns8_query_operation_support(ctx,&q,first,1,&required)==RNS8_SUCCESS);CHECK(first[0].reason_code==RNS8_SUPPORT_ADDRESS);
  q.abi_version=UINT32_MAX;CHECK(rns8_query_operation_support(ctx,&q,first,1,&required)==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

TEST_CASE("DAG discovery agrees with concrete resident graph backend ownership", "[api][provider][graph]") {
  auto* ctx=create_cpu_context();
  for(auto backend:{RNS8_BACKEND_CPU_REFERENCE,RNS8_BACKEND_HIP_DIRECT,RNS8_BACKEND_HIPBLASLT,
      RNS8_BACKEND_CK,RNS8_BACKEND_ROCWMMA,RNS8_BACKEND_AMDGPU_BUILTINS,
      RNS8_BACKEND_WRAP64_BYTE_LIMB,RNS8_BACKEND_CPU_AMX,RNS8_BACKEND_HIP_VECTOR_ALU_INT64}) {
    CAPTURE(backend);
    const bool owns_graph=backend==RNS8_BACKEND_CPU_REFERENCE || backend==RNS8_BACKEND_HIP_DIRECT ||
        backend==RNS8_BACKEND_WRAP64_BYTE_LIMB;
    rns8_context hypothetical{};hypothetical.backend=backend;
    rns8_resident_node invalid_node{};rns8_resident_graph* graph=nullptr;
    // Unsupported families fail at ownership admission; admitted families reach
    // descriptor validation. No plan, matrix, GPU probe or device allocation is needed.
    CHECK(rns8_create_resident_graph(&hypothetical,&invalid_node,1,&graph)==
        (owns_graph?RNS8_INVALID_ARGUMENT:RNS8_UNSUPPORTED_BACKEND));
    CHECK(graph==nullptr);
    for(auto operation:{RNS8_OP_GEMM,RNS8_OP_PREPARED_GROUP}) {
      rns8_operation_query query{};query.struct_size=sizeof(query);query.abi_version=RNS8_ABI_VERSION;
      query.operation=operation;query.execution_modes=RNS8_EXEC_DAG;query.scratch_budget=UINT64_MAX;
      query.problem=bounded_gemm_desc(RNS8_BOUNDED_I64,1,1,1);
      if(backend==RNS8_BACKEND_WRAP64_BYTE_LIMB) {
        query.problem.semantics=RNS8_WRAP_U64_MOD_2_64;query.problem.bound_kind=RNS8_BOUND_NONE;
        query.problem.max_prefix=0;query.problem.bound=0;
      }
      query.problem.requested_backend=backend;
      rns8_operation_support support{};uint32_t count=0;
      REQUIRE(rns8_query_operation_support(ctx,&query,&support,1,&count)==RNS8_SUCCESS);
      REQUIRE(count==1);
      CHECK(bool(support.supported_execution_modes&RNS8_EXEC_DAG)==owns_graph);
      if(!owns_graph) {
        CHECK(support.status==RNS8_UNSUPPORTED_BACKEND);
        if(support.source_implemented)CHECK(support.reason_code==RNS8_SUPPORT_EXECUTION_MODE);
        else CHECK(support.reason_code==RNS8_SUPPORT_SEMANTICS);
      }
      CHECK_FALSE(support.correctness_qualified);CHECK_FALSE(support.performance_qualified);
    }
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

TEST_CASE("operation inventory separates source semantics compilation and unknown resources", "[api][provider]") {
  struct Row {rns8_backend_kind backend;bool semantics[7];bool device;};
  const Row rows[]={
    {RNS8_BACKEND_CPU_REFERENCE,{true,true,true,true,false,true,true},false},
    {RNS8_BACKEND_CPU_AMX,{true,true,true,true,false,true,true},false},
    {RNS8_BACKEND_HIP_DIRECT,{true,true,true,true,true,true,true},true},
    {RNS8_BACKEND_HIP_VECTOR_ALU_INT64,{true,true,false,false,false,false,false},true},
    {RNS8_BACKEND_HIPBLASLT,{true,true,true,true,false,true,true},true},
    {RNS8_BACKEND_CK,{true,false,true,true,false,true,true},true},
    {RNS8_BACKEND_ROCWMMA,{true,true,true,true,false,true,true},true},
    {RNS8_BACKEND_AMDGPU_BUILTINS,{true,true,true,true,false,true,true},true},
    {RNS8_BACKEND_WRAP64_BYTE_LIMB,{false,false,false,false,true,false,false},false}};
  const rns8_semantics semantics[]={RNS8_BOUNDED_I64,RNS8_BOUNDED_U64,RNS8_EXACT_WIDE_SIGNED,
      RNS8_EXACT_WIDE_UNSIGNED,RNS8_WRAP_U64_MOD_2_64,RNS8_FINITE_RING_U8,RNS8_FINITE_FIELD_U8};
  auto* ctx=create_cpu_context();
  for(const auto& row:rows)for(unsigned i=0;i<7;++i) {
    CAPTURE(row.backend,semantics[i]);
    CHECK(rns8::detail::api::backend_source_supports_semantics(row.backend,semantics[i])==row.semantics[i]);
    CHECK(rns8::detail::api::hip_device_backend_kind(row.backend)==row.device);
    rns8_backend_capability_info capability{};capability.struct_size=sizeof(capability);capability.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_get_backend_capability_info(row.backend,&capability)==RNS8_SUCCESS);
    rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
    q.operation=RNS8_OP_GEMM;q.execution_modes=RNS8_EXEC_SYNC;q.scratch_budget=UINT64_MAX;
    q.problem=i<2?bounded_gemm_desc(semantics[i],1,1,1):i==4?wrap_gemm_desc(1,1,1):
        i>=5?finite_gemm_desc(semantics[i],1,1,1,251):exact_desc(semantics[i],1,1,1);
    q.problem.requested_backend=row.backend;
    rns8_operation_support support{};uint32_t count=0;
    REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);REQUIRE(count==1);
    CHECK(bool(support.source_implemented)==row.semantics[i]);
    if(!row.semantics[i]) {
      CHECK(support.reason_code==RNS8_SUPPORT_SEMANTICS);CHECK(support.resource_status==3);
      CHECK_FALSE(support.compiled_inventory);CHECK_FALSE(support.runtime_observed);
    } else {
      CHECK(bool(support.compiled_inventory)==bool(capability.compiled_kernel_available));
      CHECK(std::string(support.target_id)==(row.device?"":"cpu"));
      if(!capability.compiled_kernel_available) {
        CHECK(support.reason_code==RNS8_SUPPORT_NOT_COMPILED);CHECK(support.resource_status==3);
        CHECK(support.required_device_bytes==0);CHECK(support.required_host_bytes==0);
        CHECK_FALSE(support.runtime_observed);
      }
      if(row.device)CHECK_FALSE(support.runtime_observed);
      q.execution_modes=RNS8_EXEC_CALLER_CAPTURE;
      REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);
      CHECK(support.reason_code==RNS8_SUPPORT_EXECUTION_MODE);CHECK(support.resource_status==3);
      CHECK(support.source_implemented==1);
      CHECK(bool(support.compiled_inventory)==bool(capability.compiled_kernel_available));
    }
  }
  rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
  q.operation=RNS8_OP_FP8;q.execution_modes=RNS8_EXEC_SYNC;q.scratch_budget=UINT64_MAX;
  q.problem=exact_desc(RNS8_EXACT_WIDE_SIGNED,1,1,1);
  rns8_operation_support support{};uint32_t count=0;
  REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);
  CHECK(support.reason_code==RNS8_SUPPORT_OPERATION);CHECK(support.resource_status==3);
  CHECK_FALSE(support.source_implemented);CHECK_FALSE(support.compiled_inventory);
  CHECK_FALSE(support.runtime_observed);CHECK(support.supported_execution_modes==0);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

TEST_CASE("runtime shortlist separates quarter headroom from ordinary minimum and stale costs", "[api][planner]") {
  auto* ctx=create_cpu_context();rns8_pipeline_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
  q.operation.struct_size=sizeof(q.operation);q.operation.abi_version=RNS8_ABI_VERSION;q.operation.operation=RNS8_OP_GEMM;
  q.operation.execution_modes=RNS8_EXEC_SYNC;q.operation.scratch_budget=UINT64_MAX;q.operation.problem=exact_desc(RNS8_EXACT_WIDE_SIGNED,16,16,16);
  q.operation.problem.max_prefix=9;q.output_magnitude_limbs[0]=100;q.output_boundary=1;q.payload_budget=UINT64_MAX;q.reuse_count=1;
  rns8_pipeline_candidate c[2]{};uint32_t count=0;REQUIRE(rns8_plan_pipeline(ctx,&q,nullptr,0,c,2,&count)==RNS8_SUCCESS);REQUIRE(count==7);
  CHECK(c[0].prefix==1);CHECK(c[1].prefix==2);CHECK(c[0].status==RNS8_SUCCESS);CHECK(c[1].status==RNS8_SUCCESS);
  CHECK_FALSE(c[0].cost_known);CHECK_FALSE(c[0].pareto);CHECK(c[0].executable_problem.flags&RNS8_PLAN_FORCE_FIXED_PREFIX);CHECK(c[0].executable_problem.flags&RNS8_PLAN_EXACT_VALUE_BOUND);
  for(auto& candidate:c){rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&candidate.executable_problem,&plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);}
  const auto ordinary=c[0].peak_payload_bytes,quarter=c[1].peak_payload_bytes;CHECK(ordinary>0);CHECK(quarter>0);
  q.payload_budget=std::min(ordinary,quarter);REQUIRE(rns8_plan_pipeline(ctx,&q,nullptr,0,c,2,&count)==RNS8_SUCCESS);
  CHECK(c[0].status==(ordinary<=q.payload_budget?RNS8_SUCCESS:RNS8_WORKSPACE_TOO_SMALL));CHECK(c[1].status==(quarter<=q.payload_budget?RNS8_SUCCESS:RNS8_WORKSPACE_TOO_SMALL));
  q.payload_budget=UINT64_MAX;rns8_pipeline_cost_profile profile{};profile.struct_size=sizeof(profile);profile.abi_version=RNS8_ABI_VERSION;
  profile.backend=RNS8_BACKEND_CPU_REFERENCE;std::strcpy(profile.target_id,"cpu");std::strcpy(profile.calibration_identity,"synthetic-regression-not-evidence");
  REQUIRE(rns8_get_pipeline_build_identity(profile.build_identity_sha256,sizeof(profile.build_identity_sha256))==RNS8_SUCCESS);
  profile.known_mask=63;profile.lower_launch_ns=1;profile.upper_launch_ns=2;
  for(unsigned i=0;i<6;++i){profile.lower_ns_per_unit[i]=1;profile.upper_ns_per_unit[i]=2;}
  REQUIRE(rns8_plan_pipeline(ctx,&q,&profile,1,c,2,&count)==RNS8_SUCCESS);CHECK(c[0].cost_known);CHECK(c[0].upper_amortized_ns>=c[0].lower_amortized_ns);
  const auto admitted=c[0].status;profile.build_identity_sha256[0]=profile.build_identity_sha256[0]=='0'?'1':'0';
  REQUIRE(rns8_plan_pipeline(ctx,&q,&profile,1,c,2,&count)==RNS8_SUCCESS);CHECK(c[0].status==admitted);CHECK_FALSE(c[0].cost_known);
  q.output_boundary=0;REQUIRE(rns8_plan_pipeline(ctx,&q,nullptr,0,c,2,&count)==RNS8_SUCCESS);CHECK(count==1);CHECK(c[0].reconstruction==0);
  q.operation.problem.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;REQUIRE(rns8_plan_pipeline(ctx,&q,nullptr,0,c,2,&count)==RNS8_SUCCESS);CHECK(c[0].prefix==9);
  q.reuse_count=0;CHECK(rns8_plan_pipeline(ctx,&q,nullptr,0,c,2,&count)==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

TEST_CASE("generated builtin event map keeps producer and semantic family identities aligned") {
  using namespace rns8::generated_metadata;
  for(const auto& row:builtin_event_families){
    REQUIRE(std::string_view(builtin_event_label(row.family,false))==row.integer_event);
    REQUIRE(std::string_view(builtin_event_label(row.family,true))==row.finite_event);
    REQUIRE(std::string_view(builtin_event_label(std::string(row.family)+"_finite_u8_epilogue_v1",false))==row.finite_event);
  }
  REQUIRE(std::string_view(builtin_event_label("amdgpu_builtin_cdna3_smfmac_i32_32x32x32_i8_sparse_a_v1",true))=="amdgpu_builtin_cdna3_smfmac_i32_32x32x32_i8_sparse_a_finite_kernel");
}

TEST_CASE("lift provider checks source uniqueness rather than a fictitious native GEMM envelope", "[api][provider][lift]") {
  auto* ctx=create_cpu_context();rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
  q.operation=RNS8_OP_BASE_EXTEND;q.execution_modes=RNS8_EXEC_SYNC;q.scratch_budget=UINT64_MAX;
  q.problem=exact_desc(RNS8_EXACT_WIDE_SIGNED,3,5,1);q.problem.max_prefix=9;q.source_prefix=3;
  rns8_operation_support support{};uint32_t count=0;
  REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);CHECK(support.reason_code==RNS8_SUPPORT_MISSING_PROOF);
  q.proof_flags=RNS8_QUERY_CURRENT_RANGE_PROOF;REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);
  CHECK(support.status==RNS8_SUCCESS);CHECK(support.requires_binding);
  rns8_plan* plan=nullptr;CHECK(rns8_create_plan(ctx,&q.problem,&plan)==RNS8_RANGE_ERROR);CHECK(plan==nullptr);
  q.source_prefix=10;REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);CHECK(support.status==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

TEST_CASE("operation import discovery rejects export-only limb formats before inventory admission", "[api][provider][native]") {
  auto* ctx=create_cpu_context();
  rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
  q.execution_modes=RNS8_EXEC_SYNC;q.proof_flags=RNS8_QUERY_CURRENT_RANGE_PROOF;q.scratch_budget=UINT64_MAX;
  for(bool sign:{false,true}) {
    q.problem=exact_desc(sign?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED,1,1,1);
    q.problem.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    for(auto operation:{RNS8_OP_NATIVE_IMPORT,RNS8_OP_HOST_IMPORT})for(uint32_t limbs:{1u,3u,32u}) {
      q.operation=operation;q.output_limbs=limbs;
      rns8_operation_support support{};uint32_t count=0;
      REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);
      CHECK(support.status==RNS8_INVALID_ARGUMENT);CHECK(support.reason_code==RNS8_SUPPORT_BAD_DESCRIPTOR);
      CHECK(std::string(support.offending_field)=="output_limbs");
      CHECK(support.required_host_bytes==0);CHECK(support.required_device_bytes==0);
      // These are pure resource queries: neither descriptor creates a GPU lease.
      uint64_t storage[32]{};rns8_native_device_desc device{};device.struct_size=sizeof(device);device.abi_version=RNS8_ABI_VERSION;
      device.element_type=sign?RNS8_NATIVE_DEVICE_I64:RNS8_NATIVE_DEVICE_U64;device.access=RNS8_NATIVE_DEVICE_READ;
      device.allocation_base=storage;device.allocation_bytes=sizeof(storage);device.rows=device.cols=device.ld=1;device.device_id=0;
      device.producer_complete=1;device.policy=RNS8_NATIVE_DEVICE_BORROW;device.owner_identity=device.generation=1;
      device.acquire=[](void*){return RNS8_SUCCESS;};device.release=[](void*){};
      rns8_native_device_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_query_native_device(&device,&resources)==RNS8_SUCCESS);
      const auto compact=resources.compact_native_bytes;
      device.element_type=sign?RNS8_NATIVE_DEVICE_SIGNED_LIMBS:RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS;device.limb_count=limbs;
      CHECK(rns8_query_native_device(&device,&resources)==support.status);CHECK(resources.compact_native_bytes==compact);
      rns8_pinned_host_desc host{};host.struct_size=sizeof(host);host.abi_version=RNS8_ABI_VERSION;
      host.element_type=device.element_type;host.access=RNS8_NATIVE_DEVICE_READ;host.rows=host.cols=host.ld=1;host.limb_count=limbs;
      uint64_t host_bytes=99;CHECK(rns8_query_pinned_host_bytes(&host,&host_bytes)==support.status);CHECK(host_bytes==99);
    }
    for(auto operation:{RNS8_OP_NATIVE_EXPORT,RNS8_OP_HOST_EXPORT})for(uint32_t limbs:{1u,3u,32u}) {
      q.operation=operation;q.output_limbs=limbs;
      rns8_operation_support support{};uint32_t count=0;
      REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);
      CHECK((support.status==RNS8_SUCCESS || support.reason_code==RNS8_SUPPORT_NOT_COMPILED));
    }
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

TEST_CASE("static provider rejections preserve published output across synchronous and prepared owners", "[api][provider]") {
  auto* ctx=create_cpu_context();auto desc=exact_desc(RNS8_EXACT_WIDE_UNSIGNED,1,1,1);
  rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  rns8_workspace* work=nullptr;REQUIRE(rns8_create_workspace(ctx,plan,&work)==RNS8_SUCCESS);
  auto md=exact_matrix_desc(1,1,RNS8_EXACT_WIDE_UNSIGNED);std::vector<rns8_matrix*> matrices(4);
  for(auto& matrix:matrices)REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
  const uint64_t value=1;for(auto* matrix:matrices)REQUIRE(rns8_pack_u64(ctx,matrix,&value,1,1)==RNS8_SUCCESS);
  auto* output=matrices[3];const auto residues=output->residues;const auto generation=output->content_generation;
  const auto range_prefix=output->exact_range_prefix;const auto magnitude=output->exact_max_magnitude;
  const auto preserved=[&]() {
    CHECK(output->residues==residues);CHECK(output->content_generation==generation);
    CHECK(output->host_residues_current);CHECK(output->exact_range_prefix==range_prefix);CHECK(output->exact_max_magnitude==magnitude);
  };
  rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
  q.problem=desc;q.execution_modes=RNS8_EXEC_SYNC;q.scratch_budget=UINT64_MAX;q.operation=RNS8_OP_EXACT_EPILOGUE;q.alpha=-1;
  rns8_operation_support support{};uint32_t count=0;
  REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);REQUIRE(support.status==RNS8_INVALID_ARGUMENT);
  CHECK(rns8_gemm_exact_axpby(ctx,plan,matrices[0],matrices[1],matrices[2],output,-1,1,work)==support.status);preserved();
  rns8_prepared_execution* prepared=nullptr;
  CHECK(rns8_prepare_exact_axpby(ctx,plan,matrices[0],matrices[1],matrices[2],output,-1,1,work,&prepared)==support.status);
  CHECK(prepared==nullptr);preserved();
  q.operation=RNS8_OP_NARROW_NATIVE;q.alpha=0;q.algorithm=RNS8_NARROW_BYTE_WMMA;
  q.proof_flags=RNS8_QUERY_NATIVE_RETAINED|RNS8_QUERY_NARROW_INPUT_PROOF;
  REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);REQUIRE(support.status==RNS8_UNSUPPORTED_BACKEND);
  CHECK(rns8_gemm_narrow_exact(ctx,plan,matrices[0],matrices[1],output,RNS8_NARROW_BYTE_WMMA,UINT64_MAX,work)==support.status);preserved();
  q.operation=RNS8_OP_STRASSEN;q.algorithm=RNS8_BILINEAR_STRASSEN;q.proof_flags=0;
  REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);REQUIRE(support.status==RNS8_UNSUPPORTED_BACKEND);
  double ratio=99;CHECK(rns8_gemm_bilinear_exact(ctx,plan,matrices[0],matrices[1],output,RNS8_BILINEAR_STRASSEN,UINT64_MAX,work,&ratio)==support.status);
  CHECK(ratio==99);preserved();
  CHECK(rns8_prepare_bilinear_workspace(ctx,plan,RNS8_BILINEAR_STRASSEN,UINT64_MAX,work)==support.status);preserved();
  REQUIRE(rns8_destroy_workspace(work)==RNS8_SUCCESS);for(auto* matrix:matrices)REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}


TEST_CASE("pipeline checked CRT policies bind executable owners without borrowing default prices", "[api][planner][crt]") {
  auto* ctx=create_cpu_context();
  for(unsigned mode=0;mode<3;++mode) {
    const auto semantics=mode==1?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;
    rns8_pipeline_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
    q.operation.struct_size=sizeof(q.operation);q.operation.abi_version=RNS8_ABI_VERSION;
    q.operation.operation=RNS8_OP_GEMM;q.operation.execution_modes=RNS8_EXEC_SYNC;q.operation.scratch_budget=UINT64_MAX;
    q.operation.problem=exact_desc(semantics,2,3,4);q.operation.problem.max_prefix=9;q.operation.output_limbs=1;
    q.output_boundary=mode+1;q.output_magnitude_limbs[0]=100;q.payload_budget=UINT64_MAX;q.reuse_count=3;
    rns8_pipeline_cost_profile profile{};profile.struct_size=sizeof(profile);profile.abi_version=RNS8_ABI_VERSION;
    profile.backend=RNS8_BACKEND_CPU_REFERENCE;std::strcpy(profile.target_id,"cpu");std::strcpy(profile.calibration_identity,"synthetic-only-unrelated-default-prices");
    REQUIRE(rns8_get_pipeline_build_identity(profile.build_identity_sha256,sizeof(profile.build_identity_sha256))==RNS8_SUCCESS);
    profile.known_mask=63;for(unsigned kind=0;kind<6;++kind){profile.lower_ns_per_unit[kind]=1;profile.upper_ns_per_unit[kind]=2;}
    rns8_pipeline_candidate candidates[8]{};candidates[7].struct_size=UINT64_C(0xdeadbeef);uint32_t count=0;
    REQUIRE(rns8_plan_pipeline(ctx,&q,&profile,1,candidates,7,&count)==RNS8_SUCCESS);REQUIRE(count==7);
    CHECK(candidates[7].struct_size==UINT64_C(0xdeadbeef));
    if(mode==2){CHECK(candidates[0].status==RNS8_UNSUPPORTED_BACKEND);CHECK(candidates[0].reason_code==RNS8_SUPPORT_OPERATION);CHECK_FALSE(candidates[0].cost_known);}
    else CHECK(candidates[0].cost_known);
    CHECK(candidates[1].cost_known);
    for(uint32_t algorithm=0;algorithm<5;++algorithm) {
      const auto& candidate=candidates[algorithm+2];REQUIRE(candidate.status==RNS8_SUCCESS);
      CHECK(candidate.reconstruction==RNS8_PIPELINE_RECONSTRUCTION_CHECKED_GARNER+algorithm);CHECK(candidate.prefix==1);
      CHECK(candidate.requires_binding);CHECK_FALSE(candidate.cost_known);CHECK_FALSE(candidate.pareto);
      REQUIRE(candidate.stage_count>0);REQUIRE(candidate.stage_count<=8);
      CHECK(candidate.stages[candidate.stage_count-1].kind==6+algorithm);
      CHECK_FALSE(candidate.stages[candidate.stage_count-1].cost_known);
      CHECK(std::string(candidate.provider_identity_sha256)!=candidates[0].provider_identity_sha256);
      for(uint32_t other=0;other<algorithm;++other)CHECK(std::string(candidate.provider_identity_sha256)!=candidates[other+2].provider_identity_sha256);
      rns8_plan* plan=nullptr;rns8_workspace* workspace=nullptr;
      REQUIRE(rns8_create_plan(ctx,&candidate.executable_problem,&plan)==RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
      const auto matrix=[&](int64_t rows,int64_t cols){auto md=exact_matrix_desc(rows,cols,semantics);md.max_prefix=candidate.prefix;
        rns8_matrix* out=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&out)==RNS8_SUCCESS);return out;};
      auto* a=matrix(2,4);auto* b=matrix(4,3);auto* c=matrix(2,3);
      const int64_t av[]={1,1,1,1,-1,-1,-1,-1},bv[]={1,1,1,1,1,1,1,1,1,1,1,1};
      const uint64_t au[]={1,1,1,1,0,0,0,0},bu[]={1,1,1,1,1,1,1,1,1,1,1,1};
      REQUIRE((mode==1?rns8_pack_u64(ctx,a,au,4,1):rns8_pack_i64(ctx,a,av,4,1))==RNS8_SUCCESS);
      REQUIRE((mode==1?rns8_pack_u64(ctx,b,bu,3,1):rns8_pack_i64(ctx,b,bv,3,1))==RNS8_SUCCESS);
      REQUIRE(rns8_gemm_rns(ctx,plan,a,b,c,workspace)==RNS8_SUCCESS);
      rns8_crt_export_desc config{};config.struct_size=sizeof(config);config.abi_version=RNS8_ABI_VERSION;
      config.algorithm=algorithm;config.mode=mode;config.limb_count=1;config.max_scratch_bytes=65536;
      rns8_crt_export* exporter=nullptr;REQUIRE(rns8_prepare_crt_export(ctx,plan,c,&config,&exporter)==RNS8_SUCCESS);
      uint64_t output[10];std::fill(std::begin(output),std::end(output),99);
      REQUIRE(rns8_export_crt_limbs(exporter,output,5)==RNS8_SUCCESS);
      for(unsigned col=0;col<3;++col){CHECK(output[col]==4);CHECK(output[5+col]==(mode==0?uint64_t(-4):mode==1?0:252));}
      CHECK(output[3]==99);CHECK(output[4]==99);CHECK(output[8]==99);CHECK(output[9]==99);
      rns8_crt_export_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_get_crt_export_info(exporter,&info)==RNS8_SUCCESS);
      CHECK(candidate.peak_payload_bytes==uint64_t(2*3+2*4+4*3+2*3*8)+info.retained_host_bytes+info.retained_device_bytes);
      REQUIRE(rns8_destroy_crt_export(exporter)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
    }
    const auto tight=candidates[2].peak_payload_bytes;q.operation.ldc=5;
    REQUIRE(rns8_plan_pipeline(ctx,&q,nullptr,0,candidates,7,&count)==RNS8_SUCCESS);
    CHECK(candidates[2].peak_payload_bytes==tight+16);
    q.payload_budget=tight;REQUIRE(rns8_plan_pipeline(ctx,&q,nullptr,0,candidates,7,&count)==RNS8_SUCCESS);
    CHECK(candidates[2].status==RNS8_WORKSPACE_TOO_SMALL);
    q.operation.ldc=INT64_MAX;CHECK(rns8_plan_pipeline(ctx,&q,nullptr,0,candidates,7,&count)==RNS8_RANGE_ERROR);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}


TEST_CASE("CPU integer export discovery prices staging and preserves unknown worker scratch", "[api][provider][memory]") {
  auto* ctx=create_cpu_context();rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
  q.operation=RNS8_OP_EXPORT;q.execution_modes=RNS8_EXEC_SYNC;q.scratch_budget=UINT64_MAX;
  for(auto semantics:{RNS8_BOUNDED_I64,RNS8_BOUNDED_U64,RNS8_EXACT_WIDE_SIGNED,RNS8_EXACT_WIDE_UNSIGNED}) {
    q.problem=semantics==RNS8_BOUNDED_I64 || semantics==RNS8_BOUNDED_U64?bounded_gemm_desc(semantics,2,3,1):exact_desc(semantics,2,3,1);
    q.problem.max_prefix=20;
    for(uint32_t limbs:{1u,3u,32u}) {
      q.output_limbs=limbs;q.scratch_budget=UINT64_MAX;rns8_operation_support support{};uint32_t count=0;
      REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);REQUIRE(count==1);
      if((semantics==RNS8_BOUNDED_I64 || semantics==RNS8_BOUNDED_U64) && limbs>1) {
        CHECK(support.status==RNS8_INVALID_ARGUMENT);CHECK(support.reason_code==RNS8_SUPPORT_BAD_DESCRIPTOR);continue;
      }
      REQUIRE(support.status==RNS8_SUCCESS);
      const uint64_t expected=6*(uint64_t(limbs)*8+sizeof(rns8_status));
      CHECK(support.required_host_bytes==expected);CHECK(support.required_device_bytes==0);CHECK(support.resource_status==1);
      q.scratch_budget=expected-1;REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);
      CHECK(support.status==RNS8_WORKSPACE_TOO_SMALL);CHECK(support.reason_code==RNS8_SUPPORT_BUDGET);
    }
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

TEST_CASE("quarter export source discovery follows ordinary residue storage producers", "[api][provider][crt]") {
  using namespace rns8::detail::api;auto* ctx=create_cpu_context();
  rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
  q.operation=RNS8_OP_QUARTER_CRT;q.execution_modes=RNS8_EXEC_SYNC;q.scratch_budget=UINT64_MAX;q.output_limbs=1;
  q.proof_flags=RNS8_QUERY_CURRENT_RANGE_PROOF|RNS8_QUERY_QUARTER_RANGE_PROOF;
  for(auto backend:{RNS8_BACKEND_CPU_REFERENCE,RNS8_BACKEND_CPU_AMX,RNS8_BACKEND_HIP_DIRECT,RNS8_BACKEND_HIPBLASLT,
                    RNS8_BACKEND_CK,RNS8_BACKEND_ROCWMMA,RNS8_BACKEND_AMDGPU_BUILTINS,RNS8_BACKEND_HIP_VECTOR_ALU_INT64,RNS8_BACKEND_WRAP64_BYTE_LIMB}) {
    q.problem=exact_desc(RNS8_EXACT_WIDE_SIGNED,2,3,1);q.problem.requested_backend=backend;q.problem.max_prefix=20;
    rns8_operation_support support{};uint32_t count=0;REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);REQUIRE(count==1);
    if(quarter_crt_backend_supported(backend) && backend_source_supports_semantics(backend,q.problem.semantics)) {
      CHECK(support.source_implemented);CHECK(support.reason_code!=RNS8_SUPPORT_OPERATION);
      CHECK(support.supported_execution_modes==RNS8_EXEC_SYNC);
    } else CHECK_FALSE(support.source_implemented);
    CHECK_FALSE(support.correctness_qualified);CHECK_FALSE(support.performance_qualified);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}


TEST_CASE("native producer pipeline queries retain explicit representation and unpriced owner boundaries", "[api][planner][mock-target]") {
  // A synthetic inventory target: this test launches no GPU, creates no HIP
  // context and attests neither runtime availability nor execution correctness.
  rns8_context target{};target.backend=RNS8_BACKEND_HIP_DIRECT;target.device_id=0;
  target.device_info.hip_available=1;std::strcpy(target.device_info.gcn_arch,"gfx1100");
  rns8_pipeline_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
  q.operation.struct_size=sizeof(q.operation);q.operation.abi_version=RNS8_ABI_VERSION;
  q.operation.operation=RNS8_OP_NARROW_NATIVE;q.operation.execution_modes=RNS8_EXEC_SYNC;q.operation.scratch_budget=UINT64_MAX;
  q.operation.problem=exact_desc(RNS8_EXACT_WIDE_SIGNED,3,5,7);q.operation.problem.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  q.operation.problem.max_prefix=9;q.operation.output_limbs=1;q.operation.input_domain=4;
  q.operation.proof_flags=RNS8_QUERY_NATIVE_RETAINED|RNS8_QUERY_NARROW_INPUT_PROOF;
  q.output_boundary=1;q.output_magnitude_limbs[0]=100;q.payload_budget=UINT64_MAX;q.reuse_count=2;
  rns8_pipeline_cost_profile profile{};profile.struct_size=sizeof(profile);profile.abi_version=RNS8_ABI_VERSION;
  profile.backend=RNS8_BACKEND_HIP_DIRECT;std::strcpy(profile.target_id,"gfx1100");std::strcpy(profile.calibration_identity,"synthetic-native-producer-regression");
  REQUIRE(rns8_get_pipeline_build_identity(profile.build_identity_sha256,sizeof(profile.build_identity_sha256))==RNS8_SUCCESS);
  profile.known_mask=63;for(unsigned i=0;i<6;++i){profile.lower_ns_per_unit[i]=1;profile.upper_ns_per_unit[i]=2;}
  std::string identities[4];
  for(uint32_t method=0;method<4;++method) {
    q.operation.algorithm=method;rns8_pipeline_candidate candidates[7]{};uint32_t count=0;
    REQUIRE(rns8_plan_pipeline(&target,&q,&profile,1,candidates,7,&count)==RNS8_SUCCESS);REQUIRE(count==7);
    for(const auto& candidate:candidates) {
      CHECK(candidate.backend==RNS8_BACKEND_HIP_DIRECT);CHECK(candidate.requires_binding);CHECK_FALSE(candidate.cost_known);CHECK_FALSE(candidate.pareto);
      if(!rns8::detail::hip_direct_compiled()){CHECK(candidate.status==RNS8_UNSUPPORTED_BACKEND);continue;}
      REQUIRE(candidate.status==RNS8_SUCCESS);bool found=false;
      for(uint32_t stage=0;stage<candidate.stage_count;++stage)if(candidate.stages[stage].kind==11+method) {
        found=true;CHECK_FALSE(candidate.stages[stage].cost_known);CHECK(candidate.stages[stage].resources==7);
        CHECK(candidate.stages[stage].work_units==3*5*7);
      }
      CHECK(found);
    }
    if(rns8::detail::hip_direct_compiled()) {
      identities[method]=candidates[0].provider_identity_sha256;
      for(uint32_t earlier=0;earlier<method;++earlier)CHECK(identities[method]!=identities[earlier]);
      const auto peak=candidates[0].peak_payload_bytes;q.payload_budget=peak-1;
      REQUIRE(rns8_plan_pipeline(&target,&q,nullptr,0,candidates,7,&count)==RNS8_SUCCESS);CHECK(candidates[0].status==RNS8_WORKSPACE_TOO_SMALL);
      q.payload_budget=UINT64_MAX;
    }
  }
  q.operation.algorithm=0;q.operation.proof_flags=RNS8_QUERY_NATIVE_RETAINED;
  rns8_pipeline_candidate candidates[7]{};uint32_t count=0;REQUIRE(rns8_plan_pipeline(&target,&q,nullptr,0,candidates,7,&count)==RNS8_SUCCESS);
  REQUIRE(count==7);for(const auto& candidate:candidates){CHECK(candidate.status==RNS8_RANGE_ERROR);CHECK(candidate.reason_code==RNS8_SUPPORT_MISSING_PROOF);}
  q.operation.proof_flags=RNS8_QUERY_NARROW_INPUT_PROOF;REQUIRE(rns8_plan_pipeline(&target,&q,nullptr,0,candidates,7,&count)==RNS8_SUCCESS);
  for(const auto& candidate:candidates)CHECK(candidate.reason_code==RNS8_SUPPORT_MISSING_PROOF);
  q.operation.proof_flags=RNS8_QUERY_NATIVE_RETAINED|RNS8_QUERY_NARROW_INPUT_PROOF;q.operation.lda=8;
  REQUIRE(rns8_plan_pipeline(&target,&q,nullptr,0,candidates,7,&count)==RNS8_SUCCESS);
  for(const auto& candidate:candidates){CHECK(candidate.status==RNS8_UNSUPPORTED_BACKEND);CHECK(candidate.reason_code==RNS8_SUPPORT_SHAPE);}
  q.operation.lda=0;q.operation.problem.semantics=RNS8_EXACT_WIDE_UNSIGNED;q.output_boundary=2;
  for(uint32_t method:{0u,2u}) {
    q.operation.algorithm=method;REQUIRE(rns8_plan_pipeline(&target,&q,nullptr,0,candidates,7,&count)==RNS8_SUCCESS);
    for(const auto& candidate:candidates){CHECK(candidate.status==RNS8_UNSUPPORTED_BACKEND);CHECK(candidate.reason_code==RNS8_SUPPORT_SEMANTICS);}
  }
  q.operation.problem.requested_backend=RNS8_BACKEND_CPU_REFERENCE;
  CHECK(rns8_plan_pipeline(&target,&q,nullptr,0,candidates,7,&count)==RNS8_UNSUPPORTED_BACKEND);CHECK(count==0);
}
