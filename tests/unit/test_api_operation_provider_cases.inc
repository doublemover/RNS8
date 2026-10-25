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

TEST_CASE("runtime shortlist separates quarter headroom from ordinary minimum and stale costs", "[api][planner]") {
  auto* ctx=create_cpu_context();rns8_pipeline_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;
  q.operation.struct_size=sizeof(q.operation);q.operation.abi_version=RNS8_ABI_VERSION;q.operation.operation=RNS8_OP_GEMM;
  q.operation.execution_modes=RNS8_EXEC_SYNC;q.operation.scratch_budget=UINT64_MAX;q.operation.problem=exact_desc(RNS8_EXACT_WIDE_SIGNED,16,16,16);
  q.operation.problem.max_prefix=9;q.output_magnitude_limbs[0]=100;q.output_boundary=1;q.payload_budget=UINT64_MAX;q.reuse_count=1;
  rns8_pipeline_candidate c[2]{};uint32_t count=0;REQUIRE(rns8_plan_pipeline(ctx,&q,nullptr,0,c,2,&count)==RNS8_SUCCESS);REQUIRE(count==2);
  CHECK(c[0].prefix==1);CHECK(c[1].prefix==2);CHECK(c[0].status==RNS8_SUCCESS);CHECK(c[1].status==RNS8_SUCCESS);
  CHECK_FALSE(c[0].cost_known);CHECK_FALSE(c[0].pareto);CHECK(c[0].executable_problem.flags&RNS8_PLAN_FORCE_FIXED_PREFIX);CHECK(c[0].executable_problem.flags&RNS8_PLAN_EXACT_VALUE_BOUND);
  for(auto& candidate:c){rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&candidate.executable_problem,&plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);}
  const auto ordinary=c[0].peak_payload_bytes,quarter=c[1].peak_payload_bytes;CHECK(quarter>ordinary);
  q.payload_budget=ordinary;REQUIRE(rns8_plan_pipeline(ctx,&q,nullptr,0,c,2,&count)==RNS8_SUCCESS);CHECK(c[0].status==RNS8_SUCCESS);CHECK(c[1].status==RNS8_WORKSPACE_TOO_SMALL);
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
