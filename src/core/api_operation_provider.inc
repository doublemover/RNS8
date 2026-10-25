// Shared static operation predicates: no probes, cache reads or GPU allocations.
// Dynamic binding remains with each owner; source/compiled/observed/qualified
// are independent dimensions, and opaque graph/library setup is never zero cost.
rns8_operation_support operation_support(const rns8_context& ctx,const rns8_operation_query& query,rns8_backend_kind backend) {
  rns8_operation_support out{};out.struct_size=sizeof(out);out.abi_version=RNS8_ABI_VERSION;out.backend=backend;out.operation=query.operation;
  out.status=RNS8_SUCCESS;out.requires_binding=1;out.alignment=hip_device_backend(backend)?256:alignof(uint64_t);
  set_text(out.target_id,sizeof(out.target_id),hip_device_backend(backend)?ctx.device_info.gcn_arch:"cpu");
  const auto reject=[&](rns8_status status,uint32_t reason,const char* field,const char* explanation){out.status=status;out.reason_code=reason;
    set_text(out.offending_field,sizeof(out.offending_field),field);set_text(out.explanation,sizeof(out.explanation),explanation);return out;};
  if(!rns8::detail::valid_abi(query.struct_size,query.abi_version,sizeof(query)) || query.operation>=RNS8_OP_COUNT ||
      !query.execution_modes || (query.execution_modes&~127u) || (query.proof_flags&~15u) || query.device_count>64 || query.output_limbs>32 || query.input_domain>5 || query.output_domain>5 || query.orientation>1 || query.source_prefix>20)
    return reject(RNS8_INVALID_ARGUMENT,RNS8_SUPPORT_BAD_DESCRIPTOR,"query","unknown ABI, operation, execution/proof flags or width");
  const auto& d=query.problem;const bool finite=uses_finite_storage(d.semantics),rns=uses_rns_storage(d.semantics);
  const uint32_t prefix=d.max_prefix?d.max_prefix:rns8::detail::default_prefix_for_semantics(d.semantics);
  const bool host_transfer=query.operation==RNS8_OP_HOST_IMPORT || query.operation==RNS8_OP_HOST_EXPORT;
  const bool native_transfer=query.operation==RNS8_OP_NATIVE_IMPORT || query.operation==RNS8_OP_NATIVE_EXPORT;
  const bool exact_lift=query.operation==RNS8_OP_BASE_EXTEND;
  // Transfer range comes from its bound/current-value proof, not an unrelated
  // full-native GEMM envelope. The concrete owner checks the strict actual bound.
  auto validation_problem=d;uint32_t validation_prefix=prefix;
  if((exact_lift || ((host_transfer || native_transfer) && (query.proof_flags&RNS8_QUERY_CURRENT_RANGE_PROOF))) && rns8::detail::exact_wide_semantics(d.semantics)){
    if(!prefix || prefix>20 || d.k!=1)return reject(RNS8_INVALID_ARGUMENT,RNS8_SUPPORT_BAD_DESCRIPTOR,"transfer_shape","transfer query requires K=1 and a concrete 1..20 active basis");
    validation_problem.max_prefix=20;validation_prefix=20;
  }
  const auto valid=rns8::detail::validate_gemm_desc(validation_problem,validation_prefix);
  if(valid!=RNS8_SUCCESS)return reject(valid,RNS8_SUPPORT_BAD_DESCRIPTOR,"problem","ordinary semantic, range, shape or basis descriptor rejected");
  const uint32_t default_domain=finite?2:d.semantics==RNS8_WRAP_U64_MOD_2_64?3:1;
  const bool native=query.operation==RNS8_OP_NATIVE_GROUP || query.operation==RNS8_OP_NARROW_NATIVE || query.operation==RNS8_OP_NATIVE_LEASE || query.operation==RNS8_OP_NATIVE_INTEGRITY || query.operation==RNS8_OP_NATIVE_IMPORT || query.operation==RNS8_OP_HOST_IMPORT || (query.operation==RNS8_OP_GRAM && query.algorithm/4==1);
  const uint32_t input_domain=query.operation==RNS8_OP_PACK?(finite?2:4):(native?4:default_domain);
  const uint32_t output_domain=(query.operation==RNS8_OP_NATIVE_EXPORT || query.operation==RNS8_OP_HOST_EXPORT)?(query.output_limbs?5:4):(query.operation==RNS8_OP_EXPORT || query.operation==RNS8_OP_NATIVE_EXPORT)?(finite?2:(rns8::detail::exact_wide_semantics(d.semantics)?5:4)):(query.operation==RNS8_OP_QUARTER_CRT?5:default_domain);
  if((query.input_domain && query.input_domain!=input_domain) || (query.output_domain && query.output_domain!=output_domain))
    return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"domain","requested representation does not match the concrete provider adapter");
  const uint32_t max_algorithm=query.operation==RNS8_OP_NARROW_NATIVE?3:query.operation==RNS8_OP_STRASSEN?9:query.operation==RNS8_OP_GRAM?11:(query.operation==RNS8_OP_GRAM || query.operation==RNS8_OP_BASE_EXTEND || query.operation==RNS8_OP_NATIVE_IMPORT || query.operation==RNS8_OP_HOST_IMPORT || query.operation==RNS8_OP_NARROW_NATIVE || query.operation==RNS8_OP_INT4?1:0);
  if(query.algorithm>max_algorithm || (query.orientation && query.operation!=RNS8_OP_GRAM))
    return reject(RNS8_INVALID_ARGUMENT,RNS8_SUPPORT_BAD_DESCRIPTOR,"algorithm","unknown concrete family method or orientation");
  const bool cpu=backend==RNS8_BACKEND_CPU_REFERENCE,hip=backend==RNS8_BACKEND_HIP_DIRECT;
  const bool amx=backend==RNS8_BACKEND_CPU_AMX;
  if(amx && query.operation!=RNS8_OP_GEMM && query.operation!=RNS8_OP_PREPARED_GROUP && query.operation!=RNS8_OP_PACK && query.operation!=RNS8_OP_EXPORT)
    return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","AMX provider integrates dense GEMM, synchronous prepared groups, host pack and shared CRT export only");
  const bool adaptive=is_per_tile_bound_kind(d.bound_kind);
  if(!backend_supports_semantics(backend,d.semantics))return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_SEMANTICS,"semantics","backend has no implementation of these arithmetic semantics");
  if((backend==RNS8_BACKEND_CK || backend==RNS8_BACKEND_ROCWMMA || backend==RNS8_BACKEND_AMDGPU_BUILTINS) && finite &&
      !rns8::detail::finite_u8::static_byte_modulus_supported(d.finite_modulus))
    return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_MODULUS,"finite_modulus","selected provider accepts only its exact finite modulus allow-list");
  if(backend==RNS8_BACKEND_HIPBLASLT && adaptive)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_SHAPE,"bound_kind","hipBLASLt provider has no adaptive tile lowering");
  out.source_implemented=1;out.supported_execution_modes=RNS8_EXEC_SYNC;
  const std::string arch=std::string(ctx.device_info.gcn_arch).substr(0,std::string(ctx.device_info.gcn_arch).find(':'));
  const bool rdna3=arch=="gfx1100" || arch=="gfx1101" || arch=="gfx1102";
  switch(query.operation) {
    case RNS8_OP_GEMM:case RNS8_OP_PREPARED_GROUP:
      out.supported_execution_modes|=RNS8_EXEC_GROUP;
      if(!amx)out.supported_execution_modes|=RNS8_EXEC_DAG;
      if(hip && !adaptive && !(d.flags&RNS8_PLAN_DIRECT_RNS_PARALLEL))out.supported_execution_modes|=RNS8_EXEC_BORROWED_STREAM|RNS8_EXEC_OWNED_GRAPH;
      break;
    case RNS8_OP_EXACT_EPILOGUE:
      if((!cpu && !hip) || !rns8::detail::exact_wide_semantics(d.semantics) || adaptive)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","exact epilogue requires uniform CPU/direct-HIP RNS lowering");
      if(d.semantics==RNS8_EXACT_WIDE_UNSIGNED && (query.alpha<0 || query.beta<0))
        return reject(RNS8_INVALID_ARGUMENT,RNS8_SUPPORT_SEMANTICS,"coefficients","unsigned exact epilogue rejects negative coefficients");
      out.supported_execution_modes|=RNS8_EXEC_GROUP|RNS8_EXEC_DAG;
      if(hip)out.supported_execution_modes|=RNS8_EXEC_BORROWED_STREAM|RNS8_EXEC_OWNED_GRAPH;break;
    case RNS8_OP_PACK:case RNS8_OP_EXPORT:break;
    case RNS8_OP_BASE_EXTEND:
      if((!cpu && !hip) || !rns8::detail::exact_wide_semantics(d.semantics))return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_SEMANTICS,"semantics","exact base extension requires CPU/direct-HIP exact-wide values");
      if(!(query.proof_flags&RNS8_QUERY_CURRENT_RANGE_PROOF))return reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_MISSING_PROOF,"range","base extension requires current source uniqueness proof");
      if(!query.source_prefix || query.source_prefix>prefix)return reject(RNS8_INVALID_ARGUMENT,RNS8_SUPPORT_BAD_DESCRIPTOR,"source_prefix","base extension requires a concrete source no wider than its target");
      if(hip)out.supported_execution_modes|=RNS8_EXEC_BORROWED_STREAM|RNS8_EXEC_OWNED_GRAPH|RNS8_EXEC_GROUP|RNS8_EXEC_DAG;break;
    case RNS8_OP_NATIVE_INTEGRITY:
      if(!hip || !rns8::detail::exact_wide_semantics(d.semantics) || adaptive || (d.flags&RNS8_PLAN_DIRECT_RNS_PARALLEL))
        return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","native integrity requires uniform direct-HIP exact-wide prepared GEMM");
      if((query.proof_flags&(RNS8_QUERY_NATIVE_RETAINED|RNS8_QUERY_CURRENT_RANGE_PROOF))!=(RNS8_QUERY_NATIVE_RETAINED|RNS8_QUERY_CURRENT_RANGE_PROOF))
        return reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_MISSING_PROOF,"native_bound","native integrity requires retained current native inputs and independent product bound");
      out.supported_execution_modes|=RNS8_EXEC_BORROWED_STREAM|RNS8_EXEC_OWNED_GRAPH|RNS8_EXEC_GROUP|RNS8_EXEC_DAG;break;
    case RNS8_OP_NATIVE_GROUP:
      if(!hip || !rns8::detail::exact_wide_semantics(d.semantics) || adaptive)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","native group requires uniform direct-HIP exact-wide plans");
      if(!(query.proof_flags&RNS8_QUERY_NATIVE_RETAINED))return reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_MISSING_PROOF,"native_storage","native group requires retained current native inputs");
      if(!rdna3)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_TARGET,"target","current native WMMA queue code object family is gfx1100/1101/1102");
      out.supported_execution_modes|=RNS8_EXEC_BORROWED_STREAM|RNS8_EXEC_OWNED_GRAPH|RNS8_EXEC_GROUP;break;
    case RNS8_OP_STRUCTURED_SPARSE_A:
      if(!cpu && backend!=RNS8_BACKEND_AMDGPU_BUILTINS)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","structured sparse-A requires its CPU or builtin sparse provider");
      if(!cpu){const auto status=rns8::detail::amdgpu_builtins_sparse_a_contract(ctx.device_info.gcn_arch,d.m,d.n,d.k,
          query.ldb?query.ldb:d.n,query.ldc?query.ldc:d.n,finite?1:prefix,finite);
        if(status!=RNS8_SUCCESS)return reject(status,RNS8_SUPPORT_TARGET,"sparse_target_shape","shared sparse provider rejected target, address or sparse shape");}break;
    case RNS8_OP_BLOCK_SPARSE:
      if(!hip || !rns8::detail::exact_wide_semantics(d.semantics))
        return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","public block-sparse owner requires direct-HIP exact-wide semantics");break;
    case RNS8_OP_STRASSEN:
      if(!hip || !rns || adaptive)
        return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","bilinear owner requires uniform direct-HIP bounded/exact-wide RNS semantics");
      if(query.operation==RNS8_OP_STRASSEN && query.algorithm>=5) {
        if(!rns8::detail::amdgpu_builtins_compiled())return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_NOT_COMPILED,"matrix_subproducts","AMDGPU matrix-engine provider was not compiled");
        if(!rdna3 && arch!="gfx942" && arch!="gfx1200" && arch!="gfx1201")return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_TARGET,"matrix_subproducts","no matrix-engine subproduct target provider");
        const int64_t blocks=query.algorithm%5==0?2:3;
        if((d.k-1)/blocks+1>RNS8_SAFE_INT32_K_BLOCK)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_SHAPE,"subproduct_k","matrix-engine subproduct K exceeds its certified INT32 cap");
      }break;
    case RNS8_OP_GRAM:
      if(!cpu && !hip)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","operation has CPU/direct-HIP providers only");
      if(!rns && !finite)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_SEMANTICS,"semantics","operation has no wrap/native-vector lowering");
      if(query.operation==RNS8_OP_GRAM && !finite && d.semantics!=RNS8_EXACT_WIDE_SIGNED && d.semantics!=RNS8_BOUNDED_I64)
        return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_SEMANTICS,"semantics","Gram provider requires signed or finite arithmetic");
      if(d.m!=d.n)return reject(RNS8_INVALID_ARGUMENT,RNS8_SUPPORT_SHAPE,"shape","Gram query is normalized as square output M=N, reduction K");
      if((query.algorithm&1) && (!hip || !rdna3))return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_TARGET,"algorithm","current Gram WMMA provider requires gfx11 direct HIP");
      if(query.algorithm/4==1 && !(query.proof_flags&RNS8_QUERY_NATIVE_RETAINED))return reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_MISSING_PROOF,"native_storage","native Gram requires current retained signed64 input");break;
    case RNS8_OP_INT4:
      if(!hip || !finite || d.finite_modulus>16)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_MODULUS,"finite_modulus","INT4 is an explicit direct-HIP finite 2..16 representation, not byte RNS truncation");
      if(!rdna3)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_TARGET,"target","INT4 WMMA provider requires gfx1100/1101/1102");break;
    case RNS8_OP_NATIVE_IMPORT:case RNS8_OP_NATIVE_EXPORT:case RNS8_OP_HOST_IMPORT:case RNS8_OP_HOST_EXPORT:
      if(!hip || !rns8::detail::exact_wide_semantics(d.semantics) || adaptive)
        return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","prepared native transfer requires uniform direct-HIP exact-wide values");
      if(!(query.proof_flags&RNS8_QUERY_CURRENT_RANGE_PROOF))return reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_MISSING_PROOF,"range","native transfer requires a type/device-checked import envelope or current export proof");
      out.supported_execution_modes|=RNS8_EXEC_BORROWED_STREAM|RNS8_EXEC_OWNED_GRAPH|RNS8_EXEC_GROUP|RNS8_EXEC_DAG;break;
    case RNS8_OP_NATIVE_LEASE:
      if(!hip || !rns8::detail::exact_wide_semantics(d.semantics))return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","native lease requires direct-HIP exact-wide context");break;
    case RNS8_OP_NARROW_NATIVE:
      if(!hip || !rns8::detail::exact_wide_semantics(d.semantics) || (!(query.algorithm&1) && d.semantics!=RNS8_EXACT_WIDE_SIGNED))return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_SEMANTICS,"semantics","narrow native provider requires direct-HIP signed semantics");
      if((query.algorithm&1) && !rdna3)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_TARGET,"algorithm","current byte-WMMA narrow provider requires gfx11");
      if(!(query.proof_flags&RNS8_QUERY_NATIVE_RETAINED))return reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_MISSING_PROOF,"native_storage","narrow input requires retained current native storage");
      if(!(query.proof_flags&RNS8_QUERY_NARROW_INPUT_PROOF))return reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_MISSING_PROOF,"input_range","narrow arithmetic requires an actual input partial-sum proof");break;
    case RNS8_OP_QUARTER_CRT:
      if(!cpu && !hip)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","quarter owner supports CPU or direct HIP only");
      if(!rns8::detail::exact_wide_semantics(d.semantics))return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_SEMANTICS,"semantics","quarter CRT requires exact-wide sources");
      if((query.proof_flags&(RNS8_QUERY_CURRENT_RANGE_PROOF|RNS8_QUERY_QUARTER_RANGE_PROOF))!=(RNS8_QUERY_CURRENT_RANGE_PROOF|RNS8_QUERY_QUARTER_RANGE_PROOF))
        return reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_MISSING_PROOF,"source_bound","quarter CRT requires current engine-owned 4B<=M and width proof");break;
    case RNS8_OP_MULTI_DEVICE:case RNS8_OP_EXPERIMENTAL_BASIS:case RNS8_OP_FP8:
      out.source_implemented=0;return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","no integrated provider for this operation in this implementation revision");
    default:return reject(RNS8_INVALID_ARGUMENT,RNS8_SUPPORT_BAD_DESCRIPTOR,"operation","unknown operation");
  }
  if(query.device_count>1 || (query.execution_modes&RNS8_EXEC_MULTI_DEVICE))return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_TOPOLOGY,"device_count","this provider owns one device and does not imply transport support");
  if(query.execution_modes&~out.supported_execution_modes)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_EXECUTION_MODE,"execution_modes","requested execution/capture mode has no matching owner; no silent synchronous fallback");
  const bool importing_native=query.operation==RNS8_OP_NATIVE_IMPORT || query.operation==RNS8_OP_HOST_IMPORT;
  const int64_t lda=query.lda?query.lda:(importing_native?d.n:d.k),ldb=query.ldb?query.ldb:d.n,ldc=query.ldc?query.ldc:d.n;
  const uint32_t planes=finite?1:d.semantics==RNS8_WRAP_U64_MOD_2_64?8:std::max(1u,prefix);
  if(query.operation!=RNS8_OP_GRAM && (!valid_matrix_access(d.m,d.k,lda,1) || !valid_matrix_access(d.k,d.n,ldb,1) || !valid_matrix_access(d.m,d.n,ldc,1)))
    return reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_ADDRESS,"strides","complete padded host address domain overflows");
  if(query.operation!=RNS8_OP_GRAM && query.operation!=RNS8_OP_STRASSEN && !cpu && !amx && !host_transfer && backend!=RNS8_BACKEND_WRAP64_BYTE_LIMB && !rns8::detail::qualified_i32_gemm_address(d.m,d.n,d.k,lda,ldb,ldc,planes))
    return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_ADDRESS,"strides","complete padded device span exceeds this provider's signed32 index domain");
  rns8_backend_capability_info capability{};capability.struct_size=sizeof(capability);capability.abi_version=RNS8_ABI_VERSION;
  fill_backend_capability_info(backend,capability);out.compiled_inventory=capability.compiled_kernel_available && capability.is_available;
  out.runtime_observed=context_accepts_backend(ctx,backend) && (!hip_device_backend(backend) || ctx.device_info.hip_available);
  if(!out.compiled_inventory)return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_NOT_COMPILED,"backend","provider source exists but is not compiled in this build");
  if(ctx.hip_failure_poisoned)return reject(RNS8_BACKEND_FAILURE,RNS8_SUPPORT_RUNTIME,"context","context has an undrained asynchronous failure");
  rns8_plan temporary{};temporary.desc=d;temporary.prefix=prefix;temporary.backend=backend;
  out.safe_k=accumulator_k_block_size_for_plan(temporary);
  const auto add=[&](uint64_t& total,uint64_t value){if(value>UINT64_MAX-total)return false;total+=value;return true;};
  uint64_t scratch=0;std::size_t a=0,b=0,c=0,total=0;
  const bool gemm_workspace=query.operation==RNS8_OP_GEMM || query.operation==RNS8_OP_PREPARED_GROUP;
  if(gemm_workspace && amx){if(!rns8::detail::cpu_amx_workspace_bytes(d.m,d.n,d.k,finite?1:prefix,rns8::detail::cpu_amx_workers(),scratch))out.resource_status=2;}
  else if(gemm_workspace && backend==RNS8_BACKEND_HIPBLASLT){if(!rns8::detail::hipblaslt_baseline_workspace_requirements(d.m,d.n,d.k,a,b) || !add(scratch,a) || !add(scratch,b))out.resource_status=2;}
  else if(gemm_workspace && backend==RNS8_BACKEND_CK){if(!rns8::detail::ck_workspace_requirements(d.m,d.n,d.k,a,b,c,total))out.resource_status=2;else scratch=total;}
  else if(gemm_workspace && backend==RNS8_BACKEND_ROCWMMA){if(!rns8::detail::rocwmma_workspace_requirements(d.m,d.n,d.k,a,b,total))out.resource_status=2;else scratch=total;}
  if(query.operation==RNS8_OP_EXPORT || query.operation==RNS8_OP_QUARTER_CRT){
    const uint64_t limbs=query.output_limbs?query.output_limbs:(rns8::detail::exact_wide_semantics(d.semantics)?3:1);
    const uint64_t cell_bytes=finite?1:limbs*8;
    if(uint64_t(d.m)>UINT64_MAX/uint64_t(d.n)/cell_bytes)out.resource_status=2;
    else{const uint64_t output_bytes=uint64_t(d.m)*uint64_t(d.n)*cell_bytes;
      if(query.operation==RNS8_OP_QUARTER_CRT)out.required_host_bytes=output_bytes;
      const bool status_word=query.operation==RNS8_OP_EXPORT && rns && (!rns8::detail::exact_wide_semantics(d.semantics) || limbs<3);
      if(hip_device_backend(backend) && (!add(scratch,output_bytes) || !add(scratch,status_word?4:0)))out.resource_status=2;
      // Ordinary export may use an environment-controlled retained pinned pool;
      // its allocation/current capacity is intentionally not queried here.
      if(query.operation==RNS8_OP_EXPORT && hip_device_backend(backend) && out.resource_status!=2)out.resource_status=1;}}
  if(query.operation==RNS8_OP_NARROW_NATIVE){const auto support=rns8::detail::query_narrow_rns_support(d.m,d.n,d.k,lda,ldb,prefix,bool(query.algorithm&1),UINT64_MAX,query.algorithm>=2);
    if(support.status!=RNS8_SUCCESS)return reject(support.status,RNS8_SUPPORT_ADDRESS,"narrow_resources","shared narrow provider rejected complete shape/address domain");scratch=support.scratch_bytes;}
  if(query.operation==RNS8_OP_STRASSEN){const auto algorithm=query.algorithm%5;const auto resources=algorithm>=2?rns8::detail::query_rank23_slp_resources(d.m,d.n,d.k,lda,ldb,ldc,prefix,UINT64_MAX,algorithm==4):
      rns8::detail::query_bilinear_resources(d.m,d.n,d.k,lda,ldb,ldc,prefix,algorithm?3:2,UINT64_MAX);
    if(resources.status!=RNS8_SUCCESS)return reject(resources.status,RNS8_SUPPORT_ADDRESS,"bilinear_resources","shared bilinear provider rejected shape/address/resource domain");scratch=resources.total_bytes;}
  if(query.operation==RNS8_OP_BASE_EXTEND && hip && query.source_prefix){if(d.m>INT64_MAX/d.n)return reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_ADDRESS,"cells","lift cell count overflow");
    if(query.source_prefix!=prefix){const auto resources=rns8::detail::query_exact_lift_support(d.m*d.n,query.source_prefix,prefix,UINT64_MAX);
      if(resources.status!=RNS8_SUCCESS)return reject(resources.status,RNS8_SUPPORT_ADDRESS,"lift_resources","shared exact lift rejected source/target envelope");scratch=resources.scratch_bytes;}
    if(query.execution_modes&~uint32_t(RNS8_EXEC_SYNC)){out.required_host_bytes=4;out.required_pinned_host_bytes=4;}}
  if(query.operation==RNS8_OP_GRAM){rns8::detail::GramProblem p{};p.source_instance=1;p.orientation=static_cast<rns8::detail::GramOrientation>(query.orientation);p.semantics=d.semantics;
    p.rows=query.orientation?d.m:d.k;p.cols=query.orientation?d.k:d.n;p.source_ld=query.lda?query.lda:p.cols;p.output_ld=ldc;p.prefix=finite?1:prefix;p.modulus=finite?d.finite_modulus:0;
    const auto support=rns8::detail::query_gram_support(p,UINT64_MAX,hip,bool(query.algorithm&1));
    if(support.status!=RNS8_SUCCESS)return reject(support.status,RNS8_SUPPORT_ADDRESS,"gram_resources","shared Gram provider rejected orientation/shape/address domain");
    const auto storage=rns8::detail::query_gram_prepared_storage(p,hip,bool(query.algorithm&1),query.algorithm/4,query.output_limbs,UINT64_MAX);
    if(storage.status!=RNS8_SUCCESS)return reject(storage.status,RNS8_SUPPORT_ADDRESS,"gram_prepared_resources","prepared adapter/export address or semantic domain rejected");
    out.required_host_bytes=storage.host_bytes;scratch=storage.device_bytes;}
  if(query.operation==RNS8_OP_NATIVE_IMPORT || query.operation==RNS8_OP_NATIVE_EXPORT){
    const bool importing=query.operation==RNS8_OP_NATIVE_IMPORT;const uint64_t width=importing?8:8*uint64_t(query.output_limbs?query.output_limbs:1);
    if(!rns8::detail::qualified_i32_matrix_address(d.m,d.n,importing?lda:ldc,1,width))return reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_ADDRESS,"native_span","complete native scalar/limb view exceeds address domain");
    const uint64_t cells=uint64_t(d.m)*d.n;scratch=importing?cells*prefix+16:cells*width+4;
    if(importing && (query.proof_flags&RNS8_QUERY_NATIVE_RETAINED))if(!add(scratch,cells*8))out.resource_status=2;
    out.required_host_bytes=4;out.required_pinned_host_bytes=4;
  }
  if(host_transfer){
    const bool importing=query.operation==RNS8_OP_HOST_IMPORT;rns8_pinned_host_desc hd{};hd.struct_size=sizeof(hd);hd.abi_version=RNS8_ABI_VERSION;
    const bool sign=d.semantics==RNS8_EXACT_WIDE_SIGNED;hd.element_type=query.output_limbs?(sign?RNS8_NATIVE_DEVICE_SIGNED_LIMBS:RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS):(sign?RNS8_NATIVE_DEVICE_I64:RNS8_NATIVE_DEVICE_U64);
    hd.access=importing?RNS8_NATIVE_DEVICE_READ:RNS8_NATIVE_DEVICE_WRITE;hd.rows=d.m;hd.cols=d.n;hd.ld=importing?lda:ldc;hd.limb_count=query.output_limbs;
    rns8::detail::PinnedHostResources resources;const auto status=rns8::detail::query_pinned_host_contract(hd,resources);
    if(status!=RNS8_SUCCESS)return reject(status,RNS8_SUPPORT_ADDRESS,"pinned_rectangle","complete padded host rectangle or compact native device span rejected");
    uint64_t pinned=0,total=0;const bool retain=importing && (query.proof_flags&RNS8_QUERY_NATIVE_RETAINED);
    if(!rns8::detail::pinned_host_transfer_bytes(resources,prefix,importing,retain,scratch,pinned,total))out.resource_status=2;
    else {out.required_host_bytes=out.required_pinned_host_bytes=pinned;if(retain)scratch+=uint64_t(d.m)*d.n*8;}
  }
  if(query.operation==RNS8_OP_NATIVE_INTEGRITY){scratch=4;out.required_host_bytes=4;out.required_pinned_host_bytes=4;out.safe_k=1;}
  if(query.operation==RNS8_OP_INT4){out.safe_k=65536;out.setup_bytes_lower_bound=4;
    if(query.algorithm){const uint64_t blocks=(uint64_t(d.k)+15)/16;
      if(blocks>UINT64_MAX/uint64_t(d.n)/8)out.resource_status=2;else scratch=blocks*uint64_t(d.n)*8;}}
  // Events, graph executables, opaque vendor allocations and specialized
  // directories have no portable sizeof. Expose the known arithmetic lower
  // bound and unknown setup instead of pretending those owners consume zero.
  if((query.execution_modes&~uint32_t(RNS8_EXEC_SYNC)) || adaptive ||
      (query.operation!=RNS8_OP_GEMM && query.operation!=RNS8_OP_EXPORT && query.operation!=RNS8_OP_QUARTER_CRT))
    if(out.resource_status!=2)out.resource_status=1;
  if(hip_device_backend(backend))out.required_device_bytes=scratch;else if(!add(out.required_host_bytes,scratch))out.resource_status=2;
  if(out.resource_status==2)return reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_RESOURCE_OVERFLOW,"resources","checked provider resource arithmetic overflowed");
  uint64_t aggregate=out.required_host_bytes;if(!add(aggregate,out.required_device_bytes))return reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_RESOURCE_OVERFLOW,"resources","host/device aggregate overflowed");
  aggregate=std::max(aggregate,out.setup_bytes_lower_bound);
  if(aggregate>query.scratch_budget)return reject(RNS8_WORKSPACE_TOO_SMALL,RNS8_SUPPORT_BUDGET,"scratch_budget","known required scratch exceeds budget; unknown setup is additional");
  std::string identity="operation-provider-v1;backend="+std::to_string(backend)+";operation="+std::to_string(query.operation)+";modes="+std::to_string(query.execution_modes)+
    ";algorithm="+std::to_string(query.algorithm)+";orientation="+std::to_string(query.orientation)+";sourceprefix="+std::to_string(query.source_prefix)+
    ";shape="+std::to_string(d.m)+","+std::to_string(d.n)+","+std::to_string(d.k)+";strides="+std::to_string(lda)+","+std::to_string(ldb)+","+std::to_string(ldc)+
    ";semantics="+std::to_string(d.semantics)+";prefix="+std::to_string(prefix)+";modulus="+std::to_string(d.finite_modulus)+";flags="+std::to_string(d.flags)+
    ";boundkind="+std::to_string(d.bound_kind)+";bound="+std::to_string(d.bound)+";lhs="+std::to_string(d.lhs_bound)+";rhs="+std::to_string(d.rhs_bound)+
    ";tile="+std::to_string(d.tile_m)+","+std::to_string(d.tile_n)+";proof="+std::to_string(query.proof_flags)+";domains="+std::to_string(input_domain)+","+std::to_string(output_domain)+
    ";coefficients="+std::to_string(query.alpha)+","+std::to_string(query.beta)+";width="+std::to_string(query.output_limbs)+";budget="+std::to_string(query.scratch_budget)+
    ";target="+out.target_id+";runtime="+std::to_string(ctx.device_info.hip_runtime_version)+";library="+capability.library_version;
#ifdef RNS8_BUILD_MANIFEST_JSON
  identity+=RNS8_BUILD_MANIFEST_JSON;
#endif
  for(uint64_t i=0;i<d.tile_bounds_count;++i)identity+=";tilebound="+std::to_string(d.tile_bounds[i]);
  for(uint64_t i=0;i<d.zero_a_rows_count;++i)identity+=";zeroa="+std::to_string(d.zero_a_rows[i]);
  for(uint64_t i=0;i<d.zero_b_cols_count;++i)identity+=";zerob="+std::to_string(d.zero_b_cols[i]);
  set_text(out.provider_identity_sha256,sizeof(out.provider_identity_sha256),rns8::detail::sha256(identity));
  set_text(out.explanation,sizeof(out.explanation),"statically compatible; actual value/alias/generation/code-object/API binding remains required; no qualification inferred");
  return out;
}
rns8_status plan_operation_preflight(const rns8_context& ctx,const rns8_plan& plan,uint32_t operation,uint32_t modes,uint32_t proof_flags) {
  rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;q.operation=operation;q.execution_modes=modes;q.proof_flags=proof_flags;
  q.problem=plan.desc;q.problem.requested_backend=plan.backend;q.scratch_budget=UINT64_MAX;q.device_count=1;
  // Plans own copied tile bounds; the descriptor's external pointers are cleared.
  q.problem.tile_bounds=plan.tile_bounds.empty()?nullptr:plan.tile_bounds.data();q.problem.tile_bounds_count=plan.tile_bounds.size();
  q.problem.zero_a_rows=plan.zero_a_rows.empty()?nullptr:plan.zero_a_rows.data();q.problem.zero_a_rows_count=plan.zero_a_rows.size();
  q.problem.zero_b_cols=plan.zero_b_cols.empty()?nullptr:plan.zero_b_cols.data();q.problem.zero_b_cols_count=plan.zero_b_cols.size();
  return operation_support(ctx,q,plan.backend).status;
}
