// Runtime shortlist consumes the same provider predicates as plan admission.
// No cache writes, probes, allocation on a GPU, dispatch or evidence promotion.
namespace {
std::string pipeline_build_identity() {
  std::string identity="rns8-pipeline-provider-v1;";
#ifdef RNS8_BUILD_MANIFEST_JSON
  identity+=RNS8_BUILD_MANIFEST_JSON;
#else
  identity+="build-manifest-unavailable;";
#ifdef __VERSION__
  identity+=__VERSION__;
#elif defined(_MSC_VER)
  identity+="msvc-"+std::to_string(_MSC_VER);
#endif
#endif
  return rns8::detail::sha256(identity);
}
bool pipeline_text(const char* text,size_t capacity) {return std::memchr(text,0,capacity)!=nullptr;}
}
rns8_status rns8_get_pipeline_build_identity(char* destination,uint64_t capacity) {
  return guard_api([&]() -> rns8_status {if(!destination || capacity<65)return RNS8_INVALID_ARGUMENT;
    const auto identity=pipeline_build_identity();std::memcpy(destination,identity.c_str(),65);return RNS8_SUCCESS;});
}
rns8_status rns8_plan_pipeline(const rns8_context* ctx,const rns8_pipeline_query* query,
    const rns8_pipeline_cost_profile* profiles,uint32_t profile_count,rns8_pipeline_candidate* results,uint32_t capacity,uint32_t* required) {
  return guard_api([&]() -> rns8_status {
    using boost::multiprecision::cpp_int;
    using namespace rns8::detail;
    if(!ctx || !query || !required || (capacity && !results) || (profile_count && !profiles) || profile_count>8 ||
       !valid_abi(query->struct_size,query->abi_version,sizeof(*query)) ||
       !valid_abi(query->operation.struct_size,query->operation.abi_version,sizeof(query->operation)) ||
       query->operation.operation!=RNS8_OP_GEMM || query->operation.execution_modes!=RNS8_EXEC_SYNC ||
       query->operation.output_limbs>32 || query->output_boundary>3 || query->retained_operands>3 || !query->reuse_count)return RNS8_INVALID_ARGUMENT;
    *required=0;
    const auto& input=query->operation.problem;
    if(!exact_wide_semantics(input.semantics) || is_per_tile_bound_kind(input.bound_kind))return RNS8_UNSUPPORTED_BACKEND;
    const uint32_t ceiling=input.max_prefix?input.max_prefix:default_prefix_for_semantics(input.semantics);
    const auto valid=validate_gemm_desc(input,ceiling);if(valid!=RNS8_SUCCESS)return valid;
    if((query->output_boundary==1 && input.semantics!=RNS8_EXACT_WIDE_SIGNED) ||
       (query->output_boundary==2 && input.semantics!=RNS8_EXACT_WIDE_UNSIGNED))return RNS8_INVALID_ARGUMENT;
    const auto build=pipeline_build_identity();
    for(uint32_t i=0;i<profile_count;++i){const auto& p=profiles[i];
      if(!valid_abi(p.struct_size,p.abi_version,sizeof(p)) || p.known_mask>63 || !pipeline_text(p.target_id,sizeof(p.target_id)) ||
         !pipeline_text(p.build_identity_sha256,sizeof(p.build_identity_sha256)) || !pipeline_text(p.calibration_identity,sizeof(p.calibration_identity)))return RNS8_INVALID_ARGUMENT;
      for(uint32_t j=0;j<i;++j)if(profiles[j].backend==p.backend)return RNS8_INVALID_ARGUMENT;
      if(!std::isfinite(p.lower_launch_ns) || !std::isfinite(p.upper_launch_ns) || p.lower_launch_ns<0 || p.upper_launch_ns<p.lower_launch_ns)return RNS8_INVALID_ARGUMENT;
      for(unsigned j=0;j<6;++j)if(p.known_mask&(1u<<j))
        if(!std::isfinite(p.lower_ns_per_unit[j]) || !std::isfinite(p.upper_ns_per_unit[j]) || p.lower_ns_per_unit[j]<=0 || p.upper_ns_per_unit[j]<p.lower_ns_per_unit[j])return RNS8_INVALID_ARGUMENT;
    }
    cpp_int bound=0;for(int i=2;i>=0;--i){bound<<=64;bound+=query->output_magnitude_limbs[i];}
    uint32_t minimum=0,quarter=0;for(uint32_t p=1;p<=ceiling;++p){const auto product=modulus_product(p);
      if(!minimum && exact_range_fits(input.semantics,bound,product))minimum=p;
      if(!quarter && bound*4<=product)quarter=p;
    }
    if(!minimum)return RNS8_RANGE_ERROR;
    if(input.flags&RNS8_PLAN_FORCE_FIXED_PREFIX){minimum=ceiling;quarter=bound*4<=modulus_product(ceiling)?ceiling:0;}
    const uint32_t limbs=query->operation.output_limbs?query->operation.output_limbs:3;
    if(query->output_boundary==1 && bound>=(cpp_int(1)<<(limbs*64-1)))return RNS8_RANGE_ERROR;
    if(query->output_boundary==2 && bound>=(cpp_int(1)<<(limbs*64)))return RNS8_RANGE_ERROR;
    std::vector<rns8_pipeline_candidate> candidates;
    constexpr rns8_backend_kind backends[]{RNS8_BACKEND_CPU_REFERENCE,RNS8_BACKEND_CPU_AMX,RNS8_BACKEND_HIP_DIRECT,RNS8_BACKEND_HIPBLASLT,RNS8_BACKEND_CK,
      RNS8_BACKEND_ROCWMMA,RNS8_BACKEND_WRAP64_BYTE_LIMB,RNS8_BACKEND_HIP_VECTOR_ALU_INT64,RNS8_BACKEND_AMDGPU_BUILTINS};
    for(auto backend:backends){if(input.requested_backend!=RNS8_BACKEND_AUTO && input.requested_backend!=backend)continue;
      for(unsigned reconstruction=0;reconstruction<2;++reconstruction){
        if(reconstruction && (!query->output_boundary || !quarter || (backend!=RNS8_BACKEND_CPU_REFERENCE && backend!=RNS8_BACKEND_HIP_DIRECT)))continue;
        rns8_pipeline_candidate c{};c.struct_size=sizeof(c);c.abi_version=RNS8_ABI_VERSION;c.backend=backend;
        c.prefix=reconstruction?quarter:minimum;c.reconstruction=query->output_boundary?(reconstruction?2:1):0;c.requires_binding=1;
        auto q=query->operation;q.problem.max_prefix=c.prefix;q.problem.requested_backend=backend;q.problem.flags|=RNS8_PLAN_FORCE_FIXED_PREFIX;
        c.executable_problem=q.problem;
        // Caller-owned pointers cannot escape in a saved executable recipe.
        c.executable_problem.tile_bounds=nullptr;c.executable_problem.tile_bounds_count=0;
        c.executable_problem.zero_a_rows=nullptr;c.executable_problem.zero_a_rows_count=0;
        c.executable_problem.zero_b_cols=nullptr;c.executable_problem.zero_b_cols_count=0;
        auto support=operation_support(*ctx,q,backend);c.status=support.status;c.reason_code=support.reason_code;c.resource_status=support.resource_status;
        set_text(c.provider_identity_sha256,sizeof(c.provider_identity_sha256),support.provider_identity_sha256);
        set_text(c.explanation,sizeof(c.explanation),support.explanation);
        if(c.status!=RNS8_SUCCESS){candidates.push_back(c);continue;}
        if(query->output_boundary==3 && modulus_product(c.prefix)>(cpp_int(1)<<(limbs*64))){c.status=RNS8_RANGE_ERROR;c.reason_code=RNS8_SUPPORT_MISSING_PROOF;candidates.push_back(c);continue;}
        const bool gpu=hip_device_backend(backend);
        const auto* profile=static_cast<const rns8_pipeline_cost_profile*>(nullptr);
        for(uint32_t i=0;i<profile_count;++i)if(profiles[i].backend==uint32_t(backend) && profiles[i].target_id==std::string(support.target_id) &&
            profiles[i].build_identity_sha256==build && profiles[i].calibration_identity[0])profile=&profiles[i];
        c.cost_known=profile!=nullptr;
        const auto product=[&](uint64_t a,uint64_t b,uint64_t factor)->uint64_t {
          if(!b || a>UINT64_MAX/b || a*b>UINT64_MAX/factor){c.status=RNS8_RANGE_ERROR;c.reason_code=RNS8_SUPPORT_RESOURCE_OVERFLOW;return 0;}return a*b*factor;};
        const uint64_t acells=product(input.m,input.k,1),bcells=product(input.k,input.n,1),ccells=product(input.m,input.n,1);
        const uint64_t abytes=product(acells,c.prefix,1),bbytes=product(bcells,c.prefix,1),cbytes=product(ccells,c.prefix,1);
        const uint64_t native_a=product(acells,8,1),native_b=product(bcells,8,1),output=product(ccells,limbs,8);
        const uint64_t ops=product(ccells,input.k,c.prefix);
        if(c.status!=RNS8_SUCCESS){candidates.push_back(c);continue;}
        // Robust capacity bound: all owned payloads simultaneously live. This
        // remains valid at interior calibration rates, unlike endpoint peaks.
        uint64_t peak=support.required_device_bytes;
        const auto reserve=[&](uint64_t bytes){if(bytes>UINT64_MAX-peak){c.status=RNS8_RANGE_ERROR;c.reason_code=RNS8_SUPPORT_RESOURCE_OVERFLOW;}else peak+=bytes;};
        reserve(support.required_host_bytes);reserve(cbytes);
        if(!(query->retained_operands&1)){reserve(abytes);if(gpu)reserve(native_a);}
        if(!(query->retained_operands&2)){reserve(bbytes);if(gpu)reserve(native_b);}
        if(query->output_boundary){reserve(output);if(gpu)reserve(output);if(gpu && !reconstruction)reserve(4);}
        c.peak_payload_bytes=peak;c.setup_payload_bytes=peak-cbytes;
        if(peak>query->payload_budget){c.status=RNS8_WORKSPACE_TOO_SMALL;c.reason_code=RNS8_SUPPORT_BUDGET;}
        if(c.status!=RNS8_SUCCESS){candidates.push_back(c);continue;}
        double low_resource[3]{},high_resource[3]{};double low_setup=0,high_setup=0;
        const auto stage=[&](uint32_t kind,uint32_t resources,uint32_t deps,uint64_t work,uint64_t bytes,bool setup){
          auto& s=c.stages[c.stage_count];s.kind=kind;s.resources=resources;s.dependencies=deps;s.work_units=work;s.allocated_bytes=bytes;
          s.cost_known=profile && (profile->known_mask&(1u<<kind));c.cost_known&=s.cost_known;
          double lo=0,hi=0;if(s.cost_known){lo=profile->lower_launch_ns+double(work)*profile->lower_ns_per_unit[kind];hi=profile->upper_launch_ns+double(work)*profile->upper_ns_per_unit[kind];
            if(!std::isfinite(lo) || !std::isfinite(hi)){s.cost_known=0;c.cost_known=0;lo=hi=0;}}
          for(uint32_t i=0;i<c.stage_count;++i)if(deps&(1u<<i)){s.earliest_start_ns=std::max(s.earliest_start_ns,c.stages[i].earliest_end_ns);s.latest_start_ns=std::max(s.latest_start_ns,c.stages[i].latest_end_ns);}
          for(unsigned r=0;r<3;++r)if(resources&(1u<<r)){s.earliest_start_ns=std::max(s.earliest_start_ns,low_resource[r]);s.latest_start_ns=std::max(s.latest_start_ns,high_resource[r]);}
          s.earliest_end_ns=s.earliest_start_ns+lo;s.latest_end_ns=s.latest_start_ns+hi;
          for(unsigned r=0;r<3;++r)if(resources&(1u<<r)){low_resource[r]=s.earliest_end_ns;high_resource[r]=s.latest_end_ns;}
          if(setup){low_setup=std::max(low_setup,s.earliest_end_ns);high_setup=std::max(high_setup,s.latest_end_ns);}++c.stage_count;
        };
        uint32_t inputs=0;
        for(unsigned operand=0;operand<2;++operand)if(!(query->retained_operands&(1u<<operand))){uint32_t dependency=0;
          if(gpu){stage(0,2,0,operand?native_b:native_a,operand?native_b:native_a,true);dependency=1u<<(c.stage_count-1);}
          stage(1,gpu?4:1,dependency,operand?bbytes:abytes,operand?bbytes:abytes,true);inputs|=1u<<(c.stage_count-1);}
        stage(2,gpu?4:1,inputs,ops,cbytes,false);
        if(query->output_boundary){stage(reconstruction?4:3,gpu?4:1,1u<<(c.stage_count-1),cbytes,output,false);
          if(gpu)stage(5,2,1u<<(c.stage_count-1),output,output,false);}
        if(c.cost_known){const auto& last=c.stages[c.stage_count-1];c.lower_amortized_ns=last.earliest_end_ns-low_setup+low_setup/query->reuse_count;
          c.upper_amortized_ns=last.latest_end_ns-high_setup+high_setup/query->reuse_count;
          set_text(c.explanation,sizeof(c.explanation),"non-promoting calibrated shortlist; enforced resource order; robust simultaneous-payload bound; actual values and opaque setup still bind");}
        else {for(auto& stage:c.stages){stage.earliest_start_ns=stage.earliest_end_ns=stage.latest_start_ns=stage.latest_end_ns=0;}
          set_text(c.explanation,sizeof(c.explanation),"admissible static candidate; cost unknown (missing/stale target/build calibration); never ranked as zero cost");}
        candidates.push_back(c);
      }
    }
    for(auto& c:candidates)if(c.status==RNS8_SUCCESS && c.cost_known){c.pareto=1;for(const auto& other:candidates)
      if(other.status==RNS8_SUCCESS && other.cost_known && other.peak_payload_bytes<=c.peak_payload_bytes && other.upper_amortized_ns<=c.upper_amortized_ns &&
         (other.peak_payload_bytes<c.peak_payload_bytes || other.upper_amortized_ns<c.upper_amortized_ns)){c.pareto=0;break;}}
    *required=uint32_t(candidates.size());for(uint32_t i=0;i<std::min(capacity,*required);++i)results[i]=candidates[i];return RNS8_SUCCESS;
  });
}
