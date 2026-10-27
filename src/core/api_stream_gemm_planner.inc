namespace {
bool stream_profile_text(const char* text,size_t bytes){return std::memchr(text,0,bytes)!=nullptr;}
bool stream_cost_pair(double low,double high,double minimum=0) {
  return std::isfinite(low) && std::isfinite(high) && ((low==0 && high==0) || (low>0 && low>=minimum && high>=low));
}
rns8_status validate_stream_cost_profile(const rns8_stream_gemm_cost_profile* p) {
  if(!p)return RNS8_SUCCESS;
  const auto& rates=p->stages;
  if(!valid_abi(p->struct_size,p->abi_version,sizeof(*p)) || !p->reuse_count ||
      !p->max_modeled_panels || p->max_modeled_panels>65536 ||
      (p->semantics!=RNS8_BOUNDED_I64 && p->semantics!=RNS8_BOUNDED_U64) ||
      !p->prefix || p->prefix>20 || !p->schedule_mask || p->schedule_mask>3 ||
      !valid_abi(rates.struct_size,rates.abi_version,sizeof(rates)) || rates.known_mask>63 ||
      !stream_profile_text(rates.target_id,sizeof(rates.target_id)) ||
      !stream_profile_text(rates.build_identity_sha256,sizeof(rates.build_identity_sha256)) ||
      !stream_profile_text(rates.calibration_identity,sizeof(rates.calibration_identity)) ||
      !stream_cost_pair(p->lower_host_copy_ns_per_byte,p->upper_host_copy_ns_per_byte) ||
      !stream_cost_pair(p->lower_input_scan_ns_per_cell,p->upper_input_scan_ns_per_cell) ||
      !stream_cost_pair(p->lower_boundary_ns,p->upper_boundary_ns) ||
      !stream_cost_pair(p->lower_duplex_multiplier,p->upper_duplex_multiplier,1) ||
      !stream_cost_pair(rates.lower_launch_ns,rates.upper_launch_ns))return RNS8_INVALID_ARGUMENT;
  for(unsigned i=0;i<6;++i)if(rates.known_mask&(1u<<i))
    if(!stream_cost_pair(rates.lower_ns_per_unit[i],rates.upper_ns_per_unit[i]) || rates.lower_ns_per_unit[i]<=0)
      return RNS8_INVALID_ARGUMENT;
  return RNS8_SUCCESS;
}
}
rns8_status rns8_plan_stream_gemm(const rns8_context* ctx,const rns8_stream_gemm_desc* base,
    const rns8_stream_gemm_panel_choice* choices,uint32_t count,const rns8_stream_gemm_cost_profile* profile,
    rns8_stream_gemm_candidate* output,uint32_t capacity,uint32_t* required) {
  return guard_api([&]()->rns8_status {
    if(!ctx || !base || !choices || !count || count>64 || !required || (capacity && !output) ||
        !valid_abi(base->struct_size,base->abi_version,sizeof(*base)))return RNS8_INVALID_ARGUMENT;
    if(ctx->backend!=RNS8_BACKEND_HIP_DIRECT)return RNS8_UNSUPPORTED_BACKEND;
    const auto valid=validate_stream_cost_profile(profile);if(valid!=RNS8_SUCCESS)return valid;
    *required=count;if(capacity<count)return RNS8_WORKSPACE_TOO_SMALL;
    char build[65]{};const auto identity=rns8_get_pipeline_build_identity(build,sizeof(build));if(identity!=RNS8_SUCCESS)return identity;
    std::vector<rns8_stream_gemm_candidate> results;results.reserve(count);
    for(uint32_t i=0;i<count;++i) {
      rns8_stream_gemm_candidate item{};item.struct_size=sizeof(item);item.abi_version=RNS8_ABI_VERSION;item.choice_index=i;
      item.executable_problem=*base;auto& d=item.executable_problem;
      d.panel_m=choices[i].panel_m;d.panel_n=choices[i].panel_n;d.panel_k=choices[i].panel_k;
      d.schedule=choices[i].schedule;d.retained_b_budget_bytes=choices[i].retained_b_budget_bytes;
      item.resources.struct_size=sizeof(item.resources);item.resources.abi_version=RNS8_ABI_VERSION;
      item.status=rns8_query_stream_gemm(&d,&item.resources);
      const char* reason=item.status==RNS8_SUCCESS?"admitted; calibration costs unknown":"runtime range/resource/provider admission rejected";
      if(item.status==RNS8_SUCCESS && profile) {
        const auto& rates=profile->stages;
        const bool matched=rates.backend==RNS8_BACKEND_HIP_DIRECT && std::strcmp(rates.target_id,ctx->device_info.gcn_arch)==0 &&
            std::strcmp(rates.build_identity_sha256,build)==0 && rates.calibration_identity[0] &&
            profile->semantics==d.semantics && profile->prefix==d.prefix &&
            (profile->schedule_mask&(1u<<uint32_t(d.schedule)));
        const bool known=(rates.known_mask&47)==47 && rates.lower_launch_ns>0 && profile->lower_host_copy_ns_per_byte>0 &&
            profile->lower_input_scan_ns_per_cell>0 && profile->lower_boundary_ns>0 &&
            (d.schedule!=RNS8_STREAM_GEMM_OVERLAPPED || profile->lower_duplex_multiplier>=1);
        if(!matched)reason="admitted; calibration target/build identity missing or stale";
        else if(!known)reason="admitted; required stage/scan/host/boundary/duplex rates unknown";
        else {
          StreamPanelProblem p;StreamPanelResources r;uint64_t bound=0;
          const auto admitted=stream_description(&d,p,r,bound);if(admitted!=RNS8_SUCCESS)return RNS8_INTERNAL_ERROR;
          if(r.total_steps>profile->max_modeled_panels)reason="admitted; prediction panel cap exceeded, costs unknown";
          else {
            const auto low=estimate_stream_panels(p,r,*profile,false),high=estimate_stream_panels(p,r,*profile,true);
            if(std::isfinite(low) && std::isfinite(high) && low>0 && high>=low) {
              item.cost_known=1;item.lower_amortized_ns=low;item.upper_amortized_ns=high;
              reason="admitted; endpoint list-schedule estimate including scan, eviction, fill/drain and duplex penalty";
            } else reason="admitted; prediction not finite or ordered, costs unknown";
          }
        }
      }
      std::strncpy(item.explanation,reason,sizeof(item.explanation)-1);results.push_back(item);
    }
    std::stable_sort(results.begin(),results.end(),[](const auto& a,const auto& b) {
      if((a.status==RNS8_SUCCESS)!=(b.status==RNS8_SUCCESS))return a.status==RNS8_SUCCESS;
      if(a.cost_known!=b.cost_known)return a.cost_known>b.cost_known;
      if(a.cost_known && a.upper_amortized_ns!=b.upper_amortized_ns)return a.upper_amortized_ns<b.upper_amortized_ns;
      if(a.resources.device_bytes!=b.resources.device_bytes)return a.resources.device_bytes<b.resources.device_bytes;
      if(a.resources.host_bytes!=b.resources.host_bytes)return a.resources.host_bytes<b.resources.host_bytes;
      return a.choice_index<b.choice_index;
    });
    std::copy(results.begin(),results.end(),output);return RNS8_SUCCESS;
  });
}
