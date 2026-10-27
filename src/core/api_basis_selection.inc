namespace {
bool basis_cost_digest(const char* value) {
  return value[64]==0 && std::all_of(value,value+64,[](char c){return (c>='0' && c<='9') || (c>='a' && c<='f');});
}
rns8_status select_basis_plan(const rns8_context* ctx,const rns8_basis_selection_desc* request,
    const rns8_basis_candidate* candidates,uint32_t count,rns8_basis_selection* output) {
  if(!ctx || !request || !candidates || !count || count>4096 || !output ||
      !valid_abi(request->struct_size,request->abi_version,sizeof(*request)) ||
      !valid_abi(output->struct_size,output->abi_version,sizeof(*output)) ||
      !request->reuse_count || request->flags&~3u || !exact_wide_semantics(request->semantics) ||
      request->m<=0 || request->n<=0 || request->k<=0 || !basis_cost_digest(request->cohort_identity))return RNS8_INVALID_ARGUMENT;
  if(ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
  char build[65]{};auto status=rns8_get_pipeline_build_identity(build,sizeof(build));if(status!=RNS8_SUCCESS)return status;
  cpp_int magnitude=request->magnitude_bound[2];magnitude<<=64;magnitude+=request->magnitude_bound[1];magnitude<<=64;magnitude+=request->magnitude_bound[0];
  rns8_basis_selection best{};best.struct_size=sizeof(best);best.abi_version=RNS8_ABI_VERSION;
  bool found=false,budget_rejected=false;
  for(uint32_t i=0;i<count;++i){const auto& candidate=candidates[i];const auto& price=candidate.cost;const auto& d=candidate.plan;
    if(!valid_abi(d.struct_size,d.abi_version,sizeof(d)) || !valid_abi(price.struct_size,price.abi_version,sizeof(price)) ||
        (price.kind!=1 && price.kind!=2) || !basis_cost_digest(price.build_identity) || !basis_cost_digest(price.cohort_identity) ||
        !std::memchr(price.target_id,0,sizeof(price.target_id)))return RNS8_INVALID_ARGUMENT;
    const auto basis=basis_data(candidate.basis);
    if(d.semantics!=request->semantics || d.m!=request->m || d.n!=request->n || d.k!=request->k ||
        std::strcmp(price.cohort_identity,request->cohort_identity)!=0)continue;
    if(!price.count || price.count>20 || price.count!=basis->packet.count || price.prefix!=d.prefix ||
        !std::equal(price.ordered_moduli,price.ordered_moduli+20,basis->packet.moduli) ||
        price.backend!=d.backend || price.semantics!=d.semantics || price.m!=d.m || price.n!=d.n || price.k!=d.k ||
        std::strcmp(price.build_identity,build)!=0 || std::strcmp(price.target_id,d.backend==RNS8_BACKEND_CPU_REFERENCE?"cpu":ctx->device_info.gcn_arch)!=0)continue;
    if(!basis_context_matches(*ctx,d.backend) || !basis_backend_compiled(d.backend))continue;
    BasisStorage storage;status=basis_plan_storage(*basis,d,storage);
    if(status==RNS8_WORKSPACE_TOO_SMALL){budget_rejected=true;continue;}
    if(status!=RNS8_SUCCESS || !exact_range_fits(d.semantics,magnitude,basis->products[d.prefix]) ||
        price.device_bytes!=storage.device || price.host_bytes!=storage.host)continue;
    if(!price.pack_ns || !price.gemm_ns || !price.export_ns || price.preparation_ns==UINT64_MAX ||
        price.pack_ns==UINT64_MAX || price.gemm_ns==UINT64_MAX || price.export_ns==UINT64_MAX)continue;
    uint64_t projected=price.preparation_ns;
    const auto charge=[&](uint64_t value,uint64_t repeats){if(value>UINT64_MAX/repeats)return false;
      const auto total=value*repeats;if(total>UINT64_MAX-projected)return false;projected+=total;return true;};
    if(!charge(price.pack_ns,request->flags&1?request->reuse_count:1) || !charge(price.gemm_ns,request->reuse_count) ||
        !charge(price.export_ns,request->flags&2?request->reuse_count:1))continue;
    ++best.eligible_candidates;
    // Stable caller order resolves exact cost ties; it never reorders planes.
    if(!found || projected<best.projected_ns){found=true;best.selected_index=i;best.projected_ns=projected;best.price_kind=price.kind;
      best.resources.struct_size=sizeof(best.resources);best.resources.abi_version=RNS8_ABI_VERSION;
      best.resources.required_device_bytes=storage.device;best.resources.required_host_bytes=storage.host;
      best.resources.prefix=d.prefix;best.resources.representation=1;best.resources.ordered_basis_identity=basis->packet.identity;}
  }
  if(!found)return budget_rejected?RNS8_WORKSPACE_TOO_SMALL:RNS8_UNSUPPORTED_BACKEND;
  *output=best;return RNS8_SUCCESS;
}
}
rns8_status rns8_select_basis_plan(const rns8_context* ctx,const rns8_basis_selection_desc* request,
    const rns8_basis_candidate* candidates,uint32_t count,rns8_basis_selection* output) {
  return guard_api([&](){return select_basis_plan(ctx,request,candidates,count,output);});
}
rns8_status rns8_create_selected_basis_plan(rns8_context* ctx,const rns8_basis_selection_desc* request,
    const rns8_basis_candidate* candidates,uint32_t count,rns8_basis_selection* output,rns8_basis_plan** out) {
  return guard_api([&]() -> rns8_status {
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!output || !valid_abi(output->struct_size,output->abi_version,sizeof(*output)))return RNS8_INVALID_ARGUMENT;
    rns8_basis_selection choice{};choice.struct_size=sizeof(choice);choice.abi_version=RNS8_ABI_VERSION;
    const auto selected=select_basis_plan(ctx,request,candidates,count,&choice);if(selected!=RNS8_SUCCESS)return selected;
    const auto& candidate=candidates[choice.selected_index];
    const auto created=rns8_create_basis_plan(ctx,candidate.basis,&candidate.plan,out);
    if(created==RNS8_SUCCESS)*output=choice;return created;
  });
}
