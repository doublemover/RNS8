namespace {
struct ExactHipAutoLayout {
  uint64_t a_offset=0,b_offset=0,c_offset=0,lift_offset=0,total=0,lift_bytes=0;
  uint64_t a_cells=0,b_cells=0,c_cells=0;bool lift_a=false,lift_b=false;
};
bool append_exact_auto_region(uint64_t bytes,uint64_t& total,uint64_t& offset) {
  if(total>UINT64_MAX-3)return false;const auto aligned=(total+3)&~UINT64_C(3);
  if(bytes>UINT64_MAX-aligned)return false;offset=aligned;total=aligned+bytes;return true;
}
}
rns8_status rns8_gemm_exact_wide_hip_auto(rns8_context* ctx,const rns8_plan* plan,rns8_matrix* A,rns8_matrix* B,
    rns8_matrix* C,rns8_workspace* workspace,uint32_t algorithm,uint64_t budget,uint32_t* selected_out) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!ctx || !plan || !A || !B || !C || !workspace || !selected_out || A==C || B==C || algorithm>1)return RNS8_INVALID_ARGUMENT;
    if(ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
    if(ctx->backend!=RNS8_BACKEND_HIP_DIRECT || plan->backend!=ctx->backend)return RNS8_UNSUPPORTED_BACKEND;
    if(!exact_wide_semantics(plan->desc.semantics))return RNS8_INVALID_ARGUMENT;
    auto status=validate_plan_context_workspace(*ctx,*plan,*workspace);if(status!=RNS8_SUCCESS)return status;
    if(!plan->tile_schedule.empty() || plan->desc.flags&~uint32_t(RNS8_PLAN_FORCE_FIXED_PREFIX))return RNS8_UNSUPPORTED_BACKEND;
    for(const auto& item:{std::pair<rns8_matrix*,std::pair<int64_t,int64_t>>{A,{plan->desc.m,plan->desc.k}},
        {B,{plan->desc.k,plan->desc.n}},{C,{plan->desc.m,plan->desc.n}}}) {
      auto& matrix=*item.first;
      if(matrix.in_flight_writer || matrix.in_flight_readers)return RNS8_BUSY;
      if(matrix.generation_exhausted || (item.first==C && matrix.content_generation==UINT64_MAX))return RNS8_RANGE_ERROR;
      if(!matrix_descriptor_matches(matrix,plan->desc.semantics,RNS8_BOUND_NONE,item.second.first,item.second.second,plan->prefix,plan->desc.tile_m,plan->desc.tile_n) ||
          matrix.backend!=ctx->backend || matrix.hip_device_id!=ctx->device_id || matrix.materialized_rns_schedule ||
          !rns_matrix_storage_matches(matrix,ctx->backend,item.second.first,item.second.second,matrix.prefix))return RNS8_INVALID_ARGUMENT;
    }
    for(const auto* input:{A,B})if(!input->device_residues_current || !input->exact_range_prefix || input->exact_range_prefix>input->prefix ||
        !exact_range_fits(input->desc.semantics,input->exact_max_magnitude,modulus_product(input->exact_range_prefix)))return RNS8_RANGE_ERROR;
    boost::multiprecision::cpp_int output_bound;rns8_exact_matrix_ranges output_ranges;
    exact_gemm_bound(*plan,*A,*B,output_bound,&output_ranges);
    const auto ceiling=std::min({plan->desc.max_prefix,A->prefix,B->prefix,C->prefix});uint32_t selected=plan->prefix;
    while(selected<=ceiling && !exact_range_fits(plan->desc.semantics,output_bound,modulus_product(selected)))++selected;
    if(selected>ceiling)return RNS8_RANGE_ERROR;
    const auto support=query_residue_gemm_support(plan->desc.m,plan->desc.n,plan->desc.k,A->desc.cols,B->desc.cols,C->desc.cols,selected);
    if(support.status!=RNS8_SUCCESS)return support.status;
    rns8_operation_query query{};query.struct_size=sizeof(query);query.abi_version=RNS8_ABI_VERSION;query.operation=RNS8_OP_GEMM;
    query.execution_modes=RNS8_EXEC_SYNC;query.problem=plan->desc;query.problem.max_prefix=selected;query.problem.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
    query.lda=A->desc.cols;query.ldb=B->desc.cols;query.ldc=C->desc.cols;query.scratch_budget=UINT64_MAX;
    status=operation_support(*ctx,query,RNS8_BACKEND_HIP_DIRECT).status;if(status!=RNS8_SUCCESS)return status;
    ExactHipAutoLayout layout;layout.a_cells=uint64_t(A->desc.rows)*A->desc.cols;layout.b_cells=uint64_t(B->desc.rows)*B->desc.cols;layout.c_cells=uint64_t(C->desc.rows)*C->desc.cols;
    layout.lift_a=A->exact_range_prefix<selected;layout.lift_b=A!=B && B->exact_range_prefix<selected;
    uint64_t a_bound[3]{},b_bound[3]{};
    auto bound_words=[](boost::multiprecision::cpp_int bound,uint64_t* words){for(unsigned i=0;i<3;++i){words[i]=static_cast<uint64_t>(bound&UINT64_MAX);bound>>=64;}return bound==0;};
    if(!bound_words(A->exact_max_magnitude,a_bound) || !bound_words(B->exact_max_magnitude,b_bound))return RNS8_RANGE_ERROR;
    const auto prepare_lift=[&](rns8_matrix& matrix,uint64_t cells,bool needed,uint64_t& offset)->rns8_status {
      if(!needed)return RNS8_SUCCESS;const auto lift=query_exact_lift_support(int64_t(cells),matrix.exact_range_prefix,selected,UINT64_MAX);
      if(lift.status!=RNS8_SUCCESS)return lift.status;layout.lift_bytes=std::max(layout.lift_bytes,lift.scratch_bytes);
      return append_exact_auto_region(cells*selected,layout.total,offset)?RNS8_SUCCESS:RNS8_RANGE_ERROR;
    };
    status=prepare_lift(*A,layout.a_cells,layout.lift_a,layout.a_offset);if(status!=RNS8_SUCCESS)return status;
    status=prepare_lift(*B,layout.b_cells,layout.lift_b,layout.b_offset);if(status!=RNS8_SUCCESS)return status;
    if(!append_exact_auto_region(layout.c_cells*selected,layout.total,layout.c_offset) ||
        !append_exact_auto_region(layout.lift_bytes,layout.total,layout.lift_offset) || layout.total>SIZE_MAX)return RNS8_RANGE_ERROR;
    if(layout.total>budget)return RNS8_WORKSPACE_TOO_SMALL;
    auto& owner=workspace->exact_auto_scratch_owner;
    if(owner.bytes()<layout.total || owner.bytes()>budget){status=owner.allocate(ctx->device_id,size_t(layout.total));if(status!=RNS8_SUCCESS)return status;}
    size_t lease_token=0;status=begin_synchronous_lease(*ctx,*A,*B,*C,*workspace,lease_token,true);if(status!=RNS8_SUCCESS)return status;
    bool c_written=false;
    struct Guard{rns8_context& ctx;size_t token;bool& written;bool active=true;~Guard(){if(active)finish_synchronous_lease(ctx,token,RNS8_INTERNAL_ERROR,written);}} guard{*ctx,lease_token,c_written};
    auto* arena=static_cast<int8_t*>(owner.get());auto* a=layout.lift_a?arena+layout.a_offset:static_cast<int8_t*>(A->hip_residues);
    auto* b=A==B?a:layout.lift_b?arena+layout.b_offset:static_cast<int8_t*>(B->hip_residues);auto* c=arena+layout.c_offset;
    const auto stage=[&](rns8_matrix& matrix,int8_t* target,uint64_t cells,bool needed,const uint64_t* words)->rns8_status {
      if(!needed)return RNS8_SUCCESS;auto s=hip_direct_copy_device_to_device(ctx->device_id,target,matrix.hip_residues,size_t(cells*matrix.exact_range_prefix));if(s!=RNS8_SUCCESS)return s;
      bool private_commit=false;return hip_direct_lift_exact_suffix_workspace(ctx->device_id,target,int64_t(cells),matrix.exact_range_prefix,selected,
          matrix.desc.semantics==RNS8_EXACT_WIDE_SIGNED,words,arena+layout.lift_offset,layout.lift_bytes,private_commit,algorithm==1);
    };
    status=stage(*A,a,layout.a_cells,layout.lift_a,a_bound);
    if(status==RNS8_SUCCESS && A!=B)status=stage(*B,b,layout.b_cells,layout.lift_b,b_bound);
    if(status==RNS8_SUCCESS)status=hip_direct_gemm_rns_device(ctx->device_id,a,b,c,plan->desc.m,plan->desc.n,plan->desc.k,A->desc.cols,B->desc.cols,C->desc.cols,selected);
    // Publish only suffix bytes; the original unique source planes are never
    // overwritten, so any failed suffix commit remains an unobservable region.
    const auto copy_suffix=[&](rns8_matrix& matrix,const int8_t* source,uint64_t cells,bool needed){if(!needed)return RNS8_SUCCESS;
      const uint64_t begin=cells*matrix.exact_range_prefix,bytes=cells*(selected-matrix.exact_range_prefix);
      return hip_direct_copy_device_to_device(ctx->device_id,static_cast<int8_t*>(matrix.hip_residues)+begin,source+begin,size_t(bytes));};
    if(status==RNS8_SUCCESS)status=copy_suffix(*A,a,layout.a_cells,layout.lift_a);
    if(status==RNS8_SUCCESS && A!=B)status=copy_suffix(*B,b,layout.b_cells,layout.lift_b);
    if(status==RNS8_SUCCESS){clear_residue_current(*C);clear_native_current(*C);clear_byte_limb_current(*C);c_written=true;
      status=hip_direct_copy_device_to_device(ctx->device_id,C->hip_residues,c,size_t(layout.c_cells*selected));}
    const bool drained=finish_synchronous_lease(*ctx,lease_token,status,c_written);guard.active=false;
    if(status!=RNS8_SUCCESS)return status;if(!drained)return RNS8_BACKEND_FAILURE;
    // No allocation or throwing arithmetic follows the all-copy success point.
    if(layout.lift_a){A->exact_range_prefix=selected;A->materialized_rns_prefix=std::max(A->materialized_rns_prefix,selected);A->host_residues_current=false;}
    if(layout.lift_b){B->exact_range_prefix=selected;B->materialized_rns_prefix=std::max(B->materialized_rns_prefix,selected);B->host_residues_current=false;}
    mark_output_device_residues_current(*C);C->exact_max_magnitude.swap(output_bound);C->exact_axis_ranges.rows.swap(output_ranges.rows);C->exact_axis_ranges.cols.swap(output_ranges.cols);
    C->exact_range_prefix=selected;C->materialized_rns_prefix=selected;*selected_out=selected;return RNS8_SUCCESS;
  });
}
