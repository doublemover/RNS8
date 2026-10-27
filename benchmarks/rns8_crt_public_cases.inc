// Public owner controls share the conversion executable/capture schema. Every
// timed call crosses the real public proof/export/publication boundary.
struct PublicCrtOwner {
  rns8_context* ctx=nullptr;rns8_plan* plan=nullptr;rns8_matrix* matrix=nullptr;rns8_quarter_crt_export* quarter=nullptr;rns8_crt_export* checked=nullptr;
  rns8_matrix* a=nullptr;rns8_matrix* b=nullptr;rns8_workspace* workspace=nullptr;
  uint64_t completed_products=0,initial_source_generation=0,workspace_bytes=0;int64_t output_ld=0;
  PublicCrtOwner()=default;PublicCrtOwner(const PublicCrtOwner&)=delete;PublicCrtOwner& operator=(const PublicCrtOwner&)=delete;
  ~PublicCrtOwner(){if(ctx)(void)rns8::detail::hip_direct_synchronize(ctx->device_id);
    if(checked)(void)rns8_destroy_crt_export(checked);if(quarter)(void)rns8_destroy_quarter_crt_export(quarter);if(matrix)(void)rns8_destroy_matrix(matrix);
    if(a)(void)rns8_destroy_matrix(a);if(b)(void)rns8_destroy_matrix(b);if(workspace)(void)rns8_destroy_workspace(workspace);if(plan)(void)rns8_destroy_plan(plan);if(ctx)(void)rns8_destroy_context(ctx);}
  void close(){if(checked){check(rns8_destroy_crt_export(checked));checked=nullptr;}if(quarter){check(rns8_destroy_quarter_crt_export(quarter));quarter=nullptr;}if(matrix){check(rns8_destroy_matrix(matrix));matrix=nullptr;}
    if(a){check(rns8_destroy_matrix(a));a=nullptr;}if(b){check(rns8_destroy_matrix(b));b=nullptr;}if(workspace){check(rns8_destroy_workspace(workspace));workspace=nullptr;}
    if(plan){check(rns8_destroy_plan(plan));plan=nullptr;}if(ctx){check(rns8_destroy_context(ctx));ctx=nullptr;}}
  void prepare_export(bool sign,bool quotient,int algorithm) {
    const auto cells=uint64_t(plan->desc.m)*uint64_t(plan->desc.n);const auto prefix=plan->prefix;
    if(quotient){rns8_quarter_crt_desc q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;q.mode=sign?0:1;q.limb_count=1;q.max_scratch_bytes=cells*16;
      check(rns8_prepare_quarter_crt_export(ctx,plan,matrix,&q,&quarter));}
    if(algorithm>=0){rns8_crt_export_desc q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;q.algorithm=uint32_t(algorithm);q.mode=sign?0:1;q.limb_count=1;
      q.max_scratch_bytes=cells*(40+(algorithm==4?8*((prefix+7)/8):0))+4;
      check(rns8_prepare_crt_export(ctx,plan,matrix,&q,&checked));}
  }
  void initialize(const std::vector<uint64_t>& native,uint32_t prefix,bool sign,bool quotient,int algorithm=-1) {
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    check(rns8_create_context(0,&options,&ctx));rns8_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
    d.semantics=sign?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;d.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    // This owner exports an already certified value; it never computes AB.
    // A short prefix need not cover the product of two full native type envelopes.
    d.m=1;d.n=native.size();d.k=1;d.max_prefix=prefix;d.flags=RNS8_PLAN_FORCE_FIXED_PREFIX|RNS8_PLAN_EXACT_VALUE_BOUND;check(rns8_create_plan(ctx,&d,&plan));
    rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.semantics=d.semantics;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.logical_ld=native.size();
    md.rows=1;md.cols=native.size();md.max_prefix=prefix;check(rns8_create_matrix(ctx,&md,&matrix));
    if(sign)check(rns8_pack_i64(ctx,matrix,reinterpret_cast<const int64_t*>(native.data()),native.size(),1));
    else check(rns8_pack_u64(ctx,matrix,native.data(),native.size(),1));
    output_ld=plan->desc.n+8;prepare_export(sign,quotient,algorithm);
  }
  void product(){check(rns8_gemm_rns(ctx,plan,a,b,matrix,workspace));++completed_products;}
  void initialize_gemm(const std::vector<uint64_t>& av,const std::vector<uint64_t>& bv,uint32_t side,uint32_t k,uint32_t prefix,
                       bool sign,bool quotient,int algorithm,uint64_t available_workspace) {
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    check(rns8_create_context(0,&options,&ctx));rns8_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
    d.semantics=sign?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;d.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    d.m=side;d.n=side;d.k=k;d.max_prefix=prefix;d.flags=RNS8_PLAN_FORCE_FIXED_PREFIX|RNS8_PLAN_EXACT_VALUE_BOUND|RNS8_PLAN_EXACT_SCALAR_FIRST;
    check(rns8_create_plan(ctx,&d,&plan));if(plan->prefix!=prefix)throw std::runtime_error("GEMM prefix differs from requested fixed prefix");
    rns8_plan_backend_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;check(rns8_get_plan_backend_info(plan,&info));
    workspace_bytes=info.workspace_required_bytes;
    // The direct owner has at most five 256-byte-aligned arena regions. Admit
    // worst-case alignment gaps before allocation, then report retained bytes.
    if(workspace_bytes>available_workspace || (workspace_bytes && available_workspace-workspace_bytes<4*255))
      throw std::invalid_argument("GEMM aggregate workspace budget");
    check(rns8_create_workspace(ctx,plan,&workspace));workspace_bytes=workspace->hip_workspace_arena_bytes;
    if(workspace_bytes>available_workspace)throw std::runtime_error("GEMM workspace exceeded admitted arena envelope");
    const auto create=[&](uint32_t rows,uint32_t cols,rns8_matrix** output){rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;
      md.semantics=d.semantics;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.rows=rows;md.cols=cols;md.logical_ld=cols;md.max_prefix=prefix;check(rns8_create_matrix(ctx,&md,output));};
    create(side,k,&a);create(k,side,&b);create(side,side,&matrix);
    if(sign){check(rns8_pack_i64(ctx,a,reinterpret_cast<const int64_t*>(av.data()),k,1));check(rns8_pack_i64(ctx,b,reinterpret_cast<const int64_t*>(bv.data()),side,1));}
    else{check(rns8_pack_u64(ctx,a,av.data(),k,1));check(rns8_pack_u64(ctx,b,bv.data(),side,1));}
    output_ld=side;product();initial_source_generation=matrix->content_generation;prepare_export(sign,quotient,algorithm);
  }
  void execute(std::vector<uint64_t>& output,bool sign,bool refresh_product=false) {
    if(refresh_product){product();if(checked)check(rns8_rebind_crt_export(checked,matrix));
      else if(quarter){check(rns8_destroy_quarter_crt_export(quarter));quarter=nullptr;prepare_export(sign,true,-1);}}
    if(checked)check(rns8_export_crt_limbs(checked,output.data(),output_ld));
    else if(quarter)check(rns8_export_quarter_crt_limbs(quarter,output.data(),output_ld));
    else if(sign)check(rns8_export_exact_wide_signed_limbs(ctx,plan,matrix,output.data(),output_ld,1));
    else check(rns8_export_exact_wide_unsigned_limbs(ctx,plan,matrix,output.data(),output_ld,1));
  }
  nlohmann::json checked_info() const {
    rns8_crt_export_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;check(rns8_get_crt_export_info(checked,&info));
    nlohmann::json groups=nlohmann::json::array();for(uint32_t g=0;g<info.group_count;++g)
      groups.push_back({{"id",g},{"channel_mask",info.group_channel_masks[g]},{"canonical_modulus",std::to_string(info.group_products[g])}});
    return {{"algorithm",info.algorithm},{"mode",info.mode},{"limb_count",info.limb_count},{"prefix",info.prefix},{"groups",groups},
      {"producer_order",info.producer_order},{"source_instance",info.source_instance},{"source_generation",info.source_generation},
      {"successful_exports",info.successful_exports},{"retained_host_bytes",info.retained_host_bytes},{"retained_device_bytes",info.retained_device_bytes},
      {"group_scratch_bytes",info.group_scratch_bytes},{"certificate_sha256",info.certificate_sha256}};
  }
  nlohmann::json gemm_info(uint64_t envelope) const {
    return {{"workspace_payload_bytes",workspace_bytes},{"aggregate_payload_admission_bytes",envelope+workspace_bytes},
      {"gemm_completed_products",completed_products},{"gemm_initial_source_generation",initial_source_generation},
      {"gemm_final_source_generation",matrix->content_generation}};
  }
  uint64_t retained_device_payload() const {
    uint64_t bytes=0;for(const auto* source:{matrix,a,b})if(source)
      bytes+=source->hip_residue_bytes+source->hip_export_bytes+source->hip_status_bytes+source->hip_upload_bytes;
    return bytes;
  }
};
int public_crt_algorithm(const std::string& candidate) {
  const char* policies[]={"garner","direct","balanced","group_local","group_split"};
  for(int i=0;i<5;++i)for(const char* sign:{"signed","unsigned"})
    if(candidate==std::string("public_owned_")+policies[i]+"_"+sign)return i;
  return -1;
}
unsigned crt_popcount64(uint64_t bits) {
  bits-=((bits>>1)&UINT64_C(0x5555555555555555));
  bits=(bits&UINT64_C(0x3333333333333333))+((bits>>2)&UINT64_C(0x3333333333333333));
  return unsigned((((bits+(bits>>4))&UINT64_C(0x0f0f0f0f0f0f0f0f))*UINT64_C(0x0101010101010101))>>56);
}
void public_crt_gemm_dataset(uint32_t side,uint32_t k,uint32_t prefix,bool sign,
                             std::vector<uint64_t>& a,std::vector<uint64_t>& b,
                             std::vector<int8_t>& residues,std::vector<uint64_t>& expected) {
  const auto count=size_t(side)*k;const uint32_t words=(k+63)/64;uint64_t state=3903;
  for(uint32_t p=0;p<prefix;++p)if(kDefaultModuli[p]<3)throw std::runtime_error("binary GEMM residue recipe requires moduli above two");
  a.resize(count);b.resize(count);std::vector<uint64_t> row_bits(size_t(side)*words),col_bits(size_t(side)*words);
  for(unsigned role=0;role<2;++role)for(size_t index=0;index<count;++index) {
    bool bit=(random_word(state)&1)!=0;const auto row=index/(role?side:k),col=index%(role?side:k);
    if((!role && row==0) || (role && col==0))bit=true;if(!role && row==1)bit=false;
    const int value=bit?1:sign?-1:0;(role?b:a)[index]=static_cast<uint64_t>(value);
    if(bit){if(!role)row_bits[row*words+col/64]|=uint64_t{1}<<(col%64);
      else col_bits[col*words+row/64]|=uint64_t{1}<<(row%64);}
    // The three possible units are already centered for every active modulus.
    // Avoid allocating multiprecision temporaries for each identical plane.
    for(uint32_t p=0;p<prefix;++p)residues[(size_t(role)*prefix+p)*count+index]=static_cast<int8_t>(value);
  }
  // Each operand is binary. XOR counts signed disagreements; AND counts
  // unsigned unit products. All output cells are exact within the proven K bound.
  for(uint32_t row=0;row<side;++row)for(uint32_t col=0;col<side;++col) {
    unsigned population=0;for(uint32_t word=0;word<words;++word) {
      const auto x=row_bits[size_t(row)*words+word],y=col_bits[size_t(col)*words+word];population+=crt_popcount64(sign?x^y:x&y);}
    const int64_t value=sign?int64_t(k)-2*int64_t(population):int64_t(population);
    expected[size_t(row)*side+col]=static_cast<uint64_t>(value);
  }
}
int public_crt_case(const std::string& candidate,uint32_t cells,uint32_t prefix,uint32_t repeats,uint32_t warmups,
                    const std::string& run_id,const std::string& scope,uint32_t k=0) {
  const int algorithm=public_crt_algorithm(candidate);
  const bool sign=candidate.size()>=7 && candidate.compare(candidate.size()-7,7,"_signed")==0;
  const bool quotient=candidate.rfind("public_quarter_",0)==0;
  const bool gemm=scope=="gemm_resident" || scope=="gemm_cold",cold=scope=="cold" || scope=="gemm_cold";
  if((algorithm<0 && candidate!="public_quarter_signed" && candidate!="public_quarter_unsigned" && candidate!="public_garner_signed" && candidate!="public_garner_unsigned") ||
     (!gemm && scope!="resident" && scope!="cold") || (gemm!=(k!=0)) || k>8192 || run_id.empty() || run_id.size()>128 || run_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")!=std::string::npos)
    throw std::invalid_argument("invalid public CRT case");
  uint32_t side=1;if(gemm){while(uint64_t(side+1)*(side+1)<=cells)++side;
    if(uint64_t(side)*side!=cells || cpp_int(k)*4>modulus_product(prefix))throw std::invalid_argument("GEMM requires square output and matching quarter-bound K");}
  const uint64_t output_bytes=uint64_t(cells)*8+64,source_bytes=gemm?uint64_t(2)*side*k*prefix:uint64_t(cells)*prefix;
  const uint64_t owner_bytes=algorithm<0?0:uint64_t(cells)*(40+(algorithm==4?8*((prefix+7)/8):0))+4;
  const uint64_t envelope=source_bytes*(gemm?4:3)+output_bytes*5+(gemm?uint64_t(cells)*prefix*2+uint64_t(2)*side*k*24:uint64_t(cells)*8)+owner_bytes+16384,budget=UINT64_C(768)*1024*1024;
  if(envelope>budget)throw std::invalid_argument("public CRT payload budget");
  const auto oracle_start=Clock::now();const cpp_int product=modulus_product(prefix);
  const cpp_int bound=std::min(cpp_int(product/4),(cpp_int(1)<<(sign?63:64))-1);
  std::vector<uint64_t> native(gemm?0:cells),av,bv,expected(cells+8,UINT64_C(0x6363636363636363)),observed=expected;
  std::vector<int8_t> residues(source_bytes),audit(source_bytes);uint64_t state=3903;cpp_int actual_bound=0;
  if(gemm){public_crt_gemm_dataset(side,k,prefix,sign,av,bv,residues,expected);actual_bound=k;}
  else for(uint32_t cell=0;cell<cells;++cell){const auto r0=random_word(state),r1=random_word(state),r2=random_word(state);
    cpp_int value=(cpp_int(r0)+(cpp_int(r1)<<64)+(cpp_int(r2)<<128))%(bound+1);
    if(cell==0)value=0;if(cell==1)value=bound;if(cell==2)value=bound-1;if(sign && cell%2)value=-value;
    actual_bound=std::max(actual_bound,cpp_int(value<0?-value:value));
    for(uint32_t p=0;p<prefix;++p)residues[size_t(p)*cells+cell]=centered_residue(value,kDefaultModuli[p]);
    if(value<0)value+=cpp_int(1)<<64;native[cell]=expected[cell]=static_cast<uint64_t>(value);
  }
  const auto input_hash=checksum(residues.data(),residues.size()),output_hash=checksum(expected.data(),output_bytes);
  const double oracle_ns=std::chrono::duration<double,std::nano>(Clock::now()-oracle_start).count();
  rns8_device_info device{};check(hip_direct_probe(0,device));hip_direct_timing_set_enabled(false);
  const auto initialize=[&](PublicCrtOwner& owner){if(gemm)owner.initialize_gemm(av,bv,side,k,prefix,sign,quotient,algorithm,budget-envelope);
    else owner.initialize(native,prefix,sign,quotient,algorithm);};
  PublicCrtOwner resident;const auto setup_start=Clock::now();if(!cold)initialize(resident);
  const double setup_ns=std::chrono::duration<double,std::nano>(Clock::now()-setup_start).count();
  const auto source_audit=[&](PublicCrtOwner& owner){if(gemm){check(hip_direct_copy_device_to_host(0,audit.data(),owner.a->hip_residues,source_bytes/2));
      check(hip_direct_copy_device_to_host(0,audit.data()+source_bytes/2,owner.b->hip_residues,source_bytes/2));}
    else check(hip_direct_copy_device_to_host(0,audit.data(),owner.matrix->hip_residues,source_bytes));
    if(audit!=residues)throw std::runtime_error("public CRT changed source");};
  const auto execute=[&](){if(!cold)resident.execute(observed,sign,gemm);else{PublicCrtOwner fresh;initialize(fresh);fresh.execute(observed,sign);fresh.close();}};
  const auto prepare=[&](){std::fill(observed.begin(),observed.end(),UINT64_C(0x6363636363636363));};
  const auto verify=[&](){if(observed!=expected)throw std::runtime_error("public CRT complete output/canary mismatch");if(!cold)source_audit(resident);};
  nlohmann::json checked_info,gemm_info;
  if(cold){PublicCrtOwner audit_owner;initialize(audit_owner);audit_owner.execute(observed,sign);source_audit(audit_owner);
    if(observed!=expected)throw std::runtime_error("public CRT cold audit output mismatch");
    if(audit_owner.checked)checked_info=audit_owner.checked_info();if(gemm)gemm_info=audit_owner.gemm_info(envelope);audit_owner.close();}
  for(uint32_t w=0;w<warmups;++w){prepare();execute();verify();}
  nlohmann::json samples=nlohmann::json::array();
  for(uint32_t sample=0;sample<repeats;++sample){prepare();hip_direct_timing_reset();hip_direct_timing_set_enabled(true);
    auto start=Clock::now();execute();const double instrumented=std::chrono::duration<double,std::nano>(Clock::now()-start).count();hip_direct_timing_flush_pending_events();
    nlohmann::json events=nlohmann::json::array();for(const auto& e:hip_direct_timing_snapshot())events.push_back({{"label",e.label},{"elapsed_ns",e.microseconds>0?nlohmann::json(e.microseconds*1000):nlohmann::json(nullptr)}});
    hip_direct_timing_set_enabled(false);verify();prepare();start=Clock::now();execute();const double production=std::chrono::duration<double,std::nano>(Clock::now()-start).count();verify();
    samples.push_back({{"host_call_ns",instrumented},{"uninstrumented_host_call_ns",production},{"events",events},{"exact_reference_passed",true},{"uninstrumented_exact_reference_passed",true}});}
  prepare();uint32_t calls=0;double elapsed=0;const auto start=Clock::now();do{execute();++calls;elapsed=std::chrono::duration<double,std::nano>(Clock::now()-start).count();}
    while(calls<4096 && (calls<16 || elapsed<250000000));verify();
  nlohmann::json report={{"schema","rns8_crt_microcase_v5"},{"run_id",run_id},{"candidate",candidate},{"scope",scope},{"cells",cells},{"prefix",prefix},
    {"completion_inside_interval",true},{"completion_scope","public_checked_export_readback_and_publication"},
    {"plan_admission_policy","fixed_prefix_current_value_bound_export_only"},{"gemm_measured",false},
    {"seed",3903},{"dataset_recipe","splitmix64_quarter_native64_seed3903_v1"},{"repeats",repeats},{"warmups",warmups},
    {"device_requested",0},{"device_actual",0},{"backend_requested","hip-direct"},{"backend_actual","hip-direct"},{"device",device.name},{"target",device.gcn_arch},
    {"runtime_version",device.hip_runtime_version},{"build_manifest",nlohmann::json::parse(RNS8_BUILD_MANIFEST_JSON)},{"timer","std_chrono_steady_clock_nanoseconds"},
    {"production_timing_hooks_enabled",false},{"public_ownership_measured",true},{"complete_output_and_canaries_checked",true},
    {"input_preservation_scope",cold?"one_separate_resident_audit_before_warmups":"outside_each_verified_call_and_final_burst"},
    {"operation","public_narrow_crt"},{"signed_inputs",sign},{"input_contract","engine_current_quarter_bound_residue_matrix"},
    {"output_contract",sign?"signed_one_limb":"unsigned_one_limb"},{"output_boundary","host"},{"reference_method","boost_cpp_int_all_native64_values"},
    {"input_checksum",input_hash},{"output_checksum",output_hash},{"canary_bytes",64},{"source_bytes_modeled",source_bytes},{"source_physical_bytes",source_bytes},
    {"source_layout","modulus_major_v1"},{"logical_rows",1},{"logical_cols",cells},{"logical_ld",cells},{"pack_kernel",nullptr},{"public_selector_measured",false},
    {"conversion_inside_call",false},{"conversion_scratch_bytes",0},{"output_bytes_modeled",uint64_t(cells)*8},{"output_allocation_bytes",output_bytes},
    {"logical_payload_envelope_bytes",envelope},{"memory_budget_bytes",budget},{"oracle_and_dataset_ns",oracle_ns},
    {"resident_setup_ns",!cold?nlohmann::json(setup_ns):nlohmann::json(nullptr)},
    {"timing_scope",cold?"public_context_plan_pack_prepare_export_checked_free":"public_current_source_checked_export_stage_readback_publish"},
    {"cache_state",cold?"fresh_buffers_each_call":"repeated_same_buffers"},{"actual_hbm_bytes",nullptr},{"timing_resolution_ns",nullptr},
    {"isa_verified",false},{"promotion_eligible",false},{"certified_magnitude_bound",actual_bound.str()},{"samples",samples},
    {"sustained",{{"calls",calls},{"minimum_calls",16},{"maximum_calls",4096},{"target_ns",250000000},{"independent_audit_inside_interval",false},
      {"elapsed_ns",elapsed},{"mean_call_ns",elapsed/calls},{"final_exact_reference_passed",true},
      {"scope",cold?"fresh_buffers_transfer_inclusive_no_intermediate_reference":"resident_same_buffers_no_intermediate_poison_or_reference"}}}};
  if(!cold){report["retained_public_device_payload_bytes"]=resident.retained_device_payload();
    if(resident.checked)checked_info=resident.checked_info();
    if(resident.quarter){rns8_quarter_crt_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;check(rns8_get_quarter_crt_info(resident.quarter,&info));
      report["quarter_owner_payload_bytes"]=info.retained_payload_bytes;report["quarter_certificate_sha256"]=info.certificate_sha256;report["delayed_digits"]=info.delayed_digits;}}
  if(algorithm>=0){report["checked_owner"]=checked_info;report["checked_owner_info_scope"]=!cold?"final_resident_owner":"separate_untimed_cold_audit_owner";
    report["checked_owner_payload_reservation_bytes"]=owner_bytes;report["payload_envelope_scope"]="modeled_buffers_and_retained_payloads_excludes_opaque_owner_metadata";}
  if(gemm){if(!cold)gemm_info=resident.gemm_info(envelope);report.update(gemm_info);
    report["schema"]="rns8_crt_microcase_v6";report["k"]=k;report["gemm_measured"]=true;report["operation"]="public_gemm_narrow_crt";
    report["dataset_recipe"]="splitmix64_binary_gemm_seed3903_v1";report["reference_method"]="exact_binary_integer_popcount_all_output_cells";
    report["input_contract"]="engine_current_binary_integer_operands_global_k_bound";
    report["plan_admission_policy"]="fixed_prefix_scalar_first_global_k_bound_gemm_export";
    report["logical_rows"]=side;report["logical_cols"]=side;report["logical_ld"]=side;
    report["payload_envelope_scope"]="modeled_buffers_and_retained_payloads_workspace_separately_admitted_excludes_opaque_metadata";
    report["timing_scope"]=cold?"public_context_plan_workspace_pack_gemm_prepare_export_checked_free":"public_current_operands_gemm_refresh_export_stage_readback_publish";
    report["cache_state"]=cold?"fresh_buffers_each_call":"repeated_same_operands_new_output_generation";
    report["gemm"]={{"m",side},{"n",side},{"k",k},{"useful_integer_operations",uint64_t(2)*cells*k},{"selected_prefix",prefix},
      {"proof_policy","exact_scalar_first_global_k_bound"},{"initial_setup_products",cold?0:1},{"products_per_timed_call",1},
      {"export_refresh",algorithm>=0?"retained_owner_rebind":quotient?"destroy_and_prepare_current_quarter_owner":"maintained_default_export"}};
  }
  resident.close();std::cout<<report.dump(2)<<'\n';return 0;
}
