namespace {
struct ProductInputProjection {const rns8_matrix* source=nullptr;boost::multiprecision::cpp_int bound=0;};
bool materialization_text(const char* text,size_t size){return std::memchr(text,0,size)!=nullptr;}
std::string product_materialization_build() {
  std::string value="rns8-product-materialization-v2;";
#ifdef RNS8_BUILD_MANIFEST_JSON
  value+=RNS8_BUILD_MANIFEST_JSON;
#else
  return {};
#endif
  return rns8::detail::sha256(value);
}
rns8_status product_materialization_identity(rns8_context* ctx,const rns8_product_recipe_desc* desc,
    const std::vector<uint32_t>& outputs,rns8_product_materialization_identity& identity,
    const std::vector<ProductInputProjection>* projected=nullptr) {
  using namespace rns8::detail;using namespace rns8::detail::api;
  if(!ctx || !desc || !valid_abi(desc->struct_size,desc->abi_version,sizeof(*desc)) ||
      !desc->inputs || !desc->nodes || !desc->input_count || desc->input_count>4096 ||
      !desc->node_count || desc->node_count>64 ||
      (desc->prefix_policy!=RNS8_PRODUCT_FULL_PREFIX && desc->prefix_policy!=RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX))
    return RNS8_INVALID_ARGUMENT;
  if(projected && projected->size()!=desc->input_count)return RNS8_INVALID_ARGUMENT;
  if(ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
  if(ctx->backend!=RNS8_BACKEND_CPU_REFERENCE && ctx->backend!=RNS8_BACKEND_HIP_DIRECT)return RNS8_UNSUPPORTED_BACKEND;
  if(!desc->inputs[0])return RNS8_INVALID_ARGUMENT;
  const auto* first=desc->inputs[0];const auto basis=recipe_basis(*first);
  const auto ordered=ordered_basis_identity(basis->packet);const auto prefix=first->prefix;
  std::string key="rns8-product-graph-v2;";
  const auto add=[&](const std::string& text){key+=std::to_string(text.size())+":"+text;};
  add(std::to_string(ctx->backend));add(std::to_string(first->desc.semantics));add(std::to_string(prefix));
  add(std::to_string(desc->prefix_policy));add(std::to_string(ordered.representation));add(std::to_string(ordered.count));
  for(auto modulus:ordered.moduli)add(std::to_string(modulus));
  std::vector<ProductShape> shapes;std::vector<ProductNode> nodes;std::vector<cpp_int> bounds;
  add(std::to_string(desc->input_count));add(std::to_string(desc->node_count));
  for(uint32_t i=0;i<desc->input_count;++i) {
    const auto* input=desc->inputs[i];if(!input)return RNS8_INVALID_ARGUMENT;
    const auto* source=projected && (*projected)[i].source?(*projected)[i].source:input;
    const auto& bound=source!=input?(*projected)[i].bound:input->exact_max_magnitude;
    if(input->in_flight_writer)return RNS8_BUSY;
    if(input->backend!=ctx->backend || input->desc.semantics!=first->desc.semantics || input->prefix!=prefix ||
        !exact_wide_semantics(input->desc.semantics) || input->generation_exhausted ||
        !rns_residue_state_current_for_backend(*source,ctx->backend) ||
        !recipe_storage_matches(*input,ctx->backend,input->desc.rows,input->desc.cols,prefix,ordered) ||
        (ctx->backend==RNS8_BACKEND_HIP_DIRECT && input->hip_device_id!=ctx->device_id))return RNS8_INVALID_ARGUMENT;
    if(source->exact_range_prefix<source->prefix || source->exact_max_magnitude<0 ||
        !exact_range_fits(source->desc.semantics,source->exact_max_magnitude,recipe_basis(*source)->products[source->prefix]) ||
        bound<0 || bound>source->exact_max_magnitude ||
        !exact_range_fits(input->desc.semantics,bound,basis->products[prefix]))return RNS8_RANGE_ERROR;
    shapes.push_back({input->desc.rows,input->desc.cols});
    bounds.push_back(bound);
    add(std::to_string(input->desc.rows));add(std::to_string(input->desc.cols));add(bound.str());
  }
  for(uint32_t i=0;i<desc->node_count;++i){nodes.push_back({desc->nodes[i].a,desc->nodes[i].b});add(std::to_string(nodes.back().a));add(std::to_string(nodes.back().b));}
  const auto arena=plan_product_arena(shapes,nodes,prefix,UINT64_MAX,{},outputs);
  if(arena.status!=RNS8_SUCCESS)return arena.status;
  const auto observed=outputs.empty()?std::vector<uint32_t>{desc->input_count+desc->node_count-1}:outputs;
  add(std::to_string(observed.size()));for(auto id:observed)add(std::to_string(id));
  const std::string target=ctx->backend==RNS8_BACKEND_HIP_DIRECT?ctx->device_info.gcn_arch:"cpu";
  const auto build=product_materialization_build();
  if(target.empty() || target.size()>=sizeof(identity.target) || build.empty())return RNS8_UNSUPPORTED_BACKEND;
  add(target);
  identity={};identity.struct_size=sizeof(identity);identity.abi_version=RNS8_ABI_VERSION;identity.node_count=desc->node_count;
  identity.admission_status=RNS8_SUCCESS;identity.blocking_node=UINT32_MAX;
  for(uint32_t i=0;i<desc->node_count;++i) {
    const auto node=nodes[i];const auto shape=arena.values[node.a].shape;
    cpp_int bound=cpp_int(shape.cols)*bounds[node.a]*bounds[node.b];
    if(!exact_range_fits(first->desc.semantics,bound,basis->products[prefix])) {
      identity.admission_status=RNS8_RANGE_ERROR;identity.blocking_node=i;break;
    }
    uint32_t needed=1;
    while(needed<prefix && !exact_range_fits(first->desc.semantics,bound,basis->products[needed]))++needed;
    identity.minimum_prefixes[i]=needed;bounds.push_back(std::move(bound));
  }
  set_text(identity.graph_sha256,sizeof(identity.graph_sha256),sha256(key));
  set_text(identity.build_sha256,sizeof(identity.build_sha256),build);
  set_text(identity.target,sizeof(identity.target),target);return RNS8_SUCCESS;
}
}
rns8_status rns8_query_product_materialization_identity(rns8_context* ctx,const rns8_product_recipe_desc* desc,
    const uint32_t* ids,uint32_t count,rns8_product_materialization_identity* out) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if(!out || !rns8::detail::valid_abi(out->struct_size,out->abi_version,sizeof(*out)) ||
        count>64 || (count && !ids))return RNS8_INVALID_ARGUMENT;
    rns8_product_materialization_identity identity{};
    const auto status=product_materialization_identity(ctx,desc,count?std::vector<uint32_t>(ids,ids+count):std::vector<uint32_t>{},identity);
    if(status==RNS8_SUCCESS)*out=identity;return status;
  });
}
namespace {
rns8_status materialization_profile(rns8_context* ctx,const rns8_product_recipe_desc* desc,
    const std::vector<uint32_t>& outputs,const rns8_product_materialization_profile* prices,
    const rns8_product_materialization_identity& identity,rns8::detail::ProductMaterializationProfile& profile) {
  using namespace rns8::detail;
    if(!prices || !valid_abi(prices->struct_size,prices->abi_version,sizeof(*prices)) ||
        prices->cost_schema_version!=RNS8_PRODUCT_MATERIALIZATION_COST_SCHEMA_VERSION ||
        prices->node_costs_size!=sizeof(rns8_product_materialization_node_costs) ||
        prices->reserved || (prices->flags&~uint32_t(RNS8_PRODUCT_RETAIN_EXTERNAL_PACKED_B|RNS8_PRODUCT_REQUIRE_PROJECTED_AMORTIZATION)) ||
        !prices->expected_replays || prices->expected_replays==UINT64_MAX ||
        !prices->nodes || !prices->node_count || prices->node_count>64 ||
        prices->max_evaluations<4 || prices->max_evaluations>65536 ||
        !prices->max_chain_frontier || prices->max_chain_frontier>4096 ||
        !valid_abi(prices->identity.struct_size,prices->identity.abi_version,sizeof(prices->identity)) ||
        !materialization_text(prices->identity.graph_sha256,sizeof(prices->identity.graph_sha256)) ||
        !materialization_text(prices->identity.build_sha256,sizeof(prices->identity.build_sha256)) ||
        !materialization_text(prices->identity.target,sizeof(prices->identity.target)) ||
        !materialization_text(prices->calibration_identity,sizeof(prices->calibration_identity)) || !prices->calibration_identity[0])return RNS8_INVALID_ARGUMENT;
    if(identity.admission_status!=RNS8_SUCCESS)return identity.admission_status;
    if(prices->node_count!=identity.node_count || prices->identity.node_count!=identity.node_count ||
        std::strcmp(prices->identity.graph_sha256,identity.graph_sha256) ||
        std::strcmp(prices->identity.build_sha256,identity.build_sha256) ||
        std::strcmp(prices->identity.target,identity.target) ||
        prices->identity.admission_status!=identity.admission_status || prices->identity.blocking_node!=identity.blocking_node ||
        !std::equal(std::begin(prices->identity.minimum_prefixes),std::end(prices->identity.minimum_prefixes),
            std::begin(identity.minimum_prefixes)))return RNS8_INVALID_ARGUMENT;
    profile.cost_schema_version=prices->cost_schema_version;
    profile.max_frontier=prices->max_chain_frontier;
    profile.replay.replays=prices->expected_replays;
    profile.replay.preparation_ns=prices->preparation_ns;profile.replay.calibration_ns=prices->calibration_ns;
    profile.replay.orchestration_ns=prices->orchestration_ns;profile.replay.publication_ns=prices->publication_ns;
    profile.replay.baseline_replay_ns=prices->baseline_replay_ns;profile.replay.baseline_preparation_ns=prices->baseline_preparation_ns;
    profile.replay.retain_external_b=(prices->flags&RNS8_PRODUCT_RETAIN_EXTERNAL_PACKED_B)!=0;
    profile.replay.require_amortization=(prices->flags&RNS8_PRODUCT_REQUIRE_PROJECTED_AMORTIZATION)!=0;
    auto& uniform=profile.uniform;uniform.max_evaluations=prices->max_evaluations;uniform.backend=ctx->backend;
    uniform.semantics=desc->inputs[0]->desc.semantics;uniform.target=identity.target;uniform.provenance=prices->calibration_identity;
    uniform.basis=ordered_basis_identity(recipe_basis(*desc->inputs[0])->packet);
    std::vector<ProductShape> shapes;std::vector<ProductNode> nodes;
    for(uint32_t i=0;i<desc->input_count;++i){shapes.push_back({desc->inputs[i]->desc.rows,desc->inputs[i]->desc.cols});uniform.experimental_basis=uniform.experimental_basis || bool(desc->inputs[i]->experimental_basis);}
    for(uint32_t i=0;i<desc->node_count;++i)nodes.push_back({desc->nodes[i].a,desc->nodes[i].b});
    uniform.graph_key=product_graph_key(shapes,nodes,desc->inputs[0]->prefix,outputs);
    uniform.nodes.resize(prices->node_count);profile.packed.resize(prices->node_count);
    for(uint32_t i=0;i<prices->node_count;++i) {
      std::copy_n(prices->nodes[i].uniform_gemm_ns,RNS8_PRODUCT_MATERIALIZATION_UNIFORM_GEMM_COST_COUNT,uniform.nodes[i].gemm.begin());
      std::copy_n(prices->nodes[i].lift_ns,RNS8_PRODUCT_MATERIALIZATION_LIFT_COST_COUNT,uniform.nodes[i].lift.begin());
      std::copy_n(prices->nodes[i].packed_gemm_ns,RNS8_PRODUCT_MATERIALIZATION_PACKED_GEMM_COST_COUNT,profile.packed[i].gemm.begin());
      std::copy_n(prices->nodes[i].pack_ns,RNS8_PRODUCT_MATERIALIZATION_PACK_COST_COUNT,profile.packed[i].pack.begin());
    }
  return RNS8_SUCCESS;
}
rns8_status ragged_materialization_identity(rns8_context* ctx,const rns8_product_recipe_desc* desc,
    const rns8_ragged_recipe_binding* bindings,uint32_t count,const std::vector<uint32_t>& outputs,
    uint64_t budget,rns8_product_materialization_identity& identity) {
  using namespace rns8::detail;
  if(!ctx || !desc || !valid_abi(desc->struct_size,desc->abi_version,sizeof(*desc)) ||
      !desc->inputs || !desc->input_count || desc->input_count>4096 || !bindings || !count || count>desc->input_count)
    return RNS8_INVALID_ARGUMENT;
  std::vector<const rns8_matrix*> inputs(desc->inputs,desc->inputs+desc->input_count);
  std::vector<ProductInputProjection> projected(desc->input_count);
  std::vector<rns8_ragged_recipe_binding> adapted(bindings,bindings+count);RaggedRecipeResources resources;
  auto status=ragged_recipe_preflight(ctx,inputs,adapted,budget,resources);if(status!=RNS8_SUCCESS)return status;
  for(const auto& binding:adapted) {
    uint64_t retained=0,capacity=0,words[3]{};auto& fact=projected[binding.input_index];
    status=ragged_recipe_binding_info(ctx,binding.source,fact.source,retained,capacity,words);
    if(status!=RNS8_SUCCESS)return status;
    fact.bound=boost::multiprecision::cpp_int(words[0])+(boost::multiprecision::cpp_int(words[1])<<64)+
        (boost::multiprecision::cpp_int(words[2])<<128);
  }
  return product_materialization_identity(ctx,desc,outputs,identity,&projected);
}
}
rns8_status rns8_create_materialized_product_recipe(rns8_context* ctx,const rns8_product_recipe_desc* desc,
    const uint32_t* ids,uint32_t count,const rns8_product_materialization_profile* prices,rns8_product_recipe** out) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if(!out || count>64 || (count && !ids))return RNS8_INVALID_ARGUMENT;
    const auto outputs=count?std::vector<uint32_t>(ids,ids+count):std::vector<uint32_t>{};
    rns8_product_materialization_identity identity{};
    auto status=product_materialization_identity(ctx,desc,outputs,identity);if(status!=RNS8_SUCCESS)return status;
    rns8::detail::ProductMaterializationProfile profile;
    status=materialization_profile(ctx,desc,outputs,prices,identity,profile);if(status!=RNS8_SUCCESS)return status;
    return create_product_recipe(ctx,desc,outputs,out,&profile);
  });
}
rns8_status rns8_query_product_materialization_identity_ragged(rns8_context* ctx,const rns8_product_recipe_desc* desc,
    const rns8_ragged_recipe_binding* bindings,uint32_t binding_count,const uint32_t* ids,uint32_t count,
    uint64_t budget,rns8_product_materialization_identity* out) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if(!out || !rns8::detail::valid_abi(out->struct_size,out->abi_version,sizeof(*out)) ||
        count>64 || (count && !ids))return RNS8_INVALID_ARGUMENT;
    const auto outputs=count?std::vector<uint32_t>(ids,ids+count):std::vector<uint32_t>{};
    rns8_product_materialization_identity identity{};
    const auto status=ragged_materialization_identity(ctx,desc,bindings,binding_count,outputs,budget,identity);
    if(status==RNS8_SUCCESS)*out=identity;return status;
  });
}
rns8_status rns8_create_materialized_product_recipe_ragged(rns8_context* ctx,const rns8_product_recipe_desc* desc,
    const rns8_ragged_recipe_binding* bindings,uint32_t binding_count,const uint32_t* ids,uint32_t count,
    uint64_t budget,const rns8_product_materialization_profile* prices,rns8_product_recipe** out) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if(!out || count>64 || (count && !ids))return RNS8_INVALID_ARGUMENT;
    const auto outputs=count?std::vector<uint32_t>(ids,ids+count):std::vector<uint32_t>{};
    rns8_product_materialization_identity identity{};
    auto status=ragged_materialization_identity(ctx,desc,bindings,binding_count,outputs,budget,identity);
    if(status!=RNS8_SUCCESS)return status;
    rns8::detail::ProductMaterializationProfile profile;
    status=materialization_profile(ctx,desc,outputs,prices,identity,profile);if(status!=RNS8_SUCCESS)return status;
    return create_product_recipe_ragged(ctx,desc,bindings,binding_count,ids,count,budget,out,&profile);
  });
}
rns8_status rns8_get_product_materialization_info(const rns8_product_recipe* recipe,rns8_product_materialization_info* out) {
  if(!recipe || !out || !rns8::detail::valid_abi(out->struct_size,out->abi_version,sizeof(*out)))return RNS8_INVALID_ARGUMENT;
  const auto& plan=recipe->owner->materialization_selection();if(plan.status!=RNS8_SUCCESS)return RNS8_UNSUPPORTED_BACKEND;
  rns8_product_materialization_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
  info.node_count=static_cast<uint32_t>(plan.prefixes.size());info.evaluated=plan.evaluated;info.exhaustive=plan.exhaustive;
  info.chain_dynamic_programming=plan.chain_dynamic_programming;info.cost_optimal=plan.cost_optimal;info.estimated_ns=plan.estimated_ns;
  info.retained_arena_and_lift_bytes=plan.arena.arena_bytes+plan.lift_scratch_bytes;info.packed_payload_bytes=plan.packed_payload_bytes;
  info.completed_packing_calls=recipe->owner->completed_packing_calls();
  info.estimated_first_replay_ns=plan.replay_cost.first_ns;info.estimated_steady_replay_ns=plan.replay_cost.steady_ns;
  info.one_time_ns=plan.replay_cost.once_ns;info.lifetime_ns=plan.replay_cost.lifetime_ns;
  info.amortized_ns=plan.replay_cost.amortized_ns;
  info.expected_replays=recipe->owner->expected_replays();info.projected_break_even_replays=plan.replay_cost.break_even_replays;
  info.amortization_status=plan.replay_cost.amortization_status;info.retained_external_packed_bytes=plan.retained_external_packed_bytes;
  info.completed_replays=recipe->owner->completed_replays();info.reused_external_packing_calls=recipe->owner->reused_packing_calls();
  info.flags=(plan.retain_external_b?RNS8_PRODUCT_RETAIN_EXTERNAL_PACKED_B:0)|
      (recipe->owner->replay_policy().require_amortization?RNS8_PRODUCT_REQUIRE_PROJECTED_AMORTIZATION:0);
  rns8::detail::api::set_text(info.calibration_identity,sizeof(info.calibration_identity),recipe->materialization_calibration);
  *out=info;return RNS8_SUCCESS;
}
rns8_status rns8_get_product_materialization_choices(const rns8_product_recipe* recipe,
    rns8_product_materialization_choice* choices,uint32_t capacity) {
  if(!recipe || !choices)return RNS8_INVALID_ARGUMENT;
  const auto& plan=recipe->owner->materialization_selection();if(plan.status!=RNS8_SUCCESS)return RNS8_UNSUPPORTED_BACKEND;
  if(capacity<plan.prefixes.size())return RNS8_WORKSPACE_TOO_SMALL;
  for(size_t i=0;i<plan.prefixes.size();++i) {
    rns8_product_materialization_choice result{};result.compute_prefix=plan.prefixes[i];result.packed_b=plan.packed_b[i];
    // The arena plan's raw values do not encode topology; retain the B value's
    // selected view through the runtime accessor below.
    if(result.packed_b) {
      const auto& packed=plan.packed[recipe->owner->materialization_b_value(static_cast<uint32_t>(i))];
      result.packed_capacity_prefix=packed.layout.prefix;result.first_packed_reader=packed.first;result.last_packed_reader=packed.last;
      result.packed_bytes=packed.layout.bytes;result.arena_offset=packed.offset;
    }
    choices[i]=result;
  }
  return RNS8_SUCCESS;
}
