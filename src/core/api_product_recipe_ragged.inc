rns8_status rns8_create_product_recipe_ragged(rns8_context* ctx,const rns8_product_recipe_desc* desc,
    const rns8_ragged_recipe_binding* bindings,uint32_t count,const uint32_t* output_ids,uint32_t output_count,
    uint64_t budget,rns8_product_recipe** out) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!ctx || !desc || !out || !valid_abi(desc->struct_size,desc->abi_version,sizeof(*desc)) || !desc->inputs || !desc->nodes ||
        !desc->input_count || desc->input_count>4096 || !desc->node_count || desc->node_count>4096 ||
        !bindings || !count || count>desc->input_count || output_count>4096 || (output_count && !output_ids) || (!output_count && output_ids) ||
        (desc->prefix_policy!=RNS8_PRODUCT_FULL_PREFIX && desc->prefix_policy!=RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX))return RNS8_INVALID_ARGUMENT;
    *out=nullptr;std::vector<const rns8_matrix*> inputs(desc->inputs,desc->inputs+desc->input_count);
    std::vector<rns8_ragged_recipe_binding> adapted(bindings,bindings+count);RaggedRecipeResources resources;
    auto status=ragged_recipe_preflight(ctx,inputs,adapted,budget,resources);if(status!=RNS8_SUCCESS)return status;
    std::vector<ProductShape> shapes;for(auto* input:inputs){if(!input)return RNS8_INVALID_ARGUMENT;shapes.push_back({input->desc.rows,input->desc.cols});}
    std::vector<ProductNode> nodes;for(uint32_t i=0;i<desc->node_count;++i)nodes.push_back({desc->nodes[i].a,desc->nodes[i].b});
    std::vector<uint32_t> outputs;if(output_count)outputs.assign(output_ids,output_ids+output_count);
    const auto topology=plan_product_arena(shapes,nodes,inputs.front()->prefix,UINT64_MAX,{},outputs);if(topology.status!=RNS8_SUCCESS)return topology.status;
    // These explicit caller-owned matrices are conversion scratch, not the
    // recipe's observable outputs. All source/budget/topology checks precede it.
    for(const auto& binding:adapted){status=rns8_materialize_ragged_exact(ctx,binding.source,binding.destination,budget);if(status!=RNS8_SUCCESS)return status;}
    auto adjusted=*desc;adjusted.max_arena_and_lift_bytes=std::min(adjusted.max_arena_and_lift_bytes,budget-resources.total);
    rns8_product_recipe* raw=nullptr;status=create_product_recipe(ctx,&adjusted,outputs,&raw);if(status!=RNS8_SUCCESS)return status;
    std::unique_ptr<rns8_product_recipe> result(raw);result->ragged=std::move(adapted);
    status=retain_ragged_owners(resources.unique,result->ragged_holds);if(status!=RNS8_SUCCESS)return status;
    result->total_budget=budget;result->materialized_input_bytes=resources.inputs;result->ragged_bytes=resources.owners;result->materialization_peak=resources.conversion_peak;
    *out=result.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_refresh_product_recipe_ragged(rns8_product_recipe* recipe,const rns8_ragged_recipe_binding* bindings,uint32_t count) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!recipe || recipe->closing || recipe->ragged.empty() || !bindings || count!=recipe->ragged.size())return RNS8_INVALID_ARGUMENT;
    std::vector<rns8_ragged_recipe_binding> next(bindings,bindings+count);
    for(size_t i=0;i<next.size();++i)if(next[i].input_index!=recipe->ragged[i].input_index || next[i].destination!=recipe->ragged[i].destination)return RNS8_INVALID_ARGUMENT;
    RaggedRecipeResources resources;auto status=ragged_recipe_preflight(recipe->ctx,recipe->inputs,next,recipe->total_budget,resources);if(status!=RNS8_SUCCESS)return status;
    const auto arena=recipe->owner->allocation().arena_bytes,lift=recipe->owner->lift_scratch_bytes();
    if(arena>recipe->total_budget-resources.total || lift>recipe->total_budget-resources.total-arena)return RNS8_WORKSPACE_TOO_SMALL;
    recipe->binding_valid=false;
    for(const auto& binding:next){status=rns8_materialize_ragged_exact(recipe->ctx,binding.source,binding.destination,recipe->total_budget);if(status!=RNS8_SUCCESS)return status;}
    std::vector<rns8_ragged_materialization*> holds;status=retain_ragged_owners(resources.unique,holds);if(status!=RNS8_SUCCESS)return status;
    struct Rollback{std::vector<rns8_ragged_materialization*>& owners;~Rollback(){for(auto* owner:owners)release_ragged_recipe_binding(owner);}} rollback{holds};
    status=recipe->owner->rebind(recipe->inputs);if(status!=RNS8_SUCCESS)return status;
    recipe->ragged.swap(next);recipe->ragged_holds.swap(holds);recipe->materialized_input_bytes=resources.inputs;recipe->ragged_bytes=resources.owners;recipe->materialization_peak=resources.conversion_peak;
    recipe->binding_valid=true;return RNS8_SUCCESS;
  });
}
