namespace {
struct RaggedRecipeResources {uint64_t inputs=0,owners=0,conversion_peak=0,total=0;std::vector<rns8_ragged_materialization*> unique;};
rns8_status ragged_recipe_preflight(rns8_context* ctx,const std::vector<const rns8_matrix*>& inputs,
    const std::vector<rns8_ragged_recipe_binding>& bindings,uint64_t budget,RaggedRecipeResources& resources) {
  using namespace rns8::detail;using namespace rns8::detail::api;
  if(!ctx || bindings.empty() || bindings.size()>inputs.size())return RNS8_INVALID_ARGUMENT;
  std::unordered_set<uint32_t> indices;std::unordered_set<rns8_matrix*> destinations;std::unordered_set<const rns8_matrix*> sources;
  std::unordered_map<rns8_ragged_materialization*,uint64_t> needed;
  const auto add=[](uint64_t& sum,uint64_t n){if(n>UINT64_MAX-sum)return false;sum+=n;return true;};
  for(const auto& binding:bindings){auto* target=binding.destination;
    if(!target || !binding.source || binding.input_index>=inputs.size() || inputs[binding.input_index]!=target ||
        !indices.insert(binding.input_index).second || !destinations.insert(target).second)return RNS8_INVALID_ARGUMENT;
    const rns8_matrix* source=nullptr;uint64_t retained=0,capacity=0,words[3]{};
    auto status=ragged_recipe_binding_info(ctx,binding.source,source,retained,capacity,words);if(status!=RNS8_SUCCESS)return status;
    sources.insert(source);
    if(target==source || target->backend!=ctx->backend || target->desc.semantics!=source->desc.semantics || target->desc.rows!=source->desc.rows || target->desc.cols!=source->desc.cols ||
        !target->prefix || target->prefix>20 || (ctx->backend==RNS8_BACKEND_HIP_DIRECT && target->hip_device_id!=ctx->device_id) ||
        !basis_storage_matches(*target,ctx->backend) ||
        !(ordered_basis_identity(stored_basis(*source)->packet)==ordered_basis_identity(stored_basis(*target)->packet)))return RNS8_INVALID_ARGUMENT;
    if(target->in_flight_writer || target->in_flight_readers)return RNS8_BUSY;
    if(target->generation_exhausted || target->content_generation==UINT64_MAX)return RNS8_RANGE_ERROR;
    const boost::multiprecision::cpp_int bound=boost::multiprecision::cpp_int(words[0])+
        (boost::multiprecision::cpp_int(words[1])<<64)+(boost::multiprecision::cpp_int(words[2])<<128);
    if(!exact_range_fits(target->desc.semantics,bound,stored_basis(*target)->products[target->prefix]))return RNS8_RANGE_ERROR;
    rns8_matrix_allocation_info allocation{};allocation.struct_size=sizeof(allocation);allocation.abi_version=RNS8_ABI_VERSION;
    status=rns8_get_matrix_allocation_info(target,&allocation);if(status!=RNS8_SUCCESS)return status;
    if(!add(resources.inputs,allocation.host_payload_capacity_bytes) || !add(resources.inputs,allocation.device_total_capacity_bytes))return RNS8_RANGE_ERROR;
    const uint64_t bytes=uint64_t(target->desc.rows)*target->desc.cols*target->prefix;
    auto [it,inserted]=needed.emplace(binding.source,bytes);if(!inserted)it->second=std::max(it->second,bytes);
    if(ctx->backend==RNS8_BACKEND_CPU_REFERENCE)resources.conversion_peak=std::max(resources.conversion_peak,bytes);
  }
  for(auto* target:destinations)if(sources.count(target))return RNS8_INVALID_ARGUMENT;
  resources.unique.reserve(needed.size());
  for(const auto& [owner,bytes]:needed){const rns8_matrix* source=nullptr;uint64_t retained=0,capacity=0;
    auto status=ragged_recipe_binding_info(ctx,owner,source,retained,capacity);if(status!=RNS8_SUCCESS)return status;
    if(ctx->backend==RNS8_BACKEND_HIP_DIRECT && bytes>capacity && !add(retained,bytes-capacity))return RNS8_RANGE_ERROR;
    if(!add(resources.owners,retained))return RNS8_RANGE_ERROR;resources.unique.push_back(owner);}
  resources.total=resources.inputs;if(!add(resources.total,resources.owners) || !add(resources.total,resources.conversion_peak))return RNS8_RANGE_ERROR;
  return resources.total>budget?RNS8_WORKSPACE_TOO_SMALL:RNS8_SUCCESS;
}
rns8_status validate_ragged_recipe_binding(const rns8_product_recipe& recipe) {
  if(recipe.closing || !recipe.binding_valid)return RNS8_INVALID_ARGUMENT;
  if(recipe.ragged.empty())return RNS8_SUCCESS;
  RaggedRecipeResources resources;const auto status=ragged_recipe_preflight(recipe.ctx,recipe.inputs,recipe.ragged,recipe.total_budget,resources);if(status!=RNS8_SUCCESS)return status;
  const auto arena=recipe.owner->allocation().arena_bytes,lift=recipe.owner->lift_scratch_bytes();
  return arena>recipe.total_budget-resources.total || lift>recipe.total_budget-resources.total-arena?RNS8_WORKSPACE_TOO_SMALL:RNS8_SUCCESS;
}
rns8_status retain_ragged_owners(const std::vector<rns8_ragged_materialization*>& owners,std::vector<rns8_ragged_materialization*>& retained) {
  retained=owners;size_t count=0;
  for(auto* owner:owners){const auto status=rns8::detail::retain_ragged_recipe_binding(owner);if(status!=RNS8_SUCCESS){for(size_t i=0;i<count;++i)rns8::detail::release_ragged_recipe_binding(owners[i]);retained.clear();return status;}++count;}
  return RNS8_SUCCESS;
}
}
