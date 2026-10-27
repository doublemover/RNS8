// Bounded concurrency uses distinct prepared owners and their existing tokens.
// A terminal slot remains leased until explicit release, so observing another
// epoch cannot overwrite an unconsumed completion record.
struct rns8_prepared_pool {
  struct Slot {rns8_prepared_execution* execution=nullptr;rns8_completion_token token{};bool leased=false;};
  std::vector<Slot> slots;
  rns8_prepared_group* group=nullptr;
  uint32_t cursor=0;
  bool graph=false;
};
namespace {
struct PoolDispatch {
  rns8_prepared_execution& execution;
  explicit PoolDispatch(rns8_prepared_execution& e):execution(e){execution.pool_dispatch=true;}
  ~PoolDispatch(){execution.pool_dispatch=false;}
};
rns8_prepared_pool::Slot* pool_token_slot(rns8_prepared_pool* pool,rns8_completion_token token) {
  if(!pool || !token.prepared_id || !token.epoch)return nullptr;
  for(auto& slot:pool->slots)if(slot.leased && slot.token.prepared_id==token.prepared_id && slot.token.epoch==token.epoch)return &slot;
  return nullptr;
}
}
rns8_status rns8_create_prepared_pool(rns8_prepared_execution* const* executions,
    uint32_t count,uint32_t use_owned_graph,rns8_prepared_pool** out) {
  return guard_api([&]() -> rns8_status {
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!executions || !count || count>64 || use_owned_graph>1)return RNS8_INVALID_ARGUMENT;
    auto pool=std::make_unique<rns8_prepared_pool>();pool->slots.reserve(count);pool->graph=use_owned_graph!=0;
    std::unordered_set<const rns8_workspace*> workspaces;
    for(uint32_t i=0;i<count;++i) {
      auto* e=executions[i];if(!e)return RNS8_INVALID_ARGUMENT;
      if(e->running || e->pool_owned || e->caller_graph_references || e->group_references || e->producer_count || e->dependent_references)return RNS8_BUSY;
      if(e->ctx!=executions[0]->ctx)return RNS8_INVALID_ARGUMENT;
      if(e->workspace && !workspaces.insert(e->workspace).second)return RNS8_INVALID_ARGUMENT;
      if(!e->hip_submission_prepared)return RNS8_UNSUPPORTED_BACKEND;
      const auto status=prepared_hip_support_status(*e);if(status!=RNS8_SUCCESS)return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      if(pool->graph && (!e->graph_executable || e->graph_a!=prepared_device_storage(*e->a) ||
          e->graph_b!=prepared_device_storage(*e->b) || e->graph_c!=prepared_device_storage(*e->c) ||
          (e->d && e->graph_d!=prepared_device_storage(*e->d))))return RNS8_INVALID_ARGUMENT;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
      pool->slots.push_back({e,{},false});
    }
    const auto status=rns8_create_prepared_group(executions,count,&pool->group);if(status!=RNS8_SUCCESS)return status;
    for(auto& slot:pool->slots)slot.execution->pool_owned=true;
    *out=pool.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_submit_prepared_pool_hip(rns8_prepared_pool* pool,void* stream,
    uint32_t* out_slot,rns8_completion_token* token) {
  return guard_api([&]() -> rns8_status {
    if(!pool || !out_slot || !token || !stream)return RNS8_INVALID_ARGUMENT;
    for(uint32_t offset=0;offset<pool->slots.size();++offset) {
      const auto index=(pool->cursor+offset)%static_cast<uint32_t>(pool->slots.size());auto& slot=pool->slots[index];
      if(slot.leased)continue;
      auto* e=slot.execution;PoolDispatch dispatch(*e);PreparedHipProof proof;
      const auto admitted=prove_prepared_submission(e,stream,pool->graph,proof);if(admitted!=RNS8_SUCCESS)return admitted;
      const auto previous=e->epoch;rns8_completion_token accepted{};
      const auto status=start_prepared_submission(e,stream,&accepted,pool->graph,proof);
      if(e->epoch!=previous) {
        slot.token=accepted;slot.leased=true;*token=accepted;*out_slot=index;
        pool->cursor=(index+1)%static_cast<uint32_t>(pool->slots.size());
      }
      return status;
    }
    return RNS8_BUSY;
  });
}
rns8_status rns8_refresh_prepared_pool_slot(rns8_prepared_pool* pool,uint32_t index) {
  if(!pool || index>=pool->slots.size())return RNS8_INVALID_ARGUMENT;
  auto& slot=pool->slots[index];if(slot.leased)return RNS8_BUSY;
  PoolDispatch dispatch(*slot.execution);return rns8_refresh_prepared_inputs(slot.execution);
}
rns8_status rns8_query_prepared_pool_hip(rns8_prepared_pool* pool,rns8_completion_token token,rns8_completion_info* info) {
  auto* slot=pool_token_slot(pool,token);if(!slot)return RNS8_INVALID_ARGUMENT;
  return rns8_query_prepared_hip(slot->execution,token,info);
}
rns8_status rns8_wait_prepared_pool_hip_for(rns8_prepared_pool* pool,rns8_completion_token token,
    uint64_t timeout_us,rns8_completion_info* info) {
  auto* slot=pool_token_slot(pool,token);if(!slot)return RNS8_INVALID_ARGUMENT;
  return rns8_wait_prepared_hip_for(slot->execution,token,timeout_us,info);
}
rns8_status rns8_cancel_prepared_pool_hip(rns8_prepared_pool* pool,rns8_completion_token token) {
  auto* slot=pool_token_slot(pool,token);if(!slot)return RNS8_INVALID_ARGUMENT;
  return rns8_cancel_prepared_hip(slot->execution,token);
}
rns8_status rns8_release_prepared_pool_slot(rns8_prepared_pool* pool,rns8_completion_token token) {
  auto* slot=pool_token_slot(pool,token);if(!slot)return RNS8_INVALID_ARGUMENT;
  // Release acknowledges that the caller consumed this completion. Querying a
  // device event is deliberately insufficient: publication must be observed.
  if(slot->execution->running || slot->execution->dependent_references)return RNS8_BUSY;
  slot->leased=false;slot->token={};return RNS8_SUCCESS;
}
rns8_status rns8_destroy_prepared_pool(rns8_prepared_pool* pool) {
  if(!pool)return RNS8_SUCCESS;
  for(const auto& slot:pool->slots)if(slot.leased || slot.execution->running)return RNS8_BUSY;
  const auto status=rns8_destroy_prepared_group(pool->group);if(status!=RNS8_SUCCESS)return status;
  for(auto& slot:pool->slots)slot.execution->pool_owned=false;
  delete pool;return RNS8_SUCCESS;
}
