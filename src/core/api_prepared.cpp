#include "core/api_internal.hpp"
#include <atomic>
#include <array>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include "core/hip_resources.hpp"

using namespace rns8::detail::api;

struct rns8_prepared_execution {
  rns8_context* ctx = nullptr;
  const rns8_plan* plan = nullptr;
  const rns8_matrix* a = nullptr;
  const rns8_matrix* b = nullptr;
  const rns8_matrix* d = nullptr;
  int64_t alpha = 0, beta = 0;
  uint64_t d_instance = 0, d_generation = 0;
  rns8_matrix* c = nullptr;
  rns8_workspace* workspace = nullptr;
  uint64_t id = 0, epoch = 0;
  uint64_t a_instance = 0, a_generation = 0, b_instance = 0, b_generation = 0;
  rns8_completion_info completion{};
  rns8_async_failure_info async_failure{};
  bool running = false;
  bool cancel_requested = false;
  uint64_t group_references = 0;
  uint64_t dependent_references = 0;
  struct Dependency {rns8_prepared_execution* producer=nullptr;rns8_completion_token token{};};
  std::array<Dependency,3> producers{};
  uint32_t producer_count=0;
  bool hip_submission_prepared = false;
  void* pending_stream = nullptr;
  boost::multiprecision::cpp_int pending_bound;
  rns8_exact_matrix_ranges pending_ranges;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8::detail::hip_unique_event completion_event;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_executable = nullptr;
  const void* graph_a = nullptr;
  const void* graph_b = nullptr;
  const void* graph_c = nullptr;
  const void* graph_d = nullptr;
  ~rns8_prepared_execution() {
    if (!graph_executable && !graph && !completion_event.get()) return;
    rns8::detail::hip_scoped_device device(ctx ? ctx->device_id : 0);
    if (graph_executable) (void)hipGraphExecDestroy(graph_executable);
    if (graph) (void)hipGraphDestroy(graph);
    completion_event.reset();
  }
#endif
};

struct rns8_prepared_group {
  std::vector<rns8_prepared_execution*> executions;
  bool running = false;
};

namespace {
uint64_t next_prepared_id() {
  static std::atomic<uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  while (value != UINT64_MAX)
    if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
  return 0;
}
rns8_status validate_prepared(const rns8_prepared_execution& e) {
  auto status = validate_plan_context_workspace(*e.ctx, *e.plan, *e.workspace);
  if (status != RNS8_SUCCESS) return status;
  if (e.a->generation_exhausted || e.b->generation_exhausted || e.c->generation_exhausted)
    return RNS8_RANGE_ERROR;
  if (e.d) {
    if (e.d->generation_exhausted) return RNS8_RANGE_ERROR;
    return validate_exact_axpby_operands(*e.ctx, *e.plan, *e.a, *e.b, *e.d, *e.c, e.alpha, e.beta);
  }
  switch (e.plan->desc.semantics) {
    case RNS8_FINITE_RING_U8:
    case RNS8_FINITE_FIELD_U8:
      return validate_finite_gemm_operands(*e.ctx, *e.plan, static_cast<uint16_t>(e.plan->desc.finite_modulus), *e.a, *e.b, *e.c);
    case RNS8_WRAP_U64_MOD_2_64:
      return validate_wrap_gemm_operands(*e.ctx, *e.plan, *e.a, *e.b, *e.c);
    default:
      return validate_rns_gemm_operands(*e.ctx, *e.plan, *e.a, *e.b, *e.c);
  }
}
void bind_inputs(rns8_prepared_execution& e) {
  e.a_instance = e.a->matrix_instance_id; e.a_generation = e.a->content_generation;
  e.b_instance = e.b->matrix_instance_id; e.b_generation = e.b->content_generation;
  if (e.d) { e.d_instance = e.d->matrix_instance_id; e.d_generation = e.d->content_generation; }
}
bool inputs_match(const rns8_prepared_execution& e) {
  return e.a_instance == e.a->matrix_instance_id && e.a_generation == e.a->content_generation &&
         e.b_instance == e.b->matrix_instance_id && e.b_generation == e.b->content_generation &&
         (!e.d || (e.d_instance == e.d->matrix_instance_id && e.d_generation == e.d->content_generation));
}
// A==B is legal; hold that resource once. No allocation follows retention.
void reference_slots(const rns8_prepared_execution& e, uint64_t* (&slots)[7]) {
  slots[0] = &e.ctx->prepared_references; slots[1] = &e.plan->prepared_references;
  slots[2] = &e.a->prepared_references;
  slots[3] = e.a == e.b ? nullptr : &e.b->prepared_references;
  slots[4] = &e.c->prepared_references; slots[5] = &e.workspace->prepared_references;
  slots[6] = e.d && e.d != e.a && e.d != e.b ? &e.d->prepared_references : nullptr;
}
}

static rns8_status prepare_operation(
    rns8_context* ctx, const rns8_plan* plan, const rns8_matrix* a, const rns8_matrix* b,
    const rns8_matrix* d, rns8_matrix* c, int64_t alpha, int64_t beta,
    rns8_workspace* workspace, rns8_prepared_execution** out) {
  return guard_api([&]() -> rns8_status {
    if (!out) return RNS8_INVALID_ARGUMENT;
    *out = nullptr;
    if (!ctx || !plan || !a || !b || !c || !workspace) return RNS8_INVALID_ARGUMENT;
    auto e = std::make_unique<rns8_prepared_execution>();
    e->ctx = ctx; e->plan = plan; e->a = a; e->b = b; e->c = c; e->workspace = workspace;
    e->d = d; e->alpha = alpha; e->beta = beta;
    const auto status = validate_prepared(*e);
    if (status != RNS8_SUCCESS) return status;
    uint64_t* slots[7]{};
    reference_slots(*e, slots);
    for (const auto slot : slots) if (slot && *slot == UINT64_MAX) return RNS8_RANGE_ERROR;
    e->id = next_prepared_id();
    if (!e->id) return RNS8_RANGE_ERROR;
    bind_inputs(*e);
    for (auto slot : slots) if (slot) ++*slot;
    *out = e.release();
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_prepare_gemm(
    rns8_context* ctx, const rns8_plan* plan, const rns8_matrix* a, const rns8_matrix* b,
    rns8_matrix* c, rns8_workspace* workspace, rns8_prepared_execution** out) {
  return prepare_operation(ctx, plan, a, b, nullptr, c, 0, 0, workspace, out);
}
rns8_status rns8_prepare_exact_axpby(
    rns8_context* ctx, const rns8_plan* plan, const rns8_matrix* a, const rns8_matrix* b,
    const rns8_matrix* d, rns8_matrix* c, int64_t alpha, int64_t beta,
    rns8_workspace* workspace, rns8_prepared_execution** out) {
  if (!d) { if (out) *out = nullptr; return RNS8_INVALID_ARGUMENT; }
  return prepare_operation(ctx, plan, a, b, d, c, alpha, beta, workspace, out);
}
static void record_epilogue_completion(rns8_prepared_execution& e) {
  e.completion.has_exact_epilogue = e.d != nullptr;
  e.completion.alpha = e.alpha; e.completion.beta = e.beta;
  e.completion.d_instance = e.d_instance; e.completion.d_generation = e.d_generation;
}
static rns8_status prepared_range(const rns8_prepared_execution& e,
    boost::multiprecision::cpp_int& bound, rns8_exact_matrix_ranges& ranges) {
  if(uses_finite_storage(e.plan->desc.semantics) ||
     e.plan->desc.semantics==RNS8_WRAP_U64_MOD_2_64) {
    bound=0;ranges={};return RNS8_SUCCESS; // ring semantics need no integer range lift
  }
  return e.d ? rns8::detail::exact_axpby_range(*e.plan, *e.a, *e.b, *e.d, e.alpha, e.beta, bound)
             : rns8::detail::exact_gemm_range(*e.plan, *e.a, *e.b, bound, &ranges);
}
static rns8_status enqueue_prepared(const rns8_prepared_execution& e, void* stream) {
  if(e.plan->desc.semantics==RNS8_WRAP_U64_MOD_2_64)
    return rns8::detail::wrap64_hip_enqueue_byte_limbs(
        e.a->hip_byte_limbs,e.b->hip_byte_limbs,e.c->hip_byte_limbs,
        e.plan->desc.m,e.plan->desc.n,e.plan->desc.k,stream);
  if(uses_finite_storage(e.plan->desc.semantics))
    return rns8::detail::hip_direct_enqueue_finite_no_publish(e.plan,e.a,e.b,e.c,stream);
  return e.d ? rns8::detail::hip_direct_enqueue_axpby_no_publish(
                   e.plan, e.a, e.b, e.d, e.c, e.alpha, e.beta, stream)
             : rns8::detail::hip_direct_enqueue_rns_no_publish(e.plan, e.a, e.b, e.c, stream);
}
static void release_prepared_dependency(rns8_prepared_execution& e) {
  for(uint32_t i=0;i<e.producer_count;++i) --e.producers[i].producer->dependent_references;
  e.producers={};e.producer_count=0;
}
rns8_status rns8_bind_prepared_dependencies(rns8_prepared_execution* consumer,
    rns8_prepared_execution* const* producers,const rns8_completion_token* tokens,uint32_t count) {
  return guard_api([&]() -> rns8_status {
    if(!consumer || count>3 || (count && (!producers || !tokens))) return RNS8_INVALID_ARGUMENT;
    if(consumer->running || consumer->group_references) return RNS8_BUSY;
    std::array<rns8_prepared_execution::Dependency,3> bindings{};
    for(uint32_t i=0;i<count;++i) {
      auto* producer=producers[i];const auto token=tokens[i];
      if(!producer || producer==consumer || producer->ctx!=consumer->ctx || token.prepared_id!=producer->id ||
         !token.epoch || token.epoch!=producer->epoch ||
         (producer->c!=consumer->a && producer->c!=consumer->b && producer->c!=consumer->d)) return RNS8_INVALID_ARGUMENT;
      for(uint32_t j=0;j<i;++j)
        if(producers[j]==producer || producers[j]->c==producer->c) return RNS8_INVALID_ARGUMENT;
      // Walk the complete dependency DAG, including branches not on a first-
      // producer chain. All allocation/validation precedes reference mutation.
      std::vector<rns8_prepared_execution*> pending{producer};
      std::unordered_set<rns8_prepared_execution*> visited;
      while(!pending.empty()) {
        auto* node=pending.back();pending.pop_back();
        if(node==consumer) return RNS8_INVALID_ARGUMENT;
        if(!visited.insert(node).second) continue;
        for(uint32_t j=0;j<node->producer_count;++j) pending.push_back(node->producers[j].producer);
      }
      if(producer->dependent_references==UINT64_MAX) return RNS8_RANGE_ERROR;
      bindings[i]={producer,token};
    }
    for(uint32_t i=0;i<count;++i) ++bindings[i].producer->dependent_references;
    release_prepared_dependency(*consumer);consumer->producers=bindings;consumer->producer_count=count;
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_bind_prepared_dependency(rns8_prepared_execution* consumer,
    rns8_prepared_execution* producer,rns8_completion_token token) {
  if(!producer) {
    if(token.prepared_id || token.epoch) return RNS8_INVALID_ARGUMENT;
    return rns8_bind_prepared_dependencies(consumer,nullptr,nullptr,0);
  }
  return rns8_bind_prepared_dependencies(consumer,&producer,&token,1);
}
static rns8_status resolve_prepared_dependency(rns8_prepared_execution& e) {
  if(!e.producer_count) return RNS8_SUCCESS;
  for(uint32_t i=0;i<e.producer_count;++i) {
    const auto binding=e.producers[i];auto& producer=*binding.producer;
    rns8_completion_info completion{};
    completion.struct_size=sizeof(completion);completion.abi_version=RNS8_ABI_VERSION;
    const auto status=producer.running && producer.hip_submission_prepared
        ?rns8_query_prepared_hip(&producer,binding.token,&completion)
        :rns8_get_prepared_completion(&producer,binding.token,&completion);
    if(status!=RNS8_SUCCESS) return status;
    if(completion.operation_status!=RNS8_SUCCESS) return completion.operation_status;
    if(!completion.completed || !completion.output_is_current) return RNS8_INVALID_ARGUMENT;
  }
  const auto unchanged=[&](const rns8_matrix* matrix,uint64_t instance,uint64_t generation) {
    for(uint32_t i=0;i<e.producer_count;++i) if(matrix==e.producers[i].producer->c) return true;
    return matrix->matrix_instance_id==instance && matrix->content_generation==generation;
  };
  if(!unchanged(e.a,e.a_instance,e.a_generation) || !unchanged(e.b,e.b_instance,e.b_generation) ||
     (e.d && !unchanged(e.d,e.d_instance,e.d_generation))) return RNS8_INVALID_ARGUMENT;
  const auto valid=validate_prepared(e);if(valid!=RNS8_SUCCESS) return valid;
  bind_inputs(e);return RNS8_SUCCESS;
}
rns8_status rns8_refresh_prepared_inputs(rns8_prepared_execution* e) {
  return guard_api([&]() -> rns8_status {
    if (!e) return RNS8_INVALID_ARGUMENT;
    if (e->running || e->producer_count) return RNS8_BUSY;
    const auto status = validate_prepared(*e);
    if (status != RNS8_SUCCESS) return status;
    bind_inputs(*e);
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_execute_prepared(rns8_prepared_execution* e, rns8_completion_token* out_token) {
  return guard_api([&]() -> rns8_status {
    if (!e || !out_token) return RNS8_INVALID_ARGUMENT;
    if (e->running || e->dependent_references) return RNS8_BUSY;
    const auto dependency=resolve_prepared_dependency(*e);
    if (dependency!=RNS8_SUCCESS) return dependency;
    if (e->epoch == UINT64_MAX) return RNS8_RANGE_ERROR;
    const auto status = validate_prepared(*e);
    if (status != RNS8_SUCCESS) return status;
    if (!inputs_match(*e)) return RNS8_INVALID_ARGUMENT;
    e->async_failure = {};
    e->cancel_requested = false;
    e->running = true;
    ++e->epoch;
    rns8_status operation = RNS8_INTERNAL_ERROR;
    if (e->d) operation = rns8_gemm_exact_axpby(e->ctx, e->plan, e->a, e->b, e->d,
                                               e->c, e->alpha, e->beta, e->workspace);
    else switch (e->plan->desc.semantics) {
      case RNS8_FINITE_RING_U8:
      case RNS8_FINITE_FIELD_U8:
        operation = rns8_gemm_finite_u8(e->ctx, e->plan, static_cast<uint16_t>(e->plan->desc.finite_modulus),
                                      e->a, e->b, e->c, e->workspace);
        break;
      case RNS8_WRAP_U64_MOD_2_64:
        operation = rns8_gemm_wrap_u64(e->ctx, e->plan, e->a, e->b, e->c, e->workspace);
        break;
      default:
        operation = rns8_gemm_rns(e->ctx, e->plan, e->a, e->b, e->c, e->workspace);
    }
    e->completion = {};
    e->completion.operation_status = operation;
    e->completion.a_instance = e->a_instance; e->completion.a_generation = e->a_generation;
    e->completion.b_instance = e->b_instance; e->completion.b_generation = e->b_generation;
    e->completion.c_instance = e->c->matrix_instance_id;
    e->completion.c_generation = e->c->content_generation;
    e->completion.completed = 1;
    record_epilogue_completion(*e);
    e->running = false;
    *out_token = {e->id, e->epoch};
    release_prepared_dependency(*e);
    return operation;
  });
}
rns8_status rns8_get_prepared_completion(
    const rns8_prepared_execution* e, rns8_completion_token token, rns8_completion_info* out) {
  if (!e || !out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out)) ||
      token.prepared_id != e->id || !token.epoch || token.epoch != e->epoch) return RNS8_INVALID_ARGUMENT;
  if (e->running) return RNS8_BUSY;
  *out = e->completion;
  out->struct_size = sizeof(*out); out->abi_version = RNS8_ABI_VERSION;
  out->output_is_current = out->operation_status == RNS8_SUCCESS && !e->c->generation_exhausted &&
      e->c->matrix_instance_id == out->c_instance && e->c->content_generation == out->c_generation &&
      (e->c->host_residues_current || e->c->device_residues_current || e->c->host_byte_limbs_current ||
       e->c->device_byte_limbs_current || e->c->host_native_current || e->c->device_native_current);
  return RNS8_SUCCESS;
}
rns8_status rns8_destroy_prepared_execution(rns8_prepared_execution* e) {
  if (!e) return RNS8_SUCCESS;
  if (e->running || e->group_references || e->dependent_references) return RNS8_BUSY;
  release_prepared_dependency(*e);
  uint64_t* slots[7]{};
  reference_slots(*e, slots);
  for (auto slot : slots) if (slot) --*slot;
  delete e;
  return RNS8_SUCCESS;
}

#include "api_prepared_hip.inc"

rns8_status rns8_create_prepared_group(
    rns8_prepared_execution* const* executions, uint32_t count, rns8_prepared_group** out) {
  return guard_api([&]() -> rns8_status {
    if (!out) return RNS8_INVALID_ARGUMENT;
    *out = nullptr;
    if (!executions || !count || count > 65536) return RNS8_INVALID_ARGUMENT;
    auto group = std::make_unique<rns8_prepared_group>();
    group->executions.assign(executions, executions + count);
    std::unordered_set<const rns8_matrix*> writers, readers;
    writers.reserve(count); readers.reserve(static_cast<std::size_t>(count) * 3);
    for (auto* e : group->executions) {
      if (!e) return RNS8_INVALID_ARGUMENT;
      if (e->running || e->producer_count || e->dependent_references) return RNS8_BUSY;
      if (e->group_references == UINT64_MAX) return RNS8_RANGE_ERROR;
      auto status = validate_prepared(*e);
      if (status != RNS8_SUCCESS) return status;
      if (!inputs_match(*e)) return RNS8_INVALID_ARGUMENT;
      if (!writers.insert(e->c).second) return RNS8_INVALID_ARGUMENT;
      readers.insert(e->a); readers.insert(e->b); if (e->d) readers.insert(e->d);
    }
    for (auto* writer : writers) if (readers.count(writer)) return RNS8_INVALID_ARGUMENT;
    for (auto* e : group->executions) ++e->group_references;
    *out = group.release();
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_refresh_prepared_group_inputs(rns8_prepared_group* group) {
  return guard_api([&]() -> rns8_status {
    if (!group) return RNS8_INVALID_ARGUMENT;
    if (group->running) return RNS8_BUSY;
    for (auto* e : group->executions) {
      if (e->running || e->producer_count || e->dependent_references) return RNS8_BUSY;
      auto status = validate_prepared(*e);
      if (status != RNS8_SUCCESS) return status;
    }
    for (auto* e : group->executions) bind_inputs(*e);
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_execute_prepared_group(
    rns8_prepared_group* group, rns8_completion_token* tokens, uint32_t token_capacity, uint32_t* out_executed) {
  return guard_api([&]() -> rns8_status {
    if (!group || !tokens || !out_executed || token_capacity < group->executions.size())
      return RNS8_INVALID_ARGUMENT;
    *out_executed = 0;
    if (group->running) return RNS8_BUSY;
    for (auto* e : group->executions) {
      if (e->running || e->producer_count || e->dependent_references) return RNS8_BUSY;
      if (e->epoch == UINT64_MAX) return RNS8_RANGE_ERROR;
      const auto status = validate_prepared(*e);
      if (status != RNS8_SUCCESS) return status;
      if (!inputs_match(*e)) return RNS8_INVALID_ARGUMENT;
    }
    group->running = true;
    for (auto* e : group->executions) {
      const auto prior_epoch = e->epoch;
      const auto status = rns8_execute_prepared(e, tokens + *out_executed);
      if (e->epoch != prior_epoch) ++*out_executed;
      if (status != RNS8_SUCCESS) { group->running = false; return status; }
    }
    group->running = false;
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_prepared_group(rns8_prepared_group* group) {
  if (!group) return RNS8_SUCCESS;
  if (group->running) return RNS8_BUSY;
  for (auto* e : group->executions) if (e->running) return RNS8_BUSY;
  for (auto* e : group->executions) --e->group_references;
  delete group;
  return RNS8_SUCCESS;
}
