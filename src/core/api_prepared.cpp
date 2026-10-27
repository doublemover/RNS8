#include "backend_common/locality_schedule.hpp"
#include "backend_amdgpu_builtins/amdgpu_builtins_backend.hpp"
#include "core/api_internal.hpp"
#include <atomic>
#include <array>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include "core/hip_resources.hpp"
#include "core/encoded_b_internal.hpp"
#include "core/prepared_transfer.hpp"
#include "backend_hip_direct/native_wmma_prepared.hpp"

using namespace rns8::detail::api;

struct rns8_prepared_execution {
  std::unique_ptr<PreparedTransfer> transfer;
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
  bool closing = false;
  bool native_integrity_enabled = false;
  rns8_native_integrity_report native_integrity_report{};
  rns8::detail::hip_direct_device_buffer integrity_flag;
  bool cancel_requested = false;
  uint64_t group_references = 0;
  uint64_t dependent_references = 0;
  uint64_t caller_graph_references = 0;
  bool pool_owned=false,pool_dispatch=false;
  struct Dependency {rns8_prepared_execution* producer=nullptr;rns8_completion_token token{};};
  std::array<Dependency,3> producers{};
  uint32_t producer_count=0;
  bool event_dependencies = false;
  uint32_t event_dependency_depth = 0;
  bool hip_submission_prepared = false;
  void* pending_stream = nullptr;
  void* hipblaslt_async = nullptr;
  boost::multiprecision::cpp_int pending_bound;
  rns8_exact_matrix_ranges pending_ranges;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8::detail::hip_unique_event completion_event;
  rns8::detail::hip_pinned_host_buffer integrity_host_flag;
  hipStream_t integrity_stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_executable = nullptr;
  const void* graph_a = nullptr;
  const void* graph_b = nullptr;
  const void* graph_c = nullptr;
  const void* graph_d = nullptr;
  ~rns8_prepared_execution() {
    if (!graph_executable && !graph && !completion_event.get() && !integrity_stream) return;
    rns8::detail::hip_scoped_device device(ctx ? ctx->device_id : 0);
    if (graph_executable) (void)hipGraphExecDestroy(graph_executable);
    if (graph) (void)hipGraphDestroy(graph);
    completion_event.reset();
    if(integrity_stream)(void)hipStreamDestroy(integrity_stream);
  }
#endif
};

struct rns8_prepared_group {
  std::vector<rns8_prepared_execution*> executions;
  bool running = false;
  struct NativeBinding {
    std::shared_ptr<const rns8::detail::arithmetic::Guard> arithmetic_guard;
    rns8::detail::NativeWmmaTask task{};
    uint64_t a_instance=0,a_generation=0,b_instance=0,b_generation=0;
  };
  std::unique_ptr<rns8::detail::NativeWmmaPreparedQueue> native_queue;
  std::vector<NativeBinding> native_values;
  std::vector<rns8_native_group_binding> native_bindings;
  rns8_native_group_desc native_desc{};
  rns8_locality_desc native_locality{};
  uint64_t native_peak_device_bytes=0;
  std::string native_certificate_sha256;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hipStream_t native_stream=nullptr;
  hipGraph_t native_graph=nullptr;
  hipGraphExec_t native_executable=nullptr;
#endif
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
  if(e.closing)return RNS8_INVALID_ARGUMENT;
  if(e.pool_owned && !e.pool_dispatch)return RNS8_BUSY;
  if(e.transfer){boost::multiprecision::cpp_int bound;return e.transfer->validate(nullptr,0,bound);}
  auto status = validate_plan_context_workspace(*e.ctx, *e.plan, *e.workspace);
  if (status != RNS8_SUCCESS) return status;
  if (e.a->generation_exhausted || e.b->generation_exhausted || e.c->generation_exhausted)
    return RNS8_RANGE_ERROR;
  if (e.d) {
    if (e.d->generation_exhausted) return RNS8_RANGE_ERROR;
    status = validate_exact_axpby_operands(*e.ctx, *e.plan, *e.a, *e.b, *e.d, *e.c, e.alpha, e.beta);
    if (status != RNS8_SUCCESS) return status;
    if (e.plan->desc.flags & RNS8_PLAN_EXACT_VALUE_BOUND) {
      boost::multiprecision::cpp_int bound;
      return rns8::detail::exact_axpby_range(*e.plan, *e.a, *e.b, *e.d, e.alpha, e.beta, bound);
    }
    return RNS8_SUCCESS;
  }
  switch (e.plan->desc.semantics) {
    case RNS8_FINITE_RING_U8:
    case RNS8_FINITE_FIELD_U8:
      return validate_finite_gemm_operands(*e.ctx, *e.plan, static_cast<uint16_t>(e.plan->desc.finite_modulus), *e.a, *e.b, *e.c);
    case RNS8_WRAP_U64_MOD_2_64:
      return validate_wrap_gemm_operands(*e.ctx, *e.plan, *e.a, *e.b, *e.c);
    default:
      status = validate_rns_gemm_operands(*e.ctx, *e.plan, *e.a, *e.b, *e.c);
      if (status != RNS8_SUCCESS) return status;
      if (e.plan->desc.flags & RNS8_PLAN_EXACT_VALUE_BOUND) {
        boost::multiprecision::cpp_int bound;
        return rns8::detail::exact_gemm_range(*e.plan, *e.a, *e.b, bound);
      }
      return RNS8_SUCCESS;
  }
}
void bind_inputs(rns8_prepared_execution& e) {
  e.a_instance = e.a->matrix_instance_id; e.a_generation = e.a->content_generation;
  e.b_instance = e.b->matrix_instance_id; e.b_generation = e.b->content_generation;
  if (e.d) { e.d_instance = e.d->matrix_instance_id; e.d_generation = e.d->content_generation; }
}
bool inputs_match(const rns8_prepared_execution& e) {
  if(e.transfer && !e.transfer->reads_matrix())return e.c->matrix_instance_id==e.a_instance;
  return e.a_instance == e.a->matrix_instance_id && e.a_generation == e.a->content_generation &&
         e.b_instance == e.b->matrix_instance_id && e.b_generation == e.b->content_generation &&
         (!e.d || (e.d_instance == e.d->matrix_instance_id && e.d_generation == e.d->content_generation));
}
rns8_status validate_group_external_spans(const std::vector<rns8_prepared_execution*>& executions) {
  for(size_t i=0;i<executions.size();++i){const auto* e=executions[i];if(!e->transfer)continue;
    const auto external=e->transfer->external_access();if(!external.pointer)continue;
    for(size_t j=0;j<executions.size();++j){if(i==j)continue;const auto* other=executions[j];
      if(other->transfer){const auto span=other->transfer->external_access();
        if(external.reservation_owner && external.reservation_owner==span.reservation_owner)return RNS8_INVALID_ARGUMENT;
        if(span.pointer && (external.writer || span.writer) && !rns8::detail::nonoverlapping_byte_ranges(external.pointer,external.bytes,span.pointer,span.bytes))return RNS8_INVALID_ARGUMENT;}
      const auto check=[&](const rns8_matrix* matrix,bool writer){if(!matrix || (!external.writer && !writer))return true;
        for(const auto& allocation:{std::pair<const void*,uint64_t>{matrix->hip_residues,matrix->hip_residue_bytes},
            {matrix->hip_byte_limbs,matrix->hip_byte_limb_bytes},{matrix->hip_native_i64,matrix->hip_native_i64_bytes},{matrix->hip_native_u64,matrix->hip_native_u64_bytes}})
          if(allocation.first && !rns8::detail::nonoverlapping_byte_ranges(external.pointer,external.bytes,allocation.first,allocation.second))return false;
        return true;};
      if(!check(other->a,false) || !check(other->b,false) || !check(other->d,false) || !check(other->c,!other->transfer || other->transfer->writes_matrix()))return RNS8_INVALID_ARGUMENT;
    }
  }
  return RNS8_SUCCESS;
}
// A==B is legal; hold that resource once. No allocation follows retention.
void reference_slots(const rns8_prepared_execution& e, uint64_t* (&slots)[7]) {
  slots[0] = &e.ctx->prepared_references; slots[1] = e.plan ? &e.plan->prepared_references : nullptr;
  slots[2] = &e.a->prepared_references;
  slots[3] = e.a == e.b ? nullptr : &e.b->prepared_references;
  slots[4] = e.c!=e.a && e.c!=e.b ? &e.c->prepared_references : nullptr; slots[5] = e.workspace ? &e.workspace->prepared_references : nullptr;
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
    const auto provider=plan_operation_preflight(*ctx,*plan,d?RNS8_OP_EXACT_EPILOGUE:RNS8_OP_PREPARED_GROUP,RNS8_EXEC_SYNC);
    if(provider!=RNS8_SUCCESS)return provider;
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
  if(e.transfer){ranges={};return e.transfer->validate(nullptr,0,bound);}
  if(uses_finite_storage(e.plan->desc.semantics) ||
     e.plan->desc.semantics==RNS8_WRAP_U64_MOD_2_64) {
    bound=0;ranges={};return RNS8_SUCCESS; // ring semantics need no integer range lift
  }
  return e.d ? rns8::detail::exact_axpby_range(*e.plan, *e.a, *e.b, *e.d, e.alpha, e.beta, bound)
             : rns8::detail::exact_gemm_range(*e.plan, *e.a, *e.b, bound, &ranges);
}
static rns8_status enqueue_prepared(const rns8_prepared_execution& e, void* stream) {
  if(e.transfer)return e.transfer->enqueue(stream);
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  if(e.plan->backend==RNS8_BACKEND_HIPBLASLT)
    return rns8::detail::hipblaslt_enqueue_async(e.hipblaslt_async,e.a->hip_residues,e.b->hip_residues,e.c->hip_residues,
        e.a->desc.cols,e.b->desc.cols,e.c->desc.cols,stream);
#endif
  if(e.plan->backend==RNS8_BACKEND_CK || e.plan->backend==RNS8_BACKEND_ROCWMMA) {
    const auto fn=e.plan->backend==RNS8_BACKEND_CK?rns8::detail::ck_enqueue_dense:rns8::detail::rocwmma_enqueue_dense;
    return fn(e.ctx->device_id,e.a->hip_residues,e.b->hip_residues,e.c->hip_residues,
        e.workspace->accelerator_workspace,e.workspace->accelerator_workspace_bytes,
        e.plan->desc.m,e.plan->desc.n,e.plan->desc.k,e.a->desc.cols,e.b->desc.cols,e.c->desc.cols,
        uses_finite_storage(e.plan->desc.semantics)?e.plan->desc.finite_modulus:e.plan->prefix,uses_finite_storage(e.plan->desc.semantics),stream);
  }
  if(e.plan->backend==RNS8_BACKEND_AMDGPU_BUILTINS)
    return rns8::detail::amdgpu_builtins_enqueue_dense(e.ctx->device_id,e.a->hip_residues,e.b->hip_residues,e.c->hip_residues,
        e.plan->desc.m,e.plan->desc.n,e.plan->desc.k,e.a->desc.cols,e.b->desc.cols,e.c->desc.cols,
        uses_finite_storage(e.plan->desc.semantics)?e.plan->desc.finite_modulus:e.plan->prefix,uses_finite_storage(e.plan->desc.semantics),stream);
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
#include "api_prepared_integrity.inc"
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
      if(!producer || (producer->transfer && !producer->transfer->produces_resident()) || producer==consumer || producer->ctx!=consumer->ctx || token.prepared_id!=producer->id ||
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
    if (e->running || e->producer_count || (e->pool_owned && !e->pool_dispatch)) return RNS8_BUSY;
    if(e->closing)return RNS8_INVALID_ARGUMENT;
    const auto status = e->transfer?e->transfer->refresh():validate_prepared(*e);
    if (status != RNS8_SUCCESS) return status;
    if(e->transfer)e->transfer->commit_refresh();
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
    if(e->native_integrity_enabled || e->transfer) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      *out_token={};
      const auto submitted=rns8_submit_prepared_hip(e,e->transfer?e->transfer->sync_stream():e->integrity_stream,out_token);
      if(!out_token->epoch)return submitted;
      rns8_completion_info completion{};completion.struct_size=sizeof(completion);completion.abi_version=RNS8_ABI_VERSION;
      const auto waited=rns8_wait_prepared_hip(e,*out_token,&completion);
      return waited==RNS8_SUCCESS?completion.operation_status:waited;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
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
  out->output_is_current = out->operation_status == RNS8_SUCCESS && (!e->transfer || e->transfer->output_current()) && !e->c->generation_exhausted &&
      e->c->matrix_instance_id == out->c_instance && e->c->content_generation == out->c_generation &&
      (e->c->host_residues_current || e->c->device_residues_current || e->c->host_byte_limbs_current ||
       e->c->device_byte_limbs_current || e->c->host_native_current || e->c->device_native_current);
  return RNS8_SUCCESS;
}
rns8_status rns8_destroy_prepared_execution(rns8_prepared_execution* e) {
  if (!e) return RNS8_SUCCESS;
  if (e->running || e->group_references || e->dependent_references || e->caller_graph_references) return RNS8_BUSY;
  e->closing=true;
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  if(e->hipblaslt_async){const auto closed=rns8::detail::hipblaslt_destroy_async(e->hipblaslt_async);
    if(closed!=RNS8_SUCCESS)return closed;e->hipblaslt_async=nullptr;}
#endif
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(e->graph_executable || e->graph || e->completion_event.get()) {
    rns8::detail::hip_scoped_device selected(e->ctx->device_id);if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    if(e->graph_executable){if(hipGraphExecDestroy(e->graph_executable)!=hipSuccess)return RNS8_BACKEND_FAILURE;e->graph_executable=nullptr;}
    if(e->graph){if(hipGraphDestroy(e->graph)!=hipSuccess)return RNS8_BACKEND_FAILURE;e->graph=nullptr;}
    if(e->completion_event.get()){if(hipEventDestroy(e->completion_event.get())!=hipSuccess)return RNS8_BACKEND_FAILURE;(void)e->completion_event.release();}
  }
#endif
  if(e->transfer){const auto closed=e->transfer->close();if(closed!=RNS8_SUCCESS)return closed;}
  const auto closed=close_prepared_integrity(*e);
  if(closed!=RNS8_SUCCESS)return closed;
  release_prepared_dependency(*e);
  uint64_t* slots[7]{};
  reference_slots(*e, slots);
  for (auto slot : slots) if (slot) --*slot;
  delete e;
  return RNS8_SUCCESS;
}

#include "api_prepared_hip.inc"
#include "api_prepared_caller_graph.inc"
#include "api_prepared_native_group.inc"
#include "api_prepared_pool.inc"

rns8_status rns8_create_prepared_group(
    rns8_prepared_execution* const* executions, uint32_t count, rns8_prepared_group** out) {
  return guard_api([&]() -> rns8_status {
    if (!out) return RNS8_INVALID_ARGUMENT;
    *out = nullptr;
    if (!executions || !count || count > 65536) return RNS8_INVALID_ARGUMENT;
    auto group = std::make_unique<rns8_prepared_group>();
    group->executions.assign(executions, executions + count);
    std::unordered_map<const rns8_matrix*,const rns8_prepared_execution*> writers;
    std::vector<std::pair<const rns8_matrix*,const rns8_prepared_execution*>> readers;
    writers.reserve(count); readers.reserve(static_cast<std::size_t>(count) * 3);
    for (auto* e : group->executions) {
      if (!e) return RNS8_INVALID_ARGUMENT;
      if (e->running || e->producer_count || e->dependent_references) return RNS8_BUSY;
      if (e->group_references == UINT64_MAX) return RNS8_RANGE_ERROR;
      auto status = validate_prepared(*e);
      if (status != RNS8_SUCCESS) return status;
      if (!inputs_match(*e)) return RNS8_INVALID_ARGUMENT;
      if(!e->transfer || e->transfer->writes_matrix())if(!writers.emplace(e->c,e).second)return RNS8_INVALID_ARGUMENT;
      if(!e->transfer || e->transfer->reads_matrix()){readers.emplace_back(e->a,e);readers.emplace_back(e->b,e);if(e->d)readers.emplace_back(e->d,e);}
    }
    const auto external=validate_group_external_spans(group->executions);if(external!=RNS8_SUCCESS)return external;
    for(const auto& read:readers){const auto writer=writers.find(read.first);if(writer!=writers.end() && writer->second!=read.second)return RNS8_INVALID_ARGUMENT;}
    for (auto* e : group->executions) ++e->group_references;
    *out = group.release();
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_refresh_prepared_group_inputs(rns8_prepared_group* group) {
  return guard_api([&]() -> rns8_status {
    if (!group) return RNS8_INVALID_ARGUMENT;
    if (group->running) return RNS8_BUSY;
    if(group->native_queue) return group->native_locality.struct_size?rns8_prepare_native_group_locality(group,&group->native_desc,
        group->native_bindings.data(),static_cast<uint32_t>(group->native_bindings.size()),&group->native_locality):
        rns8_prepare_native_group(group,&group->native_desc,group->native_bindings.data(),static_cast<uint32_t>(group->native_bindings.size()));
    for (auto* e : group->executions) {
      if (e->running || e->producer_count || e->dependent_references) return RNS8_BUSY;
      if(e->closing)return RNS8_INVALID_ARGUMENT;
      auto status = e->transfer?e->transfer->refresh():validate_prepared(*e);
      if (status != RNS8_SUCCESS) return status;
    }
    for (auto* e : group->executions){if(e->transfer)e->transfer->commit_refresh();bind_inputs(*e);}
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_execute_prepared_group(
    rns8_prepared_group* group, rns8_completion_token* tokens, uint32_t token_capacity, uint32_t* out_executed) {
  return guard_api([&]() -> rns8_status {
    if (!group || !tokens || !out_executed || token_capacity < group->executions.size())
      return RNS8_INVALID_ARGUMENT;
    *out_executed = 0;
    if(group->native_queue) return execute_native_group_sync(group,tokens,token_capacity,out_executed);
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
  const auto close=close_native_group(*group);if(close!=RNS8_SUCCESS)return close;
  for(const auto& binding:group->native_bindings) if(binding.encoded_b)--binding.encoded_b->prepared_references;
  for (auto* e : group->executions) --e->group_references;
  delete group;
  return RNS8_SUCCESS;
}

// Factory used by exact-lift and native/host transfer owners. Resource preparation
// is already complete; ownership/reference publication is one nonthrowing step.
rns8_status rns8::detail::api::create_transfer_prepared(std::unique_ptr<PreparedTransfer> transfer,
    rns8_prepared_execution** out) {
  return guard_api([&]() -> rns8_status {
    if(!transfer || !out || !transfer->ctx || !transfer->matrix)return RNS8_INVALID_ARGUMENT;
    *out=nullptr;
    auto e=std::make_unique<rns8_prepared_execution>();e->ctx=transfer->ctx;
    e->a=transfer->input_a();e->b=transfer->input_b();e->c=transfer->matrix;e->transfer=std::move(transfer);
    const auto valid=validate_prepared(*e);if(valid!=RNS8_SUCCESS)return valid;
    uint64_t* slots[7]{};reference_slots(*e,slots);
    for(const auto* slot:slots)if(slot && *slot==UINT64_MAX)return RNS8_RANGE_ERROR;
    e->id=next_prepared_id();if(!e->id)return RNS8_RANGE_ERROR;bind_inputs(*e);
    const auto hip=rns8_prepare_hip_submission(e.get());if(hip!=RNS8_SUCCESS)return hip;
    for(auto* slot:slots)if(slot)++*slot;
    *out=e.release();return RNS8_SUCCESS;
  });
}

rns8_status rns8_get_prepared_transfer_info(const rns8_prepared_execution* e,rns8_prepared_transfer_info* info) {
  if(!e || !e->transfer || !info || !rns8::detail::valid_abi(info->struct_size,info->abi_version,sizeof(*info)))return RNS8_INVALID_ARGUMENT;
  if(e->running)return RNS8_BUSY;
  *info={};info->struct_size=sizeof(*info);info->abi_version=RNS8_ABI_VERSION;info->prepared_id=e->id;info->epoch=e->epoch;
  e->transfer->describe(*info);return RNS8_SUCCESS;
}
