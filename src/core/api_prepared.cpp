#include "core/api_internal.hpp"
#include <atomic>
#include <memory>

using namespace rns8::detail::api;

struct rns8_prepared_execution {
  rns8_context* ctx = nullptr;
  const rns8_plan* plan = nullptr;
  const rns8_matrix* a = nullptr;
  const rns8_matrix* b = nullptr;
  rns8_matrix* c = nullptr;
  rns8_workspace* workspace = nullptr;
  uint64_t id = 0, epoch = 0;
  uint64_t a_instance = 0, a_generation = 0, b_instance = 0, b_generation = 0;
  rns8_completion_info completion{};
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
  switch (e.plan->desc.semantics) {
    case RNS8_FINITE_RING_U8:
    case RNS8_FINITE_FIELD_U8:
      return validate_finite_gemm_operands(*e.ctx, *e.plan, e.plan->desc.finite_modulus, *e.a, *e.b, *e.c);
    case RNS8_WRAP_U64_MOD_2_64:
      return validate_wrap_gemm_operands(*e.ctx, *e.plan, *e.a, *e.b, *e.c);
    default:
      return validate_rns_gemm_operands(*e.ctx, *e.plan, *e.a, *e.b, *e.c);
  }
}
void bind_inputs(rns8_prepared_execution& e) {
  e.a_instance = e.a->matrix_instance_id; e.a_generation = e.a->content_generation;
  e.b_instance = e.b->matrix_instance_id; e.b_generation = e.b->content_generation;
}
// A==B is legal; hold that resource once. No allocation follows retention.
void reference_slots(const rns8_prepared_execution& e, uint64_t* (&slots)[6]) {
  slots[0] = &e.ctx->prepared_references; slots[1] = &e.plan->prepared_references;
  slots[2] = &e.a->prepared_references;
  slots[3] = e.a == e.b ? nullptr : &e.b->prepared_references;
  slots[4] = &e.c->prepared_references; slots[5] = &e.workspace->prepared_references;
}
}

rns8_status rns8_prepare_gemm(
    rns8_context* ctx, const rns8_plan* plan, const rns8_matrix* a, const rns8_matrix* b,
    rns8_matrix* c, rns8_workspace* workspace, rns8_prepared_execution** out) {
  return guard_api([&]() -> rns8_status {
    if (!out) return RNS8_INVALID_ARGUMENT;
    *out = nullptr;
    if (!ctx || !plan || !a || !b || !c || !workspace) return RNS8_INVALID_ARGUMENT;
    auto e = std::make_unique<rns8_prepared_execution>();
    e->ctx = ctx; e->plan = plan; e->a = a; e->b = b; e->c = c; e->workspace = workspace;
    const auto status = validate_prepared(*e);
    if (status != RNS8_SUCCESS) return status;
    uint64_t* slots[6]{};
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
rns8_status rns8_refresh_prepared_inputs(rns8_prepared_execution* e) {
  return guard_api([&]() -> rns8_status {
    if (!e) return RNS8_INVALID_ARGUMENT;
    if (e->running) return RNS8_BUSY;
    const auto status = validate_prepared(*e);
    if (status != RNS8_SUCCESS) return status;
    bind_inputs(*e);
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_execute_prepared(rns8_prepared_execution* e, rns8_completion_token* out_token) {
  return guard_api([&]() -> rns8_status {
    if (!e || !out_token) return RNS8_INVALID_ARGUMENT;
    if (e->running) return RNS8_BUSY;
    if (e->epoch == UINT64_MAX) return RNS8_RANGE_ERROR;
    const auto status = validate_prepared(*e);
    if (status != RNS8_SUCCESS) return status;
    if (e->a_instance != e->a->matrix_instance_id || e->a_generation != e->a->content_generation ||
        e->b_instance != e->b->matrix_instance_id || e->b_generation != e->b->content_generation)
      return RNS8_INVALID_ARGUMENT;
    e->running = true;
    ++e->epoch;
    rns8_status operation = RNS8_INTERNAL_ERROR;
    switch (e->plan->desc.semantics) {
      case RNS8_FINITE_RING_U8:
      case RNS8_FINITE_FIELD_U8:
        operation = rns8_gemm_finite_u8(e->ctx, e->plan, e->plan->desc.finite_modulus, e->a, e->b, e->c, e->workspace);
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
    e->running = false;
    *out_token = {e->id, e->epoch};
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
  if (e->running) return RNS8_BUSY;
  uint64_t* slots[6]{};
  reference_slots(*e, slots);
  for (auto slot : slots) if (slot) --*slot;
  delete e;
  return RNS8_SUCCESS;
}
