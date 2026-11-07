#include "core/hip_resources.hpp"
#include "backend_common/checked_arithmetic.hpp"
#include <memory>
#include <type_traits>

struct rns8_block_sparse_operand {
  mutable uint64_t in_flight_readers = 0;
  rns8_context* ctx = nullptr;
  const rns8_matrix* source = nullptr;
  rns8::detail::BlockSparseOperand layout;
  mutable rns8::detail::hip_direct_device_buffer payload;
  mutable rns8::detail::hip_direct_device_buffer directory_owner;
  rns8::detail::BlockSparseDeviceDirectory directory;
  uint64_t preparation_peak_bound = 0, preparation_budget = 0;
  std::vector<int64_t> download_i64;
  std::vector<uint64_t> download_u64;
  std::vector<int8_t> packed;
  bool context_reference = true, cleanup_started = false;
  rns8_block_sparse_operand* failed_cleanup_next = nullptr;
};

namespace {
struct UnpublishedBlockSparseDeleter {
  void operator()(rns8_block_sparse_operand* owner) const noexcept {
    if (!owner || rns8_destroy_block_sparse_operand(owner) == RNS8_SUCCESS)
      return;
    auto* context = owner->ctx;
    context->hip_failure_poisoned = true;
    --context->prepared_references;
    owner->context_reference = false;
    owner->failed_cleanup_next = context->failed_block_sparse_cleanup;
    context->failed_block_sparse_cleanup = owner;
  }
};
} // namespace

rns8_status rns8_prepare_block_sparse_operand(rns8_context* ctx, const rns8_matrix* source,
                                              rns8_block_sparse_role role,
                                              uint64_t max_preparation_bytes,
                                              rns8_block_sparse_operand** out) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if (!ctx || !source || !out || (role != RNS8_BLOCK_SPARSE_LHS && role != RNS8_BLOCK_SPARSE_RHS))
      return RNS8_INVALID_ARGUMENT;
    if (ctx->hip_failure_poisoned)
      return RNS8_BACKEND_FAILURE;
    if (ctx->backend != RNS8_BACKEND_HIP_DIRECT || source->backend != ctx->backend ||
        !exact_wide_semantics(source->desc.semantics))
      return RNS8_UNSUPPORTED_BACKEND;
    if (source->in_flight_writer)
      return RNS8_BUSY;
    if (source->prepared_references == UINT64_MAX || ctx->prepared_references == UINT64_MAX)
      return RNS8_RANGE_ERROR;
    if (source->hip_device_id != ctx->device_id || source->generation_exhausted ||
        !(source->desc.flags & RNS8_MATRIX_RETAIN_NATIVE_INPUT) || !source->device_native_current ||
        !rns_residue_state_current_for_backend(*source, ctx->backend) ||
        source->exact_range_prefix < source->prefix)
      return RNS8_INVALID_ARGUMENT;
    AddressDomain address;
    if (!checked_host_plane_address(source->desc.rows, source->desc.cols, source->desc.cols, 1, 8,
                                    address))
      return RNS8_RANGE_ERROR;
    const bool u = source->desc.semantics == RNS8_EXACT_WIDE_UNSIGNED;
    const void* native = u ? source->hip_native_u64 : source->hip_native_i64;
    const auto native_bytes = u ? source->hip_native_u64_bytes : source->hip_native_i64_bytes;
    if (!native || native_bytes != address.byte_extent)
      return RNS8_INVALID_ARGUMENT;
    constexpr uint64_t fixed = sizeof(rns8_block_sparse_operand) +
                               3 * sizeof(hip_deferred_buffer_cleanup) + sizeof(uint32_t);
    uint64_t base = 0;
    if (!checked::add(address.byte_extent, fixed, base))
      return RNS8_RANGE_ERROR;
    if (base > max_preparation_bytes)
      return RNS8_WORKSPACE_TOO_SMALL;
    const uint64_t pack_budget = (max_preparation_bytes - base) / 2;
    std::unique_ptr<rns8_block_sparse_operand, UnpublishedBlockSparseDeleter> result(
        new rns8_block_sparse_operand);
    result->ctx = ctx;
    result->source = source;
    result->preparation_budget = max_preparation_bytes;
    ++ctx->prepared_references;
    ++source->prepared_references;
    size_t token = 0;
    auto status = begin_synchronous_read_lease(*ctx, *source, token);
    if (status != RNS8_SUCCESS)
      return status;
    SynchronousLease lease(*ctx, token);
    status = add_synchronous_owner_hold(*ctx, token, result->in_flight_readers);
    if (status == RNS8_SUCCESS)
      status = lease.hold_buffer(result->payload);
    if (status == RNS8_SUCCESS)
      status = lease.hold_buffer(result->directory_owner);
    if (status != RNS8_SUCCESS)
      return lease.finish(status);
    const auto packed_role =
        role == RNS8_BLOCK_SPARSE_LHS ? BlockSparseRole::lhs : BlockSparseRole::rhs;
    const auto prepare = [&](auto tag) -> rns8_status {
      using T = decltype(tag);
      auto& host = [&]() -> std::vector<T>& {
        if constexpr (std::is_signed_v<T>)
          return result->download_i64;
        else
          return result->download_u64;
      }();
      host.resize(static_cast<size_t>(address.byte_extent / 8));
      auto status = synchronous_dispatch_checkpoint(*ctx, "block_sparse_download");
      if (status != RNS8_SUCCESS)
        return status;
      status = hip_direct_copy_device_to_host(ctx->device_id, host.data(), native,
                                              static_cast<size_t>(address.byte_extent),
                                              HostExportDestination::retained_owner);
      if (status != RNS8_SUCCESS)
        return status;
      if constexpr (std::is_signed_v<T>)
        return pack_block_sparse_i64(host.data(), source->desc.rows, source->desc.cols,
                                     source->desc.cols, source->prefix, packed_role,
                                     source->matrix_instance_id, source->content_generation,
                                     pack_budget, result->layout, result->packed);
      else
        return pack_block_sparse_u64(host.data(), source->desc.rows, source->desc.cols,
                                     source->desc.cols, source->prefix, packed_role,
                                     source->matrix_instance_id, source->content_generation,
                                     pack_budget, result->layout, result->packed);
    };
    status = lease.finish([&]() -> rns8_status {
      if (!ctx->native_wmma_capability.provider) {
        auto status = synchronous_dispatch_checkpoint(*ctx, "block_sparse_capability");
        if (status != RNS8_SUCCESS)
          return status;
      }
      const auto capability_status = ensure_native_wmma_capability(*ctx);
      if (capability_status != RNS8_SUCCESS)
        return capability_status;
      auto status = u ? prepare(uint64_t{}) : prepare(int64_t{});
      if (status != RNS8_SUCCESS)
        return status;
      uint64_t directory_bytes = 0, host_directory = 0, host_words = 0;
      if (!block_sparse_directory_bytes(result->layout, directory_bytes) ||
          !checked::add(result->layout.offsets.capacity(), result->layout.k_blocks.capacity(),
                        host_words) ||
          !checked::multiply(host_words, sizeof(uint32_t), host_directory))
        return RNS8_RANGE_ERROR;
      uint64_t native_capacity = 0;
      if (!checked::add(result->download_i64.capacity(), result->download_u64.capacity(),
                        native_capacity) ||
          !checked::multiply(native_capacity, sizeof(uint64_t), native_capacity))
        return RNS8_RANGE_ERROR;
      auto peak = fixed;
      for (uint64_t bytes : {native_capacity, host_directory, uint64_t(result->packed.capacity()),
                             uint64_t(result->packed.size()), directory_bytes})
        if (!checked::add(peak, bytes, peak))
          return RNS8_RANGE_ERROR;
      if (peak > max_preparation_bytes)
        return RNS8_WORKSPACE_TOO_SMALL;
      result->preparation_peak_bound = peak;
      status =
          result->directory_owner.allocate(ctx->device_id, static_cast<size_t>(directory_bytes));
      if (status != RNS8_SUCCESS)
        return status;
      if (!make_block_sparse_device_directory(result->layout, ctx->device_id,
                                              result->directory_owner.get(), directory_bytes,
                                              result->directory))
        return RNS8_INTERNAL_ERROR;
      status = synchronous_dispatch_checkpoint(*ctx, "block_sparse_directory_upload");
      if (status != RNS8_SUCCESS)
        return status;
      auto* directory = static_cast<uint32_t*>(result->directory_owner.get());
      status =
          hip_direct_copy_host_to_device(ctx->device_id, directory, result->layout.offsets.data(),
                                         result->layout.offsets.size() * sizeof(uint32_t));
      if (status != RNS8_SUCCESS)
        return status;
      if (!result->layout.k_blocks.empty()) {
        status = hip_direct_copy_host_to_device(
            ctx->device_id, directory + result->layout.offsets.size(),
            result->layout.k_blocks.data(), result->layout.k_blocks.size() * sizeof(uint32_t));
        if (status != RNS8_SUCCESS)
          return status;
      }
      if (!result->packed.empty()) {
        status = result->payload.allocate(ctx->device_id, result->packed.size());
        if (status != RNS8_SUCCESS)
          return status;
        status = synchronous_dispatch_checkpoint(*ctx, "block_sparse_upload");
        if (status != RNS8_SUCCESS)
          return status;
        status = hip_direct_copy_host_to_device(ctx->device_id, result->payload.get(),
                                                result->packed.data(), result->packed.size());
        if (status != RNS8_SUCCESS)
          return status;
      }
      return synchronous_dispatch_checkpoint(*ctx, "block_sparse_prepare_publish");
    }());
    if (status != RNS8_SUCCESS)
      return status;
    // These ordinary sources are released only after the shared completion lease.
    std::vector<int64_t>{}.swap(result->download_i64);
    std::vector<uint64_t>{}.swap(result->download_u64);
    std::vector<int8_t>{}.swap(result->packed);
    *out = result.release();
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_block_sparse_operand(rns8_block_sparse_operand* operand) {
  if (!operand)
    return RNS8_SUCCESS;
  if (operand->in_flight_readers)
    return RNS8_BUSY;
  operand->cleanup_started = true;
  // Synchronous handle use/destruction requires external thread exclusion.
  for (auto* buffer : {&operand->payload, &operand->directory_owner}) {
    const auto status = buffer->reset();
    if (status != RNS8_SUCCESS)
      return status;
  }
  --operand->source->prepared_references;
  if (operand->context_reference)
    --operand->ctx->prepared_references;
  delete operand;
  return RNS8_SUCCESS;
}

rns8_status rns8_get_block_sparse_storage_info(const rns8_block_sparse_operand* operand,
                                               rns8_block_sparse_storage_info* out) {
  if (!operand || !out || !rns8::detail::minimum_abi_header(out))
    return RNS8_INVALID_ARGUMENT;
  if (operand->cleanup_started)
    return RNS8_BUSY;
  rns8_block_sparse_storage_info result{};
  result.struct_size = sizeof(result);
  result.abi_version = RNS8_ABI_VERSION;
  result.role = operand->layout.role == rns8::detail::BlockSparseRole::lhs ? RNS8_BLOCK_SPARSE_LHS
                                                                           : RNS8_BLOCK_SPARSE_RHS;
  result.prefix = operand->layout.prefix;
  result.device_id = operand->ctx->device_id;
  result.matrix_method_support =
      operand->ctx->native_wmma_capability.supported ? RNS8_SUCCESS : RNS8_UNSUPPORTED_BACKEND;
  result.source_instance = operand->layout.value_instance;
  result.source_generation = operand->layout.value_generation;
  result.host_directory_capacity_bytes =
      (uint64_t(operand->layout.offsets.capacity()) + operand->layout.k_blocks.capacity()) *
      sizeof(uint32_t);
  result.device_payload_bytes = operand->payload.bytes();
  result.device_directory_bytes = operand->directory_owner.bytes();
  result.retained_metadata_object_bytes =
      sizeof(*operand) +
      (1 + (operand->payload ? 1 : 0)) * sizeof(rns8::detail::hip_deferred_buffer_cleanup);
  result.preparation_peak_bound_bytes = operand->preparation_peak_bound;
  result.preparation_budget_bytes = operand->preparation_budget;
  *out = result;
  return RNS8_SUCCESS;
}

rns8_status rns8::detail::api::drain_failed_block_sparse_cleanup(rns8_context& ctx) noexcept {
  while (ctx.failed_block_sparse_cleanup) {
    auto* owner = ctx.failed_block_sparse_cleanup;
    auto* next = owner->failed_cleanup_next;
    const auto status = rns8_destroy_block_sparse_operand(owner);
    if (status != RNS8_SUCCESS)
      return status;
    ctx.failed_block_sparse_cleanup = next;
  }
  return RNS8_SUCCESS;
}
rns8_status rns8_gemm_block_sparse_exact_with_method(
    rns8_context* ctx, const rns8_plan* plan, const rns8_block_sparse_operand* a,
    const rns8_block_sparse_operand* b, rns8_matrix* c, rns8_block_sparse_method method,
    uint32_t scalar_intersection_limit, uint64_t max_scratch_bytes, rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if (method < RNS8_BLOCK_SPARSE_SCALAR || method > RNS8_BLOCK_SPARSE_HYBRID ||
        (method != RNS8_BLOCK_SPARSE_HYBRID && scalar_intersection_limit))
      return RNS8_INVALID_ARGUMENT;
    if (!ctx || !plan || !a || !b || !c || !workspace || a->ctx != ctx || b->ctx != ctx ||
        a->cleanup_started || b->cleanup_started || a->layout.role != BlockSparseRole::lhs ||
        b->layout.role != BlockSparseRole::rhs)
      return RNS8_INVALID_ARGUMENT;
    if (plan->backend != RNS8_BACKEND_HIP_DIRECT || !exact_wide_semantics(plan->desc.semantics))
      return RNS8_UNSUPPORTED_BACKEND;
    auto status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (status != RNS8_SUCCESS)
      return status;
    status = validate_rns_gemm_operands(*ctx, *plan, *a->source, *b->source, *c, true);
    if (status != RNS8_SUCCESS)
      return status;
    if (c->generation_exhausted)
      return RNS8_RANGE_ERROR;
    for (const auto* operand : {a, b}) {
      const auto& source = *operand->source;
      if (source.generation_exhausted || operand->layout.prefix != plan->prefix ||
          !block_sparse_directory_matches(operand->directory, operand->layout, ctx->device_id,
                                          source.matrix_instance_id, source.content_generation))
        return RNS8_INVALID_ARGUMENT;
    }
    const auto& capability = ctx->native_wmma_capability;
    if (!native_wmma_capability_current(*ctx, capability))
      return RNS8_INVALID_ARGUMENT;
    if (method != RNS8_BLOCK_SPARSE_SCALAR && !capability.supported)
      return RNS8_UNSUPPORTED_BACKEND;
    (void)max_scratch_bytes; // This prepared route needs no transient device metadata.
    cpp_int bound;
    rns8_exact_matrix_ranges ranges;
    status = exact_gemm_range(*plan, *a->source, *b->source, bound, &ranges);
    if (status != RNS8_SUCCESS)
      return status;
    bool commit_started = false;
    size_t token = 0;
    status = begin_synchronous_lease(*ctx, *a->source, *b->source, *c, *workspace, token);
    if (status != RNS8_SUCCESS)
      return status;
    SynchronousLease write(*ctx, token);
    write.track_write(commit_started);
    status = add_synchronous_owner_hold(*ctx, token, a->in_flight_readers);
    if (status == RNS8_SUCCESS)
      status = add_synchronous_owner_hold(*ctx, token, b->in_flight_readers);
    if (status == RNS8_SUCCESS)
      status = write.hold_buffer(a->payload);
    if (status == RNS8_SUCCESS)
      status = write.hold_buffer(b->payload);
    if (status == RNS8_SUCCESS)
      status = write.hold_buffer(a->directory_owner);
    if (status == RNS8_SUCCESS)
      status = write.hold_buffer(b->directory_owner);
    if (status != RNS8_SUCCESS)
      return write.finish(status);
    return write.finish([&]() -> rns8_status {
      status = synchronous_dispatch_checkpoint(*ctx, "block_sparse", &commit_started);
      if (status != RNS8_SUCCESS)
        return status;
      status = hip_direct_gemm_prepared_block_sparse_device(
          ctx->device_id, a->payload.get(), a->layout, a->directory, b->payload.get(), b->layout,
          b->directory, capability, c->hip_residues, c->desc.cols, commit_started,
          static_cast<uint32_t>(method), scalar_intersection_limit);
      if (status != RNS8_SUCCESS)
        return status;
      mark_output_device_residues_current(*c);
      commit_exact_range(*c, *plan, bound, &ranges);
      return RNS8_SUCCESS;
    }());
  });
}

rns8_status rns8_gemm_block_sparse_exact(rns8_context* ctx, const rns8_plan* plan,
                                         const rns8_block_sparse_operand* a,
                                         const rns8_block_sparse_operand* b, rns8_matrix* c,
                                         uint64_t budget, rns8_workspace* workspace) {
  return rns8_gemm_block_sparse_exact_with_method(ctx, plan, a, b, c, RNS8_BLOCK_SPARSE_SCALAR, 0,
                                                  budget, workspace);
}
