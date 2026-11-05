#include "core/api_internal.hpp"
#include "core/hip_resources.hpp"
#include "core/experimental_basis.hpp"
#include "core/ragged_recipe_binding.hpp"
#include <utility>
using namespace rns8::detail;
using namespace rns8::detail::api;
struct rns8_ragged_materialization {
  uint64_t recipe_references = 0;
  mutable uint64_t in_flight_uses = 0;
  rns8_context* ctx = nullptr;
  const rns8_matrix* source = nullptr;
  RaggedResidueLayout layout;
  cpp_int bound;
  std::vector<int8_t> host;
  hip_direct_device_buffer payload_owner, directory_owner, status_owner, staging_owner;
  void *device_payload = nullptr, *device_directory = nullptr, *device_status = nullptr,
       *staged_output = nullptr;
  uint64_t staged_capacity = 0;
  bool context_reference = true, cleanup_started = false;
  rns8_ragged_materialization* failed_cleanup_next = nullptr;
};
namespace {
struct UnpublishedRaggedDeleter {
  void operator()(rns8_ragged_materialization* owner) const noexcept {
    if (!owner)
      return;
    auto* context = owner->ctx;
    if (rns8_destroy_ragged_materialization(owner) == RNS8_SUCCESS)
      return;
    context->hip_failure_poisoned = true;
    --context->prepared_references;
    owner->context_reference = false;
    owner->failed_cleanup_next = context->failed_ragged_cleanup;
    context->failed_ragged_cleanup = owner;
  }
};
rns8_status hold_ragged_resources(rns8_ragged_materialization& owner, SynchronousLease& lease,
                                  size_t token) {
  auto status = add_synchronous_owner_hold(*owner.ctx, token, owner.in_flight_uses);
  for (auto* buffer :
       {&owner.payload_owner, &owner.directory_owner, &owner.status_owner, &owner.staging_owner}) {
    if (status != RNS8_SUCCESS)
      break;
    status = lease.hold_buffer(*buffer);
  }
  return status;
}
} // namespace
rns8_status rns8_prepare_ragged_materialization(rns8_context* ctx, const rns8_matrix* source,
                                                uint32_t tile_rows, uint32_t tile_cols,
                                                uint64_t budget,
                                                rns8_ragged_materialization** out) {
  return guard_api([&]() -> rns8_status {
    if (!out)
      return RNS8_INVALID_ARGUMENT;
    *out = nullptr;
    if (!ctx || !source || !tile_rows || !tile_cols || tile_rows > 128 || tile_cols > 128)
      return RNS8_INVALID_ARGUMENT;
    if (ctx->hip_failure_poisoned)
      return RNS8_BACKEND_FAILURE;
    if (ctx->backend != RNS8_BACKEND_CPU_REFERENCE && ctx->backend != RNS8_BACKEND_HIP_DIRECT)
      return RNS8_UNSUPPORTED_BACKEND;
    if (source->in_flight_writer)
      return RNS8_BUSY;
    if (source->backend != ctx->backend || !exact_wide_semantics(source->desc.semantics) ||
        !source->prefix || source->prefix > 20 || source->desc.rows <= 0 ||
        source->desc.cols <= 0 || source->generation_exhausted ||
        !basis_storage_matches(*source, ctx->backend) ||
        !rns_residue_state_current_for_backend(*source, ctx->backend) ||
        source->exact_range_prefix < source->prefix ||
        !exact_range_fits(source->desc.semantics, source->exact_max_magnitude,
                          stored_basis(*source)->products[source->prefix]))
      return RNS8_INVALID_ARGUMENT;
    if (ctx->prepared_references == UINT64_MAX || source->prepared_references == UINT64_MAX ||
        source->in_flight_readers == UINT64_MAX)
      return RNS8_RANGE_ERROR;
    const bool gpu = ctx->backend == RNS8_BACKEND_HIP_DIRECT;
    if (gpu && source->hip_device_id != ctx->device_id)
      return RNS8_INVALID_ARGUMENT;
    AddressDomain address;
    if (!checked_host_plane_address(source->desc.rows, source->desc.cols, source->desc.cols,
                                    source->prefix, 1, address))
      return RNS8_RANGE_ERROR;
    std::unique_ptr<rns8_ragged_materialization, UnpublishedRaggedDeleter> owner(
        new rns8_ragged_materialization);
    owner->ctx = ctx;
    owner->source = source;
    ++ctx->prepared_references;
    ++source->prepared_references;
    size_t token = 0;
    auto status = begin_synchronous_read_lease(*ctx, *source, token);
    if (status != RNS8_SUCCESS)
      return status;
    SynchronousLease lease(*ctx, token);
    status = hold_ragged_resources(*owner, lease, token);
    if (status != RNS8_SUCCESS)
      return lease.finish(status);
    status = lease.finish([&]() -> rns8_status {
      auto& layout = owner->layout;
      layout.rows = uint64_t(source->desc.rows);
      layout.cols = uint64_t(source->desc.cols);
      layout.tile_rows = tile_rows;
      layout.tile_cols = tile_cols;
      layout.grid_rows = 1 + (layout.rows - 1) / tile_rows;
      layout.grid_cols = 1 + (layout.cols - 1) / tile_cols;
      layout.value_instance = source->matrix_instance_id;
      layout.value_generation = source->content_generation;
      layout.signed_value = source->desc.semantics == RNS8_EXACT_WIDE_SIGNED;
      layout.basis = stored_basis(*source)->packet;
      const uint64_t limit = uint64_t(PTRDIFF_MAX);
      if (layout.grid_rows > limit / layout.grid_cols / sizeof(RaggedResidueTile))
        return RNS8_RANGE_ERROR;
      const uint64_t metadata = layout.grid_rows * layout.grid_cols * sizeof(RaggedResidueTile);
      const uint64_t snapshot =
          gpu ? metadata + sizeof(int) : 0; // device directory and proof status
      if (snapshot > limit || metadata > limit - snapshot)
        return RNS8_RANGE_ERROR;
      const uint64_t base = metadata + snapshot;
      if (base > budget)
        return RNS8_WORKSPACE_TOO_SMALL;
      layout.tiles.resize(static_cast<size_t>(layout.grid_rows * layout.grid_cols));
      const int8_t* uniform = source->residues.data();
      status = synchronous_dispatch_checkpoint(*ctx, "ragged_discover");
      if (status != RNS8_SUCCESS)
        return status;
      if (gpu) {
        status = owner->directory_owner.allocate_alias(ctx->device_id, metadata,
                                                       owner->device_directory);
        if (status != RNS8_SUCCESS)
          return status;
        status =
            owner->status_owner.allocate_alias(ctx->device_id, sizeof(int), owner->device_status);
        if (status != RNS8_SUCCESS)
          return status;
        uint64_t bound[3]{};
        const cpp_int mask = (cpp_int(1) << 64) - 1;
        for (unsigned limb = 0; limb < 3; ++limb)
          bound[limb] =
              ((source->exact_max_magnitude >> (64 * limb)) & mask).convert_to<uint64_t>();
        status = hip_direct_discover_ragged_bounds(ctx->device_id, source->hip_residues,
                                                   source->prefix, bound, layout,
                                                   owner->device_directory, owner->device_status);
        if (status != RNS8_SUCCESS)
          return status;
      }
      const uint64_t cells = layout.rows * layout.cols, area = uint64_t(tile_rows) * tile_cols;
      const cpp_int product = stored_basis(*source)->products[source->prefix],
                    mask = (cpp_int(1) << 64) - 1;
      std::vector<int8_t> residues(source->prefix);
      for (uint64_t tr = 0; tr < layout.grid_rows; ++tr)
        for (uint64_t tc = 0; tc < layout.grid_cols; ++tc) {
          auto& tile = layout.tiles[tr * layout.grid_cols + tc];
          cpp_int bound = 0;
          if (gpu)
            bound = cpp_int(tile.magnitude_bound[0]) + (cpp_int(tile.magnitude_bound[1]) << 64) +
                    (cpp_int(tile.magnitude_bound[2]) << 128);
          else
            for (uint64_t row = tr * tile_rows; row < std::min(layout.rows, (tr + 1) * tile_rows);
                 ++row)
              for (uint64_t col = tc * tile_cols; col < std::min(layout.cols, (tc + 1) * tile_cols);
                   ++col) {
                for (uint32_t p = 0; p < source->prefix; ++p) {
                  const int value = uniform[p * cells + row * layout.cols + col],
                            q = layout.basis.moduli[p];
                  if (value < -(q / 2) || value > (q - 1) / 2)
                    return RNS8_INVALID_ARGUMENT;
                  residues[p] = static_cast<int8_t>(value);
                }
                uint64_t words[3]{};
                if (!basis_reconstruct_words(residues.data(), 1, source->prefix, layout.basis,
                                             false, words))
                  return RNS8_INVALID_ARGUMENT;
                cpp_int value =
                    cpp_int(words[0]) + (cpp_int(words[1]) << 64) + (cpp_int(words[2]) << 128);
                if (layout.signed_value && value >= (product + 1) / 2)
                  value -= product;
                const cpp_int magnitude = boost::multiprecision::abs(value);
                if (magnitude > source->exact_max_magnitude)
                  return RNS8_RANGE_ERROR;
                if (magnitude > bound)
                  bound = magnitude;
              }
          tile.prefix = 1;
          while (tile.prefix < source->prefix &&
                 !exact_range_fits(source->desc.semantics, bound,
                                   stored_basis(*source)->products[tile.prefix]))
            ++tile.prefix;
          tile.byte_offset = layout.bytes;
          const uint64_t bytes = area * tile.prefix;
          if (layout.bytes > limit - bytes)
            return RNS8_RANGE_ERROR;
          layout.bytes += bytes;
          for (unsigned limb = 0; limb < 3; ++limb)
            tile.magnitude_bound[limb] = ((bound >> (64 * limb)) & mask).convert_to<uint64_t>();
          if (bound > owner->bound)
            owner->bound = bound;
        }
      if (!valid_ragged_residue_layout(layout, layout.value_instance, layout.value_generation,
                                       source->prefix))
        return RNS8_INVALID_ARGUMENT;
      const uint64_t copies = 1; // payload stays on its execution backend
      if (layout.bytes > (limit - base) / copies)
        return RNS8_RANGE_ERROR;
      if (layout.bytes > (budget - base) / copies)
        return RNS8_WORKSPACE_TOO_SMALL;
      if (!gpu)
        owner->host.assign(static_cast<size_t>(layout.bytes), 0);
      if (!gpu)
        for (uint64_t row = 0; row < layout.rows; ++row)
          for (uint64_t col = 0; col < layout.cols; ++col) {
            const auto& tile = layout.tiles[(row / tile_rows) * layout.grid_cols + col / tile_cols];
            const uint64_t local = (row % tile_rows) * tile_cols + col % tile_cols;
            for (uint32_t p = 0; p < tile.prefix; ++p)
              owner->host[tile.byte_offset + p * area + local] =
                  uniform[p * cells + row * layout.cols + col];
          }
      status = synchronous_dispatch_checkpoint(*ctx, "ragged_compact");
      if (status != RNS8_SUCCESS)
        return status;
      if (gpu) {
        status = owner->payload_owner.allocate_alias(ctx->device_id, layout.bytes,
                                                     owner->device_payload);
        if (status != RNS8_SUCCESS)
          return status;
        status = hip_direct_compact_ragged_residues(ctx->device_id, source->hip_residues,
                                                    source->prefix, layout, owner->device_payload,
                                                    owner->device_directory);
        if (status != RNS8_SUCCESS)
          return status;
        // This finalized immutable directory is retained and already current on device;
        // later materializations must not upload the same records again.
      }
      return synchronous_dispatch_checkpoint(*ctx, "ragged_prepare_publish");
    }());
    if (status != RNS8_SUCCESS)
      return status;
    *out = owner.release();
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_get_ragged_resources(const rns8_ragged_materialization* owner,
                                      rns8_ragged_resources* out) {
  if (!owner || !out || !valid_abi(out->struct_size, out->abi_version, sizeof(*out)))
    return RNS8_INVALID_ARGUMENT;
  rns8_ragged_resources r{};
  r.struct_size = sizeof(r);
  r.abi_version = RNS8_ABI_VERSION;
  r.minimum_prefix = 20;
  for (const auto& tile : owner->layout.tiles) {
    r.minimum_prefix = std::min(r.minimum_prefix, tile.prefix);
    r.maximum_prefix = std::max(r.maximum_prefix, tile.prefix);
  }
  r.tile_count = owner->layout.tiles.size();
  r.payload_bytes = owner->layout.bytes;
  r.directory_bytes = r.tile_count * sizeof(RaggedResidueTile);
  r.device_metadata_bytes =
      (owner->device_directory ? r.directory_bytes : 0) + (owner->device_status ? sizeof(int) : 0);
  r.materialization_capacity_bytes = owner->staged_output ? owner->staged_capacity : 0;
  r.source_instance = owner->layout.value_instance;
  r.source_generation = owner->layout.value_generation;
  *out = r;
  return RNS8_SUCCESS;
}
rns8_status rns8_destroy_ragged_materialization(rns8_ragged_materialization* owner) {
  if (owner) {
    if (owner->recipe_references || owner->in_flight_uses)
      return RNS8_BUSY;
    owner->cleanup_started = true;
    rns8_status cleanup = RNS8_SUCCESS;
    const std::pair<hip_direct_device_buffer*, void**> allocations[]{
        {&owner->staging_owner, &owner->staged_output},
        {&owner->payload_owner, &owner->device_payload},
        {&owner->directory_owner, &owner->device_directory},
        {&owner->status_owner, &owner->device_status}};
    for (const auto& allocation : allocations) {
      auto** pointer = allocation.second;
      if (!*pointer)
        continue;
      const auto status = allocation.first->reset();
      if (status == RNS8_SUCCESS)
        *pointer = nullptr;
      else if (cleanup == RNS8_SUCCESS)
        cleanup = status;
    }
    if (cleanup != RNS8_SUCCESS)
      return cleanup;
    if (owner->context_reference)
      --owner->ctx->prepared_references;
    --owner->source->prepared_references;
    delete owner;
  }
  return RNS8_SUCCESS;
}
rns8_status rns8_materialize_ragged_exact(rns8_context* ctx,
                                          const rns8_ragged_materialization* owner,
                                          rns8_matrix* output, uint64_t budget) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !owner || owner->ctx != ctx || owner->cleanup_started || !output ||
        output == owner->source)
      return RNS8_INVALID_ARGUMENT;
    if (ctx->hip_failure_poisoned)
      return RNS8_BACKEND_FAILURE;
    const auto& source = *owner->source;
    if (owner->in_flight_uses || source.in_flight_writer || output->in_flight_writer ||
        output->in_flight_readers)
      return RNS8_BUSY;
    if (source.generation_exhausted || source.matrix_instance_id != owner->layout.value_instance ||
        source.content_generation != owner->layout.value_generation ||
        !rns_residue_state_current_for_backend(source, ctx->backend))
      return RNS8_INVALID_ARGUMENT;
    if (output->generation_exhausted || output->prepared_references == UINT64_MAX)
      return RNS8_RANGE_ERROR;
    if (output->backend != ctx->backend || !output->prefix || output->prefix > 20 ||
        output->desc.semantics != source.desc.semantics || output->desc.rows != source.desc.rows ||
        output->desc.cols != source.desc.cols || !basis_storage_matches(*output, ctx->backend) ||
        !(ordered_basis_identity(stored_basis(*output)->packet) ==
          ordered_basis_identity(owner->layout.basis)) ||
        (ctx->backend == RNS8_BACKEND_HIP_DIRECT && output->hip_device_id != ctx->device_id))
      return RNS8_INVALID_ARGUMENT;
    if (!exact_range_fits(source.desc.semantics, owner->bound,
                          stored_basis(*output)->products[output->prefix]))
      return RNS8_RANGE_ERROR;
    ResidueLayout target;
    if (!make_residue_layout(ResidueLayoutKind::modulus_major, source.desc.rows, source.desc.cols,
                             source.desc.cols, output->prefix, 0, 0, target,
                             ordered_basis_identity(owner->layout.basis)))
      return RNS8_INVALID_ARGUMENT;
    cpp_int bound = owner->bound; // stage all fallible proof work before publication
    bool started = false;
    auto& execution = *const_cast<rns8_ragged_materialization*>(owner);
    size_t token = 0;
    auto status = begin_synchronous_lease(*ctx, source, source, *output, nullptr, token);
    if (status != RNS8_SUCCESS)
      return status;
    SynchronousLease lease(*ctx, token);
    lease.track_write(started);
    status = hold_ragged_resources(execution, lease, token);
    if (status != RNS8_SUCCESS)
      return lease.finish(status);
    return lease.finish([&]() -> rns8_status {
      status = synchronous_dispatch_checkpoint(*ctx, "ragged_materialize", &started);
      if (status != RNS8_SUCCESS)
        return status;
      if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
        const uint64_t metadata =
            owner->layout.tiles.size() * sizeof(RaggedResidueTile) + sizeof(int);
        const uint64_t capacity = std::max(target.bytes, execution.staged_capacity);
        if (capacity > UINT64_MAX - metadata || capacity + metadata > budget)
          return RNS8_WORKSPACE_TOO_SMALL;
        if (execution.staged_capacity < target.bytes) {
          if (execution.recipe_references)
            return RNS8_BUSY; // pinned recipe resource envelope
          if (execution.staged_output) {
            status = execution.staging_owner.reset();
            if (status != RNS8_SUCCESS)
              return status;
            execution.staged_output = nullptr;
            execution.staged_capacity = 0;
          }
          status = execution.staging_owner.allocate_alias(ctx->device_id, target.bytes,
                                                          execution.staged_output);
          if (status != RNS8_SUCCESS)
            return status;
          execution.staged_capacity = target.bytes;
        }
        status = hip_direct_materialize_ragged_residues(
            ctx->device_id, owner->device_payload, owner->layout, output->hip_residues, target,
            source.matrix_instance_id, source.content_generation, budget, started,
            execution.staged_output, execution.device_directory, execution.device_status, true);
        if (status != RNS8_SUCCESS)
          return status;
        status = synchronous_dispatch_checkpoint(*ctx, "ragged_materialize_publish", &started);
        if (status != RNS8_SUCCESS)
          return status;
        mark_output_device_residues_current(*output);
      } else {
        status = lease.begin_write();
        if (status != RNS8_SUCCESS)
          return status;
        status = materialize_ragged_residues_host(
            owner->host.data(), owner->layout, output->residues.data(), target,
            source.matrix_instance_id, source.content_generation, budget);
        if (status != RNS8_SUCCESS)
          return status;
        status = synchronous_dispatch_checkpoint(*ctx, "ragged_materialize_publish", &started);
        if (status != RNS8_SUCCESS)
          return status;
        mark_output_host_residues_current(*output);
      }
      output->exact_max_magnitude.swap(bound);
      output->exact_axis_ranges.rows.clear();
      output->exact_axis_ranges.cols.clear();
      output->exact_range_prefix = output->prefix;
      return RNS8_SUCCESS;
    }());
  });
}

rns8_status rns8::detail::api::drain_failed_ragged_cleanup(rns8_context& ctx) {
  while (ctx.failed_ragged_cleanup) {
    auto* owner = ctx.failed_ragged_cleanup;
    auto* next = owner->failed_cleanup_next;
    const auto status = rns8_destroy_ragged_materialization(owner);
    if (status != RNS8_SUCCESS)
      return status;
    ctx.failed_ragged_cleanup = next;
  }
  return RNS8_SUCCESS;
}

rns8_status rns8::detail::ragged_recipe_binding_info(rns8_context* ctx,
                                                     const rns8_ragged_materialization* owner,
                                                     const rns8_matrix*& source, uint64_t& retained,
                                                     uint64_t& capacity,
                                                     uint64_t* exact_magnitude_words) {
  if (!ctx || !owner || owner->ctx != ctx || owner->cleanup_started)
    return RNS8_INVALID_ARGUMENT;
  if (ctx->hip_failure_poisoned)
    return RNS8_BACKEND_FAILURE;
  if (owner->in_flight_uses || owner->source->in_flight_writer)
    return RNS8_BUSY;
  source = owner->source;
  if (source->generation_exhausted || source->matrix_instance_id != owner->layout.value_instance ||
      source->content_generation != owner->layout.value_generation ||
      !rns_residue_state_current_for_backend(*source, ctx->backend))
    return RNS8_INVALID_ARGUMENT;
  const uint64_t metadata = owner->layout.tiles.capacity() * sizeof(RaggedResidueTile);
  retained = metadata;
  const auto add = [&](uint64_t n) {
    if (n > UINT64_MAX - retained)
      return false;
    retained += n;
    return true;
  };
  if (!add(owner->device_payload ? owner->layout.bytes : owner->host.capacity()) ||
      !add(owner->device_directory ? owner->layout.tiles.size() * sizeof(RaggedResidueTile) : 0) ||
      !add(owner->device_status ? sizeof(int) : 0) ||
      !add(owner->staged_output ? owner->staged_capacity : 0))
    return RNS8_RANGE_ERROR;
  if (exact_magnitude_words) {
    cpp_int magnitude = owner->bound;
    const cpp_int mask = (cpp_int(1) << 64) - 1;
    uint64_t words[3]{};
    if (magnitude < 0)
      return RNS8_RANGE_ERROR;
    for (auto& word : words) {
      word = (magnitude & mask).convert_to<uint64_t>();
      magnitude >>= 64;
    }
    if (magnitude != 0)
      return RNS8_RANGE_ERROR;
    std::copy_n(words, 3, exact_magnitude_words);
  }
  capacity = owner->staged_output ? owner->staged_capacity : 0;
  return RNS8_SUCCESS;
}
rns8_status rns8::detail::retain_ragged_recipe_binding(rns8_ragged_materialization* owner) {
  if (!owner || owner->cleanup_started)
    return RNS8_INVALID_ARGUMENT;
  if (owner->recipe_references == UINT64_MAX)
    return RNS8_RANGE_ERROR;
  ++owner->recipe_references;
  return RNS8_SUCCESS;
}
void rns8::detail::release_ragged_recipe_binding(rns8_ragged_materialization* owner) noexcept {
  if (owner && owner->recipe_references)
    --owner->recipe_references;
}
