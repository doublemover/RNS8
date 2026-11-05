#include "core/runtime_product_recipe.hpp"
#include "core/api_internal.hpp"
#include "core/hip_resources.hpp"
#include "core/ragged_recipe_binding.hpp"
#include "core/experimental_basis.hpp"
#include "rns8/experimental_basis.h"
#include "rns8/experimental_product.h"
#include <array>
#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <unordered_set>
#include <unordered_map>
namespace rns8::detail {
using namespace api;
namespace {
// A nonblocking owner gate also rejects same-thread callbacks without invoking
// std::mutex's undefined recursive try_lock behavior. No waiter spins or blocks.
struct RecipeTryMutex {
  std::atomic_flag entered = ATOMIC_FLAG_INIT;
  bool try_lock() noexcept { return !entered.test_and_set(std::memory_order_acquire); }
  void unlock() noexcept { entered.clear(std::memory_order_release); }
};
std::shared_ptr<const ExperimentalBasis> recipe_basis(const rns8_matrix& matrix) {
  return matrix.experimental_basis ? matrix.experimental_basis : default_experimental_basis();
}
std::vector<const rns8_matrix*>
unique_recipe_inputs(const std::vector<const rns8_matrix*>& inputs) {
  auto unique = inputs;
  std::sort(unique.begin(), unique.end(), std::less<const rns8_matrix*>{});
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  return unique;
}
bool recipe_storage_matches(const rns8_matrix& matrix, rns8_backend_kind backend, int64_t rows,
                            int64_t cols, uint32_t prefix, const OrderedBasisIdentity& basis) {
  if (!(ordered_basis_identity(recipe_basis(matrix)->packet) == basis) || !prefix ||
      prefix > basis.count || matrix.prefix != prefix || matrix.materialized_rns_schedule ||
      matrix.desc.bound_kind != RNS8_BOUND_NONE ||
      matrix.desc.logical_layout != RNS8_LAYOUT_ROW_MAJOR || matrix.desc.rows != rows ||
      matrix.desc.cols != cols || !matrix.byte_limbs.empty() || matrix.hip_byte_limbs ||
      matrix.hip_byte_limb_bytes || matrix.host_byte_limbs_current ||
      matrix.device_byte_limbs_current)
    return false;
  AddressDomain domain;
  if (!checked_host_plane_address(rows, cols, cols, prefix, 1, domain))
    return false;
  if (backend == RNS8_BACKEND_HIP_DIRECT)
    return matrix.hip_residues && matrix.hip_residue_bytes == domain.byte_extent &&
           (matrix.residues.size() == domain.byte_extent ||
            (matrix.residues.empty() && !matrix.host_residues_current));
  return matrix.residues.size() == domain.byte_extent && !matrix.hip_residues &&
         !matrix.hip_residue_bytes && !matrix.device_residues_current;
}
} // namespace
struct RuntimeProductRecipe::Impl {
  rns8_context* ctx = nullptr;
  struct Input {
    const rns8_matrix* matrix;
    uint64_t instance, generation;
  };
  std::vector<Input> inputs;
  std::vector<ProductNode> nodes;
  std::vector<uint32_t> outputs;
  ProductArenaPlan allocation;
  std::vector<cpp_int> bounds;
  std::vector<uint32_t> compute_prefixes, capacities;
  // Immutable topology prepares these host metadata capacities once. The
  // execution mutex excludes reuse until publication/drain releases readers.
  std::vector<uint32_t> current, packed_current;
  std::vector<ResidueLayout> layouts;
  std::vector<const rns8_matrix*> unique_inputs;
  std::vector<rns8_matrix*> sorted_outputs;
  std::vector<cpp_int> publication_bounds;
  std::vector<std::array<uint64_t, 3>> magnitude_words;
  uint64_t lift_scratch = 0;
  std::shared_ptr<const ExperimentalBasis> basis;
  OrderedBasisIdentity basis_identity;
  bool experimental = false;
  uint64_t packet_offset = 0;
  int lift_status = 0;
  ProductPrefixSelection selection;
  ProductMaterializationSelection materializations;
  ReplayCostPolicy replay_policy;
  uint64_t packing_calls = 0, reused_packing_calls = 0, replays = 0, expected_replays = 1;
  uint32_t prefix = 0;
  rns8_semantics semantics = RNS8_EXACT_WIDE_SIGNED;
  std::vector<int8_t> host;
  void* device = nullptr;
  struct OrderedResources {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    hip_unique_stream stream;
    hip_direct_device_buffer status;
    hip_pinned_host_buffer result;
#endif
    bool pending = false;
  };
  std::unique_ptr<OrderedResources> ordered;
  rns8_product_execution_policy execution_policy = RNS8_PRODUCT_EXECUTION_SYNCHRONOUS_WRAPPERS;
  rns8_product_recipe_execution_info observations{};
  uint64_t max_budget = UINT64_MAX;
  bool needs_lift_guard = false;
  bool quarantined = false, closed = false, closing = false, output_touched = false;
  uint64_t control_device_bytes() const noexcept {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    return ordered ? ordered->status.bytes() : 0;
#else
    return 0;
#endif
  }
  void* stream() const noexcept {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    return ordered ? ordered->stream.get() : nullptr;
#else
    return nullptr;
#endif
  }
  int* sticky() const noexcept {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    return ordered ? static_cast<int*>(ordered->status.get()) : nullptr;
#else
    return nullptr;
#endif
  }
  rns8_status drain_ordered(bool record_wait) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (!ordered || !ordered->pending)
      return RNS8_SUCCESS;
    if (record_wait)
      ++observations.stream_wait_calls;
    if (drain_synchronous_context(*ctx, ordered->stream.get()) != RNS8_SUCCESS)
      return RNS8_BACKEND_FAILURE;
    ordered->pending = false;
    ordered->status.mark_completed();
    ordered->result.mark_completed();
    return RNS8_SUCCESS;
#else
    (void)record_wait;
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  }
  std::vector<const rns8_matrix*> active_inputs;
  std::vector<rns8_matrix*> active_outputs;
  std::vector<uint64_t*> holds;
  RecipeTryMutex mutex;
  void release_active() noexcept {
    if (active_outputs.empty())
      return;
    for (const auto* input : active_inputs)
      --input->in_flight_readers;
    for (auto* output : active_outputs) {
      output->in_flight_writer = false;
      --output->prepared_references;
    }
    active_outputs.clear();
    active_inputs.clear();
    output_touched = false;
  }
  void invalidate_touched() noexcept {
    if (output_touched)
      for (auto* output : active_outputs) {
        clear_residue_current(*output);
        clear_byte_limb_current(*output);
        clear_native_current(*output);
      }
  }
  ~Impl() {
    // An executing/quarantined owner is never destroyed: close preserves it.
    if (device)
      (void)hip_direct_free(ctx->device_id, device);
    for (auto* hold : holds)
      --*hold;
  }
};
RuntimeProductRecipe::RuntimeProductRecipe(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}
RuntimeProductRecipe::~RuntimeProductRecipe() {
  // Private RAII users cannot return a destruction error. Preserve the complete
  // owner (including matrix/context holds) rather than free in-flight storage.
  if (close() != RNS8_SUCCESS)
    (void)impl_.release();
}
rns8_status RuntimeProductRecipe::close() {
  return guard_api([&]() -> rns8_status {
    auto& state = *impl_;
    std::unique_lock<RecipeTryMutex> lock(state.mutex, std::try_to_lock);
    if (!lock.owns_lock())
      return RNS8_BUSY;
    if (state.closed)
      return RNS8_SUCCESS;
    state.closing = true;
    if (state.quarantined || !state.active_outputs.empty()) {
      if (state.ctx->backend == RNS8_BACKEND_HIP_DIRECT &&
          (state.execution_policy == RNS8_PRODUCT_EXECUTION_ORDERED_HIP
               ? state.drain_ordered(false)
               : hip_direct_synchronize(state.ctx->device_id)) != RNS8_SUCCESS) {
        state.ctx->hip_failure_poisoned = true;
        return RNS8_BACKEND_FAILURE;
      }
      state.invalidate_touched();
      state.release_active();
      state.quarantined = false;
    }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (state.ordered) {
      if (state.ordered->stream.reset() != hipSuccess ||
          state.ordered->result.reset() != hipSuccess ||
          state.ordered->status.reset() != RNS8_SUCCESS)
        return RNS8_BACKEND_FAILURE;
    }
#endif
    if (state.device) {
      const auto status = hip_direct_free(state.ctx->device_id, state.device);
      if (status != RNS8_SUCCESS)
        return status;
      state.device = nullptr;
    }
    state.closed = true;
    return RNS8_SUCCESS;
  });
}
const ProductPrefixSelection& RuntimeProductRecipe::cost_selection() const {
  return impl_->selection;
}
const ProductMaterializationSelection& RuntimeProductRecipe::materialization_selection() const {
  return impl_->materializations;
}
uint64_t RuntimeProductRecipe::completed_packing_calls() const {
  return impl_->packing_calls;
}
uint64_t RuntimeProductRecipe::completed_replays() const {
  return impl_->replays;
}
uint64_t RuntimeProductRecipe::reused_packing_calls() const {
  return impl_->reused_packing_calls;
}
uint64_t RuntimeProductRecipe::expected_replays() const {
  return impl_->expected_replays;
}
const ReplayCostPolicy& RuntimeProductRecipe::replay_policy() const {
  return impl_->replay_policy;
}
uint32_t RuntimeProductRecipe::materialization_b_value(uint32_t node) const {
  return impl_->nodes[node].b;
}
const ProductArenaPlan& RuntimeProductRecipe::allocation() const {
  return impl_->allocation;
}
uint64_t RuntimeProductRecipe::lift_scratch_bytes() const {
  return impl_->lift_scratch + impl_->control_device_bytes();
}
const std::vector<uint32_t>& RuntimeProductRecipe::compute_prefixes() const {
  return impl_->compute_prefixes;
}
const BasisPacket& RuntimeProductRecipe::basis_packet() const {
  return impl_->basis->packet;
}
uint32_t RuntimeProductRecipe::output_prefix() const {
  return impl_->prefix;
}
uint64_t RuntimeProductRecipe::additional_ordered_control_bytes() const {
  return impl_->needs_lift_guard && !impl_->control_device_bytes() ? sizeof(int) : 0;
}
rns8_product_recipe_execution_info RuntimeProductRecipe::execution_info() const {
  const auto& state = *impl_;
  auto out = state.observations;
  out.struct_size = sizeof(out);
  out.abi_version = RNS8_ABI_VERSION;
  out.policy = state.execution_policy;
  out.stream_prepared = state.stream() != nullptr;
  out.guarded_layout_products =
      state.execution_policy == RNS8_PRODUCT_EXECUTION_ORDERED_HIP && state.needs_lift_guard;
  out.retained_control_device_bytes = state.control_device_bytes();
  out.retained_layout_host_bytes = state.layouts.capacity() * sizeof(ResidueLayout);
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  out.retained_control_pinned_bytes = state.ordered ? state.ordered->result.capacity() : 0;
#endif
  return out;
}
rns8_status RuntimeProductRecipe::set_execution_policy(rns8_product_execution_policy policy) {
  return guard_api([&]() -> rns8_status {
    auto& state = *impl_;
    std::unique_lock<RecipeTryMutex> lock(state.mutex, std::try_to_lock);
    if (!lock.owns_lock())
      return RNS8_BUSY;
    if (uint32_t(policy) > 1 || state.closed || state.closing)
      return RNS8_INVALID_ARGUMENT;
    if (state.ctx->hip_failure_poisoned || state.quarantined)
      return RNS8_BACKEND_FAILURE;
    if (policy == RNS8_PRODUCT_EXECUTION_ORDERED_HIP) {
      if (state.ctx->backend != RNS8_BACKEND_HIP_DIRECT)
        return RNS8_UNSUPPORTED_BACKEND;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      if (!state.ordered) {
        for (size_t i = 0; i < state.nodes.size(); ++i) {
          const auto node = state.nodes[i];
          const auto prefix = state.compute_prefixes[i];
          const bool packed =
              state.materializations.status == RNS8_SUCCESS && state.materializations.packed_b[i];
          const auto& al = state.layouts[node.a];
          const auto& cl = state.layouts[state.inputs.size() + i];
          if (packed) {
            const auto& bl = state.materializations.packed[node.b].layout;
            const auto converted =
                query_hip_layout_conversion_support(state.layouts[node.b], bl, 0, prefix);
            if (converted != RNS8_SUCCESS)
              return converted;
            const auto multiplied = query_hip_layout_gemm_support(al, bl, cl, prefix);
            if (multiplied != RNS8_SUCCESS)
              return multiplied;
          } else if (state.needs_lift_guard) {
            const auto multiplied =
                query_hip_layout_gemm_support(al, state.layouts[node.b], cl, prefix);
            if (multiplied != RNS8_SUCCESS)
              return multiplied;
          }
        }
        const uint64_t base = state.allocation.arena_bytes + state.lift_scratch;
        const uint64_t extra = additional_ordered_control_bytes();
        if (base > state.max_budget || extra > state.max_budget - base)
          return RNS8_WORKSPACE_TOO_SMALL;
        auto candidate = std::make_unique<Impl::OrderedResources>();
        hip_scoped_device selected(state.ctx->device_id);
        if (selected.status() != hipSuccess)
          return RNS8_BACKEND_FAILURE;
        if (candidate->stream.create_non_blocking() != hipSuccess)
          return RNS8_BACKEND_FAILURE;
        if (state.needs_lift_guard) {
          const auto allocated = candidate->status.allocate(state.ctx->device_id, sizeof(int));
          if (allocated != RNS8_SUCCESS)
            return allocated;
          if (candidate->result.ensure(state.ctx->device_id, sizeof(int)) != hipSuccess)
            return RNS8_BACKEND_FAILURE;
        }
        state.ordered = std::move(candidate);
      }
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    if (state.execution_policy != policy)
      state.observations = {};
    state.execution_policy = policy;
    return RNS8_SUCCESS;
  });
}
rns8_status RuntimeProductRecipe::create(rns8_context* ctx,
                                         const std::vector<const rns8_matrix*>& inputs,
                                         const std::vector<ProductNode>& nodes, uint64_t budget,
                                         std::unique_ptr<RuntimeProductRecipe>& out,
                                         bool minimize_compute_prefix,
                                         const ProductCostProfile* costs,
                                         const std::vector<uint32_t>& outputs,
                                         const ProductMaterializationProfile* materializations) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || inputs.empty() || inputs.size() > 4096 || nodes.empty() || nodes.size() > 4096 ||
        out)
      return RNS8_INVALID_ARGUMENT;
    if (ctx->hip_failure_poisoned)
      return RNS8_BACKEND_FAILURE;
    if (costs && materializations)
      return RNS8_INVALID_ARGUMENT;
    if (materializations)
      costs = &materializations->uniform;
    if (ctx->backend != RNS8_BACKEND_CPU_REFERENCE && ctx->backend != RNS8_BACKEND_HIP_DIRECT)
      return RNS8_UNSUPPORTED_BACKEND;
    auto state = std::make_unique<Impl>();
    state->max_budget = budget;
    state->ctx = ctx;
    if (!inputs.front())
      return RNS8_INVALID_ARGUMENT;
    state->basis = recipe_basis(*inputs.front());
    state->basis_identity = ordered_basis_identity(state->basis->packet);
    state->experimental = bool(inputs.front()->experimental_basis);
    // Equal content may come from separate basis handles. Provider selection is
    // frozen for this owner; all non-default content uses the runtime-modulus path.
    for (const auto* input : inputs)
      if (input && input->experimental_basis)
        state->experimental = true;
    std::vector<ProductShape> shapes;
    std::unordered_set<uint64_t*> holds{&ctx->prepared_references};
    for (const auto* input : inputs) {
      if (!input)
        return RNS8_INVALID_ARGUMENT;
      if (input->in_flight_writer)
        return RNS8_BUSY;
      if (!exact_wide_semantics(input->desc.semantics) || input->backend != ctx->backend ||
          input->generation_exhausted ||
          !rns_residue_state_current_for_backend(*input, ctx->backend) ||
          !recipe_storage_matches(*input, ctx->backend, input->desc.rows, input->desc.cols,
                                  input->prefix, state->basis_identity) ||
          (ctx->backend == RNS8_BACKEND_HIP_DIRECT && input->hip_device_id != ctx->device_id))
        return RNS8_INVALID_ARGUMENT;
      if (state->inputs.empty()) {
        state->prefix = input->prefix;
        state->semantics = input->desc.semantics;
      }
      if (input->prefix != state->prefix || input->desc.semantics != state->semantics ||
          input->exact_range_prefix < state->prefix ||
          !exact_range_fits(state->semantics, input->exact_max_magnitude,
                            state->basis->products[state->prefix]))
        return RNS8_RANGE_ERROR;
      state->inputs.push_back({input, input->matrix_instance_id, input->content_generation});
      shapes.push_back({input->desc.rows, input->desc.cols});
      state->bounds.push_back(input->exact_max_magnitude);
      holds.insert(&input->prepared_references);
    }
    state->outputs = outputs;
    if (state->outputs.empty())
      state->outputs.push_back(static_cast<uint32_t>(inputs.size() + nodes.size() - 1));
    state->allocation =
        plan_product_arena(shapes, nodes, state->prefix, UINT64_MAX, {}, state->outputs);
    if (state->allocation.status != RNS8_SUCCESS)
      return state->allocation.status;
    state->nodes = nodes;
    for (size_t i = 0; i < nodes.size(); ++i) {
      const auto node = nodes[i];
      const auto a = state->allocation.values[node.a].shape,
                 b = state->allocation.values[node.b].shape;
      if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
        const auto support = query_residue_gemm_support(a.rows, b.cols, a.cols, a.cols, b.cols,
                                                        b.cols, state->prefix);
        if (support.status != RNS8_SUCCESS)
          return support.status;
      }
      cpp_int bound = cpp_int(a.cols) * state->bounds[node.a] * state->bounds[node.b];
      if (!exact_range_fits(state->semantics, bound, state->basis->products[state->prefix]))
        return RNS8_RANGE_ERROR;
      uint32_t selected = state->prefix;
      if (minimize_compute_prefix || (costs && !materializations)) {
        selected = 1;
        while (selected < state->prefix &&
               !exact_range_fits(state->semantics, bound, state->basis->products[selected]))
          ++selected;
      }
      state->compute_prefixes.push_back(selected);
      state->bounds.push_back(std::move(bound));
    }
    if (costs) {
      const std::string target =
          ctx->backend == RNS8_BACKEND_HIP_DIRECT ? ctx->device_info.gcn_arch : "cpu";
      if (costs->backend != ctx->backend || costs->semantics != state->semantics ||
          costs->target != target || !(costs->basis == state->basis_identity) ||
          costs->experimental_basis != state->experimental)
        return RNS8_INVALID_ARGUMENT;
      if (materializations) {
        state->expected_replays = materializations->replay.replays;
        state->replay_policy = materializations->replay;
        state->materializations = select_product_materializations(
            shapes, nodes, state->prefix, state->compute_prefixes, budget,
            ctx->backend == RNS8_BACKEND_HIP_DIRECT, *materializations, state->outputs);
        if (state->materializations.status != RNS8_SUCCESS)
          return state->materializations.status;
        state->compute_prefixes = state->materializations.prefixes;
      } else {
        state->selection = select_product_prefixes(
            shapes, nodes, state->prefix, state->compute_prefixes, budget,
            ctx->backend == RNS8_BACKEND_HIP_DIRECT, *costs, state->outputs);
        if (state->selection.status != RNS8_SUCCESS)
          return state->selection.status;
        state->compute_prefixes = state->selection.prefixes;
      }
    }
    state->capacities = state->compute_prefixes;
    for (size_t i = 0; i < nodes.size(); ++i)
      for (auto id : {nodes[i].a, nodes[i].b})
        if (id >= inputs.size())
          state->capacities[id - inputs.size()] =
              std::max(state->capacities[id - inputs.size()], state->compute_prefixes[i]);
    for (auto id : state->outputs)
      state->capacities[id - inputs.size()] = state->prefix; // complete observed values
    for (size_t i = 0; i < nodes.size(); ++i)
      if (state->capacities[i] > state->compute_prefixes[i]) {
        const auto shape = state->allocation.values[inputs.size() + i].shape;
        if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
          if (state->experimental) {
            const auto lift = basis_lift_storage(shape.rows * shape.cols,
                                                 state->compute_prefixes[i], state->capacities[i]);
            if (lift.status != RNS8_SUCCESS)
              return lift.status;
            if (lift.scratch_bytes > state->lift_scratch) {
              state->lift_scratch = lift.scratch_bytes;
              state->packet_offset = lift.packet_offset;
            }
          } else {
            const auto lift =
                query_exact_lift_support(shape.rows * shape.cols, state->compute_prefixes[i],
                                         state->capacities[i], UINT64_MAX);
            if (lift.status != RNS8_SUCCESS)
              return lift.status;
            state->lift_scratch = std::max(state->lift_scratch, lift.scratch_bytes);
          }
        }
      }
    if (state->lift_scratch > budget)
      return RNS8_WORKSPACE_TOO_SMALL;
    if (materializations) {
      if (state->capacities != state->materializations.capacities ||
          state->lift_scratch != state->materializations.lift_scratch_bytes)
        return RNS8_INTERNAL_ERROR;
      state->allocation = state->materializations.arena;
    } else
      state->allocation =
          plan_product_arena(shapes, nodes, state->prefix, budget - state->lift_scratch,
                             state->capacities, state->outputs);
    if (state->allocation.status != RNS8_SUCCESS)
      return state->allocation.status;
    for (size_t i = 0; i < state->compute_prefixes.size(); ++i)
      state->needs_lift_guard |= state->compute_prefixes[i] < state->capacities[i];
    state->layouts.resize(state->allocation.values.size());
    for (size_t id = 0; id < state->layouts.size(); ++id) {
      const auto shape = state->allocation.values[id].shape;
      const auto capacity =
          id < inputs.size() ? state->prefix : state->capacities[id - inputs.size()];
      if (!make_residue_layout(ResidueLayoutKind::modulus_major, shape.rows, shape.cols, shape.cols,
                               capacity, 0, 0, state->layouts[id], state->basis_identity))
        return RNS8_RANGE_ERROR;
    }
    state->current.resize(state->allocation.values.size());
    if (std::any_of(state->materializations.packed_b.begin(),
                    state->materializations.packed_b.end(),
                    [](uint8_t value) { return value != 0; }))
      state->packed_current.resize(state->allocation.values.size());
    state->unique_inputs = unique_recipe_inputs(inputs);
    state->active_inputs.reserve(inputs.size());
    state->active_outputs.reserve(state->outputs.size());
    state->sorted_outputs.reserve(state->outputs.size());
    state->publication_bounds.reserve(state->outputs.size());
    for (auto id : state->outputs)
      state->publication_bounds.push_back(state->bounds[id]);
    if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
      state->magnitude_words.resize(state->bounds.size());
      for (size_t id = 0; id < state->bounds.size(); ++id) {
        auto magnitude = state->bounds[id];
        for (auto& word : state->magnitude_words[id]) {
          word = static_cast<uint64_t>(magnitude & UINT64_MAX);
          magnitude >>= 64;
        }
        if (magnitude != 0)
          return RNS8_RANGE_ERROR;
      }
    }
    for (auto* hold : holds)
      if (*hold == UINT64_MAX)
        return RNS8_RANGE_ERROR;
    if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
      // Arena slots are 256-byte aligned; append one int-aligned maximum lift
      // region to the same retained allocation. No per-materialization malloc.
      if (state->allocation.arena_bytes > uint64_t(PTRDIFF_MAX) ||
          state->lift_scratch > uint64_t(PTRDIFF_MAX) - state->allocation.arena_bytes)
        return RNS8_RANGE_ERROR;
      const auto status = hip_direct_allocate(
          ctx->device_id, static_cast<size_t>(state->allocation.arena_bytes + state->lift_scratch),
          &state->device);
      if (status != RNS8_SUCCESS)
        return status;
    } else
      state->host.resize(static_cast<size_t>(state->allocation.arena_bytes));
    // Allocate the outer owner and reference list before taking any references.
    auto result = std::unique_ptr<RuntimeProductRecipe>(new RuntimeProductRecipe(std::move(state)));
    result->impl_->holds.assign(holds.begin(), holds.end());
    for (auto* hold : result->impl_->holds)
      ++*hold;
    if (ctx->backend == RNS8_BACKEND_HIP_DIRECT && result->impl_->experimental &&
        result->impl_->lift_scratch) {
      auto& retained = *result->impl_;
      auto* packet = static_cast<int8_t*>(retained.device) + retained.allocation.arena_bytes +
                     retained.packet_offset;
      // Immutable packet upload belongs to preparation, including the first
      // replay. Retain the complete owner if initialization cannot be drained.
      retained.quarantined = true;
      const auto uploaded = hip_direct_copy_host_to_device(
          ctx->device_id, packet, &retained.basis->packet, sizeof(BasisPacket));
      const auto drained = hip_direct_synchronize(ctx->device_id);
      if (drained != RNS8_SUCCESS) {
        ctx->hip_failure_poisoned = true;
        return drained;
      }
      retained.quarantined = false;
      if (uploaded != RNS8_SUCCESS)
        return uploaded;
    }
    out = std::move(result);
    return RNS8_SUCCESS;
  });
}
rns8_status RuntimeProductRecipe::rebind(const std::vector<const rns8_matrix*>& inputs) {
  return guard_api([&]() -> rns8_status {
    auto& state = *impl_;
    std::unique_lock<RecipeTryMutex> lock(state.mutex, std::try_to_lock);
    if (!lock.owns_lock())
      return RNS8_BUSY;
    if (state.closed || state.closing || inputs.size() != state.inputs.size())
      return RNS8_INVALID_ARGUMENT;
    if (state.ctx->hip_failure_poisoned || state.quarantined)
      return RNS8_BACKEND_FAILURE;
    std::vector<Impl::Input> replacement;
    replacement.reserve(inputs.size());
    std::unordered_set<uint64_t*> new_holds{&state.ctx->prepared_references};
    const std::unordered_set<uint64_t*> old_holds(state.holds.begin(), state.holds.end());
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto* input = inputs[i];
      if (!input)
        return RNS8_INVALID_ARGUMENT;
      if (input->in_flight_writer)
        return RNS8_BUSY;
      const auto shape = state.allocation.values[i].shape;
      if (input->backend != state.ctx->backend || input->desc.semantics != state.semantics ||
          input->prefix != state.prefix || input->generation_exhausted ||
          input->desc.rows != shape.rows || input->desc.cols != shape.cols ||
          !rns_residue_state_current_for_backend(*input, state.ctx->backend) ||
          !recipe_storage_matches(*input, state.ctx->backend, shape.rows, shape.cols, state.prefix,
                                  state.basis_identity) ||
          (state.ctx->backend == RNS8_BACKEND_HIP_DIRECT &&
           input->hip_device_id != state.ctx->device_id))
        return RNS8_INVALID_ARGUMENT;
      if (input->exact_range_prefix < state.prefix || input->exact_max_magnitude < 0 ||
          input->exact_max_magnitude > state.bounds[i] ||
          !exact_range_fits(state.semantics, input->exact_max_magnitude,
                            state.basis->products[state.prefix]))
        return RNS8_RANGE_ERROR;
      replacement.push_back({input, input->matrix_instance_id, input->content_generation});
      new_holds.insert(&input->prepared_references);
    }
    std::vector<uint64_t*> replacement_holds(new_holds.begin(), new_holds.end());
    auto unique_inputs = unique_recipe_inputs(inputs);
    for (auto* hold : replacement_holds)
      if (!old_holds.count(hold) && *hold == UINT64_MAX)
        return RNS8_RANGE_ERROR;
    // Nothing below allocates or validates. Take new unique owners before
    // dropping old owners, then atomically publish all generation snapshots.
    for (auto* hold : replacement_holds)
      if (!old_holds.count(hold))
        ++*hold;
    for (auto* hold : state.holds)
      if (!new_holds.count(hold))
        --*hold;
    for (size_t i = 0; i < replacement.size(); ++i)
      if (replacement[i].instance != state.inputs[i].instance ||
          replacement[i].generation != state.inputs[i].generation)
        if (i < state.packed_current.size())
          state.packed_current[i] = 0;
    state.inputs.swap(replacement);
    state.holds.swap(replacement_holds);
    state.unique_inputs.swap(unique_inputs);
    return RNS8_SUCCESS;
  });
}
rns8_status RuntimeProductRecipe::execute(rns8_matrix* output) {
  return execute(&output, 1);
}
rns8_status RuntimeProductRecipe::execute(const std::vector<rns8_matrix*>& outputs) {
  return execute(outputs.data(), outputs.size());
}
rns8_status RuntimeProductRecipe::execute(rns8_matrix* const* outputs, size_t count) {
  return guard_api([&]() -> rns8_status {
    auto& state = *impl_;
    std::unique_lock<RecipeTryMutex> lock(state.mutex, std::try_to_lock);
    if (!lock.owns_lock())
      return RNS8_BUSY;
    if (state.closed || state.closing || !outputs || count != state.outputs.size())
      return RNS8_INVALID_ARGUMENT;
    if (state.replays == UINT64_MAX)
      return RNS8_RANGE_ERROR;
    if (state.ctx->hip_failure_poisoned || state.quarantined)
      return RNS8_BACKEND_FAILURE;
    state.sorted_outputs.assign(outputs, outputs + count);
    std::sort(state.sorted_outputs.begin(), state.sorted_outputs.end(), std::less<rns8_matrix*>{});
    if (std::adjacent_find(state.sorted_outputs.begin(), state.sorted_outputs.end()) !=
        state.sorted_outputs.end())
      return RNS8_INVALID_ARGUMENT;
    for (size_t i = 0; i < count; ++i) {
      auto* output = outputs[i];
      if (!output)
        return RNS8_INVALID_ARGUMENT;
      if (output->in_flight_readers || output->in_flight_writer)
        return RNS8_BUSY;
      const auto shape = state.allocation.values[state.outputs[i]].shape;
      if (output->backend != state.ctx->backend || output->desc.semantics != state.semantics ||
          output->desc.rows != shape.rows || output->desc.cols != shape.cols ||
          output->prefix != state.prefix ||
          !recipe_storage_matches(*output, state.ctx->backend, shape.rows, shape.cols, state.prefix,
                                  state.basis_identity) ||
          (state.ctx->backend == RNS8_BACKEND_HIP_DIRECT &&
           output->hip_device_id != state.ctx->device_id))
        return RNS8_INVALID_ARGUMENT;
      if (output->generation_exhausted || output->prepared_references == UINT64_MAX)
        return RNS8_RANGE_ERROR;
    }
    for (const auto& input : state.inputs) {
      const auto& matrix = *input.matrix;
      if (std::binary_search(state.sorted_outputs.begin(), state.sorted_outputs.end(),
                             const_cast<rns8_matrix*>(input.matrix), std::less<rns8_matrix*>{}))
        return RNS8_INVALID_ARGUMENT;
      if (matrix.in_flight_writer)
        return RNS8_BUSY;
      if (matrix.generation_exhausted || matrix.matrix_instance_id != input.instance ||
          matrix.content_generation != input.generation ||
          !rns_residue_state_current_for_backend(matrix, state.ctx->backend))
        return RNS8_INVALID_ARGUMENT;
    }
    // Copy every proof before work; final publication swaps without allocation.
    auto& bounds = state.publication_bounds;
    for (size_t i = 0; i < count; ++i)
      bounds[i] = state.bounds[state.outputs[i]];
    const bool hip = state.ctx->backend == RNS8_BACKEND_HIP_DIRECT;
    const bool ordered = hip && state.execution_policy == RNS8_PRODUCT_EXECUTION_ORDERED_HIP;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    std::optional<hip_scoped_device> ordered_device;
    if (ordered) {
      ordered_device.emplace(state.ctx->device_id);
      if (ordered_device->status() != hipSuccess)
        return RNS8_BACKEND_FAILURE;
    }
#endif
    auto* arena = hip ? static_cast<int8_t*>(state.device) : state.host.data();
    const auto pointer = [&](uint32_t id) -> const int8_t* {
      if (id < state.inputs.size()) {
        const auto& matrix = *state.inputs[id].matrix;
        return hip ? static_cast<const int8_t*>(matrix.hip_residues) : matrix.residues.data();
      }
      return arena + state.allocation.values[id].offset;
    };
    auto& current = state.current;
    std::fill(current.begin(), current.end(), 0);
    auto& packed_current = state.packed_current;
    const auto retained = state.materializations.retain_external_b ? state.inputs.size() : 0;
    if (!packed_current.empty())
      std::fill(packed_current.begin() + retained, packed_current.end(), 0);
    state.packing_calls = 0;
    state.reused_packing_calls = 0;
    std::fill_n(current.begin(), state.inputs.size(), state.prefix);
    // Populate and validate retained reservation storage before taking any
    // counters. Duplicate external input handles get one reader reservation.
    state.active_inputs.assign(state.unique_inputs.begin(), state.unique_inputs.end());
    for (const auto* input : state.active_inputs)
      if (input->in_flight_readers == UINT64_MAX)
        return RNS8_RANGE_ERROR;
    // Capacity was reserved at preparation for the fixed output count.
    state.active_outputs.assign(outputs, outputs + count);
    for (const auto* input : state.active_inputs)
      ++input->in_flight_readers;
    for (auto* output : state.active_outputs) {
      ++output->prepared_references;
      output->in_flight_writer = true;
    }
    state.output_touched = false;
    state.observations = {};
    struct RunGuard {
      Impl& state;
      bool completed = false, quiescent = false, use_ordered = false;
      rns8_status wait() {
        const auto status = state.drain_ordered(true);
        quiescent = status == RNS8_SUCCESS;
        return status;
      }
      void clear_packing() noexcept {
        std::fill(state.packed_current.begin(), state.packed_current.end(), 0);
        state.packing_calls = state.reused_packing_calls = 0;
      }
      ~RunGuard() noexcept {
        if (completed)
          return;
        clear_packing();
        state.invalidate_touched();
        if (state.ctx->backend == RNS8_BACKEND_HIP_DIRECT && !quiescent) {
          try {
            const auto drained = use_ordered ? state.drain_ordered(true)
                                             : hip_direct_synchronize(state.ctx->device_id);
            if (drained != RNS8_SUCCESS) {
              state.quarantined = true;
              state.ctx->hip_failure_poisoned = true;
              return;
            }
          } catch (...) {
            state.quarantined = true;
            state.ctx->hip_failure_poisoned = true;
            return;
          }
        }
        state.release_active();
      }
      void finish() noexcept {
        state.release_active();
        completed = true;
      }
      void reject_guarded() noexcept {
        state.output_touched = false;
        clear_packing();
        finish();
      }
    } run{state, false, false, ordered};
    if (ordered && state.ctx->synchronous_test_hooks) {
      const auto checked = synchronous_dispatch_checkpoint(*state.ctx, "product_recipe_prepare");
      if (checked != RNS8_SUCCESS)
        return checked;
    }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (ordered) {
      if (!state.ordered || !state.stream())
        return RNS8_INTERNAL_ERROR;
      state.ordered->pending = true;
      if (state.sticky()) {
        state.ordered->status.mark_pending();
        state.ordered->result.mark_pending();
        if (hipMemsetAsync(state.sticky(), 0, sizeof(int),
                           static_cast<hipStream_t>(state.stream())) != hipSuccess)
          return RNS8_BACKEND_FAILURE;
      }
    }
#endif
    const auto materialize = [&](uint32_t id, uint32_t target) -> rns8_status {
      if (current[id] >= target)
        return RNS8_SUCCESS;
      if (id < state.inputs.size() || !current[id])
        return RNS8_INVALID_ARGUMENT;
      auto* residues = arena + state.allocation.values[id].offset;
      const auto shape = state.allocation.values[id].shape;
      const int64_t cells = shape.rows * shape.cols;
      if (hip) {
        const auto* words = state.magnitude_words[id].data();
        bool touched = false;
        auto* lift_workspace = static_cast<int8_t*>(state.device) + state.allocation.arena_bytes;
        rns8_status status;
        if (ordered) {
          ++state.observations.ordered_lift_calls;
          status = hip_direct_enqueue_recipe_lift_current(
              residues, cells, current[id], target, state.semantics == RNS8_EXACT_WIDE_SIGNED,
              words, lift_workspace, state.lift_scratch,
              state.experimental ? &state.basis->packet : nullptr, state.sticky(), state.stream());
        } else if (state.experimental) {
          const auto storage = basis_lift_storage(cells, current[id], target);
          if (storage.status != RNS8_SUCCESS || storage.packet_offset > state.packet_offset)
            return RNS8_RANGE_ERROR;
          auto* packet = lift_workspace + state.packet_offset;
          BasisMagnitude bound;
          std::copy_n(words, 3, bound.words);
          status = hip_direct_basis_convert(
              state.ctx->device_id, residues, lift_workspace, cells, packet, packet, current[id],
              target, state.semantics == RNS8_EXACT_WIDE_SIGNED, bound,
              lift_workspace + storage.flag_offset, &state.lift_status);
          if (status == RNS8_SUCCESS)
            status = hip_direct_copy_device_to_device(
                state.ctx->device_id, residues + uint64_t(cells) * current[id],
                lift_workspace + uint64_t(cells) * current[id],
                size_t(uint64_t(cells) * (target - current[id])));
        } else
          status = hip_direct_lift_exact_suffix_workspace(
              state.ctx->device_id, residues, cells, current[id], target,
              state.semantics == RNS8_EXACT_WIDE_SIGNED, words, lift_workspace, state.lift_scratch,
              touched);
        if (status != RNS8_SUCCESS)
          return status;
      } else {
        std::array<int8_t, 20> cell_residues{};
        const auto& product = state.basis->products[current[id]];
        const cpp_int threshold = (product + 1) / 2;
        for (int64_t cell = 0; cell < cells; ++cell) {
          for (uint32_t p = 0; p < current[id]; ++p) {
            const int value = residues[int64_t(p) * cells + cell],
                      modulus = state.basis->packet.moduli[p];
            if (value < -(modulus / 2) || value > (modulus - 1) / 2)
              return RNS8_INVALID_ARGUMENT;
            cell_residues[p] = static_cast<int8_t>(value);
          }
          cpp_int value = 0;
          for (uint32_t p = 0; p < current[id]; ++p) {
            const int modulus = state.basis->packet.moduli[p];
            int difference =
                (int(cell_residues[p]) - (value % modulus).convert_to<int>()) % modulus;
            if (difference < 0)
              difference += modulus;
            value += state.basis->products[p] *
                     ((difference * state.basis->packet.inverse_prefix[p]) % modulus);
          }
          if (state.semantics == RNS8_EXACT_WIDE_SIGNED && value >= threshold)
            value -= product;
          if (boost::multiprecision::abs(value) > state.bounds[id])
            return RNS8_RANGE_ERROR;
          for (uint32_t p = current[id]; p < target; ++p)
            residues[int64_t(p) * cells + cell] =
                centered_residue(value, state.basis->packet.moduli[p]);
        }
      }
      current[id] = target;
      return RNS8_SUCCESS;
    };
    for (size_t i = 0; i < state.nodes.size(); ++i) {
      const auto node = state.nodes[i];
      const auto prefix = state.compute_prefixes[i];
      const bool packed =
          state.materializations.status == RNS8_SUCCESS && state.materializations.packed_b[i];
      for (auto id : {node.a, node.b}) {
        if (id == node.b && id != node.a && packed && packed_current[id] >= prefix)
          continue;
        const auto status = materialize(id, prefix);
        if (status != RNS8_SUCCESS)
          return status;
      }
      const auto a = state.allocation.values[node.a].shape,
                 b = state.allocation.values[node.b].shape;
      auto* c = arena + state.allocation.values[state.inputs.size() + i].offset;
      const auto* av = pointer(node.a);
      const auto* bv = pointer(node.b);
      const auto layout = [&](uint32_t id, ResidueLayout& value) {
        if (id >= state.layouts.size())
          return false;
        value = state.layouts[id];
        return true;
      };
      if (packed) {
        const auto& view = state.materializations.packed[node.b];
        auto* packed_b = arena + view.offset;
        ResidueLayout source, al, cl;
        if (!layout(node.b, source) || !layout(node.a, al) ||
            !layout(static_cast<uint32_t>(state.inputs.size() + i), cl))
          return RNS8_INTERNAL_ERROR;
        if (packed_current[node.b] < prefix) {
          if (hip) {
            if (ordered)
              ++state.observations.ordered_conversion_calls;
            const auto status =
                ordered
                    ? hip_direct_enqueue_residue_planes_current(bv, source, packed_b, view.layout,
                                                                packed_current[node.b], prefix,
                                                                state.stream(), state.sticky())
                    : hip_direct_convert_residue_planes(state.ctx->device_id, bv, source, packed_b,
                                                        view.layout, packed_current[node.b],
                                                        prefix);
            if (status != RNS8_SUCCESS)
              return status;
          } else if (!convert_residue_planes_host(bv, source, packed_b, view.layout,
                                                  packed_current[node.b], prefix))
            return RNS8_INTERNAL_ERROR;
          packed_current[node.b] = prefix;
          ++state.packing_calls;
        } else if (node.b < state.inputs.size() && state.materializations.retain_external_b)
          ++state.reused_packing_calls;
        if (hip) {
          if (ordered)
            ++state.observations.ordered_gemm_calls;
          const auto status =
              ordered
                  ? hip_direct_enqueue_residue_layout_gemm_current(av, al, packed_b, view.layout, c,
                                                                   cl, prefix, state.stream(),
                                                                   state.sticky())
                  : hip_direct_gemm_residue_layout(state.ctx->device_id, av, al, packed_b,
                                                   view.layout, c, cl, prefix);
          if (status != RNS8_SUCCESS)
            return status;
        } else if (!residue_layout_gemm_host(av, al, packed_b, view.layout, c, cl, prefix))
          return RNS8_INTERNAL_ERROR;
        current[state.inputs.size() + i] = prefix;
        continue;
      }
      if (hip) {
        rns8_status status;
        if (ordered) {
          ++state.observations.ordered_gemm_calls;
          if (state.sticky()) {
            ResidueLayout al, bl, cl;
            if (!layout(node.a, al) || !layout(node.b, bl) ||
                !layout(static_cast<uint32_t>(state.inputs.size() + i), cl))
              return RNS8_INTERNAL_ERROR;
            status = hip_direct_enqueue_residue_layout_gemm_current(av, al, bv, bl, c, cl, prefix,
                                                                    state.stream(), state.sticky());
          } else
            status = hip_direct_enqueue_recipe_gemm_current(
                av, bv, c, a.rows, b.cols, a.cols, prefix, state.stream(),
                state.experimental ? state.basis->packet.moduli : nullptr);
        } else
          status = state.experimental
                       ? hip_direct_basis_gemm(state.ctx->device_id, av, bv, c, a.rows, b.cols,
                                               a.cols, state.basis->packet.moduli, prefix)
                       : hip_direct_gemm_rns_device(state.ctx->device_id, av, bv, c, a.rows, b.cols,
                                                    a.cols, a.cols, b.cols, b.cols, prefix);
        if (status != RNS8_SUCCESS)
          return status;
      } else {
        ResidueLayout al, bl, cl;
        if (!layout(node.a, al) || !layout(node.b, bl) ||
            !layout(static_cast<uint32_t>(state.inputs.size() + i), cl) ||
            !residue_layout_gemm_host(av, al, bv, bl, c, cl, prefix))
          return RNS8_INTERNAL_ERROR;
      }
      current[state.inputs.size() + i] = prefix;
    }
    // Finish all private materializations before starting any public write.
    for (auto id : state.outputs) {
      const auto completed = materialize(id, state.prefix);
      if (completed != RNS8_SUCCESS)
        return completed;
    }
    if (ordered && state.ctx->synchronous_test_hooks) {
      const auto checked = synchronous_dispatch_checkpoint(*state.ctx, "product_recipe_compute");
      if (checked != RNS8_SUCCESS)
        return checked;
    }
    state.output_touched = true;
    // Guarded publication leaves old bytes/metadata intact on logical rejection.
    // Writer reservations prohibit every data consumer until final observation.
    if (!ordered || !state.sticky())
      state.invalidate_touched();
    for (size_t i = 0; i < count; ++i) {
      auto* output = outputs[i];
      const auto& value = state.allocation.values[state.outputs[i]];
      const auto* result = arena + value.offset;
      if (hip) {
        if (ordered)
          ++state.observations.ordered_copy_calls;
        const auto status =
            ordered ? hip_direct_enqueue_recipe_copy_current(
                          result, output->hip_residues, value.bytes, state.stream(), state.sticky())
                    : hip_direct_copy_device_to_device(state.ctx->device_id, output->hip_residues,
                                                       result, static_cast<size_t>(value.bytes));
        if (status != RNS8_SUCCESS)
          return status;
      } else
        std::copy_n(result, static_cast<size_t>(value.bytes), output->residues.data());
    }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (ordered) {
      if (state.ctx->synchronous_test_hooks) {
        const auto checked =
            synchronous_dispatch_checkpoint(*state.ctx, "product_recipe_publication");
        if (checked != RNS8_SUCCESS)
          return checked;
      }
      if (state.sticky()) {
        if (hipMemcpyAsync(state.ordered->result.get(), state.sticky(), sizeof(int),
                           hipMemcpyDeviceToHost,
                           static_cast<hipStream_t>(state.stream())) != hipSuccess)
          return RNS8_BACKEND_FAILURE;
        state.observations.ordered_status_readback_bytes += sizeof(int);
      }
      const auto waited = run.wait();
      if (waited != RNS8_SUCCESS)
        return waited;
      if (state.sticky()) {
        const int flag = *static_cast<const int*>(state.ordered->result.get());
        if (flag) {
          run.reject_guarded();
          if (flag == int(RNS8_INVALID_ARGUMENT))
            return RNS8_INVALID_ARGUMENT;
          if (flag == int(RNS8_RANGE_ERROR))
            return RNS8_RANGE_ERROR;
          state.ctx->hip_failure_poisoned = true;
          return RNS8_BACKEND_FAILURE;
        }
      }
    }
#endif
    // Publish only after every copy succeeds. Any failed publication invalidates
    // the entire output batch, including copies which had already completed.
    for (size_t i = 0; i < count; ++i) {
      auto* output = outputs[i];
      if (hip)
        mark_output_device_residues_current(*output);
      else
        mark_output_host_residues_current(*output);
      output->exact_max_magnitude.swap(bounds[i]);
      output->exact_axis_ranges.rows.clear();
      output->exact_axis_ranges.cols.clear();
      output->exact_range_prefix = state.prefix;
    }
    ++state.replays;
    run.finish();
    return RNS8_SUCCESS;
  });
}
} // namespace rns8::detail

struct rns8_product_recipe {
  std::unique_ptr<rns8::detail::RuntimeProductRecipe> owner;
  rns8_context* ctx = nullptr;
  std::vector<const rns8_matrix*> inputs;
  std::vector<rns8_ragged_recipe_binding> ragged;
  std::vector<rns8_ragged_materialization*> ragged_holds;
  uint64_t total_budget = UINT64_MAX, materialized_input_bytes = 0, ragged_bytes = 0,
           materialization_peak = 0;
  std::string materialization_calibration;
  bool binding_valid = true, closing = false;
  ~rns8_product_recipe() {
    owner.reset();
    for (auto* source : ragged_holds)
      rns8::detail::release_ragged_recipe_binding(source);
  }
};
#include "api_product_recipe_ragged_helpers.inc"

static rns8_status create_product_recipe(
    rns8_context* ctx, const rns8_product_recipe_desc* desc, const std::vector<uint32_t>& outputs,
    rns8_product_recipe** out,
    const rns8::detail::ProductMaterializationProfile* materializations = nullptr) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if (!ctx || !desc || !out || !minimum_abi_header(desc) || !desc->inputs || !desc->nodes ||
        !desc->input_count || desc->input_count > 4096 || !desc->node_count ||
        desc->node_count > 4096 ||
        (desc->prefix_policy != RNS8_PRODUCT_FULL_PREFIX &&
         desc->prefix_policy != RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX))
      return RNS8_INVALID_ARGUMENT;
    std::vector<const rns8_matrix*> inputs(desc->inputs, desc->inputs + desc->input_count);
    std::vector<ProductNode> nodes;
    nodes.reserve(desc->node_count);
    for (uint32_t i = 0; i < desc->node_count; ++i)
      nodes.push_back({desc->nodes[i].a, desc->nodes[i].b});
    auto result = std::make_unique<rns8_product_recipe>();
    result->ctx = ctx;
    result->inputs = inputs;
    const auto status = RuntimeProductRecipe::create(
        ctx, inputs, nodes, desc->max_arena_and_lift_bytes, result->owner,
        desc->prefix_policy == RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX, nullptr, outputs,
        materializations);
    if (status != RNS8_SUCCESS)
      return status;
    if (materializations)
      result->materialization_calibration = materializations->uniform.provenance;
    *out = result.release();
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_create_product_recipe(rns8_context* ctx, const rns8_product_recipe_desc* desc,
                                       rns8_product_recipe** out) {
  return create_product_recipe(ctx, desc, {}, out);
}
rns8_status rns8_create_product_recipe_outputs(rns8_context* ctx,
                                               const rns8_product_recipe_desc* desc,
                                               const uint32_t* ids, uint32_t count,
                                               rns8_product_recipe** out) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!ids || !count || count > 4096)
      return RNS8_INVALID_ARGUMENT;
    return create_product_recipe(ctx, desc, std::vector<uint32_t>(ids, ids + count), out);
  });
}
rns8_status rns8_execute_product_recipe_outputs(rns8_product_recipe* recipe,
                                                rns8_matrix* const* outputs, uint32_t count) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!recipe || !outputs || !count || count > 4096)
      return RNS8_INVALID_ARGUMENT;
    const auto binding = validate_ragged_recipe_binding(*recipe);
    if (binding != RNS8_SUCCESS)
      return binding;
    return recipe->owner->execute(outputs, count);
  });
}
rns8_status rns8_execute_product_recipe(rns8_product_recipe* recipe, rns8_matrix* output) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!recipe)
      return RNS8_INVALID_ARGUMENT;
    const auto binding = validate_ragged_recipe_binding(*recipe);
    return binding == RNS8_SUCCESS ? recipe->owner->execute(output) : binding;
  });
}
rns8_status rns8_set_product_recipe_execution_policy(rns8_product_recipe* recipe,
                                                     rns8_product_execution_policy policy) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!recipe || recipe->closing || uint32_t(policy) > 1)
      return RNS8_INVALID_ARGUMENT;
    const auto extra = policy == RNS8_PRODUCT_EXECUTION_ORDERED_HIP
                           ? recipe->owner->additional_ordered_control_bytes()
                           : 0;
    const auto binding = validate_ragged_recipe_binding(*recipe, extra);
    if (binding != RNS8_SUCCESS)
      return binding;
    return recipe->owner->set_execution_policy(policy);
  });
}
rns8_status rns8_get_product_recipe_execution_info(const rns8_product_recipe* recipe,
                                                   rns8_product_recipe_execution_info* out) {
  if (!recipe || recipe->closing || !out || !rns8::detail::minimum_abi_header(out))
    return RNS8_INVALID_ARGUMENT;
  *out = recipe->owner->execution_info();
  return RNS8_SUCCESS;
}
rns8_status rns8_get_product_recipe_resources(const rns8_product_recipe* recipe,
                                              rns8_product_recipe_resources* out) {
  if (!recipe || !out || !rns8::detail::minimum_abi_header(out))
    return RNS8_INVALID_ARGUMENT;
  const auto& allocation = recipe->owner->allocation();
  out->node_count = static_cast<uint32_t>(recipe->owner->compute_prefixes().size());
  out->arena_bytes = allocation.arena_bytes;
  out->lift_scratch_bytes = recipe->owner->lift_scratch_bytes();
  out->unreused_payload_bytes = allocation.unreused_bytes;
  out->materialized_input_bytes = recipe->materialized_input_bytes;
  out->ragged_retained_bytes = recipe->ragged_bytes;
  out->materialization_peak_bytes = recipe->materialization_peak;
  return RNS8_SUCCESS;
}
rns8_status rns8_get_product_recipe_prefixes(const rns8_product_recipe* recipe, uint32_t* prefixes,
                                             uint32_t capacity) {
  if (!recipe || !prefixes)
    return RNS8_INVALID_ARGUMENT;
  const auto& selected = recipe->owner->compute_prefixes();
  if (capacity < selected.size())
    return RNS8_WORKSPACE_TOO_SMALL;
  std::copy(selected.begin(), selected.end(), prefixes);
  return RNS8_SUCCESS;
}
rns8_status rns8_get_product_recipe_basis(const rns8_product_recipe* recipe, rns8_basis_info* out) {
  if (!recipe || !out || !rns8::detail::minimum_abi_header(out))
    return RNS8_INVALID_ARGUMENT;
  const auto& packet = recipe->owner->basis_packet();
  const auto prefix = recipe->owner->output_prefix();
  rns8_basis_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  info.representation = packet.version;
  info.identity = packet.identity;
  info.count = packet.count;
  info.selected_prefix = prefix;
  std::copy_n(packet.moduli, 20, info.moduli);
  std::copy_n(packet.products[prefix], 3, info.prefix_product);
  *out = info;
  return RNS8_SUCCESS;
}
rns8_status rns8_destroy_product_recipe(rns8_product_recipe* recipe) {
  if (!recipe)
    return RNS8_SUCCESS;
  const auto status = recipe->owner->close();
  if (status != RNS8_BUSY)
    recipe->closing = true;
  if (status != RNS8_SUCCESS)
    return status;
  delete recipe;
  return RNS8_SUCCESS;
}

rns8_status rns8_rebind_product_recipe_inputs(rns8_product_recipe* recipe,
                                              const rns8_matrix* const* inputs, uint32_t count) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if (!recipe || !inputs || !count || count > 4096)
      return RNS8_INVALID_ARGUMENT;
    if (recipe->closing || !recipe->ragged.empty())
      return RNS8_INVALID_ARGUMENT;
    std::vector<const rns8_matrix*> replacement(inputs, inputs + count);
    const auto status = recipe->owner->rebind(replacement);
    if (status == RNS8_SUCCESS)
      recipe->inputs.swap(replacement);
    return status;
  });
}

#include "api_product_recipe_ragged.inc"
#include "api_product_recipe_materializations.inc"
