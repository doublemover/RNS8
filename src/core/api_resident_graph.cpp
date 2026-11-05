#include "core/api_internal.hpp"
#include "core/resident_native_import.hpp"
#include "backend_common/complex_residue_plan.hpp"
#include <array>
#include <cstring>
#include "backend_common/resident_transfer_plan.hpp"
#include <atomic>
#include "core/hip_resources.hpp"
#include <unordered_map>
#include <unordered_set>

using namespace rns8::detail::api;
struct rns8_resident_graph {
  struct OutputProof {
    boost::multiprecision::cpp_int bound;
    rns8_exact_matrix_ranges ranges;
    boost::multiprecision::cpp_int imag_bound;
    rns8::detail::ComplexResidueScratch complex_scratch;
    uint32_t source_prefix = 0, target_prefix = 0;
  };
  struct Input {
    const rns8_matrix* matrix;
    uint64_t instance, generation;
  };
  struct ProofValue : rns8::detail::ExactRangeValue {
    uint32_t materialized_rns_prefix = 0;
    uint16_t finite_modulus = 0;
    uint64_t expected_instance = 0, expected_generation = 0;
  };
  struct MathProofPlan {
    struct PlanKey {
      const rns8_plan* plan = nullptr;
      std::shared_ptr<const rns8_validated_plan_contract> contract;
      uint32_t prefix = 0, flags = 0;
      boost::multiprecision::cpp_int product;
    };
    std::vector<Input> inputs;
    std::vector<PlanKey> plans;
    std::vector<OutputProof> outputs;
  };
  rns8_context* ctx = nullptr;
  std::vector<rns8_resident_node> nodes;
  std::vector<std::unique_ptr<rns8::detail::api::ResidentNativeImport>> native_imports;
  std::vector<rns8::detail::api::ResidentNativeImport*> native_readers;
  std::vector<Input> inputs;
  std::vector<uint64_t*> holds;
  uint64_t epoch = 0;
  bool running = false;
  bool cancel_requested = false;
  uint64_t id = 0;
  bool hip_prepared = false;
  bool retiring = false;
  void* pending_stream = nullptr;
  std::vector<rns8_workspace*> hip_workspaces;
  std::vector<rns8_matrix*> hip_outputs;
  std::vector<OutputProof> pending_proofs;
  mutable std::vector<OutputProof> proof_staging;
  mutable std::unordered_map<const rns8_matrix*, ProofValue> proof_metadata;
  std::shared_ptr<const MathProofPlan> math_proof;
  mutable std::shared_ptr<const MathProofPlan> staged_math_proof;
  uint64_t math_proof_builds = 0, math_proof_reuses = 0;
  rns8_status execute_proven_exact_node(uint32_t index) {
    return rns8::detail::api::guard_api([&]() -> rns8_status {
      if (!running || !math_proof || index >= nodes.size() || index >= math_proof->outputs.size())
        return RNS8_INTERNAL_ERROR;
      const auto& node = nodes[index];
      const bool epilogue = node.kind == RNS8_RESIDENT_EXACT_AXPBY;
      rns8::detail::ExactOperationProof proof(*node.plan, *node.a, *node.b,
                                              epilogue ? node.d : nullptr,
                                              epilogue ? node.alpha : 0, epilogue ? node.beta : 0);
      proof.bound_view_ = &math_proof->outputs[index].bound;
      proof.ranges_view_ = &math_proof->outputs[index].ranges;
      return epilogue ? rns8::detail::api::gemm_axpby_with_proof(ctx, node.plan, node.a, node.b,
                                                                 node.d, node.c, node.alpha,
                                                                 node.beta, node.workspace, &proof)
                      : rns8::detail::api::gemm_rns_with_proof(ctx, node.plan, node.a, node.b,
                                                               node.c, node.workspace, &proof);
    });
  }
  rns8_graph_execution_info completion{};
  rns8_async_failure_info async_failure{};
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8::detail::hip_unique_event completion_event;
  rns8::detail::hip_unique_stream synchronous_stream;
  rns8::detail::hip_direct_device_buffer complex_storage;
  std::vector<rns8::detail::ComplexResidueScratch> complex_resources;
  struct Transfer {
    rns8::detail::ResidentTransferLayout layout;
    uint32_t source = 0, target = 0;
    boost::multiprecision::cpp_int envelope;
    std::array<uint64_t, 3> bound{};
    rns8::detail::hip_pinned_host_buffer host;
  };
  std::vector<Transfer> transfers;
  hipGraph_t captured_graph = nullptr;
  hipGraphExec_t captured_executable = nullptr;
  std::vector<std::array<const void*, 12>> captured_addresses;
  rns8::detail::hip_unique_graph captured_graph_owner;
  rns8::detail::hip_unique_graph_exec captured_executable_owner;

#endif
};

static rns8_status submit_resident_graph_hip(rns8_resident_graph*, void*,
                                             rns8_graph_completion_token*, bool,
                                             uint32_t* failed_preflight = nullptr);

rns8::detail::api::PreparedProofObservations
rns8::detail::api::resident_graph_proof_observations(const rns8_resident_graph* graph) noexcept {
  if (!graph)
    return {};
  PreparedProofObservations out{graph->math_proof_builds, graph->math_proof_reuses};
  if (!graph->math_proof)
    return out;
  const auto& proof = *graph->math_proof;
  out.retained_object_bytes =
      sizeof(proof) + proof.inputs.capacity() * sizeof(rns8_resident_graph::Input) +
      proof.plans.capacity() * sizeof(rns8_resident_graph::MathProofPlan::PlanKey) +
      proof.outputs.capacity() * sizeof(rns8_resident_graph::OutputProof);
  using rns8::detail::exact_proof_limb_bytes;
  using rns8::detail::exact_proof_range_memory;
  for (const auto& key : proof.plans)
    out.logical_limb_bytes += exact_proof_limb_bytes(key.product);
  const auto count_output = [&](const auto& output) {
    out.logical_limb_bytes +=
        exact_proof_limb_bytes(output.bound) + exact_proof_limb_bytes(output.imag_bound);
    exact_proof_range_memory(output.ranges, out.retained_object_bytes, out.logical_limb_bytes);
  };
  for (const auto& output : proof.outputs)
    count_output(output);
  // Retained staging/map nodes are metadata only, never matrix payloads.
  out.retained_object_bytes +=
      graph->proof_staging.capacity() * sizeof(rns8_resident_graph::OutputProof) +
      graph->pending_proofs.capacity() * sizeof(rns8_resident_graph::OutputProof) +
      graph->proof_metadata.size() * sizeof(decltype(graph->proof_metadata)::value_type) +
      graph->proof_metadata.bucket_count() * sizeof(void*);
  for (const auto& output : graph->proof_staging)
    count_output(output);
  for (const auto& output : graph->pending_proofs)
    count_output(output);
  for (const auto& entry : graph->proof_metadata) {
    out.logical_limb_bytes += exact_proof_limb_bytes(entry.second.exact_max_magnitude);
    exact_proof_range_memory(entry.second.exact_axis_ranges, out.retained_object_bytes,
                             out.logical_limb_bytes);
  }
  return out;
}

namespace {
struct GraphMathStaging {
  rns8_resident_graph& graph;
  ~GraphMathStaging() { graph.staged_math_proof.reset(); }
};
bool graph_math_proof_matches(const rns8_resident_graph& graph) {
  const auto& proof = graph.math_proof;
  if (!proof || proof->inputs.size() != graph.inputs.size() ||
      proof->plans.size() != graph.nodes.size() || proof->outputs.size() != graph.nodes.size())
    return false;
  for (size_t i = 0; i < graph.inputs.size(); ++i) {
    const auto& input = proof->inputs[i];
    const auto& matrix = *graph.inputs[i].matrix;
    if (input.matrix != &matrix || input.instance != matrix.matrix_instance_id ||
        input.generation != matrix.content_generation)
      return false;
  }
  for (size_t i = 0; i < graph.nodes.size(); ++i) {
    const auto* plan = graph.nodes[i].plan;
    const auto& key = proof->plans[i];
    if (key.plan != plan ||
        (plan && (key.contract != plan->validated_contract || key.prefix != plan->prefix ||
                  key.flags != plan->desc.flags || key.product != plan->modulus_product)))
      return false;
  }
  return true;
}
void stage_graph_math_proof(const rns8_resident_graph& graph,
                            const std::vector<rns8_resident_graph::OutputProof>& outputs) {
  auto proof = std::make_shared<rns8_resident_graph::MathProofPlan>();
  proof->inputs.reserve(graph.inputs.size());
  proof->plans.reserve(graph.nodes.size());
  for (const auto& input : graph.inputs)
    proof->inputs.push_back(
        {input.matrix, input.matrix->matrix_instance_id, input.matrix->content_generation});
  for (const auto& node : graph.nodes) {
    rns8_resident_graph::MathProofPlan::PlanKey key;
    key.plan = node.plan;
    if (node.plan) {
      key.contract = node.plan->validated_contract;
      key.prefix = node.plan->prefix;
      key.flags = node.plan->desc.flags;
      key.product = node.plan->modulus_product;
    }
    proof->plans.push_back(std::move(key));
  }
  proof->outputs = outputs;
  graph.staged_math_proof = std::move(proof);
}
void commit_graph_math_proof(rns8_resident_graph& graph) noexcept {
  if (graph.staged_math_proof) {
    graph.math_proof.swap(graph.staged_math_proof);
    graph.staged_math_proof.reset();
    if (graph.math_proof_builds != UINT64_MAX)
      ++graph.math_proof_builds;
  } else if (graph.math_proof_reuses != UINT64_MAX)
    ++graph.math_proof_reuses;
}
rns8_status
validate_resident_transfer_envelopes(const rns8_resident_graph& graph,
                                     const std::vector<rns8_resident_graph::OutputProof>& proofs);
bool bounded_graph_semantics(rns8_semantics semantics) {
  return semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64;
}
uint64_t next_graph_id() {
  static std::atomic<uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  while (value != UINT64_MAX)
    if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
      return value;
  return 0;
}
std::array<const rns8_matrix*, 4> node_inputs(const rns8_resident_node& node) {
  if (node.kind == RNS8_RESIDENT_NATIVE_TO_RNS)
    return {};
  if (node.kind == RNS8_RESIDENT_COMPLEX_MATMUL)
    return {node.a, node.a_imag, node.b, node.b_imag};
  if (node.kind == RNS8_RESIDENT_BASE_EXTEND || node.kind == RNS8_RESIDENT_EXPORT_LIMBS)
    return {node.c, nullptr, nullptr, nullptr};
  return {node.a, node.b, node.d, nullptr};
}
rns8_status cpu_graph_node_inputs_current(const rns8_resident_graph& graph,
                                          const rns8_resident_node& node) {
  const auto current = [&](const rns8_matrix* matrix) -> rns8_status {
    if (!matrix)
      return RNS8_SUCCESS;
    if (matrix->generation_exhausted)
      return RNS8_RANGE_ERROR;
    const auto found = graph.proof_metadata.find(matrix);
    return found != graph.proof_metadata.end() &&
                   found->second.expected_instance == matrix->matrix_instance_id &&
                   found->second.expected_generation == matrix->content_generation
               ? RNS8_SUCCESS
               : RNS8_INVALID_ARGUMENT;
  };
  if (node.kind == RNS8_RESIDENT_BASE_EXTEND || node.kind == RNS8_RESIDENT_EXPORT_LIMBS)
    return current(node.c);
  for (const auto* input : node_inputs(node)) {
    const auto status = current(input);
    if (status != RNS8_SUCCESS)
      return status;
  }
  return RNS8_SUCCESS;
}
void bind_cpu_graph_node_outputs(rns8_resident_graph& graph,
                                 const rns8_resident_node& node) noexcept {
  if (node.kind == RNS8_RESIDENT_EXPORT_LIMBS)
    return;
  for (const auto* output : {node.c, node.c_imag})
    if (output) {
      const auto found = graph.proof_metadata.find(output);
      if (found != graph.proof_metadata.end()) {
        found->second.expected_instance = output->matrix_instance_id;
        found->second.expected_generation = output->content_generation;
      }
    }
}

rns8_status export_span(const rns8_resident_node& node, uint64_t& bytes) {
  if (!node.export_limbs || !node.plan ||
      !valid_limb_export_access(node.plan->desc.m, node.plan->desc.n, node.export_ld,
                                node.export_limb_count) ||
      reinterpret_cast<uintptr_t>(node.export_limbs) % alignof(uint64_t))
    return RNS8_INVALID_ARGUMENT;
  rns8::detail::AddressDomain domain;
  if (!rns8::detail::checked_host_plane_address(node.plan->desc.m, node.plan->desc.n,
                                                node.export_ld, 1,
                                                uint64_t{8} * node.export_limb_count, domain))
    return RNS8_RANGE_ERROR;
  bytes = domain.byte_extent;
  if (reinterpret_cast<uintptr_t>(node.export_limbs) > UINTPTR_MAX - bytes)
    return RNS8_RANGE_ERROR;
  return RNS8_SUCCESS;
}

rns8_status prove_export(const rns8_context& ctx, const rns8_resident_node& node,
                         const rns8_resident_graph::ProofValue& proof) {
  if (node.c->in_flight_writer)
    return RNS8_BUSY;
  const auto status = validate_export_matrix(ctx, *node.plan, *node.c, node.plan->desc.semantics,
                                             RNS8_BOUND_NONE, node.plan->prefix, false);
  if (status != RNS8_SUCCESS)
    return status;
  if (proof.exact_range_prefix < node.plan->prefix ||
      !rns8::detail::exact_range_fits(node.plan->desc.semantics, proof.exact_max_magnitude,
                                      node.plan->modulus_product))
    return RNS8_RANGE_ERROR;
  const unsigned bits =
      64 * node.export_limb_count - (node.plan->desc.semantics == RNS8_EXACT_WIDE_SIGNED ? 1 : 0);
  // Symmetric magnitude proof deliberately does not infer the sign of a value
  // at the signed minimum. Full-width exports remain available with more limbs.
  if (proof.exact_max_magnitude >= (boost::multiprecision::cpp_int(1) << bits))
    return RNS8_RANGE_ERROR;
  return RNS8_SUCCESS;
}

rns8_status prove_extension(const rns8_context& ctx, const rns8_resident_node& node,
                            rns8_resident_graph::ProofValue& proof) {
  const auto& matrix = *node.c;
  if (matrix.in_flight_writer || matrix.in_flight_readers)
    return RNS8_BUSY;
  if (matrix.backend != ctx.backend || !rns8::detail::exact_wide_semantics(matrix.desc.semantics) ||
      !node.target_prefix || node.target_prefix > matrix.prefix ||
      node.target_prefix > RNS8_MAX_SUPPORTED_PREFIX || matrix.desc.max_prefix != matrix.prefix ||
      rns8::detail::validate_matrix_desc(matrix.desc, matrix.prefix) != RNS8_SUCCESS ||
      !rns_matrix_storage_matches(matrix, ctx.backend, matrix.desc.rows, matrix.desc.cols,
                                  matrix.prefix) ||
      (ctx.backend == RNS8_BACKEND_HIP_DIRECT && matrix.hip_device_id != ctx.device_id))
    return RNS8_INVALID_ARGUMENT;
  if (!proof.exact_range_prefix || proof.exact_range_prefix > matrix.prefix ||
      !rns8::detail::exact_range_fits(matrix.desc.semantics, proof.exact_max_magnitude,
                                      rns8::detail::modulus_product(proof.exact_range_prefix)))
    return RNS8_RANGE_ERROR;
  if (node.target_prefix <= proof.exact_range_prefix)
    return RNS8_SUCCESS;
  std::size_t cells = 0;
  const auto suffix = node.target_prefix - proof.exact_range_prefix;
  if (!matrix_cell_count(matrix.desc.rows, matrix.desc.cols, cells) || cells > SIZE_MAX / suffix ||
      (ctx.backend == RNS8_BACKEND_HIP_DIRECT && cells > INT32_MAX - 256))
    return RNS8_RANGE_ERROR;
  if (cells * suffix > node.max_staged_residue_bytes)
    return RNS8_WORKSPACE_TOO_SMALL;
  proof.exact_range_prefix = node.target_prefix;
  return RNS8_SUCCESS;
}

// Metadata-only proof values: no copies of host/device matrix payloads.
rns8_status prove_graph(const rns8_resident_graph& graph, uint32_t& failed_node,
                        bool check_bindings,
                        std::vector<rns8_resident_graph::OutputProof>* pending = nullptr) {
  if (graph.retiring)
    return RNS8_BUSY;
  graph.staged_math_proof.reset();
  const bool reuse = check_bindings && graph_math_proof_matches(graph);
  if (!pending)
    pending = &graph.proof_staging;
  if (reuse)
    *pending = graph.math_proof->outputs;
  else
    pending->resize(graph.nodes.size());
  auto& proofs = graph.proof_metadata;
  proofs.reserve(graph.inputs.size() + 2 * graph.nodes.size());
  for (const auto& input : graph.inputs) {
    const auto& m = *input.matrix;
    if (m.in_flight_writer)
      return RNS8_BUSY;
    if (m.generation_exhausted ||
        !(m.desc.semantics == RNS8_WRAP_U64_MOD_2_64
              ? wrap_byte_limb_state_current_for_backend(m, graph.ctx->backend)
              : rns_residue_state_current_for_backend(m, graph.ctx->backend)) ||
        (check_bindings &&
         (m.matrix_instance_id != input.instance || m.content_generation != input.generation)))
      return RNS8_INVALID_ARGUMENT;
    auto& proof = proofs.try_emplace(input.matrix).first->second;
    proof.expected_instance = m.matrix_instance_id;
    proof.expected_generation = m.content_generation;
    if (m.desc.semantics == RNS8_WRAP_U64_MOD_2_64 || bounded_graph_semantics(m.desc.semantics)) {
      proof.materialized_rns_prefix = m.materialized_rns_prefix;
      continue;
    }
    if (uses_finite_storage(m.desc.semantics)) {
      if (!rns8::detail::valid_finite_modulus_for_semantics(m.desc.semantics, m.finite_modulus))
        return RNS8_INVALID_ARGUMENT;
      proof.finite_modulus = m.finite_modulus;
      continue;
    }
    if (!m.exact_range_prefix || m.exact_range_prefix > m.prefix ||
        !rns8::detail::exact_range_fits(m.desc.semantics, m.exact_max_magnitude,
                                        rns8::detail::modulus_product(m.exact_range_prefix)))
      return RNS8_RANGE_ERROR;
    proof.exact_max_magnitude = m.exact_max_magnitude;
    if (!reuse)
      proof.exact_axis_ranges = m.exact_axis_ranges;
    proof.exact_range_prefix = m.exact_range_prefix;
  }
  for (uint32_t index = 0; index < graph.nodes.size(); ++index) {
    failed_node = index;
    const auto& node = graph.nodes[index];
    if (node.kind == RNS8_RESIDENT_NATIVE_TO_RNS) {
      rns8_resident_graph::ProofValue output;
      const auto status =
          graph.native_imports[index]->validate(check_bindings, output.exact_max_magnitude);
      if (status != RNS8_SUCCESS)
        return status;
      output.exact_range_prefix = node.target_prefix;
      if (pending) {
        auto& value = (*pending)[index];
        value.bound = output.exact_max_magnitude;
        value.target_prefix = node.target_prefix;
      }
      proofs.insert_or_assign(node.c, std::move(output));
      continue;
    }
    if (node.kind == RNS8_RESIDENT_EXPORT_LIMBS) {
      const auto input = proofs.find(node.c);
      if (input == proofs.end())
        return RNS8_INVALID_ARGUMENT;
      const auto status = prove_export(*graph.ctx, node, input->second);
      if (status != RNS8_SUCCESS)
        return status;
      if (pending)
        (*pending)[index].bound = input->second.exact_max_magnitude;
      continue;
    }
    if (node.kind == RNS8_RESIDENT_BASE_EXTEND) {
      auto input = proofs.find(node.c);
      if (input == proofs.end())
        return RNS8_INVALID_ARGUMENT;
      const auto source = input->second.exact_range_prefix;
      const auto status = prove_extension(*graph.ctx, node, input->second);
      if (status != RNS8_SUCCESS)
        return status;
      if (pending) {
        auto& value = (*pending)[index];
        value.bound = input->second.exact_max_magnitude;
        value.source_prefix = source;
        value.target_prefix = input->second.exact_range_prefix;
      }
      continue;
    }
    auto status = validate_plan_context_workspace(*graph.ctx, *node.plan, *node.workspace);
    if (status != RNS8_SUCCESS)
      return status;
    status =
        node.plan->desc.semantics == RNS8_WRAP_U64_MOD_2_64
            ? validate_wrap_gemm_operands(*graph.ctx, *node.plan, *node.a, *node.b, *node.c, false)
        : uses_finite_storage(node.plan->desc.semantics)
            ? validate_finite_gemm_operands(*graph.ctx, *node.plan,
                                            static_cast<uint16_t>(node.plan->desc.finite_modulus),
                                            *node.a, *node.b, *node.c, false)
        : node.kind == RNS8_RESIDENT_EXACT_AXPBY
            ? validate_exact_axpby_operands(*graph.ctx, *node.plan, *node.a, *node.b, *node.d,
                                            *node.c, node.alpha, node.beta, false)
            : validate_rns_gemm_operands(*graph.ctx, *node.plan, *node.a, *node.b, *node.c, false);
    if (status != RNS8_SUCCESS)
      return status;
    if (node.c->generation_exhausted)
      return RNS8_RANGE_ERROR;
    auto ai = proofs.find(node.a), bi = proofs.find(node.b);
    if (ai == proofs.end() || bi == proofs.end())
      return RNS8_INVALID_ARGUMENT;
    if (node.plan->desc.semantics == RNS8_WRAP_U64_MOD_2_64 ||
        bounded_graph_semantics(node.plan->desc.semantics)) {
      if (node.kind != RNS8_RESIDENT_MATMUL)
        return RNS8_INVALID_ARGUMENT;
      // Internal prefix metadata describes written planes, not an integer proof.
      if (bounded_graph_semantics(node.plan->desc.semantics) &&
          (ai->second.materialized_rns_prefix < rns_storage_prefix_for_plan(*node.plan) ||
           bi->second.materialized_rns_prefix < rns_storage_prefix_for_plan(*node.plan)))
        return RNS8_RANGE_ERROR;
      auto& output = proofs.try_emplace(node.c).first->second;
      output.materialized_rns_prefix = node.plan->tile_schedule.empty()
                                           ? node.plan->prefix
                                           : node.plan->schedule_min_selected_prefix;
      if (pending)
        (*pending)[index] = {};
      continue;
    }
    if (uses_finite_storage(node.plan->desc.semantics)) {
      if (node.kind != RNS8_RESIDENT_MATMUL ||
          ai->second.finite_modulus != node.plan->desc.finite_modulus ||
          bi->second.finite_modulus != node.plan->desc.finite_modulus)
        return RNS8_INVALID_ARGUMENT;
      proofs.try_emplace(node.c).first->second.finite_modulus =
          static_cast<uint16_t>(node.plan->desc.finite_modulus);
      if (pending)
        (*pending)[index] = {};
      continue;
    }
    if (node.kind == RNS8_RESIDENT_COMPLEX_MATMUL) {
      status = validate_rns_gemm_operands(*graph.ctx, *node.plan, *node.a_imag, *node.b_imag,
                                          *node.c_imag, false);
      if (status != RNS8_SUCCESS)
        return status;
      if (node.c_imag->generation_exhausted)
        return RNS8_RANGE_ERROR;
      const auto aii = proofs.find(node.a_imag), bii = proofs.find(node.b_imag);
      if (aii == proofs.end() || bii == proofs.end())
        return RNS8_INVALID_ARGUMENT;
      for (const auto* proof : {&ai->second, &bi->second, &aii->second, &bii->second})
        if (proof->exact_range_prefix < node.plan->prefix)
          return RNS8_RANGE_ERROR;
      using namespace rns8::detail;
      const ComplexResidueBasis basis{ComplexResidueDomain::wide_signed, 1, node.plan->prefix, 0};
      const bool hip = graph.ctx->backend == RNS8_BACKEND_HIP_DIRECT;
      const auto view = [&](const rns8_matrix* matrix) {
        return ComplexResidueView<const int8_t>{
            hip ? static_cast<const int8_t*>(matrix->hip_residues) : matrix->residues.data(),
            matrix->desc.rows, matrix->desc.cols, matrix->desc.cols, basis};
      };
      const auto destination = [&](rns8_matrix* matrix) {
        return ComplexResidueView<int8_t>{
            hip ? static_cast<int8_t*>(matrix->hip_residues) : matrix->residues.data(),
            matrix->desc.rows, matrix->desc.cols, matrix->desc.cols, basis};
      };
      auto admitted = plan_complex_residue_product(
          {view(node.a), view(node.a_imag)}, {view(node.b), view(node.b_imag)},
          {destination(node.c), destination(node.c_imag)},
          {ai->second.exact_max_magnitude, aii->second.exact_max_magnitude,
           bi->second.exact_max_magnitude, bii->second.exact_max_magnitude},
          node.complex_method == RNS8_COMPLEX_THREE_REAL_PRODUCTS ? ComplexProductMethod::three_real
                                                                  : ComplexProductMethod::four_real,
          node.max_staged_residue_bytes);
      if (admitted.status != RNS8_SUCCESS)
        return admitted.status;
      if (pending)
        (*pending)[index] = {admitted.real_bound,
                             {},
                             admitted.imag_bound,
                             static_cast<const ComplexResidueScratch&>(admitted)};
      rns8_resident_graph::ProofValue real, imag;
      real.exact_range_prefix = imag.exact_range_prefix = node.plan->prefix;
      real.exact_max_magnitude.swap(admitted.real_bound);
      imag.exact_max_magnitude.swap(admitted.imag_bound);
      proofs.insert_or_assign(node.c, std::move(real));
      proofs.insert_or_assign(node.c_imag, std::move(imag));
      continue;
    }
    if (reuse) {
      if (node.kind == RNS8_RESIDENT_EXACT_AXPBY && proofs.find(node.d) == proofs.end())
        return RNS8_INVALID_ARGUMENT;
      if (ai->second.exact_range_prefix < node.plan->prefix ||
          bi->second.exact_range_prefix < node.plan->prefix)
        return RNS8_RANGE_ERROR;
      if (node.kind == RNS8_RESIDENT_EXACT_AXPBY &&
          proofs.at(node.d).exact_range_prefix < node.plan->prefix)
        return RNS8_RANGE_ERROR;
      auto& output = proofs.at(node.c);
      output.exact_range_prefix = node.plan->prefix;
      output.exact_max_magnitude = graph.math_proof->outputs[index].bound;
      continue;
    }
    rns8_resident_graph::ProofValue output;
    if (node.kind == RNS8_RESIDENT_EXACT_AXPBY) {
      const auto di = proofs.find(node.d);
      if (di == proofs.end())
        return RNS8_INVALID_ARGUMENT;
      status = rns8::detail::exact_axpby_range(*node.plan, ai->second, bi->second, di->second,
                                               node.alpha, node.beta, output.exact_max_magnitude);
    } else {
      status =
          rns8::detail::exact_gemm_range(*node.plan, ai->second, bi->second,
                                         output.exact_max_magnitude, &output.exact_axis_ranges);
    }
    if (status != RNS8_SUCCESS)
      return status;
    output.exact_range_prefix = node.plan->prefix;
    if (pending)
      (*pending)[index] = {output.exact_max_magnitude, output.exact_axis_ranges};
    proofs.insert_or_assign(node.c, std::move(output));
  }
  failed_node = UINT32_MAX;
  if (!reuse)
    stage_graph_math_proof(graph, *pending);
  return RNS8_SUCCESS;
}
void bind_graph_inputs(rns8_resident_graph& graph) {
  for (auto& input : graph.native_imports)
    if (input)
      input->commit_refresh();
  for (auto& input : graph.inputs) {
    input.instance = input.matrix->matrix_instance_id;
    input.generation = input.matrix->content_generation;
  }
}
} // namespace

rns8_status rns8_create_resident_graph(rns8_context* ctx, const rns8_resident_node* nodes,
                                       uint32_t count, rns8_resident_graph** out) {
  return guard_api([&]() -> rns8_status {
    if (!out)
      return RNS8_INVALID_ARGUMENT;
    *out = nullptr;
    if (!ctx || !nodes || !count || count > 4096)
      return RNS8_INVALID_ARGUMENT;
    if (!resident_graph_backend_supported(ctx->backend))
      return RNS8_UNSUPPORTED_BACKEND;
    auto graph = std::make_unique<rns8_resident_graph>();
    GraphMathStaging math_staging{*graph};
    graph->ctx = ctx;
    graph->nodes.assign(nodes, nodes + count);
    graph->native_imports.resize(count);
    std::unordered_map<const rns8_matrix*, uint32_t> producers;
    std::vector<std::pair<const void*, uint64_t>> exports;
    producers.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      const auto& node = nodes[i];
      if (!rns8::detail::valid_abi(node.struct_size, node.abi_version, sizeof(node)) || !node.c)
        return RNS8_INVALID_ARGUMENT;
      if (ctx->backend == RNS8_BACKEND_WRAP64_BYTE_LIMB &&
          (node.kind != RNS8_RESIDENT_MATMUL || !node.plan ||
           node.plan->desc.semantics != RNS8_WRAP_U64_MOD_2_64))
        return RNS8_UNSUPPORTED_BACKEND;
      if (node.kind != RNS8_RESIDENT_COMPLEX_MATMUL &&
          (node.a_imag || node.b_imag || node.c_imag ||
           node.complex_method != RNS8_COMPLEX_THREE_REAL_PRODUCTS))
        return RNS8_INVALID_ARGUMENT;
      if (node.kind != RNS8_RESIDENT_NATIVE_TO_RNS &&
          (node.native_source || node.native_max_magnitude || node.verify_native_bound))
        return RNS8_INVALID_ARGUMENT;
      if (node.kind != RNS8_RESIDENT_BASE_EXTEND && node.kind != RNS8_RESIDENT_EXPORT_LIMBS &&
          node.kind != RNS8_RESIDENT_NATIVE_TO_RNS && node.max_transfer_storage_bytes)
        return RNS8_INVALID_ARGUMENT;
      if (node.kind != RNS8_RESIDENT_EXPORT_LIMBS &&
          (node.export_limbs || node.export_ld || node.export_limb_count))
        return RNS8_INVALID_ARGUMENT;
      if (node.kind == RNS8_RESIDENT_NATIVE_TO_RNS) {
        if (node.plan || node.a || node.b || node.workspace || node.d || node.alpha || node.beta ||
            node.max_staged_residue_bytes || !producers.emplace(node.c, i).second)
          return RNS8_INVALID_ARGUMENT;
        const auto status = create_resident_native_import(ctx, node, graph->native_imports[i]);
        if (status != RNS8_SUCCESS)
          return status;
      } else if (node.kind == RNS8_RESIDENT_MATMUL || node.kind == RNS8_RESIDENT_EXACT_AXPBY) {
        if (node.kind == RNS8_RESIDENT_EXACT_AXPBY ? !node.d : (node.d || node.alpha || node.beta))
          return RNS8_INVALID_ARGUMENT;
        if (!node.plan || !node.a || !node.b || !node.workspace || node.target_prefix ||
            node.max_staged_residue_bytes ||
            (!rns8::detail::exact_wide_semantics(node.plan->desc.semantics) &&
             !(node.kind == RNS8_RESIDENT_MATMUL &&
               (uses_finite_storage(node.plan->desc.semantics) ||
                node.plan->desc.semantics == RNS8_WRAP_U64_MOD_2_64 ||
                bounded_graph_semantics(node.plan->desc.semantics)))) ||
            !producers.emplace(node.c, i).second)
          return RNS8_INVALID_ARGUMENT;
      } else if (node.kind == RNS8_RESIDENT_COMPLEX_MATMUL) {
        if (!node.plan || !node.a || !node.a_imag || !node.b || !node.b_imag || !node.c_imag ||
            !node.workspace || node.c == node.c_imag || node.target_prefix || node.d ||
            node.alpha || node.beta || node.plan->desc.semantics != RNS8_EXACT_WIDE_SIGNED ||
            (node.complex_method != RNS8_COMPLEX_THREE_REAL_PRODUCTS &&
             node.complex_method != RNS8_COMPLEX_FOUR_REAL_PRODUCTS) ||
            !producers.emplace(node.c, i).second || !producers.emplace(node.c_imag, i).second)
          return RNS8_INVALID_ARGUMENT;
      } else if (node.kind == RNS8_RESIDENT_BASE_EXTEND) {
        if (node.plan || node.a || node.b || node.workspace || node.d || node.alpha || node.beta)
          return RNS8_INVALID_ARGUMENT;
      } else if (node.kind == RNS8_RESIDENT_EXPORT_LIMBS) {
        if (!node.plan || !rns8::detail::exact_wide_semantics(node.plan->desc.semantics) ||
            node.a || node.b || node.workspace || node.d || node.alpha || node.beta ||
            node.target_prefix || node.max_staged_residue_bytes)
          return RNS8_INVALID_ARGUMENT;
        uint64_t bytes = 0;
        const auto status = export_span(node, bytes);
        if (status != RNS8_SUCCESS)
          return status;
        for (const auto& previous : exports)
          if (!rns8::detail::nonoverlapping_byte_ranges(node.export_limbs, bytes, previous.first,
                                                        previous.second))
            return RNS8_INVALID_ARGUMENT;
        exports.emplace_back(node.export_limbs, bytes);
      } else
        return RNS8_INVALID_ARGUMENT;
    }
    std::unordered_set<const rns8_matrix*> external;
    std::unordered_set<uint64_t*> holds{&ctx->prepared_references};
    for (uint32_t i = 0; i < count; ++i) {
      const auto& node = nodes[i];
      for (const auto* input : node_inputs(node)) {
        if (!input)
          continue;
        const auto found = producers.find(input);
        if (found == producers.end())
          external.insert(input);
        else if (found->second >= i)
          return RNS8_INVALID_ARGUMENT;
        holds.insert(&input->prepared_references);
      }
      holds.insert(&node.c->prepared_references);
      if (node.c_imag)
        holds.insert(&node.c_imag->prepared_references);
      if (node.plan)
        holds.insert(&node.plan->prepared_references);
      if (node.workspace)
        holds.insert(&node.workspace->prepared_references);
    }
    // Input binding order is deterministic, independent of hash iteration.
    for (const auto& node : graph->nodes)
      for (const auto* input : node_inputs(node))
        if (input && external.erase(input))
          graph->inputs.push_back({input, input->matrix_instance_id, input->content_generation});
    graph->holds.assign(holds.begin(), holds.end());
    for (auto* hold : graph->holds)
      if (*hold == UINT64_MAX)
        return RNS8_RANGE_ERROR;
    std::unordered_set<const void*> native_owners;
    for (const auto& input : graph->native_imports)
      if (input) {
        if (native_owners.insert(input->owner_key()).second)
          graph->native_readers.push_back(input.get());
        for (const auto& node : graph->nodes) {
          const std::array<const rns8_matrix*, 7> matrices{
              node.a, node.b, node.c, node.d, node.a_imag, node.b_imag, node.c_imag};
          for (const auto* matrix : matrices)
            if (matrix)
              for (const auto& storage : {std::pair<const void*, uint64_t>{
                                              matrix->hip_residues, matrix->hip_residue_bytes},
                                          {matrix->hip_byte_limbs, matrix->hip_byte_limb_bytes},
                                          {matrix->hip_native_i64, matrix->hip_native_i64_bytes},
                                          {matrix->hip_native_u64, matrix->hip_native_u64_bytes}})
                if (storage.first &&
                    !rns8::detail::nonoverlapping_byte_ranges(
                        input->source(), input->source_bytes(), storage.first, storage.second))
                  return RNS8_INVALID_ARGUMENT;
        }
      }
    uint32_t failed = UINT32_MAX;
    const auto status = prove_graph(*graph, failed, true);
    if (status != RNS8_SUCCESS)
      return status;
    commit_graph_math_proof(*graph);
    graph->id = next_graph_id();
    if (!graph->id)
      return RNS8_RANGE_ERROR;
    for (auto* hold : graph->holds)
      ++*hold;
    *out = graph.release();
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_refresh_resident_graph_inputs(rns8_resident_graph* graph) {
  return guard_api([&]() -> rns8_status {
    if (!graph)
      return RNS8_INVALID_ARGUMENT;
    if (graph->running)
      return RNS8_BUSY;
    GraphMathStaging math_staging{*graph};
    uint32_t failed = UINT32_MAX;
    auto& proofs = graph->proof_staging;
    auto status = prove_graph(*graph, failed, false, &proofs);
    if (status != RNS8_SUCCESS)
      return status;
    if (graph->hip_prepared) {
      status = validate_resident_transfer_envelopes(*graph, proofs);
      if (status != RNS8_SUCCESS)
        return status;
    }
    bind_graph_inputs(*graph);
    commit_graph_math_proof(*graph);
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_execute_resident_graph(rns8_resident_graph* graph,
                                        rns8_graph_execution_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!graph || !out ||
        !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out)))
      return RNS8_INVALID_ARGUMENT;
    if (graph->running)
      return RNS8_BUSY;
    GraphMathStaging math_staging{*graph};
    if (graph->epoch == UINT64_MAX)
      return RNS8_RANGE_ERROR;
    if (!graph->native_readers.empty() && !graph->hip_prepared) {
      const auto status = rns8_prepare_resident_graph_hip(graph);
      if (status != RNS8_SUCCESS)
        return status;
    }
    out->completed_nodes = 0;
    out->failed_node = UINT32_MAX;
    out->epoch = graph->epoch;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    // Prepared complex graphs reuse their bounded scratch and owned stream.
    // Keep a failed drain attached to the graph exactly as for borrowed streams.
    if (graph->hip_prepared) {
      rns8_graph_completion_token token{};
      auto status = submit_resident_graph_hip(graph, graph->synchronous_stream.get(), &token, false,
                                              &out->failed_node);
      if (status != RNS8_SUCCESS) {
        out->operation_status = status;
        out->epoch = graph->epoch;
        return status;
      }
      status = rns8_wait_resident_graph_hip(graph, token, out);
      if (status != RNS8_SUCCESS)
        out->operation_status = status;
      return status == RNS8_SUCCESS ? out->operation_status : status;
    }
#endif
    out->operation_status = prove_graph(*graph, out->failed_node, true);
    if (out->operation_status != RNS8_SUCCESS)
      return out->operation_status;
    commit_graph_math_proof(*graph);
    out->epoch = ++graph->epoch;
    graph->async_failure = {};
    graph->cancel_requested = false;
    graph->running = true;
    for (uint32_t i = 0; i < graph->nodes.size(); ++i) {
      const auto& node = graph->nodes[i];
      const auto input_status = cpu_graph_node_inputs_current(*graph, node);
      if (input_status != RNS8_SUCCESS)
        out->operation_status = input_status;
      else if ((node.kind == RNS8_RESIDENT_MATMUL || node.kind == RNS8_RESIDENT_EXACT_AXPBY) &&
               rns8::detail::exact_wide_semantics(node.plan->desc.semantics))
        out->operation_status = graph->execute_proven_exact_node(i);
      else if (node.kind == RNS8_RESIDENT_MATMUL)
        out->operation_status =
            node.plan->desc.semantics == RNS8_WRAP_U64_MOD_2_64
                ? rns8_gemm_wrap_u64(graph->ctx, node.plan, node.a, node.b, node.c, node.workspace)
            : uses_finite_storage(node.plan->desc.semantics)
                ? rns8_gemm_finite_u8(graph->ctx, node.plan,
                                      static_cast<uint16_t>(node.plan->desc.finite_modulus), node.a,
                                      node.b, node.c, node.workspace)
                : rns8_gemm_rns(graph->ctx, node.plan, node.a, node.b, node.c, node.workspace);
      else if (node.kind == RNS8_RESIDENT_EXACT_AXPBY)
        out->operation_status =
            rns8_gemm_exact_axpby(graph->ctx, node.plan, node.a, node.b, node.d, node.c, node.alpha,
                                  node.beta, node.workspace);
      else if (node.kind == RNS8_RESIDENT_COMPLEX_MATMUL)
        out->operation_status = rns8_gemm_complex_exact_signed(
            graph->ctx, node.plan, node.a, node.a_imag, node.b, node.b_imag, node.c, node.c_imag,
            node.complex_method, node.max_staged_residue_bytes, node.workspace);
      else if (node.kind == RNS8_RESIDENT_EXPORT_LIMBS)
        out->operation_status =
            node.plan->desc.semantics == RNS8_EXACT_WIDE_SIGNED
                ? rns8_export_exact_wide_signed_limbs(graph->ctx, node.plan, node.c,
                                                      node.export_limbs, node.export_ld,
                                                      node.export_limb_count)
                : rns8_export_exact_wide_unsigned_limbs(graph->ctx, node.plan, node.c,
                                                        node.export_limbs, node.export_ld,
                                                        node.export_limb_count);
      else if (graph->ctx->backend == RNS8_BACKEND_HIP_DIRECT)
        out->operation_status = rns8_lift_exact_wide_hip(graph->ctx, node.c, node.target_prefix,
                                                         node.max_staged_residue_bytes);
      else
        out->operation_status = rns8_lift_exact_wide_cpu(graph->ctx, node.c, node.target_prefix,
                                                         node.max_staged_residue_bytes);
      if (out->operation_status != RNS8_SUCCESS) {
        out->failed_node = i;
        graph->running = false;
        return out->operation_status;
      }
      bind_cpu_graph_node_outputs(*graph, node);
      ++out->completed_nodes;
    }
    graph->running = false;
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_resident_graph(rns8_resident_graph* graph) {
  if (!graph)
    return RNS8_SUCCESS;
  if (graph->running)
    return RNS8_BUSY;
  graph->retiring = true;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  // Retain both handle and references on cleanup failure so destruction can be
  // retried. Release wrappers only after the corresponding HIP destroy succeeds.
  if (graph->ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
    rns8::detail::hip_scoped_device device(graph->ctx->device_id);
    if (device.status() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (graph->captured_executable) {
      if (graph->captured_executable_owner.reset() != hipSuccess)
        return RNS8_BACKEND_FAILURE;
      graph->captured_executable = nullptr;
    }
    if (graph->captured_graph) {
      if (graph->captured_graph_owner.reset() != hipSuccess)
        return RNS8_BACKEND_FAILURE;
      graph->captured_graph = nullptr;
    }
    if (graph->completion_event.get()) {
      if (graph->completion_event.reset() != hipSuccess)
        return RNS8_BACKEND_FAILURE;
    }
    if (graph->synchronous_stream.get()) {
      if (graph->synchronous_stream.reset() != hipSuccess)
        return RNS8_BACKEND_FAILURE;
    }
    if (graph->complex_storage.reset() != RNS8_SUCCESS)
      return RNS8_BACKEND_FAILURE;
    for (auto& transfer : graph->transfers)
      if (transfer.host.reset() != hipSuccess)
        return RNS8_BACKEND_FAILURE;
  }
#endif
  for (auto* hold : graph->holds)
    --*hold;
  delete graph;
  return RNS8_SUCCESS;
}

rns8_status rns8_get_resident_graph_resources(const rns8_resident_graph* graph,
                                              rns8_resident_graph_resources* out) {
  if (!graph || !out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out)))
    return RNS8_INVALID_ARGUMENT;
  rns8_resident_graph_resources info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  info.prepared = graph->hip_prepared;
  for (const auto& node : graph->nodes)
    info.transfer_nodes += node.kind == RNS8_RESIDENT_BASE_EXTEND ||
                           node.kind == RNS8_RESIDENT_EXPORT_LIMBS ||
                           node.kind == RNS8_RESIDENT_NATIVE_TO_RNS;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  info.retained_device_scratch_bytes = graph->complex_storage.bytes();
  for (const auto& transfer : graph->transfers) {
    if (transfer.host.capacity() > UINT64_MAX - info.retained_pinned_host_bytes)
      return RNS8_RANGE_ERROR;
    info.retained_pinned_host_bytes += transfer.host.capacity();
  }
#endif
  for (const auto& input : graph->native_imports)
    if (input) {
      if (input->retained_native_bytes() > UINT64_MAX - info.retained_native_matrix_bytes)
        return RNS8_RANGE_ERROR;
      info.retained_native_matrix_bytes += input->retained_native_bytes();
    }
  *out = info;
  return RNS8_SUCCESS;
}

#include "api_resident_graph_hip.inc"
