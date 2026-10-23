#include "core/api_internal.hpp"
#include <unordered_map>
#include <unordered_set>

using namespace rns8::detail::api;
struct rns8_resident_graph {
  struct Input {
    const rns8_matrix* matrix;
    uint64_t instance, generation;
  };
  rns8_context* ctx = nullptr;
  std::vector<rns8_resident_matmul_node> nodes;
  std::vector<Input> inputs;
  std::vector<uint64_t*> holds;
  uint64_t epoch = 0;
  bool running = false;
};

namespace {
// Metadata-only proof values: no copies of host/device matrix payloads.
rns8_status prove_graph(const rns8_resident_graph& graph, uint32_t& failed_node, bool check_bindings) {
  std::unordered_map<const rns8_matrix*, rns8_matrix> proofs;
  proofs.reserve(graph.inputs.size() + graph.nodes.size());
  for (const auto& input : graph.inputs) {
    const auto& m = *input.matrix;
    if (m.generation_exhausted || !rns_residue_state_current_for_backend(m, graph.ctx->backend) ||
        (check_bindings && (m.matrix_instance_id != input.instance || m.content_generation != input.generation)))
      return RNS8_INVALID_ARGUMENT;
    if (!m.exact_range_prefix || m.exact_range_prefix > m.prefix ||
        !rns8::detail::exact_range_fits(m.desc.semantics, m.exact_max_magnitude,
                                      rns8::detail::modulus_product(m.exact_range_prefix)))
      return RNS8_RANGE_ERROR;
    auto& proof = proofs.try_emplace(input.matrix).first->second;
    proof.exact_max_magnitude = m.exact_max_magnitude;
    proof.exact_axis_ranges = m.exact_axis_ranges;
    proof.exact_range_prefix = m.exact_range_prefix;
  }
  for (uint32_t index = 0; index < graph.nodes.size(); ++index) {
    failed_node = index;
    const auto& node = graph.nodes[index];
    auto status = validate_plan_context_workspace(*graph.ctx, *node.plan, *node.workspace);
    if (status != RNS8_SUCCESS) return status;
    status = validate_rns_gemm_operands(*graph.ctx, *node.plan, *node.a, *node.b, *node.c, false);
    if (status != RNS8_SUCCESS) return status;
    if (node.c->generation_exhausted) return RNS8_RANGE_ERROR;
    auto ai = proofs.find(node.a), bi = proofs.find(node.b);
    if (ai == proofs.end() || bi == proofs.end()) return RNS8_INVALID_ARGUMENT;
    rns8_matrix output;
    status = rns8::detail::exact_gemm_range(*node.plan, ai->second, bi->second,
                                         output.exact_max_magnitude, &output.exact_axis_ranges);
    if (status != RNS8_SUCCESS) return status;
    output.exact_range_prefix = node.plan->prefix;
    proofs.emplace(node.c, std::move(output));
  }
  failed_node = UINT32_MAX;
  return RNS8_SUCCESS;
}
void bind_graph_inputs(rns8_resident_graph& graph) {
  for (auto& input : graph.inputs) {
    input.instance = input.matrix->matrix_instance_id;
    input.generation = input.matrix->content_generation;
  }
}
}

rns8_status rns8_create_resident_graph(
    rns8_context* ctx, const rns8_resident_matmul_node* nodes, uint32_t count, rns8_resident_graph** out) {
  return guard_api([&]() -> rns8_status {
    if (!out) return RNS8_INVALID_ARGUMENT;
    *out = nullptr;
    if (!ctx || !nodes || !count || count > 4096) return RNS8_INVALID_ARGUMENT;
    if (ctx->backend != RNS8_BACKEND_CPU_REFERENCE && ctx->backend != RNS8_BACKEND_HIP_DIRECT)
      return RNS8_UNSUPPORTED_BACKEND;
    auto graph = std::make_unique<rns8_resident_graph>();
    graph->ctx = ctx;
    graph->nodes.assign(nodes, nodes + count);
    std::unordered_map<const rns8_matrix*, uint32_t> producers;
    producers.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      const auto& node = nodes[i];
      if (!rns8::detail::valid_abi(node.struct_size, node.abi_version, sizeof(node)) ||
          !node.plan || !node.a || !node.b || !node.c || !node.workspace ||
          !rns8::detail::exact_wide_semantics(node.plan->desc.semantics) ||
          !producers.emplace(node.c, i).second) return RNS8_INVALID_ARGUMENT;
    }
    std::unordered_set<const rns8_matrix*> external;
    std::unordered_set<uint64_t*> holds{&ctx->prepared_references};
    for (uint32_t i = 0; i < count; ++i) {
      const auto& node = nodes[i];
      for (const auto* input : {node.a, node.b}) {
        const auto found = producers.find(input);
        if (found == producers.end()) external.insert(input);
        else if (found->second >= i) return RNS8_INVALID_ARGUMENT;
        holds.insert(&input->prepared_references);
      }
      holds.insert(&node.c->prepared_references);
      holds.insert(&node.plan->prepared_references);
      holds.insert(&node.workspace->prepared_references);
    }
    // Input binding order is deterministic, independent of hash iteration.
    for (const auto& node : graph->nodes)
      for (const auto* input : {node.a, node.b})
        if (external.erase(input)) graph->inputs.push_back({input, input->matrix_instance_id, input->content_generation});
    graph->holds.assign(holds.begin(), holds.end());
    for (auto* hold : graph->holds) if (*hold == UINT64_MAX) return RNS8_RANGE_ERROR;
    uint32_t failed = UINT32_MAX;
    const auto status = prove_graph(*graph, failed, true);
    if (status != RNS8_SUCCESS) return status;
    for (auto* hold : graph->holds) ++*hold;
    *out = graph.release();
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_refresh_resident_graph_inputs(rns8_resident_graph* graph) {
  return guard_api([&]() -> rns8_status {
    if (!graph) return RNS8_INVALID_ARGUMENT;
    if (graph->running) return RNS8_BUSY;
    uint32_t failed = UINT32_MAX;
    const auto status = prove_graph(*graph, failed, false);
    if (status != RNS8_SUCCESS) return status;
    bind_graph_inputs(*graph);
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_execute_resident_graph(rns8_resident_graph* graph, rns8_graph_execution_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!graph || !out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out)))
      return RNS8_INVALID_ARGUMENT;
    if (graph->running) return RNS8_BUSY;
    if (graph->epoch == UINT64_MAX) return RNS8_RANGE_ERROR;
    out->completed_nodes = 0; out->failed_node = UINT32_MAX; out->epoch = graph->epoch;
    out->operation_status = prove_graph(*graph, out->failed_node, true);
    if (out->operation_status != RNS8_SUCCESS) return out->operation_status;
    out->epoch = ++graph->epoch;
    graph->running = true;
    for (uint32_t i = 0; i < graph->nodes.size(); ++i) {
      const auto& node = graph->nodes[i];
      out->operation_status = rns8_gemm_rns(graph->ctx, node.plan, node.a, node.b, node.c, node.workspace);
      if (out->operation_status != RNS8_SUCCESS) {
        out->failed_node = i; graph->running = false; return out->operation_status;
      }
      ++out->completed_nodes;
    }
    graph->running = false;
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_resident_graph(rns8_resident_graph* graph) {
  if (!graph) return RNS8_SUCCESS;
  if (graph->running) return RNS8_BUSY;
  for (auto* hold : graph->holds) --*hold;
  delete graph;
  return RNS8_SUCCESS;
}
