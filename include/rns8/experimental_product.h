#ifndef RNS8_EXPERIMENTAL_PRODUCT_H
#define RNS8_EXPERIMENTAL_PRODUCT_H
#include "rns8.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Explicit synchronous CPU/direct-HIP exact-wide materialization planning.
 * These integer-nanosecond tables are caller estimates, not qualification or
 * AUTO promotion evidence. UINT64_MAX is unknown and never becomes zero.
 * Entries use prefix indices 1..20, with growth indexed old_prefix*21+new_prefix.
 * Packing is shared by logical B value within a replay, including incremental
 * plane growth. Each replay repacks; no hidden cross-execution cache is assumed. */
typedef struct rns8_product_materialization_identity {
  uint64_t struct_size;
  uint32_t abi_version;
  uint32_t node_count;
  rns8_status admission_status;
  uint32_t blocking_node; /* UINT32_MAX when every edge is range-admitted */
  uint32_t minimum_prefixes[64];
  char graph_sha256[65];
  char build_sha256[65];
  char target[128];
} rns8_product_materialization_identity;
typedef struct rns8_product_materialization_node_costs {
  uint64_t uniform_gemm_ns[21];
  uint64_t lift_ns[441];
  uint64_t packed_gemm_ns[21];
  uint64_t pack_ns[441];
} rns8_product_materialization_node_costs;
typedef struct rns8_product_materialization_profile {
  uint64_t struct_size;
  uint32_t abi_version;
  uint32_t max_evaluations; /* 4..65536 complete schedule evaluations */
  uint32_t max_chain_frontier; /* 1..4096 cost states; truncation is explicit */
  uint32_t node_count;
  rns8_product_materialization_identity identity;
  char calibration_identity[128];
  const rns8_product_materialization_node_costs* nodes;
} rns8_product_materialization_profile;
typedef struct rns8_product_materialization_info {
  uint64_t struct_size;
  uint32_t abi_version;
  uint32_t node_count;
  uint32_t evaluated;
  uint32_t exhaustive;
  uint32_t chain_dynamic_programming;
  uint32_t cost_optimal;
  uint64_t estimated_ns;
  uint64_t retained_arena_and_lift_bytes;
  uint64_t packed_payload_bytes; /* sum of packed views before arena reuse */
  uint64_t completed_packing_calls; /* last replay's completed conversions */
  char calibration_identity[128];
} rns8_product_materialization_info;
typedef struct rns8_product_materialization_choice {
  uint32_t compute_prefix;
  uint32_t packed_b;
  uint32_t packed_capacity_prefix;
  uint32_t first_packed_reader;
  uint32_t last_packed_reader;
  uint32_t reserved;
  uint64_t packed_bytes;
  uint64_t arena_offset;
} rns8_product_materialization_choice;

/* Read-only query binds shape/topology, full ordered basis, semantics, proof
 * envelopes, requested outputs, prefix policy, target and compiled build. It
 * does not allocate on the GPU. output_count=0 selects only the final value.
 * Successful queries report admission_status separately, including the first
 * range-blocking node; no downstream enormous bound is constructed after it. */
RNS8_API rns8_status rns8_query_product_materialization_identity(rns8_context* ctx,
    const rns8_product_recipe_desc* desc,const uint32_t* output_ids,uint32_t output_count,
    rns8_product_materialization_identity* out);
/* FULL_PREFIX preserves every compute prefix; MINIMUM_COMPUTE_PREFIX allows
 * cost selection above independently proved minimum prefixes. Prices cannot
 * weaken those proofs. The descriptor's hard budget covers the single retained
 * uniform/packed arena and lift scratch, excluding external matrices and host
 * descriptors/proofs. Larger DAG search is bounded and need not be optimal.
 * A failed bounded search is unsupported, not proof of memory infeasibility.
 * The ordinary recipe execution/rebind/destroy APIs consume the resulting owner.
 * No async/capture behavior is introduced by this function. */
RNS8_API rns8_status rns8_create_materialized_product_recipe(rns8_context* ctx,
    const rns8_product_recipe_desc* desc,const uint32_t* output_ids,uint32_t output_count,
    const rns8_product_materialization_profile* profile,rns8_product_recipe** out);
/* Ragged variants price the post-gather graph using the current source proofs.
 * Query does not modify the conversion destinations. Creation validates prices
 * before gathering; the shared budget additionally covers retained ragged
 * owners, conversion destinations and peak gather scratch. Refresh and lifetime
 * rules are those of rns8_create_product_recipe_ragged. The source's producer
 * tile shape need not match the consumer's packed 16x16 B tile shape. */
RNS8_API rns8_status rns8_query_product_materialization_identity_ragged(rns8_context* ctx,
    const rns8_product_recipe_desc* desc,const rns8_ragged_recipe_binding* bindings,uint32_t binding_count,
    const uint32_t* output_ids,uint32_t output_count,uint64_t max_total_bytes,
    rns8_product_materialization_identity* out);
RNS8_API rns8_status rns8_create_materialized_product_recipe_ragged(rns8_context* ctx,
    const rns8_product_recipe_desc* desc,const rns8_ragged_recipe_binding* bindings,uint32_t binding_count,
    const uint32_t* output_ids,uint32_t output_count,uint64_t max_total_bytes,
    const rns8_product_materialization_profile* profile,rns8_product_recipe** out);
RNS8_API rns8_status rns8_get_product_materialization_info(const rns8_product_recipe* recipe,
    rns8_product_materialization_info* out);
RNS8_API rns8_status rns8_get_product_materialization_choices(const rns8_product_recipe* recipe,
    rns8_product_materialization_choice* choices,uint32_t capacity);
#ifdef __cplusplus
}
#endif
#endif
