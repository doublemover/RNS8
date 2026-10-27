#ifndef RNS8_EXPERIMENTAL_PRODUCT_H
#define RNS8_EXPERIMENTAL_PRODUCT_H
#include "rns8.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Explicit synchronous CPU/direct-HIP exact-wide materialization planning.
 * These integer-nanosecond tables are caller estimates, not qualification or
 * AUTO promotion evidence. UINT64_MAX is unknown and never becomes zero.
 * Prefix indices are 1..20. Uniform GEMM uses compute_prefix; lift uses
 * old_prefix*21+new_prefix. Packed GEMM uses capacity*21+compute_prefix;
 * packing uses (capacity*21+old_prefix)*21+new_prefix. Capacity is the
 * physical packed B prefix shared by all readers, not its current prefix.
 * Valid packed entries have 1<=compute_prefix<=capacity<=20 and
 * 0<=old_prefix<new_prefix<=capacity. Unused cells must remain unknown.
 * Packing is shared by logical B value, including incremental plane growth.
 * Cross-replay external-input retention is explicit, generation-bound and
 * charged for the owner's entire lifetime. Intermediates never survive replay. */
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
enum {
  RNS8_PRODUCT_MATERIALIZATION_COST_SCHEMA_VERSION = 3,
  RNS8_PRODUCT_RETAIN_EXTERNAL_PACKED_B = 1,
  RNS8_PRODUCT_REQUIRE_PROJECTED_AMORTIZATION = 2,
  RNS8_PRODUCT_MATERIALIZATION_PREFIX_COUNT = 21,
  RNS8_PRODUCT_MATERIALIZATION_UNIFORM_GEMM_COST_COUNT = 21,
  RNS8_PRODUCT_MATERIALIZATION_LIFT_COST_COUNT = 441,
  RNS8_PRODUCT_MATERIALIZATION_PACKED_GEMM_COST_COUNT = 441,
  RNS8_PRODUCT_MATERIALIZATION_PACK_COST_COUNT = 9261
};

typedef struct rns8_product_materialization_node_costs {
  uint64_t uniform_gemm_ns[21];
  uint64_t lift_ns[441];
  uint64_t packed_gemm_ns[441];
  uint64_t pack_ns[9261];
} rns8_product_materialization_node_costs;
typedef struct rns8_product_materialization_profile {
  uint64_t struct_size;
  uint32_t abi_version;
  uint32_t max_evaluations; /* 4..65536 complete schedule evaluations */
  uint32_t max_chain_frontier; /* 1..4096 cost states; truncation is explicit */
  uint32_t node_count;
  uint32_t cost_schema_version; /* RNS8_PRODUCT_MATERIALIZATION_COST_SCHEMA_VERSION */
  uint32_t node_costs_size; /* sizeof(rns8_product_materialization_node_costs) */
  rns8_product_materialization_identity identity;
  char calibration_identity[128];
  const rns8_product_materialization_node_costs* nodes;
  uint32_t flags; /* RETAIN_EXTERNAL_PACKED_B | REQUIRE_PROJECTED_AMORTIZATION */
  uint32_t reserved;
  uint64_t expected_replays; /* 1..UINT64_MAX-1, explicit lifetime estimate */
  uint64_t preparation_ns,calibration_ns,orchestration_ns,publication_ns;
  uint64_t baseline_replay_ns,baseline_preparation_ns; /* UINT64_MAX = unknown */
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
  uint64_t estimated_first_replay_ns,estimated_steady_replay_ns,one_time_ns,lifetime_ns,amortized_ns;
  uint64_t expected_replays,projected_break_even_replays,retained_external_packed_bytes;
  uint64_t completed_replays,reused_external_packing_calls;
  uint32_t amortization_status; /* 0 unknown,1 projected profitable,2 nonprofitable */
  uint32_t flags;
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
