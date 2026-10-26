#ifndef RNS8_PIPELINE_RECIPE_H
#define RNS8_PIPELINE_RECIPE_H
#include "rns8.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Bound whole-route planning. This explicit surface never changes AUTO or
 * promotes a profile. Enumeration is host-only and does not probe/allocate on
 * a device. It binds current inputs and reports dynamic preparation separately.
 * Source/target eligibility is shared with the existing operation provider. */
typedef enum rns8_pipeline_route_kind {
  RNS8_ROUTE_UNIFORM=0,RNS8_ROUTE_ENCODED_NATIVE_B=1,RNS8_ROUTE_SPLIT_K=2,
  RNS8_ROUTE_NARROW=3,RNS8_ROUTE_NATIVE_QUEUE=4
} rns8_pipeline_route_kind;
typedef struct rns8_pipeline_recipe_options {
  uint64_t struct_size;uint32_t abi_version,route_mask;
  uint32_t execution_mode; /* exactly SYNC, BORROWED_STREAM or OWNED_GRAPH */
  uint32_t output_mode; /* resident0; signed/unsigned/canonical host limbs1/2/3 */
  uint32_t output_limbs,export_algorithm; /* checked CRT0..4, resident requires0 */
  int64_t output_ld; /* bound host cell stride; resident requires0 */
  uint32_t modulus_group,workers,fragments,fragment_window,locality_order_mask;
  rns8_locality_desc locality;
  uint64_t reuse_count,max_extra_payload_bytes;
} rns8_pipeline_recipe_options;
typedef struct rns8_pipeline_recipe_candidate {
  uint64_t struct_size;uint32_t abi_version,route,method,execution_mode;
  rns8_status status;uint32_t reason_code,resource_status,cost_known,cost_origin,pareto;
  uint64_t retained_payload_bytes,preparation_peak_bytes,export_payload_bytes,peak_extra_payload_bytes;
  double lower_amortized_ns,upper_amortized_ns;
  char identity_sha256[65],cost_identity_sha256[65],provider_identity_sha256[65],build_identity_sha256[65],target_id[128],explanation[160];
} rns8_pipeline_recipe_candidate;
/* Complete-route prices, explicitly modeled1 or measured2. All three stage
 * costs include the real preparation/enqueue/status/completion/publication
 * boundary. Profile identity matches candidate.cost_identity_sha256/build/target;
 * instance/generation binding has its separate identity, so independent process
 * profiles are reusable only for the same logical/proof/materialization facts.
 * Neither origin is hardware qualification or permission to install a cache. */
typedef struct rns8_pipeline_recipe_prices {
  uint64_t struct_size;uint32_t abi_version,known_mask,origin;
  char identity_sha256[65],build_identity_sha256[65],target_id[128],calibration_id[128];
  double lower_ns[3],upper_ns[3]; /* prepare, complete producer, final export */
} rns8_pipeline_recipe_prices;
RNS8_API rns8_status rns8_query_pipeline_recipes(const rns8_context*,const rns8_plan*,
    const rns8_matrix*,const rns8_matrix*,const rns8_matrix*,const rns8_pipeline_recipe_options*,
    const rns8_pipeline_recipe_prices*,uint32_t prices_count,
    rns8_pipeline_recipe_candidate*,uint32_t capacity,uint32_t* required_count);
RNS8_API rns8_status rns8_select_pipeline_recipe(const rns8_pipeline_recipe_candidate*,uint32_t count,uint32_t* selected_index);
typedef struct rns8_pipeline_recipe rns8_pipeline_recipe;
/* Revalidates the candidate against current values before acquiring resources.
 * Holds all borrowed matrices/plan/context until checked destruction. Changed
 * input generation requires recreation; a cost never weakens that admission.
 * On an undrainable preparation/cleanup failure *out may be cleanup-only. */
RNS8_API rns8_status rns8_prepare_pipeline_recipe(rns8_context*,const rns8_plan*,
    const rns8_matrix*,const rns8_matrix*,rns8_matrix*,const rns8_pipeline_recipe_options*,
    const rns8_pipeline_recipe_candidate*,rns8_pipeline_recipe**);
/* SYNC only. A resident call requires dst=NULL,ld=0. Host limb export stages
 * before writing the caller. The producer's successful resident value remains
 * current if a later host export fails; this is a sequential pipeline. */
RNS8_API rns8_status rns8_execute_pipeline_recipe(rns8_pipeline_recipe*,uint64_t* dst,int64_t ld);
/* Async resident routes only; borrowed nondefault stream and existing token
 * semantics. The token does not retain this owner. Partial enqueue returns its
 * issued token. Query/wait/cancel delegate to the actual prepared execution. */
RNS8_API rns8_status rns8_submit_pipeline_recipe_hip(rns8_pipeline_recipe*,void* stream,rns8_completion_token*);
RNS8_API rns8_status rns8_query_pipeline_recipe_hip(rns8_pipeline_recipe*,rns8_completion_token,rns8_completion_info*);
RNS8_API rns8_status rns8_wait_pipeline_recipe_hip(rns8_pipeline_recipe*,rns8_completion_token,rns8_completion_info*);
RNS8_API rns8_status rns8_cancel_pipeline_recipe_hip(rns8_pipeline_recipe*,rns8_completion_token);
RNS8_API rns8_status rns8_get_pipeline_recipe_candidate(const rns8_pipeline_recipe*,rns8_pipeline_recipe_candidate*);
RNS8_API rns8_status rns8_destroy_pipeline_recipe(rns8_pipeline_recipe*);
/* Preparation exception cleanup uses an allocation-free complete-owner queue;
 * failed drains remain owned. Explicit retry does not launch new arithmetic. */
RNS8_API void rns8_reap_pipeline_recipe_cleanup(void);
#ifdef __cplusplus
}
#endif
#endif
