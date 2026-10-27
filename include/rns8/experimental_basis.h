#ifndef RNS8_EXPERIMENTAL_BASIS_H
#define RNS8_EXPERIMENTAL_BASIS_H
#include "rns8/rns8.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct rns8_basis rns8_basis;
typedef struct rns8_basis_plan rns8_basis_plan;

/* Explicit experimental ordered centered-INT8 basis. Moduli are copied, each
 * lies in 2..256, and every pair must be coprime. Order is storage identity.
 * The explicit cost-bound selector below is opt-in; the default ladder is unchanged. */
typedef struct rns8_basis_desc {
  uint64_t struct_size;
  uint32_t abi_version,representation;
  const uint16_t* moduli;
  uint32_t count,flags;
} rns8_basis_desc;
typedef struct rns8_basis_info {
  uint64_t struct_size;
  uint32_t abi_version,representation;
  uint64_t identity;
  uint32_t count,selected_prefix;
  uint16_t moduli[20];
  uint64_t prefix_product[3];
} rns8_basis_info;
typedef struct rns8_basis_plan_desc {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_semantics semantics;
  /* Compute provider. CPU requires a CPU context. HIP/direct and optional
   * CK/rocWMMA/hipBLASLt/builtins require direct-HIP storage/context ownership. */
  rns8_backend_kind backend;
  int64_t m,n,k;
  uint32_t prefix,flags;
  uint64_t max_device_bytes,max_host_bytes;
} rns8_basis_plan_desc;
typedef struct rns8_basis_resources {
  uint64_t struct_size;
  uint32_t abi_version,reserved;
  uint64_t required_device_bytes,required_host_bytes;
  uint64_t ordered_basis_identity;
  uint32_t prefix,representation;
} rns8_basis_resources;

/* Explicit opt-in runtime choice. Prices bind the entire ordered basis and
 * compiled workload; they are caller-supplied observations/models, never an
 * arithmetic certificate or automatic performance qualification. */
typedef struct rns8_basis_cost {
  uint64_t struct_size;
  uint32_t abi_version,kind; /* 1=observed, 2=modeled */
  char build_identity[65],cohort_identity[65],target_id[64];
  uint16_t ordered_moduli[20];
  uint32_t count,prefix;
  rns8_backend_kind backend;
  rns8_semantics semantics;
  int64_t m,n,k;
  uint64_t device_bytes,host_bytes;
  uint64_t preparation_ns,pack_ns,gemm_ns,export_ns; /* UINT64_MAX=unknown */
} rns8_basis_cost;
typedef struct rns8_basis_candidate {
  const rns8_basis* basis; /* null denotes the default ordered basis */
  rns8_basis_plan_desc plan;
  rns8_basis_cost cost;
} rns8_basis_candidate;
typedef struct rns8_basis_selection_desc {
  uint64_t struct_size;
  uint32_t abi_version,flags; /* bit0: repack each call, bit1: export each call */
  rns8_semantics semantics;
  int64_t m,n,k;
  char cohort_identity[65];
  uint64_t reuse_count,magnitude_bound[3];
} rns8_basis_selection_desc;
typedef struct rns8_basis_selection {
  uint64_t struct_size;
  uint32_t abi_version,selected_index;
  uint32_t eligible_candidates,price_kind;
  uint64_t projected_ns;
  rns8_basis_resources resources;
} rns8_basis_selection;
RNS8_API rns8_status rns8_select_basis_plan(const rns8_context*,const rns8_basis_selection_desc*,
    const rns8_basis_candidate*,uint32_t,rns8_basis_selection*);
RNS8_API rns8_status rns8_create_selected_basis_plan(rns8_context*,const rns8_basis_selection_desc*,
    const rns8_basis_candidate*,uint32_t,rns8_basis_selection*,rns8_basis_plan**);
RNS8_API rns8_status rns8_create_basis(const rns8_basis_desc*,rns8_basis**);
RNS8_API rns8_status rns8_destroy_basis(rns8_basis*);
RNS8_API rns8_status rns8_get_basis_info(const rns8_basis*,uint32_t prefix,rns8_basis_info*);
/* Matrix/plan owners retain immutable basis data independently of this handle.
 * NULL basis explicitly requests the unchanged default basis for controls or
 * conversion destinations. Exact-wide signed/unsigned only; no type inference. */
RNS8_API rns8_status rns8_create_basis_matrix(rns8_context*,const rns8_basis*,const rns8_matrix_desc*,rns8_matrix**);
RNS8_API rns8_status rns8_get_matrix_basis(const rns8_matrix*,rns8_basis_info*);
/* Product recipes require one complete ordered basis on every input/output.
 * They retain that immutable identity independently of the public basis handle. */
RNS8_API rns8_status rns8_get_product_recipe_basis(const rns8_product_recipe*,rns8_basis_info*);
RNS8_API rns8_status rns8_query_basis_plan(const rns8_basis*,const rns8_basis_plan_desc*,rns8_basis_resources*);
/* Typed static inventory for the explicit basis owner: GEMM, PACK, EXPORT,
 * BASE_EXTEND (integer conversion), or EXPERIMENTAL_BASIS (plan construction).
 * This query owns no GPU resources and never certifies current input values.
 * The returned status is also recorded in support; basis contains full order. */
RNS8_API rns8_status rns8_query_basis_operation(const rns8_context*,const rns8_basis*,
    const rns8_basis_plan_desc*,rns8_operation_kind,uint32_t execution_modes,
    rns8_operation_support*,rns8_basis_info*);
RNS8_API rns8_status rns8_create_basis_plan(rns8_context*,const rns8_basis*,const rns8_basis_plan_desc*,rns8_basis_plan**);
RNS8_API rns8_status rns8_destroy_basis_plan(rns8_basis_plan*);
RNS8_API rns8_status rns8_get_basis_plan_info(const rns8_basis_plan*,rns8_basis_info*,rns8_basis_resources*);
RNS8_API rns8_status rns8_pack_basis_i64(rns8_basis_plan*,rns8_matrix*,const int64_t*,int64_t ld,uint64_t version);
RNS8_API rns8_status rns8_pack_basis_u64(rns8_basis_plan*,rns8_matrix*,const uint64_t*,int64_t ld,uint64_t version);
RNS8_API rns8_status rns8_gemm_basis(rns8_basis_plan*,const rns8_matrix*,const rns8_matrix*,rns8_matrix*);
/* Direct-HIP fixed-address prepared GEMM with the ordinary stream, owned graph,
 * completion, cancellation and refresh APIs. Retains the full immutable basis
 * and matrices independently of the basis-plan handle. No hidden synchronization
 * or default-ladder reinterpretation; input generation changes require refresh.
 * Native transfers and exact conversion use the explicit prepared adapters below. */
RNS8_API rns8_status rns8_prepare_basis_gemm_hip(rns8_basis_plan*,const rns8_matrix*,const rns8_matrix*,rns8_matrix*,rns8_prepared_execution**);
/* Fixed-address exact conversion/lift between two complete ordered bases.
 * Uses the ordinary prepared stream/owned-graph lifecycle and retains both
 * immutable bases independently of the plan. Budget includes device staging
 * and pinned status. Native-device import/export also accept explicit-basis
 * matrices through rns8_prepare_native_transfer_hip and its real lease owner. */
RNS8_API rns8_status rns8_prepare_basis_conversion_hip(rns8_basis_plan*,const rns8_matrix*,rns8_matrix*,uint64_t max_total_scratch_bytes,rns8_prepared_execution**);
RNS8_API rns8_status rns8_export_basis_limbs(rns8_basis_plan*,const rns8_matrix*,uint64_t*,int64_t ld,uint32_t limbs);
/* Explicit uniquely represented integer conversion, including reordered bases.
 * Source and destination signedness must agree. The source proof must be valid
 * in its actual written prefix, and the destination must uniquely cover it.
 * Source identity/generation is preserved; destination publishes one new value.
 * Ordinary default-basis plans/caches never reinterpret experimental matrices. */
RNS8_API rns8_status rns8_convert_basis_values(rns8_basis_plan*,const rns8_matrix*,rns8_matrix*);
#ifdef __cplusplus
}
#endif
#endif
