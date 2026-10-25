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
 * No automatic basis selection or default-ladder replacement occurs. */
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
