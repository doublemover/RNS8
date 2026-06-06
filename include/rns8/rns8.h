#ifndef RNS8_RNS8_H
#define RNS8_RNS8_H

#include <stdint.h>

#include "rns8/bounds.h"
#include "rns8/moduli.h"
#include "rns8/semantics.h"
#include "rns8/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rns8_context rns8_context;
typedef struct rns8_plan rns8_plan;
typedef struct rns8_matrix rns8_matrix;
typedef struct rns8_workspace rns8_workspace;
typedef struct rns8_prepack_cache rns8_prepack_cache;

typedef struct rns8_context_options {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_backend_kind requested_backend;
  /* Reserved for future hard-cut ABI versions. Must be zero. */
  uint32_t flags;
} rns8_context_options;

typedef struct rns8_device_info {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_backend_kind backend;
  int32_t device_id;
  uint32_t hip_available;
  uint32_t hip_runtime_version;
  uint32_t hip_driver_version;
  uint64_t global_mem_bytes;
  char name[128];
  char gcn_arch[64];
  char detail[256];
} rns8_device_info;

/*
 * Plan creation flags for rns8_gemm_desc.flags.
 *
 * By default, max_prefix is an upper correctness budget: bounded and
 * exact-wide RNS plans execute the minimum proven prefix for the contract.
 * Set RNS8_PLAN_FORCE_FIXED_PREFIX only for controlled experiments or
 * compatibility captures that intentionally execute exactly max_prefix planes.
 *
 * RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS is valid only with per-tile bounded
 * contracts whose zero tile bounds came from a trusted exact scan or proof.
 * Without that opt-in, zero bounds remain ordinary range contracts and do not
 * authorize execution shortcuts.
 *
 * RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS is valid only with per-tile
 * bounded contracts and explicit zero_a_rows/zero_b_cols proof masks. A set A
 * row or B column mask bit proves every output cell in that row or column is
 * zero for this specific input pair. The proof is trusted input metadata; RNS8
 * does not infer or verify it during plan creation.
 */
#define RNS8_PLAN_FORCE_FIXED_PREFIX 0x00000001u
#define RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS 0x00000002u
#define RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS 0x00000004u

/*
 * Per-tile schedule flags returned through rns8_plan_schedule_info.flags and
 * rns8_plan_tile_schedule_entry.flags.
 */
#define RNS8_TILE_SCHEDULE_ZERO_OUTPUT 0x00000001u
#define RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT 0x00000002u

typedef struct rns8_gemm_desc {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_semantics semantics;
  rns8_bound_kind bound_kind;
  rns8_backend_kind requested_backend;
  int64_t m;
  int64_t n;
  int64_t k;
  uint64_t bound;
  uint32_t max_prefix;
  uint32_t tile_m;
  uint32_t tile_n;
  /*
   * Required for RNS8_FINITE_RING_U8 and RNS8_FINITE_FIELD_U8 plans, and must
   * be zero for every non-finite semantic contract. Persistent finite
   * pack/GEMM/export calls still take an explicit modulus and reject any value
   * that differs from this plan-level contract.
   */
  uint32_t finite_modulus;
  /* RNS8_PLAN_* flags. Unknown bits are rejected. */
  uint32_t flags;
  const uint64_t* tile_bounds;
  uint64_t tile_bounds_count;
  /*
   * Required only for RNS8_BOUND_INPUT_RANGE_AND_K. These are explicit
   * per-operand input magnitude contracts; plan creation derives the effective
   * output bound as k * lhs_bound * rhs_bound and rejects contracts that cannot
   * fit the selected bounded output semantic.
   */
  uint64_t lhs_bound;
  uint64_t rhs_bound;
  const uint8_t* zero_a_rows;
  uint64_t zero_a_rows_count;
  const uint8_t* zero_b_cols;
  uint64_t zero_b_cols_count;
} rns8_gemm_desc;

typedef struct rns8_matrix_desc {
  uint64_t struct_size;
  uint32_t abi_version;
  int64_t rows;
  int64_t cols;
  int64_t logical_ld;
  rns8_semantics semantics;
  rns8_layout logical_layout;
  rns8_bound_kind bound_kind;
  uint32_t tile_m;
  uint32_t tile_n;
  uint32_t max_prefix;
  /* Reserved for future hard-cut ABI versions. Must be zero. */
  uint32_t flags;
} rns8_matrix_desc;

typedef struct rns8_plan_schedule_info {
  uint64_t struct_size;
  uint32_t abi_version;
  uint32_t tile_m;
  uint32_t tile_n;
  uint64_t tile_rows;
  uint64_t tile_cols;
  uint64_t tile_count;
  uint32_t min_required_prefix;
  uint32_t max_required_prefix;
  uint32_t min_selected_prefix;
  uint32_t max_selected_prefix;
  uint32_t prefix_group_count;
  uint32_t adaptive_prefix_active;
  uint32_t adaptive_skip_active;
  uint32_t range_bit_length;
  uint32_t flags;
  rns8_bound_kind bound_kind;
  uint32_t reserved0;
  uint64_t effective_bound;
  uint64_t lhs_bound;
  uint64_t rhs_bound;
  uint64_t zero_a_row_count;
  uint64_t zero_b_col_count;
  uint64_t zero_row_col_product_count;
  char bound_contract[96];
} rns8_plan_schedule_info;

typedef struct rns8_plan_tile_schedule_entry {
  uint64_t struct_size;
  uint32_t abi_version;
  uint32_t flags;
  uint64_t tile_row;
  uint64_t tile_col;
  int64_t row_offset;
  int64_t col_offset;
  int64_t row_extent;
  int64_t col_extent;
  uint32_t required_prefix;
  uint32_t selected_prefix;
  uint32_t group_index;
  uint32_t range_bit_length;
} rns8_plan_tile_schedule_entry;

typedef struct rns8_backend_capability_info {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_backend_kind backend;
  uint32_t is_accelerator;
  uint32_t is_available;
  uint32_t is_correctness_backend;
  uint32_t is_matrix_engine_backend;
  uint32_t supports_bounded_rns;
  uint32_t supports_exact_wide_rns;
  uint32_t supports_finite_u8;
  uint32_t supports_wrap64;
  uint32_t requires_feature_detection;
  uint32_t enable_flag_fail_fast;
  uint32_t candidate_evidence_only;
  uint32_t compiled_kernel_available;
  uint32_t exact_differential_validated;
  uint32_t performance_validated;
  uint32_t flags;
  char backend_name[64];
  char selected_kernel[128];
  char library_name[64];
  char library_version[64];
  char enable_flag[64];
  char epilogue_mode[64];
  char workspace_mode[64];
  char isa_evidence[128];
  char status[128];
  char detail[256];
} rns8_backend_capability_info;

typedef struct rns8_plan_backend_info {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_backend_kind backend;
  uint32_t is_accelerator;
  uint32_t is_correctness_backend;
  uint32_t is_matrix_engine_backend;
  uint32_t compiled_kernel_available;
  uint32_t exact_differential_validated;
  uint32_t performance_validated;
  uint32_t flags;
  uint64_t workspace_required_bytes;
  char selected_kernel[128];
  char accelerator_library[64];
  char accelerator_version[64];
  char capability_status[128];
  char epilogue_mode[64];
  char workspace_mode[64];
  char isa_evidence[128];
  char autotune_key[1024];
  uint64_t accumulator_k_block_size;
  uint64_t accumulator_k_block_cap;
  uint64_t accumulator_modulus;
  uint64_t accumulator_max_lhs_abs;
  uint64_t accumulator_max_rhs_abs;
  uint64_t accumulator_max_product;
  uint32_t accumulator_uses_int32_inner_product;
  uint32_t accumulator_safe_for_k_block;
  char accumulator_input_domain[64];
  char accumulator_signedness[64];
  char accumulator_type[64];
  char accumulator_modulus_policy[128];
  char accumulator_safety_status[128];
} rns8_plan_backend_info;

typedef enum rns8_output_domain {
  RNS8_OUTPUT_DOMAIN_RNS_RESIDUE = 1,
  RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64 = 2,
  RNS8_OUTPUT_DOMAIN_FINITE_U8 = 3,
  RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB = 4
} rns8_output_domain;

typedef enum rns8_next_op_flags {
  RNS8_NEXT_OP_FINAL_EXPORT = 1u << 0,
  RNS8_NEXT_OP_RNS_GEMM = 1u << 1,
  RNS8_NEXT_OP_NATIVE_GEMM = 1u << 2,
  RNS8_NEXT_OP_NATIVE_TO_RNS_CONVERTIBLE = 1u << 3,
  RNS8_NEXT_OP_REUSABLE_B_PREPACK = 1u << 4
} rns8_next_op_flags;

typedef struct rns8_plan_packing_info {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_backend_kind backend;
  rns8_semantics semantics;
  uint32_t uses_resident_matrix_inputs;
  uint32_t uses_transient_pack_workspace;
  uint32_t uses_matrix_engine_pack_layout;
  uint32_t reusable_prepack_cache_available;
  uint32_t production_prepack_cache_available;
  uint32_t flags;
  rns8_output_domain input_domain;
  rns8_output_domain output_domain;
  uint32_t output_host_current;
  uint32_t output_device_current;
  uint32_t next_op_flags;
  uint32_t reserved0;
  uint64_t a_pack_workspace_bytes;
  uint64_t b_pack_workspace_bytes;
  uint64_t accumulator_workspace_bytes;
  uint64_t library_workspace_bytes;
  uint64_t total_transient_workspace_bytes;
  char input_domain_name[64];
  char output_domain_name[64];
  char next_op_hint[160];
  char a_layout_version[96];
  char b_layout_version[96];
  char output_layout_version[96];
  char prepack_cache_scope[96];
  char detail[256];
} rns8_plan_packing_info;

#define RNS8_GROUPED_DISPATCH_CONTRACT_SAME_SHAPE_REQUIRED 0x00000001u
#define RNS8_GROUPED_DISPATCH_CONTRACT_COMPACT_ROW_MAJOR_REQUIRED 0x00000002u
#define RNS8_GROUPED_DISPATCH_CONTRACT_UNIQUE_TASK_HANDLES_REQUIRED 0x00000004u
#define RNS8_GROUPED_DISPATCH_CONTRACT_ONE_WORKSPACE_PER_TASK_REQUIRED 0x00000008u
#define RNS8_GROUPED_DISPATCH_CONTRACT_SAME_DEVICE_REQUIRED 0x00000010u
#define RNS8_GROUPED_DISPATCH_CONTRACT_PER_TASK_SOURCE_VERSION_REQUIRED 0x00000020u
#define RNS8_GROUPED_DISPATCH_CONTRACT_DEVICE_CURRENT_OUTPUT_REQUIRED 0x00000040u
#define RNS8_GROUPED_DISPATCH_CONTRACT_BENCHMARK_ONLY_EXECUTION 0x00000080u

typedef struct rns8_grouped_dispatch_contract_info {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_backend_kind backend;
  rns8_semantics semantics;
  uint32_t task_count;
  uint32_t descriptor_contract_supported;
  uint32_t public_execution_available;
  uint32_t same_shape_required;
  uint32_t same_semantics_required;
  uint32_t compact_row_major_required;
  uint32_t unique_matrix_handles_required;
  uint32_t unique_workspace_handles_required;
  uint32_t same_plan_fingerprint_required;
  uint32_t same_device_required;
  uint32_t device_resident_inputs_required;
  uint32_t per_task_source_versions_required;
  uint32_t produces_device_current_output;
  uint32_t final_export_required_for_host_output;
  uint32_t per_task_status_required;
  uint32_t checksum_policy_required;
  uint32_t descriptor_reuse_validated;
  uint32_t flags;
  char descriptor_layout[96];
  char bucket_policy[96];
  char source_version_policy[96];
  char workspace_policy[96];
  char matrix_ownership_policy[128];
  char descriptor_reuse_policy[128];
  char stride_policy[128];
  char output_currentness_policy[128];
  char lifetime_policy[128];
  char checksum_policy[96];
  char status_policy[96];
  char device_descriptor_policy[96];
  char output_domain[64];
  char unsupported_reason[160];
  char detail[256];
} rns8_grouped_dispatch_contract_info;

typedef struct rns8_grouped_gemm_task {
  uint64_t struct_size;
  uint32_t abi_version;
  const rns8_matrix* a;
  const rns8_matrix* b;
  rns8_matrix* c;
  rns8_workspace* workspace;
} rns8_grouped_gemm_task;

typedef struct rns8_matrix_storage_info {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_backend_kind backend;
  rns8_semantics semantics;
  rns8_layout logical_layout;
  rns8_bound_kind bound_kind;
  int64_t rows;
  int64_t cols;
  int64_t logical_ld;
  uint32_t max_prefix;
  uint32_t finite_modulus;
  uint64_t source_version;
  uint32_t host_residues_current;
  uint32_t device_residues_current;
  uint32_t host_byte_limbs_current;
  uint32_t device_byte_limbs_current;
  uint32_t host_native_current;
  uint32_t device_native_current;
  uint32_t uses_residue_storage;
  uint32_t uses_byte_limb_storage;
  uint32_t uses_native_storage;
  int32_t hip_device_id;
  uint32_t flags;
  uint64_t host_residue_bytes;
  uint64_t device_residue_bytes;
  uint64_t host_byte_limb_bytes;
  uint64_t device_byte_limb_bytes;
  uint64_t host_native_bytes;
  uint64_t device_native_bytes;
  char layout_version[96];
  char storage_scope[96];
  char detail[256];
} rns8_matrix_storage_info;

typedef enum rns8_operand_role {
  RNS8_OPERAND_A = 1,
  RNS8_OPERAND_B = 2
} rns8_operand_role;

typedef struct rns8_prepack_cache_key_info {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_backend_kind backend;
  rns8_semantics semantics;
  rns8_operand_role operand_role;
  uint32_t cache_key_valid;
  uint32_t reusable_prepack_cache_available;
  uint32_t production_prepack_cache_available;
  uint32_t flags;
  int32_t hip_device_id;
  uint32_t reserved0;
  int64_t matrix_rows;
  int64_t matrix_cols;
  uint32_t max_prefix;
  uint32_t finite_modulus;
  uint64_t source_version;
  uint64_t plan_fingerprint;
  uint64_t cache_key_hash;
  char matrix_layout_version[96];
  char operand_layout_version[96];
  char cache_scope[96];
  char cache_key[512];
  char detail[256];
} rns8_prepack_cache_key_info;

typedef struct rns8_prepack_cache_info {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_backend_kind backend;
  rns8_semantics semantics;
  rns8_operand_role operand_role;
  uint32_t cache_key_valid;
  uint32_t reusable_prepack_cache_available;
  uint32_t production_prepack_cache_available;
  uint32_t flags;
  int32_t hip_device_id;
  uint32_t reserved0;
  int64_t matrix_rows;
  int64_t matrix_cols;
  int64_t k;
  uint32_t max_prefix;
  uint32_t finite_modulus;
  uint64_t source_version;
  uint64_t plan_fingerprint;
  uint64_t cache_key_hash;
  uint64_t device_bytes;
  uint64_t operand_pack_bytes;
  char matrix_layout_version[96];
  char operand_layout_version[96];
  char cache_scope[96];
  char cache_key[512];
  char detail[256];
} rns8_prepack_cache_info;

/*
 * Public ABI hard-cut status precedence: invalid struct size/version, reserved
 * flags, unknown semantics/bound/layout enum values, or malformed semantic
 * metadata return RNS8_INVALID_ARGUMENT before backend routing. Valid
 * descriptors that request unavailable or future backend enum values return
 * RNS8_UNSUPPORTED_BACKEND.
 */
RNS8_API rns8_status rns8_create_context(
    int device_id,
    const rns8_context_options* options,
    rns8_context** out);

RNS8_API rns8_status rns8_destroy_context(rns8_context* ctx);

RNS8_API rns8_status rns8_get_device_info(
    rns8_context* ctx,
    rns8_device_info* out);

RNS8_API rns8_status rns8_get_backend_capability_info(
    rns8_backend_kind backend,
    rns8_backend_capability_info* out);

RNS8_API rns8_status rns8_create_plan(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    rns8_plan** out);

RNS8_API rns8_status rns8_destroy_plan(rns8_plan* plan);

RNS8_API rns8_status rns8_get_plan_schedule_info(
    const rns8_plan* plan,
    rns8_plan_schedule_info* out);

RNS8_API rns8_status rns8_get_plan_tile_schedule(
    const rns8_plan* plan,
    rns8_plan_tile_schedule_entry* entries,
    uint64_t capacity,
    uint64_t* written);

RNS8_API rns8_status rns8_get_plan_backend_info(
    const rns8_plan* plan,
    rns8_plan_backend_info* out);

/*
 * Report the concrete packing and resident-layout contract for a created plan.
 * The byte counts are derived from the selected backend and plan shape. Current
 * accelerator backends use transient per-dispatch pack workspaces; no reusable
 * production prepack cache is reported until a real cache is implemented and
 * validated.
 */
RNS8_API rns8_status rns8_get_plan_packing_info(
    const rns8_plan* plan,
    rns8_plan_packing_info* out);

/*
 * Report the current grouped-dispatch descriptor and lifetime contract for a
 * created plan. This is read-only introspection. public_execution_available is
 * set only for the narrow same-shape Direct-HIP resident grouped GEMM contracts
 * currently exposed by rns8_gemm_rns_grouped and rns8_gemm_finite_u8_grouped.
 */
RNS8_API rns8_status rns8_get_grouped_dispatch_contract_info(
    const rns8_plan* plan,
    uint32_t task_count,
    rns8_grouped_dispatch_contract_info* out);

RNS8_API rns8_status rns8_create_workspace(
    rns8_context* ctx,
    const rns8_plan* plan,
    rns8_workspace** out);

RNS8_API rns8_status rns8_destroy_workspace(rns8_workspace* workspace);

RNS8_API rns8_status rns8_create_matrix(
    rns8_context* ctx,
    const rns8_matrix_desc* desc,
    rns8_matrix** out);

RNS8_API rns8_status rns8_destroy_matrix(rns8_matrix* matrix);

/*
 * Report the resident storage state for a matrix handle. This exposes the
 * source version, currentness flags, byte counts, and layout version needed to
 * key or reject future prepack caches without mutating the matrix.
 */
RNS8_API rns8_status rns8_get_matrix_storage_info(
    const rns8_matrix* matrix,
    rns8_matrix_storage_info* out);

/*
 * Validate and report deterministic key material for a future reusable prepack
 * cache entry. This does not create a cache or report production cache
 * availability; it rejects incompatible plan, operand role, matrix shape,
 * storage layout, finite-modulus, backend, device id, or currentness before
 * returning a key.
 */
RNS8_API rns8_status rns8_get_prepack_cache_key_info(
    const rns8_plan* plan,
    const rns8_matrix* matrix,
    rns8_operand_role operand_role,
    rns8_prepack_cache_key_info* out);

/*
 * Create or destroy a reusable accelerator prepack cache. Current production
 * runtime code supports a narrow rocWMMA B-operand RNS cache for non-tiled
 * plans with K <= 65536; unsupported roles, backends, or shapes return
 * RNS8_UNSUPPORTED_BACKEND instead of silently falling back to transient pack
 * workspaces.
 */
RNS8_API rns8_status rns8_create_prepack_cache(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* matrix,
    rns8_operand_role operand_role,
    rns8_prepack_cache** out);

/*
 * Report deterministic metadata for a created reusable prepack cache. This is
 * a read-only inspection API: it exposes the cache key, source version, device
 * id, layout versions, and allocation byte counts without making the cache a
 * production autotune artifact.
 */
RNS8_API rns8_status rns8_get_prepack_cache_info(
    const rns8_prepack_cache* cache,
    rns8_prepack_cache_info* out);

RNS8_API rns8_status rns8_destroy_prepack_cache(rns8_prepack_cache* cache);

RNS8_API rns8_status rns8_pack_i64(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const int64_t* src,
    int64_t ld,
    uint64_t source_version);

RNS8_API rns8_status rns8_pack_u64(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const uint64_t* src,
    int64_t ld,
    uint64_t source_version);

/*
 * Pack persistent finite-ring or finite-field uint8_t host storage into a
 * resident finite matrix. The matrix descriptor semantics select ring versus
 * field validation. The modulus is explicit on every finite persistent call:
 * RNS8_FINITE_RING_U8 accepts moduli in [2, 256], while RNS8_FINITE_FIELD_U8
 * accepts prime moduli <= 251. The matrix descriptor must use
 * RNS8_BOUND_NONE, max_prefix = 0, row-major layout, and no CRT/bounded
 * metadata. Successful pack stamps the resident matrix with this modulus;
 * later finite GEMM/export calls reject mismatched input or output modulus.
 */
RNS8_API rns8_status rns8_pack_finite_u8(
    rns8_context* ctx,
    rns8_matrix* matrix,
    uint16_t modulus,
    const uint8_t* src,
    int64_t ld,
    uint64_t source_version);

RNS8_API rns8_status rns8_gemm_rns(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace);

/*
 * Execute a same-shape group of resident RNS GEMMs through the Direct-HIP
 * grouped descriptor path. Inputs must already be packed/current resident
 * matrices for this plan; the grouped call does not perform host packing,
 * native-to-RNS conversion, final host export, or AUTO routing.
 */
RNS8_API rns8_status rns8_gemm_rns_grouped(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_grouped_gemm_task* tasks,
    uint32_t task_count);

/*
 * GEMM variant that consumes a reusable B prepack cache created from the same
 * plan and source-versioned B matrix. Currently implemented for the narrow
 * rocWMMA non-tiled RNS B-cache path only.
 */
RNS8_API rns8_status rns8_gemm_rns_prepacked_b(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_prepack_cache* B,
    rns8_matrix* C,
    rns8_workspace* workspace);

RNS8_API rns8_status rns8_gemm_wrap_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace);

/*
 * Persistent finite-ring/finite-field GEMM over resident uint8_t matrices.
 * The plan and all matrices must have matching finite semantics,
 * RNS8_BOUND_NONE, max_prefix = 0, row-major resident storage, and an explicit
 * descriptor finite_modulus valid for that semantic. The modulus argument must
 * match the plan descriptor. Inputs must already be packed with the same
 * modulus. Output is resident centered finite residues and must be exported
 * with rns8_export_finite_u8 using the same modulus. This API does not route
 * through bounded CRT, exact-wide export, or strict mod 2^64 byte-limb paths.
 */
RNS8_API rns8_status rns8_gemm_finite_u8(
    rns8_context* ctx,
    const rns8_plan* plan,
    uint16_t modulus,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace);

/*
 * Execute a same-shape group of resident finite-u8 GEMMs through the Direct-HIP
 * grouped descriptor path. The modulus stays explicit as in
 * rns8_gemm_finite_u8. Inputs must already be packed/current for the same
 * modulus; the grouped call does not perform host packing or final export.
 */
RNS8_API rns8_status rns8_gemm_finite_u8_grouped(
    rns8_context* ctx,
    const rns8_plan* plan,
    uint16_t modulus,
    const rns8_grouped_gemm_task* tasks,
    uint32_t task_count);

RNS8_API rns8_status rns8_export_i64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    int64_t* dst,
    int64_t ld);

RNS8_API rns8_status rns8_export_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld);

RNS8_API rns8_status rns8_export_wrap_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld);

/* Export persistent finite-ring/finite-field output as canonical uint8_t
 * residues. The output matrix must be resident-current and stamped with the
 * same explicit modulus supplied here. For modulus <= 255, every output byte is
 * in [0, modulus - 1]; for modulus 256, the full byte is the canonical
 * representative. Padded host output columns outside plan->n are preserved.
 */
RNS8_API rns8_status rns8_export_finite_u8(
    rns8_context* ctx,
    const rns8_plan* plan,
    uint16_t modulus,
    const rns8_matrix* C,
    uint8_t* dst,
    int64_t ld);

/* Export persistent RNS output for RNS8_EXACT_WIDE_SIGNED only.
 *
 * Output is row-major by matrix element. `ld` is a leading dimension in
 * elements, not limbs, and each element stores exactly `limb_count` contiguous
 * little-endian uint64_t limbs:
 *
 *   dst[((row * ld) + col) * limb_count + limb]
 *
 * `limb_count` is the fixed output width and must be in [1, 32].
 *
 * The reconstructed centered integer uses the selected modulus product P and
 * maps canonical CRT value x to x - P when x >= ceil(P / 2). It must fit the
 * fixed-width signed range [-2^(64 * limb_count - 1),
 * 2^(64 * limb_count - 1) - 1]. Successful exports use two's-complement
 * representation in exactly the requested width. Too few limbs return
 * RNS8_RANGE_ERROR; invalid handles, invalid leading dimensions, null
 * destinations, and invalid limb counts return RNS8_INVALID_ARGUMENT. Range
 * errors preserve the caller's destination storage. The value is never
 * truncated, saturated, or treated as mod 2^64 wraparound. This API is separate
 * from bounded i64/u64 export and from strict mod 2^64 wraparound byte-limb
 * export. The plan and matrix handles must both be exact-wide signed,
 * bound-none RNS handles; bounded, wrap64, stale-prefix, or cross-semantics
 * handles are malformed for this export surface. Direct HIP exact-wide export
 * requires device-current resident RNS output and does not perform an implicit
 * host-to-device upload.
 */
RNS8_API rns8_status rns8_export_exact_wide_signed_limbs(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count);

/* Export persistent RNS output for RNS8_EXACT_WIDE_UNSIGNED only.
 *
 * Layout matches rns8_export_exact_wide_signed_limbs: row-major elements,
 * element-stride `ld`, and exactly `limb_count` little-endian uint64_t limbs per
 * element at dst[((row * ld) + col) * limb_count + limb].
 *
 * `limb_count` is the fixed output width and must be in [1, 32].
 *
 * The reconstructed canonical integer must fit the fixed-width unsigned range
 * [0, 2^(64 * limb_count) - 1]. Successful exports use magnitude limbs in
 * exactly the requested width. Too few limbs return RNS8_RANGE_ERROR; invalid
 * handles, invalid leading dimensions, null destinations, and invalid limb
 * counts return RNS8_INVALID_ARGUMENT. Range errors preserve the caller's
 * destination storage. The value is never truncated, saturated, or treated as
 * strict mod 2^64 wraparound. This API is separate from bounded i64/u64 export
 * and from strict mod 2^64 wraparound byte-limb export. The plan and matrix
 * handles must both be exact-wide unsigned, bound-none RNS handles; bounded,
 * wrap64, stale-prefix, or cross-semantics handles are malformed for this
 * export surface. Direct HIP exact-wide export requires device-current
 * resident RNS output and does not perform an implicit host-to-device upload.
 */
RNS8_API rns8_status rns8_export_exact_wide_unsigned_limbs(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count);

RNS8_API rns8_status rns8_gemm_i64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const int64_t* A,
    int64_t lda,
    const int64_t* B,
    int64_t ldb,
    int64_t* C,
    int64_t ldc);

RNS8_API rns8_status rns8_gemm_u64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    uint64_t* C,
    int64_t ldc);

RNS8_API rns8_status rns8_gemm_wrap_u64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    uint64_t* C,
    int64_t ldc);

/*
 * One-shot finite-ring GEMM over uint8_t storage with an explicit modulus.
 *
 * The descriptor must use RNS8_FINITE_RING_U8, RNS8_BOUND_NONE, bound = 0,
 * max_prefix = 0, finite_modulus = `modulus`, no tile bounds, and row-major
 * byte matrices. `modulus` must be in [2, 256]. Inputs are reduced modulo
 * `modulus`; outputs are canonical residues in [0, modulus - 1] for
 * modulus <= 255 and full bytes for modulus == 256. This API is separate from
 * bounded CRT, exact-wide export, and strict mod 2^64 wraparound.
 */
RNS8_API rns8_status rns8_gemm_finite_ring_u8_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    uint16_t modulus,
    const uint8_t* A,
    int64_t lda,
    const uint8_t* B,
    int64_t ldb,
    uint8_t* C,
    int64_t ldc);

/*
 * One-shot finite-field GEMM over uint8_t storage with an explicit prime
 * modulus. Contract matches rns8_gemm_finite_ring_u8_oneshot, except
 * `modulus` must be prime and <= 251, and the descriptor semantics must be
 * RNS8_FINITE_FIELD_U8.
 */
RNS8_API rns8_status rns8_gemm_finite_field_u8_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    uint16_t modulus,
    const uint8_t* A,
    int64_t lda,
    const uint8_t* B,
    int64_t ldb,
    uint8_t* C,
    int64_t ldc);

#ifdef __cplusplus
}
#endif

#endif
