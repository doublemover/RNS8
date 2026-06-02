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
  /* Reserved for future hard-cut ABI versions. Must be zero. */
  uint32_t flags;
  const uint64_t* tile_bounds;
  uint64_t tile_bounds_count;
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
  char isa_evidence[64];
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
  char isa_evidence[64];
  char autotune_key[256];
} rns8_plan_backend_info;

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
 * modulus valid for that semantic. Inputs must already be packed with the same
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
 * max_prefix = 0, no tile bounds, and row-major byte matrices. `modulus` must
 * be in [2, 256]. Inputs are reduced modulo `modulus`; outputs are canonical
 * residues in [0, modulus - 1] for modulus <= 255 and full bytes for
 * modulus == 256. This API is separate from bounded CRT, exact-wide export,
 * and strict mod 2^64 wraparound.
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
