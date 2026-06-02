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

RNS8_API rns8_status rns8_create_context(
    int device_id,
    const rns8_context_options* options,
    rns8_context** out);

RNS8_API rns8_status rns8_destroy_context(rns8_context* ctx);

RNS8_API rns8_status rns8_get_device_info(
    rns8_context* ctx,
    rns8_device_info* out);

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

#ifdef __cplusplus
}
#endif

#endif
