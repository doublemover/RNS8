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
  uint32_t flags;
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
  uint32_t flags;
} rns8_matrix_desc;

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

RNS8_API rns8_status rns8_export_exact_wide_signed_limbs(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count);

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
