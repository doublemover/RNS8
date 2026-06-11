#ifndef RNS8_SEMANTICS_H
#define RNS8_SEMANTICS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rns8_semantics {
  RNS8_BOUNDED_I64 = 1,
  RNS8_BOUNDED_U64 = 2,
  RNS8_EXACT_WIDE_SIGNED = 3,
  RNS8_EXACT_WIDE_UNSIGNED = 4,
  RNS8_WRAP_U64_MOD_2_64 = 5,
  RNS8_FINITE_RING_U8 = 6,
  RNS8_FINITE_FIELD_U8 = 7,

  // === Research semantics (schema-gated, not in default builds) ===
  RNS8_INT4_RESEARCH = 100,
  RNS8_IU4_RESEARCH = 101,
  RNS8_OZAKI_FP8_RESEARCH = 200,
  RNS8_STRASSEN_RESEARCH = 201,
  RNS8_FREIVALDS_RESEARCH = 202
} rns8_semantics;

typedef enum rns8_layout {
  RNS8_LAYOUT_ROW_MAJOR = 1,
  RNS8_LAYOUT_COLUMN_MAJOR = 2
} rns8_layout;

typedef enum rns8_backend_kind {
  RNS8_BACKEND_AUTO = 0,
  RNS8_BACKEND_CPU_REFERENCE = 1,
  RNS8_BACKEND_HIP_DIRECT = 2,
  RNS8_BACKEND_HIPBLASLT = 3,
  RNS8_BACKEND_CK = 4,
  RNS8_BACKEND_ROCWMMA = 5,
  RNS8_BACKEND_WRAP64_BYTE_LIMB = 6,
  RNS8_BACKEND_HIP_VECTOR_ALU_INT64 = 7,
  RNS8_BACKEND_AMDGPU_BUILTINS = 8
} rns8_backend_kind;

#ifdef __cplusplus
}
#endif

#endif
