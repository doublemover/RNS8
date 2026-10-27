#ifndef RNS8_INTEGER_SPLIT_H
#define RNS8_INTEGER_SPLIT_H

#include "rns8/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rns8_integer_split_format RNS8_ENUM_BASE {
  RNS8_INTEGER_SPLIT_SIGNED_RADIX128 = 1
} rns8_integer_split_format;

typedef struct rns8_integer_split_i64_desc {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_integer_split_format format;
  int64_t rows, cols, native_ld;
  /* Byte pitches. Zero selects cols and rows*component_ld respectively.
   * An explicit plane stride must cover rows*component_ld bytes. */
  int64_t component_ld, component_plane_stride;
} rns8_integer_split_i64_desc;

typedef struct rns8_integer_split_i64_info {
  uint64_t struct_size;
  uint32_t abi_version;
  rns8_integer_split_format format;
  uint32_t radix_bits, component_count, layout_version, reserved;
  int64_t component_ld, component_plane_stride;
  uint64_t native_span_bytes, required_component_bytes;
  int32_t digit_min, digit_max;
} rns8_integer_split_i64_info;

/* Exact host integer representation, independent of any GPU/backend/context.
 * Each logical x has ten signed radix128 digits, low component first:
 *   x = sum(d[s]*2^(7*s)), s=0..9.
 * Digits are sign(x) times the unsigned magnitude digits (including INT64_MIN),
 * so -127<=d<=127 and all nonzero digits have the same sign. The last digit
 * has magnitude at most1. This format contains no floating encodings/scales.
 * All buffers are caller-owned; calls allocate nothing and retain no pointers.
 * ABI/shape/format/span failure leaves query output unchanged. */
RNS8_API rns8_status rns8_query_integer_split_i64(
    const rns8_integer_split_i64_desc*, rns8_integer_split_i64_info*);

/* Synchronous split/merge of host-accessible buffers. native_ld counts int64
 * elements; component addresses are s*plane_stride + row*component_ld + col.
 * Capacity counts accessible component bytes through the final logical cell;
 * it must cover the queried required_component_bytes. Native buffers must
 * cover native_span_bytes. Logical cells alone are read/written; all padding
 * remains untouched. The conservative pitched native/component spans must be
 * disjoint, aligned as their element types require and addressable by ptrdiff_t.
 * Invalid arguments, address/range overflow, insufficient capacity and malformed
 * component input leave the entire destination unchanged. Merge rejects -128,
 * mixed-sign digits, a noncanonical high digit and out-of-int64 magnitudes.
 * Callers must keep descriptors/input bytes stable throughout the call.
 * These functions neither execute GEMM nor approximate floating-point Ozaki.
 */
RNS8_API rns8_status rns8_split_i64(
    const rns8_integer_split_i64_desc*, const int64_t* native,
    int8_t* components, uint64_t component_capacity_bytes);
RNS8_API rns8_status rns8_merge_i64(
    const rns8_integer_split_i64_desc*, const int8_t* components,
    uint64_t component_capacity_bytes, int64_t* native);

#ifdef __cplusplus
}
#endif
#endif
