#ifndef RNS8_PUBLIC_ABI_H
#define RNS8_PUBLIC_ABI_H

#include "rns8/status.h"

#define RNS8_PUBLIC_ABI_METRICS_VERSION 1u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rns8_public_abi_metric {
  const char* name;
  uint64_t value;
} rns8_public_abi_metric;

typedef struct rns8_public_abi_info {
  uint64_t struct_size;
  uint32_t abi_version;
  uint32_t metric_schema_version;
  uint64_t metric_count;
  const rns8_public_abi_metric* metrics;
  const char* declaration_sha256;
  const char* compiler_id;
  const char* runtime_model;
} rns8_public_abi_info;

/* CPU-only, allocation-free inspection of this compiled library's public C ABI.
 * Supply a readable eight-byte struct_size prefix; success requires exactly
 * sizeof(rns8_public_abi_info) bytes and the current abi_version. Failure
 * preserves every output byte.
 * The returned sorted table/strings are immutable and owned by the loaded
 * library; never free them or retain them after unloading it. Values describe
 * actual compiled sizes, alignments, offsets, enum values and public constants.
 * This does not qualify GPU code, all runtime contracts or C++ source compatibility.
 */
RNS8_API rns8_status rns8_get_public_abi_info(rns8_public_abi_info* out);

#ifdef __cplusplus
}
#endif
#endif
