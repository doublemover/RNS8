#ifndef RNS8_CPU_I8MM_H
#define RNS8_CPU_I8MM_H
#include "rns8/rns8.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct rns8_prepared_cpu_i8mm rns8_prepared_cpu_i8mm;
typedef struct rns8_cpu_i8mm_info {
  uint64_t struct_size;
  uint32_t abi_version,compiled,linux_aarch64,os_asimd,os_i8mm,admitted;
  uint32_t safe_k,layout_version,last_execution_threads,reserved;
  uint64_t packed_b_bytes,output_staging_bytes,b_pack_count,b_reuse_count;
  uint64_t a_generation,b_generation;
} rns8_cpu_i8mm_info;
/* Write-free runtime admission; never executes SMMLA or changes process state. */
RNS8_API rns8_status rns8_query_cpu_i8mm(rns8_cpu_i8mm_info*);
/* Explicit synchronous Arm integer-matrix owner over ordinary CPU-resident RNS
 * matrices and a CPU-reference semantic plan. It executes real SMMLA, never the
 * scalar oracle. Supports uniform bounded/wide/finite contracts; wrap/adaptive
 * and experimental bases reject. Payload budget covers packed B and staged C.
 * Preparation retains all inputs, plan, context and output. */
RNS8_API rns8_status rns8_prepare_cpu_i8mm(rns8_context*,const rns8_plan*,const rns8_matrix*,const rns8_matrix*,rns8_matrix*,uint64_t max_payload_bytes,rns8_prepared_cpu_i8mm**);
RNS8_API rns8_status rns8_execute_cpu_i8mm(rns8_prepared_cpu_i8mm*);
/* Accept current generations within the original proof envelope. B is rebuilt
 * only when its instance/generation changes; A is staged on demand by tiles. */
RNS8_API rns8_status rns8_refresh_cpu_i8mm(rns8_prepared_cpu_i8mm*);
RNS8_API rns8_status rns8_get_cpu_i8mm_info(const rns8_prepared_cpu_i8mm*,rns8_cpu_i8mm_info*);
RNS8_API rns8_status rns8_destroy_cpu_i8mm(rns8_prepared_cpu_i8mm*);
#ifdef __cplusplus
}
#endif
#endif
