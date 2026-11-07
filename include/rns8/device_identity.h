#ifndef RNS8_DEVICE_IDENTITY_H
#define RNS8_DEVICE_IDENTITY_H

#include "rns8/status.h"

enum { RNS8_DEVICE_IDENTITY_PCI_BDF = 1u, RNS8_DEVICE_IDENTITY_UUID = 2u };

typedef struct rns8_runtime_device_identity {
  uint64_t struct_size;
  uint32_t abi_version;
  int32_t runtime_ordinal;
  uint32_t valid_fields;
  uint32_t reserved;
  uint8_t uuid[16];
  char pci_bdf[32];
  uint32_t native_pci_status;
  uint32_t native_uuid_status;
} rns8_runtime_device_identity;

#ifdef __cplusplus
extern "C" {
#endif

/* Query an ordinal in this process's current HIP visibility namespace.
 * Supply a readable eight-byte struct_size prefix; success requires exactly
 * sizeof(rns8_runtime_device_identity) bytes and the current ABI version.
 * Failure preserves output bytes. A successful inquiry can have valid_fields
 * zero when the runtime supplies neither portable identity; this is unknown,
 * not proof of a physical-device association. UUID bytes are opaque HIP data.
 * No device switch, allocation, kernel launch or synchronization is requested.
 */
RNS8_API rns8_status rns8_query_runtime_device_identity(int device_ordinal,
                                                        rns8_runtime_device_identity* out);

#ifdef __cplusplus
}
#endif
#endif
