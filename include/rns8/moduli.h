#ifndef RNS8_MODULI_H
#define RNS8_MODULI_H

#include <stdint.h>

#include "rns8/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RNS8_DEFAULT_MODULUS_COUNT 28u
#define RNS8_DEFAULT_BOUNDED_PREFIX 9u
#define RNS8_MAX_SUPPORTED_PREFIX 20u
#define RNS8_SAFE_INT32_K_BLOCK 65536u

RNS8_API uint32_t rns8_default_modulus_count(void);
RNS8_API uint16_t rns8_default_modulus(uint32_t index);
RNS8_API double rns8_prefix_range_bits(uint32_t prefix);
RNS8_API rns8_status rns8_validate_default_moduli(void);

#ifdef __cplusplus
}
#endif

#endif

