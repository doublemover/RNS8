#ifndef RNS8_BOUNDS_H
#define RNS8_BOUNDS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rns8_bound_kind {
  RNS8_BOUND_NONE = 0,
  RNS8_BOUND_GLOBAL_MAX_ABS = 1,
  RNS8_BOUND_GLOBAL_MAX_UNSIGNED = 2,
  RNS8_BOUND_PER_TILE_MAX_ABS = 3,
  RNS8_BOUND_PER_TILE_MAX_UNSIGNED = 4,
  RNS8_BOUND_INPUT_RANGE_AND_K = 5
} rns8_bound_kind;

#ifdef __cplusplus
}
#endif

#endif

