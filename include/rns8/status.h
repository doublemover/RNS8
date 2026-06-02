#ifndef RNS8_STATUS_H
#define RNS8_STATUS_H

#include <stdint.h>

#define RNS8_ABI_VERSION 1u

#if defined(RNS8_STATIC)
#  define RNS8_API
#elif defined(_WIN32)
#  if defined(RNS8_BUILDING_LIBRARY)
#    define RNS8_API __declspec(dllexport)
#  else
#    define RNS8_API __declspec(dllimport)
#  endif
#else
#  define RNS8_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rns8_status {
  RNS8_SUCCESS = 0,
  RNS8_INVALID_ARGUMENT = 1,
  RNS8_UNSUPPORTED_OS = 2,
  RNS8_UNSUPPORTED_ARCH = 3,
  RNS8_UNSUPPORTED_BACKEND = 4,
  RNS8_RANGE_ERROR = 5,
  RNS8_ACCUMULATION_OVERFLOW_RISK = 6,
  RNS8_WORKSPACE_TOO_SMALL = 7,
  RNS8_BACKEND_FAILURE = 8,
  RNS8_VERIFICATION_FAILED = 9,
  RNS8_INTERNAL_ERROR = 10
} rns8_status;

RNS8_API const char* rns8_status_string(rns8_status status);

#ifdef __cplusplus
}
#endif

#endif

