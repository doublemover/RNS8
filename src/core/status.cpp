#include "rns8/status.h"

const char* rns8_status_string(rns8_status status) {
  switch (status) {
    case RNS8_SUCCESS:
      return "success";
    case RNS8_INVALID_ARGUMENT:
      return "invalid argument";
    case RNS8_UNSUPPORTED_OS:
      return "unsupported operating system";
    case RNS8_UNSUPPORTED_ARCH:
      return "unsupported architecture";
    case RNS8_UNSUPPORTED_BACKEND:
      return "unsupported backend";
    case RNS8_RANGE_ERROR:
      return "range error";
    case RNS8_ACCUMULATION_OVERFLOW_RISK:
      return "accumulation overflow risk";
    case RNS8_WORKSPACE_TOO_SMALL:
      return "workspace too small";
    case RNS8_BACKEND_FAILURE:
      return "backend failure";
    case RNS8_VERIFICATION_FAILED:
      return "verification failed";
    case RNS8_INTERNAL_ERROR:
      return "internal error";
  }
  return "unknown status";
}

