#ifndef RNS8_CORE_BACKEND_COMMON_HPP
#define RNS8_CORE_BACKEND_COMMON_HPP

#include <cstdint>
#include <functional>

#include "rns8/status.h"

namespace rns8::detail {

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t& out);
bool round_up_aligned_u64(uint64_t value, uint64_t alignment, uint64_t& out);
int run_timed_device_code(const char* label, const std::function<int()>& fn);
rns8_status status_from_device_code(int code);

}  // namespace rns8::detail

#endif
