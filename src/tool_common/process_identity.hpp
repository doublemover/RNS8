#pragma once

#include <cstdint>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace rns8::tools {
inline uint64_t process_id() noexcept {
#ifdef _WIN32
  return static_cast<uint64_t>(static_cast<uint32_t>(_getpid()));
#else
  return static_cast<uint64_t>(getpid());
#endif
}
}  // namespace rns8::tools
