#pragma once

// Core SDK 10.1's host E5M3 decoder calls the Clang/GCC clz builtin even
// under native MSVC. Keep that adaptation local to the third-party include;
// AMD Clang device compilation still uses the SDK's native implementation.
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
namespace rns8::detail {
inline int hipblaslt_host_clz(unsigned int value) noexcept {
  unsigned long index = 0;
  return _BitScanReverse(&index, value) ? 31 - int(index) : 32;
}
}
#pragma push_macro("__builtin_clz")
#undef __builtin_clz
#define __builtin_clz(value) ::rns8::detail::hipblaslt_host_clz(value)
#endif

#include <hipblaslt/hipblaslt.h>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma pop_macro("__builtin_clz")
#endif
