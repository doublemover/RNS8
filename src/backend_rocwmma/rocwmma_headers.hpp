#pragma once
#include <hip/hip_runtime.h>

// rocWMMA's floating wrapper traits intentionally specialize std traits. MSVC
// 14.51 adds clang::no_specializations to those templates. Keep this vendor
// compatibility diagnostic scoped to the dependency include; RNS8 uses its
// integer fragments, and project code retains the normal diagnostic policy.
#if defined(__clang__) && defined(_MSC_VER) && _MSC_VER >= 1950
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winvalid-specialization"
#endif
#include <rocwmma/rocwmma.hpp>
#if defined(__clang__) && defined(_MSC_VER) && _MSC_VER >= 1950
#pragma clang diagnostic pop
#endif
