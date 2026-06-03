#include <iostream>

#if defined(RNS8_ENABLE_GMP)
#include <gmp.h>
#endif

#if defined(RNS8_ENABLE_FLINT)
#include <flint/flint.h>
#endif

int main() {
#if defined(RNS8_ENABLE_GMP)
  if (gmp_version == nullptr || gmp_version[0] == '\0') {
    return 2;
  }
  std::cout << "GMP " << gmp_version << '\n';
#endif

#if defined(RNS8_ENABLE_FLINT)
  if (flint_version[0] == '\0') {
    return 3;
  }
  std::cout << "FLINT " << flint_version << '\n';
#endif

  return 0;
}
