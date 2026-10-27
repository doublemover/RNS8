#include <iostream>

#if defined(RNS8_ENABLE_GMP)
#include <gmp.h>
#endif

#if defined(RNS8_ENABLE_FLINT)
#include <flint/flint.h>
#include <flint/fmpz.h>
#endif

int main() {
#if defined(RNS8_ENABLE_GMP)
  if (gmp_version == nullptr || gmp_version[0] == '\0') {
    return 2;
  }
  mpz_t gmp_a, gmp_b, gmp_product, gmp_expected;
  mpz_inits(gmp_a, gmp_b, gmp_product, gmp_expected, nullptr);
  mpz_set_ui(gmp_a, 1);
  mpz_mul_2exp(gmp_a, gmp_a, 130);
  mpz_add_ui(gmp_a, gmp_a, 3);
  mpz_set_si(gmp_b, -7);
  mpz_mul(gmp_product, gmp_a, gmp_b);
  mpz_set_si(gmp_expected, -7);
  mpz_mul_2exp(gmp_expected, gmp_expected, 130);
  mpz_sub_ui(gmp_expected, gmp_expected, 21);
  const bool gmp_exact = mpz_cmp(gmp_product, gmp_expected) == 0;
  mpz_clears(gmp_a, gmp_b, gmp_product, gmp_expected, nullptr);
  if (!gmp_exact) return 4;
  std::cout << "GMP " << gmp_version << '\n';
#endif

#if defined(RNS8_ENABLE_FLINT)
  if (flint_version[0] == '\0') {
    return 3;
  }
  fmpz_t flint_a, flint_product, flint_expected;
  fmpz_init(flint_a); fmpz_init(flint_product); fmpz_init(flint_expected);
  fmpz_one(flint_a);
  fmpz_mul_2exp(flint_a, flint_a, 130);
  fmpz_add_ui(flint_a, flint_a, 3);
  fmpz_mul_si(flint_product, flint_a, -7);
  fmpz_set_si(flint_expected, -7);
  fmpz_mul_2exp(flint_expected, flint_expected, 130);
  fmpz_sub_ui(flint_expected, flint_expected, 21);
  const bool flint_exact = fmpz_equal(flint_product, flint_expected) != 0;
  fmpz_clear(flint_a); fmpz_clear(flint_product); fmpz_clear(flint_expected);
  if (!flint_exact) return 5;
  std::cout << "FLINT " << flint_version << '\n';
#endif

  return 0;
}
