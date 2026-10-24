#include "backend_common/garner_constants.hpp"
#include "backend_common/integrity_identity.hpp"
#include <catch2/catch_test_macros.hpp>
#include <numeric>
TEST_CASE("integrity identity and independent primes exclude computational "
          "channels") {
  using namespace rns8::detail;
  IntegrityIdentity value{1, 2, 3, 4, 5, 6, 1, 20};
  REQUIRE(integrity_identity_matches(value, value));
  for (auto member :
       {&IntegrityIdentity::operation, &IntegrityIdentity::epoch,
        &IntegrityIdentity::a_instance, &IntegrityIdentity::a_generation,
        &IntegrityIdentity::b_instance, &IntegrityIdentity::b_generation}) {
    auto changed = value;
    ++(changed.*member);
    REQUIRE_FALSE(integrity_identity_matches(value, changed));
  }
  auto changed = value;
  changed.prefix = 19;
  REQUIRE_FALSE(integrity_identity_matches(value, changed));
  changed = value;
  changed.basis_version = 2;
  REQUIRE_FALSE(integrity_identity_matches(value, changed));
  for (uint32_t q : {257u, 263u})
    for (auto m : garner::kConstants.moduli)
      REQUIRE(std::gcd(q, m) == 1);
  REQUIRE(std::gcd(257, 263) == 1);
}
