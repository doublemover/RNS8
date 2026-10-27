#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "rns8_generated_centered_reducers.cuh"

namespace {
template<std::size_t Index>
void check_generated_reducer() {
  constexpr uint16_t modulus = rns8::generated::moduli[Index];
  using Reducer = rns8::generated::CenteredReducer<modulus>;
  auto check = [=](int32_t value) {
    // Widen first so the independent formula handles INT32_MIN without negation.
    int64_t canonical = static_cast<int64_t>(value) % modulus;
    if (canonical < 0) canonical += modulus;
    if (canonical >= (modulus + 1) / 2) canonical -= modulus;
    REQUIRE(static_cast<int64_t>(Reducer::apply(value)) == canonical);
  };
  for (int32_t value = -4096; value <= 4096; ++value) check(value);
  for (int32_t value : {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min() + 1,
                        std::numeric_limits<int32_t>::max() - 1, std::numeric_limits<int32_t>::max()})
    check(value);
  STATIC_REQUIRE(Reducer::safe_k == 65536);
  STATIC_REQUIRE(sizeof(Reducer::certificate_sha256) == 65);
  if constexpr (Index + 1 < sizeof(rns8::generated::moduli) / sizeof(uint16_t))
    check_generated_reducer<Index + 1>();
}
}

TEST_CASE("generated centered reducers compile and match widened signed arithmetic") {
  check_generated_reducer<0>();
}
