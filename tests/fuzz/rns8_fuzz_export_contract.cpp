#include "arithmetic_replay.hpp"

using namespace rns8_fuzz;
using rns8_fuzz::replay_support::replay;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
  ArithmeticCase value;
  if (!decode(data, size, value))
    return 0;
  const auto expected = oracle(value);
  // Outside-envelope cases are not valid arithmetic executions. Descriptor
  // admission/status mutations have a separate harness.
  if (expected.admitted)
    replay(std::move(value), expected);
  return 0;
}
