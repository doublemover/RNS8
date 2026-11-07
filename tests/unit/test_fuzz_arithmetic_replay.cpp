#include <catch2/catch_test_macros.hpp>
#include "../fuzz/arithmetic_replay.hpp"
#include "../fuzz/lifecycle_case.hpp"

namespace {
using namespace rns8_fuzz;
using namespace rns8_fuzz::replay_support;
struct CapturedFailure : std::exception {
  Failure value;
  explicit CapturedFailure(const Failure& failure) : value(failure) {}
  const char* what() const noexcept override { return value.phase; }
};
void capture_failure(const Failure& failure) {
  throw CapturedFailure{failure};
}
struct FailureScope {
  FailureHandler old = failure_handler;
  FailureScope() { failure_handler = capture_failure; }
  ~FailureScope() { failure_handler = old; }
};
} // namespace
TEST_CASE("native history models prepared generations publication and retained ownership",
          "[fuzz][history][cpu]") {
  FailureScope failures;
  const uint8_t history[] = {'R', '8', 'H', '1', 3, 4, 9,  0,  5, 3,  6, 5,
                             3,   7,   8,   4,   5, 3, 10, 11, 1, 10, 11};
  rns8_fuzz::history_support::Observation observed;
  REQUIRE_NOTHROW(observed = rns8_fuzz::history_support::replay(history, sizeof(history)));
  CHECK(observed.transitions == 19);
  CHECK(observed.stale_rejections == 1);
  CHECK(observed.successful_executions == 2);
  CHECK(observed.busy_checks == 1);
  CHECK(observed.cache_queries == 2);
  CHECK(observed.unsupported_caches == 2);
  const uint8_t invalid[] = {'R', '8', 'H', '2', 5};
  CHECK(rns8_fuzz::history_support::replay(invalid, sizeof(invalid)).transitions == 0);
  CHECK_FALSE(rns8_fuzz::history_support::admitted(nullptr, 5));
  CHECK_FALSE(rns8_fuzz::history_support::admitted(history, 4));
  std::array<uint8_t, 37> oversized{};
  std::copy_n(history, 4, oversized.begin());
  CHECK_FALSE(rns8_fuzz::history_support::admitted(oversized.data(), oversized.size()));
}
TEST_CASE("native corpus detects production signed result mutations", "[fuzz][replay][mutation]") {
  FailureScope failures;
  ArithmeticCase value;
  value.m = value.n = value.k = 1;
  value.prefix = 20;
  value.a = {uint64_t(-2)};
  value.b = {3};
  const auto expected = oracle(value);
  REQUIRE(expected.admitted);
  REQUIRE(expected.cells.front() == -6);
  REQUIRE_NOTHROW(replay(value, expected));
}
TEST_CASE("native corpus detects production preflight status mutations",
          "[fuzz][replay][mutation]") {
  FailureScope failures;
  ArithmeticCase value;
  value.m = value.n = value.k = 1;
  value.prefix = 20;
  value.operations = 2;
  value.a = {2};
  value.b = {3};
  const auto expected = oracle(value);
  REQUIRE(expected.admitted);
  REQUIRE_NOTHROW(replay(value, expected));
}
TEST_CASE("native corpus detects production stale prepared mutations", "[fuzz][replay][mutation]") {
  FailureScope failures;
  ArithmeticCase value;
  value.m = value.n = value.k = 1;
  value.prefix = 20;
  value.operations = 4;
  value.a = {2};
  value.b = {3};
  const auto expected = oracle(value);
  REQUIRE(expected.admitted);
  REQUIRE_NOTHROW(replay(value, expected));
}
TEST_CASE("shared arithmetic replay checks complete CPU results and real lifecycle counts",
          "[fuzz][replay][cpu]") {
  FailureScope failures;
  for (uint8_t semantics = 0; semantics < 6; ++semantics) {
    ArithmeticCase value;
    value.semantic = semantics;
    value.m = value.n = value.k = 1;
    value.a = {semantics == 2 ? 255u : 2u};
    value.b = {semantics == 2 ? 255u : 3u};
    value.prefix = value.finite() || value.wrap() ? 0 : 20;
    value.modulus = value.finite() ? 254 : 0;
    value.padding_a = 1;
    value.padding_b = 3;
    value.padding_c = 2;
    value.operations = value.finite() || value.wrap() ? 3 : 7;
    const auto expected = oracle(value);
    REQUIRE(expected.admitted);
    Observation observed;
    const auto backend = value.wrap() ? RNS8_BACKEND_WRAP64_BYTE_LIMB : RNS8_BACKEND_CPU_REFERENCE;
    replay(value, expected, backend, -1, &observed);
    CHECK(observed.device_observed);
    CHECK(observed.plan_observed);
    CHECK(observed.plan.backend == backend);
    CHECK(observed.plan.selected_kernel[0]);
    CHECK(observed.gemm_calls == ((value.operations & 4) ? 3 : 2));
    CHECK(observed.comparisons == ((value.operations & 4) ? 4 : 2));
  }
}
TEST_CASE("shared arithmetic replay refutes wrong result status and no-write witnesses",
          "[fuzz][replay][negative-control]") {
  FailureScope failures;
  ArithmeticCase value;
  value.m = value.n = value.k = 1;
  value.a = {2};
  value.b = {3};
  value.prefix = 20;
  const auto expected = oracle(value);
  try {
    compare<int64_t>(value, expected, 1, [](int64_t* output, int64_t) {
      *output = 7;
      return RNS8_SUCCESS;
    });
    FAIL("wrong successful value was accepted");
  } catch (const CapturedFailure& failure) {
    CHECK(std::string(failure.value.kind) == "value_or_padding");
    CHECK(failure.value.storage_element == 1);
    CHECK(failure.value.actual_value == "7");
    CHECK(failure.value.expected_value == "6");
  }
  try {
    compare<int64_t>(value, expected, 1, [](int64_t*, int64_t) { return RNS8_RANGE_ERROR; });
    FAIL("wrong export status was accepted");
  } catch (const CapturedFailure& failure) {
    CHECK(std::string(failure.value.kind) == "status");
    CHECK(failure.value.actual == RNS8_RANGE_ERROR);
    CHECK(failure.value.expected == RNS8_SUCCESS);
  }
  value.operations = 2;
  try {
    compare<int64_t>(value, expected, 1, [](int64_t* output, int64_t) {
      *output = 7;
      return RNS8_INVALID_ARGUMENT;
    });
    FAIL("invalid-LD write was accepted");
  } catch (const CapturedFailure& failure) {
    CHECK(std::string(failure.value.kind) == "no_write");
  }
}
