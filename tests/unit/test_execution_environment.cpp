#include <catch2/catch_test_macros.hpp>
#include <optional>
#include <sstream>
#include <cstdlib>
#include "tool_common/execution_environment_json.hpp"

namespace {
constexpr const char* fixture_key = "RNS8_ENV_WITNESS_FIXTURE";
void assign_fixture(const char* value) {
#if defined(_WIN32)
  if (_putenv_s(fixture_key, value ? value : "") != 0)
    throw std::runtime_error("authored environment fixture setup failed");
#else
  const int status = value ? ::setenv(fixture_key, value, 1) : ::unsetenv(fixture_key);
  if (status)
    throw std::runtime_error("authored environment fixture setup failed");
#endif
}
struct RestoreFixture {
  std::optional<std::string> original;
  RestoreFixture() {
#if defined(_WIN32)
    char* value = nullptr;
    std::size_t count = 0;
    if (_dupenv_s(&value, &count, fixture_key) != 0)
      throw std::runtime_error("authored environment fixture read failed");
    if (value)
      original = value;
    std::free(value);
#else
    if (const char* value = std::getenv(fixture_key))
      original = value;
#endif
  }
  ~RestoreFixture() {
    try {
      assign_fixture(original ? original->c_str() : nullptr);
    } catch (...) {
    }
  }
};
nlohmann::json serialize(const rns8::tools::ExecutionEnvironmentWitness& witness) {
  std::ostringstream out;
  witness.print(out);
  return nlohmann::json::parse(out.str());
}
} // namespace

TEST_CASE("disabled execution observations do not dereference argv", "[cpu][identity]") {
  rns8::tools::ExecutionEnvironmentWitness witness(false, 0, nullptr);
  witness.finish();
  const auto value = serialize(witness);
  CHECK(value.at("state") == "not_requested");
  CHECK(value.at("arguments").is_null());
  CHECK(value.at("before").is_null());
}

TEST_CASE("native execution observations bind actual arguments and environment changes",
          "[cpu][identity]") {
  RestoreFixture restore;
  assign_fixture("before");
  char program[] = "authored-benchmark";
  char option[] = "--record-execution-environment";
  char* arguments[]{program, option};
  rns8::tools::ExecutionEnvironmentWitness witness(true, 2, arguments);
  CHECK(serialize(witness).at("state") == "incomplete");
  witness.finish();
  const auto stable = serialize(witness);
  CHECK(stable.at("state") == "stable");
  CHECK(stable.at("arguments") == nlohmann::json::array({option}));
  CHECK(stable.at("before").at(fixture_key) == "before");
  assign_fixture("after");
  witness.finish();
  const auto changed = serialize(witness);
  CHECK(changed.at("state") == "changed");
  CHECK(changed.at("after").at(fixture_key) == "after");
  using rns8::tools::environment_observation_detail::relevant;
  CHECK(relevant("ROCPROFILER_TOOL_CONFIG"));
  CHECK(relevant("HIP_VISIBLE_DEVICES"));
  CHECK(relevant("LD_PRELOAD"));
  CHECK_FALSE(relevant("UNRELATED_AUTH_TOKEN"));
}
