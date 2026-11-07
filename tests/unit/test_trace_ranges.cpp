#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "diagnostics/trace_ranges.hpp"
#include "diagnostics/trace_label.hpp"

namespace {
nlohmann::json range_snapshot() {
  std::ostringstream out;
  rns8::detail::trace::write_snapshot(out);
  return nlohmann::json::parse(out.str());
}
} // namespace

TEST_CASE("native marker labels bind exact snapshot slots without allocation",
          "[cpu][trace][identity]") {
  char label[80]{};
  using rns8::detail::trace::format_marker_label;
  REQUIRE(format_marker_label(label, sizeof(label), "rns8.public_call", 0));
  CHECK(std::string(label) == "rns8.public_call#slot=0");
  REQUIRE(format_marker_label(label, sizeof(label), "rns8.completion", UINT64_MAX));
  CHECK(std::string(label) == "rns8.completion#slot=18446744073709551615");
  char small[] = "unchanged";
  CHECK_FALSE(format_marker_label(small, sizeof(small), "rns8.proof", 0));
  CHECK(std::string(small) == "unchanged");
  CHECK_FALSE(format_marker_label(nullptr, sizeof(label), "rns8.proof", 0));
  CHECK_FALSE(format_marker_label(label, sizeof(label), {}, 0));
}

TEST_CASE("profiler ranges distinguish disabled and unavailable providers", "[cpu][trace]") {
  const auto snapshot = range_snapshot();
  CHECK(snapshot.at("schema") == "rns8_profiler_ranges_v1");
#if !defined(RNS8_ENABLE_ROCTX) || !RNS8_ENABLE_ROCTX
  CHECK(snapshot.at("provider_status") == "disabled_at_build");
  CHECK(snapshot.at("capacity") == 0);
  CHECK(snapshot.at("records").empty());
#else
  const auto status = snapshot.at("provider_status").get<std::string>();
  CHECK((status == "disabled_at_runtime" || status == "requested_provider_unavailable" ||
         status == "active"));
#endif
}

TEST_CASE("enabled ROCTx host scopes balance exceptions and cross-thread stops",
          "[cpu][trace][roctx]") {
#if defined(RNS8_ENABLE_ROCTX) && RNS8_ENABLE_ROCTX
  using namespace rns8::detail::trace;
  if (std::string(provider_status()) != "active")
    SKIP("requires explicit RNS8_PROFILE_RANGES=1 and a current ROCTx provider");
  const auto before = range_snapshot().at("claimed").get<uint64_t>();
  if (before + 3 > 4096)
    SKIP("bounded trace metadata capacity already consumed");
  try {
    Scope outer(Phase::public_call, {1234, 7, Role::measured});
    Scope inner(Phase::proof);
    throw std::runtime_error("authored scope unwind");
  } catch (const std::runtime_error&) {
  }
  AsyncRange operation;
  operation.start(Phase::completion, {1234, 7, Role::measured});
  AsyncRange moved(std::move(operation));
  std::thread worker([range = std::move(moved)]() mutable { range.complete(Outcome::completed); });
  worker.join();
  const auto after = range_snapshot();
  REQUIRE(after.at("claimed") == before + 3);
  CHECK(after.at("marker_label_contract") == "rns8_phase_slot_v1");
  const auto& records = after.at("records");
  REQUIRE(records.size() >= before + 3);
  CHECK(records.at(before).at("stop_state") == 1); // host scope closure
  CHECK(records.at(before + 1).at("stop_state") == 1);
  CHECK(records.at(before + 2).at("stop_state") == 2); // explicit completion
  CHECK(records.at(before + 2).at("parent_range_id").is_null());
  CHECK(records.at(before + 2).at("operation_id") == 1234);
  CHECK(after.at("host_clock").at("schema") == "rns8_linux_host_range_clock_v1");
  CHECK(after.at("host_clock").at("clock_id") == 1);
  CHECK(after.at("host_clock").at("unit") == "nanoseconds");
  for (uint64_t index = before; index < before + 3; ++index) {
    const auto& record = records.at(index);
    REQUIRE(record.at("process_id").get<uint64_t>() > 0);
    REQUIRE(record.at("start_thread_id").get<uint64_t>() > 0);
    REQUIRE(record.at("stop_thread_id").get<uint64_t>() > 0);
    const auto a = record.at("start_before_ns").get<uint64_t>();
    const auto b = record.at("start_after_ns").get<uint64_t>();
    const auto c = record.at("stop_before_ns").get<uint64_t>();
    const auto d = record.at("stop_after_ns").get<uint64_t>();
    CHECK(a <= b);
    CHECK(b <= c);
    CHECK(c <= d);
  }
  CHECK(records.at(before).at("start_thread_id") == records.at(before).at("stop_thread_id"));
  CHECK(records.at(before + 2).at("start_thread_id") !=
        records.at(before + 2).at("stop_thread_id"));
#else
  SKIP("optional ROCTx build feature is disabled");
#endif
}
