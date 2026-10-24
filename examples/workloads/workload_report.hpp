#ifndef RNS8_WORKLOAD_REPORT_HPP
#define RNS8_WORKLOAD_REPORT_HPP
#include "workload_common.hpp"
namespace rns8_workload {
inline Json sustained_sample(Variant& variant, uint32_t target_ms, const Words& expected) {
  // All variants recompute and serialize their complete output on every call.
  // Keep both reference work and diagnostic phase clocks outside this interval.
  Phases phases;
  phases.enabled = false;
  uint32_t calls = 0;
  double elapsed = 0;
  const auto start = Clock::now();
  do {
    variant.execute(phases, 1);
    ++calls;
    elapsed = since(start);
  } while (calls < 4096 && elapsed < double(target_ms) * 1e6);
  const auto verification_start = Clock::now();
  require(variant.output() == expected, "sustained final exact output mismatch: " + variant.name);
  const auto hash = checksum(variant.output());
  return {{"calls", calls}, {"elapsed_ns", elapsed}, {"mean_call_ns", elapsed / calls},
          {"target_ms", target_ms}, {"call_limit", 4096},
          {"target_reached", elapsed >= double(target_ms) * 1e6},
          {"checksum_u64", hash}, {"final_exact_reference_passed", true},
          {"verification_ns", since(verification_start)}, {"phase_timing_enabled", false},
          {"scope", "same inputs/version and retained allocations; full synchronous application output every call; no reference, checksum or phase timing between calls; complete final output verified after timing"}};
}
inline Json verified_sample(Variant& variant, uint64_t version, const Words& expected) {
  Phases phases;
  const auto start = Clock::now(); variant.execute(phases, version); const double application = since(start);
  const auto verification_start = Clock::now();
  require(variant.output() == expected, "full exact output mismatch: " + variant.name);
  const auto hash = checksum(variant.output());
  const double verification = since(verification_start);
  return {{"application_ns", application}, {"phases_ns", phases.values}, {"verification_ns", verification},
          {"checksum_u64", hash}, {"exact_reference_passed", true}};
}
inline void run_variants(Json& report, const Options& options, const Dataset& data, std::vector<Variant>& variants,
    const Words& expected, const std::function<Words()>& mutate_and_reference, Json skipped = Json::array()) {
  Json records = Json::array(), order = Json::array();
  for (auto& variant : variants) {
    const auto cold = verified_sample(variant, 1, expected);
    records.push_back({{"name", variant.name}, {"status", "passed"}, {"metadata", variant.metadata},
        {"setup_ns", variant.setup_ns}, {"cold_call", cold},
        {"cold_application_total_ns", data.preparation_ns + variant.setup_ns + cold["application_ns"].get<double>()},
        {"cold_total_scope", "sum of measured common input/layout preparation, per-variant resource setup and first synchronous call; verification/reporting excluded"},
        {"samples", Json::array()}});
  }
  for (uint32_t repeat = 0; repeat < options.warmups + options.repeats; ++repeat) {
    Json iteration = Json::array();
    for (std::size_t slot = 0; slot < variants.size(); ++slot) {
      const auto index = (slot + repeat + options.seed % variants.size()) % variants.size();
      auto sample = verified_sample(variants[index], 1, expected);
      if (repeat >= options.warmups) records[index]["samples"].push_back(std::move(sample));
      iteration.push_back(variants[index].name);
    }
    order.push_back({{"warmup", repeat < options.warmups}, {"order", iteration}});
  }
  Json sustained_order = Json::array();
  if (options.sustained_ms) {
    for (std::size_t slot = 0; slot < variants.size(); ++slot) {
      const auto index = (slot + options.seed % variants.size()) % variants.size();
      records[index]["sustained"] = sustained_sample(variants[index], options.sustained_ms, expected);
      sustained_order.push_back(variants[index].name);
    }
  }
  report["sustained_order"] = sustained_order;
  const auto reference_start = Clock::now();
  const Words changed = mutate_and_reference();
  report["changed_input_reference_ns"] = since(reference_start);
  for (std::size_t index = 0; index < variants.size(); ++index) {
    records[index]["changed_input"] = verified_sample(variants[index], 2, changed);
    records[index]["changed_input"]["source_version"] = 2;
    std::vector<double> times;
    for (const auto& sample : records[index]["samples"]) times.push_back(sample["application_ns"].get<double>());
    std::sort(times.begin(), times.end());
    const double median = times.size() % 2 ? times[times.size() / 2]
        : (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2;
    records[index]["median_steady_ns"] = median;
    records[index]["min_steady_ns"] = times.front(); records[index]["max_steady_ns"] = times.back();
    records[index]["calls_per_second"] = 1e9 / median;
    records[index]["within_process_only"] = true;
    const auto teardown_start = Clock::now();
    variants[index].execute = {}; variants[index].output = {};
    const double teardown = since(teardown_start);
    records[index]["resource_teardown_ns"] = teardown;
    records[index]["cold_application_total_ns"] = records[index]["cold_application_total_ns"].get<double>() + teardown;
    records[index]["cold_total_scope"] = "sum of measured common input/layout preparation, per-variant resource setup, first synchronous call and eventual resource teardown; verification/reporting excluded; teardown occurs after the reuse experiment";
  }
  for (auto& value : skipped) records.push_back(std::move(value));
  report["variants"] = records;
  report["aggregate_memory_envelope_bytes"] = options.memory->reserved;
  report["sample_order"] = order;
  report["order_method"] = "deterministic cyclic rotation across variants; warmups and all repetitions fully verified outside timing";
  report["reference_method"] = "independent naive Boost cpp_int complete output; no checksum-only verification";
  report["dataset_preparation_ns"] = data.preparation_ns;
  report["checksum_u64"] = checksum(expected); report["checksum_hex"] = hex64(checksum(expected));
  report["changed_checksum_u64"] = checksum(changed); report["changed_checksum_hex"] = hex64(checksum(changed));
  report["checksum_contract"] = "fnv1a64_bytes_v1; row-major cells then least-significant u64 limb first then little-endian bytes; every output word included";
  report["comparison_note"] = "raw same-process diagnostic observations, not established speedups; collector preserves independent processes and calibration admission remains separate";
}
inline void mutate_input(Dataset& data) {
  require(!data.a.empty(), "missing mutable input");
  data.a[0] = data.a[0] == INT64_MAX ? data.a[0] - 1 : data.a[0] + 1;
}
} // namespace rns8_workload
#endif
