#include "call_observation.hpp"
#include "host_control_json.hpp"
// Completed PUBLIC ownership/proof/export boundary; private queue microbenchmarks
// remain separate and are never substituted for these generation-owned calls.
#include "core/internal.hpp"
#include "core/hip_resources.hpp"
#include "backend_common/native_wmma_queue.hpp"
#include "backend_common/native_wmma_schedule.hpp"
static_assert(sizeof(rns8::detail::NativeWmmaTask) == 104,
              "public queue capture ABI requires 64-bit pointer layout");
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using boost::multiprecision::cpp_int;
namespace {
void need(bool b, const char* why) {
  if (!b)
    throw std::invalid_argument(why);
}
void check(rns8_status s) {
  if (s != RNS8_SUCCESS)
    throw std::runtime_error(rns8_status_string(s));
}
double us(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}
uint32_t number(const json& c, const char* key, uint32_t lo, uint32_t hi) {
  need(c.at(key).is_number_integer(), "integer field required");
  const auto v = c.at(key).get<int64_t>();
  need(v >= lo && v <= hi, "integer field outside admitted range");
  return uint32_t(v);
}
struct Input {
  rns8_matrix* matrix = nullptr;
  rns8_encoded_b* cache = nullptr;
  uint32_t rows = 0, cols = 0, prefix = 0;
  bool u = false;
  std::vector<int64_t> signed_values;
  std::vector<uint64_t> unsigned_values;
  cpp_int value(size_t i) const {
    return u ? cpp_int(unsigned_values[i]) : cpp_int(signed_values[i]);
  }
};
struct Task {
  rns8_plan* plan = nullptr;
  rns8_workspace* work = nullptr;
  rns8_matrix* c = nullptr;
  rns8_prepared_execution* execution = nullptr;
  Input a;
  uint32_t m = 0, n = 0, k = 0, prefix = 0, group = 0, b_slot = 0;
  std::vector<uint64_t> output;
};
struct Owner {
  rns8_context* ctx = nullptr;
  rns8_prepared_group* group = nullptr;
  std::vector<Task> tasks;
  std::map<uint32_t, Input> bs;
  ~Owner() {
    // On an uncertain drain, the public owner deliberately refuses destruction.
    // Do not bypass its reservations by freeing private pointers in a benchmark.
    if (group && rns8_destroy_prepared_group(group) != RNS8_SUCCESS)
      return;
    for (auto& task : tasks) {
      if (task.execution)
        (void)rns8_destroy_prepared_execution(task.execution);
      if (task.c)
        (void)rns8_destroy_matrix(task.c);
      if (task.a.matrix)
        (void)rns8_destroy_matrix(task.a.matrix);
      if (task.work)
        (void)rns8_destroy_workspace(task.work);
      if (task.plan)
        (void)rns8_destroy_plan(task.plan);
    }
    for (auto& [slot, b] : bs) {
      if (b.cache)
        (void)rns8_destroy_encoded_b(b.cache);
      if (b.matrix)
        (void)rns8_destroy_matrix(b.matrix);
    }
    if (ctx)
      (void)rns8_destroy_context(ctx);
  }
};
struct Observation {
  double prepare_us, execute_us, export_us, host_loop_us, total_us;
  uint32_t complete_groups;
};
json observation_json(const Observation& value) {
  return {{"prepare_us", value.prepare_us}, {"execute_us", value.execute_us},
          {"export_us", value.export_us},   {"host_loop_us", value.host_loop_us},
          {"total_us", value.total_us},     {"complete_groups", value.complete_groups}};
}
using rns8::bench::host_control_json;
json allocation_delta(const rns8::detail::hip_direct_allocation_counters& before,
                      const rns8::detail::hip_direct_allocation_counters& after) {
  // This calibrator owns the process; these are existing successful API-payload
  // counters, not host allocations, physical traffic or device residency.
  return {{"allocate_calls", after.allocate_calls - before.allocate_calls},
          {"free_calls", after.free_calls - before.free_calls},
          {"allocated_bytes", after.allocated_bytes - before.allocated_bytes},
          {"h2d_calls", after.h2d_calls - before.h2d_calls},
          {"h2d_bytes", after.h2d_bytes - before.h2d_bytes},
          {"d2h_calls", after.d2h_calls - before.d2h_calls},
          {"d2h_bytes", after.d2h_bytes - before.d2h_bytes},
          {"d2d_calls", after.d2d_calls - before.d2d_calls},
          {"d2d_bytes", after.d2d_bytes - before.d2d_bytes}};
}
constexpr uint64_t sentinel = 0x5ac37e195ac37e19ull;
} // namespace
int main(int argc, char** argv) {
  try {
    need(argc == 3, "usage: rns8-public-queue-calibrate CASE.json unique-run-id");
    std::ifstream file(argv[1]);
    json config;
    file >> config;
    need(config.is_object() &&
             config.size() == 13 + (config.contains("locality_order") ? 1 : 0) +
                                  (config.contains("dispatch_policy") ? 1 : 0) +
                                  (config.contains("host_diagnostics") ? 1 : 0) +
                                  (config.contains("integrity") ? 1 : 0) &&
             config.at("schema") == "public_native_group_case_v1",
         "case ABI");
    const auto mode = config.at("mode").get<std::string>(),
               policy = config.at("input_policy").get<std::string>();
    need(mode == "persistent" || mode == "ordinary" || mode == "graph" || mode == "resident",
         "mode");
    need(policy == "resident" || policy == "mutated-b", "input policy");
    need(config.at("encoded_b").is_boolean(), "encoded B must be bool");
    const bool encoded = config.at("encoded_b");
    need(mode != "resident" || !encoded, "resident RNS control does not consume encoded B");
    const uint32_t schedule = number(config, "schedule", 0, 3),
                   chunk = number(config, "chunk", 1, 32),
                   workers = number(config, "workers", 1, 1024),
                   reuse = number(config, "reuse", 1, 64),
                   warmups = number(config, "warmups", 1, 1000),
                   repeats = number(config, "repeats", 3, 1000),
                   seed = number(config, "seed", 0, UINT32_MAX),
                   padding = number(config, "padding", 0, 64);
    for (const auto* key : {"host_diagnostics", "integrity"})
      need(!config.contains(key) || config.at(key).is_boolean(),
           "diagnostic/integrity controls must be bool");
    const bool diagnostics = config.value("host_diagnostics", false),
               integrity = config.value("integrity", false);
    need(!diagnostics || rns8::detail::host_control_diagnostics_compiled(),
         "host diagnostics require RNS8_ENABLE_HOST_CONTROL_DIAGNOSTICS=ON");
    need(!integrity || mode != "resident", "native integrity requires native queue inputs");
    const auto dispatch =
        config.contains("dispatch_policy") ? number(config, "dispatch_policy", 0, 2) : 0u;
    need(!dispatch || mode == "persistent" || mode == "graph",
         "dispatch policy requires a native queue");
    const bool locality_enabled = config.contains("locality_order");
    const std::string locality_order = config.value("locality_order", std::string("ordinary"));
    need(!locality_enabled ||
             (!schedule && (mode == "persistent" || mode == "graph") &&
              (locality_order == "ordinary" || locality_order == "output-neighbor" ||
               locality_order == "modulus" || locality_order == "same-output")),
         "locality requires explicit persistent/graph work directory");
    rns8_locality_desc locality{
        sizeof(locality), RNS8_ABI_VERSION, RNS8_LOCALITY_ORDINARY, 0, 1, 1, 1, 0};
    locality.order = locality_order == "output-neighbor" ? RNS8_LOCALITY_OUTPUT_NEIGHBOR
                     : locality_order == "modulus"       ? RNS8_LOCALITY_MODULUS
                     : locality_order == "same-output"   ? RNS8_LOCALITY_SAME_OUTPUT
                                                         : RNS8_LOCALITY_ORDINARY;
    need(config.at("tasks").is_array() && !config["tasks"].empty() && config["tasks"].size() <= 64,
         "task count");
    Owner owner;
    owner.tasks.resize(config["tasks"].size());
    uint64_t oracle_work = 0, useful_work = 0;
    for (size_t i = 0; i < owner.tasks.size(); ++i) {
      const auto& t = config["tasks"][i];
      auto& task = owner.tasks[i];
      need(t.is_object() && t.size() == 7 && t.at("unsigned_inputs").is_boolean(), "task ABI");
      task.m = number(t, "m", 1, 512);
      task.n = number(t, "n", 1, 512);
      task.k = number(t, "k", 1, 65537);
      task.prefix = number(t, "prefix", 1, 20);
      task.group = number(t, "group", 1, 4);
      need(task.group != 3, "modulus group");
      task.b_slot = number(t, "b_slot", 0, 63);
      task.a = {};
      task.a.rows = task.m;
      task.a.cols = task.k;
      task.a.prefix = task.prefix;
      task.a.u = t["unsigned_inputs"];
      auto& b = owner.bs[task.b_slot];
      if (b.rows)
        need(b.rows == task.k && b.cols == task.n && b.prefix == task.prefix && b.u == task.a.u,
             "shared B contract mismatch");
      else {
        b.rows = task.k;
        b.cols = task.n;
        b.prefix = task.prefix;
        b.u = task.a.u;
      }
      oracle_work += uint64_t(task.m) * task.n * task.k;
      useful_work += uint64_t((task.m + 15) / 16) * ((task.n + 15) / 16) *
                     ((task.prefix + task.group - 1) / task.group);
      cpp_int product = 1;
      for (uint32_t p = 0; p < task.prefix; ++p)
        product *= rns8_default_modulus(p);
      need(product > cpp_int(task.k) * 49 * (task.a.u ? 1 : 2),
           "prefix fails independent dataset bound");
    }
    need(oracle_work <= 134217728, "full Boost oracle work budget");
    std::mt19937_64 rng(seed);
    auto values = [&](Input& input) {
      const size_t count = size_t(input.rows) * input.cols;
      if (input.u) {
        input.unsigned_values.resize(count);
        for (auto& v : input.unsigned_values)
          v = rng() % 8;
      } else {
        input.signed_values.resize(count);
        for (auto& v : input.signed_values)
          v = int64_t(rng() % 15) - 7;
      }
    };
    for (auto& [slot, b] : owner.bs)
      values(b);
    for (auto& task : owner.tasks)
      values(task.a);
    rns8::detail::hip_direct_timing_set_enabled(false);
    rns8::detail::HostControlObservations cold_counts;
    const auto cold_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    std::optional<rns8::detail::HostControlObservationScope> cold_observer;
    if (diagnostics)
      cold_observer.emplace(cold_counts);
    const auto setup_start = Clock::now();
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    check(rns8_create_context(0, &options, &owner.ctx));
    auto matrix = [&](Input& input) {
      rns8_matrix_desc md{};
      md.struct_size = sizeof(md);
      md.abi_version = RNS8_ABI_VERSION;
      md.semantics = input.u ? RNS8_EXACT_WIDE_UNSIGNED : RNS8_EXACT_WIDE_SIGNED;
      md.bound_kind = RNS8_BOUND_NONE;
      md.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
      md.rows = input.rows;
      md.cols = input.cols;
      md.logical_ld = input.cols;
      md.max_prefix = input.prefix;
      md.flags = RNS8_MATRIX_RETAIN_NATIVE_INPUT;
      check(rns8_create_matrix(owner.ctx, &md, &input.matrix));
    };
    auto pack = [&](Input& input) {
      check(
          input.u
              ? rns8_pack_u64(owner.ctx, input.matrix, input.unsigned_values.data(), input.cols, 0)
              : rns8_pack_i64(owner.ctx, input.matrix, input.signed_values.data(), input.cols, 0));
    };
    for (auto& [slot, b] : owner.bs) {
      matrix(b);
      pack(b);
    }
    std::vector<rns8_prepared_execution*> executions;
    std::vector<rns8_native_group_binding> bindings(owner.tasks.size());
    for (size_t i = 0; i < owner.tasks.size(); ++i) {
      auto& task = owner.tasks[i];
      matrix(task.a);
      pack(task.a);
      rns8_gemm_desc desc{};
      desc.struct_size = sizeof(desc);
      desc.abi_version = RNS8_ABI_VERSION;
      desc.semantics = task.a.u ? RNS8_EXACT_WIDE_UNSIGNED : RNS8_EXACT_WIDE_SIGNED;
      desc.bound_kind = RNS8_BOUND_NONE;
      desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      desc.m = task.m;
      desc.n = task.n;
      desc.k = task.k;
      desc.max_prefix = task.prefix;
      desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX | RNS8_PLAN_EXACT_VALUE_BOUND;
      check(rns8_create_plan(owner.ctx, &desc, &task.plan));
      check(rns8_create_workspace(owner.ctx, task.plan, &task.work));
      rns8_matrix_desc md{};
      md.struct_size = sizeof(md);
      md.abi_version = RNS8_ABI_VERSION;
      md.semantics = desc.semantics;
      md.bound_kind = RNS8_BOUND_NONE;
      md.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
      md.rows = task.m;
      md.cols = task.n;
      md.logical_ld = task.n;
      md.max_prefix = task.prefix;
      check(rns8_create_matrix(owner.ctx, &md, &task.c));
      check(rns8_prepare_gemm(owner.ctx, task.plan, task.a.matrix, owner.bs.at(task.b_slot).matrix,
                              task.c, task.work, &task.execution));
      if (integrity) {
        rns8_native_integrity_options options{sizeof(options), RNS8_ABI_VERSION, 1, 8, 0, 0};
        check(rns8_set_prepared_native_integrity(task.execution, &options));
      }
      executions.push_back(task.execution);
      bindings[i] = {sizeof(bindings[i]), RNS8_ABI_VERSION, task.group, nullptr};
      task.output.assign(size_t(task.m) * (task.n + padding) * 3, sentinel);
    }
    check(rns8_create_prepared_group(executions.data(), uint32_t(executions.size()), &owner.group));
    rns8_native_group_desc desc{sizeof(desc), RNS8_ABI_VERSION, schedule,
                                chunk,        workers,          mode == "ordinary" ? 1u : 0u,
                                UINT64_MAX};
    uint64_t encoded_bytes = 0, peak_encoded_bytes = 0, rebuilds = 0;
    const auto rebuild = [&]() {
      if (mode == "resident") {
        check(rns8_refresh_prepared_group_inputs(owner.group));
        ++rebuilds;
        return;
      }
      std::map<uint32_t, rns8_encoded_b*> replacement;
      uint64_t new_bytes = 0;
      try {
        if (encoded)
          for (auto& [slot, b] : owner.bs) {
            rns8_encoded_b* cache = nullptr;
            check(rns8_prepare_encoded_b(owner.ctx, b.matrix, UINT64_MAX, &cache));
            replacement.emplace(slot, cache);
            rns8_encoded_b_info info{sizeof(info), RNS8_ABI_VERSION};
            check(rns8_get_encoded_b_info(cache, &info));
            need(info.source_is_current == 1, "new cache stale");
            new_bytes += info.retained_device_bytes;
          }
        for (size_t i = 0; i < bindings.size(); ++i)
          bindings[i].encoded_b = encoded ? replacement.at(owner.tasks[i].b_slot) : nullptr;
        check(rns8_prepare_native_group_dispatch(owner.group, &desc, bindings.data(),
                                                 uint32_t(bindings.size()),
                                                 static_cast<rns8_native_dispatch_policy>(dispatch),
                                                 locality_enabled ? &locality : nullptr));
        peak_encoded_bytes = std::max(peak_encoded_bytes, encoded_bytes + new_bytes);
        encoded_bytes = new_bytes;
        for (auto& [slot, b] : owner.bs) {
          if (b.cache)
            check(rns8_destroy_encoded_b(b.cache));
          b.cache = encoded ? replacement.at(slot) : nullptr;
          replacement.erase(slot);
        }
        if (mode == "graph")
          check(rns8_capture_native_group_hip_graph(owner.group));
        ++rebuilds;
      } catch (...) {
        for (auto& [slot, cache] : replacement)
          (void)rns8_destroy_encoded_b(cache);
        throw;
      }
    };
    rebuild();
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    rns8::detail::hip_unique_stream stream;
    if (mode == "graph" && stream.create_non_blocking() != hipSuccess)
      throw std::runtime_error("stream creation failed");
#endif
    const double setup_us = us(setup_start, Clock::now());
    cold_observer.reset();
    const auto cold_finished_allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
    uint64_t generation = 1, groups_executed = 0;
    std::vector<rns8_completion_token> tokens(owner.tasks.size());
    const auto execute = [&](bool timed = true) {
      const rns8::bench::CallClock<> clock(timed);
      std::array<double, 3> phases{};
      const auto start = clock.now();
      for (uint32_t r = 0; r < reuse; ++r) {
        const auto before = clock.now();
        if (policy == "mutated-b") {
          ++generation;
          for (auto& [slot, b] : owner.bs) {
            if (b.u)
              b.unsigned_values[0] = (b.unsigned_values[0] + 1) % 8;
            else
              b.signed_values[0] = (b.signed_values[0] + 8) % 15 - 7;
            pack(b);
          }
          rebuild();
        }
        const auto prepared = clock.now();
        uint32_t completed = 0;
        rns8_status status = RNS8_SUCCESS;
        if (mode != "graph")
          status = rns8_execute_prepared_group(owner.group, tokens.data(), uint32_t(tokens.size()),
                                               &completed);
        else {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
          status = rns8_submit_native_group_hip_graph(owner.group, stream.get(), tokens.data(),
                                                      uint32_t(tokens.size()), &completed);
          for (uint32_t i = 0; i < completed; ++i) {
            rns8_completion_info info{sizeof(info), RNS8_ABI_VERSION};
            const auto waited = rns8_wait_prepared_hip(executions[i], tokens[i], &info);
            if (status == RNS8_SUCCESS)
              status = waited == RNS8_SUCCESS ? info.operation_status : waited;
          }
#else
          throw std::runtime_error("native graph requires a HIP build");
#endif
        }
        check(status);
        need(completed == owner.tasks.size(), "incomplete public group");
        const auto computed = clock.now();
        for (auto& task : owner.tasks)
          check(task.a.u
                    ? rns8_export_exact_wide_unsigned_limbs(owner.ctx, task.plan, task.c,
                                                            task.output.data(), task.n + padding, 3)
                    : rns8_export_exact_wide_signed_limbs(owner.ctx, task.plan, task.c,
                                                          task.output.data(), task.n + padding, 3));
        const auto exported = clock.now();
        phases[0] += us(before, prepared);
        phases[1] += us(prepared, computed);
        phases[2] += us(computed, exported);
        ++groups_executed;
      }
      const double total = us(start, clock.now());
      return Observation{phases[0], phases[1],
                         phases[2], std::max(0.0, total - phases[0] - phases[1] - phases[2]),
                         total,     reuse};
    };
    json integrity_receipts = nullptr;
    const auto audit = [&]() {
      uint64_t digest = 14695981039346656037ull;
      for (const auto& [slot, b] : owner.bs)
        need(b.matrix->content_generation == generation,
             "public B generation differs from recorded interval");
      if (integrity)
        integrity_receipts = json::array();
      size_t task_index = 0;
      for (const auto& task : owner.tasks) {
        if (integrity) {
          rns8_native_integrity_report report{sizeof(report), RNS8_ABI_VERSION};
          check(rns8_get_prepared_native_integrity_report(task.execution, tokens[task_index],
                                                          &report));
          need(report.enabled && report.outcome == 1 && !report.failure_mask &&
                   report.operation_status == RNS8_SUCCESS,
               "native integrity mismatch");
          integrity_receipts.push_back(
              {{"task", task_index},
               {"operation", report.operation_id},
               {"epoch", report.epoch},
               {"a_generation", report.a_generation},
               {"b_generation", report.b_generation},
               {"prefix", report.computational_prefix},
               {"basis_version", report.computational_basis_version},
               {"check_moduli", {report.check_moduli[0], report.check_moduli[1]}},
               {"enabled", true},
               {"outcome", report.outcome},
               {"failure_mask", report.failure_mask}});
        }
        ++task_index;
        const auto& b = owner.bs.at(task.b_slot);
        const auto ld = task.n + padding;
        for (uint32_t row = 0; row < task.m; ++row)
          for (uint32_t col = 0; col < ld; ++col) {
            const auto index = (size_t(row) * ld + col) * 3;
            if (col >= task.n) {
              for (int limb = 0; limb < 3; ++limb)
                need(task.output[index + limb] == sentinel, "padding modified");
              continue;
            }
            cpp_int expected = 0;
            for (uint32_t q = 0; q < task.k; ++q)
              expected +=
                  task.a.value(size_t(row) * task.k + q) * b.value(size_t(q) * task.n + col);
            cpp_int actual = cpp_int(task.output[index]) + (cpp_int(task.output[index + 1]) << 64) +
                             (cpp_int(task.output[index + 2]) << 128);
            if (!task.a.u && (task.output[index + 2] >> 63))
              actual -= cpp_int(1) << 192;
            need(actual == expected, "full Boost output mismatch");
            for (int limb = 0; limb < 3; ++limb)
              for (int byte = 0; byte < 8; ++byte) {
                digest ^= (task.output[index + limb] >> (byte * 8)) & 255u;
                digest *= 1099511628211ull;
              }
          }
      }
      return std::to_string(digest);
    };
    const auto first = observation_json(execute());
    audit();
    for (uint32_t i = 0; i < warmups; ++i)
      execute(false);
    json samples = json::array();
    for (uint32_t i = 0; i < repeats; ++i) {
      auto sample = observation_json(execute());
      sample["output_digest"] = audit();
      samples.push_back(std::move(sample));
    }
    json sustained = json::array();
    for (int batch = 0; batch < 3; ++batch) {
      const auto start_generation = generation, start_groups = groups_executed,
                 start_rebuilds = rebuilds;
      const auto interval = rns8::bench::measure_completed_calls([&]() { execute(false); });
      const auto calls = interval.calls;
      const double duration = interval.duration_us;
      sustained.push_back({{"calls", calls},
                           {"complete_groups", groups_executed - start_groups},
                           {"duration_us", duration},
                           {"first_input_generation", start_generation},
                           {"last_input_generation", generation},
                           {"rebuilds", rebuilds - start_rebuilds},
                           {"output_digest", audit()},
                           {"exact_reference_passed", true}});
    }
    json host_diagnostic = nullptr;
    if (diagnostics) {
      rns8::detail::HostControlObservations warm_counts;
      const auto before = rns8::detail::hip_direct_allocation_counters_snapshot();
      {
        rns8::detail::HostControlObservationScope observation(warm_counts);
        execute();
      }
      const auto after = rns8::detail::hip_direct_allocation_counters_snapshot();
      host_diagnostic = {
          {"schema", "rns8_public_queue_host_control_v2"},
          {"phase_time_contract", rns8::bench::host_proof_phase_contract},
          {"scope", "calling_thread_one_complete_observation_after_timing"},
          {"complete_groups", reuse},
          {"cold_setup", host_control_json(cold_counts)},
          {"warm_observation", host_control_json(warm_counts)},
          {"cold_setup_allocations", allocation_delta(cold_allocations, cold_finished_allocations)},
          {"warm_observation_allocations", allocation_delta(before, after)},
          {"output_digest", audit()},
          {"exact_reference_passed", true},
          {"attempt_count_scope", "prepared_submission_and_managed_runtime_handles"},
          {"latency_evidence", false}};
    }
    rns8_device_info device{};
    device.struct_size = sizeof(device);
    device.abi_version = RNS8_ABI_VERSION;
    check(rns8_get_device_info(owner.ctx, &device));
    json metadata = nullptr;
    if (mode != "resident") {
      rns8_native_group_info info{sizeof(info), RNS8_ABI_VERSION};
      check(rns8_get_native_group_info(owner.group, &info));
      rns8_native_group_dispatch_info hint{sizeof(hint), RNS8_ABI_VERSION};
      check(rns8_get_native_group_dispatch_info(owner.group, &hint));
      metadata = {
          {"dispatch_hint",
           {{"policy", uint32_t(hint.policy)},
            {"group", hint.uniform_group},
            {"unsigned_inputs", hint.unsigned_inputs},
            {"encoded_b", hint.encoded_b}}},
          {"schedule_mode", info.schedule_mode},
          {"schedule_identity", rns8::detail::native_wmma_schedule_identity(
                                    rns8::detail::NativeWmmaScheduleMode(info.schedule_mode))},
          {"tasks", info.task_count},
          {"work", info.work_count},
          {"device_bytes", info.retained_device_metadata_bytes},
          {"host_bytes", info.retained_host_metadata_bytes},
          {"preparation_peak_device_bytes", info.preparation_peak_device_metadata_bytes},
          {"graph_ready", info.graph_ready != 0}};
    }
    json result = {
        {"schema", "rns8_public_native_group_microcase_v1"},
        {"case", config},
        {"run_id", argv[2]},
        {"build_manifest", json::parse(RNS8_BUILD_MANIFEST_JSON)},
        {"device", device.name},
        {"target", device.gcn_arch},
        {"runtime_version", device.hip_runtime_version},
        {"seed", seed},
        {"dataset", "bounded_mt19937_64_shared_b_v1"},
        {"native_integrity_receipts", integrity_receipts},
        {"host_control_diagnostics_compiled", rns8::detail::host_control_diagnostics_compiled()},
        {"host_control_diagnostic", host_diagnostic},
        {"hot_loop_serialization", "typed_observation_no_phase_clocks_in_sustained_v2"},
        {"host_instrumentation_state", "hooks_disabled"},
        {"output_boundary", "each_public_group_and_all_padded_three_limb_exports"},
        {"setup_us", setup_us},
        {"setup_scope", "context_plans_matrices_input_pack_encoded_B_descriptors_graph"},
        {"first_call", first},
        {"samples", samples},
        {"sustained", sustained},
        {"metadata", metadata},
        {"encoded_b_retained_device_bytes", encoded_bytes},
        {"encoded_b_peak_replacement_device_bytes", peak_encoded_bytes},
        {"useful_tile_channel_groups_per_call", useful_work},
        {"exact_reference_passed", true},
        {"oracle_inside_production_interval", false},
        {"oracle_scope", "complete_final_output_after_each_observation_and_sustained_interval"},
        {"per_task_device_completion_latency_measured", false},
        {"promotion_eligible", false}};
    if (locality_enabled) {
      rns8_locality_info info{sizeof(info), RNS8_ABI_VERSION};
      check(rns8_get_native_group_locality_info(owner.group, &info));
      result["locality"] = {{"version", 1},
                            {"requested_order", locality_order},
                            {"applied_order", info.applied_order},
                            {"applied", bool(info.applied)},
                            {"target_id", info.target_id},
                            {"topology_version", 1},
                            {"physical_affinity_established", false},
                            {"active_entries", info.active_entries},
                            {"directory_bytes", info.directory_bytes},
                            {"requested_source_bytes", info.requested_source_bytes},
                            {"unique_source_bytes", info.unique_source_bytes},
                            {"partial_output_bytes", info.output_read_write_bytes},
                            {"permutation_fingerprint", info.permutation_fingerprint}};
    }
    uint64_t output_bytes = 0;
    for (const auto& task : owner.tasks)
      output_bytes += uint64_t(task.m) * task.n * 24;
    result["output_payload_bytes_per_call"] = output_bytes;
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
