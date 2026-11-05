#include "call_observation.hpp"
#include "native_diagnostic_observation.hpp"
#include "core/api_internal.hpp"
#include "tool_common/build_identity.hpp"
#include "tool_common/device_identity_json.hpp"
#include "tool_common/execution_environment_json.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

namespace {
using json = nlohmann::json;
using boost::multiprecision::cpp_int;
using Clock = std::chrono::steady_clock;
static_assert(sizeof(rns8::detail::NativeNarrowTask) == 32,
              "group calibration requires native64 task pointers");
void check(rns8_status status) {
  if (status != RNS8_SUCCESS)
    throw std::runtime_error(rns8_status_string(status));
}
uint32_t number(const json& c, const char* key, uint32_t low, uint32_t high) {
  if (!c.at(key).is_number_integer())
    throw std::invalid_argument(key);
  const auto value = c.at(key).get<int64_t>();
  if (value < low || value > high)
    throw std::invalid_argument(key);
  return uint32_t(value);
}
double microseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}
cpp_int exact_native(uint64_t value, bool unsigned_inputs) {
  return unsigned_inputs || value <= uint64_t(INT64_MAX) ? cpp_int(value)
                                                         : cpp_int(value) - (cpp_int(1) << 64);
}
struct Dataset {
  uint32_t m, n, k, count, prefix;
  bool u, shared_b;
  std::vector<std::vector<uint64_t>> a, b;
  std::vector<cpp_int> expected;
  Dataset(const json& c)
      : m(c["m"]), n(c["n"]), k(c["k"]), count(c["tasks"]), prefix(c["prefix"]),
        u(c["unsigned_inputs"]), shared_b(c["shared_b"]), a(count), b(shared_b ? 1 : count),
        expected(size_t(count) * m * n) {
    const auto seed = c["seed"].get<uint32_t>();
    for (uint32_t task = 0; task < count; ++task) {
      a[task].assign(size_t(m) * (k + 1), UINT64_MAX);
      for (uint32_t r = 0; r < m; ++r)
        for (uint32_t q = 0; q < k; ++q)
          a[task][size_t(r) * (k + 1) + q] = uint64_t(
              int64_t((uint64_t(r) * 71 + uint64_t(q) * 19 + task * 13 + seed) % (u ? 8 : 15)) -
              (u ? 0 : 7));
    }
    for (uint32_t task = 0; task < b.size(); ++task) {
      b[task].assign(size_t(k) * (n + 2), UINT64_MAX);
      for (uint32_t q = 0; q < k; ++q)
        for (uint32_t col = 0; col < n; ++col)
          b[task][size_t(q) * (n + 2) + col] = uint64_t(
              int64_t((uint64_t(q) * 37 + uint64_t(col) * 11 + task * 17 + seed) % (u ? 8 : 15)) -
              (u ? 0 : 7));
    }
    for (uint32_t task = 0; task < count; ++task)
      for (uint32_t r = 0; r < m; ++r)
        for (uint32_t col = 0; col < n; ++col)
          for (uint32_t q = 0; q < k; ++q) {
            const auto av = a[task][size_t(r) * (k + 1) + q];
            const auto bv = b[shared_b ? 0 : task][size_t(q) * (n + 2) + col];
            expected[(size_t(task) * m + r) * n + col] += exact_native(av, u) * exact_native(bv, u);
          }
    if (!prefix) {
      cpp_int product = 1;
      for (prefix = 1; prefix <= 20; ++prefix) {
        product *= rns8_default_modulus(prefix - 1);
        if (product > uint64_t(k) * 49 * (u ? 1 : 2))
          break;
      }
    }
    cpp_int product = 1;
    for (uint32_t p = 0; p < prefix; ++p)
      product *= rns8_default_modulus(p);
    if (prefix > 20 || product <= uint64_t(k) * 49 * (u ? 1 : 2))
      throw std::invalid_argument("strict prefix capacity");
  }
};
struct TaskOwner {
  rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr;
  rns8_workspace* workspace = nullptr;
};
struct Batch {
  const Dataset& data;
  const json& config;
  rns8_context* ctx = nullptr;
  rns8_plan* plan = nullptr;
  std::vector<TaskOwner> owned;
  std::vector<rns8_grouped_gemm_task> tasks;
  Batch(const Dataset& d, const json& c) : data(d), config(c), owned(d.count), tasks(d.count) {}
  rns8_status close() {
    for (auto& task : owned) {
      for (auto** matrix : {&task.c, &task.b, &task.a})
        if (*matrix) {
          const auto status = rns8_destroy_matrix(*matrix);
          if (status != RNS8_SUCCESS)
            return status;
          *matrix = nullptr;
        }
      if (task.workspace) {
        const auto status = rns8_destroy_workspace(task.workspace);
        if (status != RNS8_SUCCESS)
          return status;
        task.workspace = nullptr;
      }
    }
    if (plan) {
      const auto status = rns8_destroy_plan(plan);
      if (status != RNS8_SUCCESS)
        return status;
      plan = nullptr;
    }
    if (ctx) {
      const auto status = rns8_destroy_context(ctx);
      if (status != RNS8_SUCCESS)
        return status;
      ctx = nullptr;
    }
    return RNS8_SUCCESS;
  }
  ~Batch() { (void)close(); }
  void pack(bool left = true, bool right = true) {
    const auto one = [&](rns8_matrix* matrix, const std::vector<uint64_t>& values, int64_t ld) {
      check(data.u ? rns8_pack_u64(ctx, matrix, values.data(), ld, 0)
                   : rns8_pack_i64(ctx, matrix, reinterpret_cast<const int64_t*>(values.data()), ld,
                                   0));
    };
    for (uint32_t i = 0; i < data.count; ++i) {
      if (left)
        one(owned[i].a, data.a[i], data.k + 1);
      if (right && owned[i].b)
        one(owned[i].b, data.b[data.shared_b ? 0 : i], data.n + 2);
    }
  }
  void initialize() {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    check(rns8_create_context(0, &options, &ctx));
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = data.u ? RNS8_EXACT_WIDE_UNSIGNED : RNS8_EXACT_WIDE_SIGNED;
    desc.bound_kind = RNS8_BOUND_NONE;
    desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    desc.m = data.m;
    desc.n = data.n;
    desc.k = data.k;
    desc.max_prefix = data.prefix;
    desc.flags =
        RNS8_PLAN_FORCE_FIXED_PREFIX | RNS8_PLAN_EXACT_VALUE_BOUND | RNS8_PLAN_EXACT_SCALAR_FIRST;
    check(rns8_create_plan(ctx, &desc, &plan));
    const auto matrix = [&](uint32_t rows, uint32_t cols, bool input, rns8_matrix** out) {
      rns8_matrix_desc md{};
      md.struct_size = sizeof(md);
      md.abi_version = RNS8_ABI_VERSION;
      md.semantics = desc.semantics;
      md.bound_kind = RNS8_BOUND_NONE;
      md.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
      md.rows = rows;
      md.cols = cols;
      md.logical_ld = cols;
      md.max_prefix = data.prefix;
      md.flags = input && config["mode"] != "rns" ? RNS8_MATRIX_RETAIN_NATIVE_INPUT : 0;
      check(rns8_create_matrix(ctx, &md, out));
    };
    for (uint32_t i = 0; i < data.count; ++i) {
      auto& task = owned[i];
      matrix(data.m, data.k, true, &task.a);
      if (!data.shared_b || !i)
        matrix(data.k, data.n, true, &task.b);
      matrix(data.m, data.n, false, &task.c);
      check(rns8_create_workspace(ctx, plan, &task.workspace));
      tasks[i] = {sizeof(tasks[i]), RNS8_ABI_VERSION, task.a, data.shared_b ? owned[0].b : task.b,
                  task.c,           task.workspace};
    }
    pack();
  }
  void product(rns8_narrow_product_method method, uint64_t budget) {
    if (config["mode"] == "group")
      check(rns8_gemm_narrow_exact_grouped(ctx, plan, tasks.data(), data.count, method, budget));
    else
      for (auto& task : tasks)
        if (config["mode"] == "rns")
          check(rns8_gemm_rns(ctx, plan, task.a, task.b, task.c, task.workspace));
        else
          check(rns8_gemm_narrow_exact(ctx, plan, task.a, task.b, task.c, method, budget,
                                       task.workspace));
  }
  void export_all(std::vector<uint64_t>& output) {
    const size_t stride = size_t(data.m) * (data.n + 3) * 3;
    for (uint32_t i = 0; i < data.count; ++i)
      check(data.u ? rns8_export_exact_wide_unsigned_limbs(
                         ctx, plan, owned[i].c, output.data() + i * stride, data.n + 3, 3)
                   : rns8_export_exact_wide_signed_limbs(
                         ctx, plan, owned[i].c, output.data() + i * stride, data.n + 3, 3));
  }
  json summaries() const {
    if (config["mode"] == "rns")
      return nullptr;
    auto rows = json::array();
    for (const auto& task : owned) {
      rns8_narrow_summary_info info{};
      info.struct_size = sizeof(info);
      info.abi_version = RNS8_ABI_VERSION;
      check(rns8_get_narrow_summary_info(task.workspace, &info));
      rows.push_back({{"a_instance", info.a_instance},
                      {"b_instance", info.b_instance},
                      {"a_generation", info.a_generation},
                      {"b_generation", info.b_generation},
                      {"a_host_proofs", info.a_host_proofs},
                      {"b_host_proofs", info.b_host_proofs},
                      {"a_reuses", info.a_reuses},
                      {"b_reuses", info.b_reuses},
                      {"a_device_scans", info.a_device_scans},
                      {"b_device_scans", info.b_device_scans},
                      {"scan_bytes", info.retained_scan_bytes},
                      {"partial_bytes", info.retained_product_bytes},
                      {"directory_bytes", info.retained_group_directory_bytes},
                      {"directory_tasks", info.retained_group_task_count}});
    }
    return rows;
  }
};
} // namespace

int main(int argc, char** argv) {
  const int identity = rns8::tools::emit_build_identity(argc, argv, std::cout);
  if (identity >= 0)
    return identity;
  try {
    if (argc != 4 || std::strcmp(argv[3], "--record-execution-environment"))
      throw std::invalid_argument(
          "usage: rns8-narrow-group-calibrate CASE.json unique-run-id --record-execution-environment");
    rns8::tools::ExecutionEnvironmentWitness environment(true, argc, argv);
    json config;
    std::ifstream(argv[1]) >> config;
    if (config.size() != 15 || config.at("schema") != "narrow_group_case_v1" ||
        !config.at("unsigned_inputs").is_boolean() || !config.at("shared_b").is_boolean())
      throw std::invalid_argument("case ABI");
    const auto m = number(config, "m", 1, 512), n = number(config, "n", 1, 512),
               k = number(config, "k", 1, 131073), count = number(config, "tasks", 2, 64),
               repeats = number(config, "repeats", 3, 100),
               warmups = number(config, "warmups", 1, 100);
    number(config, "prefix", 0, 20);
    number(config, "seed", 0, UINT32_MAX);
    const std::string mode = config.at("mode"), method_name = config.at("method"),
                      residency = config.at("residency"), output_boundary = config.at("output");
    const bool u = config.at("unsigned_inputs"), resident_output = output_boundary == "resident";
    if ((mode != "single" && mode != "group" && mode != "rns") ||
        (method_name != "scalar" && method_name != "wmma" && method_name != "split-k") ||
        (residency != "cold" && residency != "resident" && residency != "repack" &&
         residency != "repack-a" && residency != "repack-b") ||
        (output_boundary != "resident" && output_boundary != "limbs") ||
        (resident_output && residency == "cold") || (u && method_name != "wmma") ||
        uint64_t(count) * m * n * k > 134217728)
      throw std::invalid_argument("bounded group recipe/domain");
    const auto method = method_name == "wmma"      ? RNS8_NARROW_BYTE_WMMA_FUSED_RNS
                        : method_name == "split-k" ? RNS8_NARROW_SPLIT_K_I64_FUSED_RNS
                                                   : RNS8_NARROW_SCALAR_FUSED_RNS;
    Dataset data(config); // All arbitrary-precision reference work is outside timing.
    rns8::tools::DeviceIdentityWitness device_witness(0, true);
    rns8::detail::hip_direct_timing_set_enabled(false);
    const auto make = [&] {
      auto value = std::make_unique<Batch>(data, config);
      value->initialize();
      return value;
    };
    const auto start = Clock::now();
    auto owner = make();
    const double setup = microseconds(start, Clock::now());
    rns8_device_info device{};
    device.struct_size = sizeof(device);
    device.abi_version = RNS8_ABI_VERSION;
    check(rns8_get_device_info(owner->ctx, &device));
    const auto support =
        rns8::detail::query_narrow_group_support(m, n, k, data.prefix, method, count, UINT64_MAX);
    check(support.status);
    const uint64_t budget =
        mode == "rns" ? 0
        : mode == "group"
            ? support.scratch_bytes
            : (support.scratch_bytes - count * sizeof(rns8::detail::NativeNarrowTask)) / count;
    std::vector<uint64_t> output(size_t(count) * m * (n + 3) * 3, UINT64_C(0x5a5a5a5a5a5a5a5a));
    if (residency == "cold") {
      check(owner->close());
      owner.reset();
    }
    const auto call = [&](bool timed) {
      const rns8::bench::CallClock<> clock(timed);
      const auto begin = clock.now();
      if (!owner)
        owner = make();
      else if (residency != "resident")
        owner->pack(residency != "repack-b", residency != "repack-a");
      const auto prepared = clock.now();
      owner->product(method, budget);
      const auto computed = clock.now();
      if (!resident_output)
        owner->export_all(output);
      const auto exported = resident_output ? computed : clock.now();
      if (residency == "cold") {
        check(owner->close());
        owner.reset();
      }
      const auto end = clock.now();
      return rns8::bench::PhaseObservation<4>{
          {microseconds(begin, prepared), microseconds(prepared, computed),
           microseconds(computed, exported), microseconds(exported, end)},
          microseconds(begin, end)};
    };
    const auto verify = [&] {
      if (resident_output)
        owner->export_all(output);
      for (uint32_t task = 0; task < count; ++task)
        for (uint32_t row = 0; row < m; ++row)
          for (uint32_t col = 0; col < n + 3; ++col) {
            cpp_int value = col < n ? data.expected[(size_t(task) * m + row) * n + col] : 0;
            if (value < 0)
              value += cpp_int(1) << 192;
            for (unsigned limb = 0; limb < 3; ++limb) {
              const auto expected = col < n ? uint64_t((value >> (64 * limb)) & UINT64_MAX)
                                            : UINT64_C(0x5a5a5a5a5a5a5a5a);
              if (output[((size_t(task) * m + row) * (n + 3) + col) * 3 + limb] != expected)
                throw std::runtime_error(
                    "complete independent task/output/padding reference mismatch");
            }
          }
    };
    const auto sample = [&](const auto& value) {
      return rns8::bench::phase_observation_json(
          value, {"prepare_us", "execute_us", "export_us", "cleanup_us"});
    };
    const auto first = sample(call(true));
    verify();
    const auto summaries_first = owner ? owner->summaries() : json(nullptr);
    for (uint32_t i = 0; i < warmups; ++i)
      call(false);
    auto samples = json::array(), sustained = json::array();
    for (uint32_t i = 0; i < repeats; ++i) {
      samples.push_back(sample(call(true)));
      verify();
    }
    for (unsigned i = 0; i < 3; ++i) {
      const auto burst = rns8::bench::measure_completed_calls([&] { call(false); });
      verify();
      sustained.push_back({{"calls", burst.calls},
                           {"total_us", burst.duration_us},
                           {"per_call_us", burst.duration_us / burst.calls},
                           {"minimum_duration_reached", burst.duration_us >= 250000}});
    }
    const auto summaries_final = owner ? owner->summaries() : json(nullptr);
    rns8_context* observer = owner ? owner->ctx : nullptr;
    rns8_context* separate = nullptr;
    if (!observer) {
      rns8_context_options options{};
      options.struct_size = sizeof(options);
      options.abi_version = RNS8_ABI_VERSION;
      options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      check(rns8_create_context(0, &options, &separate));
      observer = separate;
    }
    auto diagnostic = rns8_benchmark::observe_native_batch(observer, [&] { call(false); });
    verify();
    diagnostic["scope"] = "one_separate_complete_task_batch";
    diagnostic["inside_timed_samples"] = false;
    diagnostic["inside_sustained_interval"] = false;
    diagnostic["exact_reference_passed"] = true;
    diagnostic["public_product_calls"] = mode == "group" ? 1 : count;
    if (separate)
      check(rns8_destroy_context(separate));
    if (!owner) {
      owner = make();
      owner->product(method, budget);
    }
    std::vector<int8_t> planes(size_t(data.prefix) * m * n);
    for (uint32_t task = 0; task < count; ++task) {
      check(rns8::detail::hip_direct_copy_device_to_host(
          0, planes.data(), owner->owned[task].c->hip_residues, planes.size()));
      for (uint32_t p = 0; p < data.prefix; ++p)
        for (size_t cell = 0; cell < size_t(m) * n; ++cell) {
          const auto modulus = rns8_default_modulus(p);
          cpp_int residue = data.expected[size_t(task) * m * n + cell] % modulus;
          if (residue < 0)
            residue += modulus;
          if (residue >= (modulus + 1) / 2)
            residue -= modulus;
          if (planes[size_t(p) * m * n + cell] != residue)
            throw std::runtime_error("complete independent task residue audit mismatch");
        }
    }
    check(owner->close());
    owner.reset();
    device_witness.finish();
    environment.finish();
    std::ostringstream physical, native;
    device_witness.print(physical);
    environment.print(native);
    std::cout << json{{"schema", "rns8_narrow_group_microcase_v1"},
                      {"case", config},
                      {"run_id", argv[2]},
                      {"process_id", rns8::tools::process_id()},
                      {"build_manifest", json::parse(RNS8_BUILD_MANIFEST_JSON)},
                      {"device",
                       {{"device_id", device.device_id},
                        {"hip_available", device.hip_available},
                        {"gcn_arch", device.gcn_arch},
                        {"name", device.name},
                        {"hip_runtime_version", device.hip_runtime_version},
                        {"hip_driver_version", device.hip_driver_version},
                        {"global_mem_bytes", device.global_mem_bytes}}},
                      {"device_identity_witness", json::parse(physical.str())},
                      {"execution_environment_witness", json::parse(native.str())},
                      {"dataset", "affine_independent_native_tasks_v1"},
                      {"selected_prefix", data.prefix},
                      {"exact_proof_policy", "scalar_first"},
                      {"host_instrumentation_state", "hooks_disabled"},
                      {"hot_loop_serialization",
                       "typed_observation_no_phase_clocks_in_sustained_v2"},
                      {"output_boundary", resident_output
                                              ? "complete_task_batch_resident_rns"
                                              : "complete_task_batch_padded_three_limb_export"},
                      {"private_narrow_budget_bytes", budget},
                      {"setup_us", setup},
                      {"first_call", first},
                      {"samples", samples},
                      {"sustained", sustained},
                      {"task_directory_entry_bytes", sizeof(rns8::detail::NativeNarrowTask)},
                      {"summary_first", summaries_first},
                      {"summary_final", summaries_final},
                      {"separate_native_observation", diagnostic},
                      {"native_useful_products", uint64_t(count) * m * n * k},
                      {"full_output_reference_values", uint64_t(count) * m * n},
                      {"exact_reference_passed", true},
                      {"all_task_residue_planes_audited", true},
                      {"oracle_inside_production_interval", false},
                      {"promotion_eligible", false}}
                     .dump(2)
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
