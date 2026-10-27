#include "core/distributed_hip.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include "backend_common/sha256.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
namespace d = rns8::detail::distributed;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };
void check(d::Admission a) {
  if (a.status == RNS8_UNSUPPORTED_BACKEND || a.status == RNS8_UNSUPPORTED_ARCH) throw Unsupported(a.reason);
  if (!a) throw std::runtime_error(std::string(a.reason) + ";status=" + rns8_status_string(a.status) +
      ";rank=" + std::to_string(a.rank) + ";work=" + std::to_string(a.work));
}
uint64_t integer(const json& c, const char* key, uint64_t lo, uint64_t hi) {
  const auto& value = c.at(key);
  if (!value.is_number_unsigned() || value.get<uint64_t>() < lo || value.get<uint64_t>() > hi)
    throw std::invalid_argument(key);
  return value.get<uint64_t>();
}
uint64_t elapsed(Clock::time_point start) {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}
}
int main(int argc, char** argv) { try {
  if (argc != 3) throw std::invalid_argument("usage: rns8-distributed-calibrate CASE.json unique-run-id");
  json c; std::ifstream(argv[1]) >> c;
  if (c.at("schema") != "rns8_distributed_case_v3" || c.size() != 21 || std::string(argv[2]).empty())
    throw std::invalid_argument("case/run identity");
  d::Request r; r.operation = 5001; r.semantics = RNS8_BOUNDED_I64; r.host_storage = d::HostStorage::limbs_only;
  if(!c["native_integrity"].is_boolean())throw std::invalid_argument("integrity policy");r.native_integrity=c["native_integrity"].get<bool>();
  if(!c.at("basis").is_array() || c["basis"].size()>20)throw std::invalid_argument("basis domain");
  for(const auto& q:c["basis"]){if(!q.is_number_unsigned() || q.get<uint64_t>()<2 || q.get<uint64_t>()>256)throw std::invalid_argument("basis modulus");r.basis.push_back(q.get<uint16_t>());}
  r.m = integer(c, "m", 1, 8192); r.n = integer(c, "n", 1, 8192); r.k = integer(c, "k", 1, 131073);
  {const std::string axis=c.at("axis");
    if(axis=="modulus")r.axis=d::Axis::modulus;else if(axis=="rows")r.axis=d::Axis::rows;else if(axis=="columns")r.axis=d::Axis::columns;
    else if(axis=="2d")r.axis=d::Axis::hybrid;else if(axis=="contraction")r.axis=d::Axis::contraction;else throw std::invalid_argument("axis");
    r.k_fragments=uint32_t(integer(c,"k_fragments",0,64));r.rank_grid_rows=uint32_t(integer(c,"rank_grid_rows",0,64));r.rank_grid_cols=uint32_t(integer(c,"rank_grid_cols",0,64));}
  r.lda = r.k; r.ldb = r.n; r.prefix = uint32_t(integer(c, "prefix", 1, 20));
  r.tile_rows = r.tile_cols = uint32_t(integer(c, "tile", 1, 256));
  r.pipeline_slots = uint32_t(integer(c, "slots", 1, 3)); r.magnitude_bound = 6 * r.k;
  r.host_budget = integer(c, "host_budget", 1, uint64_t{1} << 60);
  r.pinned_budget = integer(c, "pinned_budget", 4, uint64_t{1} << 30);
  const auto device_budget = integer(c, "device_budget", 1, uint64_t{1} << 60);
  const auto warmups = integer(c, "warmups", 1, 100), repeats = integer(c, "repeats", 3, 100);
  const auto scope = c.at("scope").get<std::string>();
  if (scope != "resident-device-limbs" && scope != "resident-host-limbs" && scope != "cold-host-limbs")
    throw std::invalid_argument("scope");
  const auto mode = c.at("transport").get<std::string>();
  if (mode != "local" && mode != "peer" && mode != "pinned") throw std::invalid_argument("transport");
  std::vector<uint32_t> ordinals;
  if (!c.at("devices").is_array() || c.at("devices").empty() || c.at("devices").size() > 64)
    throw std::invalid_argument("devices");
  for (const auto& item : c.at("devices")) {
    if (!item.is_number_unsigned() || item.get<uint64_t>() > INT32_MAX) throw std::invalid_argument("device ordinal");
    ordinals.push_back(item.get<uint32_t>());
  }
  check(d::HipExecution::discover(ordinals, r.ranks, r.edges));
  for (auto& rank : r.ranks) rank.memory_budget = std::min(rank.memory_budget, device_budget);
  r.transport = mode == "local" ? d::Transport::local : mode == "peer" ? d::Transport::peer : d::Transport::pinned_host;
  r.a = {7001, 1}; r.b = {7002, 1};
  r.certificate_sha256 = rns8::detail::sha256("distributed-native-bound-v1;separable_signed_narrow;K=" + std::to_string(r.k));
  r.layout_sha256 = rns8::detail::sha256("distributed-global-plane-row-major-v1;tile=" + std::to_string(r.tile_rows));
  const auto plan_start = Clock::now();
  std::shared_ptr<const d::Plan> plan; check(d::Plan::create(r, plan));
  const auto plan_compilation_ns = elapsed(plan_start);
  const uint64_t caller_output_bytes = r.m * (r.n + 2) * 24;
  if (caller_output_bytes > r.host_budget || plan->resources().host_model > r.host_budget - caller_output_bytes)
    throw std::invalid_argument("complete host reservation including caller output exceeds budget");
  std::vector<uint64_t> av(size_t(r.m * r.k)), bv(size_t(r.k * r.n));
  for (uint64_t row = 0; row < r.m; ++row) for (uint64_t k = 0; k < r.k; ++k)
    av[row * r.k + k] = uint64_t((int64_t(row % 7) - 3) * (int64_t(k % 3) - 1));
  for (uint64_t k = 0; k < r.k; ++k) for (uint64_t col = 0; col < r.n; ++col)
    bv[k * r.n + col] = uint64_t((int64_t(k % 3) - 1) * (int64_t(col % 5) - 2));
  auto inputs = std::make_shared<const d::Inputs>(r.a, r.b, std::move(av), std::move(bv));
  // An independent Boost dot checks the recipe's closed form before timing.
  // All output cells are checked against that form, avoiding a cubic CPU oracle
  // for the required 8192-square qualification workload.
  const int64_t inner = int64_t(2 * (r.k / 3) + (r.k % 3 != 0));
  for (uint64_t row : {uint64_t{0}, r.m - 1}) for (uint64_t col : {uint64_t{0}, r.n - 1}) {
    d::Integer exact = 0;
    const auto decode = [](uint64_t bits) -> d::Integer {
      return bits > uint64_t(INT64_MAX) ? d::Integer(bits) - (d::Integer(1) << 64) : d::Integer(bits);
    };
    for (uint64_t k = 0; k < r.k; ++k) exact += decode(inputs->lhs()[row * r.k + k]) * decode(inputs->rhs()[k * r.n + col]);
    if (exact != (int64_t(row % 7) - 3) * (int64_t(col % 5) - 2) * inner) throw std::runtime_error("closed-form recipe audit");
  }
  const bool cold = scope == "cold-host-limbs", host = scope != "resident-device-limbs";
  std::unique_ptr<d::HipExecution> execution;
  std::vector<std::string> certificates;
  const auto setup_start = Clock::now();
  check(d::HipExecution::create(plan, inputs, execution));
  certificates = execution->arithmetic_certificates();
  if (cold) { check(execution->close()); execution.reset(); }
  const auto setup_ns = elapsed(setup_start);
  std::vector<uint64_t> output(size_t(caller_output_bytes / 8), UINT64_C(0xdeadbeef));
  uint64_t epoch = 0, calls = 0; d::HipProgress last;
  const auto invoke = [&] {
    if (cold) { check(d::HipExecution::create(plan, inputs, execution)); epoch = 0; }
    check(execution->submit(++epoch, UINT64_C(120000000000)));
    check(execution->wait(epoch, UINT64_C(120000000000), last)); check(last.operation);
    if (!last.output_current || last.local_completed != plan->work().size() || last.transfers_queued != plan->work().size())
      throw std::runtime_error("incomplete distributed coverage/publication");
    if (host) check(execution->export_limbs(epoch, output.data(), r.n + 2));
    if (cold) { check(execution->close()); execution.reset(); }
    ++calls;
  };
  const auto audit = [&] {
    if (!host) check(execution->export_limbs(epoch, output.data(), r.n + 2));
    for (uint64_t row = 0; row < r.m; ++row) for (uint64_t col = 0; col < r.n + 2; ++col) {
      const int64_t exact = (int64_t(row % 7) - 3) * (int64_t(col % 5) - 2) * inner;
      for (unsigned limb = 0; limb < 3; ++limb) {
        const uint64_t expected = col >= r.n ? UINT64_C(0xdeadbeef) : limb == 0 ? uint64_t(exact) : exact < 0 ? UINT64_MAX : 0;
        if (output[(row * (r.n + 2) + col) * 3 + limb] != expected) throw std::runtime_error("exact output/padding audit");
      }
    }
  };
  auto start = Clock::now(); invoke(); const auto first_ns = elapsed(start); audit();
  for (uint64_t i = 0; i < warmups; ++i) { invoke(); audit(); }
  std::vector<uint64_t> samples; samples.reserve(size_t(repeats));
  for (uint64_t i = 0; i < repeats; ++i) { start = Clock::now(); invoke(); samples.push_back(elapsed(start)); audit(); }
  json bursts = json::array();
  for (unsigned burst = 0; burst < 3; ++burst) {
    uint64_t count = 0; start = Clock::now();
    do { invoke(); ++count; } while (count < 4096 && (count < 16 || elapsed(start) < 250000000));
    const auto ns = elapsed(start); audit();
    bursts.push_back({{"calls", count}, {"elapsed_ns", ns}, {"mean_call_ns", double(ns) / double(count)},
        {"duration_reached", ns >= 250000000}, {"stop_reason", ns >= 250000000 ? "duration" : "call_cap"}});
  }
  if (!cold) { check(execution->close()); execution.reset(); }
  json ranks = json::array();
  for (uint32_t rank = 0; rank < r.ranks.size(); ++rank) {
    rns8_device_info device{};
    if (rns8::detail::hip_direct_probe(int(r.ranks[rank].ordinal), device) != RNS8_SUCCESS) throw std::runtime_error("final device identity");
    const auto& bytes = plan->resources().ranks[rank];
    ranks.push_back({{"rank", rank}, {"ordinal", r.ranks[rank].ordinal}, {"physical_id", r.ranks[rank].physical_id},
        {"physical_package_id", r.ranks[rank].physical_package_id}, {"compute_partition", r.ranks[rank].compute_partition},
        {"memory_partition", r.ranks[rank].memory_partition}, {"target", r.ranks[rank].target},
        {"runtime_version", device.hip_runtime_version}, {"driver_version", device.hip_driver_version},
        {"requested_device_reservation", bytes.peak}, {"native_inputs", bytes.native_inputs},
        {"packed_inputs", bytes.packed_inputs}, {"local_output", bytes.local_outputs}, {"gather", bytes.gather},{"partial_gather",bytes.partial_gather}});
  }
  json manifest = nullptr;
  const char* host_platform = "other";
#if defined(_WIN32)
  host_platform = "native-windows";
#elif defined(__linux__)
  host_platform = "native-linux";
#endif
#ifdef RNS8_BUILD_MANIFEST_JSON
  manifest = json::parse(RNS8_BUILD_MANIFEST_JSON);
#endif
  std::cout << json{{"schema", "rns8_distributed_microcase_v1"}, {"case", c}, {"run_id", argv[2]},
      {"build_manifest", manifest}, {"dataset", "signed_separable_7x3x5_v1"}, {"ranks", ranks},
      {"host_platform", host_platform}, {"host_hardware_threads", std::thread::hardware_concurrency()},
      {"affinity_policy", "inherited_not_modified"}, {"numa_placement_observed", false},
      {"physical_card_count_known", false}, {"plan_sha256", rns8::detail::sha256(plan->identity())}, {"basis_moduli",plan->request().basis},
      {"arithmetic_certificates", certificates},{"distribution_axis",unsigned(r.axis)},{"k_fragments",plan->k_fragments()}, {"expected_work", plan->work().size()},
      {"local_completed", last.local_completed}, {"transfers_queued", last.transfers_queued},
      {"published", last.output_current}, {"calls", calls}, {"setup_ns", setup_ns}, {"first_ns", first_ns},
      {"integrity_requested",last.integrity_requested},{"integrity_completed",last.integrity_completed},
      {"integrity_failure_mask",last.integrity_failure_mask},{"checking_moduli",json::array({257,263})},
      {"setup_scope", cold ? "untimed_preparation_admission_probe_and_cleanup" : "retained_preparation"},
      {"plan_compilation_ns", plan_compilation_ns}, {"plan_reused", true}, {"production_timing_hooks_enabled", false},
      {"samples_ns", samples}, {"sustained", bursts}, {"transfer_bytes_per_call", plan->resources().transfer_bytes},
      {"host_owner_reservation", plan->resources().host_model}, {"caller_output_bytes", caller_output_bytes},
      {"pinned_reservation", plan->resources().pinned_staging}, {"timer", "single_host_steady_clock_nanoseconds"},
      {"completion_inside_interval", true}, {"oracle_inside_interval", false}, {"exact_reference_passed", true},
      {"promotion_eligible", false}, {"allocator_overhead_known", false}}.dump(2) << '\n';
  return 0;
} catch (const Unsupported& e) { std::cerr << "unsupported: " << e.what() << '\n'; return 77;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
