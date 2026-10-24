#ifndef RNS8_WORKLOAD_COMMON_HPP
#define RNS8_WORKLOAD_COMMON_HPP

#include <rns8/rns8.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if __has_include("workload_source_identity.hpp")
#include "workload_source_identity.hpp"
#endif

namespace rns8_workload {
using boost::multiprecision::cpp_int;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Words = std::vector<uint64_t>;

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}
inline double since(Clock::time_point start) {
  return std::chrono::duration<double, std::nano>(Clock::now() - start).count();
}
inline uint64_t number(const std::string& text, uint64_t low, uint64_t high, const char* option) {
  uint64_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  require(result.ec == std::errc{} && result.ptr == text.data() + text.size() && value >= low && value <= high,
          std::string("invalid ") + option + " (expected " + std::to_string(low) + ".." + std::to_string(high) + ")");
  return value;
}
inline const char* backend_name(rns8_backend_kind backend) {
  switch (backend) {
    case RNS8_BACKEND_CPU_REFERENCE: return "cpu-reference";
    case RNS8_BACKEND_HIP_DIRECT: return "hip-direct";
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64: return "hip-vector-alu-int64";
    case RNS8_BACKEND_HIPBLASLT: return "hipblaslt";
    case RNS8_BACKEND_CK: return "ck";
    case RNS8_BACKEND_ROCWMMA: return "rocwmma";
    case RNS8_BACKEND_AMDGPU_BUILTINS: return "amdgpu-builtins";
    default: throw std::invalid_argument("backend has no workload implementation");
  }
}
inline rns8_backend_kind parse_backend(const std::string& name) {
  for (auto backend : {RNS8_BACKEND_CPU_REFERENCE, RNS8_BACKEND_HIP_DIRECT,
       RNS8_BACKEND_HIP_VECTOR_ALU_INT64, RNS8_BACKEND_HIPBLASLT, RNS8_BACKEND_CK,
       RNS8_BACKEND_ROCWMMA, RNS8_BACKEND_AMDGPU_BUILTINS})
    if (name == backend_name(backend)) return backend;
  throw std::invalid_argument("choose an explicit supported backend; AUTO/fallback is not used");
}
struct MemoryBudget {
  uint64_t reserved = 0;
  void reserve(uint64_t bytes, uint64_t limit, const char* label) {
    require(reserved <= limit && bytes <= limit - reserved, std::string(label) + " exceeds aggregate declared memory envelope");
    reserved += bytes;
  }
};
struct Options {
  uint32_t m = 64, n = 64, k = 96, bits = 8, repeats = 3, warmups = 1, sustained_ms = 0, gram_reuse = 0;
  uint64_t seed = 3135, memory_mib = 256, baseline_max_products = 4000000;
  rns8_backend_kind backend = RNS8_BACKEND_HIP_DIRECT;
  int device = 0;
  bool json = false, wide = false, gram = false, reference_only = false;
  std::string size = "modest", pattern = "seeded", run_id = "standalone", gram_export = "once";
  std::set<std::string> supplied;
  std::shared_ptr<MemoryBudget> memory = std::make_shared<MemoryBudget>();
};
inline Options options(int argc, char** argv) {
  Options out;
  bool device_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    require(out.supplied.insert(key).second, "duplicate option: " + key);
    if (key == "--help") {
      std::cout << "Checked RNS8 workload; explicit backend/device, no AUTO fallback.\n"
          "--backend hip-direct|cpu-reference|hip-vector-alu-int64|hipblaslt|ck|rocwmma|amdgpu-builtins\n"
          "--device 0..63 (CPU uses -1) --reference-only --json --run-id ID\n"
          "--size tiny|modest|large --m M --n N --k K (1..512) --bits 1..63\n"
          "--seed U64 --pattern seeded|zero|sparse --repeats 1..100 --warmups 0..20\n"
          "--gram-reuse 1..256 --gram-export each|once (explicit --gram batch only; default once)\n"
          "--sustained-ms 0..10000 (0 disables production-style repeated calls)\n"
          "--memory-mib 1..4096 --baseline-max-products 1..134217728 --wide --gram\n"
          "tiny uses a named hand-checkable fixture; large is explicit opt-in.\n"
          "Field/resident/complex/exactness programs reject irrelevant controls.\n";
      std::exit(0);
    }
    if (key == "--json") { out.json = true; continue; }
    if (key == "--wide") { out.wide = true; continue; }
    if (key == "--gram") { out.gram = true; continue; }
    if (key == "--reference-only") { out.reference_only = true; continue; }
    require(i + 1 < argc, "missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--backend") out.backend = parse_backend(value);
    else if (key == "--device") {
      out.device = value == "-1" ? -1 : static_cast<int>(number(value, 0, 63, "device")); device_set = true;
    } else if (key == "--size") {
      require(value == "tiny" || value == "modest" || value == "large", "invalid --size");
      out.size = value;
    } else if (key == "--m") out.m = static_cast<uint32_t>(number(value, 1, 512, "m"));
    else if (key == "--n") out.n = static_cast<uint32_t>(number(value, 1, 512, "n"));
    else if (key == "--k") out.k = static_cast<uint32_t>(number(value, 1, 512, "k"));
    else if (key == "--bits") out.bits = static_cast<uint32_t>(number(value, 1, 63, "bits"));
    else if (key == "--gram-reuse") out.gram_reuse = static_cast<uint32_t>(number(value, 1, 256, "gram-reuse"));
    else if (key == "--gram-export") {
      require(value == "each" || value == "once", "invalid --gram-export"); out.gram_export = value;
    }
    else if (key == "--seed") out.seed = number(value, 0, UINT64_MAX, "seed");
    else if (key == "--repeats") out.repeats = static_cast<uint32_t>(number(value, 1, 100, "repeats"));
    else if (key == "--warmups") out.warmups = static_cast<uint32_t>(number(value, 0, 20, "warmups"));
    else if (key == "--sustained-ms") out.sustained_ms = static_cast<uint32_t>(number(value, 0, 10000, "sustained-ms"));
    else if (key == "--memory-mib") out.memory_mib = number(value, 1, 4096, "memory-mib");
    else if (key == "--baseline-max-products") out.baseline_max_products = number(value, 1, 134217728, "baseline-max-products");
    else if (key == "--pattern") {
      require(value == "seeded" || value == "zero" || value == "sparse", "invalid --pattern"); out.pattern = value;
    } else if (key == "--run-id") { require(!value.empty() && value.size() <= 128, "invalid --run-id"); out.run_id = value; }
    else throw std::invalid_argument("unknown option: " + key);
  }
  if (out.reference_only) {
    require(!out.supplied.count("--backend") || out.backend == RNS8_BACKEND_CPU_REFERENCE,
            "--reference-only conflicts with a GPU backend");
    out.backend = RNS8_BACKEND_CPU_REFERENCE;
  }
  if (out.backend == RNS8_BACKEND_CPU_REFERENCE) {
    require(!device_set || out.device == -1, "CPU backend uses --device -1"); out.device = -1;
  } else require(out.device >= 0, "GPU requires a nonnegative explicit device");
  const uint32_t tiny[3] = {2, 2, 3}, large[3] = {256, 256, 256};
  if (out.size != "modest") {
    const auto* shape = out.size == "tiny" ? tiny : large;
    if (!out.supplied.count("--m")) out.m = shape[0];
    if (!out.supplied.count("--n")) out.n = shape[1];
    if (!out.supplied.count("--k")) out.k = shape[2];
  }
  if (out.gram) {
    require(!out.supplied.count("--n") || out.n == out.m, "Gram requires N=M features"); out.n = out.m;
  }
  require(!out.gram_reuse || out.gram, "--gram-reuse requires --gram");
  require(!out.supplied.count("--gram-export") || out.gram_reuse, "--gram-export requires --gram-reuse");
  if (out.gram_reuse)
    require(uint64_t(out.m)*out.n*out.k*out.gram_reuse <= 134217728,
            "Gram batch exceeds134217728 mathematical products; reduce shape/reuse");
  return out;
}
inline Json gram_batch_contract(const Options& options) {
  return {{"version",1},{"products_per_batch",options.gram_reuse},
          {"export_policy",options.gram_export},{"reference_outputs_per_batch",1},
          {"input_preparation","pack_once_per_generation"},
          {"scope","R identical products recomputed synchronously; complete final native host output"}};
}
inline void reject_options(const Options& options, std::initializer_list<const char*> keys) {
  for (const char* key : keys) require(!options.supplied.count(key), std::string(key) + " is irrelevant for this workload");
}
inline uint64_t next_word(uint64_t& state) {
  uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}
inline uint64_t magnitude(int64_t value) {
  return value < 0 ? uint64_t{0} - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
}
inline uint64_t bits_of(int64_t value) {
  uint64_t bits; std::memcpy(&bits, &value, sizeof(bits)); return bits;
}
inline uint64_t checksum(const Words& words) {
  // Canonical FNV-1a over every u64, serialized least-significant byte first.
  uint64_t hash = UINT64_C(14695981039346656037);
  for (uint64_t word : words) for (unsigned byte = 0; byte < 8; ++byte) {
    hash ^= (word >> (byte * 8)) & 255;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}
inline std::string hex64(uint64_t value) {
  std::ostringstream text; text << std::hex << std::setfill('0') << std::setw(16) << value; return text.str();
}
inline Words integer_words(const std::vector<cpp_int>& values, uint32_t limbs) {
  Words result(values.size() * limbs);
  const cpp_int limit = cpp_int(1) << (64 * limbs), half = limit >> 1, mask = (cpp_int(1) << 64) - 1;
  for (std::size_t cell = 0; cell < values.size(); ++cell) {
    cpp_int value = values[cell];
    require(value >= -half && value < half, "exact output does not fit requested signed export width");
    if (value < 0) value += limit;
    for (uint32_t limb = 0; limb < limbs; ++limb) {
      result[cell * limbs + limb] = (value & mask).convert_to<uint64_t>(); value >>= 64;
    }
  }
  return result;
}
inline cpp_int decode(const Words& values, std::size_t cell, uint32_t limbs) {
  cpp_int value = 0;
  for (uint32_t limb = limbs; limb-- > 0;) { value <<= 64; value += values[cell * limbs + limb]; }
  if (values[cell * limbs + limbs - 1] >> 63) value -= cpp_int(1) << (64 * limbs);
  return value;
}
inline Json preview(const Words& words, uint32_t limbs) {
  Json result = Json::array();
  for (std::size_t cell = 0; cell < std::min<std::size_t>(6, words.size() / limbs); ++cell)
    result.push_back(decode(words, cell, limbs).str());
  return result;
}
struct Dataset {
  std::vector<int64_t> a, b, t;
  double preparation_ns = 0;
  std::string recipe;
};
inline Dataset make_data(const Options& options, bool chain = false) {
  const auto start = Clock::now();
  Dataset data;
  data.a.resize(std::size_t(options.m) * options.k);
  data.b.resize(std::size_t(options.k) * options.n);
  if (chain) data.t.resize(std::size_t(options.n) * options.n);
  data.recipe = "splitmix64_signed_magnitude_v1";
  uint64_t state = options.seed;
  const uint64_t max = (uint64_t{1} << options.bits) - 1, span = max * 2 + 1;
  for (auto* operand : {&data.a, &data.b, &data.t}) {
    for (std::size_t index = 0; index < operand->size(); ++index) {
      const uint64_t word = next_word(state), value = word % span;
      (*operand)[index] = options.pattern == "zero" || (options.pattern == "sparse" && index % 8)
          ? 0 : value <= max ? static_cast<int64_t>(value) : -static_cast<int64_t>(value - max);
    }
  }
  if (options.size == "tiny" && options.m == 2 && options.n == 2 && options.k == 3 && options.pattern == "seeded"
      && !options.supplied.count("--seed") && !options.supplied.count("--bits")) {
    data.a = {2, -3, 4, -5, 6, 7}; data.b = {1, 8, -2, 9, 3, -4};
    if (chain) data.t = {2, 1, -1, 2};
    data.recipe = chain ? "tiny_cross_chain_v1" : "tiny_cross_v1";
  }
  if (options.gram) {
    for (uint32_t row = 0; row < options.m; ++row)
      for (uint32_t q = 0; q < options.k; ++q) data.a[std::size_t(row) * options.k + q] = data.b[std::size_t(q) * options.n + row];
    data.recipe += "_gram_transpose_v1";
  }
  data.preparation_ns = since(start);
  return data;
}
inline Words dataset_words(const Dataset& data) {
  Words values;
  values.reserve(data.a.size() + data.b.size() + data.t.size());
  for (const auto* operand : {&data.a, &data.b, &data.t}) for (auto value : *operand) values.push_back(bits_of(value));
  return values;
}
inline cpp_int maximum(const std::vector<int64_t>& values) {
  uint64_t max = 0; for (int64_t value : values) max = std::max(max, magnitude(value)); return cpp_int(max);
}
inline void memory_admission(const Options& options, bool chain = false) {
  const uint64_t cells = uint64_t(options.m) * options.k + uint64_t(options.k) * options.n +
      uint64_t(options.m) * options.n + (chain ? uint64_t(options.n) * options.n + uint64_t(options.m) * options.n * 2 : 0);
  // One aggregate host allowance covers simultaneously retained comparators,
  // inputs and old/new exact references. GPU matrix/workspace envelopes are
  // charged separately for every candidate before its allocations.
  options.memory->reserve(cells * 1024, options.memory_mib * 1024 * 1024, "host comparator/reference storage");
  require(options.size == "large" || uint64_t(options.m) * options.n * options.k <= 4000000,
          "larger reference work requires explicit --size large");
}
inline rns8_gemm_desc gemm_desc(const Options& options, rns8_semantics semantics,
                               uint32_t m, uint32_t n, uint32_t k, const cpp_int& bound, bool fixed = false) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc); desc.abi_version = RNS8_ABI_VERSION;
  desc.requested_backend = options.backend; desc.semantics = semantics;
  desc.m = m; desc.n = n; desc.k = k;
  const bool finite = semantics == RNS8_FINITE_FIELD_U8;
  desc.bound_kind = semantics == RNS8_BOUNDED_I64 ? RNS8_BOUND_GLOBAL_MAX_ABS : RNS8_BOUND_NONE;
  if (semantics == RNS8_BOUNDED_I64) {
    require(bound <= INT64_MAX, "bounded-i64 envelope is insufficient; select --wide or reduce inputs");
    desc.bound = bound.convert_to<uint64_t>();
  }
  desc.max_prefix = finite ? 0 : RNS8_MAX_SUPPORTED_PREFIX;
  desc.finite_modulus = finite ? 251 : 0;
  if (fixed) desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
  return desc;
}
inline rns8_matrix_desc matrix_desc(uint32_t rows, uint32_t cols, rns8_semantics semantics,
                                    rns8_bound_kind kind, uint32_t prefix) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc); desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows; desc.cols = cols; desc.logical_ld = cols;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR; desc.semantics = semantics;
  desc.bound_kind = kind; desc.max_prefix = prefix;
  return desc;
}
inline Json device_info(const rns8::Context& context, const Options& options) {
  const auto info = context.device_info();
  require(info.backend == options.backend && info.device_id == options.device,
          "runtime did not select requested backend/device");
  return {{"backend_requested", backend_name(options.backend)}, {"backend_selected", backend_name(info.backend)},
          {"device_id", info.device_id}, {"device", info.name},
          {"target", options.backend == RNS8_BACKEND_CPU_REFERENCE ? "host" : info.gcn_arch},
          {"hip_runtime_version", info.hip_runtime_version}, {"hip_driver_version", info.hip_driver_version},
          {"global_mem_bytes", info.global_mem_bytes}, {"gpu_execution", options.backend != RNS8_BACKEND_CPU_REFERENCE}};
}
inline Json plan_info(const rns8::Plan& plan) {
  const auto backend = plan.backend_info(); const auto schedule = plan.schedule_info();
  return {{"backend_selected", backend_name(backend.backend)}, {"selected_kernel", backend.selected_kernel},
          {"workspace_required_bytes", backend.workspace_required_bytes},
          {"selected_prefix", schedule.max_selected_prefix}, {"range_bits", schedule.max_selected_prefix ? rns8_prefix_range_bits(schedule.max_selected_prefix) : 0},
          {"accelerator_library", backend.accelerator_library}, {"accelerator_version", backend.accelerator_version}};
}
struct Phases {
  Json values = Json::object();
  bool enabled = true;
  template<class Function> void measure(const char* name, Function&& function) {
    if (!enabled) { function(); return; }
    const auto start = Clock::now(); function(); values[name] = values.value(name, 0.0) + since(start);
  }
};
struct Variant {
  std::string name;
  Json metadata = Json::object();
  double setup_ns = 0;
  std::function<void(Phases&, uint64_t)> execute;
  std::function<const Words&()> output;
  // Optional lifecycle evidence is read after timing, never between burst calls.
  std::function<Json()> execution_evidence;
};
inline Json clock_info() {
  std::array<double, 1001> times{};
  for (auto& value : times) { const auto start = Clock::now(); value = since(start); }
  std::sort(times.begin(), times.end());
  const double tick = 1e9 * double(Clock::period::num) / double(Clock::period::den);
  return {{"source", "host_monotonic"}, {"period_numerator", Clock::period::num}, {"period_denominator", Clock::period::den},
          {"resolution_ns", tick}, {"pair_median_ns", times[500]}, {"pair_p95_ns", times[950]},
          {"censor_threshold_ns", std::max(tick, 5 * times[950])}, {"calibration_samples", times.size()},
          {"device_event_ns", nullptr}, {"device_event_reason", "public synchronous API exposes no event hook; measured host calls include required synchronization"}};
}
inline Json base_report(const Options& options, const std::string& workload) {
  Json manifest = nullptr, source = nullptr;
#ifdef RNS8_BUILD_MANIFEST_JSON
  manifest = Json::parse(RNS8_BUILD_MANIFEST_JSON);
#elif defined(RNS8_WORKLOAD_LIBRARY_MANIFEST_JSON)
  manifest = Json::parse(RNS8_WORKLOAD_LIBRARY_MANIFEST_JSON);
#endif
#ifdef RNS8_WORKLOAD_SOURCE_IDENTITY_JSON
  source = Json::parse(RNS8_WORKLOAD_SOURCE_IDENTITY_JSON);
#endif
#ifdef _MSC_FULL_VER
  const std::string compiler = "MSVC " + std::to_string(_MSC_FULL_VER);
#elif defined(__VERSION__)
  const std::string compiler = __VERSION__;
#else
  const std::string compiler = "unavailable";
#endif
#ifdef _WIN32
  const char* os = "windows";
#elif defined(__linux__)
  const char* os = "linux";
#else
  const char* os = "other";
#endif
  return {{"schema", "rns8_workload_observation_v1"}, {"status", "passed"}, {"workload", workload}, {"run_id", options.run_id},
          {"backend_requested", backend_name(options.backend)}, {"device_requested", options.device},
          {"m", options.m}, {"n", options.n}, {"k", options.k}, {"bits", options.bits}, {"seed", options.seed},
          {"logical_layout", "row_major"}, {"a_ld", options.k}, {"b_ld", options.n}, {"output_ld", options.n},
          {"pattern", options.pattern}, {"repeats", options.repeats}, {"warmups", options.warmups}, {"size", options.size},
          {"sustained_ms", options.sustained_ms},
          {"declared_memory_budget_bytes", options.memory_mib * 1024 * 1024},
          {"os", os}, {"compiler", compiler}, {"cpu_comparator_threads", 1}, {"host_logical_threads", std::thread::hardware_concurrency()},
          {"build_manifest", manifest}, {"workload_source_identity", source}, {"clock", clock_info()},
          {"independent_process_runs", 1}, {"isa_verified", false}, {"promotion_eligible", false},
          {"executable_sha256", nullptr}, {"artifact_identity_note", "collector binds executable hash before/after process; direct output alone is not compiled-library attestation"},
          {"timing_scope", "host synchronous calls; common dataset/layout preparation charged once to each cold application total; verification and report I/O excluded"},
          {"unavailable_phases", {{"isolated_upload_ns", nullptr}, {"isolated_download_ns", nullptr}, {"kernel_only_ns", nullptr}}},
          {"unavailable_phase_reason", "public API combines pack/upload and reconstruction/download; no invented separation"},
          {"optional_references", {{"gmp", "not_run: no same-workload GMP adapter"}, {"flint", "not_run: no same-workload FLINT adapter"}}}};
}
inline void print_report(const Options& options, const Json& report) {
  if (options.json) { std::cout << report.dump(2) << '\n'; return; }
  std::cout << report.at("task").get<std::string>() << '\n';
  std::cout << "status: " << report.at("status") << "; full output comparison, not checksum-only verification\n";
  if (report.contains("output_preview_decimal")) std::cout << "sample exact output: " << report["output_preview_decimal"].dump() << '\n';
  if (report.contains("checksum_hex")) std::cout << "all-output checksum: " << report["checksum_hex"] << '\n';
  if (report.contains("variants")) for (const auto& variant : report["variants"]) {
    std::cout << variant.at("name").get<std::string>() << ": " << variant.at("status").get<std::string>();
    if (variant.contains("median_steady_ns")) std::cout << ", median " << variant["median_steady_ns"].get<double>() / 1000 << " us";
    if (variant.contains("reason")) std::cout << ", " << variant["reason"].get<std::string>();
    std::cout << '\n';
  }
  if (report.contains("witnesses")) for (const auto& witness : report["witnesses"]) std::cout << witness.dump(2) << '\n';
  std::cout << "Use --json for raw timings, phase scopes, bounds and provenance. One process does not establish a performance winner.\n";
}
template<class Function> int program(int argc, char** argv, Function&& function) {
  bool json = false; for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--json") json = true;
  try { const auto opts = options(argc, argv); function(opts); return 0; }
  catch (const rns8::Error& error) {
    const bool unsupported = error.status() == RNS8_UNSUPPORTED_BACKEND || error.status() == RNS8_UNSUPPORTED_ARCH
        || error.status() == RNS8_UNSUPPORTED_OS;
    if (json) std::cout << Json{{"schema", "rns8_workload_observation_v1"}, {"status", unsupported ? "unsupported" : "failed"},
                              {"reason", error.what()}, {"promotion_eligible", false}}.dump() << '\n';
    else std::cerr << error.what() << '\n';
    return unsupported ? 77 : 1;
  } catch (const std::exception& error) {
    if (json) std::cout << Json{{"schema", "rns8_workload_observation_v1"}, {"status", "failed"},
                              {"reason", error.what()}, {"promotion_eligible", false}}.dump() << '\n';
    else std::cerr << error.what() << '\n';
    return 1;
  }
}
} // namespace rns8_workload
#endif
