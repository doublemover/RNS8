#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "rns8/rns8.h"

#ifndef RNS8_CONFIGURED_AMDGPU_TARGETS
#  define RNS8_CONFIGURED_AMDGPU_TARGETS "not-configured"
#endif

#ifndef RNS8_GIT_COMMIT
#  define RNS8_GIT_COMMIT "unknown"
#endif

#ifndef RNS8_SOURCE_DIR
#  define RNS8_SOURCE_DIR "."
#endif

namespace {

constexpr uint32_t kBenchmarkSchemaVersion = 2;

enum class BenchSemantics {
  BoundedI64,
  BoundedU64,
};

struct Args {
  int64_t m = 64;
  int64_t n = 64;
  int64_t k = 64;
  uint32_t warmups = 1;
  uint32_t repeats = 5;
  uint64_t seed = 1;
  int device_id = std::numeric_limits<int>::min();
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;
  BenchSemantics semantics = BenchSemantics::BoundedI64;
};

struct TimingSamples {
  std::vector<uint64_t> pack_us;
  std::vector<uint64_t> gemm_us;
  std::vector<uint64_t> export_us;
  std::vector<uint64_t> end_to_end_us;
};

struct GpuEventSamples {
  bool requested = false;
  bool complete = true;
  std::vector<std::string> unavailable_reasons;
  std::vector<double> pack_h2d_us;
  std::vector<double> pack_kernel_us;
  std::vector<double> pack_us;
  std::vector<double> rns_gemm_kernel_group_us;
  std::vector<double> rns_gemm_us;
  std::vector<double> crt_export_status_memset_us;
  std::vector<double> crt_export_kernel_us;
  std::vector<double> crt_export_status_d2h_us;
  std::vector<double> crt_export_d2h_us;
  std::vector<double> crt_export_us;
};

struct BenchmarkResult {
  uint64_t plan_us = 0;
  uint64_t matrix_alloc_us = 0;
  TimingSamples samples{};
  GpuEventSamples gpu_events{};
  uint64_t checksum = 0;
};

uint64_t elapsed_us(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

[[noreturn]] void usage_error(const std::string& message) {
  std::cerr << message << "\n";
  std::cerr
      << "usage: rns8-bench [--backend cpu|hip-direct] [--semantics bounded-i64|bounded-u64]\n"
      << "                  [--device N] [--m M] [--n N] [--k K]\n"
      << "                  [--warmups W] [--repeats R] [--seed S]\n";
  std::exit(2);
}

int64_t parse_i64(const char* text, const char* label) {
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (!end || *end != '\0') {
    usage_error(std::string("invalid integer for ") + label + ": " + text);
  }
  return static_cast<int64_t>(value);
}

uint32_t parse_u32(const char* text, const char* label) {
  const int64_t value = parse_i64(text, label);
  if (value < 0 || value > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    usage_error(std::string("out-of-range integer for ") + label + ": " + text);
  }
  return static_cast<uint32_t>(value);
}

uint64_t parse_u64_seed(const char* text) {
  const int64_t value = parse_i64(text, "--seed");
  if (value < 0) {
    usage_error("seed must be non-negative");
  }
  return static_cast<uint64_t>(value);
}

rns8_backend_kind parse_backend(const std::string& value) {
  if (value == "cpu" || value == "cpu-reference") return RNS8_BACKEND_CPU_REFERENCE;
  if (value == "hip-direct") return RNS8_BACKEND_HIP_DIRECT;
  usage_error("unknown backend: " + value);
}

BenchSemantics parse_semantics(const std::string& value) {
  if (value == "bounded-i64") return BenchSemantics::BoundedI64;
  if (value == "bounded-u64") return BenchSemantics::BoundedU64;
  usage_error("unknown semantics: " + value);
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--m" && i + 1 < argc) {
      args.m = parse_i64(argv[++i], "--m");
    } else if (arg == "--n" && i + 1 < argc) {
      args.n = parse_i64(argv[++i], "--n");
    } else if (arg == "--k" && i + 1 < argc) {
      args.k = parse_i64(argv[++i], "--k");
    } else if (arg == "--warmups" && i + 1 < argc) {
      args.warmups = parse_u32(argv[++i], "--warmups");
    } else if (arg == "--repeats" && i + 1 < argc) {
      args.repeats = parse_u32(argv[++i], "--repeats");
    } else if (arg == "--seed" && i + 1 < argc) {
      args.seed = parse_u64_seed(argv[++i]);
    } else if (arg == "--device" && i + 1 < argc) {
      args.device_id = static_cast<int>(parse_i64(argv[++i], "--device"));
    } else if (arg == "--backend" && i + 1 < argc) {
      args.backend = parse_backend(argv[++i]);
    } else if (arg == "--semantics" && i + 1 < argc) {
      args.semantics = parse_semantics(argv[++i]);
    } else if (arg == "--help") {
      std::cout
          << "usage: rns8-bench [--backend cpu|hip-direct] [--semantics bounded-i64|bounded-u64]\n"
          << "                  [--device N] [--m M] [--n N] [--k K]\n"
          << "                  [--warmups W] [--repeats R] [--seed S]\n";
      std::exit(0);
    } else {
      usage_error("unknown or incomplete argument: " + arg);
    }
  }

  if (args.m <= 0 || args.n <= 0 || args.k <= 0 || args.repeats == 0) {
    usage_error("matrix dimensions must be positive and repeats must be nonzero");
  }
  if (args.device_id == std::numeric_limits<int>::min()) {
    args.device_id = args.backend == RNS8_BACKEND_HIP_DIRECT ? 0 : -1;
  }
  return args;
}

const char* backend_name(rns8_backend_kind backend) {
  switch (backend) {
    case RNS8_BACKEND_AUTO:
      return "auto";
    case RNS8_BACKEND_CPU_REFERENCE:
      return "cpu-reference";
    case RNS8_BACKEND_HIP_DIRECT:
      return "hip-direct";
    case RNS8_BACKEND_HIPBLASLT:
      return "hipblaslt";
    case RNS8_BACKEND_CK:
      return "ck";
    case RNS8_BACKEND_WMMA:
      return "wmma";
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
      return "wrap64-byte-limb";
  }
  return "unknown";
}

const char* semantics_name(BenchSemantics semantics) {
  return semantics == BenchSemantics::BoundedI64 ? "bounded_i64" : "bounded_u64";
}

rns8_semantics c_semantics(BenchSemantics semantics) {
  return semantics == BenchSemantics::BoundedI64 ? RNS8_BOUNDED_I64 : RNS8_BOUNDED_U64;
}

rns8_bound_kind bound_kind(BenchSemantics semantics) {
  return semantics == BenchSemantics::BoundedI64 ? RNS8_BOUND_GLOBAL_MAX_ABS : RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
}

const char* bound_kind_name(BenchSemantics semantics) {
  return semantics == BenchSemantics::BoundedI64 ? "global_max_abs" : "global_max_unsigned";
}

std::string json_escape(const std::string& input) {
  std::ostringstream out;
  for (const char ch : input) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

std::string command_line(int argc, char** argv) {
  std::ostringstream out;
  for (int i = 0; i < argc; ++i) {
    if (i > 0) {
      out << ' ';
    }
    out << argv[i];
  }
  return out.str();
}

std::string trim_ascii_whitespace(std::string value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' ' || value.back() == '\t')) {
    value.pop_back();
  }
  std::size_t first = 0;
  while (first < value.size() && (value[first] == '\n' || value[first] == '\r' || value[first] == ' ' || value[first] == '\t')) {
    ++first;
  }
  if (first > 0) {
    value.erase(0, first);
  }
  return value;
}

std::string runtime_git_commit() {
#if defined(_WIN32)
  const std::string command =
      std::string("git -C \"") + RNS8_SOURCE_DIR + "\" rev-parse --short=12 HEAD 2>NUL";
  FILE* pipe = _popen(command.c_str(), "r");
#else
  const std::string command =
      std::string("git -C \"") + RNS8_SOURCE_DIR + "\" rev-parse --short=12 HEAD 2>/dev/null";
  FILE* pipe = popen(command.c_str(), "r");
#endif
  if (!pipe) {
    return RNS8_GIT_COMMIT;
  }
  std::array<char, 64> buffer{};
  std::string output;
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) {
    output += buffer.data();
  }
#if defined(_WIN32)
  const int close_status = _pclose(pipe);
#else
  const int close_status = pclose(pipe);
#endif
  output = trim_ascii_whitespace(output);
  if (close_status != 0 || output.empty()) {
    return RNS8_GIT_COMMIT;
  }
  return output;
}

std::string compiler_id() {
#if defined(_MSC_VER)
  return "msvc";
#elif defined(__clang__)
  return "clang";
#elif defined(__GNUC__)
  return "gcc";
#else
  return "unknown";
#endif
}

std::string compiler_version() {
#if defined(_MSC_VER)
  return std::to_string(_MSC_VER) + "." + std::to_string(_MSC_FULL_VER);
#elif defined(__clang__)
  return std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." +
         std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
  return std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
#else
  return "unknown";
#endif
}

std::size_t checked_elements(int64_t rows, int64_t cols, const char* label) {
  const auto u_rows = static_cast<uint64_t>(rows);
  const auto u_cols = static_cast<uint64_t>(cols);
  if (u_cols != 0 && u_rows > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) / u_cols) {
    usage_error(std::string("matrix size overflows size_t for ") + label);
  }
  return static_cast<std::size_t>(u_rows * u_cols);
}

uint64_t benchmark_bound(const Args& args) {
  const uint64_t max_term = 16u * 16u;
  const auto u_k = static_cast<uint64_t>(args.k);
  if (u_k > std::numeric_limits<uint64_t>::max() / max_term) {
    usage_error("k is too large for the benchmark bound");
  }
  const uint64_t bound = u_k * max_term;
  if (args.semantics == BenchSemantics::BoundedI64 &&
      bound > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    usage_error("bounded-i64 benchmark bound exceeds int64 output range");
  }
  return bound;
}

rns8_matrix_desc matrix_desc(int64_t rows, int64_t cols, BenchSemantics semantics) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = c_semantics(semantics);
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind(semantics);
  desc.tile_m = 128;
  desc.tile_n = 128;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

rns8_gemm_desc gemm_desc(const Args& args, uint64_t bound) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = c_semantics(args.semantics);
  desc.bound_kind = bound_kind(args.semantics);
  desc.requested_backend = args.backend;
  desc.m = args.m;
  desc.n = args.n;
  desc.k = args.k;
  desc.bound = bound;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  desc.tile_m = 128;
  desc.tile_n = 128;
  return desc;
}

void fail_status(const char* label, rns8_status status) {
  std::cerr << label << ": " << rns8_status_string(status) << "\n";
  std::exit(1);
}

void mix_checksum(uint64_t& checksum, uint64_t value) {
  checksum ^= value;
  checksum *= 1099511628211ull;
}

double average(const std::vector<uint64_t>& values) {
  if (values.empty()) {
    return 0.0;
  }
  uint64_t sum = 0;
  for (const uint64_t value : values) {
    sum += value;
  }
  return static_cast<double>(sum) / static_cast<double>(values.size());
}

double percentile(std::vector<uint64_t> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double scaled = p * static_cast<double>(values.size() - 1);
  const auto index = static_cast<std::size_t>(scaled + 0.999999);
  return static_cast<double>(values[index]);
}

double average_double(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const double value : values) {
    sum += value;
  }
  return sum / static_cast<double>(values.size());
}

double percentile_double(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double scaled = p * static_cast<double>(values.size() - 1);
  const auto index = static_cast<std::size_t>(scaled + 0.999999);
  return values[index];
}

void print_u64_array(const std::vector<uint64_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void print_double_array(const std::vector<double>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void print_string_array(const std::vector<std::string>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << "\"" << json_escape(values[i]) << "\"";
  }
  std::cout << "]";
}

void print_single_u64_array(uint64_t value) {
  std::cout << "[" << value << "]";
}

void print_timing_summary(const char* name, const std::vector<uint64_t>& values, bool trailing_comma) {
  std::cout << "    \"" << name << "\": {\n";
  std::cout << "      \"avg\": " << average(values) << ",\n";
  std::cout << "      \"median\": " << percentile(values, 0.50) << ",\n";
  std::cout << "      \"p95\": " << percentile(values, 0.95) << "\n";
  std::cout << "    }" << (trailing_comma ? "," : "") << "\n";
}

void print_single_timing_summary(const char* name, uint64_t value, bool trailing_comma) {
  std::cout << "    \"" << name << "\": {\n";
  std::cout << "      \"avg\": " << static_cast<double>(value) << ",\n";
  std::cout << "      \"median\": " << static_cast<double>(value) << ",\n";
  std::cout << "      \"p95\": " << static_cast<double>(value) << "\n";
  std::cout << "    }" << (trailing_comma ? "," : "") << "\n";
}

void print_gpu_event_summary(const char* name, const std::vector<double>& values, bool trailing_comma) {
  std::cout << "    \"" << name << "\": {\n";
  std::cout << "      \"avg\": " << average_double(values) << ",\n";
  std::cout << "      \"median\": " << percentile_double(values, 0.50) << ",\n";
  std::cout << "      \"p95\": " << percentile_double(values, 0.95) << "\n";
  std::cout << "    }" << (trailing_comma ? "," : "") << "\n";
}

void print_named_gpu_event_array(const char* name, const std::vector<double>& values, bool trailing_comma) {
  std::cout << "    \"" << name << "\": ";
  print_double_array(values);
  std::cout << (trailing_comma ? "," : "") << "\n";
}

void print_gpu_event_timings(const GpuEventSamples& events) {
  std::cout << "  \"gpu_event_timings_us\": {\n";
  print_named_gpu_event_array("pack_h2d", events.pack_h2d_us, true);
  print_named_gpu_event_array("pack_kernel", events.pack_kernel_us, true);
  print_named_gpu_event_array("pack", events.pack_us, true);
  print_named_gpu_event_array("rns_gemm_kernel_group", events.rns_gemm_kernel_group_us, true);
  print_named_gpu_event_array("rns_gemm", events.rns_gemm_us, true);
  print_named_gpu_event_array("crt_export_status_memset", events.crt_export_status_memset_us, true);
  print_named_gpu_event_array("crt_export_kernel", events.crt_export_kernel_us, true);
  print_named_gpu_event_array("crt_export_status_d2h", events.crt_export_status_d2h_us, true);
  print_named_gpu_event_array("crt_export_d2h", events.crt_export_d2h_us, true);
  print_named_gpu_event_array("crt_export", events.crt_export_us, false);
  std::cout << "  },\n";
}

void print_gpu_event_timing_summary(const GpuEventSamples& events) {
  std::cout << "  \"gpu_event_timing_summary_us\": {\n";
  print_gpu_event_summary("pack_h2d", events.pack_h2d_us, true);
  print_gpu_event_summary("pack_kernel", events.pack_kernel_us, true);
  print_gpu_event_summary("pack", events.pack_us, true);
  print_gpu_event_summary("rns_gemm_kernel_group", events.rns_gemm_kernel_group_us, true);
  print_gpu_event_summary("rns_gemm", events.rns_gemm_us, true);
  print_gpu_event_summary("crt_export_status_memset", events.crt_export_status_memset_us, true);
  print_gpu_event_summary("crt_export_kernel", events.crt_export_kernel_us, true);
  print_gpu_event_summary("crt_export_status_d2h", events.crt_export_status_d2h_us, true);
  print_gpu_event_summary("crt_export_d2h", events.crt_export_d2h_us, true);
  print_gpu_event_summary("crt_export", events.crt_export_us, false);
  std::cout << "  },\n";
}

uint64_t checksum_i64(const std::vector<int64_t>& values) {
  uint64_t checksum = 1469598103934665603ull;
  for (const int64_t value : values) {
    mix_checksum(checksum, static_cast<uint64_t>(value));
  }
  return checksum;
}

uint64_t checksum_u64(const std::vector<uint64_t>& values) {
  uint64_t checksum = 1469598103934665603ull;
  for (const uint64_t value : values) {
    mix_checksum(checksum, value);
  }
  return checksum;
}

bool gpu_event_capture_requested(const Args& args) {
  return args.backend == RNS8_BACKEND_HIP_DIRECT;
}

void add_unavailable_reason(GpuEventSamples& events, const std::string& reason) {
  events.complete = false;
  if (std::find(events.unavailable_reasons.begin(), events.unavailable_reasons.end(), reason) ==
      events.unavailable_reasons.end()) {
    events.unavailable_reasons.push_back(reason);
  }
}

double sum_event_label(
    GpuEventSamples& events,
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    const char* phase,
    const char* label) {
  bool found = false;
  double total = 0.0;
  for (const auto& sample : samples) {
    if (sample.label == label) {
      found = true;
      total += sample.microseconds;
    }
  }
  if (!found) {
    add_unavailable_reason(events, std::string(phase) + " missing backend HIP event label " + label);
  }
  return total;
}

void collect_pack_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double h2d = sum_event_label(events, samples, "pack", "pack_h2d");
  const double kernel = sum_event_label(events, samples, "pack", "pack_kernel");
  if (events.complete) {
    events.pack_h2d_us.push_back(h2d);
    events.pack_kernel_us.push_back(kernel);
    events.pack_us.push_back(h2d + kernel);
  }
}

void collect_gemm_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double kernel_group = sum_event_label(events, samples, "rns_gemm", "rns_gemm_kernel_group");
  if (events.complete) {
    events.rns_gemm_kernel_group_us.push_back(kernel_group);
    events.rns_gemm_us.push_back(kernel_group);
  }
}

void collect_export_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double status_memset = sum_event_label(events, samples, "crt_export", "crt_export_status_memset");
  const double kernel = sum_event_label(events, samples, "crt_export", "crt_export_kernel");
  const double status_d2h = sum_event_label(events, samples, "crt_export", "crt_export_status_d2h");
  const double d2h = sum_event_label(events, samples, "crt_export", "crt_export_d2h");
  if (events.complete) {
    events.crt_export_status_memset_us.push_back(status_memset);
    events.crt_export_kernel_us.push_back(kernel);
    events.crt_export_status_d2h_us.push_back(status_d2h);
    events.crt_export_d2h_us.push_back(d2h);
    events.crt_export_us.push_back(status_memset + kernel + status_d2h + d2h);
  }
}

void begin_gpu_event_phase(bool collect) {
  if (!collect) {
    return;
  }
  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
}

void end_gpu_event_phase(bool collect) {
  if (!collect) {
    return;
  }
  rns8::detail::hip_direct_timing_set_enabled(false);
}

bool gpu_event_timing_available(const BenchmarkResult& result) {
  if (!result.gpu_events.requested || !result.gpu_events.complete) {
    return false;
  }
  const std::size_t repeats = result.samples.pack_us.size();
  return repeats > 0 && result.gpu_events.pack_us.size() == repeats &&
         result.gpu_events.rns_gemm_us.size() == repeats && result.gpu_events.crt_export_us.size() == repeats;
}

BenchmarkResult run_bounded_i64(rns8_context* ctx, const Args& args, uint64_t bound) {
  std::mt19937_64 rng(args.seed);
  std::uniform_int_distribution<int64_t> dist(-16, 16);
  std::vector<int64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<int64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<int64_t> C(checked_elements(args.m, args.n, "C"));
  for (auto& value : A) value = dist(rng);
  for (auto& value : B) value = dist(rng);

  BenchmarkResult result{};
  result.gpu_events.requested = gpu_event_capture_requested(args);
  auto desc = gemm_desc(args, bound);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  const auto alloc_start = std::chrono::steady_clock::now();
  auto a_desc = matrix_desc(args.m, args.k, args.semantics);
  auto b_desc = matrix_desc(args.k, args.n, args.semantics);
  auto c_desc = matrix_desc(args.m, args.n, args.semantics);
  status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_pack_i64(ctx, a_matrix, A.data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(A)", status);
    status = rns8_pack_i64(ctx, b_matrix, B.data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(B)", status);
    if (collect_gpu_events) {
      collect_pack_gpu_events(result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto pack_end = std::chrono::steady_clock::now();

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns", status);
    if (collect_gpu_events) {
      collect_gemm_gpu_events(result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_export_i64(ctx, plan, c_matrix, C.data(), args.n);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_i64", status);
    if (collect_gpu_events) {
      collect_export_gpu_events(result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto export_end = std::chrono::steady_clock::now();

    if (samples) {
      samples->pack_us.push_back(elapsed_us(pack_start, pack_end));
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_iteration(static_cast<uint64_t>(r) + 1, nullptr);
  }
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
  }
  result.checksum = checksum_i64(C);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_bounded_u64(rns8_context* ctx, const Args& args, uint64_t bound) {
  std::mt19937_64 rng(args.seed);
  std::uniform_int_distribution<uint64_t> dist(0, 16);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<uint64_t> C(checked_elements(args.m, args.n, "C"));
  for (auto& value : A) value = dist(rng);
  for (auto& value : B) value = dist(rng);

  BenchmarkResult result{};
  result.gpu_events.requested = gpu_event_capture_requested(args);
  auto desc = gemm_desc(args, bound);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  const auto alloc_start = std::chrono::steady_clock::now();
  auto a_desc = matrix_desc(args.m, args.k, args.semantics);
  auto b_desc = matrix_desc(args.k, args.n, args.semantics);
  auto c_desc = matrix_desc(args.m, args.n, args.semantics);
  status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_pack_u64(ctx, a_matrix, A.data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(A)", status);
    status = rns8_pack_u64(ctx, b_matrix, B.data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(B)", status);
    if (collect_gpu_events) {
      collect_pack_gpu_events(result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto pack_end = std::chrono::steady_clock::now();

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns", status);
    if (collect_gpu_events) {
      collect_gemm_gpu_events(result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_export_u64(ctx, plan, c_matrix, C.data(), args.n);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_u64", status);
    if (collect_gpu_events) {
      collect_export_gpu_events(result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto export_end = std::chrono::steady_clock::now();

    if (samples) {
      samples->pack_us.push_back(elapsed_us(pack_start, pack_end));
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_iteration(static_cast<uint64_t>(r) + 1, nullptr);
  }
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
  }
  result.checksum = checksum_u64(C);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}

void print_json(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result,
    uint64_t bound,
    const std::string& cmdline) {
  const double avg_pack_us = average(result.samples.pack_us);
  const double avg_gemm_us = average(result.samples.gemm_us);
  const double avg_export_us = average(result.samples.export_us);
  const double avg_end_to_end_us = average(result.samples.end_to_end_us);
  const double avg_per_modulus_gemm_estimate_us = avg_gemm_us / static_cast<double>(RNS8_DEFAULT_BOUNDED_PREFIX);
  const bool gpu_events_available = gpu_event_timing_available(result);
  const char* gpu_event_reason = gpu_events_available
                                     ? "captured_by_direct_hip_backend_hooks"
                                     : (result.gpu_events.requested ? "backend_event_capture_incomplete"
                                                                    : "backend_not_hip_direct");
  const char* gpu_event_status = gpu_events_available
                                     ? "available"
                                     : (result.gpu_events.requested ? "unavailable_missing_expected_events"
                                                                    : "not_requested_for_selected_backend");

  std::cout << "{\n";
  std::cout << "  \"schema_version\": " << kBenchmarkSchemaVersion << ",\n";
  std::cout << "  \"benchmark\": \"rns8_bounded_gemm_persistent_rns\",\n";
  std::cout << "  \"backend_requested\": \"" << backend_name(args.backend) << "\",\n";
  std::cout << "  \"backend_selected\": \"" << backend_name(info.backend) << "\",\n";
  std::cout << "  \"selected_kernel\": null,\n";
  std::cout << "  \"semantics\": \"" << semantics_name(args.semantics) << "\",\n";
  std::cout << "  \"bound_kind\": \"" << bound_kind_name(args.semantics) << "\",\n";
  std::cout << "  \"bound\": " << bound << ",\n";
  std::cout << "  \"m\": " << args.m << ",\n";
  std::cout << "  \"n\": " << args.n << ",\n";
  std::cout << "  \"k\": " << args.k << ",\n";
  std::cout << "  \"prefix\": " << RNS8_DEFAULT_BOUNDED_PREFIX << ",\n";
  std::cout << "  \"tile_m\": 128,\n";
  std::cout << "  \"tile_n\": 128,\n";
  std::cout << "  \"layout\": \"row_major\",\n";
  std::cout << "  \"k_block_size\": " << std::min<int64_t>(args.k, RNS8_SAFE_INT32_K_BLOCK) << ",\n";
  std::cout << "  \"adaptive_tile_size\": null,\n";
  std::cout << "  \"epilogue_type\": \"crt_export\",\n";
  std::cout << "  \"packed_layout_version\": null,\n";
  std::cout << "  \"seed\": " << args.seed << ",\n";
  std::cout << "  \"warmups\": " << args.warmups << ",\n";
  std::cout << "  \"repeats\": " << args.repeats << ",\n";
  std::cout << "  \"input_distribution\": \""
            << (args.semantics == BenchSemantics::BoundedI64 ? "signed_uniform_-16_16"
                                                             : "unsigned_uniform_0_16")
            << "\",\n";
  std::cout << "  \"command_line\": \"" << json_escape(cmdline) << "\",\n";
  std::cout << "  \"git_commit\": \"" << json_escape(runtime_git_commit()) << "\",\n";
  std::cout << "  \"compiler\": {\n";
  std::cout << "    \"id\": \"" << compiler_id() << "\",\n";
  std::cout << "    \"version\": \"" << compiler_version() << "\"\n";
  std::cout << "  },\n";
  std::cout << "  \"configured_amdgpu_targets\": \"" << json_escape(RNS8_CONFIGURED_AMDGPU_TARGETS) << "\",\n";
  std::cout << "  \"device\": {\n";
  std::cout << "    \"device_id\": " << info.device_id << ",\n";
  std::cout << "    \"name\": \"" << json_escape(info.name) << "\",\n";
  std::cout << "    \"gcn_arch\": \"" << json_escape(info.gcn_arch) << "\",\n";
  std::cout << "    \"hip_available\": " << info.hip_available << ",\n";
  std::cout << "    \"hip_runtime_version\": " << info.hip_runtime_version << ",\n";
  std::cout << "    \"hip_driver_version\": " << info.hip_driver_version << ",\n";
  std::cout << "    \"global_mem_bytes\": " << info.global_mem_bytes << "\n";
  std::cout << "  },\n";
  std::cout << "  \"clock_power_settings\": null,\n";
  std::cout << "  \"comparison_baseline\": null,\n";
  std::cout << "  \"derived_tops_equivalent\": null,\n";
  std::cout << "  \"timing_source\": \"std::chrono::steady_clock\",\n";
  std::cout << "  \"timing_note\": \"host wall-clock timings; direct-HIP calls include current backend "
               "synchronization, first-use persistent buffer allocation, copies, kernel launches, fused reduction, "
               "and GPU bounded export when using HIP\",\n";
  std::cout << "  \"timing_metadata\": {\n";
  std::cout << "    \"unit\": \"microseconds\",\n";
  std::cout << "    \"source\": \"std::chrono::steady_clock\",\n";
  std::cout << "    \"source_scope\": \"host_wall_clock\",\n";
  std::cout << "    \"gpu_event_timing\": " << (gpu_events_available ? "true" : "false") << ",\n";
  std::cout << "    \"gpu_event_timing_reason\": \"" << gpu_event_reason << "\",\n";
  std::cout << "    \"gpu_event_timing_status\": \"" << gpu_event_status << "\",\n";
  std::cout << "    \"gpu_event_timing_source\": "
            << (gpu_events_available ? "\"hipEventElapsedTime\"" : "null") << ",\n";
  std::cout << "    \"gpu_event_timing_source_scope\": "
            << (gpu_events_available ? "\"direct_hip_default_stream_backend_operation_groups\"" : "null") << ",\n";
  std::cout << "    \"gpu_event_timing_caveat\": "
            << (gpu_events_available
                    ? "\"HIP event timings record backend default-stream operation groups only; host wall-clock timings remain required for CPU scheduling overhead, API dispatch, allocations, and any synchronous host-side copy overhead not represented on the HIP stream\""
                    : "null")
            << ",\n";
  if (!gpu_events_available && !result.gpu_events.unavailable_reasons.empty()) {
    std::cout << "    \"gpu_event_timing_unavailable_reasons\": ";
    print_string_array(result.gpu_events.unavailable_reasons);
    std::cout << ",\n";
  }
  std::cout << "    \"phase_order\": [\"planning\", \"matrix_alloc\", \"pack\", \"rns_gemm\", \"crt_export\", \"end_to_end\"],\n";
  std::cout << "    \"gpu_event_phase_order\": [\"pack_h2d\", \"pack_kernel\", \"rns_gemm_kernel_group\", \"crt_export_status_memset\", \"crt_export_kernel\", \"crt_export_status_d2h\", \"crt_export_d2h\"],\n";
  std::cout << "    \"phase_notes\": {\n";
  std::cout << "      \"planning\": \"one-time rns8_create_plan plus rns8_create_workspace host timing\",\n";
  std::cout << "      \"matrix_alloc\": \"one-time persistent matrix allocation host timing\",\n";
  std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into persistent RNS matrices\",\n";
  std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_rns\",\n";
  std::cout << "      \"crt_export\": \"per-repeat host timing for export/reconstruction into logical output\",\n";
  std::cout << "      \"end_to_end\": \"per-repeat pack plus rns_gemm plus crt_export host timing\"\n";
  std::cout << "    }\n";
  std::cout << "  },\n";
  if (gpu_events_available) {
    print_gpu_event_timings(result.gpu_events);
    print_gpu_event_timing_summary(result.gpu_events);
  } else {
    std::cout << "  \"gpu_event_timings_us\": null,\n";
    std::cout << "  \"gpu_event_timing_summary_us\": null,\n";
  }
  std::cout << "  \"plan_us\": " << result.plan_us << ",\n";
  std::cout << "  \"avg_planning_us\": " << static_cast<double>(result.plan_us) << ",\n";
  std::cout << "  \"matrix_alloc_us\": " << result.matrix_alloc_us << ",\n";
  std::cout << "  \"avg_matrix_alloc_us\": " << static_cast<double>(result.matrix_alloc_us) << ",\n";
  std::cout << "  \"avg_pack_us\": " << avg_pack_us << ",\n";
  std::cout << "  \"avg_rns_gemm_us\": " << avg_gemm_us << ",\n";
  std::cout << "  \"avg_per_modulus_gemm_estimate_us\": " << avg_per_modulus_gemm_estimate_us << ",\n";
  std::cout << "  \"avg_crt_export_us\": " << avg_export_us << ",\n";
  std::cout << "  \"avg_end_to_end_us\": " << avg_end_to_end_us << ",\n";
  std::cout << "  \"raw_timings_us\": {\n";
  std::cout << "    \"planning\": ";
  print_single_u64_array(result.plan_us);
  std::cout << ",\n";
  std::cout << "    \"matrix_alloc\": ";
  print_single_u64_array(result.matrix_alloc_us);
  std::cout << ",\n";
  std::cout << "    \"pack\": ";
  print_u64_array(result.samples.pack_us);
  std::cout << ",\n";
  std::cout << "    \"rns_gemm\": ";
  print_u64_array(result.samples.gemm_us);
  std::cout << ",\n";
  std::cout << "    \"crt_export\": ";
  print_u64_array(result.samples.export_us);
  std::cout << ",\n";
  std::cout << "    \"end_to_end\": ";
  print_u64_array(result.samples.end_to_end_us);
  std::cout << "\n";
  std::cout << "  },\n";
  std::cout << "  \"timing_summary_us\": {\n";
  print_single_timing_summary("planning", result.plan_us, true);
  print_single_timing_summary("matrix_alloc", result.matrix_alloc_us, true);
  print_timing_summary("pack", result.samples.pack_us, true);
  print_timing_summary("rns_gemm", result.samples.gemm_us, true);
  print_timing_summary("crt_export", result.samples.export_us, true);
  print_timing_summary("end_to_end", result.samples.end_to_end_us, false);
  std::cout << "  },\n";
  std::cout << "  \"checksum_u64\": " << result.checksum << "\n";
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  const uint64_t bound = benchmark_bound(args);
  const std::string cmdline = command_line(argc, argv);

  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = args.backend;
  rns8_context* ctx = nullptr;
  rns8_status status = rns8_create_context(args.device_id, &options, &ctx);
  if (status != RNS8_SUCCESS) {
    std::cerr << "rns8_create_context: " << rns8_status_string(status) << "\n";
    return 1;
  }

  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  status = rns8_get_device_info(ctx, &info);
  if (status != RNS8_SUCCESS) {
    rns8_destroy_context(ctx);
    std::cerr << "rns8_get_device_info: " << rns8_status_string(status) << "\n";
    return 1;
  }

  const BenchmarkResult result = args.semantics == BenchSemantics::BoundedI64 ? run_bounded_i64(ctx, args, bound)
                                                                             : run_bounded_u64(ctx, args, bound);
  rns8_destroy_context(ctx);
  print_json(args, info, result, bound, cmdline);
  return 0;
}
