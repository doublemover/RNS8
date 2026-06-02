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
#include <utility>
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

constexpr uint32_t kBenchmarkSchemaVersion = 4;

enum class BenchSemantics {
  BoundedI64,
  BoundedU64,
  WrapU64Mod2_64,
};

enum class BoundMode {
  Global,
  PerTile,
};

struct Args {
  int64_t m = 64;
  int64_t n = 64;
  int64_t k = 64;
  uint32_t warmups = 1;
  uint32_t repeats = 5;
  uint64_t seed = 1;
  uint32_t tile_m = 128;
  uint32_t tile_n = 128;
  int device_id = std::numeric_limits<int>::min();
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;
  BenchSemantics semantics = BenchSemantics::BoundedI64;
  BoundMode bound_mode = BoundMode::Global;
  bool require_adaptive_execution = false;
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
  std::vector<double> wrap64_byte_gemm36_kernel_us;
  std::vector<double> rns_gemm_us;
  std::vector<double> crt_export_status_memset_us;
  std::vector<double> crt_export_kernel_us;
  std::vector<double> crt_export_status_d2h_us;
  std::vector<double> crt_export_d2h_us;
  std::vector<double> wrap64_export_kernel_us;
  std::vector<double> wrap64_export_d2h_us;
  std::vector<double> crt_export_us;
};

struct BenchmarkResult {
  uint64_t plan_us = 0;
  uint64_t schedule_query_us = 0;
  uint64_t matrix_alloc_us = 0;
  std::vector<uint64_t> tile_bounds{};
  uint64_t tile_bound_min = 0;
  uint64_t tile_bound_max = 0;
  uint64_t tile_bound_hash = 0;
  rns8_plan_schedule_info schedule_info{};
  bool schedule_info_available = false;
  TimingSamples samples{};
  GpuEventSamples gpu_events{};
  uint64_t checksum = 0;
};

uint64_t elapsed_us(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

void mix_checksum(uint64_t& checksum, uint64_t value);

[[noreturn]] void usage_error(const std::string& message) {
  std::cerr << message << "\n";
  std::cerr
      << "usage: rns8-bench [--backend cpu|hip-direct|wrap64-byte-limb]\n"
      << "                  [--semantics bounded-i64|bounded-u64|wrap-u64]\n"
      << "                  [--device N] [--m M] [--n N] [--k K]\n"
      << "                  [--tile-m M] [--tile-n N]\n"
      << "                  [--bound-mode global|per-tile]\n"
      << "                  [--require-adaptive-execution]\n"
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

bool valid_tile_size(uint32_t value) {
  return value >= 64 && value <= 512 && (value & (value - 1u)) == 0;
}

rns8_backend_kind parse_backend(const std::string& value) {
  if (value == "cpu" || value == "cpu-reference") return RNS8_BACKEND_CPU_REFERENCE;
  if (value == "hip-direct") return RNS8_BACKEND_HIP_DIRECT;
  if (value == "wrap64-byte-limb") return RNS8_BACKEND_WRAP64_BYTE_LIMB;
  usage_error("unknown backend: " + value);
}

BenchSemantics parse_semantics(const std::string& value) {
  if (value == "bounded-i64") return BenchSemantics::BoundedI64;
  if (value == "bounded-u64") return BenchSemantics::BoundedU64;
  if (value == "wrap-u64" || value == "wrap-u64-mod-2-64") return BenchSemantics::WrapU64Mod2_64;
  usage_error("unknown semantics: " + value);
}

BoundMode parse_bound_mode(const std::string& value) {
  if (value == "global") return BoundMode::Global;
  if (value == "per-tile" || value == "per_tile") return BoundMode::PerTile;
  usage_error("unknown bound mode: " + value);
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
    } else if (arg == "--tile-m" && i + 1 < argc) {
      args.tile_m = parse_u32(argv[++i], "--tile-m");
    } else if (arg == "--tile-n" && i + 1 < argc) {
      args.tile_n = parse_u32(argv[++i], "--tile-n");
    } else if (arg == "--device" && i + 1 < argc) {
      args.device_id = static_cast<int>(parse_i64(argv[++i], "--device"));
    } else if (arg == "--backend" && i + 1 < argc) {
      args.backend = parse_backend(argv[++i]);
    } else if (arg == "--semantics" && i + 1 < argc) {
      args.semantics = parse_semantics(argv[++i]);
    } else if (arg == "--bound-mode" && i + 1 < argc) {
      args.bound_mode = parse_bound_mode(argv[++i]);
    } else if (arg == "--require-adaptive-execution") {
      args.require_adaptive_execution = true;
    } else if (arg == "--help") {
      std::cout
          << "usage: rns8-bench [--backend cpu|hip-direct|wrap64-byte-limb]\n"
          << "                  [--semantics bounded-i64|bounded-u64|wrap-u64]\n"
          << "                  [--device N] [--m M] [--n N] [--k K]\n"
          << "                  [--tile-m M] [--tile-n N]\n"
          << "                  [--bound-mode global|per-tile]\n"
          << "                  [--require-adaptive-execution]\n"
          << "                  [--warmups W] [--repeats R] [--seed S]\n";
      std::exit(0);
    } else {
      usage_error("unknown or incomplete argument: " + arg);
    }
  }

  if (args.m <= 0 || args.n <= 0 || args.k <= 0 || args.repeats == 0) {
    usage_error("matrix dimensions must be positive and repeats must be nonzero");
  }
  if (!valid_tile_size(args.tile_m) || !valid_tile_size(args.tile_n)) {
    usage_error("tile dimensions must be powers of two from 64 through 512");
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 && args.backend != RNS8_BACKEND_WRAP64_BYTE_LIMB &&
      args.backend != RNS8_BACKEND_HIP_DIRECT) {
    usage_error("wrap-u64 benchmark requires --backend wrap64-byte-limb or --backend hip-direct");
  }
  if (args.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB && args.semantics != BenchSemantics::WrapU64Mod2_64) {
    usage_error("wrap64-byte-limb backend requires --semantics wrap-u64");
  }
  if (args.bound_mode == BoundMode::PerTile) {
    if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
      usage_error("--bound-mode per-tile is only valid for bounded semantics");
    }
    if (args.backend != RNS8_BACKEND_HIP_DIRECT) {
      usage_error("--bound-mode per-tile currently captures the direct HIP adaptive path; use --backend hip-direct");
    }
  }
  if (args.require_adaptive_execution && args.bound_mode != BoundMode::PerTile) {
    usage_error("--require-adaptive-execution requires --bound-mode per-tile");
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
  switch (semantics) {
    case BenchSemantics::BoundedI64:
      return "bounded_i64";
    case BenchSemantics::BoundedU64:
      return "bounded_u64";
    case BenchSemantics::WrapU64Mod2_64:
      return "wrap_u64_mod_2_64";
  }
  return "unknown";
}

rns8_semantics c_semantics(BenchSemantics semantics) {
  switch (semantics) {
    case BenchSemantics::BoundedI64:
      return RNS8_BOUNDED_I64;
    case BenchSemantics::BoundedU64:
      return RNS8_BOUNDED_U64;
    case BenchSemantics::WrapU64Mod2_64:
      return RNS8_WRAP_U64_MOD_2_64;
  }
  return RNS8_BOUNDED_I64;
}

rns8_bound_kind global_bound_kind(BenchSemantics semantics) {
  switch (semantics) {
    case BenchSemantics::BoundedI64:
      return RNS8_BOUND_GLOBAL_MAX_ABS;
    case BenchSemantics::BoundedU64:
      return RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
    case BenchSemantics::WrapU64Mod2_64:
      return RNS8_BOUND_NONE;
  }
  return RNS8_BOUND_NONE;
}

rns8_bound_kind bound_kind(const Args& args) {
  if (args.bound_mode == BoundMode::PerTile) {
    switch (args.semantics) {
      case BenchSemantics::BoundedI64:
        return RNS8_BOUND_PER_TILE_MAX_ABS;
      case BenchSemantics::BoundedU64:
        return RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
      case BenchSemantics::WrapU64Mod2_64:
        return RNS8_BOUND_NONE;
    }
  }
  return global_bound_kind(args.semantics);
}

const char* bound_kind_name(const Args& args) {
  if (args.bound_mode == BoundMode::PerTile) {
    switch (args.semantics) {
      case BenchSemantics::BoundedI64:
        return "per_tile_max_abs";
      case BenchSemantics::BoundedU64:
        return "per_tile_max_unsigned";
      case BenchSemantics::WrapU64Mod2_64:
        return "none";
    }
  }
  switch (args.semantics) {
    case BenchSemantics::BoundedI64:
      return "global_max_abs";
    case BenchSemantics::BoundedU64:
      return "global_max_unsigned";
    case BenchSemantics::WrapU64Mod2_64:
      return "none";
  }
  return "unknown";
}

const char* bound_mode_name(BoundMode mode) {
  switch (mode) {
    case BoundMode::Global:
      return "global";
    case BoundMode::PerTile:
      return "per_tile";
  }
  return "unknown";
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

uint64_t ceil_div_i64_u32(int64_t value, uint32_t divisor) {
  const auto unsigned_value = static_cast<uint64_t>(value);
  return (unsigned_value + static_cast<uint64_t>(divisor) - 1u) / static_cast<uint64_t>(divisor);
}

std::size_t row_major_index(int64_t row, int64_t col, int64_t ld, const char* label) {
  const auto u_row = static_cast<uint64_t>(row);
  const auto u_col = static_cast<uint64_t>(col);
  const auto u_ld = static_cast<uint64_t>(ld);
  if (u_ld != 0 && u_row > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) / u_ld) {
    usage_error(std::string("matrix index overflows size_t for ") + label);
  }
  const uint64_t row_offset = u_row * u_ld;
  if (u_col > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) - row_offset) {
    usage_error(std::string("matrix index overflows size_t for ") + label);
  }
  return static_cast<std::size_t>(row_offset + u_col);
}

uint64_t benchmark_bound(const Args& args) {
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 || args.bound_mode == BoundMode::PerTile) {
    return 0;
  }
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

uint64_t checked_tile_count(const Args& args) {
  const uint64_t tile_rows = ceil_div_i64_u32(args.m, args.tile_m);
  const uint64_t tile_cols = ceil_div_i64_u32(args.n, args.tile_n);
  if (tile_cols != 0 && tile_rows > std::numeric_limits<uint64_t>::max() / tile_cols) {
    usage_error("tile grid overflows uint64_t");
  }
  const uint64_t tile_count = tile_rows * tile_cols;
  if (tile_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    usage_error("tile grid overflows size_t");
  }
  return tile_count;
}

std::vector<uint64_t> compute_i64_tile_bounds(
    const Args& args,
    const std::vector<int64_t>& A,
    const std::vector<int64_t>& B) {
  const auto u_k = static_cast<uint64_t>(args.k);
  if (u_k > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 256u) {
    usage_error("bounded-i64 per-tile benchmark k is too large for exact int64 tile-bound prepass");
  }
  const uint64_t tile_rows = ceil_div_i64_u32(args.m, args.tile_m);
  const uint64_t tile_cols = ceil_div_i64_u32(args.n, args.tile_n);
  std::vector<uint64_t> bounds(static_cast<std::size_t>(checked_tile_count(args)), 0);
  for (uint64_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
    const int64_t row_begin = static_cast<int64_t>(tile_row * static_cast<uint64_t>(args.tile_m));
    const int64_t row_end = std::min<int64_t>(args.m, row_begin + static_cast<int64_t>(args.tile_m));
    for (uint64_t tile_col = 0; tile_col < tile_cols; ++tile_col) {
      const int64_t col_begin = static_cast<int64_t>(tile_col * static_cast<uint64_t>(args.tile_n));
      const int64_t col_end = std::min<int64_t>(args.n, col_begin + static_cast<int64_t>(args.tile_n));
      uint64_t& tile_max = bounds[static_cast<std::size_t>(tile_row * tile_cols + tile_col)];
      for (int64_t row = row_begin; row < row_end; ++row) {
        for (int64_t col = col_begin; col < col_end; ++col) {
          int64_t acc = 0;
          for (int64_t kk = 0; kk < args.k; ++kk) {
            acc += A[row_major_index(row, kk, args.k, "A")] * B[row_major_index(kk, col, args.n, "B")];
          }
          const uint64_t abs_value = acc < 0 ? static_cast<uint64_t>(-acc) : static_cast<uint64_t>(acc);
          tile_max = std::max(tile_max, abs_value);
        }
      }
    }
  }
  return bounds;
}

std::vector<uint64_t> compute_u64_tile_bounds(
    const Args& args,
    const std::vector<uint64_t>& A,
    const std::vector<uint64_t>& B) {
  const auto u_k = static_cast<uint64_t>(args.k);
  if (u_k > std::numeric_limits<uint64_t>::max() / 256u) {
    usage_error("bounded-u64 per-tile benchmark k is too large for exact uint64 tile-bound prepass");
  }
  const uint64_t tile_rows = ceil_div_i64_u32(args.m, args.tile_m);
  const uint64_t tile_cols = ceil_div_i64_u32(args.n, args.tile_n);
  std::vector<uint64_t> bounds(static_cast<std::size_t>(checked_tile_count(args)), 0);
  for (uint64_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
    const int64_t row_begin = static_cast<int64_t>(tile_row * static_cast<uint64_t>(args.tile_m));
    const int64_t row_end = std::min<int64_t>(args.m, row_begin + static_cast<int64_t>(args.tile_m));
    for (uint64_t tile_col = 0; tile_col < tile_cols; ++tile_col) {
      const int64_t col_begin = static_cast<int64_t>(tile_col * static_cast<uint64_t>(args.tile_n));
      const int64_t col_end = std::min<int64_t>(args.n, col_begin + static_cast<int64_t>(args.tile_n));
      uint64_t& tile_max = bounds[static_cast<std::size_t>(tile_row * tile_cols + tile_col)];
      for (int64_t row = row_begin; row < row_end; ++row) {
        for (int64_t col = col_begin; col < col_end; ++col) {
          uint64_t acc = 0;
          for (int64_t kk = 0; kk < args.k; ++kk) {
            acc += A[row_major_index(row, kk, args.k, "A")] * B[row_major_index(kk, col, args.n, "B")];
          }
          tile_max = std::max(tile_max, acc);
        }
      }
    }
  }
  return bounds;
}

void record_tile_bounds(BenchmarkResult& result, std::vector<uint64_t> bounds) {
  result.tile_bounds = std::move(bounds);
  if (result.tile_bounds.empty()) {
    result.tile_bound_min = 0;
    result.tile_bound_max = 0;
    result.tile_bound_hash = 0;
    return;
  }
  result.tile_bound_min = *std::min_element(result.tile_bounds.begin(), result.tile_bounds.end());
  result.tile_bound_max = *std::max_element(result.tile_bounds.begin(), result.tile_bounds.end());
  uint64_t hash = 1469598103934665603ull;
  mix_checksum(hash, static_cast<uint64_t>(result.tile_bounds.size()));
  for (const uint64_t bound : result.tile_bounds) {
    mix_checksum(hash, bound);
  }
  result.tile_bound_hash = hash;
}

rns8_matrix_desc matrix_desc(int64_t rows, int64_t cols, const Args& args) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = c_semantics(args.semantics);
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind(args);
  desc.tile_m = args.tile_m;
  desc.tile_n = args.tile_n;
  desc.max_prefix = args.semantics == BenchSemantics::WrapU64Mod2_64 ? 0 : RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

rns8_gemm_desc gemm_desc(const Args& args, uint64_t bound, const std::vector<uint64_t>* tile_bounds = nullptr) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = c_semantics(args.semantics);
  desc.bound_kind = bound_kind(args);
  desc.requested_backend = args.backend;
  desc.m = args.m;
  desc.n = args.n;
  desc.k = args.k;
  desc.bound = bound;
  desc.max_prefix = args.semantics == BenchSemantics::WrapU64Mod2_64 ? 0 : RNS8_DEFAULT_BOUNDED_PREFIX;
  desc.tile_m = args.tile_m;
  desc.tile_n = args.tile_n;
  if (args.bound_mode == BoundMode::PerTile) {
    if (!tile_bounds) {
      usage_error("per-tile bound mode requires generated tile bounds");
    }
    desc.bound = 0;
    desc.tile_bounds = tile_bounds->data();
    desc.tile_bounds_count = static_cast<uint64_t>(tile_bounds->size());
  }
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

void capture_schedule_info(rns8_plan* plan, BenchmarkResult& result) {
  result.schedule_info.struct_size = sizeof(result.schedule_info);
  result.schedule_info.abi_version = RNS8_ABI_VERSION;
  const auto start = std::chrono::steady_clock::now();
  const rns8_status status = rns8_get_plan_schedule_info(plan, &result.schedule_info);
  const auto end = std::chrono::steady_clock::now();
  if (status != RNS8_SUCCESS) {
    fail_status("rns8_get_plan_schedule_info", status);
  }
  result.schedule_query_us = elapsed_us(start, end);
  result.schedule_info_available = true;
}

std::vector<std::string> gpu_event_phase_order(const Args& args) {
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return {
        "pack_h2d",
        "pack_kernel",
        "pack",
        "wrap64_byte_gemm36_kernel",
        "rns_gemm",
        "wrap64_export_kernel",
        "wrap64_export_d2h",
        "crt_export"};
  }
  return {
      "pack_h2d",
      "pack_kernel",
      "pack",
      "rns_gemm_kernel_group",
      "rns_gemm",
      "crt_export_status_memset",
      "crt_export_kernel",
      "crt_export_status_d2h",
      "crt_export_d2h",
      "crt_export"};
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

void print_gpu_event_timings(const Args& args, const GpuEventSamples& events) {
  std::cout << "  \"gpu_event_timings_us\": {\n";
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    print_named_gpu_event_array("pack_h2d", events.pack_h2d_us, true);
    print_named_gpu_event_array("pack_kernel", events.pack_kernel_us, true);
    print_named_gpu_event_array("pack", events.pack_us, true);
    print_named_gpu_event_array("wrap64_byte_gemm36_kernel", events.wrap64_byte_gemm36_kernel_us, true);
    print_named_gpu_event_array("rns_gemm", events.rns_gemm_us, true);
    print_named_gpu_event_array("wrap64_export_kernel", events.wrap64_export_kernel_us, true);
    print_named_gpu_event_array("wrap64_export_d2h", events.wrap64_export_d2h_us, true);
    print_named_gpu_event_array("crt_export", events.crt_export_us, false);
  } else {
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
  }
  std::cout << "  },\n";
}

void print_gpu_event_timing_summary(const Args& args, const GpuEventSamples& events) {
  std::cout << "  \"gpu_event_timing_summary_us\": {\n";
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    print_gpu_event_summary("pack_h2d", events.pack_h2d_us, true);
    print_gpu_event_summary("pack_kernel", events.pack_kernel_us, true);
    print_gpu_event_summary("pack", events.pack_us, true);
    print_gpu_event_summary("wrap64_byte_gemm36_kernel", events.wrap64_byte_gemm36_kernel_us, true);
    print_gpu_event_summary("rns_gemm", events.rns_gemm_us, true);
    print_gpu_event_summary("wrap64_export_kernel", events.wrap64_export_kernel_us, true);
    print_gpu_event_summary("wrap64_export_d2h", events.wrap64_export_d2h_us, true);
    print_gpu_event_summary("crt_export", events.crt_export_us, false);
  } else {
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
  }
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

void collect_wrap64_gemm_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double kernel = sum_event_label(events, samples, "rns_gemm", "wrap64_byte_gemm36_kernel");
  if (events.complete) {
    events.wrap64_byte_gemm36_kernel_us.push_back(kernel);
    events.rns_gemm_us.push_back(kernel);
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

void collect_wrap64_export_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double kernel = sum_event_label(events, samples, "crt_export", "wrap64_export_kernel");
  const double d2h = sum_event_label(events, samples, "crt_export", "wrap64_export_d2h");
  if (events.complete) {
    events.wrap64_export_kernel_us.push_back(kernel);
    events.wrap64_export_d2h_us.push_back(d2h);
    events.crt_export_us.push_back(kernel + d2h);
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

bool gpu_event_timing_available(const Args& args, const BenchmarkResult& result) {
  if (!result.gpu_events.requested || !result.gpu_events.complete) {
    return false;
  }
  const std::size_t repeats = result.samples.pack_us.size();
  if (repeats == 0 || result.gpu_events.pack_us.size() != repeats ||
      result.gpu_events.rns_gemm_us.size() != repeats || result.gpu_events.crt_export_us.size() != repeats) {
    return false;
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return result.gpu_events.wrap64_byte_gemm36_kernel_us.size() == repeats &&
           result.gpu_events.wrap64_export_kernel_us.size() == repeats &&
           result.gpu_events.wrap64_export_d2h_us.size() == repeats;
  }
  return result.gpu_events.rns_gemm_kernel_group_us.size() == repeats &&
         result.gpu_events.crt_export_status_memset_us.size() == repeats &&
         result.gpu_events.crt_export_kernel_us.size() == repeats &&
         result.gpu_events.crt_export_status_d2h_us.size() == repeats &&
         result.gpu_events.crt_export_d2h_us.size() == repeats;
}

bool schedule_uses_adaptive_work(const BenchmarkResult& result) {
  return result.schedule_info.adaptive_prefix_active != 0 || result.schedule_info.adaptive_skip_active != 0;
}

void enforce_per_tile_capture_contract(const Args& args, const BenchmarkResult& result) {
  if (args.bound_mode != BoundMode::PerTile) {
    return;
  }
  if (result.tile_bounds.empty()) {
    usage_error("per-tile benchmark capture has no generated tile bounds");
  }
  if (!schedule_uses_adaptive_work(result)) {
    usage_error(
        "per-tile benchmark capture did not produce adaptive prefix grouping or prefix skipping; adjust shape, seed, or inputs");
  }
  if (args.require_adaptive_execution && !schedule_uses_adaptive_work(result)) {
    usage_error("--require-adaptive-execution was requested but the plan is fixed-prefix");
  }
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
  if (args.bound_mode == BoundMode::PerTile) {
    record_tile_bounds(result, compute_i64_tile_bounds(args, A, B));
  }
  auto desc = gemm_desc(args, bound, &result.tile_bounds);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  enforce_per_tile_capture_contract(args, result);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  const auto alloc_start = std::chrono::steady_clock::now();
  auto a_desc = matrix_desc(args.m, args.k, args);
  auto b_desc = matrix_desc(args.k, args.n, args);
  auto c_desc = matrix_desc(args.m, args.n, args);
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
  if (args.bound_mode == BoundMode::PerTile) {
    record_tile_bounds(result, compute_u64_tile_bounds(args, A, B));
  }
  auto desc = gemm_desc(args, bound, &result.tile_bounds);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  enforce_per_tile_capture_contract(args, result);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  const auto alloc_start = std::chrono::steady_clock::now();
  auto a_desc = matrix_desc(args.m, args.k, args);
  auto b_desc = matrix_desc(args.k, args.n, args);
  auto c_desc = matrix_desc(args.m, args.n, args);
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

BenchmarkResult run_wrap_u64(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<uint64_t> C(checked_elements(args.m, args.n, "C"));
  for (auto& value : A) value = rng();
  for (auto& value : B) value = rng();

  BenchmarkResult result{};
  result.gpu_events.requested = gpu_event_capture_requested(args);
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  const auto alloc_start = std::chrono::steady_clock::now();
  auto a_desc = matrix_desc(args.m, args.k, args);
  auto b_desc = matrix_desc(args.k, args.n, args);
  auto c_desc = matrix_desc(args.m, args.n, args);
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
    status = rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_wrap_u64", status);
    if (collect_gpu_events) {
      collect_wrap64_gemm_gpu_events(result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_export_wrap_u64(ctx, plan, c_matrix, C.data(), args.n);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_wrap_u64", status);
    if (collect_gpu_events) {
      collect_wrap64_export_gpu_events(result.gpu_events);
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

uint32_t benchmark_prefix(const Args& args) {
  return args.semantics == BenchSemantics::WrapU64Mod2_64 ? 0 : RNS8_DEFAULT_BOUNDED_PREFIX;
}

const char* benchmark_name(const Args& args) {
  return args.semantics == BenchSemantics::WrapU64Mod2_64 ? "rns8_wrap_u64_persistent_byte_limb"
                                                          : "rns8_bounded_gemm_persistent_rns";
}

const char* epilogue_type(const Args& args) {
  return args.semantics == BenchSemantics::WrapU64Mod2_64 ? "low64_wrap_export" : "crt_export";
}

const char* input_distribution(const Args& args) {
  switch (args.semantics) {
    case BenchSemantics::BoundedI64:
      return "signed_uniform_-16_16";
    case BenchSemantics::BoundedU64:
      return "unsigned_uniform_0_16";
    case BenchSemantics::WrapU64Mod2_64:
      return "unsigned_rng_u64_full_range";
  }
  return "unknown";
}

const char* tile_bound_pattern(const Args& args) {
  switch (args.semantics) {
    case BenchSemantics::BoundedI64:
      return "exact_output_tile_max_abs_v1";
    case BenchSemantics::BoundedU64:
      return "exact_output_tile_max_unsigned_v1";
    case BenchSemantics::WrapU64Mod2_64:
      return "none";
  }
  return "unknown";
}

bool adaptive_execution_applied(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result) {
  return args.bound_mode == BoundMode::PerTile && info.backend == RNS8_BACKEND_HIP_DIRECT &&
         schedule_uses_adaptive_work(result);
}

const char* selected_kernel_name(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result) {
  if (adaptive_execution_applied(args, info, result)) {
    return "direct_hip_tiled_rns_gemm_v1";
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 && info.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "direct_hip_wrap64_byte_gemm36_correctness_v1";
  }
  return nullptr;
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
  const uint32_t prefix = benchmark_prefix(args);
  const bool adaptive_applied = adaptive_execution_applied(args, info, result);
  const bool per_modulus_estimate_applicable = prefix > 0 && !adaptive_applied;
  const double avg_per_modulus_gemm_estimate_us =
      per_modulus_estimate_applicable ? avg_gemm_us / static_cast<double>(prefix) : avg_gemm_us;
  const bool gpu_events_available = gpu_event_timing_available(args, result);
  const bool wrap64_hip_events = gpu_events_available && args.semantics == BenchSemantics::WrapU64Mod2_64;
  const bool adaptive_hip_events = gpu_events_available && adaptive_applied;
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
  std::cout << "  \"benchmark\": \"" << benchmark_name(args) << "\",\n";
  std::cout << "  \"backend_requested\": \"" << backend_name(args.backend) << "\",\n";
  std::cout << "  \"backend_selected\": \"" << backend_name(info.backend) << "\",\n";
  const char* selected_kernel = selected_kernel_name(args, info, result);
  std::cout << "  \"selected_kernel\": ";
  if (selected_kernel) {
    std::cout << "\"" << selected_kernel << "\"";
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"semantics\": \"" << semantics_name(args.semantics) << "\",\n";
  std::cout << "  \"bound_kind\": \"" << bound_kind_name(args) << "\",\n";
  std::cout << "  \"bound_mode\": \"" << bound_mode_name(args.bound_mode) << "\",\n";
  std::cout << "  \"bound\": " << bound << ",\n";
  std::cout << "  \"m\": " << args.m << ",\n";
  std::cout << "  \"n\": " << args.n << ",\n";
  std::cout << "  \"k\": " << args.k << ",\n";
  std::cout << "  \"prefix\": " << prefix << ",\n";
  std::cout << "  \"tile_m\": " << args.tile_m << ",\n";
  std::cout << "  \"tile_n\": " << args.tile_n << ",\n";
  std::cout << "  \"layout\": \"row_major\",\n";
  std::cout << "  \"k_block_size\": "
            << (args.semantics == BenchSemantics::WrapU64Mod2_64 ? args.k
                                                                  : std::min<int64_t>(args.k, RNS8_SAFE_INT32_K_BLOCK))
            << ",\n";
  std::cout << "  \"adaptive_tile_size\": null,\n";
  std::cout << "  \"tile_bounds_u64\": ";
  if (args.bound_mode == BoundMode::PerTile) {
    std::cout << "{\n";
    std::cout << "    \"source\": \"exact_seeded_input_prepass\",\n";
    std::cout << "    \"pattern\": \"" << tile_bound_pattern(args) << "\",\n";
    std::cout << "    \"order\": \"row_major_output_tiles\",\n";
    std::cout << "    \"count\": " << result.tile_bounds.size() << ",\n";
    std::cout << "    \"min\": " << result.tile_bound_min << ",\n";
    std::cout << "    \"max\": " << result.tile_bound_max << ",\n";
    std::cout << "    \"hash_u64\": " << result.tile_bound_hash << "\n";
    std::cout << "  },\n";
  } else {
    std::cout << "null,\n";
  }
  std::cout << "  \"schedule_metadata\": {\n";
  std::cout << "    \"source\": \"rns8_get_plan_schedule_info\",\n";
  std::cout << "    \"tile_m\": " << result.schedule_info.tile_m << ",\n";
  std::cout << "    \"tile_n\": " << result.schedule_info.tile_n << ",\n";
  std::cout << "    \"tile_rows\": " << result.schedule_info.tile_rows << ",\n";
  std::cout << "    \"tile_cols\": " << result.schedule_info.tile_cols << ",\n";
  std::cout << "    \"tile_count\": " << result.schedule_info.tile_count << ",\n";
  std::cout << "    \"min_required_prefix\": " << result.schedule_info.min_required_prefix << ",\n";
  std::cout << "    \"max_required_prefix\": " << result.schedule_info.max_required_prefix << ",\n";
  std::cout << "    \"min_selected_prefix\": " << result.schedule_info.min_selected_prefix << ",\n";
  std::cout << "    \"max_selected_prefix\": " << result.schedule_info.max_selected_prefix << ",\n";
  std::cout << "    \"prefix_group_count\": " << result.schedule_info.prefix_group_count << ",\n";
  std::cout << "    \"adaptive_prefix_active\": "
            << (result.schedule_info.adaptive_prefix_active ? "true" : "false") << ",\n";
  std::cout << "    \"adaptive_skip_active\": "
            << (result.schedule_info.adaptive_skip_active ? "true" : "false") << ",\n";
  std::cout << "    \"adaptive_execution_applied\": " << (adaptive_applied ? "true" : "false") << ",\n";
  std::cout << "    \"range_bit_length\": " << result.schedule_info.range_bit_length << "\n";
  std::cout << "  },\n";
  std::cout << "  \"epilogue_type\": \"" << epilogue_type(args) << "\",\n";
  std::cout << "  \"packed_layout_version\": "
            << (args.semantics == BenchSemantics::WrapU64Mod2_64 ? "\"byte_limb_v1\"" : "null") << ",\n";
  std::cout << "  \"seed\": " << args.seed << ",\n";
  std::cout << "  \"warmups\": " << args.warmups << ",\n";
  std::cout << "  \"repeats\": " << args.repeats << ",\n";
  std::cout << "  \"input_distribution\": \"" << input_distribution(args) << "\",\n";
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
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 && args.backend == RNS8_BACKEND_HIP_DIRECT) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the direct-HIP wrap64 byte-GEMM36 "
                 "correctness path; GPU event timing uses wrap64-specific byte-GEMM36/export labels plus "
                 "schema-compatible rns_gemm/crt_export aggregate aliases\",\n";
  } else if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the CPU wrap64 byte-limb reference; "
                 "no GPU event timing is requested for this backend\",\n";
  } else if (adaptive_applied) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the direct-HIP adaptive per-tile bounded "
                 "correctness path; GPU event timing aggregates all selected-prefix tiled GEMM launches and tiled "
                 "CRT export work into backend operation-group labels\",\n";
  } else {
    std::cout << "  \"timing_note\": \"host wall-clock timings; direct-HIP calls include current backend "
                 "synchronization, first-use persistent buffer allocation, copies, kernel launches, fused reduction, "
                 "and GPU bounded export when using HIP\",\n";
  }
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
            << (gpu_events_available
                    ? (wrap64_hip_events
                           ? "\"direct_hip_wrap64_byte_gemm36_default_stream_backend_operation_groups\""
                           : (adaptive_hip_events
                                  ? "\"direct_hip_bounded_adaptive_default_stream_backend_operation_groups\""
                                  : "\"direct_hip_default_stream_backend_operation_groups\""))
                    : "null")
            << ",\n";
  std::cout << "    \"gpu_event_timing_caveat\": "
            << (wrap64_hip_events
                    ? "\"HIP event timings record backend default-stream operation groups only; wrap64 uses a one-thread-per-output byte-GEMM36 correctness kernel and schema-compatible rns_gemm/crt_export aggregate aliases; host wall-clock timings remain required for CPU scheduling overhead, API dispatch, allocations, and synchronous host-side overhead not represented on the HIP stream\""
                    : (adaptive_hip_events
                           ? "\"HIP event timings record backend default-stream operation groups only; adaptive bounded captures aggregate all selected-prefix tile launches and tiled export kernels rather than exposing per-tile or per-prefix timings; host wall-clock timings remain required for scheduling overhead, API dispatch, allocations, and synchronous host-side overhead not represented on the HIP stream\""
                           : (gpu_events_available
                                  ? "\"HIP event timings record backend default-stream operation groups only; host wall-clock timings remain required for CPU scheduling overhead, API dispatch, allocations, and any synchronous host-side copy overhead not represented on the HIP stream\""
                                  : "null")))
            << ",\n";
  if (!gpu_events_available && !result.gpu_events.unavailable_reasons.empty()) {
    std::cout << "    \"gpu_event_timing_unavailable_reasons\": ";
    print_string_array(result.gpu_events.unavailable_reasons);
    std::cout << ",\n";
  }
  std::cout
      << "    \"phase_order\": [\"planning\", \"scheduling\", \"matrix_alloc\", \"pack\", \"rns_gemm\", "
         "\"crt_export\", \"end_to_end\"],\n";
  std::cout << "    \"gpu_event_phase_order\": ";
  print_string_array(gpu_event_phase_order(args));
  std::cout << ",\n";
  std::cout << "    \"phase_notes\": {\n";
  std::cout << "      \"planning\": \"one-time rns8_create_plan plus rns8_create_workspace host timing\",\n";
  std::cout << "      \"scheduling\": \"one-time rns8_get_plan_schedule_info host timing; planning remains the legacy aggregate that also includes this query\",\n";
  std::cout << "      \"matrix_alloc\": \"one-time persistent matrix allocation host timing\",\n";
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into persistent byte-limb matrices\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_wrap_u64; key retained for schema compatibility\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for low-64-bit rns8_export_wrap_u64; key retained for schema compatibility\",\n";
  } else {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into persistent RNS matrices\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_rns\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for export/reconstruction into logical output\",\n";
  }
  std::cout << "      \"end_to_end\": \"per-repeat pack plus rns_gemm plus crt_export host timing\"\n";
  std::cout << "    },\n";
  std::cout << "    \"phase_availability\": {\n";
  std::cout << "      \"scheduling\": {\n";
  std::cout << "        \"timed\": true,\n";
  std::cout << "        \"timing_key\": \"scheduling\",\n";
  std::cout << "        \"scope\": \"one_time_schedule_info_query\",\n";
  std::cout << "        \"reason\": \"measured with host steady_clock around rns8_get_plan_schedule_info\"\n";
  std::cout << "      },\n";
  std::cout << "      \"reduction\": {\n";
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    std::cout << "        \"timed\": false,\n";
    std::cout << "        \"timing_key\": null,\n";
    std::cout << "        \"scope\": \"not_applicable_wrap64_byte_limb\",\n";
    std::cout << "        \"reason\": \"strict wrap64 byte-limb captures do not use centered RNS residue reduction\"\n";
  } else {
    std::cout << "        \"timed\": false,\n";
    std::cout << "        \"timing_key\": null,\n";
    std::cout << "        \"scope\": \"fused_into_rns_gemm\",\n";
    std::cout << "        \"reason\": \"current CPU and direct-HIP RNS GEMM paths reduce to centered residues inside rns_gemm, so no separate reduction kernel timing exists\"\n";
  }
  std::cout << "      }\n";
  std::cout << "    }\n";
  std::cout << "  },\n";
  if (gpu_events_available) {
    print_gpu_event_timings(args, result.gpu_events);
    print_gpu_event_timing_summary(args, result.gpu_events);
  } else {
    std::cout << "  \"gpu_event_timings_us\": null,\n";
    std::cout << "  \"gpu_event_timing_summary_us\": null,\n";
  }
  std::cout << "  \"plan_us\": " << result.plan_us << ",\n";
  std::cout << "  \"avg_planning_us\": " << static_cast<double>(result.plan_us) << ",\n";
  std::cout << "  \"schedule_query_us\": " << result.schedule_query_us << ",\n";
  std::cout << "  \"avg_scheduling_us\": " << static_cast<double>(result.schedule_query_us) << ",\n";
  std::cout << "  \"matrix_alloc_us\": " << result.matrix_alloc_us << ",\n";
  std::cout << "  \"avg_matrix_alloc_us\": " << static_cast<double>(result.matrix_alloc_us) << ",\n";
  std::cout << "  \"avg_pack_us\": " << avg_pack_us << ",\n";
  std::cout << "  \"avg_rns_gemm_us\": " << avg_gemm_us << ",\n";
  std::cout << "  \"per_modulus_gemm_estimate_applicable\": "
            << (per_modulus_estimate_applicable ? "true" : "false") << ",\n";
  std::cout << "  \"avg_per_modulus_gemm_estimate_us\": " << avg_per_modulus_gemm_estimate_us << ",\n";
  std::cout << "  \"avg_crt_export_us\": " << avg_export_us << ",\n";
  std::cout << "  \"avg_end_to_end_us\": " << avg_end_to_end_us << ",\n";
  std::cout << "  \"raw_timings_us\": {\n";
  std::cout << "    \"planning\": ";
  print_single_u64_array(result.plan_us);
  std::cout << ",\n";
  std::cout << "    \"scheduling\": ";
  print_single_u64_array(result.schedule_query_us);
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
  print_single_timing_summary("scheduling", result.schedule_query_us, true);
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

  BenchmarkResult result{};
  switch (args.semantics) {
    case BenchSemantics::BoundedI64:
      result = run_bounded_i64(ctx, args, bound);
      break;
    case BenchSemantics::BoundedU64:
      result = run_bounded_u64(ctx, args, bound);
      break;
    case BenchSemantics::WrapU64Mod2_64:
      result = run_wrap_u64(ctx, args, bound);
      break;
  }
  rns8_destroy_context(ctx);
  print_json(args, info, result, bound, cmdline);
  return 0;
}
