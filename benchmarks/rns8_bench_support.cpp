#include "rns8_bench_support.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

#ifndef RNS8_CONFIGURED_HIP_ENABLED
#  define RNS8_CONFIGURED_HIP_ENABLED 0
#endif

#if RNS8_CONFIGURED_HIP_ENABLED
#  include <hip/hip_runtime_api.h>
#endif

#ifndef RNS8_GIT_COMMIT
#  define RNS8_GIT_COMMIT "unknown"
#endif

#ifndef RNS8_SOURCE_DIR
#  define RNS8_SOURCE_DIR "."
#endif

namespace rns8::bench {

uint64_t elapsed_us(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

[[noreturn]] void usage_error(const std::string& message) {
  std::cerr << message << "\n";
  std::cerr
      << "usage: rns8-bench [--backend auto|cpu|hip-direct|hipblaslt|ck|rocwmma|wrap64-byte-limb|hip-vector-alu-int64|hip-vector-alu-int64-baseline]\n"
      << "                  [--semantics bounded-i64|bounded-u64|wrap-u64|finite-u8-ring|finite-u8-field]\n"
      << "                  [--modulus M]\n"
      << "                  [--device N] [--m M] [--n N] [--k K]\n"
      << "                  [--output-ld-padding N]\n"
      << "                  [--tile-m M] [--tile-n N]\n"
      << "                  [--bound-mode global|per-tile]\n"
      << "                  [--input-profile uniform-small|adaptive-bands|finite-binary|finite-sparse|\n"
      << "                                   finite-low-hamming|finite-small-centered|finite-full-uniform]\n"
      << "                  [--bound-source static-profile|input-scan]\n"
      << "                  [--prefix-policy minimum-proven|fixed-requested] [--max-prefix N]\n"
      << "                  [--exact-wide-limbs 1..32]\n"
      << "                  [--residue-chain-length N]\n"
      << "                  [--residue-chain-final-export]\n"
      << "                  [--residue-chain-independent-final-export]\n"
      << "                  [--host-api-batch-size N]\n"
      << "                  [--next-op-hint final-export|rns-gemm|native-gemm|native-to-rns|reuse-b]\n"
      << "                  [--modulus-set default|experimental:NAME]\n"
      << "                  [--tile-shape-variant NAME]\n"
      << "                  [--export-variant NAME]\n"
      << "                  [--reconstruction-variant NAME]\n"
      << "                  [--grouped-dispatch N]\n"
      << "                  [--hip-graph-replay]\n"
      << "                  [--workload-proxy NAME]\n"
      << "                  [--resident-lifetime]\n"
      << "                  [--workspace-arena]\n"
      << "                  [--adaptive-grouped-scheduler]\n"
      << "                  [--streaming-overlap]\n"
      << "                  [--release-gate NAME]\n"
      << "                  [--verification-amortization NAME]\n"
      << "                  [--error-detection-policy NAME]\n"
      << "                  [--cpu-small-shape-selector NAME]\n"
      << "                  [--incremental-result-cache NAME]\n"
      << "                  [--require-adaptive-execution]\n"
      << "                  [--residue-channel-fusion]\n"
      << "                  [--oneshot]\n"
      << "                  [--transient-uniform-small-inputs]\n"
      << "                  [--native-to-rns-bridge]\n"
      << "                  [--vector-to-rns-chain]\n"
      << "                  [--vector-to-rns-chain-host-repack-control]\n"
      << "                  [--reuse-packed-inputs|--reuse-packed-a|--reuse-packed-b]\n"
      << "                  [--write-autotune-cache]  # refused; use release benchmark_sweep promotion\n"
      << "                  [--warmups W] [--repeats R] [--seed S]\n";
  std::exit(2);
}

void fail_status(const char* label, rns8_status status) {
  std::cerr << label << ": " << rns8_status_string(status) << "\n";
  std::exit(1);
}

void fail_hip_runtime(const char* label, int status) {
  std::cerr << label << ": HIP runtime status " << status << "\n";
  std::exit(1);
}

void mix_checksum(uint64_t& checksum, uint64_t value) {
  checksum ^= value;
  checksum *= 1099511628211ull;
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

std::string environment_value(const char* name) {
  if (!name || name[0] == '\0') {
    return {};
  }
#if defined(_MSC_VER)
  char* buffer = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&buffer, &length, name) != 0 || !buffer) {
    return {};
  }
  std::string value(buffer);
  std::free(buffer);
  return value;
#else
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string{};
#endif
}

uint32_t count_visible_devices_list(const std::string& value) {
  uint32_t count = 0;
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t comma = value.find(',', start);
    const std::size_t end = comma == std::string::npos ? value.size() : comma;
    std::string token = value.substr(start, end - start);
    token = trim_ascii_whitespace(std::move(token));
    if (!token.empty()) {
      ++count;
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return count;
}

uint32_t visible_device_count_from_environment() {
  for (const char* name : {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}) {
    const std::string value = environment_value(name);
    if (!trim_ascii_whitespace(value).empty()) {
      return count_visible_devices_list(value);
    }
  }
  return 0;
}

uint32_t runtime_hip_device_count() {
#if RNS8_CONFIGURED_HIP_ENABLED
  int count = 0;
  if (hipGetDeviceCount(&count) == hipSuccess && count > 0) {
    return static_cast<uint32_t>(count);
  }
#endif
  return 0;
}

uint32_t benchmark_node_gpu_count() {
  const uint32_t visible_count = visible_device_count_from_environment();
  const uint32_t runtime_count = runtime_hip_device_count();
  if (visible_count > runtime_count) {
    return visible_count;
  }
  return runtime_count;
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

void print_nullable_string(const char* value) {
  if (value && value[0] != '\0') {
    std::cout << "\"" << json_escape(value) << "\"";
  } else {
    std::cout << "null";
  }
}

void print_json_string_or_null(const char* value) {
  if (value && value[0] != '\0') {
    std::cout << "\"" << json_escape(value) << "\"";
  } else {
    std::cout << "null";
  }
}

void print_json_string_or_null(const std::string& value) {
  if (!value.empty()) {
    std::cout << "\"" << json_escape(value) << "\"";
  } else {
    std::cout << "null";
  }
}

std::size_t checked_elements(int64_t rows, int64_t cols, const char* label) {
  const auto u_rows = static_cast<uint64_t>(rows);
  const auto u_cols = static_cast<uint64_t>(cols);
  if (u_cols != 0 && u_rows > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) / u_cols) {
    usage_error(std::string("matrix size overflows size_t for ") + label);
  }
  return static_cast<std::size_t>(u_rows * u_cols);
}

std::size_t checked_limb_elements(int64_t rows, int64_t cols, uint32_t limb_count, const char* label) {
  const std::size_t elements = checked_elements(rows, cols, label);
  if (limb_count == 0 || elements > std::numeric_limits<std::size_t>::max() / limb_count) {
    usage_error(std::string("limb output size overflows size_t for ") + label);
  }
  return elements * static_cast<std::size_t>(limb_count);
}

std::size_t checked_bytes(std::size_t elements, std::size_t element_size, const char* label) {
  if (element_size != 0 && elements > std::numeric_limits<std::size_t>::max() / element_size) {
    usage_error(std::string("device buffer size overflows size_t for ") + label);
  }
  return elements * element_size;
}

std::size_t checked_add_bytes(std::size_t lhs, std::size_t rhs, const char* label) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    usage_error(std::string("device buffer size overflows size_t for ") + label);
  }
  return lhs + rhs;
}

}  // namespace rns8::bench
