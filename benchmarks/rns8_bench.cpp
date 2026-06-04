#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "backend_rocwmma/rocwmma_backend.hpp"
#include "backend_wrap64/wrap64_hip.hpp"
#include "core/backend_common.hpp"
#include "rns8/rns8.h"

#ifndef RNS8_CONFIGURED_AMDGPU_TARGETS
#  define RNS8_CONFIGURED_AMDGPU_TARGETS "not-configured"
#endif

#ifndef RNS8_CONFIGURED_HIP_ENABLED
#  define RNS8_CONFIGURED_HIP_ENABLED 0
#endif

#ifndef RNS8_CONFIGURED_HIP_ROOT
#  define RNS8_CONFIGURED_HIP_ROOT ""
#endif

#ifndef RNS8_CONFIGURED_HIPCC_PATH
#  define RNS8_CONFIGURED_HIPCC_PATH ""
#endif

#ifndef RNS8_CONFIGURED_HIPCC_VERSION
#  define RNS8_CONFIGURED_HIPCC_VERSION ""
#endif

#ifndef RNS8_CONFIGURED_HIP_SDK_OR_ROCM_VERSION
#  define RNS8_CONFIGURED_HIP_SDK_OR_ROCM_VERSION ""
#endif

#ifndef RNS8_GIT_COMMIT
#  define RNS8_GIT_COMMIT "unknown"
#endif

#ifndef RNS8_SOURCE_DIR
#  define RNS8_SOURCE_DIR "."
#endif

namespace {

constexpr uint32_t kBenchmarkSchemaVersion = 4;
constexpr uint32_t kDefaultExactWideBenchmarkLimbCount = 4;
constexpr uint32_t kWrap64RocwmmaCandidateTile = 16;
constexpr int64_t kWrap64RocwmmaCandidateMaxK = 32768;
constexpr const char* kWrap64RocwmmaCandidateRequestedBackend = "rocwmma-wrap64-candidate";
constexpr const char* kWrap64RocwmmaCandidateSelectedKernel = "rocwmma_wrap64_byte_gemm36_candidate_v0";
constexpr const char* kWrap64RocwmmaCandidateScheduleSource = "rns8_bench_wrap64_rocwmma_candidate_static_schedule";
constexpr const char* kWrap64RocwmmaCandidateBackendSource = "rns8_bench_wrap64_rocwmma_candidate";
constexpr const char* kWrap64RocwmmaCandidateEventLabel = "wrap64_rocwmma_candidate_gemm36_kernel_group";

enum class BenchSemantics {
  BoundedI64,
  BoundedU64,
  ExactWideSigned,
  ExactWideUnsigned,
  WrapU64Mod2_64,
  FiniteRingU8,
  FiniteFieldU8,
};

enum class BoundMode {
  Global,
  PerTile,
};

enum class InputProfile {
  UniformSmall,
  AdaptiveBands,
};

enum class BoundSource {
  StaticProfile,
  InputScan,
};

enum class PrefixPolicy {
  MinimumProven,
  FixedRequested,
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
  bool vector_alu_baseline = false;
  bool wrap64_rocwmma_candidate = false;
  BenchSemantics semantics = BenchSemantics::BoundedI64;
  uint16_t finite_modulus = 251;
  BoundMode bound_mode = BoundMode::Global;
  InputProfile input_profile = InputProfile::UniformSmall;
  BoundSource bound_source = BoundSource::StaticProfile;
  PrefixPolicy prefix_policy = PrefixPolicy::MinimumProven;
  uint32_t max_prefix_override = 0;
  uint32_t exact_wide_limb_count = kDefaultExactWideBenchmarkLimbCount;
  uint32_t residue_chain_length = 1;
  bool require_adaptive_execution = false;
  bool write_autotune_cache = false;
  bool oneshot = false;
  bool reuse_packed_inputs = false;
  bool reuse_packed_a = false;
  bool reuse_packed_b = false;
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
  std::map<std::string, std::vector<double>> timings_us;
};

enum class PrepackReuseStrategy {
  None,
  PersistentMatrixResidency,
  RocwmmaReusableBCache,
};

struct BenchmarkResult {
  uint64_t plan_us = 0;
  uint64_t schedule_query_us = 0;
  uint64_t global_bound_scan_us = 0;
  bool global_bound_scan_available = false;
  uint64_t tile_bound_scan_us = 0;
  bool tile_bound_scan_available = false;
  uint64_t matrix_alloc_us = 0;
  uint64_t static_bound = 0;
  uint64_t effective_bound = 0;
  bool effective_bound_available = false;
  uint64_t discovered_global_bound = 0;
  uint64_t bound_candidate_row_sum_col_max = 0;
  uint64_t bound_candidate_row_max_col_sum = 0;
  uint64_t row_abs_sum_max = 0;
  uint64_t row_abs_max = 0;
  uint64_t col_abs_sum_max = 0;
  uint64_t col_abs_max = 0;
  uint64_t zero_row_count = 0;
  uint64_t zero_col_count = 0;
  std::vector<uint64_t> tile_bounds{};
  uint64_t tile_bound_min = 0;
  uint64_t tile_bound_max = 0;
  uint64_t tile_bound_hash = 0;
  rns8_plan_schedule_info schedule_info{};
  bool schedule_info_available = false;
  uint64_t zero_output_tile_count = 0;
  uint64_t zero_output_selected_residue_plane_count = 0;
  std::string schedule_source = "rns8_get_plan_schedule_info";
  std::string target_id = "cpu";
  rns8_plan_backend_info backend_info{};
  bool backend_info_available = false;
  TimingSamples samples{};
  GpuEventSamples gpu_events{};
  uint64_t prepack_setup_us = 0;
  bool prepack_setup_available = false;
  PrepackReuseStrategy prepack_reuse_strategy = PrepackReuseStrategy::None;
  uint64_t checksum = 0;
};

#if RNS8_CONFIGURED_HIP_ENABLED
extern "C" int rns8_bench_vector_i64_gemm_device(
    int device_id,
    const int64_t* a,
    const int64_t* b,
    int64_t* c,
    uint32_t* status,
    int64_t m,
    int64_t n,
    int64_t k);
extern "C" int rns8_bench_vector_u64_gemm_device(
    int device_id,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    uint32_t* status,
    int64_t m,
    int64_t n,
    int64_t k);
#endif

uint64_t elapsed_us(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

void mix_checksum(uint64_t& checksum, uint64_t value);
uint32_t benchmark_prefix(const Args& args);
uint32_t selected_execution_prefix(const Args& args, const BenchmarkResult& result);

[[noreturn]] void usage_error(const std::string& message) {
  std::cerr << message << "\n";
  std::cerr
      << "usage: rns8-bench [--backend auto|cpu|hip-direct|hipblaslt|ck|rocwmma|wrap64-byte-limb|hip-vector-alu-int64|hip-vector-alu-int64-baseline]\n"
      << "                  [--semantics bounded-i64|bounded-u64|wrap-u64|finite-u8-ring|finite-u8-field]\n"
      << "                  [--modulus M]\n"
      << "                  [--device N] [--m M] [--n N] [--k K]\n"
      << "                  [--tile-m M] [--tile-n N]\n"
      << "                  [--bound-mode global|per-tile]\n"
      << "                  [--input-profile uniform-small|adaptive-bands]\n"
      << "                  [--bound-source static-profile|input-scan]\n"
      << "                  [--prefix-policy minimum-proven|fixed-requested] [--max-prefix N]\n"
      << "                  [--exact-wide-limbs 1..32]\n"
      << "                  [--residue-chain-length N]\n"
      << "                  [--require-adaptive-execution]\n"
      << "                  [--oneshot]\n"
      << "                  [--reuse-packed-inputs|--reuse-packed-a|--reuse-packed-b]\n"
      << "                  [--write-autotune-cache]  # refused; use release benchmark_sweep promotion\n"
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

bool finite_benchmark_semantics(BenchSemantics semantics) {
  return semantics == BenchSemantics::FiniteRingU8 || semantics == BenchSemantics::FiniteFieldU8;
}

bool bounded_benchmark_semantics(BenchSemantics semantics) {
  return semantics == BenchSemantics::BoundedI64 || semantics == BenchSemantics::BoundedU64;
}

bool exact_wide_benchmark_semantics(BenchSemantics semantics) {
  return semantics == BenchSemantics::ExactWideSigned || semantics == BenchSemantics::ExactWideUnsigned;
}

bool rns_chain_benchmark_semantics(BenchSemantics semantics) {
  return bounded_benchmark_semantics(semantics) || exact_wide_benchmark_semantics(semantics);
}

bool residue_current_output_mode(const Args& args) {
  return rns_chain_benchmark_semantics(args.semantics) && args.residue_chain_length > 1;
}

bool exact_wide_export_status_check_required(const Args& args) {
  if (args.semantics == BenchSemantics::ExactWideUnsigned) {
    return args.exact_wide_limb_count < 3;
  }
  if (args.semantics == BenchSemantics::ExactWideSigned) {
    return args.exact_wide_limb_count < 4;
  }
  return true;
}

bool valid_finite_field_modulus(uint16_t modulus) {
  if (modulus < 2 || modulus > 251) {
    return false;
  }
  for (uint16_t divisor = 2; divisor * divisor <= modulus; ++divisor) {
    if (modulus % divisor == 0) {
      return false;
    }
  }
  return true;
}

bool valid_finite_modulus(BenchSemantics semantics, uint16_t modulus) {
  if (semantics == BenchSemantics::FiniteRingU8) {
    return modulus >= 2 && modulus <= 256;
  }
  if (semantics == BenchSemantics::FiniteFieldU8) {
    return valid_finite_field_modulus(modulus);
  }
  return true;
}

void parse_backend_option(const std::string& value, Args& args) {
  args.vector_alu_baseline = false;
  args.wrap64_rocwmma_candidate = false;
  if (value == "auto") {
    args.backend = RNS8_BACKEND_AUTO;
    return;
  }
  if (value == "cpu" || value == "cpu-reference") {
    args.backend = RNS8_BACKEND_CPU_REFERENCE;
    return;
  }
  if (value == "hip-direct") {
    args.backend = RNS8_BACKEND_HIP_DIRECT;
    return;
  }
  if (value == "hipblaslt" || value == "hipblaslt-baseline") {
    args.backend = RNS8_BACKEND_HIPBLASLT;
    return;
  }
  if (value == "ck") {
    args.backend = RNS8_BACKEND_CK;
    return;
  }
  if (value == "rocwmma") {
    args.backend = RNS8_BACKEND_ROCWMMA;
    return;
  }
  if (value == kWrap64RocwmmaCandidateRequestedBackend) {
    args.backend = RNS8_BACKEND_ROCWMMA;
    args.wrap64_rocwmma_candidate = true;
    return;
  }
  if (value == "wrap64-byte-limb") {
    args.backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
    return;
  }
  if (value == "hip-vector-alu-int64" || value == "vector-alu-int64" ||
      value == "hip-vector-alu-int64-runtime" || value == "vector-alu-int64-runtime") {
    args.backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
    return;
  }
  if (value == "hip-vector-alu-int64-baseline" || value == "vector-alu-int64-baseline" ||
      value == "hip-vector-alu-int64-benchmark" || value == "vector-alu-int64-benchmark") {
    args.backend = RNS8_BACKEND_HIP_DIRECT;
    args.vector_alu_baseline = true;
    return;
  }
  usage_error("unknown backend: " + value);
}

BenchSemantics parse_semantics(const std::string& value) {
  if (value == "bounded-i64") return BenchSemantics::BoundedI64;
  if (value == "bounded-u64") return BenchSemantics::BoundedU64;
  if (value == "exact-wide-signed" || value == "exact-wide-i64" || value == "exact_wide_signed")
    return BenchSemantics::ExactWideSigned;
  if (value == "exact-wide-unsigned" || value == "exact-wide-u64" || value == "exact_wide_unsigned")
    return BenchSemantics::ExactWideUnsigned;
  if (value == "wrap-u64" || value == "wrap-u64-mod-2-64") return BenchSemantics::WrapU64Mod2_64;
  if (value == "finite-u8-ring" || value == "finite-ring-u8") return BenchSemantics::FiniteRingU8;
  if (value == "finite-u8-field" || value == "finite-field-u8") return BenchSemantics::FiniteFieldU8;
  usage_error("unknown semantics: " + value);
}

BoundMode parse_bound_mode(const std::string& value) {
  if (value == "global") return BoundMode::Global;
  if (value == "per-tile" || value == "per_tile") return BoundMode::PerTile;
  usage_error("unknown bound mode: " + value);
}

InputProfile parse_input_profile(const std::string& value) {
  if (value == "uniform-small" || value == "uniform_small") return InputProfile::UniformSmall;
  if (value == "adaptive-bands" || value == "adaptive_bands") return InputProfile::AdaptiveBands;
  usage_error("unknown input profile: " + value);
}

BoundSource parse_bound_source(const std::string& value) {
  if (value == "static-profile" || value == "static_profile" || value == "static") {
    return BoundSource::StaticProfile;
  }
  if (value == "input-scan" || value == "input_scan" || value == "scan" || value == "discovered-global") {
    return BoundSource::InputScan;
  }
  usage_error("unknown bound source: " + value);
}

PrefixPolicy parse_prefix_policy(const std::string& value) {
  if (value == "minimum-proven" || value == "minimum_proven" || value == "min" || value == "auto") {
    return PrefixPolicy::MinimumProven;
  }
  if (value == "fixed-requested" || value == "fixed_requested" || value == "fixed") {
    return PrefixPolicy::FixedRequested;
  }
  usage_error("unknown prefix policy: " + value);
}

Args parse_args(int argc, char** argv) {
  Args args;
  bool tile_m_set = false;
  bool tile_n_set = false;
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
      tile_m_set = true;
    } else if (arg == "--tile-n" && i + 1 < argc) {
      args.tile_n = parse_u32(argv[++i], "--tile-n");
      tile_n_set = true;
    } else if (arg == "--device" && i + 1 < argc) {
      args.device_id = static_cast<int>(parse_i64(argv[++i], "--device"));
    } else if (arg == "--backend" && i + 1 < argc) {
      parse_backend_option(argv[++i], args);
    } else if (arg == "--semantics" && i + 1 < argc) {
      args.semantics = parse_semantics(argv[++i]);
    } else if (arg == "--modulus" && i + 1 < argc) {
      const uint32_t parsed = parse_u32(argv[++i], "--modulus");
      if (parsed > 256) {
        usage_error("finite-u8 modulus must be <= 256");
      }
      args.finite_modulus = static_cast<uint16_t>(parsed);
    } else if (arg == "--bound-mode" && i + 1 < argc) {
      args.bound_mode = parse_bound_mode(argv[++i]);
    } else if (arg == "--input-profile" && i + 1 < argc) {
      args.input_profile = parse_input_profile(argv[++i]);
    } else if (arg == "--bound-source" && i + 1 < argc) {
      args.bound_source = parse_bound_source(argv[++i]);
    } else if (arg == "--prefix-policy" && i + 1 < argc) {
      args.prefix_policy = parse_prefix_policy(argv[++i]);
    } else if (arg == "--max-prefix" && i + 1 < argc) {
      args.max_prefix_override = parse_u32(argv[++i], "--max-prefix");
    } else if (arg == "--exact-wide-limbs" && i + 1 < argc) {
      args.exact_wide_limb_count = parse_u32(argv[++i], "--exact-wide-limbs");
    } else if (arg == "--residue-chain-length" && i + 1 < argc) {
      args.residue_chain_length = parse_u32(argv[++i], "--residue-chain-length");
    } else if (arg == "--require-adaptive-execution") {
      args.require_adaptive_execution = true;
    } else if (arg == "--oneshot" || arg == "--one-shot") {
      args.oneshot = true;
    } else if (arg == "--reuse-packed-inputs") {
      args.reuse_packed_inputs = true;
      args.reuse_packed_a = true;
      args.reuse_packed_b = true;
    } else if (arg == "--reuse-packed-a") {
      args.reuse_packed_inputs = true;
      args.reuse_packed_a = true;
    } else if (arg == "--reuse-packed-b") {
      args.reuse_packed_inputs = true;
      args.reuse_packed_b = true;
    } else if (arg == "--write-autotune-cache") {
      args.write_autotune_cache = true;
    } else if (arg == "--help") {
      std::cout
          << "usage: rns8-bench [--backend auto|cpu|hip-direct|hipblaslt|ck|rocwmma|wrap64-byte-limb|rocwmma-wrap64-candidate|hip-vector-alu-int64|hip-vector-alu-int64-baseline]\n"
          << "                  [--semantics bounded-i64|bounded-u64|exact-wide-signed|exact-wide-unsigned|wrap-u64|finite-u8-ring|finite-u8-field]\n"
          << "                  [--modulus M]\n"
          << "                  [--device N] [--m M] [--n N] [--k K]\n"
          << "                  [--tile-m M] [--tile-n N]\n"
          << "                  [--bound-mode global|per-tile]\n"
          << "                  [--input-profile uniform-small|adaptive-bands]\n"
          << "                  [--bound-source static-profile|input-scan]\n"
          << "                  [--prefix-policy minimum-proven|fixed-requested] [--max-prefix N]\n"
          << "                  [--exact-wide-limbs 1..32]\n"
          << "                  [--residue-chain-length N]\n"
          << "                  [--require-adaptive-execution]\n"
          << "                  [--oneshot]\n"
          << "                  [--reuse-packed-inputs|--reuse-packed-a|--reuse-packed-b]\n"
          << "                  [--write-autotune-cache]\n"
          << "                  [--warmups W] [--repeats R] [--seed S]\n";
      std::exit(0);
    } else {
      usage_error("unknown or incomplete argument: " + arg);
    }
  }

  if (args.m <= 0 || args.n <= 0 || args.k <= 0 || args.repeats == 0) {
    usage_error("matrix dimensions must be positive and repeats must be nonzero");
  }
  if (args.wrap64_rocwmma_candidate) {
    if (args.semantics != BenchSemantics::WrapU64Mod2_64) {
      usage_error("rocwmma-wrap64-candidate is only valid with --semantics wrap-u64");
    }
    if (!tile_m_set) {
      args.tile_m = kWrap64RocwmmaCandidateTile;
    }
    if (!tile_n_set) {
      args.tile_n = kWrap64RocwmmaCandidateTile;
    }
    if (args.tile_m != kWrap64RocwmmaCandidateTile || args.tile_n != kWrap64RocwmmaCandidateTile) {
      usage_error("rocwmma-wrap64-candidate uses a fixed 16x16 WMMA tile; pass --tile-m 16 --tile-n 16 or omit tile args");
    }
  } else if (!valid_tile_size(args.tile_m) || !valid_tile_size(args.tile_n)) {
    usage_error("tile dimensions must be powers of two from 64 through 512");
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 && args.backend != RNS8_BACKEND_WRAP64_BYTE_LIMB &&
      args.backend != RNS8_BACKEND_HIP_DIRECT && args.backend != RNS8_BACKEND_AUTO && !args.wrap64_rocwmma_candidate) {
    usage_error("wrap-u64 benchmark requires --backend auto, wrap64-byte-limb, hip-direct, or rocwmma-wrap64-candidate");
  }
  if (args.vector_alu_baseline &&
      (exact_wide_benchmark_semantics(args.semantics) || args.semantics == BenchSemantics::WrapU64Mod2_64 ||
       finite_benchmark_semantics(args.semantics))) {
    usage_error("hip-vector-alu-int64-baseline is only valid for bounded-i64 or bounded-u64 semantics");
  }
  if (args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64 &&
      (exact_wide_benchmark_semantics(args.semantics) || args.semantics == BenchSemantics::WrapU64Mod2_64 ||
       finite_benchmark_semantics(args.semantics))) {
    usage_error("hip-vector-alu-int64 is only valid for bounded-i64 or bounded-u64 semantics");
  }
  if (args.reuse_packed_inputs && args.vector_alu_baseline) {
    usage_error("--reuse-packed-inputs is only valid for persistent matrix benchmark paths");
  }
  if (args.oneshot) {
    if (!bounded_benchmark_semantics(args.semantics) && !finite_benchmark_semantics(args.semantics)) {
      usage_error("--oneshot is currently only valid for bounded-i64, bounded-u64, or finite-u8 semantics");
    }
    if (args.bound_mode != BoundMode::Global) {
      usage_error("--oneshot currently requires --bound-mode global");
    }
    if (args.reuse_packed_inputs) {
      usage_error("--oneshot cannot be combined with packed-input reuse modes");
    }
    if (args.residue_chain_length != 1) {
      usage_error("--oneshot cannot be combined with --residue-chain-length > 1");
    }
    if (args.vector_alu_baseline || args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64 ||
        args.backend == RNS8_BACKEND_AUTO || args.backend == RNS8_BACKEND_HIPBLASLT ||
        args.backend == RNS8_BACKEND_CK || args.backend == RNS8_BACKEND_ROCWMMA ||
        args.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      usage_error("--oneshot currently requires --backend cpu or --backend hip-direct");
    }
  }
  if (args.input_profile != InputProfile::UniformSmall && !bounded_benchmark_semantics(args.semantics)) {
    usage_error("--input-profile adaptive-bands is only valid for bounded-i64 or bounded-u64 semantics");
  }
  if (args.bound_source == BoundSource::InputScan) {
    if (!bounded_benchmark_semantics(args.semantics)) {
      usage_error("--bound-source input-scan is only valid for bounded-i64 or bounded-u64 semantics");
    }
  }
  const bool rns_prefix_semantics = bounded_benchmark_semantics(args.semantics) ||
                                    exact_wide_benchmark_semantics(args.semantics);
  if (!rns_prefix_semantics &&
      (args.max_prefix_override != 0 || args.prefix_policy != PrefixPolicy::MinimumProven)) {
    usage_error("--max-prefix and --prefix-policy are only valid for bounded or exact-wide RNS semantics");
  }
  if (rns_prefix_semantics && args.max_prefix_override > RNS8_MAX_SUPPORTED_PREFIX) {
    usage_error("--max-prefix must be <= RNS8_MAX_SUPPORTED_PREFIX");
  }
  if (args.exact_wide_limb_count == 0 || args.exact_wide_limb_count > 32) {
    usage_error("--exact-wide-limbs must be in [1, 32]");
  }
  if (args.exact_wide_limb_count != kDefaultExactWideBenchmarkLimbCount &&
      !exact_wide_benchmark_semantics(args.semantics)) {
    usage_error("--exact-wide-limbs is only valid for exact-wide semantics");
  }
  if (args.residue_chain_length == 0) {
    usage_error("--residue-chain-length must be positive");
  }
  if (args.residue_chain_length > 1) {
    if (!rns_chain_benchmark_semantics(args.semantics)) {
      usage_error("--residue-chain-length > 1 is only valid for bounded or exact-wide RNS semantics");
    }
    if (args.m != args.n || args.n != args.k) {
      usage_error("--residue-chain-length > 1 currently requires square m=n=k RNS shapes");
    }
    if (bounded_benchmark_semantics(args.semantics)) {
      if (args.bound_mode != BoundMode::Global) {
        usage_error("bounded --residue-chain-length > 1 currently requires --bound-mode global");
      }
      if (args.vector_alu_baseline || args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64 ||
          args.backend == RNS8_BACKEND_AUTO) {
        usage_error("bounded --residue-chain-length > 1 requires an explicit RNS backend, not auto or vector-ALU");
      }
    }
  }
#if !RNS8_CONFIGURED_HIP_ENABLED
  if (args.vector_alu_baseline || args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    usage_error("hip-vector-alu-int64 requires a HIP-enabled benchmark build");
  }
#endif
  if (args.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB && args.semantics != BenchSemantics::WrapU64Mod2_64) {
    usage_error("wrap64-byte-limb backend requires --semantics wrap-u64");
  }
  if (finite_benchmark_semantics(args.semantics)) {
    if (!valid_finite_modulus(args.semantics, args.finite_modulus)) {
      usage_error("invalid modulus for selected finite-u8 benchmark semantics");
    }
    if (args.bound_mode != BoundMode::Global) {
      usage_error("finite-u8 benchmarks require --bound-mode global");
    }
  }
  if (args.bound_mode == BoundMode::PerTile) {
    if (exact_wide_benchmark_semantics(args.semantics) || args.semantics == BenchSemantics::WrapU64Mod2_64 ||
        finite_benchmark_semantics(args.semantics)) {
      usage_error("--bound-mode per-tile is only valid for bounded semantics");
    }
    if (!args.vector_alu_baseline && args.backend != RNS8_BACKEND_CPU_REFERENCE &&
        args.backend != RNS8_BACKEND_HIP_DIRECT && args.backend != RNS8_BACKEND_CK &&
        args.backend != RNS8_BACKEND_ROCWMMA && args.backend != RNS8_BACKEND_AUTO &&
        args.backend != RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      usage_error(
          "--bound-mode per-tile currently captures CPU, direct HIP, CK, rocWMMA, or hip-vector-alu-int64 paths");
    }
  }
  if (args.require_adaptive_execution && args.bound_mode != BoundMode::PerTile) {
    usage_error("--require-adaptive-execution requires --bound-mode per-tile");
  }
  if (args.device_id == std::numeric_limits<int>::min()) {
    args.device_id =
        (args.vector_alu_baseline || args.backend == RNS8_BACKEND_HIP_DIRECT || args.backend == RNS8_BACKEND_HIPBLASLT ||
         args.backend == RNS8_BACKEND_CK || args.backend == RNS8_BACKEND_ROCWMMA || args.backend == RNS8_BACKEND_AUTO ||
         args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64)
            ? 0
            : -1;
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
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
      return "hip-vector-alu-int64";
    case RNS8_BACKEND_HIPBLASLT:
      return "hipblaslt";
    case RNS8_BACKEND_CK:
      return "ck";
    case RNS8_BACKEND_ROCWMMA:
      return "rocwmma";
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
      return "wrap64-byte-limb";
  }
  return "unknown";
}

const char* requested_backend_name(const Args& args) {
  if (args.wrap64_rocwmma_candidate) {
    return kWrap64RocwmmaCandidateRequestedBackend;
  }
  return args.vector_alu_baseline ? "hip-vector-alu-int64" : backend_name(args.backend);
}

const char* selected_backend_name(const Args& args, const rns8_device_info& info, const BenchmarkResult* result = nullptr) {
  if (args.vector_alu_baseline) {
    return "hip-vector-alu-int64";
  }
  if (result && result->backend_info_available) {
    return backend_name(result->backend_info.backend);
  }
  return backend_name(info.backend);
}

bool benchmark_gpu_target_backend(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_HIP_DIRECT || backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64 ||
         backend == RNS8_BACKEND_HIPBLASLT || backend == RNS8_BACKEND_CK || backend == RNS8_BACKEND_ROCWMMA;
}

bool concrete_benchmark_target_id(const char* target) {
  if (!target || target[0] == '\0') {
    return false;
  }
  const std::string value(target);
  return value != "none" && value != "cpu" && value != "unknown";
}

std::string benchmark_target_id_for_context(rns8_context* ctx, rns8_backend_kind backend) {
  if (!benchmark_gpu_target_backend(backend)) {
    return "cpu";
  }
  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  if (rns8_get_device_info(ctx, &info) == RNS8_SUCCESS && concrete_benchmark_target_id(info.gcn_arch)) {
    return info.gcn_arch;
  }
  return "unknown";
}

std::string autotune_key_field(const char* key, const char* field) {
  if (!key || !field || field[0] == '\0') {
    return {};
  }
  const std::string text(key);
  const std::string prefix = std::string(field) + "=";
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t end = text.find(';', start);
    const std::string part = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (part.rfind(prefix, 0) == 0) {
      return part.substr(prefix.size());
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return {};
}

void update_result_target_id_from_key(BenchmarkResult& result) {
  const std::string target = autotune_key_field(result.backend_info.autotune_key, "target_id");
  if (!target.empty()) {
    result.target_id = target;
  } else if (result.backend_info_available && !benchmark_gpu_target_backend(result.backend_info.backend)) {
    result.target_id = "cpu";
  }
}

std::string benchmark_key_target_id(const BenchmarkResult& result) {
  return result.target_id.empty() ? "unknown" : result.target_id;
}

bool bounded_native_a_reuse_b_requested(const Args& args) {
  return !args.oneshot && bounded_benchmark_semantics(args.semantics) && args.reuse_packed_b &&
         !args.reuse_packed_a && args.backend == RNS8_BACKEND_HIP_DIRECT &&
         args.bound_mode == BoundMode::Global && args.residue_chain_length == 1;
}

bool bounded_native_a_reuse_b_uniform_small_a(const Args& args) {
  return args.input_profile == InputProfile::UniformSmall;
}

const char* input_profile_name(const Args& args) {
  return args.input_profile == InputProfile::UniformSmall ? "uniform-small" : "adaptive-bands";
}

const char* bound_source_name(const Args& args) {
  return args.bound_source == BoundSource::InputScan ? "input_scan" : "static_profile";
}

const char* bound_discovery_source_name(const Args& args, bool global_bound_scan_available) {
  if (global_bound_scan_available) {
    return "input_row_column_abs_summary";
  }
  if (args.bound_source == BoundSource::InputScan && args.bound_mode == BoundMode::PerTile) {
    return "input_exact_tile_bounds";
  }
  return "static_profile_contract";
}

const char* backend_metadata_source(const Args& args) {
  if (args.wrap64_rocwmma_candidate) {
    return kWrap64RocwmmaCandidateBackendSource;
  }
  if (args.oneshot) {
    return "rns8_bench_public_oneshot_api";
  }
  if (bounded_native_a_reuse_b_requested(args)) {
    if (bounded_native_a_reuse_b_uniform_small_a(args)) {
      return "rns8_bench_uniform_small_i8_ab_reuse_b_path";
    }
    return "rns8_bench_native_a_reuse_b_path";
  }
  if (finite_benchmark_semantics(args.semantics) && args.reuse_packed_b && !args.reuse_packed_a &&
      args.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "rns8_bench_native_a_reuse_b_path";
  }
  return args.vector_alu_baseline ? "rns8_bench_vector_alu_baseline" : "rns8_get_plan_backend_info";
}

bool runtime_vector_alu_backend(const Args& args) {
  return !args.vector_alu_baseline && args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
}

bool oneshot_benchmark_mode(const Args& args) {
  return args.oneshot;
}

const char* benchmark_execution_mode_name(const Args& args) {
  if (args.oneshot) {
    return "public_oneshot_transient_native_inputs";
  }
  if (args.vector_alu_baseline) {
    return "benchmark_owned_vector_alu_native_buffers";
  }
  if (runtime_vector_alu_backend(args)) {
    return "public_runtime_vector_alu_native_buffers";
  }
  if (args.wrap64_rocwmma_candidate) {
    return "internal_wrap64_rocwmma_candidate";
  }
  if (bounded_native_a_reuse_b_requested(args)) {
    if (bounded_native_a_reuse_b_uniform_small_a(args)) {
      return "transient_uniform_small_i8_a_resident_i8_b_reuse";
    }
    return "transient_native_a_resident_b_reuse";
  }
  if (finite_benchmark_semantics(args.semantics) && args.reuse_packed_b && !args.reuse_packed_a &&
      args.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "transient_native_a_resident_b_reuse";
  }
  return "persistent_resident_matrices";
}

const char* pack_mode_name(const Args& args) {
  if (!args.reuse_packed_inputs) {
    return "per_repeat_repack";
  }
  if (args.reuse_packed_a && args.reuse_packed_b) {
    return "prepacked_reuse";
  }
  return args.reuse_packed_a ? "prepacked_reuse_a" : "prepacked_reuse_b";
}

bool reuses_all_packed_inputs(const Args& args) {
  return args.reuse_packed_a && args.reuse_packed_b;
}

bool expects_hipblaslt_pack_transpose_event(const Args& args) {
  return !reuses_all_packed_inputs(args);
}

template <typename PackA, typename PackB>
void pack_preused_inputs(const Args& args, uint64_t source_version, const PackA& pack_a, const PackB& pack_b) {
  if (args.reuse_packed_a) {
    pack_a(source_version);
  }
  if (args.reuse_packed_b) {
    pack_b(source_version);
  }
}

template <typename PackA, typename PackB>
void pack_per_repeat_inputs(const Args& args, uint64_t source_version, const PackA& pack_a, const PackB& pack_b) {
  if (!args.reuse_packed_a) {
    pack_a(source_version);
  }
  if (!args.reuse_packed_b) {
    pack_b(source_version);
  }
}

std::string prepack_reuse_operand_text(const Args& args) {
  if (args.reuse_packed_a && args.reuse_packed_b) {
    return "A and B";
  }
  if (args.reuse_packed_a) {
    return "A";
  }
  if (args.reuse_packed_b) {
    return "B";
  }
  return "none";
}

std::string per_repeat_pack_operand_text(const Args& args) {
  if (!args.reuse_packed_inputs) {
    return "A and B";
  }
  if (reuses_all_packed_inputs(args)) {
    return "none";
  }
  return args.reuse_packed_a ? "B" : "A";
}

std::vector<std::string> prepack_reuse_operands(const Args& args) {
  std::vector<std::string> operands;
  if (args.reuse_packed_a) {
    operands.push_back("A");
  }
  if (args.reuse_packed_b) {
    operands.push_back("B");
  }
  return operands;
}

void fail_status(const char* label, rns8_status status);

const char* prepack_reuse_strategy_name(PrepackReuseStrategy strategy) {
  switch (strategy) {
    case PrepackReuseStrategy::None:
      return "none";
    case PrepackReuseStrategy::PersistentMatrixResidency:
      return "persistent_matrix_residency";
    case PrepackReuseStrategy::RocwmmaReusableBCache:
      return "rocwmma_reusable_b_cache";
  }
  return "unknown";
}

bool uses_runtime_b_prepack_cache(const BenchmarkResult& result) {
  return result.prepack_reuse_strategy == PrepackReuseStrategy::RocwmmaReusableBCache;
}

bool should_probe_reusable_b_prepack_cache(const Args& args) {
  return args.reuse_packed_inputs && args.reuse_packed_b && !args.reuse_packed_a &&
         args.backend != RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
}

rns8_status run_rns_gemm_with_optional_b_cache(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* a_matrix,
    const rns8_matrix* b_matrix,
    const rns8_prepack_cache* b_cache,
    rns8_matrix* c_matrix,
    rns8_workspace* workspace) {
  if (b_cache) {
    return rns8_gemm_rns_prepacked_b(ctx, plan, a_matrix, b_cache, c_matrix, workspace);
  }
  return rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
}

bool maybe_create_reusable_b_prepack_cache(
    rns8_context* ctx,
    const rns8_plan* plan,
    rns8_matrix* b_matrix,
    rns8_prepack_cache** out) {
  *out = nullptr;
  rns8_prepack_cache_key_info key{};
  key.struct_size = sizeof(key);
  key.abi_version = RNS8_ABI_VERSION;
  const rns8_status key_status = rns8_get_prepack_cache_key_info(plan, b_matrix, RNS8_OPERAND_B, &key);
  if (key_status != RNS8_SUCCESS) {
    fail_status("rns8_get_prepack_cache_key_info(B)", key_status);
  }
  if (key.reusable_prepack_cache_available == 0) {
    return false;
  }
  const rns8_status cache_status = rns8_create_prepack_cache(ctx, plan, b_matrix, RNS8_OPERAND_B, out);
  if (cache_status != RNS8_SUCCESS) {
    fail_status("rns8_create_prepack_cache(B)", cache_status);
  }
  return true;
}

void set_backend_text(char* dst, std::size_t dst_size, const char* text) {
  if (dst_size == 0) {
    return;
  }
  std::snprintf(dst, dst_size, "%s", text ? text : "");
  dst[dst_size - 1] = '\0';
}

const char* semantics_name(BenchSemantics semantics) {
  switch (semantics) {
    case BenchSemantics::BoundedI64:
      return "bounded_i64";
    case BenchSemantics::BoundedU64:
      return "bounded_u64";
    case BenchSemantics::ExactWideSigned:
      return "exact_wide_signed";
    case BenchSemantics::ExactWideUnsigned:
      return "exact_wide_unsigned";
    case BenchSemantics::WrapU64Mod2_64:
      return "wrap_u64_mod_2_64";
    case BenchSemantics::FiniteRingU8:
      return "finite_ring_u8";
    case BenchSemantics::FiniteFieldU8:
      return "finite_field_u8";
  }
  return "unknown";
}

rns8_semantics c_semantics(BenchSemantics semantics) {
  switch (semantics) {
    case BenchSemantics::BoundedI64:
      return RNS8_BOUNDED_I64;
    case BenchSemantics::BoundedU64:
      return RNS8_BOUNDED_U64;
    case BenchSemantics::ExactWideSigned:
      return RNS8_EXACT_WIDE_SIGNED;
    case BenchSemantics::ExactWideUnsigned:
      return RNS8_EXACT_WIDE_UNSIGNED;
    case BenchSemantics::WrapU64Mod2_64:
      return RNS8_WRAP_U64_MOD_2_64;
    case BenchSemantics::FiniteRingU8:
      return RNS8_FINITE_RING_U8;
    case BenchSemantics::FiniteFieldU8:
      return RNS8_FINITE_FIELD_U8;
  }
  return RNS8_BOUNDED_I64;
}

rns8_bound_kind global_bound_kind(BenchSemantics semantics) {
  switch (semantics) {
    case BenchSemantics::BoundedI64:
      return RNS8_BOUND_GLOBAL_MAX_ABS;
    case BenchSemantics::BoundedU64:
      return RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
    case BenchSemantics::ExactWideSigned:
    case BenchSemantics::ExactWideUnsigned:
    case BenchSemantics::WrapU64Mod2_64:
    case BenchSemantics::FiniteRingU8:
    case BenchSemantics::FiniteFieldU8:
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
      case BenchSemantics::ExactWideSigned:
      case BenchSemantics::ExactWideUnsigned:
      case BenchSemantics::WrapU64Mod2_64:
      case BenchSemantics::FiniteRingU8:
      case BenchSemantics::FiniteFieldU8:
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
      case BenchSemantics::ExactWideSigned:
      case BenchSemantics::ExactWideUnsigned:
      case BenchSemantics::WrapU64Mod2_64:
      case BenchSemantics::FiniteRingU8:
      case BenchSemantics::FiniteFieldU8:
        return "none";
    }
  }
  switch (args.semantics) {
    case BenchSemantics::BoundedI64:
      return "global_max_abs";
    case BenchSemantics::BoundedU64:
      return "global_max_unsigned";
    case BenchSemantics::ExactWideSigned:
    case BenchSemantics::ExactWideUnsigned:
    case BenchSemantics::WrapU64Mod2_64:
    case BenchSemantics::FiniteRingU8:
    case BenchSemantics::FiniteFieldU8:
      return "none";
  }
  return "unknown";
}

const char* bound_kind_name(rns8_bound_kind bound_kind) {
  switch (bound_kind) {
    case RNS8_BOUND_NONE:
      return "none";
    case RNS8_BOUND_GLOBAL_MAX_ABS:
      return "global_max_abs";
    case RNS8_BOUND_GLOBAL_MAX_UNSIGNED:
      return "global_max_unsigned";
    case RNS8_BOUND_PER_TILE_MAX_ABS:
      return "per_tile_max_abs";
    case RNS8_BOUND_PER_TILE_MAX_UNSIGNED:
      return "per_tile_max_unsigned";
    case RNS8_BOUND_INPUT_RANGE_AND_K:
      return "input_range_and_k";
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

uint64_t profile_band_for_coordinate(int64_t coordinate, uint32_t tile_size, bool allow_zero) {
  constexpr std::array<uint64_t, 4> zero_based = {0, 1, 4, 16};
  constexpr std::array<uint64_t, 4> nonzero = {1, 2, 8, 16};
  const uint64_t tile = static_cast<uint64_t>(coordinate) / static_cast<uint64_t>(tile_size);
  return (allow_zero ? zero_based : nonzero)[static_cast<std::size_t>(tile % 4u)];
}

int64_t sample_signed_band(std::mt19937_64& rng, uint64_t magnitude) {
  if (magnitude == 0) {
    return 0;
  }
  std::uniform_int_distribution<int64_t> dist(
      -static_cast<int64_t>(magnitude),
      static_cast<int64_t>(magnitude));
  return dist(rng);
}

uint64_t sample_unsigned_band(std::mt19937_64& rng, uint64_t magnitude) {
  if (magnitude == 0) {
    return 0;
  }
  std::uniform_int_distribution<uint64_t> dist(0, magnitude);
  return dist(rng);
}

void fill_bounded_i64_inputs(
    const Args& args,
    std::vector<int64_t>& A,
    std::vector<int64_t>& B,
    std::mt19937_64& rng) {
  if (args.input_profile == InputProfile::UniformSmall) {
    std::uniform_int_distribution<int64_t> dist(-16, 16);
    for (auto& value : A) value = dist(rng);
    for (auto& value : B) value = dist(rng);
    return;
  }
  for (int64_t row = 0; row < args.m; ++row) {
    const uint64_t row_band = profile_band_for_coordinate(row, args.tile_m, true);
    for (int64_t kk = 0; kk < args.k; ++kk) {
      A[row_major_index(row, kk, args.k, "A")] = sample_signed_band(rng, row_band);
    }
  }
  for (int64_t kk = 0; kk < args.k; ++kk) {
    for (int64_t col = 0; col < args.n; ++col) {
      const uint64_t col_band = profile_band_for_coordinate(col, args.tile_n, false);
      B[row_major_index(kk, col, args.n, "B")] = sample_signed_band(rng, col_band);
    }
  }
}

void fill_bounded_u64_inputs(
    const Args& args,
    std::vector<uint64_t>& A,
    std::vector<uint64_t>& B,
    std::mt19937_64& rng) {
  if (args.input_profile == InputProfile::UniformSmall) {
    std::uniform_int_distribution<uint64_t> dist(0, 16);
    for (auto& value : A) value = dist(rng);
    for (auto& value : B) value = dist(rng);
    return;
  }
  for (int64_t row = 0; row < args.m; ++row) {
    const uint64_t row_band = profile_band_for_coordinate(row, args.tile_m, true);
    for (int64_t kk = 0; kk < args.k; ++kk) {
      A[row_major_index(row, kk, args.k, "A")] = sample_unsigned_band(rng, row_band);
    }
  }
  for (int64_t kk = 0; kk < args.k; ++kk) {
    for (int64_t col = 0; col < args.n; ++col) {
      const uint64_t col_band = profile_band_for_coordinate(col, args.tile_n, false);
      B[row_major_index(kk, col, args.n, "B")] = sample_unsigned_band(rng, col_band);
    }
  }
}

int8_t uniform_small_i64_to_i8(int64_t value) {
  if (value < -16 || value > 16) {
    usage_error("uniform-small i64 fast path received an input outside [-16, 16]");
  }
  return static_cast<int8_t>(value);
}

int8_t uniform_small_u64_to_i8(uint64_t value) {
  if (value > 16) {
    usage_error("uniform-small u64 fast path received an input outside [0, 16]");
  }
  return static_cast<int8_t>(value);
}

std::vector<int8_t> make_uniform_small_i8_inputs(const std::vector<int64_t>& values) {
  std::vector<int8_t> packed(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    packed[index] = uniform_small_i64_to_i8(values[index]);
  }
  return packed;
}

std::vector<int8_t> make_uniform_small_i8_inputs(const std::vector<uint64_t>& values) {
  std::vector<int8_t> packed(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    packed[index] = uniform_small_u64_to_i8(values[index]);
  }
  return packed;
}

uint64_t benchmark_bound(const Args& args) {
  if (exact_wide_benchmark_semantics(args.semantics) || args.semantics == BenchSemantics::WrapU64Mod2_64 ||
      finite_benchmark_semantics(args.semantics) || args.bound_mode == BoundMode::PerTile) {
    return 0;
  }
  const uint64_t max_term = 16u * 16u;
  const auto u_k = static_cast<uint64_t>(args.k);
  if (u_k > std::numeric_limits<uint64_t>::max() / max_term) {
    usage_error("k is too large for the benchmark bound");
  }
  uint64_t bound = u_k * max_term;
  if (args.residue_chain_length > 1) {
    if (u_k > std::numeric_limits<uint64_t>::max() / 16u) {
      usage_error("k is too large for the bounded residue chain benchmark bound");
    }
    const uint64_t chain_multiplier = u_k * 16u;
    for (uint32_t chain_index = 1; chain_index < args.residue_chain_length; ++chain_index) {
      if (chain_multiplier != 0 && bound > std::numeric_limits<uint64_t>::max() / chain_multiplier) {
        usage_error("bounded residue chain benchmark bound exceeds uint64_t");
      }
      bound *= chain_multiplier;
    }
  }
  if (args.semantics == BenchSemantics::BoundedI64 &&
      bound > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    usage_error("bounded-i64 benchmark bound exceeds int64 output range");
  }
  return bound;
}

uint64_t checked_add_bound(uint64_t a, uint64_t b, const char* label) {
  if (a > std::numeric_limits<uint64_t>::max() - b) {
    usage_error(std::string(label) + " bound scan overflowed uint64_t");
  }
  return a + b;
}

uint64_t checked_mul_bound(uint64_t a, uint64_t b, const char* label) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    usage_error(std::string(label) + " bound scan overflowed uint64_t");
  }
  return a * b;
}

uint64_t magnitude_for_bound_scan(int64_t value) {
  if (value == std::numeric_limits<int64_t>::min()) {
    usage_error("bounded-i64 input scan cannot represent abs(INT64_MIN) in int64_t");
  }
  return value < 0 ? static_cast<uint64_t>(-value) : static_cast<uint64_t>(value);
}

uint64_t magnitude_for_bound_scan(uint64_t value) {
  return value;
}

struct GlobalBoundScanStats {
  uint64_t discovered_bound = 0;
  uint64_t candidate_row_sum_col_max = 0;
  uint64_t candidate_row_max_col_sum = 0;
  uint64_t row_abs_sum_max = 0;
  uint64_t row_abs_max = 0;
  uint64_t col_abs_sum_max = 0;
  uint64_t col_abs_max = 0;
  uint64_t zero_row_count = 0;
  uint64_t zero_col_count = 0;
};

template <typename T>
GlobalBoundScanStats compute_global_bound_scan(
    const Args& args,
    const std::vector<T>& A,
    const std::vector<T>& B,
    uint64_t static_bound) {
  GlobalBoundScanStats stats{};

  for (int64_t row = 0; row < args.m; ++row) {
    uint64_t sum = 0;
    uint64_t maximum = 0;
    for (int64_t kk = 0; kk < args.k; ++kk) {
      const uint64_t value = magnitude_for_bound_scan(A[row_major_index(row, kk, args.k, "A")]);
      sum = checked_add_bound(sum, value, "row absolute sum");
      maximum = std::max(maximum, value);
    }
    stats.row_abs_sum_max = std::max(stats.row_abs_sum_max, sum);
    stats.row_abs_max = std::max(stats.row_abs_max, maximum);
    if (sum == 0) {
      ++stats.zero_row_count;
    }
  }

  for (int64_t col = 0; col < args.n; ++col) {
    uint64_t sum = 0;
    uint64_t maximum = 0;
    for (int64_t kk = 0; kk < args.k; ++kk) {
      const uint64_t value = magnitude_for_bound_scan(B[row_major_index(kk, col, args.n, "B")]);
      sum = checked_add_bound(sum, value, "column absolute sum");
      maximum = std::max(maximum, value);
    }
    stats.col_abs_sum_max = std::max(stats.col_abs_sum_max, sum);
    stats.col_abs_max = std::max(stats.col_abs_max, maximum);
    if (sum == 0) {
      ++stats.zero_col_count;
    }
  }

  stats.candidate_row_sum_col_max =
      checked_mul_bound(stats.row_abs_sum_max, stats.col_abs_max, "row-sum/column-max");
  stats.candidate_row_max_col_sum =
      checked_mul_bound(stats.row_abs_max, stats.col_abs_sum_max, "row-max/column-sum");
  stats.discovered_bound = std::min(stats.candidate_row_sum_col_max, stats.candidate_row_max_col_sum);
  if (static_bound != 0) {
    stats.discovered_bound = std::min(stats.discovered_bound, static_bound);
  }
  return stats;
}

void record_global_bound_scan(BenchmarkResult& result, const GlobalBoundScanStats& stats, uint64_t static_bound) {
  result.global_bound_scan_available = true;
  result.static_bound = static_bound;
  result.discovered_global_bound = stats.discovered_bound;
  result.bound_candidate_row_sum_col_max = stats.candidate_row_sum_col_max;
  result.bound_candidate_row_max_col_sum = stats.candidate_row_max_col_sum;
  result.row_abs_sum_max = stats.row_abs_sum_max;
  result.row_abs_max = stats.row_abs_max;
  result.col_abs_sum_max = stats.col_abs_sum_max;
  result.col_abs_max = stats.col_abs_max;
  result.zero_row_count = stats.zero_row_count;
  result.zero_col_count = stats.zero_col_count;
}

template <typename T>
uint64_t resolve_bounded_global_bound(
    const Args& args,
    const std::vector<T>& A,
    const std::vector<T>& B,
    uint64_t static_bound,
    BenchmarkResult& result) {
  result.static_bound = static_bound;
  result.effective_bound = static_bound;
  result.effective_bound_available = true;
  if (args.bound_source != BoundSource::InputScan || args.bound_mode != BoundMode::Global) {
    return static_bound;
  }
  const auto scan_start = std::chrono::steady_clock::now();
  const GlobalBoundScanStats stats = compute_global_bound_scan(args, A, B, static_bound);
  const auto scan_end = std::chrono::steady_clock::now();
  result.global_bound_scan_us = elapsed_us(scan_start, scan_end);
  record_global_bound_scan(result, stats, static_bound);
  result.effective_bound = stats.discovered_bound;
  return result.effective_bound;
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

template <typename Fn>
void record_timed_tile_bounds(BenchmarkResult& result, Fn&& compute_bounds) {
  const auto scan_start = std::chrono::steady_clock::now();
  auto bounds = compute_bounds();
  const auto scan_end = std::chrono::steady_clock::now();
  result.tile_bound_scan_us = elapsed_us(scan_start, scan_end);
  result.tile_bound_scan_available = true;
  record_tile_bounds(result, std::move(bounds));
}

uint32_t benchmark_prefix(const Args& args) {
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 || finite_benchmark_semantics(args.semantics)) {
    return 0;
  }
  if (args.max_prefix_override != 0) {
    return args.max_prefix_override;
  }
  if (exact_wide_benchmark_semantics(args.semantics)) {
    return RNS8_MAX_SUPPORTED_PREFIX;
  }
  return RNS8_DEFAULT_BOUNDED_PREFIX;
}

bool fixed_requested_prefix_policy(const Args& args) {
  return args.prefix_policy == PrefixPolicy::FixedRequested || args.residue_chain_length > 1;
}

const char* prefix_policy_name(const Args& args, const BenchmarkResult& result) {
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 || finite_benchmark_semantics(args.semantics)) {
    return "semantic_specific_no_rns_prefix";
  }
  if (args.bound_mode == BoundMode::PerTile && !fixed_requested_prefix_policy(args)) {
    return "per_tile_minimum";
  }
  if (args.residue_chain_length > 1 && args.prefix_policy != PrefixPolicy::FixedRequested) {
    return "fixed_requested_residue_chain";
  }
  if (fixed_requested_prefix_policy(args)) {
    return "fixed_requested";
  }
  if (result.schedule_info_available && result.schedule_info.max_selected_prefix < benchmark_prefix(args)) {
    return "minimum_proven";
  }
  return "minimum_proven";
}

uint32_t selected_execution_prefix(const Args& args, const BenchmarkResult& result) {
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 || finite_benchmark_semantics(args.semantics)) {
    return 0;
  }
  if (result.schedule_info_available && result.schedule_info.max_selected_prefix > 0) {
    return result.schedule_info.max_selected_prefix;
  }
  return benchmark_prefix(args);
}

rns8_matrix_desc matrix_desc(int64_t rows, int64_t cols, const Args& args, uint32_t max_prefix) {
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
  desc.max_prefix = max_prefix;
  return desc;
}

rns8_matrix_desc matrix_desc(int64_t rows, int64_t cols, const Args& args) {
  return matrix_desc(rows, cols, args, benchmark_prefix(args));
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
  desc.max_prefix = benchmark_prefix(args);
  desc.tile_m = args.tile_m;
  desc.tile_n = args.tile_n;
  if (fixed_requested_prefix_policy(args) &&
      (bounded_benchmark_semantics(args.semantics) || exact_wide_benchmark_semantics(args.semantics))) {
    desc.flags |= RNS8_PLAN_FORCE_FIXED_PREFIX;
  }
  if (finite_benchmark_semantics(args.semantics)) {
    desc.finite_modulus = args.finite_modulus;
  }
  if (args.bound_mode == BoundMode::PerTile) {
    if (!tile_bounds) {
      usage_error("per-tile bound mode requires generated tile bounds");
    }
    desc.bound = 0;
    desc.tile_bounds = tile_bounds->data();
    desc.tile_bounds_count = static_cast<uint64_t>(tile_bounds->size());
    if (bounded_benchmark_semantics(args.semantics) && !tile_bounds->empty()) {
      desc.flags |= RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
    }
  }
  return desc;
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

struct DeviceBuffer {
  int device_id = -1;
  void* ptr = nullptr;
  std::size_t bytes = 0;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    reset();
  }

  void allocate(int device, std::size_t requested_bytes, const char* label) {
    reset();
    device_id = device;
    bytes = requested_bytes;
    const rns8_status status = rns8::detail::hip_direct_allocate(device_id, requested_bytes, &ptr);
    if (status != RNS8_SUCCESS) {
      fail_status(label, status);
    }
  }

  void reset() {
    if (ptr) {
      const rns8_status status = rns8::detail::hip_direct_free(device_id, ptr);
      ptr = nullptr;
      bytes = 0;
      if (status != RNS8_SUCCESS) {
        std::cerr << "hip_direct_free: " << rns8_status_string(status) << "\n";
      }
    }
  }
};

template <typename T>
uint64_t vector_alu_workspace_bytes(const Args& args) {
  const std::size_t a_bytes = checked_bytes(checked_elements(args.m, args.k, "A"), sizeof(T), "A");
  const std::size_t b_bytes = checked_bytes(checked_elements(args.k, args.n, "B"), sizeof(T), "B");
  const std::size_t c_bytes = checked_bytes(checked_elements(args.m, args.n, "C"), sizeof(T), "C");
  const std::size_t status_bytes = sizeof(uint32_t);
  const std::size_t ab_bytes = checked_add_bytes(a_bytes, b_bytes, "vector workspace A+B");
  const std::size_t abc_bytes = checked_add_bytes(ab_bytes, c_bytes, "vector workspace A+B+C");
  const std::size_t total = checked_add_bytes(abc_bytes, status_bytes, "vector workspace A+B+C+status");
  return static_cast<uint64_t>(total);
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

double median(std::vector<uint64_t> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if ((values.size() & 1u) != 0u) {
    return static_cast<double>(values[middle]);
  }
  return (static_cast<double>(values[middle - 1]) + static_cast<double>(values[middle])) / 2.0;
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

std::vector<std::string> required_speedup_baselines(const Args& args, const char* selected_backend) {
  std::vector<std::string> baselines;
  const std::string selected = selected_backend ? selected_backend : "";
  switch (args.semantics) {
    case BenchSemantics::BoundedI64:
    case BenchSemantics::BoundedU64:
      baselines.push_back("same_contract_cpu_reference");
      if (selected != "hip-vector-alu-int64") {
        baselines.push_back("same_contract_direct_hip_vector_alu_int64");
      }
      if (selected != "hip-direct") {
        baselines.push_back("same_contract_direct_hip_correctness");
      }
      if (args.oneshot) {
        baselines.push_back("same_contract_direct_hip_persistent_rns");
      }
      break;
    case BenchSemantics::ExactWideSigned:
    case BenchSemantics::ExactWideUnsigned:
      baselines.push_back("same_contract_cpu_reference");
      if (selected != "hip-direct") {
        baselines.push_back("same_contract_direct_hip_correctness");
      }
      break;
    case BenchSemantics::WrapU64Mod2_64:
      baselines.push_back("same_contract_cpu_wrap64_byte_limb_reference");
      baselines.push_back("same_contract_direct_hip_wrap64_byte_gemm36");
      break;
    case BenchSemantics::FiniteRingU8:
    case BenchSemantics::FiniteFieldU8:
      baselines.push_back("same_contract_cpu_reference");
      if (selected != "hip-direct") {
        baselines.push_back("same_contract_direct_hip_correctness");
      }
      if (args.oneshot) {
        baselines.push_back("same_contract_direct_hip_persistent_finite_u8");
      }
      break;
  }
  return baselines;
}

void print_comparison_baseline(const Args& args, const rns8_device_info& info, const BenchmarkResult& result) {
  const bool performance_validated = result.backend_info.performance_validated != 0;
  const char* selected = selected_backend_name(args, info, &result);
  std::cout << "  \"comparison_baseline\": {\n";
  std::cout << "    \"status\": \""
            << (performance_validated ? "reviewed_release_same_contract_baseline"
                                      : "required_not_recorded")
            << "\",\n";
  std::cout << "    \"speedup_claimed\": false,\n";
  std::cout << "    \"selected_reference\": null,\n";
  std::cout << "    \"required_before_speedup_claim\": ";
  print_string_array(required_speedup_baselines(args, selected));
  std::cout << ",\n";
  std::cout << "    \"reason\": \"";
  if (performance_validated) {
    std::cout << "performance_validated=true from an exact reviewed release autotune cache hit for this plan key";
  } else {
    std::cout << "performance_validated=false; raw capture has not been promoted against same-contract CPU and GPU "
                 "baseline evidence";
  }
  std::cout << "\"\n";
  std::cout << "  },\n";
}

void print_single_u64_array(uint64_t value) {
  std::cout << "[" << value << "]";
}

void capture_schedule_info(rns8_plan* plan, BenchmarkResult& result) {
  result.schedule_info.struct_size = sizeof(result.schedule_info);
  result.schedule_info.abi_version = RNS8_ABI_VERSION;
  const auto start = std::chrono::steady_clock::now();
  rns8_status status = rns8_get_plan_schedule_info(plan, &result.schedule_info);
  if (status == RNS8_SUCCESS && result.schedule_info.tile_count != 0) {
    if (result.schedule_info.tile_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
      usage_error("plan tile schedule is too large to inspect for skip counters");
    }
    std::vector<rns8_plan_tile_schedule_entry> entries(
        static_cast<std::size_t>(result.schedule_info.tile_count));
    uint64_t written = 0;
    status = rns8_get_plan_tile_schedule(plan, entries.data(), entries.size(), &written);
    if (status == RNS8_SUCCESS && written != static_cast<uint64_t>(entries.size())) {
      usage_error("plan tile schedule query returned an unexpected entry count");
    }
    if (status == RNS8_SUCCESS) {
      for (const auto& entry : entries) {
        if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0) {
          ++result.zero_output_tile_count;
          result.zero_output_selected_residue_plane_count += entry.selected_prefix;
        }
      }
    }
  }
  const auto end = std::chrono::steady_clock::now();
  if (status != RNS8_SUCCESS) {
    fail_status("rns8_get_plan_schedule_info/rns8_get_plan_tile_schedule", status);
  }
  result.schedule_query_us = elapsed_us(start, end);
  result.schedule_info_available = true;
}

void capture_backend_info(rns8_plan* plan, BenchmarkResult& result) {
  result.backend_info.struct_size = sizeof(result.backend_info);
  result.backend_info.abi_version = RNS8_ABI_VERSION;
  const rns8_status status = rns8_get_plan_backend_info(plan, &result.backend_info);
  if (status != RNS8_SUCCESS) {
    fail_status("rns8_get_plan_backend_info", status);
  }
  result.backend_info_available = true;
  update_result_target_id_from_key(result);
}

void append_accumulator_key_fields(std::ostringstream& out, const rns8_plan_backend_info& info) {
  out << ";accumulator_type=" << info.accumulator_type
      << ";accumulator_signedness=" << info.accumulator_signedness
      << ";accumulator_modulus_policy=" << info.accumulator_modulus_policy
      << ";k_block_size=" << info.accumulator_k_block_size
      << ";k_block_cap=" << info.accumulator_k_block_cap;
}

void fill_vector_alu_accumulator_info(const Args& args, rns8_plan_backend_info& info) {
  info.accumulator_k_block_size = args.k > 0 ? static_cast<uint64_t>(args.k) : 0u;
  info.accumulator_k_block_cap = 0;
  info.accumulator_modulus = 0;
  info.accumulator_max_lhs_abs = 0;
  info.accumulator_max_rhs_abs = 0;
  info.accumulator_max_product = 0;
  info.accumulator_uses_int32_inner_product = 0;
  info.accumulator_safe_for_k_block = 1;
  set_backend_text(
      info.accumulator_input_domain,
      sizeof(info.accumulator_input_domain),
      args.semantics == BenchSemantics::BoundedI64 ? "native_i64_values" : "native_u64_values");
  set_backend_text(
      info.accumulator_signedness,
      sizeof(info.accumulator_signedness),
      args.semantics == BenchSemantics::BoundedI64 ? "signed_i64x_signed_i64" : "unsigned_u64x_unsigned_u64");
  set_backend_text(info.accumulator_type, sizeof(info.accumulator_type), "software_192bit_limb");
  set_backend_text(
      info.accumulator_modulus_policy,
      sizeof(info.accumulator_modulus_policy),
      "native_exact_integer_output");
  set_backend_text(
      info.accumulator_safety_status,
      sizeof(info.accumulator_safety_status),
      "exact_192bit_limb_no_int32_k_cap");
}

void fill_wrap64_rocwmma_candidate_accumulator_info(const Args& args, rns8_plan_backend_info& info) {
  info.accumulator_k_block_size = args.k > 0 ? static_cast<uint64_t>(args.k) : 0u;
  info.accumulator_k_block_cap = 32768u;
  info.accumulator_modulus = 0;
  info.accumulator_max_lhs_abs = 255u;
  info.accumulator_max_rhs_abs = 255u;
  info.accumulator_max_product = 255u * 255u;
  info.accumulator_uses_int32_inner_product = 1;
  info.accumulator_safe_for_k_block =
      info.accumulator_k_block_size > 0 && info.accumulator_k_block_size <= info.accumulator_k_block_cap &&
              info.accumulator_max_product <=
                  static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / info.accumulator_k_block_size
          ? 1u
          : 0u;
  set_backend_text(info.accumulator_input_domain, sizeof(info.accumulator_input_domain), "compact_u8_byte_limb_pairs");
  set_backend_text(info.accumulator_signedness, sizeof(info.accumulator_signedness), "unsigned_u8x_unsigned_u8");
  set_backend_text(info.accumulator_type, sizeof(info.accumulator_type), "int32_then_int64_diagonal");
  set_backend_text(
      info.accumulator_modulus_policy,
      sizeof(info.accumulator_modulus_policy),
      "mod_2_64_wraparound_byte_limb");
  set_backend_text(
      info.accumulator_safety_status,
      sizeof(info.accumulator_safety_status),
      info.accumulator_safe_for_k_block ? "safe_int32_byte_limb_gemm36_k_block"
                                        : "unsafe_int32_byte_limb_gemm36_k_block");
}

std::string bounded_oneshot_autotune_key(
    const Args& args,
    const BenchmarkResult& result,
    const char* kernel,
    const char* epilogue) {
  std::ostringstream out;
  const uint32_t selected_prefix = selected_execution_prefix(args, result);
  out << "backend=" << backend_name(result.backend_info.backend)
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";prefix=" << selected_prefix
      << ";requested_max_prefix=" << benchmark_prefix(args)
      << ";prefix_policy=" << prefix_policy_name(args, result)
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";groups=" << result.schedule_info.prefix_group_count
      << ";adaptive_prefix=" << result.schedule_info.adaptive_prefix_active
      << ";adaptive_skip=" << result.schedule_info.adaptive_skip_active
      << ";schedule_flags=" << result.schedule_info.flags
      << ";zero_output_tiles=" << result.zero_output_tile_count
      << ";execution=public_oneshot_transient_native_inputs";
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=" << epilogue;
  return out.str();
}

const char* bounded_oneshot_kernel(const Args& args) {
  return args.semantics == BenchSemantics::BoundedU64 &&
             args.m >= 512 && args.n >= 512 && args.k >= 512
      ? "direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2"
      : "direct_hip_prefix9_native_input_grouped_rns_gemm_v1";
}

std::string finite_oneshot_autotune_key(
    const Args& args,
    const BenchmarkResult& result,
    const char* kernel,
    const char* epilogue) {
  std::ostringstream out;
  out << "backend=" << backend_name(result.backend_info.backend)
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";finite_modulus=" << args.finite_modulus
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";execution=public_oneshot_transient_native_inputs";
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=" << epilogue;
  return out.str();
}

const char* finite_native_a_reuse_b_kernel(uint16_t modulus) {
  if (modulus == 256) {
    return "direct_hip_native_a_finite_u8_gemm_mod256_v1";
  }
  if (modulus == 255) {
    return "direct_hip_native_a_finite_u8_gemm_mod255_v1";
  }
  if (modulus == 251) {
    return "direct_hip_native_a_finite_u8_gemm_mod251_v1";
  }
  return "direct_hip_native_a_finite_u8_gemm_v1";
}

const char* bounded_native_a_reuse_b_kernel(const Args& args) {
  if (bounded_native_a_reuse_b_uniform_small_a(args)) {
    return "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_b_grouped_rns_gemm_v2";
  }
  return args.semantics == BenchSemantics::BoundedI64
      ? "direct_hip_native_a_i64_prefix9_reuse_b_grouped_rns_gemm_v1"
      : "direct_hip_native_a_u64_prefix9_reuse_b_grouped_rns_gemm_v1";
}

const char* bounded_native_a_reuse_b_epilogue(const Args& args) {
  return bounded_native_a_reuse_b_uniform_small_a(args)
      ? "uniform_small_i8_ab_resident_b_residue_then_crt_export"
      : "native_a_centered_resident_b_residue_then_crt_export";
}

const char* bounded_native_a_reuse_b_event_label(const Args& args) {
  return bounded_native_a_reuse_b_uniform_small_a(args)
      ? "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
      : "bounded_native_a_reuse_b_gemm_kernel_group";
}

const char* bounded_native_a_reuse_b_pack_h2d_label(const Args& args) {
  return bounded_native_a_reuse_b_uniform_small_a(args)
      ? "bounded_uniform_small_i8_a_h2d"
      : "bounded_native_a_h2d";
}

const char* bounded_native_a_reuse_b_b_h2d_label(const Args& args) {
  return bounded_native_a_reuse_b_uniform_small_a(args)
      ? "bounded_uniform_small_i8_b_h2d"
      : "bounded_native_b_h2d";
}

bool bounded_native_a_reuse_b_path(const Args& args, const BenchmarkResult& result) {
  return bounded_native_a_reuse_b_requested(args) &&
         result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT &&
         result.schedule_info_available &&
         result.schedule_info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.prefix_group_count == 1 &&
         !result.schedule_info.adaptive_prefix_active &&
         !result.schedule_info.adaptive_skip_active;
}

bool finite_native_a_reuse_b_path(const Args& args, const BenchmarkResult& result) {
  return finite_benchmark_semantics(args.semantics) && args.reuse_packed_b && !args.reuse_packed_a &&
         result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT;
}

std::string bounded_native_a_reuse_b_autotune_key(
    const Args& args,
    const BenchmarkResult& result,
    const char* kernel,
    const char* epilogue,
    uint64_t bound) {
  std::ostringstream out;
  out << "backend=" << backend_name(result.backend_info.backend)
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";bound=" << bound
      << ";input_profile=" << input_profile_name(args)
      << ";prefix=" << RNS8_DEFAULT_BOUNDED_PREFIX
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";groups=" << result.schedule_info.prefix_group_count
      << ";adaptive_prefix=" << result.schedule_info.adaptive_prefix_active
      << ";adaptive_skip=" << result.schedule_info.adaptive_skip_active
      << ";schedule_flags=" << result.schedule_info.flags
      << ";zero_output_tiles=" << result.zero_output_tile_count
      << ";execution=" << benchmark_execution_mode_name(args);
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=" << epilogue;
  return out.str();
}

std::string finite_native_a_reuse_b_autotune_key(
    const Args& args,
    const BenchmarkResult& result,
    const char* kernel,
    const char* epilogue) {
  std::ostringstream out;
  out << "backend=" << backend_name(result.backend_info.backend)
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";finite_modulus=" << args.finite_modulus
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";execution=transient_native_a_resident_b_reuse";
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=" << epilogue;
  return out.str();
}

void apply_bounded_oneshot_backend_metadata(const Args& args, BenchmarkResult& result) {
  if (!args.oneshot || !result.backend_info_available) {
    return;
  }
  result.backend_info.performance_validated = 0;
  if (result.backend_info.backend != RNS8_BACKEND_HIP_DIRECT ||
      args.bound_mode != BoundMode::Global ||
      selected_execution_prefix(args, result) != 9) {
    return;
  }
  const char* kernel = bounded_oneshot_kernel(args);
  const char* epilogue = "native_input_centered_residue_then_crt_export";
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "transient_native_inputs_to_resident_rns_output");
  const std::string key = bounded_oneshot_autotune_key(args, result, kernel, epilogue);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_finite_oneshot_backend_metadata(const Args& args, BenchmarkResult& result) {
  if (!args.oneshot || !finite_benchmark_semantics(args.semantics) || !result.backend_info_available) {
    return;
  }
  result.backend_info.performance_validated = 0;
  if (result.backend_info.backend != RNS8_BACKEND_HIP_DIRECT) {
    return;
  }
  const char* kernel = "direct_hip_native_finite_u8_gemm_v1";
  if (args.finite_modulus == 256) {
    kernel = "direct_hip_native_finite_u8_gemm_mod256_v1";
  } else if (args.finite_modulus == 255) {
    kernel = "direct_hip_native_finite_u8_gemm_mod255_v1";
  } else if (args.finite_modulus == 251) {
    kernel = "direct_hip_native_finite_u8_gemm_mod251_v1";
  }
  const char* epilogue = "native_u8_centered_residue_then_canonical_u8_export";
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "transient_native_u8_inputs_to_resident_finite_output");
  const std::string key = finite_oneshot_autotune_key(args, result, kernel, epilogue);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_bounded_native_a_reuse_b_backend_metadata(const Args& args, BenchmarkResult& result, uint64_t bound) {
  if (!bounded_native_a_reuse_b_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = bounded_native_a_reuse_b_kernel(args);
  const char* epilogue = bounded_native_a_reuse_b_epilogue(args);
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "transient_native_a_resident_rns_b_output");
  const std::string key = bounded_native_a_reuse_b_autotune_key(args, result, kernel, epilogue, bound);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_finite_native_a_reuse_b_backend_metadata(const Args& args, BenchmarkResult& result) {
  if (!finite_native_a_reuse_b_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = finite_native_a_reuse_b_kernel(args.finite_modulus);
  const char* epilogue = "native_a_centered_resident_b_residue_then_canonical_u8_export";
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "transient_native_u8_a_resident_finite_b_output");
  const std::string key = finite_native_a_reuse_b_autotune_key(args, result, kernel, epilogue);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

std::string vector_alu_autotune_key(const Args& args, const BenchmarkResult& result, const char* kernel) {
  std::ostringstream out;
  const uint32_t selected_prefix = selected_execution_prefix(args, result);
  out << "backend=hip-vector-alu-int64"
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";prefix=" << selected_prefix
      << ";requested_max_prefix=" << benchmark_prefix(args)
      << ";prefix_policy=" << prefix_policy_name(args, result)
      << ";tile_m=" << args.tile_m
      << ";tile_n=" << args.tile_n
      << ";groups=" << result.schedule_info.prefix_group_count
      << ";adaptive_prefix=" << result.schedule_info.adaptive_prefix_active
      << ";adaptive_skip=" << result.schedule_info.adaptive_skip_active
      << ";schedule_flags=" << result.schedule_info.flags
      << ";zero_output_tiles=" << result.zero_output_tile_count;
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kernel
      << ";epilogue=direct_int64_export";
  return out.str();
}

void fill_vector_alu_backend_info(const Args& args, BenchmarkResult& result, uint64_t workspace_bytes) {
  const bool signed_semantics = args.semantics == BenchSemantics::BoundedI64;
  const char* kernel = signed_semantics ? "hip_vector_alu_i64_exact_192b_v1" : "hip_vector_alu_u64_exact_192b_v1";
  result.backend_info = {};
  result.backend_info.struct_size = sizeof(result.backend_info);
  result.backend_info.abi_version = RNS8_ABI_VERSION;
  result.backend_info.backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
  result.backend_info.is_accelerator = 0;
  result.backend_info.is_correctness_backend = 1;
  result.backend_info.is_matrix_engine_backend = 0;
  result.backend_info.compiled_kernel_available = 1;
  result.backend_info.exact_differential_validated = 1;
  result.backend_info.performance_validated = 0;
  result.backend_info.workspace_required_bytes = workspace_bytes;
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.accelerator_library, sizeof(result.backend_info.accelerator_library), "HIP runtime");
  set_backend_text(result.backend_info.accelerator_version, sizeof(result.backend_info.accelerator_version), RNS8_CONFIGURED_HIP_SDK_OR_ROCM_VERSION);
  set_backend_text(result.backend_info.capability_status, sizeof(result.backend_info.capability_status), "benchmark_only_vector_alu_baseline");
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), "direct_int64_export");
  set_backend_text(result.backend_info.workspace_mode, sizeof(result.backend_info.workspace_mode), "benchmark_owned_device_buffers");
  set_backend_text(
      result.backend_info.isa_evidence,
      sizeof(result.backend_info.isa_evidence),
      "source_level_192bit_limb_accumulator_no_matrix_engine");
  fill_vector_alu_accumulator_info(args, result.backend_info);
  const std::string key = vector_alu_autotune_key(args, result, kernel);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
  result.backend_info_available = true;
}

std::size_t wrap64_compact_limb_bytes(int64_t rows, int64_t cols, const char* label) {
  return checked_bytes(checked_elements(rows, cols, label), sizeof(uint64_t), label);
}

uint64_t wrap64_rocwmma_candidate_workspace_bytes(const Args& args) {
  const std::size_t a_bytes = wrap64_compact_limb_bytes(args.m, args.k, "candidate A byte limbs");
  const std::size_t b_bytes = wrap64_compact_limb_bytes(args.k, args.n, "candidate B byte limbs");
  const std::size_t c_bytes = wrap64_compact_limb_bytes(args.m, args.n, "candidate C byte limbs");
  const std::size_t ab_bytes = checked_add_bytes(a_bytes, b_bytes, "wrap64 candidate A+B byte limbs");
  const std::size_t total = checked_add_bytes(ab_bytes, c_bytes, "wrap64 candidate A+B+C byte limbs");
  return static_cast<uint64_t>(total);
}

void fill_wrap64_rocwmma_candidate_schedule(const Args& args, BenchmarkResult& result) {
  result.schedule_source = kWrap64RocwmmaCandidateScheduleSource;
  result.schedule_info = {};
  result.schedule_info.struct_size = sizeof(result.schedule_info);
  result.schedule_info.abi_version = RNS8_ABI_VERSION;
  result.schedule_info.tile_m = kWrap64RocwmmaCandidateTile;
  result.schedule_info.tile_n = kWrap64RocwmmaCandidateTile;
  result.schedule_info.tile_rows = ceil_div_i64_u32(args.m, kWrap64RocwmmaCandidateTile);
  result.schedule_info.tile_cols = ceil_div_i64_u32(args.n, kWrap64RocwmmaCandidateTile);
  result.schedule_info.tile_count = result.schedule_info.tile_rows * result.schedule_info.tile_cols;
  result.schedule_info.min_required_prefix = 0;
  result.schedule_info.max_required_prefix = 0;
  result.schedule_info.min_selected_prefix = 0;
  result.schedule_info.max_selected_prefix = 0;
  result.schedule_info.prefix_group_count = 0;
  result.schedule_info.adaptive_prefix_active = 0;
  result.schedule_info.adaptive_skip_active = 0;
  result.schedule_info.range_bit_length = 0;
  result.schedule_info_available = true;
}

std::string wrap64_rocwmma_candidate_autotune_key(const Args& args, const BenchmarkResult& result) {
  std::ostringstream out;
  out << "backend=" << kWrap64RocwmmaCandidateRequestedBackend
      << ";target_id=" << benchmark_key_target_id(result)
      << ";semantics=" << semantics_name(args.semantics)
      << ";m=" << args.m
      << ";n=" << args.n
      << ";k=" << args.k
      << ";prefix=0"
      << ";tile_m=" << result.schedule_info.tile_m
      << ";tile_n=" << result.schedule_info.tile_n
      << ";groups=0;adaptive_prefix=0;adaptive_skip=0";
  append_accumulator_key_fields(out, result.backend_info);
  out << ";kernel=" << kWrap64RocwmmaCandidateSelectedKernel
      << ";epilogue=low64_wrap_export";
  return out.str();
}

void fill_wrap64_rocwmma_candidate_backend_info(const Args& args, BenchmarkResult& result, uint64_t workspace_bytes) {
  result.backend_info = {};
  result.backend_info.struct_size = sizeof(result.backend_info);
  result.backend_info.abi_version = RNS8_ABI_VERSION;
  result.backend_info.backend = RNS8_BACKEND_ROCWMMA;
  result.backend_info.is_accelerator = 1;
  result.backend_info.is_correctness_backend = 0;
  result.backend_info.is_matrix_engine_backend = 1;
  result.backend_info.compiled_kernel_available = 1;
  result.backend_info.exact_differential_validated = 1;
  result.backend_info.performance_validated = 0;
  result.backend_info.workspace_required_bytes = workspace_bytes;
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kWrap64RocwmmaCandidateSelectedKernel);
  set_backend_text(result.backend_info.accelerator_library, sizeof(result.backend_info.accelerator_library), "rocWMMA");
  set_backend_text(
      result.backend_info.accelerator_version,
      sizeof(result.backend_info.accelerator_version),
      "repo-local release/rocm-rel-7.1");
  set_backend_text(
      result.backend_info.capability_status,
      sizeof(result.backend_info.capability_status),
      "internal_wrap64_matrix_engine_candidate");
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), "low64_wrap_export");
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "benchmark_owned_compact_byte_limb_device_buffers");
  set_backend_text(
      result.backend_info.isa_evidence,
      sizeof(result.backend_info.isa_evidence),
      "rocwmma_wrap64_byte_gemm36_wmma_isa_gate_no_int32_global_store_no_divide");
  fill_wrap64_rocwmma_candidate_accumulator_info(args, result.backend_info);
  const std::string key = wrap64_rocwmma_candidate_autotune_key(args, result);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
  result.backend_info_available = true;
}

std::string prefix_event_label(const char* prefix, uint32_t index, const char* suffix) {
  char buffer[96];
  std::snprintf(buffer, sizeof(buffer), "%s%02u_%s", prefix, index, suffix);
  return std::string(buffer);
}

uint32_t gpu_event_selected_prefix_count(const Args& args, const BenchmarkResult& result) {
  if (finite_benchmark_semantics(args.semantics) || args.semantics == BenchSemantics::WrapU64Mod2_64 ||
      args.vector_alu_baseline || runtime_vector_alu_backend(args)) {
    return 0;
  }
  if (result.schedule_info_available && result.schedule_info.max_selected_prefix > 0) {
    return std::min<uint32_t>(result.schedule_info.max_selected_prefix, RNS8_DEFAULT_MODULUS_COUNT);
  }
  return std::min<uint32_t>(benchmark_prefix(args), RNS8_DEFAULT_MODULUS_COUNT);
}

void append_ck_deep_event_phases(std::vector<std::string>& phases, uint32_t prefix_count, bool zero_output_tiles) {
  phases.push_back("ck_pack_a_kernel");
  phases.push_back("ck_pack_b_kernel");
  phases.push_back("ck_wmma_cshuffle_matmul");
  phases.push_back("ck_copy_centered_kernel");
  phases.push_back("ck_add_centered_kernel");
  if (zero_output_tiles) {
    phases.push_back("ck_zero_output_tile_memset");
  }
  for (uint32_t index = 0; index < prefix_count; ++index) {
    phases.push_back(prefix_event_label("ck_prefix_", index, "pack_a"));
    phases.push_back(prefix_event_label("ck_prefix_", index, "pack_b"));
    phases.push_back(prefix_event_label("ck_prefix_", index, "matmul"));
    phases.push_back(prefix_event_label("ck_prefix_", index, "copy_centered"));
    phases.push_back(prefix_event_label("ck_prefix_", index, "add_centered"));
  }
}

void append_rocwmma_deep_event_phases(
    std::vector<std::string>& phases,
    uint32_t prefix_count,
    bool use_prepacked_b_cache,
    bool zero_output_tiles) {
  phases.push_back(use_prepacked_b_cache ? "rocwmma_pack_a_prepacked_b_kernel" : "rocwmma_pack_a_kernel");
  if (!use_prepacked_b_cache) {
    phases.push_back("rocwmma_pack_b_kernel");
  }
  phases.push_back(use_prepacked_b_cache ? "rocwmma_matmul_prepacked_b_kernel" : "rocwmma_matmul_kernel");
  if (zero_output_tiles) {
    phases.push_back("rocwmma_zero_output_tile_memset");
  }
  for (uint32_t index = 0; index < prefix_count; ++index) {
    if (use_prepacked_b_cache) {
      phases.push_back(prefix_event_label("rocwmma_prefix_", index, "pack_a_prepacked_b"));
      phases.push_back(prefix_event_label("rocwmma_prefix_", index, "matmul_prepacked_b"));
    } else {
      phases.push_back(prefix_event_label("rocwmma_prefix_", index, "pack_a"));
      phases.push_back(prefix_event_label("rocwmma_prefix_", index, "pack_b"));
      phases.push_back(prefix_event_label("rocwmma_prefix_", index, "matmul"));
    }
  }
}

void append_accelerator_deep_event_phases(
    std::vector<std::string>& phases,
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend,
    bool use_prepacked_b_cache) {
  if (finite_benchmark_semantics(args.semantics)) {
    return;
  }
  const uint32_t prefix_count = gpu_event_selected_prefix_count(args, result);
  const bool zero_output_tiles = result.zero_output_tile_count != 0;
  if (selected_backend == RNS8_BACKEND_CK) {
    append_ck_deep_event_phases(phases, prefix_count, zero_output_tiles);
  } else if (selected_backend == RNS8_BACKEND_ROCWMMA && !args.wrap64_rocwmma_candidate) {
    append_rocwmma_deep_event_phases(phases, prefix_count, use_prepacked_b_cache, zero_output_tiles);
  }
}

void append_hipblaslt_gemm_event_phases(std::vector<std::string>& phases, const Args& args) {
  if (expects_hipblaslt_pack_transpose_event(args)) {
    phases.push_back("hipblaslt_pack_transpose_centered");
  }
  phases.push_back("hipblaslt_int8_i32_matmul");
  phases.push_back("hipblaslt_i32_to_residue_reduce");
  phases.push_back("rns_gemm");
}

std::vector<std::string> gpu_event_phase_order(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend,
    bool use_prepacked_b_cache) {
  if (args.oneshot && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
    if (finite_benchmark_semantics(args.semantics)) {
      return {
          "oneshot_native_input_h2d",
          "finite_native_gemm_kernel",
          "rns_gemm",
          "finite_export_kernel",
          "finite_export_d2h",
          "crt_export",
          "oneshot_api_gpu"};
    }
    return {
        "oneshot_native_input_h2d",
        "rns_gemm_kernel_group",
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
        "oneshot_api_gpu"};
  }
  if (args.vector_alu_baseline || selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    const char* kernel =
        args.semantics == BenchSemantics::BoundedI64 ? "vector_alu_i64_kernel" : "vector_alu_u64_kernel";
    return {
        "vector_alu_pack_a_h2d",
        "vector_alu_pack_b_h2d",
        "pack",
        "vector_alu_status_memset",
        kernel,
        "rns_gemm",
        "vector_alu_status_d2h",
        "vector_alu_output_d2h",
        "crt_export"};
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    if (args.wrap64_rocwmma_candidate) {
      return {
          "pack_h2d",
          "pack_kernel",
          "pack",
          kWrap64RocwmmaCandidateEventLabel,
          "rns_gemm",
          "wrap64_export_kernel",
          "wrap64_export_d2h",
          "crt_export"};
    }
    return {
        "pack_h2d",
        "pack_kernel",
        "pack",
        "wrap64_byte_gemm36_tiled_2d_kernel",
        "rns_gemm",
        "wrap64_export_kernel",
        "wrap64_export_d2h",
        "crt_export"};
  }
  if (finite_benchmark_semantics(args.semantics)) {
    if (selected_backend == RNS8_BACKEND_HIPBLASLT) {
      std::vector<std::string> phases = {
          "finite_pack_h2d",
          "finite_pack_kernel",
          "pack",
      };
      append_hipblaslt_gemm_event_phases(phases, args);
      phases.push_back("finite_export_kernel");
      phases.push_back("finite_export_d2h");
      phases.push_back("crt_export");
      return phases;
    }
    std::vector<std::string> phases = {
        "finite_pack_h2d",
        "finite_pack_kernel",
        "pack",
        finite_native_a_reuse_b_path(args, result)
            ? "finite_native_a_gemm_kernel"
            : selected_backend == RNS8_BACKEND_HIP_DIRECT ? "finite_resident_gemm_kernel" : "rns_gemm_kernel_group",
    };
    append_accelerator_deep_event_phases(phases, args, result, selected_backend, use_prepacked_b_cache);
    phases.push_back("rns_gemm");
    phases.push_back("finite_export_kernel");
    phases.push_back("finite_export_d2h");
    phases.push_back("crt_export");
    return phases;
  }
  if (exact_wide_benchmark_semantics(args.semantics)) {
    if (selected_backend == RNS8_BACKEND_HIPBLASLT) {
      std::vector<std::string> phases = {
          "pack_h2d",
          "pack_kernel",
          "pack",
      };
      append_hipblaslt_gemm_event_phases(phases, args);
      if (residue_current_output_mode(args)) {
        return phases;
      }
      phases.push_back("exact_wide_export_status_memset");
      phases.push_back("exact_wide_export_kernel");
      phases.push_back("exact_wide_export_status_d2h");
      phases.push_back("exact_wide_export_d2h");
      phases.push_back("crt_export");
      return phases;
    }
    std::vector<std::string> phases = {
        "pack_h2d",
        "pack_kernel",
        "pack",
        use_prepacked_b_cache ? "rns_gemm_prepacked_b_kernel_group" : "rns_gemm_kernel_group",
    };
    append_accelerator_deep_event_phases(phases, args, result, selected_backend, use_prepacked_b_cache);
    phases.push_back("rns_gemm");
    if (residue_current_output_mode(args)) {
      return phases;
    }
    phases.push_back("exact_wide_export_status_memset");
    phases.push_back("exact_wide_export_kernel");
    phases.push_back("exact_wide_export_status_d2h");
    phases.push_back("exact_wide_export_d2h");
    phases.push_back("crt_export");
    return phases;
  }
  if (selected_backend == RNS8_BACKEND_HIPBLASLT) {
    std::vector<std::string> phases = {
        "pack_h2d",
        "pack_kernel",
        "pack",
    };
    append_hipblaslt_gemm_event_phases(phases, args);
    phases.push_back("crt_export_status_memset");
    phases.push_back("crt_export_kernel");
    phases.push_back("crt_export_status_d2h");
    phases.push_back("crt_export_d2h");
    phases.push_back("crt_export");
    return phases;
  }
  std::vector<std::string> phases = {
      "pack_h2d",
      "pack_kernel",
      "pack",
      bounded_native_a_reuse_b_path(args, result)
          ? bounded_native_a_reuse_b_event_label(args)
          : use_prepacked_b_cache ? "rns_gemm_prepacked_b_kernel_group" : "rns_gemm_kernel_group",
  };
  append_accelerator_deep_event_phases(phases, args, result, selected_backend, use_prepacked_b_cache);
  phases.push_back("rns_gemm");
  phases.push_back("crt_export_status_memset");
  phases.push_back("crt_export_kernel");
  phases.push_back("crt_export_status_d2h");
  phases.push_back("crt_export_d2h");
  phases.push_back("crt_export");
  return phases;
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

const std::vector<double>& gpu_event_values(const GpuEventSamples& events, const std::string& label) {
  static const std::vector<double> empty;
  const auto it = events.timings_us.find(label);
  return it == events.timings_us.end() ? empty : it->second;
}

void print_gpu_event_timings(const std::vector<std::string>& phase_order, const GpuEventSamples& events) {
  std::cout << "  \"gpu_event_timings_us\": {\n";
  for (std::size_t index = 0; index < phase_order.size(); ++index) {
    print_named_gpu_event_array(
        phase_order[index].c_str(), gpu_event_values(events, phase_order[index]), index + 1 != phase_order.size());
  }
  std::cout << "  },\n";
}

void print_gpu_event_timing_summary(const std::vector<std::string>& phase_order, const GpuEventSamples& events) {
  std::cout << "  \"gpu_event_timing_summary_us\": {\n";
  for (std::size_t index = 0; index < phase_order.size(); ++index) {
    print_gpu_event_summary(
        phase_order[index].c_str(), gpu_event_values(events, phase_order[index]), index + 1 != phase_order.size());
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

uint64_t checksum_u8(const std::vector<uint8_t>& values) {
  uint64_t checksum = 1469598103934665603ull;
  for (const uint8_t value : values) {
    mix_checksum(checksum, static_cast<uint64_t>(value));
  }
  return checksum;
}

bool backend_supports_gpu_event_capture(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_HIP_DIRECT || backend == RNS8_BACKEND_HIPBLASLT ||
         backend == RNS8_BACKEND_CK || backend == RNS8_BACKEND_ROCWMMA ||
         backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
}

rns8_backend_kind selected_backend_for_events(const Args& args, const BenchmarkResult& result) {
  return result.backend_info_available ? result.backend_info.backend : args.backend;
}

bool gpu_event_capture_requested(const Args& args, rns8_backend_kind selected_backend) {
  if (residue_current_output_mode(args)) {
    return false;
  }
  return backend_supports_gpu_event_capture(selected_backend);
}

void add_unavailable_reason(GpuEventSamples& events, const std::string& reason) {
  events.complete = false;
  if (std::find(events.unavailable_reasons.begin(), events.unavailable_reasons.end(), reason) ==
      events.unavailable_reasons.end()) {
    events.unavailable_reasons.push_back(reason);
  }
}

void push_gpu_event_value(GpuEventSamples& events, const std::string& label, double value) {
  if (events.complete) {
    events.timings_us[label].push_back(value);
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

double optional_event_label(
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    const char* label) {
  double total = 0.0;
  for (const auto& sample : samples) {
    if (sample.label == label) {
      total += sample.microseconds;
    }
  }
  return total;
}

bool sum_event_label_if_present(
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    const char* label,
    double& total) {
  bool found = false;
  total = 0.0;
  for (const auto& sample : samples) {
    if (sample.label == label) {
      found = true;
      total += sample.microseconds;
    }
  }
  return found;
}

std::vector<double> event_label_values(
    const std::vector<rns8::detail::hip_direct_timing_sample>& samples,
    const char* label) {
  std::vector<double> values;
  for (const auto& sample : samples) {
    if (sample.label == label) {
      values.push_back(sample.microseconds);
    }
  }
  return values;
}

void collect_pack_gpu_events(const Args& args, rns8_backend_kind selected_backend, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  if (args.vector_alu_baseline || selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    const auto h2d_samples = event_label_values(samples, "residue_h2d_sync");
    std::size_t h2d_index = 0;
    const auto operand_h2d = [&](bool reused, const char* label) -> double {
      if (reused) {
        return 0.0;
      }
      const bool has_fallback = h2d_index < h2d_samples.size();
      const double fallback = has_fallback ? h2d_samples[h2d_index] : 0.0;
      ++h2d_index;
      double explicit_total = 0.0;
      if (sum_event_label_if_present(samples, label, explicit_total)) {
        return explicit_total;
      }
      if (has_fallback) {
        return fallback;
      }
      add_unavailable_reason(
          events, std::string("pack missing backend HIP event label ") + label + " or residue_h2d_sync");
      return 0.0;
    };
    const double a_h2d = operand_h2d(args.reuse_packed_a, "vector_alu_pack_a_h2d");
    const double b_h2d = operand_h2d(args.reuse_packed_b, "vector_alu_pack_b_h2d");
    if (events.complete) {
      push_gpu_event_value(events, "vector_alu_pack_a_h2d", a_h2d);
      push_gpu_event_value(events, "vector_alu_pack_b_h2d", b_h2d);
      push_gpu_event_value(events, "pack", a_h2d + b_h2d);
    }
    return;
  }
  const bool finite = finite_benchmark_semantics(args.semantics);
  const char* h2d_label = finite ? "finite_pack_h2d" : "pack_h2d";
  const char* kernel_label = finite ? "finite_pack_kernel" : "pack_kernel";
  const double h2d = sum_event_label(events, samples, "pack", h2d_label);
  const double kernel = sum_event_label(events, samples, "pack", kernel_label);
  if (events.complete) {
    push_gpu_event_value(events, h2d_label, h2d);
    push_gpu_event_value(events, kernel_label, kernel);
    push_gpu_event_value(events, "pack", h2d + kernel);
  }
}

void record_reused_pack_gpu_events(const Args& args, rns8_backend_kind selected_backend, GpuEventSamples& events) {
  if (args.vector_alu_baseline || selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    push_gpu_event_value(events, "vector_alu_pack_a_h2d", 0.0);
    push_gpu_event_value(events, "vector_alu_pack_b_h2d", 0.0);
    push_gpu_event_value(events, "pack", 0.0);
    return;
  }
  const bool finite = finite_benchmark_semantics(args.semantics);
  if (events.complete) {
    push_gpu_event_value(events, finite ? "finite_pack_h2d" : "pack_h2d", 0.0);
    push_gpu_event_value(events, finite ? "finite_pack_kernel" : "pack_kernel", 0.0);
    push_gpu_event_value(events, "pack", 0.0);
  }
}

void collect_gemm_gpu_events(GpuEventSamples& events, bool use_prepacked_b_cache) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = use_prepacked_b_cache ? "rns_gemm_prepacked_b_kernel_group" : "rns_gemm_kernel_group";
  const double kernel_group = sum_event_label(events, samples, "rns_gemm", label);
  if (events.complete) {
    push_gpu_event_value(events, label, kernel_group);
    push_gpu_event_value(events, "rns_gemm", kernel_group);
  }
}

void collect_finite_direct_gemm_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double kernel = sum_event_label(events, samples, "rns_gemm", "finite_resident_gemm_kernel");
  if (events.complete) {
    push_gpu_event_value(events, "finite_resident_gemm_kernel", kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_finite_native_a_pack_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double h2d = sum_event_label(events, samples, "pack", "finite_native_a_h2d");
  if (events.complete) {
    push_gpu_event_value(events, "finite_pack_h2d", h2d);
    push_gpu_event_value(events, "finite_pack_kernel", 0.0);
    push_gpu_event_value(events, "pack", h2d);
  }
}

void collect_finite_native_a_gemm_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double kernel = sum_event_label(events, samples, "rns_gemm", "finite_native_a_gemm_kernel");
  if (events.complete) {
    push_gpu_event_value(events, "finite_native_a_gemm_kernel", kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_bounded_native_a_pack_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double h2d = sum_event_label(events, samples, "pack", bounded_native_a_reuse_b_pack_h2d_label(args));
  if (events.complete) {
    push_gpu_event_value(events, "pack_h2d", h2d);
    push_gpu_event_value(events, "pack_kernel", 0.0);
    push_gpu_event_value(events, "pack", h2d);
  }
}

void collect_bounded_native_a_gemm_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = bounded_native_a_reuse_b_event_label(args);
  const double kernel = sum_event_label(events, samples, "rns_gemm", label);
  if (events.complete) {
    push_gpu_event_value(events, label, kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_hipblaslt_gemm_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const bool expect_pack_transpose = expects_hipblaslt_pack_transpose_event(args);
  const double pack = expect_pack_transpose
      ? sum_event_label(events, samples, "rns_gemm", "hipblaslt_pack_transpose_centered")
      : 0.0;
  const double matmul = sum_event_label(events, samples, "rns_gemm", "hipblaslt_int8_i32_matmul");
  const double reduce = sum_event_label(events, samples, "rns_gemm", "hipblaslt_i32_to_residue_reduce");
  if (events.complete) {
    if (expect_pack_transpose) {
      push_gpu_event_value(events, "hipblaslt_pack_transpose_centered", pack);
    }
    push_gpu_event_value(events, "hipblaslt_int8_i32_matmul", matmul);
    push_gpu_event_value(events, "hipblaslt_i32_to_residue_reduce", reduce);
    push_gpu_event_value(events, "rns_gemm", pack + matmul + reduce);
  }
}

void collect_vector_alu_gemm_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* kernel_label =
      args.semantics == BenchSemantics::BoundedI64 ? "vector_alu_i64_kernel" : "vector_alu_u64_kernel";
  const double status_memset = optional_event_label(samples, "vector_alu_status_memset");
  const double kernel = sum_event_label(events, samples, "rns_gemm", kernel_label);
  const double status_d2h = optional_event_label(samples, "vector_alu_status_d2h");
  if (events.complete) {
    push_gpu_event_value(events, "vector_alu_status_memset", status_memset);
    push_gpu_event_value(events, kernel_label, kernel);
    if (!args.vector_alu_baseline) {
      push_gpu_event_value(events, "vector_alu_status_d2h", status_d2h);
    }
    push_gpu_event_value(events, "rns_gemm", status_memset + kernel + status_d2h);
  }
}

void collect_ck_deep_gemm_gpu_events(
    const Args& args,
    const BenchmarkResult& result,
    GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const bool all_tiles_zero =
      result.zero_output_tile_count != 0 && result.zero_output_tile_count == result.schedule_info.tile_count;
  const auto required_or_zero = [&](const char* label) {
    return all_tiles_zero ? optional_event_label(samples, label) : sum_event_label(events, samples, "rns_gemm", label);
  };
  const double pack_a = required_or_zero("ck_pack_a_kernel");
  const double pack_b = required_or_zero("ck_pack_b_kernel");
  const double matmul = required_or_zero("ck_wmma_cshuffle_matmul");
  const double copy = optional_event_label(samples, "ck_copy_centered_kernel");
  const double add = optional_event_label(samples, "ck_add_centered_kernel");
  const double zero_fill = result.zero_output_tile_count == 0
                               ? 0.0
                               : sum_event_label(events, samples, "rns_gemm", "ck_zero_output_tile_memset");
  if (events.complete) {
    push_gpu_event_value(events, "ck_pack_a_kernel", pack_a);
    push_gpu_event_value(events, "ck_pack_b_kernel", pack_b);
    push_gpu_event_value(events, "ck_wmma_cshuffle_matmul", matmul);
    push_gpu_event_value(events, "ck_copy_centered_kernel", copy);
    push_gpu_event_value(events, "ck_add_centered_kernel", add);
    if (result.zero_output_tile_count != 0) {
      push_gpu_event_value(events, "ck_zero_output_tile_memset", zero_fill);
    }
    const uint32_t prefix_count = gpu_event_selected_prefix_count(args, result);
    for (uint32_t index = 0; index < prefix_count; ++index) {
      const std::string pack_a_label = prefix_event_label("ck_prefix_", index, "pack_a");
      const std::string pack_b_label = prefix_event_label("ck_prefix_", index, "pack_b");
      const std::string matmul_label = prefix_event_label("ck_prefix_", index, "matmul");
      push_gpu_event_value(
          events,
          pack_a_label,
          all_tiles_zero ? optional_event_label(samples, pack_a_label.c_str())
                         : sum_event_label(events, samples, "rns_gemm", pack_a_label.c_str()));
      push_gpu_event_value(
          events,
          pack_b_label,
          all_tiles_zero ? optional_event_label(samples, pack_b_label.c_str())
                         : sum_event_label(events, samples, "rns_gemm", pack_b_label.c_str()));
      push_gpu_event_value(
          events,
          matmul_label,
          all_tiles_zero ? optional_event_label(samples, matmul_label.c_str())
                         : sum_event_label(events, samples, "rns_gemm", matmul_label.c_str()));
      const std::string copy_label = prefix_event_label("ck_prefix_", index, "copy_centered");
      const std::string add_label = prefix_event_label("ck_prefix_", index, "add_centered");
      push_gpu_event_value(events, copy_label, optional_event_label(samples, copy_label.c_str()));
      push_gpu_event_value(events, add_label, optional_event_label(samples, add_label.c_str()));
    }
  }
}

void collect_rocwmma_deep_gemm_gpu_events(
    const Args& args,
    const BenchmarkResult& result,
    GpuEventSamples& events,
    bool use_prepacked_b_cache) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* pack_a_label =
      use_prepacked_b_cache ? "rocwmma_pack_a_prepacked_b_kernel" : "rocwmma_pack_a_kernel";
  const char* matmul_label =
      use_prepacked_b_cache ? "rocwmma_matmul_prepacked_b_kernel" : "rocwmma_matmul_kernel";
  const bool all_tiles_zero =
      result.zero_output_tile_count != 0 && result.zero_output_tile_count == result.schedule_info.tile_count;
  const auto required_or_zero = [&](const char* label) {
    return all_tiles_zero ? optional_event_label(samples, label) : sum_event_label(events, samples, "rns_gemm", label);
  };
  const double pack_a = required_or_zero(pack_a_label);
  const double pack_b = use_prepacked_b_cache ? 0.0 : required_or_zero("rocwmma_pack_b_kernel");
  const double matmul = required_or_zero(matmul_label);
  const double zero_fill = result.zero_output_tile_count == 0
                               ? 0.0
                               : sum_event_label(events, samples, "rns_gemm", "rocwmma_zero_output_tile_memset");
  if (events.complete) {
    push_gpu_event_value(events, pack_a_label, pack_a);
    if (!use_prepacked_b_cache) {
      push_gpu_event_value(events, "rocwmma_pack_b_kernel", pack_b);
    }
    push_gpu_event_value(events, matmul_label, matmul);
    if (result.zero_output_tile_count != 0) {
      push_gpu_event_value(events, "rocwmma_zero_output_tile_memset", zero_fill);
    }
    const uint32_t prefix_count = gpu_event_selected_prefix_count(args, result);
    for (uint32_t index = 0; index < prefix_count; ++index) {
      if (use_prepacked_b_cache) {
        const std::string pack_a_prefix = prefix_event_label("rocwmma_prefix_", index, "pack_a_prepacked_b");
        const std::string matmul_prefix = prefix_event_label("rocwmma_prefix_", index, "matmul_prepacked_b");
        push_gpu_event_value(
            events,
            pack_a_prefix,
            all_tiles_zero ? optional_event_label(samples, pack_a_prefix.c_str())
                           : sum_event_label(events, samples, "rns_gemm", pack_a_prefix.c_str()));
        push_gpu_event_value(
            events,
            matmul_prefix,
            all_tiles_zero ? optional_event_label(samples, matmul_prefix.c_str())
                           : sum_event_label(events, samples, "rns_gemm", matmul_prefix.c_str()));
      } else {
        const std::string pack_a_prefix = prefix_event_label("rocwmma_prefix_", index, "pack_a");
        const std::string pack_b_prefix = prefix_event_label("rocwmma_prefix_", index, "pack_b");
        const std::string matmul_prefix = prefix_event_label("rocwmma_prefix_", index, "matmul");
        push_gpu_event_value(
            events,
            pack_a_prefix,
            all_tiles_zero ? optional_event_label(samples, pack_a_prefix.c_str())
                           : sum_event_label(events, samples, "rns_gemm", pack_a_prefix.c_str()));
        push_gpu_event_value(
            events,
            pack_b_prefix,
            all_tiles_zero ? optional_event_label(samples, pack_b_prefix.c_str())
                           : sum_event_label(events, samples, "rns_gemm", pack_b_prefix.c_str()));
        push_gpu_event_value(
            events,
            matmul_prefix,
            all_tiles_zero ? optional_event_label(samples, matmul_prefix.c_str())
                           : sum_event_label(events, samples, "rns_gemm", matmul_prefix.c_str()));
      }
    }
  }
}

void collect_rns_gemm_gpu_events(
    const Args& args,
    rns8_backend_kind selected_backend,
    const BenchmarkResult& result,
    GpuEventSamples& events,
    bool use_prepacked_b_cache = false) {
  if (selected_backend == RNS8_BACKEND_HIPBLASLT) {
    collect_hipblaslt_gemm_gpu_events(args, events);
  } else if (selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    collect_vector_alu_gemm_gpu_events(args, events);
  } else if (finite_benchmark_semantics(args.semantics) && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
    collect_finite_direct_gemm_gpu_events(events);
  } else {
    collect_gemm_gpu_events(events, use_prepacked_b_cache);
    if (finite_benchmark_semantics(args.semantics)) {
      return;
    }
    if (selected_backend == RNS8_BACKEND_CK) {
      collect_ck_deep_gemm_gpu_events(args, result, events);
    } else if (selected_backend == RNS8_BACKEND_ROCWMMA && !args.wrap64_rocwmma_candidate) {
      collect_rocwmma_deep_gemm_gpu_events(args, result, events, use_prepacked_b_cache);
    }
  }
}

void collect_wrap64_gemm_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = args.wrap64_rocwmma_candidate ? kWrap64RocwmmaCandidateEventLabel : "wrap64_byte_gemm36_tiled_2d_kernel";
  const double kernel = sum_event_label(events, samples, "rns_gemm", label);
  if (events.complete) {
    push_gpu_event_value(events, label, kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_export_gpu_events(const Args& args, rns8_backend_kind selected_backend, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  if (args.vector_alu_baseline || selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    const auto d2h_samples = event_label_values(samples, "residue_d2h_sync");
    std::size_t d2h_index = 0;
    double status_d2h = 0.0;
    if (args.vector_alu_baseline) {
      const bool has_status_fallback = d2h_index < d2h_samples.size();
      const double status_fallback = has_status_fallback ? d2h_samples[d2h_index] : 0.0;
      ++d2h_index;
      if (!sum_event_label_if_present(samples, "vector_alu_status_d2h", status_d2h) && has_status_fallback) {
        status_d2h = status_fallback;
      }
    }
    const bool has_output_fallback = d2h_index < d2h_samples.size();
    const double output_fallback = has_output_fallback ? d2h_samples[d2h_index] : 0.0;
    double output_d2h = 0.0;
    if (!sum_event_label_if_present(samples, "vector_alu_output_d2h", output_d2h)) {
      if (has_output_fallback) {
        output_d2h = output_fallback;
      } else {
        add_unavailable_reason(
            events, "crt_export missing backend HIP event label vector_alu_output_d2h or residue_d2h_sync");
      }
    }
    if (events.complete) {
      if (args.vector_alu_baseline) {
        push_gpu_event_value(events, "vector_alu_status_d2h", status_d2h);
      }
      push_gpu_event_value(events, "vector_alu_output_d2h", output_d2h);
      push_gpu_event_value(events, "crt_export", status_d2h + output_d2h);
    }
    return;
  }
  if (finite_benchmark_semantics(args.semantics)) {
    const double kernel = sum_event_label(events, samples, "crt_export", "finite_export_kernel");
    const double d2h = sum_event_label(events, samples, "crt_export", "finite_export_d2h");
    if (events.complete) {
      push_gpu_event_value(events, "finite_export_kernel", kernel);
      push_gpu_event_value(events, "finite_export_d2h", d2h);
      push_gpu_event_value(events, "crt_export", kernel + d2h);
    }
    return;
  }
  if (exact_wide_benchmark_semantics(args.semantics)) {
    const bool requires_status = exact_wide_export_status_check_required(args);
    const double status_memset =
        requires_status ? optional_event_label(samples, "exact_wide_export_status_memset") : 0.0;
    const double kernel = sum_event_label(events, samples, "crt_export", "exact_wide_export_kernel");
    const double status_d2h =
        requires_status ? sum_event_label(events, samples, "crt_export", "exact_wide_export_status_d2h") : 0.0;
    const double d2h = sum_event_label(events, samples, "crt_export", "exact_wide_export_d2h");
    if (events.complete) {
      push_gpu_event_value(events, "exact_wide_export_status_memset", status_memset);
      push_gpu_event_value(events, "exact_wide_export_kernel", kernel);
      push_gpu_event_value(events, "exact_wide_export_status_d2h", status_d2h);
      push_gpu_event_value(events, "exact_wide_export_d2h", d2h);
      push_gpu_event_value(events, "crt_export", status_memset + kernel + status_d2h + d2h);
    }
    return;
  }
  const double status_memset = optional_event_label(samples, "crt_export_status_memset");
  const double kernel = sum_event_label(events, samples, "crt_export", "crt_export_kernel");
  const double status_d2h = sum_event_label(events, samples, "crt_export", "crt_export_status_d2h");
  const double d2h = sum_event_label(events, samples, "crt_export", "crt_export_d2h");
  if (events.complete) {
    push_gpu_event_value(events, "crt_export_status_memset", status_memset);
    push_gpu_event_value(events, "crt_export_kernel", kernel);
    push_gpu_event_value(events, "crt_export_status_d2h", status_d2h);
    push_gpu_event_value(events, "crt_export_d2h", d2h);
    push_gpu_event_value(events, "crt_export", status_memset + kernel + status_d2h + d2h);
  }
}

void collect_bounded_oneshot_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double native_h2d = sum_event_label(events, samples, "oneshot", "residue_h2d_sync");
  const double gemm = sum_event_label(events, samples, "oneshot", "rns_gemm_kernel_group");
  const double status_memset = optional_event_label(samples, "crt_export_status_memset");
  const double export_kernel = sum_event_label(events, samples, "oneshot", "crt_export_kernel");
  const double status_d2h = sum_event_label(events, samples, "oneshot", "crt_export_status_d2h");
  const double output_d2h = sum_event_label(events, samples, "oneshot", "crt_export_d2h");
  const double export_total = status_memset + export_kernel + status_d2h + output_d2h;
  if (events.complete) {
    push_gpu_event_value(events, "oneshot_native_input_h2d", native_h2d);
    push_gpu_event_value(events, "rns_gemm_kernel_group", gemm);
    push_gpu_event_value(events, "rns_gemm", gemm);
    push_gpu_event_value(events, "crt_export_status_memset", status_memset);
    push_gpu_event_value(events, "crt_export_kernel", export_kernel);
    push_gpu_event_value(events, "crt_export_status_d2h", status_d2h);
    push_gpu_event_value(events, "crt_export_d2h", output_d2h);
    push_gpu_event_value(events, "crt_export", export_total);
    push_gpu_event_value(events, "oneshot_api_gpu", native_h2d + gemm + export_total);
  }
}

void collect_finite_oneshot_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double native_h2d = sum_event_label(events, samples, "oneshot", "residue_h2d_sync");
  const double gemm = sum_event_label(events, samples, "oneshot", "finite_native_gemm_kernel");
  const double export_kernel = sum_event_label(events, samples, "oneshot", "finite_export_kernel");
  const double output_d2h = sum_event_label(events, samples, "oneshot", "finite_export_d2h");
  const double export_total = export_kernel + output_d2h;
  if (events.complete) {
    push_gpu_event_value(events, "oneshot_native_input_h2d", native_h2d);
    push_gpu_event_value(events, "finite_native_gemm_kernel", gemm);
    push_gpu_event_value(events, "rns_gemm", gemm);
    push_gpu_event_value(events, "finite_export_kernel", export_kernel);
    push_gpu_event_value(events, "finite_export_d2h", output_d2h);
    push_gpu_event_value(events, "crt_export", export_total);
    push_gpu_event_value(events, "oneshot_api_gpu", native_h2d + gemm + export_total);
  }
}

void collect_wrap64_export_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double kernel = sum_event_label(events, samples, "crt_export", "wrap64_export_kernel");
  const double d2h = sum_event_label(events, samples, "crt_export", "wrap64_export_d2h");
  if (events.complete) {
    push_gpu_event_value(events, "wrap64_export_kernel", kernel);
    push_gpu_event_value(events, "wrap64_export_d2h", d2h);
    push_gpu_event_value(events, "crt_export", kernel + d2h);
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
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const std::size_t repeats = result.samples.pack_us.size();
  const bool use_prepacked_b_cache = uses_runtime_b_prepack_cache(result);
  const auto phase_order = gpu_event_phase_order(args, result, selected_backend, use_prepacked_b_cache);
  if (repeats == 0 || phase_order.empty()) {
    return false;
  }
  for (const auto& label : phase_order) {
    const auto it = result.gpu_events.timings_us.find(label);
    if (it == result.gpu_events.timings_us.end() || it->second.size() != repeats) {
      return false;
    }
  }
  for (const auto& item : result.gpu_events.timings_us) {
    if (std::find(phase_order.begin(), phase_order.end(), item.first) == phase_order.end()) {
      return false;
    }
  }
  return true;
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
  if (args.require_adaptive_execution && !schedule_uses_adaptive_work(result)) {
    usage_error("--require-adaptive-execution was requested but the plan is fixed-prefix");
  }
  if (!fixed_requested_prefix_policy(args) && !schedule_uses_adaptive_work(result)) {
    usage_error(
        "per-tile benchmark capture did not produce adaptive prefix grouping or prefix skipping; adjust shape, seed, or inputs");
  }
}

template <typename Fn>
rns8_status run_timed_status_operation(const char* label, Fn&& fn) {
  rns8_status status = RNS8_SUCCESS;
  const int code = rns8::detail::run_timed_device_code(label, [&]() {
    status = fn();
    return status == RNS8_SUCCESS ? 0 : 3;
  });
  if (code != 0 && status == RNS8_SUCCESS) {
    return RNS8_BACKEND_FAILURE;
  }
  return status;
}

void capture_vector_alu_schedule(rns8_context* ctx, const Args& args, uint64_t bound, BenchmarkResult& result) {
  result.target_id = benchmark_target_id_for_context(ctx, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);
  auto desc = gemm_desc(args, bound, &result.tile_bounds);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(vector-alu schedule)", status);
  capture_schedule_info(plan, result);
  enforce_per_tile_capture_contract(args, result);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);
  rns8_destroy_plan(plan);
}

BenchmarkResult run_vector_alu_i64(rns8_context* ctx, const Args& args, uint64_t bound) {
#if !RNS8_CONFIGURED_HIP_ENABLED
  (void)ctx;
  (void)args;
  (void)bound;
  usage_error("hip-vector-alu-int64-baseline requires a HIP-enabled benchmark build");
#else
  std::mt19937_64 rng(args.seed);
  std::vector<int64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<int64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<int64_t> C(checked_elements(args.m, args.n, "C"));
  fill_bounded_i64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  if (args.bound_mode == BoundMode::PerTile) {
    record_timed_tile_bounds(result, [&]() { return compute_i64_tile_bounds(args, A, B); });
  }
  capture_vector_alu_schedule(ctx, args, bound, result);

  const std::size_t a_bytes = checked_bytes(A.size(), sizeof(int64_t), "A");
  const std::size_t b_bytes = checked_bytes(B.size(), sizeof(int64_t), "B");
  const std::size_t c_bytes = checked_bytes(C.size(), sizeof(int64_t), "C");
  const std::size_t status_bytes = sizeof(uint32_t);
  DeviceBuffer d_a;
  DeviceBuffer d_b;
  DeviceBuffer d_c;
  DeviceBuffer d_status;
  const auto alloc_start = std::chrono::steady_clock::now();
  d_a.allocate(args.device_id, a_bytes, "hip_direct_allocate(vector A)");
  d_b.allocate(args.device_id, b_bytes, "hip_direct_allocate(vector B)");
  d_c.allocate(args.device_id, c_bytes, "hip_direct_allocate(vector C)");
  d_status.allocate(args.device_id, status_bytes, "hip_direct_allocate(vector status)");
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  fill_vector_alu_backend_info(args, result, vector_alu_workspace_bytes<int64_t>(args));
  result.gpu_events.requested = gpu_event_capture_requested(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);

  const auto run_iteration = [&](TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    begin_gpu_event_phase(collect_gpu_events);
    rns8_status status = run_timed_status_operation("vector_alu_pack_a_h2d", [&]() {
      return rns8::detail::hip_direct_copy_host_to_device(args.device_id, d_a.ptr, A.data(), a_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(vector A)", status);
    status = run_timed_status_operation("vector_alu_pack_b_h2d", [&]() {
      return rns8::detail::hip_direct_copy_host_to_device(args.device_id, d_b.ptr, B.data(), b_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(vector B)", status);
    if (collect_gpu_events) {
      collect_pack_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto pack_end = std::chrono::steady_clock::now();

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = run_timed_status_operation("vector_alu_status_memset", [&]() {
      return rns8::detail::hip_direct_zero(args.device_id, d_status.ptr, status_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_zero(vector status)", status);
    const int kernel_status = rns8::detail::run_timed_device_code("vector_alu_i64_kernel", [&]() {
      return rns8_bench_vector_i64_gemm_device(
          args.device_id,
          static_cast<const int64_t*>(d_a.ptr),
          static_cast<const int64_t*>(d_b.ptr),
          static_cast<int64_t*>(d_c.ptr),
          static_cast<uint32_t*>(d_status.ptr),
          args.m,
          args.n,
          args.k);
    });
    if (kernel_status != 0) fail_hip_runtime("rns8_bench_vector_i64_gemm_device", kernel_status);
    if (collect_gpu_events) {
      collect_rns_gemm_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    uint32_t range_status = 0;
    status = run_timed_status_operation("vector_alu_status_d2h", [&]() {
      return rns8::detail::hip_direct_copy_device_to_host(args.device_id, &range_status, d_status.ptr, status_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_device_to_host(vector status)", status);
    if (range_status != 0) {
      fail_status("hip-vector-alu-int64 range check", RNS8_RANGE_ERROR);
    }
    status = run_timed_status_operation("vector_alu_output_d2h", [&]() {
      return rns8::detail::hip_direct_copy_device_to_host(args.device_id, C.data(), d_c.ptr, c_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_device_to_host(vector C)", status);
    if (collect_gpu_events) {
      collect_export_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result.gpu_events);
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
    run_iteration(nullptr);
  }
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_i64(C);
  return result;
#endif
}

BenchmarkResult run_vector_alu_u64(rns8_context* ctx, const Args& args, uint64_t bound) {
#if !RNS8_CONFIGURED_HIP_ENABLED
  (void)ctx;
  (void)args;
  (void)bound;
  usage_error("hip-vector-alu-int64-baseline requires a HIP-enabled benchmark build");
#else
  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<uint64_t> C(checked_elements(args.m, args.n, "C"));
  fill_bounded_u64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  if (args.bound_mode == BoundMode::PerTile) {
    record_timed_tile_bounds(result, [&]() { return compute_u64_tile_bounds(args, A, B); });
  }
  capture_vector_alu_schedule(ctx, args, bound, result);

  const std::size_t a_bytes = checked_bytes(A.size(), sizeof(uint64_t), "A");
  const std::size_t b_bytes = checked_bytes(B.size(), sizeof(uint64_t), "B");
  const std::size_t c_bytes = checked_bytes(C.size(), sizeof(uint64_t), "C");
  const std::size_t status_bytes = sizeof(uint32_t);
  DeviceBuffer d_a;
  DeviceBuffer d_b;
  DeviceBuffer d_c;
  DeviceBuffer d_status;
  const auto alloc_start = std::chrono::steady_clock::now();
  d_a.allocate(args.device_id, a_bytes, "hip_direct_allocate(vector A)");
  d_b.allocate(args.device_id, b_bytes, "hip_direct_allocate(vector B)");
  d_c.allocate(args.device_id, c_bytes, "hip_direct_allocate(vector C)");
  d_status.allocate(args.device_id, status_bytes, "hip_direct_allocate(vector status)");
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  fill_vector_alu_backend_info(args, result, vector_alu_workspace_bytes<uint64_t>(args));
  result.gpu_events.requested = gpu_event_capture_requested(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64);

  const auto run_iteration = [&](TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    begin_gpu_event_phase(collect_gpu_events);
    rns8_status status = run_timed_status_operation("vector_alu_pack_a_h2d", [&]() {
      return rns8::detail::hip_direct_copy_host_to_device(args.device_id, d_a.ptr, A.data(), a_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(vector A)", status);
    status = run_timed_status_operation("vector_alu_pack_b_h2d", [&]() {
      return rns8::detail::hip_direct_copy_host_to_device(args.device_id, d_b.ptr, B.data(), b_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(vector B)", status);
    if (collect_gpu_events) {
      collect_pack_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto pack_end = std::chrono::steady_clock::now();

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = run_timed_status_operation("vector_alu_status_memset", [&]() {
      return rns8::detail::hip_direct_zero(args.device_id, d_status.ptr, status_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_zero(vector status)", status);
    const int kernel_status = rns8::detail::run_timed_device_code("vector_alu_u64_kernel", [&]() {
      return rns8_bench_vector_u64_gemm_device(
          args.device_id,
          static_cast<const uint64_t*>(d_a.ptr),
          static_cast<const uint64_t*>(d_b.ptr),
          static_cast<uint64_t*>(d_c.ptr),
          static_cast<uint32_t*>(d_status.ptr),
          args.m,
          args.n,
          args.k);
    });
    if (kernel_status != 0) fail_hip_runtime("rns8_bench_vector_u64_gemm_device", kernel_status);
    if (collect_gpu_events) {
      collect_rns_gemm_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    uint32_t range_status = 0;
    status = run_timed_status_operation("vector_alu_status_d2h", [&]() {
      return rns8::detail::hip_direct_copy_device_to_host(args.device_id, &range_status, d_status.ptr, status_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_device_to_host(vector status)", status);
    if (range_status != 0) {
      fail_status("hip-vector-alu-int64 range check", RNS8_RANGE_ERROR);
    }
    status = run_timed_status_operation("vector_alu_output_d2h", [&]() {
      return rns8::detail::hip_direct_copy_device_to_host(args.device_id, C.data(), d_c.ptr, c_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_device_to_host(vector C)", status);
    if (collect_gpu_events) {
      collect_export_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result.gpu_events);
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
    run_iteration(nullptr);
  }
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_u64(C);
  return result;
#endif
}

BenchmarkResult initialize_bounded_oneshot_result(
    rns8_context* ctx,
    const Args& args,
    uint64_t bound,
    const rns8_gemm_desc& desc,
    BenchmarkResult result = {}) {
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(oneshot metadata)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  apply_bounded_oneshot_backend_metadata(args, result);
  apply_finite_oneshot_backend_metadata(args, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  status = rns8_destroy_plan(plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_destroy_plan(oneshot metadata)", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);
  result.matrix_alloc_us = 0;
  (void)bound;
  return result;
}

BenchmarkResult run_bounded_i64_oneshot(rns8_context* ctx, const Args& args, uint64_t bound) {
  std::mt19937_64 rng(args.seed);
  std::vector<int64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<int64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<int64_t> C(checked_elements(args.m, args.n, "C"));
  fill_bounded_i64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  auto desc = gemm_desc(args, bound);
  result = initialize_bounded_oneshot_result(ctx, args, bound, desc, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);

  const auto run_iteration = [&](TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    const rns8_status status =
        rns8_gemm_i64_oneshot(ctx, &desc, A.data(), args.k, B.data(), args.n, C.data(), args.n);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_i64_oneshot", status);
    if (collect_gpu_events && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      collect_bounded_oneshot_gpu_events(result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto repeat_end = std::chrono::steady_clock::now();

    if (samples) {
      const uint64_t elapsed = elapsed_us(repeat_start, repeat_end);
      samples->pack_us.push_back(0);
      samples->gemm_us.push_back(elapsed);
      samples->export_us.push_back(0);
      samples->end_to_end_us.push_back(elapsed);
    }
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_iteration(nullptr);
  }
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_i64(C);
  return result;
}

BenchmarkResult run_bounded_u64_oneshot(rns8_context* ctx, const Args& args, uint64_t bound) {
  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<uint64_t> C(checked_elements(args.m, args.n, "C"));
  fill_bounded_u64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  auto desc = gemm_desc(args, bound);
  result = initialize_bounded_oneshot_result(ctx, args, bound, desc, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);

  const auto run_iteration = [&](TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    const rns8_status status =
        rns8_gemm_u64_oneshot(ctx, &desc, A.data(), args.k, B.data(), args.n, C.data(), args.n);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_u64_oneshot", status);
    if (collect_gpu_events && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      collect_bounded_oneshot_gpu_events(result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto repeat_end = std::chrono::steady_clock::now();

    if (samples) {
      const uint64_t elapsed = elapsed_us(repeat_start, repeat_end);
      samples->pack_us.push_back(0);
      samples->gemm_us.push_back(elapsed);
      samples->export_us.push_back(0);
      samples->end_to_end_us.push_back(elapsed);
    }
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_iteration(nullptr);
  }
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_u64(C);
  return result;
}

BenchmarkResult run_bounded_i64(rns8_context* ctx, const Args& args, uint64_t bound) {
  if (args.oneshot) {
    return run_bounded_i64_oneshot(ctx, args, bound);
  }
  if (args.vector_alu_baseline) {
    return run_vector_alu_i64(ctx, args, bound);
  }
  std::mt19937_64 rng(args.seed);
  std::vector<int64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<int64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<int64_t> C(checked_elements(args.m, args.n, "C"));
  fill_bounded_i64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  if (args.bound_mode == BoundMode::PerTile) {
    record_timed_tile_bounds(result, [&]() { return compute_i64_tile_bounds(args, A, B); });
  }
  auto desc = gemm_desc(args, bound, &result.tile_bounds);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  enforce_per_tile_capture_contract(args, result);
  apply_bounded_native_a_reuse_b_backend_metadata(args, result, bound);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const bool use_native_a_reuse_b = bounded_native_a_reuse_b_path(args, result);
  const bool use_uniform_small_i8_ab_reuse_b =
      use_native_a_reuse_b && bounded_native_a_reuse_b_uniform_small_a(args);
  std::vector<int8_t> uniform_small_a;
  std::vector<int8_t> uniform_small_b;
  if (use_uniform_small_i8_ab_reuse_b) {
    uniform_small_a = make_uniform_small_i8_inputs(A);
    uniform_small_b = make_uniform_small_i8_inputs(B);
  }
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  rns8_matrix* scratch_matrix = nullptr;
  rns8_prepack_cache* b_prepack_cache = nullptr;
  DeviceBuffer native_a;
  DeviceBuffer uniform_small_a_device;
  DeviceBuffer uniform_small_b_device;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  const std::size_t native_a_bytes =
      use_native_a_reuse_b && !use_uniform_small_i8_ab_reuse_b
          ? checked_bytes(A.size(), sizeof(int64_t), "native A")
          : 0;
  const std::size_t uniform_small_a_bytes =
      use_uniform_small_i8_ab_reuse_b ? checked_bytes(uniform_small_a.size(), sizeof(int8_t), "uniform-small A") : 0;
  const std::size_t uniform_small_b_bytes =
      use_uniform_small_i8_ab_reuse_b ? checked_bytes(uniform_small_b.size(), sizeof(int8_t), "uniform-small B") : 0;
  if (use_uniform_small_i8_ab_reuse_b) {
    uniform_small_a_device.allocate(args.device_id, uniform_small_a_bytes, "hip_direct_allocate(bounded i64 i8 A)");
    uniform_small_b_device.allocate(args.device_id, uniform_small_b_bytes, "hip_direct_allocate(bounded i64 i8 B)");
  } else if (use_native_a_reuse_b) {
    native_a.allocate(args.device_id, native_a_bytes, "hip_direct_allocate(bounded i64 native A)");
  } else {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  }
  if (!use_uniform_small_i8_ab_reuse_b) {
    status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  }
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  if (residue_current_output_mode(args)) {
    status = rns8_create_matrix(ctx, &c_desc, &scratch_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(chain scratch)", status);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_a_input = [&](uint64_t source_version) {
    if (use_native_a_reuse_b) {
      fail_status("rns8_pack_i64(A native-A reuse-B path)", RNS8_INVALID_ARGUMENT);
    }
    if (selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      status = run_timed_status_operation("vector_alu_pack_a_h2d", [&]() {
        return rns8_pack_i64(ctx, a_matrix, A.data(), args.k, source_version);
      });
    } else {
      status = rns8_pack_i64(ctx, a_matrix, A.data(), args.k, source_version);
    }
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(A)", status);
  };
  const auto pack_b_input = [&](uint64_t source_version) {
    if (use_uniform_small_i8_ab_reuse_b) {
      (void)source_version;
      status = run_timed_status_operation(bounded_native_a_reuse_b_b_h2d_label(args), [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(
            args.device_id, uniform_small_b_device.ptr, uniform_small_b.data(), uniform_small_b_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded i64 uniform-small B)", status);
      return;
    }
    if (selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      status = run_timed_status_operation("vector_alu_pack_b_h2d", [&]() {
        return rns8_pack_i64(ctx, b_matrix, B.data(), args.n, source_version);
      });
    } else {
      status = rns8_pack_i64(ctx, b_matrix, B.data(), args.n, source_version);
    }
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(B)", status);
  };
  if (args.reuse_packed_inputs) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    pack_preused_inputs(args, 1, pack_a_input, pack_b_input);
    if (!use_native_a_reuse_b && should_probe_reusable_b_prepack_cache(args) &&
        maybe_create_reusable_b_prepack_cache(ctx, plan, b_matrix, &b_prepack_cache)) {
      result.prepack_reuse_strategy = PrepackReuseStrategy::RocwmmaReusableBCache;
    }
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  rns8_matrix* latest_output_matrix = c_matrix;
  const bool chain_residue_output = residue_current_output_mode(args);

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    auto pack_end = pack_start;
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else if (use_native_a_reuse_b) {
      begin_gpu_event_phase(collect_gpu_events);
      status = run_timed_status_operation(bounded_native_a_reuse_b_pack_h2d_label(args), [&]() {
        return use_uniform_small_i8_ab_reuse_b
            ? rns8::detail::hip_direct_copy_host_to_device(
                  args.device_id, uniform_small_a_device.ptr, uniform_small_a.data(), uniform_small_a_bytes)
            : rns8::detail::hip_direct_copy_host_to_device(args.device_id, native_a.ptr, A.data(), native_a_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded i64 native A)", status);
      if (collect_gpu_events) {
        collect_bounded_native_a_pack_gpu_events(args, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    }

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    rns8_matrix* lhs_matrix = a_matrix;
    rns8_matrix* out_matrix = c_matrix;
    rns8_matrix* final_output_matrix = c_matrix;
    if (use_native_a_reuse_b) {
      if (use_uniform_small_i8_ab_reuse_b) {
        status = rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix(
            args.device_id,
            uniform_small_a_device.ptr,
            uniform_small_b_device.ptr,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix", status);
        }
      } else {
        status = rns8::detail::hip_direct_gemm_i64_native_a_resident_b_prefix9_matrix(
            args.device_id,
            native_a.ptr,
            b_matrix,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_gemm_i64_native_a_resident_b_prefix9_matrix", status);
      }
      final_output_matrix = c_matrix;
      if (collect_gpu_events) {
        collect_bounded_native_a_gemm_gpu_events(args, result.gpu_events);
      }
    } else {
      for (uint32_t chain_index = 0; chain_index < args.residue_chain_length; ++chain_index) {
        status = run_rns_gemm_with_optional_b_cache(
            ctx, plan, lhs_matrix, b_matrix, b_prepack_cache, out_matrix, workspace);
        if (status != RNS8_SUCCESS) {
          fail_status(b_prepack_cache ? "rns8_gemm_rns_prepacked_b" : "rns8_gemm_rns", status);
        }
        final_output_matrix = out_matrix;
        lhs_matrix = out_matrix;
        out_matrix = out_matrix == c_matrix ? scratch_matrix : c_matrix;
      }
      if (collect_gpu_events) {
        collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events, b_prepack_cache != nullptr);
      }
    }
    latest_output_matrix = final_output_matrix;
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    auto export_start = gemm_end;
    auto export_end = gemm_end;
    if (!chain_residue_output) {
      export_start = std::chrono::steady_clock::now();
      begin_gpu_event_phase(collect_gpu_events);
      status = rns8_export_i64(ctx, plan, c_matrix, C.data(), args.n);
      if (status != RNS8_SUCCESS) fail_status("rns8_export_i64", status);
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      export_end = std::chrono::steady_clock::now();
    }

    if (samples) {
      samples->pack_us.push_back(reuses_all_packed_inputs(args) ? 0 : elapsed_us(pack_start, pack_end));
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
  if (chain_residue_output) {
    status = rns8_export_i64(ctx, plan, latest_output_matrix, C.data(), args.n);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_i64(final chain checksum)", status);
  }
  result.checksum = checksum_i64(C);

  if (b_prepack_cache) {
    status = rns8_destroy_prepack_cache(b_prepack_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_prepack_cache(B)", status);
  }
  if (scratch_matrix) {
    rns8_destroy_matrix(scratch_matrix);
  }
  rns8_destroy_matrix(c_matrix);
  if (b_matrix) {
    rns8_destroy_matrix(b_matrix);
  }
  if (a_matrix) {
    rns8_destroy_matrix(a_matrix);
  }
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_bounded_u64(rns8_context* ctx, const Args& args, uint64_t bound) {
  if (args.oneshot) {
    return run_bounded_u64_oneshot(ctx, args, bound);
  }
  if (args.vector_alu_baseline) {
    return run_vector_alu_u64(ctx, args, bound);
  }
  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<uint64_t> C(checked_elements(args.m, args.n, "C"));
  fill_bounded_u64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  if (args.bound_mode == BoundMode::PerTile) {
    record_timed_tile_bounds(result, [&]() { return compute_u64_tile_bounds(args, A, B); });
  }
  auto desc = gemm_desc(args, bound, &result.tile_bounds);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  enforce_per_tile_capture_contract(args, result);
  apply_bounded_native_a_reuse_b_backend_metadata(args, result, bound);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const bool use_native_a_reuse_b = bounded_native_a_reuse_b_path(args, result);
  const bool use_uniform_small_i8_ab_reuse_b =
      use_native_a_reuse_b && bounded_native_a_reuse_b_uniform_small_a(args);
  std::vector<int8_t> uniform_small_a;
  std::vector<int8_t> uniform_small_b;
  if (use_uniform_small_i8_ab_reuse_b) {
    uniform_small_a = make_uniform_small_i8_inputs(A);
    uniform_small_b = make_uniform_small_i8_inputs(B);
  }
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  rns8_matrix* scratch_matrix = nullptr;
  rns8_prepack_cache* b_prepack_cache = nullptr;
  DeviceBuffer native_a;
  DeviceBuffer uniform_small_a_device;
  DeviceBuffer uniform_small_b_device;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  const std::size_t native_a_bytes =
      use_native_a_reuse_b && !use_uniform_small_i8_ab_reuse_b
          ? checked_bytes(A.size(), sizeof(uint64_t), "native A")
          : 0;
  const std::size_t uniform_small_a_bytes =
      use_uniform_small_i8_ab_reuse_b ? checked_bytes(uniform_small_a.size(), sizeof(int8_t), "uniform-small A") : 0;
  const std::size_t uniform_small_b_bytes =
      use_uniform_small_i8_ab_reuse_b ? checked_bytes(uniform_small_b.size(), sizeof(int8_t), "uniform-small B") : 0;
  if (use_uniform_small_i8_ab_reuse_b) {
    uniform_small_a_device.allocate(args.device_id, uniform_small_a_bytes, "hip_direct_allocate(bounded u64 i8 A)");
    uniform_small_b_device.allocate(args.device_id, uniform_small_b_bytes, "hip_direct_allocate(bounded u64 i8 B)");
  } else if (use_native_a_reuse_b) {
    native_a.allocate(args.device_id, native_a_bytes, "hip_direct_allocate(bounded u64 native A)");
  } else {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  }
  if (!use_uniform_small_i8_ab_reuse_b) {
    status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  }
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  if (residue_current_output_mode(args)) {
    status = rns8_create_matrix(ctx, &c_desc, &scratch_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(chain scratch)", status);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_a_input = [&](uint64_t source_version) {
    if (use_native_a_reuse_b) {
      fail_status("rns8_pack_u64(A native-A reuse-B path)", RNS8_INVALID_ARGUMENT);
    }
    if (selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      status = run_timed_status_operation("vector_alu_pack_a_h2d", [&]() {
        return rns8_pack_u64(ctx, a_matrix, A.data(), args.k, source_version);
      });
    } else {
      status = rns8_pack_u64(ctx, a_matrix, A.data(), args.k, source_version);
    }
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(A)", status);
  };
  const auto pack_b_input = [&](uint64_t source_version) {
    if (use_uniform_small_i8_ab_reuse_b) {
      (void)source_version;
      status = run_timed_status_operation(bounded_native_a_reuse_b_b_h2d_label(args), [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(
            args.device_id, uniform_small_b_device.ptr, uniform_small_b.data(), uniform_small_b_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded u64 uniform-small B)", status);
      return;
    }
    if (selected_backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      status = run_timed_status_operation("vector_alu_pack_b_h2d", [&]() {
        return rns8_pack_u64(ctx, b_matrix, B.data(), args.n, source_version);
      });
    } else {
      status = rns8_pack_u64(ctx, b_matrix, B.data(), args.n, source_version);
    }
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(B)", status);
  };
  if (args.reuse_packed_inputs) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    pack_preused_inputs(args, 1, pack_a_input, pack_b_input);
    if (!use_native_a_reuse_b && should_probe_reusable_b_prepack_cache(args) &&
        maybe_create_reusable_b_prepack_cache(ctx, plan, b_matrix, &b_prepack_cache)) {
      result.prepack_reuse_strategy = PrepackReuseStrategy::RocwmmaReusableBCache;
    }
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  rns8_matrix* latest_output_matrix = c_matrix;
  const bool chain_residue_output = residue_current_output_mode(args);

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    auto pack_end = pack_start;
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else if (use_native_a_reuse_b) {
      begin_gpu_event_phase(collect_gpu_events);
      status = run_timed_status_operation(bounded_native_a_reuse_b_pack_h2d_label(args), [&]() {
        return use_uniform_small_i8_ab_reuse_b
            ? rns8::detail::hip_direct_copy_host_to_device(
                  args.device_id, uniform_small_a_device.ptr, uniform_small_a.data(), uniform_small_a_bytes)
            : rns8::detail::hip_direct_copy_host_to_device(args.device_id, native_a.ptr, A.data(), native_a_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded u64 native A)", status);
      if (collect_gpu_events) {
        collect_bounded_native_a_pack_gpu_events(args, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    }

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    rns8_matrix* lhs_matrix = a_matrix;
    rns8_matrix* out_matrix = c_matrix;
    rns8_matrix* final_output_matrix = c_matrix;
    if (use_native_a_reuse_b) {
      if (use_uniform_small_i8_ab_reuse_b) {
        status = rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix(
            args.device_id,
            uniform_small_a_device.ptr,
            uniform_small_b_device.ptr,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix", status);
        }
      } else {
        status = rns8::detail::hip_direct_gemm_u64_native_a_resident_b_prefix9_matrix(
            args.device_id,
            native_a.ptr,
            b_matrix,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            source_version);
        if (status != RNS8_SUCCESS) fail_status("hip_direct_gemm_u64_native_a_resident_b_prefix9_matrix", status);
      }
      final_output_matrix = c_matrix;
      if (collect_gpu_events) {
        collect_bounded_native_a_gemm_gpu_events(args, result.gpu_events);
      }
    } else {
      for (uint32_t chain_index = 0; chain_index < args.residue_chain_length; ++chain_index) {
        status = run_rns_gemm_with_optional_b_cache(
            ctx, plan, lhs_matrix, b_matrix, b_prepack_cache, out_matrix, workspace);
        if (status != RNS8_SUCCESS) {
          fail_status(b_prepack_cache ? "rns8_gemm_rns_prepacked_b" : "rns8_gemm_rns", status);
        }
        final_output_matrix = out_matrix;
        lhs_matrix = out_matrix;
        out_matrix = out_matrix == c_matrix ? scratch_matrix : c_matrix;
      }
      if (collect_gpu_events) {
        collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events, b_prepack_cache != nullptr);
      }
    }
    latest_output_matrix = final_output_matrix;
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    auto export_start = gemm_end;
    auto export_end = gemm_end;
    if (!chain_residue_output) {
      export_start = std::chrono::steady_clock::now();
      begin_gpu_event_phase(collect_gpu_events);
      status = rns8_export_u64(ctx, plan, c_matrix, C.data(), args.n);
      if (status != RNS8_SUCCESS) fail_status("rns8_export_u64", status);
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      export_end = std::chrono::steady_clock::now();
    }

    if (samples) {
      samples->pack_us.push_back(reuses_all_packed_inputs(args) ? 0 : elapsed_us(pack_start, pack_end));
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
  if (chain_residue_output) {
    status = rns8_export_u64(ctx, plan, latest_output_matrix, C.data(), args.n);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_u64(final chain checksum)", status);
  }
  result.checksum = checksum_u64(C);

  if (b_prepack_cache) {
    status = rns8_destroy_prepack_cache(b_prepack_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_prepack_cache(B)", status);
  }
  if (scratch_matrix) {
    rns8_destroy_matrix(scratch_matrix);
  }
  rns8_destroy_matrix(c_matrix);
  if (b_matrix) {
    rns8_destroy_matrix(b_matrix);
  }
  if (a_matrix) {
    rns8_destroy_matrix(a_matrix);
  }
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_exact_wide_signed(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  std::uniform_int_distribution<int64_t> dist(-16, 16);
  std::vector<int64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<int64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<uint64_t> C(checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, "C"));
  for (auto& value : A) value = dist(rng);
  for (auto& value : B) value = dist(rng);

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  rns8_matrix* scratch_matrix = nullptr;
  rns8_prepack_cache* b_prepack_cache = nullptr;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  if (residue_current_output_mode(args)) {
    status = rns8_create_matrix(ctx, &c_desc, &scratch_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(chain scratch)", status);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_a_input = [&](uint64_t source_version) {
    status = rns8_pack_i64(ctx, a_matrix, A.data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(A)", status);
  };
  const auto pack_b_input = [&](uint64_t source_version) {
    status = rns8_pack_i64(ctx, b_matrix, B.data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(B)", status);
  };
  if (args.reuse_packed_inputs) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    pack_preused_inputs(args, 1, pack_a_input, pack_b_input);
    if (should_probe_reusable_b_prepack_cache(args) &&
        maybe_create_reusable_b_prepack_cache(ctx, plan, b_matrix, &b_prepack_cache)) {
      result.prepack_reuse_strategy = PrepackReuseStrategy::RocwmmaReusableBCache;
    }
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  rns8_matrix* latest_output_matrix = c_matrix;
  const bool chain_residue_output = residue_current_output_mode(args);

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    auto pack_end = pack_start;
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    }

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    rns8_matrix* lhs_matrix = a_matrix;
    rns8_matrix* out_matrix = c_matrix;
    rns8_matrix* final_output_matrix = c_matrix;
    for (uint32_t chain_index = 0; chain_index < args.residue_chain_length; ++chain_index) {
      status = run_rns_gemm_with_optional_b_cache(
          ctx, plan, lhs_matrix, b_matrix, b_prepack_cache, out_matrix, workspace);
      if (status != RNS8_SUCCESS) {
        fail_status(b_prepack_cache ? "rns8_gemm_rns_prepacked_b" : "rns8_gemm_rns", status);
      }
      final_output_matrix = out_matrix;
      lhs_matrix = out_matrix;
      out_matrix = out_matrix == c_matrix ? scratch_matrix : c_matrix;
    }
    latest_output_matrix = final_output_matrix;
    if (collect_gpu_events) {
      collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events, b_prepack_cache != nullptr);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    auto export_start = gemm_end;
    auto export_end = gemm_end;
    if (!chain_residue_output) {
      export_start = std::chrono::steady_clock::now();
      begin_gpu_event_phase(collect_gpu_events);
      status =
          rns8_export_exact_wide_signed_limbs(ctx, plan, c_matrix, C.data(), args.n, args.exact_wide_limb_count);
      if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_signed_limbs", status);
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      export_end = std::chrono::steady_clock::now();
    }

    if (samples) {
      samples->pack_us.push_back(reuses_all_packed_inputs(args) ? 0 : elapsed_us(pack_start, pack_end));
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
  if (chain_residue_output) {
    status = rns8_export_exact_wide_signed_limbs(
        ctx, plan, latest_output_matrix, C.data(), args.n, args.exact_wide_limb_count);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_signed_limbs(final chain checksum)", status);
  }
  result.checksum = checksum_u64(C);

  if (b_prepack_cache) {
    status = rns8_destroy_prepack_cache(b_prepack_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_prepack_cache(B)", status);
  }
  if (scratch_matrix) {
    rns8_destroy_matrix(scratch_matrix);
  }
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_exact_wide_unsigned(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  std::uniform_int_distribution<uint64_t> dist(0, 16);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<uint64_t> C(checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, "C"));
  for (auto& value : A) value = dist(rng);
  for (auto& value : B) value = dist(rng);

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  rns8_matrix* scratch_matrix = nullptr;
  rns8_prepack_cache* b_prepack_cache = nullptr;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  if (residue_current_output_mode(args)) {
    status = rns8_create_matrix(ctx, &c_desc, &scratch_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(chain scratch)", status);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_a_input = [&](uint64_t source_version) {
    status = rns8_pack_u64(ctx, a_matrix, A.data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(A)", status);
  };
  const auto pack_b_input = [&](uint64_t source_version) {
    status = rns8_pack_u64(ctx, b_matrix, B.data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(B)", status);
  };
  if (args.reuse_packed_inputs) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    pack_preused_inputs(args, 1, pack_a_input, pack_b_input);
    if (should_probe_reusable_b_prepack_cache(args) &&
        maybe_create_reusable_b_prepack_cache(ctx, plan, b_matrix, &b_prepack_cache)) {
      result.prepack_reuse_strategy = PrepackReuseStrategy::RocwmmaReusableBCache;
    }
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  rns8_matrix* latest_output_matrix = c_matrix;
  const bool chain_residue_output = residue_current_output_mode(args);

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    auto pack_end = pack_start;
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    }

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    rns8_matrix* lhs_matrix = a_matrix;
    rns8_matrix* out_matrix = c_matrix;
    rns8_matrix* final_output_matrix = c_matrix;
    for (uint32_t chain_index = 0; chain_index < args.residue_chain_length; ++chain_index) {
      status = run_rns_gemm_with_optional_b_cache(
          ctx, plan, lhs_matrix, b_matrix, b_prepack_cache, out_matrix, workspace);
      if (status != RNS8_SUCCESS) {
        fail_status(b_prepack_cache ? "rns8_gemm_rns_prepacked_b" : "rns8_gemm_rns", status);
      }
      final_output_matrix = out_matrix;
      lhs_matrix = out_matrix;
      out_matrix = out_matrix == c_matrix ? scratch_matrix : c_matrix;
    }
    latest_output_matrix = final_output_matrix;
    if (collect_gpu_events) {
      collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events, b_prepack_cache != nullptr);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    auto export_start = gemm_end;
    auto export_end = gemm_end;
    if (!chain_residue_output) {
      export_start = std::chrono::steady_clock::now();
      begin_gpu_event_phase(collect_gpu_events);
      status =
          rns8_export_exact_wide_unsigned_limbs(ctx, plan, c_matrix, C.data(), args.n, args.exact_wide_limb_count);
      if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_unsigned_limbs", status);
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      export_end = std::chrono::steady_clock::now();
    }

    if (samples) {
      samples->pack_us.push_back(reuses_all_packed_inputs(args) ? 0 : elapsed_us(pack_start, pack_end));
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
  if (chain_residue_output) {
    status = rns8_export_exact_wide_unsigned_limbs(
        ctx, plan, latest_output_matrix, C.data(), args.n, args.exact_wide_limb_count);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_unsigned_limbs(final chain checksum)", status);
  }
  result.checksum = checksum_u64(C);

  if (b_prepack_cache) {
    status = rns8_destroy_prepack_cache(b_prepack_cache);
    if (status != RNS8_SUCCESS) fail_status("rns8_destroy_prepack_cache(B)", status);
  }
  if (scratch_matrix) {
    rns8_destroy_matrix(scratch_matrix);
  }
  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_finite_u8_oneshot(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  const uint32_t high = args.finite_modulus == 256 ? 255u : static_cast<uint32_t>(args.finite_modulus - 1u);
  std::uniform_int_distribution<uint32_t> dist(0, high);
  std::vector<uint8_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint8_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<uint8_t> C(checked_elements(args.m, args.n, "C"));
  for (auto& value : A) value = static_cast<uint8_t>(dist(rng));
  for (auto& value : B) value = static_cast<uint8_t>(dist(rng));

  auto desc = gemm_desc(args, 0);
  BenchmarkResult result = initialize_bounded_oneshot_result(ctx, args, 0, desc);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);

  const auto run_iteration = [&](TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    rns8_status status = RNS8_SUCCESS;
    if (args.semantics == BenchSemantics::FiniteFieldU8) {
      status = rns8_gemm_finite_field_u8_oneshot(
          ctx, &desc, args.finite_modulus, A.data(), args.k, B.data(), args.n, C.data(), args.n);
    } else {
      status = rns8_gemm_finite_ring_u8_oneshot(
          ctx, &desc, args.finite_modulus, A.data(), args.k, B.data(), args.n, C.data(), args.n);
    }
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_finite_u8_oneshot", status);
    if (collect_gpu_events && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      collect_finite_oneshot_gpu_events(result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto repeat_end = std::chrono::steady_clock::now();

    if (samples) {
      const uint64_t elapsed = elapsed_us(repeat_start, repeat_end);
      samples->pack_us.push_back(0);
      samples->gemm_us.push_back(elapsed);
      samples->export_us.push_back(0);
      samples->end_to_end_us.push_back(elapsed);
    }
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_iteration(nullptr);
  }
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_u8(C);
  return result;
}

BenchmarkResult run_finite_u8(rns8_context* ctx, const Args& args, uint64_t bound) {
  if (args.oneshot) {
    return run_finite_u8_oneshot(ctx, args, bound);
  }
  (void)bound;
  std::mt19937_64 rng(args.seed);
  const uint32_t high = args.finite_modulus == 256 ? 255u : static_cast<uint32_t>(args.finite_modulus - 1u);
  std::uniform_int_distribution<uint32_t> dist(0, high);
  std::vector<uint8_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint8_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<uint8_t> C(checked_elements(args.m, args.n, "C"));
  for (auto& value : A) value = static_cast<uint8_t>(dist(rng));
  for (auto& value : B) value = static_cast<uint8_t>(dist(rng));

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  apply_finite_native_a_reuse_b_backend_metadata(args, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const bool use_native_a_reuse_b = finite_native_a_reuse_b_path(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  DeviceBuffer native_a;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  const std::size_t native_a_bytes = use_native_a_reuse_b ? checked_bytes(A.size(), sizeof(uint8_t), "native A") : 0;
  if (use_native_a_reuse_b) {
    native_a.allocate(args.device_id, native_a_bytes, "hip_direct_allocate(finite native A)");
  } else {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  }
  status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_a_input = [&](uint64_t source_version) {
    if (use_native_a_reuse_b) {
      fail_status("rns8_pack_finite_u8(A native-A reuse-B path)", RNS8_INVALID_ARGUMENT);
    }
    status = rns8_pack_finite_u8(ctx, a_matrix, args.finite_modulus, A.data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_finite_u8(A)", status);
  };
  const auto pack_b_input = [&](uint64_t source_version) {
    status = rns8_pack_finite_u8(ctx, b_matrix, args.finite_modulus, B.data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_finite_u8(B)", status);
  };
  if (args.reuse_packed_inputs) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    pack_preused_inputs(args, 1, pack_a_input, pack_b_input);
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    auto pack_end = pack_start;
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else if (use_native_a_reuse_b) {
      begin_gpu_event_phase(collect_gpu_events);
      status = run_timed_status_operation("finite_native_a_h2d", [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(args.device_id, native_a.ptr, A.data(), native_a_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(finite native A)", status);
      if (collect_gpu_events) {
        collect_finite_native_a_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    }

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    if (use_native_a_reuse_b) {
      status = rns8::detail::hip_direct_gemm_finite_u8_native_a_resident_b_matrix(
          args.device_id,
          native_a.ptr,
          b_matrix,
          c_matrix,
          args.m,
          args.n,
          args.k,
          args.k,
          args.finite_modulus,
          source_version);
      if (status != RNS8_SUCCESS) fail_status("hip_direct_gemm_finite_u8_native_a_resident_b_matrix", status);
      if (collect_gpu_events) {
        collect_finite_native_a_gemm_gpu_events(result.gpu_events);
      }
    } else {
      status = rns8_gemm_finite_u8(ctx, plan, args.finite_modulus, a_matrix, b_matrix, c_matrix, workspace);
      if (status != RNS8_SUCCESS) fail_status("rns8_gemm_finite_u8", status);
      if (collect_gpu_events) {
        collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events);
      }
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_export_finite_u8(ctx, plan, args.finite_modulus, c_matrix, C.data(), args.n);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_finite_u8", status);
    if (collect_gpu_events) {
      collect_export_gpu_events(args, selected_backend, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto export_end = std::chrono::steady_clock::now();

    if (samples) {
      samples->pack_us.push_back(reuses_all_packed_inputs(args) ? 0 : elapsed_us(pack_start, pack_end));
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
  result.checksum = checksum_u8(C);

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  if (a_matrix) {
    rns8_destroy_matrix(a_matrix);
  }
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_wrap_u64_rocwmma_candidate(rns8_context* ctx, const Args& args, uint64_t bound) {
#if !RNS8_CONFIGURED_HIP_ENABLED
  (void)ctx;
  (void)args;
  (void)bound;
  usage_error("rocwmma-wrap64-candidate requires a HIP-enabled benchmark build");
#else
  (void)ctx;
  (void)bound;
  if (args.k > kWrap64RocwmmaCandidateMaxK) {
    usage_error("rocwmma-wrap64-candidate currently supports K <= 32768");
  }

  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<uint64_t> C(checked_elements(args.m, args.n, "C"));
  for (auto& value : A) value = rng();
  for (auto& value : B) value = rng();

  BenchmarkResult result{};
  result.target_id = benchmark_target_id_for_context(ctx, RNS8_BACKEND_ROCWMMA);
  const auto plan_start = std::chrono::steady_clock::now();
  fill_wrap64_rocwmma_candidate_schedule(args, result);
  fill_wrap64_rocwmma_candidate_backend_info(args, result, wrap64_rocwmma_candidate_workspace_bytes(args));
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);
  result.schedule_query_us = 0;
  result.gpu_events.requested = gpu_event_capture_requested(args, RNS8_BACKEND_ROCWMMA);

  const std::size_t a_limb_bytes = wrap64_compact_limb_bytes(args.m, args.k, "candidate A byte limbs");
  const std::size_t b_limb_bytes = wrap64_compact_limb_bytes(args.k, args.n, "candidate B byte limbs");
  const std::size_t c_limb_bytes = wrap64_compact_limb_bytes(args.m, args.n, "candidate C byte limbs");
  DeviceBuffer a_limbs;
  DeviceBuffer b_limbs;
  DeviceBuffer c_limbs;
  DeviceBuffer upload_buffer;
  DeviceBuffer export_buffer;
  upload_buffer.device_id = args.device_id;
  export_buffer.device_id = args.device_id;
  const auto alloc_start = std::chrono::steady_clock::now();
  a_limbs.allocate(args.device_id, a_limb_bytes, "hip_direct_allocate(wrap64 candidate A limbs)");
  b_limbs.allocate(args.device_id, b_limb_bytes, "hip_direct_allocate(wrap64 candidate B limbs)");
  c_limbs.allocate(args.device_id, c_limb_bytes, "hip_direct_allocate(wrap64 candidate C limbs)");
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  rns8_status status = RNS8_SUCCESS;
  const auto pack_a_input = [&](uint64_t) {
    status = rns8::detail::wrap64_hip_pack_u64_device(
        args.device_id, A.data(), &upload_buffer.ptr, &upload_buffer.bytes, a_limbs.ptr, args.m, args.k, args.k);
    if (status != RNS8_SUCCESS) fail_status("wrap64_hip_pack_u64_device(A)", status);
  };
  const auto pack_b_input = [&](uint64_t) {
    status = rns8::detail::wrap64_hip_pack_u64_device(
        args.device_id, B.data(), &upload_buffer.ptr, &upload_buffer.bytes, b_limbs.ptr, args.k, args.n, args.n);
    if (status != RNS8_SUCCESS) fail_status("wrap64_hip_pack_u64_device(B)", status);
  };
  if (args.reuse_packed_inputs) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    pack_preused_inputs(args, 1, pack_a_input, pack_b_input);
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  const auto run_iteration = [&](TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    auto pack_end = pack_start;
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, RNS8_BACKEND_ROCWMMA, result.gpu_events);
      }
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, 1, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, RNS8_BACKEND_ROCWMMA, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    }

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8::detail::rocwmma_wrap64_gemm_byte_limbs_candidate_device(
        args.device_id, a_limbs.ptr, b_limbs.ptr, c_limbs.ptr, args.m, args.n, args.k);
    if (status != RNS8_SUCCESS) fail_status("rocwmma_wrap64_gemm_byte_limbs_candidate_device", status);
    if (collect_gpu_events) {
      collect_wrap64_gemm_gpu_events(args, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8::detail::wrap64_hip_export_u64_device(
        args.device_id, c_limbs.ptr, &export_buffer.ptr, &export_buffer.bytes, args.m, args.n, C.data(), args.n);
    if (status != RNS8_SUCCESS) fail_status("wrap64_hip_export_u64_device(C)", status);
    if (collect_gpu_events) {
      collect_wrap64_export_gpu_events(result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto export_end = std::chrono::steady_clock::now();

    if (samples) {
      samples->pack_us.push_back(reuses_all_packed_inputs(args) ? 0 : elapsed_us(pack_start, pack_end));
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_iteration(nullptr);
  }
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_u64(C);
  return result;
#endif
}

BenchmarkResult run_wrap_u64(rns8_context* ctx, const Args& args, uint64_t bound) {
  if (args.wrap64_rocwmma_candidate) {
    return run_wrap_u64_rocwmma_candidate(ctx, args, bound);
  }
  (void)bound;
  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  std::vector<uint64_t> C(checked_elements(args.m, args.n, "C"));
  for (auto& value : A) value = rng();
  for (auto& value : B) value = rng();

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_a_input = [&](uint64_t source_version) {
    status = rns8_pack_u64(ctx, a_matrix, A.data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(A)", status);
  };
  const auto pack_b_input = [&](uint64_t source_version) {
    status = rns8_pack_u64(ctx, b_matrix, B.data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(B)", status);
  };
  if (args.reuse_packed_inputs) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    pack_preused_inputs(args, 1, pack_a_input, pack_b_input);
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    auto pack_end = pack_start;
    if (reuses_all_packed_inputs(args)) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    }

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_gemm_wrap_u64(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_wrap_u64", status);
    if (collect_gpu_events) {
      collect_wrap64_gemm_gpu_events(args, result.gpu_events);
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
      samples->pack_us.push_back(reuses_all_packed_inputs(args) ? 0 : elapsed_us(pack_start, pack_end));
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

const char* benchmark_name(const Args& args) {
  if (args.oneshot) {
    return finite_benchmark_semantics(args.semantics) ? "rns8_finite_u8_public_oneshot"
                                                      : "rns8_bounded_gemm_public_oneshot";
  }
  if (args.vector_alu_baseline) {
    return "rns8_bounded_gemm_hip_vector_alu_int64_baseline";
  }
  if (runtime_vector_alu_backend(args)) {
    return "rns8_bounded_gemm_hip_vector_alu_int64_runtime";
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return "rns8_wrap_u64_persistent_byte_limb";
  }
  if (finite_benchmark_semantics(args.semantics)) {
    return "rns8_finite_u8_persistent_residue";
  }
  if (exact_wide_benchmark_semantics(args.semantics)) {
    return "rns8_exact_wide_persistent_rns";
  }
  return "rns8_bounded_gemm_persistent_rns";
}

const char* epilogue_type(const Args& args) {
  if (args.vector_alu_baseline || runtime_vector_alu_backend(args)) {
    return "direct_int64_export";
  }
  if (finite_benchmark_semantics(args.semantics)) {
    return "canonical_u8_export";
  }
  if (residue_current_output_mode(args)) {
    return "residue_current_rns_output";
  }
  if (args.semantics == BenchSemantics::ExactWideSigned) {
    return "exact_wide_signed_limb_export";
  }
  if (args.semantics == BenchSemantics::ExactWideUnsigned) {
    return "exact_wide_unsigned_limb_export";
  }
  return args.semantics == BenchSemantics::WrapU64Mod2_64 ? "low64_wrap_export" : "crt_export";
}

const char* residue_output_mode_name(const Args& args) {
  return residue_current_output_mode(args) ? "residue_current_rns" : "host_export";
}

const char* exact_wide_export_status_check_name(const Args& args) {
  if (!exact_wide_benchmark_semantics(args.semantics)) {
    return nullptr;
  }
  return exact_wide_export_status_check_required(args) ? "required_for_range_check"
                                                      : "elided_full_width_device_reconstruction";
}

const char* packed_layout_version(const Args& args) {
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return "byte_limb_v1";
  }
  if (runtime_vector_alu_backend(args)) {
    return args.semantics == BenchSemantics::BoundedI64 ? "native_i64_rowmajor_v1" : "native_u64_rowmajor_v1";
  }
  return nullptr;
}

const char* input_distribution(const Args& args) {
  if (args.input_profile == InputProfile::AdaptiveBands) {
    return args.semantics == BenchSemantics::BoundedI64 ? "signed_adaptive_bands_-16_16"
                                                        : "unsigned_adaptive_bands_0_16";
  }
  switch (args.semantics) {
    case BenchSemantics::BoundedI64:
    case BenchSemantics::ExactWideSigned:
      return "signed_uniform_-16_16";
    case BenchSemantics::BoundedU64:
    case BenchSemantics::ExactWideUnsigned:
      return "unsigned_uniform_0_16";
    case BenchSemantics::WrapU64Mod2_64:
      return "unsigned_rng_u64_full_range";
    case BenchSemantics::FiniteRingU8:
    case BenchSemantics::FiniteFieldU8:
      return "u8_uniform_0_modulus_minus_1";
  }
  return "unknown";
}

const char* tile_bound_pattern(const Args& args) {
  switch (args.semantics) {
    case BenchSemantics::BoundedI64:
      return "exact_output_tile_max_abs_v1";
    case BenchSemantics::BoundedU64:
      return "exact_output_tile_max_unsigned_v1";
    case BenchSemantics::ExactWideSigned:
    case BenchSemantics::ExactWideUnsigned:
    case BenchSemantics::WrapU64Mod2_64:
    case BenchSemantics::FiniteRingU8:
    case BenchSemantics::FiniteFieldU8:
      return "none";
  }
  return "unknown";
}

bool adaptive_execution_applied(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result) {
  const rns8_backend_kind selected_backend =
      result.backend_info_available ? result.backend_info.backend : info.backend;
  return args.bound_mode == BoundMode::PerTile &&
         (selected_backend == RNS8_BACKEND_CPU_REFERENCE || selected_backend == RNS8_BACKEND_HIP_DIRECT ||
          selected_backend == RNS8_BACKEND_CK || selected_backend == RNS8_BACKEND_ROCWMMA) &&
         schedule_uses_adaptive_work(result);
}

const char* selected_kernel_name(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result) {
  if (result.backend_info_available && result.backend_info.selected_kernel[0] != '\0') {
    return result.backend_info.selected_kernel;
  }
  if (adaptive_execution_applied(args, info, result)) {
    return "direct_hip_tiled_active_prefix_rns_gemm_v2";
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 && info.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "direct_hip_wrap64_byte_gemm36_tiled_2d_v3";
  }
  return nullptr;
}

int64_t benchmark_k_block_size(const Args& args, const BenchmarkResult& result) {
  if (result.backend_info_available && result.backend_info.accumulator_k_block_size > 0 &&
      result.backend_info.accumulator_k_block_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return static_cast<int64_t>(result.backend_info.accumulator_k_block_size);
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return args.k;
  }
  return std::min<int64_t>(args.k, RNS8_SAFE_INT32_K_BLOCK);
}

std::string schedule_hash_string(const Args& args, const BenchmarkResult& result) {
  std::ostringstream out;
  out << "tile_rows=" << result.schedule_info.tile_rows << ";tile_cols=" << result.schedule_info.tile_cols
      << ";selected_prefix=" << selected_execution_prefix(args, result)
      << ";requested_max_prefix=" << benchmark_prefix(args)
      << ";prefix_policy=" << prefix_policy_name(args, result)
      << ";groups=" << result.schedule_info.prefix_group_count
      << ";adaptive_prefix=" << result.schedule_info.adaptive_prefix_active
      << ";adaptive_skip=" << result.schedule_info.adaptive_skip_active
      << ";tile_bound_hash=" << (args.bound_mode == BoundMode::PerTile ? result.tile_bound_hash : 0);
  return out.str();
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
  const uint32_t selected_prefix = selected_execution_prefix(args, result);
  const bool global_bound_scan_available = result.global_bound_scan_available;
  const uint32_t residue_planes_skipped = prefix > selected_prefix ? prefix - selected_prefix : 0;
  const double residue_plane_skip_fraction =
      prefix == 0 ? 0.0 : static_cast<double>(residue_planes_skipped) / static_cast<double>(prefix);
  const bool adaptive_applied = adaptive_execution_applied(args, info, result);
  const bool per_modulus_estimate_applicable =
      selected_prefix > 0 && args.bound_mode != BoundMode::PerTile && !adaptive_applied && !args.oneshot &&
      !args.vector_alu_baseline && !runtime_vector_alu_backend(args);
  const double avg_per_modulus_gemm_estimate_us =
      per_modulus_estimate_applicable ? avg_gemm_us / static_cast<double>(selected_prefix) : avg_gemm_us;
  const bool gpu_events_available = gpu_event_timing_available(args, result);
  const rns8_backend_kind selected_backend_kind = selected_backend_for_events(args, result);
  const bool use_prepacked_b_cache = uses_runtime_b_prepack_cache(result);
  const bool chain_residue_output = residue_current_output_mode(args);
  const auto event_phase_order =
      gpu_events_available ? gpu_event_phase_order(args, result, selected_backend_kind, use_prepacked_b_cache)
                           : std::vector<std::string>{};
  const bool wrap64_rocwmma_candidate_events = gpu_events_available && args.wrap64_rocwmma_candidate;
  const bool wrap64_hip_events =
      gpu_events_available && args.semantics == BenchSemantics::WrapU64Mod2_64 && !args.wrap64_rocwmma_candidate;
  const bool adaptive_hip_events = gpu_events_available && adaptive_applied;
  const char* selected_backend = selected_backend_name(args, info, &result);
  const std::string selected_backend_string(selected_backend);
  const bool hipblaslt_events = gpu_events_available && selected_backend_string == "hipblaslt";
  const bool accelerator_events =
      gpu_events_available && (selected_backend_string == "ck" || selected_backend_string == "rocwmma");
  const bool finite_accelerator_operation_group_events =
      accelerator_events && finite_benchmark_semantics(args.semantics);
  const bool accelerator_deep_kernel_events = accelerator_events && !finite_accelerator_operation_group_events;
  const bool vector_alu_events = gpu_events_available && selected_backend_string == "hip-vector-alu-int64";
  const bool oneshot_hip_events =
      gpu_events_available && args.oneshot && selected_backend_kind == RNS8_BACKEND_HIP_DIRECT;
  const char* gpu_event_reason = "backend_has_no_gpu_event_hooks";
  const char* gpu_event_status = "not_requested_for_selected_backend";
  const char* gpu_event_scope = "null";
  const char* gpu_event_caveat = "null";
  if (gpu_events_available) {
    gpu_event_status = "available";
    gpu_event_scope = "\"direct_hip_default_stream_backend_operation_groups\"";
    gpu_event_caveat =
        "\"HIP event timings record backend default-stream operation groups only; host wall-clock timings remain "
        "required for CPU scheduling overhead, API dispatch, allocations, and any synchronous host-side copy overhead "
        "not represented on the HIP stream\"";
    if (oneshot_hip_events) {
      gpu_event_reason = "captured_by_direct_hip_oneshot_api_hooks";
      gpu_event_scope = "\"direct_hip_oneshot_default_stream_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record the public one-shot API's native input H2D copies, direct-HIP GEMM "
          "kernel group, and logical export operation groups; host wall-clock timings remain required for plan creation, "
          "transient allocations, API dispatch, CPU scheduling, teardown, and synchronous host-side overhead not "
          "represented on the HIP stream\"";
    } else if (hipblaslt_events) {
      gpu_event_reason = "captured_by_hipblaslt_backend_hooks";
      gpu_event_scope = "\"hipblaslt_baseline_default_stream_backend_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record hipBLASLt baseline default-stream operation groups only; host wall-clock "
          "timings remain required for API dispatch, descriptor setup, allocations, and synchronous host-side "
          "overhead not represented on the HIP stream\"";
    } else if (wrap64_rocwmma_candidate_events) {
      gpu_event_reason = "captured_by_internal_rocwmma_wrap64_candidate_hooks";
      gpu_event_scope = "\"rocwmma_wrap64_byte_gemm36_candidate_default_stream_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record direct-HIP byte-limb pack/export operation groups plus one internal rocWMMA "
          "wrap64 byte-GEMM36 candidate operation group; host wall-clock timings remain required for CPU scheduling "
          "overhead, API dispatch, allocations, and synchronous host-side overhead not represented on the HIP stream\"";
    } else if (wrap64_hip_events) {
      gpu_event_reason = "captured_by_direct_hip_backend_hooks";
      gpu_event_scope = "\"direct_hip_wrap64_byte_gemm36_default_stream_backend_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record backend default-stream operation groups only; wrap64 uses a tiled byte-limb "
          "GEMM kernel and current rns_gemm/crt_export aggregate phase labels; host wall-clock timings remain "
          "required for CPU scheduling overhead, API dispatch, allocations, and synchronous host-side overhead not "
          "represented on the HIP stream\"";
    } else if (finite_accelerator_operation_group_events) {
      gpu_event_reason = "captured_by_accelerator_backend_operation_group_hooks";
      gpu_event_scope = "\"accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export\"";
      gpu_event_caveat =
          "\"HIP event timings record direct-HIP finite-u8 pack/export operation groups plus one accelerator GEMM "
          "operation group; host wall-clock timings remain required for API dispatch, CPU scheduling, allocations, "
          "and synchronous host-side overhead not represented on the HIP stream\"";
    } else if (accelerator_deep_kernel_events) {
      gpu_event_reason = "captured_by_accelerator_backend_deep_kernel_hooks";
      gpu_event_scope = "\"accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export\"";
      gpu_event_caveat =
          "\"HIP event timings record direct-HIP pack/export operation groups plus accelerator GEMM group, aggregate "
          "kernel-class labels, and selected-prefix labels, with synchronized host fallback only if a platform event "
          "read fails; host wall-clock timings remain required for API dispatch, CPU scheduling, allocations, and "
          "synchronous host-side overhead not represented on the HIP stream\"";
    } else if (vector_alu_events) {
      gpu_event_reason = "captured_by_vector_alu_native_backend_hooks";
      gpu_event_scope = "\"vector_alu_default_stream_native_int64_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record benchmark/API vector-ALU native-buffer operation groups; host wall-clock "
          "timings remain required for CPU staging, range checks, API dispatch, allocations, and synchronous "
          "host-side overhead not represented on the HIP stream\"";
    } else if (adaptive_hip_events) {
      gpu_event_reason = "captured_by_direct_hip_backend_hooks";
      gpu_event_scope = "\"direct_hip_bounded_adaptive_default_stream_backend_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record backend default-stream operation groups only; adaptive bounded captures "
          "aggregate all selected-prefix tile launches and tiled export kernels rather than exposing per-tile or "
          "per-prefix timings; host wall-clock timings remain required for scheduling overhead, API dispatch, "
          "allocations, and synchronous host-side overhead not represented on the HIP stream\"";
    } else {
      gpu_event_reason = "captured_by_direct_hip_backend_hooks";
    }
  } else if (chain_residue_output) {
    gpu_event_reason = "not_supported_for_residue_current_chain_mode";
    gpu_event_status = "not_requested_for_residue_current_chain_mode";
  } else if (result.gpu_events.requested) {
    gpu_event_reason = "backend_event_capture_incomplete";
    gpu_event_status = "unavailable_missing_expected_events";
  }

  std::cout << "{\n";
  std::cout << "  \"schema_version\": " << kBenchmarkSchemaVersion << ",\n";
  std::cout << "  \"benchmark\": \"" << benchmark_name(args) << "\",\n";
  std::cout << "  \"benchmark_execution_mode\": \"" << benchmark_execution_mode_name(args) << "\",\n";
  std::cout << "  \"backend_requested\": \"" << requested_backend_name(args) << "\",\n";
  std::cout << "  \"backend_selected\": \"" << selected_backend << "\",\n";
  const char* selected_kernel = selected_kernel_name(args, info, result);
  std::cout << "  \"selected_kernel\": ";
  if (selected_kernel) {
    std::cout << "\"" << selected_kernel << "\"";
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"backend_metadata\": {\n";
  std::cout << "    \"source\": \"" << backend_metadata_source(args) << "\",\n";
  std::cout << "    \"selected_kernel\": ";
  print_json_string_or_null(result.backend_info.selected_kernel);
  std::cout << ",\n";
  std::cout << "    \"accelerator_backend\": "
            << (result.backend_info.is_accelerator ? "true" : "false") << ",\n";
  std::cout << "    \"correctness_backend\": "
            << (result.backend_info.is_correctness_backend ? "true" : "false") << ",\n";
  std::cout << "    \"matrix_engine_backend\": "
            << (result.backend_info.is_matrix_engine_backend ? "true" : "false") << ",\n";
  std::cout << "    \"compiled_kernel_available\": "
            << (result.backend_info.compiled_kernel_available ? "true" : "false") << ",\n";
  std::cout << "    \"exact_differential_validated\": "
            << (result.backend_info.exact_differential_validated ? "true" : "false") << ",\n";
  std::cout << "    \"performance_validated\": "
            << (result.backend_info.performance_validated ? "true" : "false") << ",\n";
  std::cout << "    \"accelerator_library\": ";
  print_json_string_or_null(result.backend_info.accelerator_library);
  std::cout << ",\n";
  std::cout << "    \"accelerator_version\": ";
  print_json_string_or_null(result.backend_info.accelerator_version);
  std::cout << ",\n";
  std::cout << "    \"capability_status\": ";
  print_json_string_or_null(result.backend_info.capability_status);
  std::cout << ",\n";
  std::cout << "    \"epilogue_mode\": ";
  print_json_string_or_null(result.backend_info.epilogue_mode);
  std::cout << ",\n";
  std::cout << "    \"workspace_mode\": ";
  print_json_string_or_null(result.backend_info.workspace_mode);
  std::cout << ",\n";
  std::cout << "    \"workspace_required_bytes\": " << result.backend_info.workspace_required_bytes << ",\n";
  std::cout << "    \"isa_evidence\": ";
  print_json_string_or_null(result.backend_info.isa_evidence);
  std::cout << ",\n";
  std::cout << "    \"autotune_key\": ";
  print_json_string_or_null(result.backend_info.autotune_key);
  std::cout << ",\n";
  std::cout << "    \"accumulator_safety\": {\n";
  std::cout << "      \"input_domain\": ";
  print_json_string_or_null(result.backend_info.accumulator_input_domain);
  std::cout << ",\n";
  std::cout << "      \"signedness\": ";
  print_json_string_or_null(result.backend_info.accumulator_signedness);
  std::cout << ",\n";
  std::cout << "      \"accumulator_type\": ";
  print_json_string_or_null(result.backend_info.accumulator_type);
  std::cout << ",\n";
  std::cout << "      \"modulus_policy\": ";
  print_json_string_or_null(result.backend_info.accumulator_modulus_policy);
  std::cout << ",\n";
  std::cout << "      \"modulus\": " << result.backend_info.accumulator_modulus << ",\n";
  std::cout << "      \"uses_int32_inner_product\": "
            << (result.backend_info.accumulator_uses_int32_inner_product ? "true" : "false") << ",\n";
  std::cout << "      \"k_block_size\": " << result.backend_info.accumulator_k_block_size << ",\n";
  std::cout << "      \"k_block_cap\": " << result.backend_info.accumulator_k_block_cap << ",\n";
  std::cout << "      \"max_lhs_abs\": " << result.backend_info.accumulator_max_lhs_abs << ",\n";
  std::cout << "      \"max_rhs_abs\": " << result.backend_info.accumulator_max_rhs_abs << ",\n";
  std::cout << "      \"max_product\": " << result.backend_info.accumulator_max_product << ",\n";
  std::cout << "      \"safe_for_k_block\": "
            << (result.backend_info.accumulator_safe_for_k_block ? "true" : "false") << ",\n";
  std::cout << "      \"status\": ";
  print_json_string_or_null(result.backend_info.accumulator_safety_status);
  std::cout << "\n";
  std::cout << "    }\n";
  std::cout << "  },\n";
  std::cout << "  \"semantics\": \"" << semantics_name(args.semantics) << "\",\n";
  std::cout << "  \"bound_kind\": \"" << bound_kind_name(args) << "\",\n";
  std::cout << "  \"bound_mode\": \"" << bound_mode_name(args.bound_mode) << "\",\n";
  std::cout << "  \"bound\": " << bound << ",\n";
  std::cout << "  \"bound_source\": \"" << bound_source_name(args) << "\",\n";
  std::cout << "  \"bound_discovery\": ";
  if (bounded_benchmark_semantics(args.semantics)) {
    std::cout << "{\n";
    std::cout << "    \"source\": \"" << bound_discovery_source_name(args, global_bound_scan_available) << "\",\n";
    std::cout << "    \"static_bound\": "
              << (result.effective_bound_available ? result.static_bound : bound) << ",\n";
    std::cout << "    \"selected_bound\": " << bound << ",\n";
    std::cout << "    \"discovered_global_bound\": ";
    if (global_bound_scan_available) {
      std::cout << result.discovered_global_bound;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"candidate_row_sum_col_max\": ";
    if (global_bound_scan_available) {
      std::cout << result.bound_candidate_row_sum_col_max;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"candidate_row_max_col_sum\": ";
    if (global_bound_scan_available) {
      std::cout << result.bound_candidate_row_max_col_sum;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"row_abs_sum_max\": ";
    if (global_bound_scan_available) {
      std::cout << result.row_abs_sum_max;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"row_abs_max\": ";
    if (global_bound_scan_available) {
      std::cout << result.row_abs_max;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"col_abs_sum_max\": ";
    if (global_bound_scan_available) {
      std::cout << result.col_abs_sum_max;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"col_abs_max\": ";
    if (global_bound_scan_available) {
      std::cout << result.col_abs_max;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"zero_row_count\": ";
    if (global_bound_scan_available) {
      std::cout << result.zero_row_count;
    } else {
      std::cout << "null";
    }
    std::cout << ",\n";
    std::cout << "    \"zero_col_count\": ";
    if (global_bound_scan_available) {
      std::cout << result.zero_col_count;
    } else {
      std::cout << "null";
    }
    std::cout << "\n";
    std::cout << "  },\n";
  } else {
    std::cout << "null,\n";
  }
  std::cout << "  \"m\": " << args.m << ",\n";
  std::cout << "  \"n\": " << args.n << ",\n";
  std::cout << "  \"k\": " << args.k << ",\n";
  std::cout << "  \"prefix\": " << prefix << ",\n";
  std::cout << "  \"selected_prefix\": " << selected_prefix << ",\n";
  std::cout << "  \"requested_max_prefix\": " << prefix << ",\n";
  std::cout << "  \"contract_prefix_policy\": \"" << prefix_policy_name(args, result) << "\",\n";
  std::cout << "  \"residue_planes_requested\": " << prefix << ",\n";
  std::cout << "  \"residue_planes_selected\": " << selected_prefix << ",\n";
  std::cout << "  \"residue_planes_skipped\": " << residue_planes_skipped << ",\n";
  std::cout << "  \"residue_plane_skip_fraction\": " << residue_plane_skip_fraction << ",\n";
  std::cout << "  \"finite_modulus\": ";
  if (finite_benchmark_semantics(args.semantics)) {
    std::cout << args.finite_modulus;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"tile_m\": " << args.tile_m << ",\n";
  std::cout << "  \"tile_n\": " << args.tile_n << ",\n";
  std::cout << "  \"layout\": \"row_major\",\n";
  std::cout << "  \"k_block_size\": "
            << benchmark_k_block_size(args, result)
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
  std::cout << "    \"source\": \"" << json_escape(result.schedule_source) << "\",\n";
  std::cout << "    \"bound_kind\": \"" << bound_kind_name(result.schedule_info.bound_kind) << "\",\n";
  std::cout << "    \"effective_bound\": " << result.schedule_info.effective_bound << ",\n";
  std::cout << "    \"lhs_bound\": " << result.schedule_info.lhs_bound << ",\n";
  std::cout << "    \"rhs_bound\": " << result.schedule_info.rhs_bound << ",\n";
  std::cout << "    \"bound_contract\": \"" << json_escape(result.schedule_info.bound_contract) << "\",\n";
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
  std::cout << "    \"flags\": " << result.schedule_info.flags << ",\n";
  std::cout << "    \"zero_output_tile_count\": " << result.zero_output_tile_count << ",\n";
  const double zero_output_tile_fraction =
      result.schedule_info.tile_count == 0
          ? 0.0
          : static_cast<double>(result.zero_output_tile_count) /
                static_cast<double>(result.schedule_info.tile_count);
  std::cout << "    \"zero_output_tile_fraction\": ";
  std::cout << zero_output_tile_fraction;
  std::cout << ",\n";
  std::cout << "    \"zero_output_selected_residue_planes\": "
            << result.zero_output_selected_residue_plane_count << ",\n";
  std::cout << "    \"zero_output_skip_active\": "
            << (result.zero_output_tile_count != 0 ? "true" : "false") << ",\n";
  std::cout << "    \"range_bit_length\": " << result.schedule_info.range_bit_length << "\n";
  std::cout << "  },\n";
  std::cout << "  \"epilogue_type\": \"" << epilogue_type(args) << "\",\n";
  std::cout << "  \"exact_wide_limb_count\": ";
  if (exact_wide_benchmark_semantics(args.semantics)) {
    std::cout << args.exact_wide_limb_count;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"exact_wide_export_status_check\": ";
  print_json_string_or_null(exact_wide_export_status_check_name(args));
  std::cout << ",\n";
  std::cout << "  \"residue_chain_length\": " << args.residue_chain_length << ",\n";
  std::cout << "  \"residue_output_mode\": \"" << residue_output_mode_name(args) << "\",\n";
  std::cout << "  \"packed_layout_version\": ";
  print_json_string_or_null(packed_layout_version(args));
  std::cout << ",\n";
  std::cout << "  \"seed\": " << args.seed << ",\n";
  std::cout << "  \"warmups\": " << args.warmups << ",\n";
  std::cout << "  \"repeats\": " << args.repeats << ",\n";
  std::cout << "  \"reuse_packed_inputs\": " << (args.reuse_packed_inputs ? "true" : "false") << ",\n";
  std::cout << "  \"pack_mode\": \"" << pack_mode_name(args) << "\",\n";
  std::cout << "  \"prepack_reuse_operands\": ";
  print_string_array(prepack_reuse_operands(args));
  std::cout << ",\n";
  std::cout << "  \"prepack_reuse_strategy\": \"" << prepack_reuse_strategy_name(result.prepack_reuse_strategy)
            << "\",\n";
  std::cout << "  \"prepack_setup_us\": ";
  if (result.prepack_setup_available) {
    std::cout << result.prepack_setup_us;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"input_distribution\": \"" << input_distribution(args) << "\",\n";
  std::cout << "  \"command_line\": \"" << json_escape(cmdline) << "\",\n";
  std::cout << "  \"git_commit\": \"" << json_escape(runtime_git_commit()) << "\",\n";
  std::cout << "  \"compiler\": {\n";
  std::cout << "    \"id\": \"" << compiler_id() << "\",\n";
  std::cout << "    \"version\": \"" << compiler_version() << "\"\n";
  std::cout << "  },\n";
  std::cout << "  \"configured_amdgpu_targets\": \"" << json_escape(RNS8_CONFIGURED_AMDGPU_TARGETS) << "\",\n";
  std::cout << "  \"hip_toolchain\": {\n";
  std::cout << "    \"enabled\": " << (RNS8_CONFIGURED_HIP_ENABLED ? "true" : "false") << ",\n";
  std::cout << "    \"hip_root\": ";
  print_nullable_string(RNS8_CONFIGURED_HIP_ROOT);
  std::cout << ",\n";
  std::cout << "    \"hipcc_path\": ";
  print_nullable_string(RNS8_CONFIGURED_HIPCC_PATH);
  std::cout << ",\n";
  std::cout << "    \"hipcc_version\": ";
  print_nullable_string(RNS8_CONFIGURED_HIPCC_VERSION);
  std::cout << ",\n";
  std::cout << "    \"hip_sdk_or_rocm_version\": ";
  print_nullable_string(RNS8_CONFIGURED_HIP_SDK_OR_ROCM_VERSION);
  std::cout << ",\n";
  std::cout << "    \"version_source\": "
            << (RNS8_CONFIGURED_HIP_ENABLED && RNS8_CONFIGURED_HIPCC_VERSION[0] != '\0'
                    ? "\"hipcc --version\""
                    : "null")
            << "\n";
  std::cout << "  },\n";
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
  print_comparison_baseline(args, info, result);
  std::cout << "  \"derived_tops_equivalent\": null,\n";
  std::cout << "  \"timing_source\": \"std::chrono::steady_clock\",\n";
  if (chain_residue_output) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for a residue-current RNS GEMM chain; "
              << "each measured repeat runs " << args.residue_chain_length
              << " resident RNS GEMM calls before host export, raw_timings_us.crt_export is intentionally zero, and "
                 "one final logical export runs after measured repeats only to produce checksum_u64\",\n";
  } else if (args.oneshot) {
    if (finite_benchmark_semantics(args.semantics)) {
      std::cout << "  \"timing_note\": \"host wall-clock timings for the public finite-u8 one-shot API; "
                   "raw_timings_us.rns_gemm and raw_timings_us.end_to_end both measure one complete "
                   "rns8_gemm_finite_ring_u8_oneshot or rns8_gemm_finite_field_u8_oneshot call, while "
                   "raw_timings_us.pack and raw_timings_us.crt_export are zero because native uint8 input copies, "
                   "direct-HIP native-input finite GEMM work, canonical export, and teardown happen inside the "
                   "measured API call\",\n";
    } else {
      std::cout << "  \"timing_note\": \"host wall-clock timings for the public bounded one-shot API; "
                   "raw_timings_us.rns_gemm and raw_timings_us.end_to_end both measure one complete "
                   "rns8_gemm_i64_oneshot or rns8_gemm_u64_oneshot call, while raw_timings_us.pack and "
                   "raw_timings_us.crt_export are zero because transient input copies, any fused native-input "
                   "direct-HIP GEMM work, logical export, and teardown happen inside the measured API call\",\n";
    }
  } else if (args.reuse_packed_inputs) {
    if (use_prepacked_b_cache) {
      std::cout << "  \"timing_note\": \"host wall-clock timings with a reusable rocWMMA B prepack cache; "
                   "prepack_setup_us records persistent B packing plus cache materialization before warmups, "
                   "end_to_end includes per-repeat A packing, cached-B rns_gemm, and export\",\n";
    } else if (reuses_all_packed_inputs(args)) {
      std::cout << "  \"timing_note\": \"host wall-clock timings for repeated GEMM/export against persistent "
                   "packed A/B inputs; pack is a zero-valued per-repeat phase and prepack_setup_us records the "
                   "one-time pack before warmups, which end_to_end excludes\",\n";
    } else {
      std::cout << "  \"timing_note\": \"host wall-clock timings with persistent packed "
                << prepack_reuse_operand_text(args) << " input and per-repeat packing of "
                << per_repeat_pack_operand_text(args)
                << "; prepack_setup_us records the one-time pack before warmups and end_to_end includes "
                   "the per-repeat pack phase for the non-reused operand\",\n";
    }
  } else if (args.vector_alu_baseline) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the benchmark-only HIP vector-ALU exact "
                 "int64 baseline; phases are raw input H2D copies, one 192-bit-limb exact output kernel, "
                 "and direct output D2H export with range-status validation\",\n";
  } else if (runtime_vector_alu_backend(args)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the public HIP vector-ALU bounded runtime "
                 "backend; phases are native device packing, one 192-bit-limb exact output kernel, and direct "
                 "native output export with range-status validation\",\n";
  } else if (args.wrap64_rocwmma_candidate) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the internal rocWMMA wrap64 byte-GEMM36 "
                 "candidate; GPU event timing uses direct-HIP byte-limb pack/export labels plus one candidate "
                 "operation-group label and this path is not public or AUTO-selected\",\n";
  } else if (args.semantics == BenchSemantics::WrapU64Mod2_64 && selected_backend_kind == RNS8_BACKEND_HIP_DIRECT) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the direct-HIP wrap64 tiled byte-limb "
                 "path; GPU event timing uses wrap64-specific tiled byte-GEMM/export labels plus "
                 "current rns_gemm/crt_export aggregate phase labels\",\n";
  } else if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the CPU wrap64 byte-limb reference; "
                 "no GPU event timing is requested for this backend\",\n";
  } else if (finite_benchmark_semantics(args.semantics)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for persistent finite-u8 packing, finite residue "
                 "GEMM, and canonical uint8 export with an explicit benchmark modulus\",\n";
  } else if (exact_wide_benchmark_semantics(args.semantics)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for persistent exact-wide RNS packing, RNS GEMM, "
                 "and fixed-width little-endian limb export; GPU event timing names exact-wide export operation "
                 "groups when backend hooks are available";
    if (!exact_wide_export_status_check_required(args)) {
      std::cout << "; the selected limb count covers the backend 192-bit reconstruction width, so exact-wide "
                   "range-status memset and status D2H are elided and reported as zero-valued event phases";
    }
    std::cout << "\",\n";
  } else if (adaptive_applied && info.backend == RNS8_BACKEND_CPU_REFERENCE) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the CPU adaptive per-tile bounded "
                 "reference path; no GPU event timing is requested for this backend\",\n";
  } else if (adaptive_applied && accelerator_deep_kernel_events) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for an adaptive per-tile bounded accelerator "
                 "path; GPU event timing combines direct-HIP pack/export labels with one accelerator GEMM "
                 "operation-group label\",\n";
  } else if (adaptive_applied && gpu_events_available) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the direct-HIP adaptive per-tile bounded "
                 "correctness path; GPU event timing aggregates all selected-prefix tiled GEMM launches and tiled "
                 "CRT export work into backend operation-group labels\",\n";
  } else if (adaptive_applied) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for an adaptive per-tile bounded accelerator path; "
                 "GPU event timing was not captured for this selected backend/path\",\n";
  } else if (info.backend == RNS8_BACKEND_HIPBLASLT) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for the hipBLASLt baseline path; GPU event timing "
                 "separates persistent RNS packing, hipBLASLt INT8-to-INT32 matmul, separate centered-residue "
                 "reduction, and CRT export operation groups\",\n";
  } else {
    std::cout << "  \"timing_note\": \"host wall-clock timings; direct-HIP calls include current backend "
                 "synchronization, first-use persistent buffer allocation, copies, kernel launches, fused reduction, "
                 "and GPU bounded export when using HIP\",\n";
  }
  std::cout << "  \"timing_metadata\": {\n";
  std::cout << "    \"unit\": \"microseconds\",\n";
  std::cout << "    \"source\": \"std::chrono::steady_clock\",\n";
  std::cout << "    \"source_scope\": \"host_wall_clock\",\n";
  std::cout << "    \"benchmark_execution_mode\": \"" << benchmark_execution_mode_name(args) << "\",\n";
  std::cout << "    \"pack_mode\": \"" << pack_mode_name(args) << "\",\n";
  std::cout << "    \"prepack_reuse_operands\": ";
  print_string_array(prepack_reuse_operands(args));
  std::cout << ",\n";
  std::cout << "    \"prepack_reuse_strategy\": \"" << prepack_reuse_strategy_name(result.prepack_reuse_strategy)
            << "\",\n";
  std::cout << "    \"gpu_event_timing\": " << (gpu_events_available ? "true" : "false") << ",\n";
  std::cout << "    \"gpu_event_timing_reason\": \"" << gpu_event_reason << "\",\n";
  std::cout << "    \"gpu_event_timing_status\": \"" << gpu_event_status << "\",\n";
  std::cout << "    \"gpu_event_timing_source\": "
            << (gpu_events_available ? "\"hipEventElapsedTime\"" : "null") << ",\n";
  std::cout << "    \"gpu_event_timing_source_scope\": " << gpu_event_scope << ",\n";
  std::cout << "    \"gpu_event_timing_caveat\": " << gpu_event_caveat << ",\n";
  if (!gpu_events_available && !result.gpu_events.unavailable_reasons.empty()) {
    std::cout << "    \"gpu_event_timing_unavailable_reasons\": ";
    print_string_array(result.gpu_events.unavailable_reasons);
    std::cout << ",\n";
  }
  std::cout
      << "    \"phase_order\": [";
  if (global_bound_scan_available) {
    std::cout << "\"global_bound_scan\", ";
  }
  std::cout << "\"planning\", \"scheduling\", ";
  if (result.tile_bound_scan_available) {
    std::cout << "\"tile_bound_scan\", ";
  }
  std::cout << "\"matrix_alloc\", \"pack\", \"rns_gemm\", \"crt_export\", \"end_to_end\"],\n";
  std::cout << "    \"gpu_event_phase_order\": ";
  if (gpu_events_available) {
    print_string_array(event_phase_order);
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"phase_notes\": {\n";
  if (global_bound_scan_available) {
    std::cout << "      \"global_bound_scan\": \"one-time exact seeded input prepass that computes row/column absolute-summary global bounds before plan creation\",\n";
  }
  if (args.oneshot) {
    std::cout << "      \"planning\": \"one-time metadata-only rns8_create_plan timing; each measured one-shot API call performs its own internal setup inside rns_gemm/end_to_end\",\n";
  } else if (args.wrap64_rocwmma_candidate) {
    std::cout << "      \"planning\": \"one-time benchmark-owned metadata initialization for the internal rocWMMA wrap64 candidate\",\n";
  } else if (args.vector_alu_baseline) {
    std::cout << "      \"planning\": \"one-time rns8_create_plan schedule validation for the same bounded semantic contract\",\n";
  } else {
    std::cout << "      \"planning\": \"one-time rns8_create_plan plus rns8_create_workspace host timing\",\n";
  }
  if (args.wrap64_rocwmma_candidate) {
    std::cout << "      \"scheduling\": \"one-time fixed 16x16 WMMA candidate schedule derivation from the matrix shape\",\n";
  } else {
    std::cout << "      \"scheduling\": \"one-time rns8_get_plan_schedule_info host timing\",\n";
  }
  if (result.tile_bound_scan_available) {
    std::cout << "      \"tile_bound_scan\": \"one-time exact seeded input prepass that computes per-output-tile bounds before plan creation\",\n";
  }
  if (args.wrap64_rocwmma_candidate) {
    std::cout << "      \"matrix_alloc\": \"one-time benchmark-owned compact byte-limb HIP device buffer allocation host timing\",\n";
  } else if (args.vector_alu_baseline) {
    std::cout << "      \"matrix_alloc\": \"one-time benchmark-owned HIP device buffer allocation host timing\",\n";
  } else if (args.oneshot) {
    std::cout << "      \"matrix_alloc\": \"zero-valued external phase; transient API allocations, if any, are inside the measured one-shot call\",\n";
  } else {
    std::cout << "      \"matrix_alloc\": \"one-time persistent matrix allocation host timing\",\n";
  }
  if (chain_residue_output) {
    if (args.reuse_packed_inputs) {
      if (reuses_all_packed_inputs(args)) {
        std::cout << "      \"pack\": \"zero-valued per-repeat phase; A and B were packed once into persistent RNS matrices before warmups\",\n";
      } else {
        std::cout << "      \"pack\": \"per-repeat host timing for packing " << per_repeat_pack_operand_text(args)
                  << "; " << prepack_reuse_operand_text(args)
                  << " was packed once into a persistent RNS matrix before warmups\",\n";
      }
    } else {
      std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into persistent RNS matrices\",\n";
    }
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for " << args.residue_chain_length
              << " chained rns8_gemm_rns calls that keep the intermediate output resident in RNS form\",\n";
    std::cout << "      \"crt_export\": \"zero-valued per-repeat phase; residue-current chain mode defers host logical export until one final checksum export after measured repeats\",\n";
  } else if (args.oneshot) {
    std::cout << "      \"pack\": \"zero-valued external phase; native input copies and any backend-local transformation are inside the measured one-shot API call\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for one complete public one-shot API call\",\n";
    std::cout << "      \"crt_export\": \"zero-valued external phase; logical output export is inside the measured one-shot API call\",\n";
  } else if (args.reuse_packed_inputs) {
    if (reuses_all_packed_inputs(args)) {
      std::cout << "      \"pack\": \"zero-valued per-repeat phase; A and B were packed once into persistent matrices before warmups\",\n";
    } else {
      std::cout << "      \"pack\": \"per-repeat host timing for packing " << per_repeat_pack_operand_text(args)
                << "; " << prepack_reuse_operand_text(args)
                << " was packed once into a persistent matrix before warmups\",\n";
    }
    if (args.wrap64_rocwmma_candidate) {
      std::cout << "      \"rns_gemm\": \"per-repeat host timing for the internal rocWMMA wrap64 byte-GEMM36 candidate against reused compact byte-limb inputs\",\n";
      std::cout << "      \"crt_export\": \"per-repeat host timing for direct-HIP low-64-bit byte-limb export\",\n";
    } else if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
      std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_wrap_u64 against reused byte-limb inputs\",\n";
      std::cout << "      \"crt_export\": \"per-repeat host timing for low-64-bit rns8_export_wrap_u64\",\n";
    } else if (finite_benchmark_semantics(args.semantics)) {
      std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_finite_u8 against reused finite-u8 inputs\",\n";
      std::cout << "      \"crt_export\": \"per-repeat host timing for canonical uint8 rns8_export_finite_u8\",\n";
    } else if (exact_wide_benchmark_semantics(args.semantics)) {
      std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_rns against reused exact-wide persistent RNS inputs\",\n";
      std::cout << "      \"crt_export\": \"per-repeat host timing for fixed-width exact-wide limb export\",\n";
    } else {
      std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_rns against reused persistent RNS inputs\",\n";
      std::cout << "      \"crt_export\": \"per-repeat host timing for export/reconstruction into logical output\",\n";
    }
  } else if (args.vector_alu_baseline) {
    std::cout << "      \"pack\": \"per-repeat host timing for raw bounded inputs copied to benchmark-owned HIP buffers\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for one exact 192-bit-limb vector-ALU output kernel\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for range-status D2H plus direct logical output D2H\",\n";
  } else if (runtime_vector_alu_backend(args)) {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into native int64/uint64 HIP device matrices\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_rns dispatch to one exact 192-bit-limb vector-ALU output kernel\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for native output D2H plus range validation\",\n";
  } else if (args.wrap64_rocwmma_candidate) {
    std::cout << "      \"pack\": \"per-repeat host timing for direct-HIP packing of A and B into compact byte-limb device buffers\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for the internal rocWMMA wrap64 byte-GEMM36 candidate\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for direct-HIP low-64-bit byte-limb export\",\n";
  } else if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into persistent byte-limb matrices\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_wrap_u64\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for low-64-bit rns8_export_wrap_u64\",\n";
  } else if (finite_benchmark_semantics(args.semantics)) {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into persistent finite-u8 matrices\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_finite_u8\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for canonical uint8 rns8_export_finite_u8\",\n";
  } else if (exact_wide_benchmark_semantics(args.semantics)) {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into exact-wide persistent RNS matrices\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_rns\",\n";
    if (exact_wide_export_status_check_required(args)) {
      std::cout << "      \"crt_export\": \"per-repeat host timing for fixed-width exact-wide limb export with range-status memset and status D2H\",\n";
    } else {
      std::cout << "      \"crt_export\": \"per-repeat host timing for fixed-width exact-wide limb export; selected limb width covers backend 192-bit reconstruction so range-status memset and status D2H are elided\",\n";
    }
  } else {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into persistent RNS matrices\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for rns8_gemm_rns\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for export/reconstruction into logical output\",\n";
  }
  if (chain_residue_output) {
    std::cout << "      \"end_to_end\": \"per-repeat pack plus chained rns_gemm host timing; excludes the final checksum-only logical export";
    if (args.reuse_packed_inputs) {
      std::cout << " and one-time prepack_setup_us";
    }
    std::cout << "\"\n";
  } else if (args.oneshot) {
    std::cout << "      \"end_to_end\": \"same measured duration as rns_gemm for one complete public one-shot API call\"\n";
  } else if (args.reuse_packed_inputs) {
    if (reuses_all_packed_inputs(args)) {
      std::cout << "      \"end_to_end\": \"per-repeat rns_gemm plus crt_export host timing; excludes one-time prepack_setup_us\"\n";
    } else {
      std::cout << "      \"end_to_end\": \"per-repeat pack of non-reused input plus rns_gemm plus crt_export host timing; excludes one-time prepack_setup_us\"\n";
    }
  } else {
    std::cout << "      \"end_to_end\": \"per-repeat pack plus rns_gemm plus crt_export host timing\"\n";
  }
  std::cout << "    },\n";
  std::cout << "    \"phase_availability\": {\n";
  if (global_bound_scan_available) {
    std::cout << "      \"global_bound_scan\": {\n";
    std::cout << "        \"timed\": true,\n";
    std::cout << "        \"timing_key\": \"global_bound_scan\",\n";
    std::cout << "        \"scope\": \"input_row_column_abs_summary\",\n";
    std::cout << "        \"reason\": \"measured with host steady_clock around seeded input row/column absolute-summary bound discovery before plan creation\"\n";
    std::cout << "      },\n";
  }
  std::cout << "      \"scheduling\": {\n";
  std::cout << "        \"timed\": true,\n";
  std::cout << "        \"timing_key\": \"scheduling\",\n";
  if (args.wrap64_rocwmma_candidate) {
    std::cout << "        \"scope\": \"benchmark_static_wrap64_rocwmma_candidate_schedule\",\n";
    std::cout << "        \"reason\": \"measured with host steady_clock around fixed 16x16 candidate schedule metadata initialization\"\n";
  } else {
    std::cout << "        \"scope\": \"one_time_schedule_info_query\",\n";
    std::cout << "        \"reason\": \"measured with host steady_clock around rns8_get_plan_schedule_info\"\n";
  }
  std::cout << "      },\n";
  if (result.tile_bound_scan_available) {
    std::cout << "      \"tile_bound_scan\": {\n";
    std::cout << "        \"timed\": true,\n";
    std::cout << "        \"timing_key\": \"tile_bound_scan\",\n";
    std::cout << "        \"scope\": \"exact_seeded_input_prepass\",\n";
    std::cout << "        \"reason\": \"measured with host steady_clock around exact per-output-tile bound computation before plan creation\"\n";
    std::cout << "      },\n";
  }
  std::cout << "      \"prepack_setup\": {\n";
  std::cout << "        \"timed\": " << (result.prepack_setup_available ? "true" : "false") << ",\n";
  std::cout << "        \"timing_key\": "
            << (result.prepack_setup_available ? "\"prepack_setup_us\"" : "null") << ",\n";
  std::cout << "        \"scope\": \""
            << (result.prepack_setup_available ? "one_time_before_warmups" : "not_requested_per_repeat_repack")
            << "\",\n";
  std::cout << "        \"reason\": \""
            << (result.prepack_setup_available
                    ? (use_prepacked_b_cache
                           ? "packed B into persistent matrix storage, materialized a reusable rocWMMA B cache before warmups, and reused that cache for every measured repeat"
                           : "prepacked " + prepack_reuse_operand_text(args) +
                                 " once before warmups and reused for every measured repeat")
                    : (args.oneshot
                           ? "one-shot benchmark mode does not expose a benchmark-side prepack phase"
                           : "benchmark mode packs A and B inside every measured repeat"))
            << "\"\n";
  std::cout << "      },\n";
  std::cout << "      \"reduction\": {\n";
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    std::cout << "        \"timed\": false,\n";
    std::cout << "        \"timing_key\": null,\n";
    std::cout << "        \"scope\": \"not_applicable_wrap64_byte_limb\",\n";
    std::cout << "        \"reason\": \"strict wrap64 byte-limb captures do not use centered RNS residue reduction\"\n";
  } else if (std::string(selected_backend) == "hipblaslt") {
    std::cout << "        \"timed\": false,\n";
    std::cout << "        \"timing_key\": null,\n";
    std::cout << "        \"scope\": \"separate_hipblaslt_i32_scratch_residue_reduce\",\n";
    std::cout << "        \"reason\": \"hipBLASLt baseline GEMM writes INT32 scratch and runs a separate centered-residue reduction inside rns_gemm\"\n";
  } else if (args.vector_alu_baseline) {
    std::cout << "        \"timed\": false,\n";
    std::cout << "        \"timing_key\": null,\n";
    std::cout << "        \"scope\": \"not_applicable_direct_int64_export\",\n";
    std::cout << "        \"reason\": \"benchmark-only vector-ALU baseline computes exact logical outputs directly and does not use centered RNS residue reduction\"\n";
  } else if (runtime_vector_alu_backend(args)) {
    std::cout << "        \"timed\": false,\n";
    std::cout << "        \"timing_key\": null,\n";
    std::cout << "        \"scope\": \"not_applicable_native_vector_output\",\n";
    std::cout << "        \"reason\": \"runtime vector-ALU backend computes exact native int64/uint64 outputs directly and does not use centered RNS residue reduction\"\n";
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
    print_gpu_event_timings(event_phase_order, result.gpu_events);
    print_gpu_event_timing_summary(event_phase_order, result.gpu_events);
  } else {
    std::cout << "  \"gpu_event_timings_us\": null,\n";
    std::cout << "  \"gpu_event_timing_summary_us\": null,\n";
  }
  std::cout << "  \"plan_us\": " << result.plan_us << ",\n";
  if (global_bound_scan_available) {
    std::cout << "  \"global_bound_scan_us\": " << result.global_bound_scan_us << ",\n";
    std::cout << "  \"avg_global_bound_scan_us\": " << static_cast<double>(result.global_bound_scan_us) << ",\n";
  }
  std::cout << "  \"avg_planning_us\": " << static_cast<double>(result.plan_us) << ",\n";
  std::cout << "  \"schedule_query_us\": " << result.schedule_query_us << ",\n";
  std::cout << "  \"avg_scheduling_us\": " << static_cast<double>(result.schedule_query_us) << ",\n";
  if (result.tile_bound_scan_available) {
    std::cout << "  \"tile_bound_scan_us\": " << result.tile_bound_scan_us << ",\n";
    std::cout << "  \"avg_tile_bound_scan_us\": " << static_cast<double>(result.tile_bound_scan_us) << ",\n";
  }
  std::cout << "  \"matrix_alloc_us\": " << result.matrix_alloc_us << ",\n";
  std::cout << "  \"avg_matrix_alloc_us\": " << static_cast<double>(result.matrix_alloc_us) << ",\n";
  std::cout << "  \"avg_prepack_setup_us\": ";
  if (result.prepack_setup_available) {
    std::cout << static_cast<double>(result.prepack_setup_us);
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"avg_pack_us\": " << avg_pack_us << ",\n";
  std::cout << "  \"avg_rns_gemm_us\": " << avg_gemm_us << ",\n";
  std::cout << "  \"per_modulus_gemm_estimate_applicable\": "
            << (per_modulus_estimate_applicable ? "true" : "false") << ",\n";
  std::cout << "  \"avg_per_modulus_gemm_estimate_us\": " << avg_per_modulus_gemm_estimate_us << ",\n";
  std::cout << "  \"avg_crt_export_us\": " << avg_export_us << ",\n";
  std::cout << "  \"avg_end_to_end_us\": " << avg_end_to_end_us << ",\n";
  std::cout << "  \"raw_timings_us\": {\n";
  if (global_bound_scan_available) {
    std::cout << "    \"global_bound_scan\": ";
    print_single_u64_array(result.global_bound_scan_us);
    std::cout << ",\n";
  }
  std::cout << "    \"planning\": ";
  print_single_u64_array(result.plan_us);
  std::cout << ",\n";
  std::cout << "    \"scheduling\": ";
  print_single_u64_array(result.schedule_query_us);
  std::cout << ",\n";
  if (result.tile_bound_scan_available) {
    std::cout << "    \"tile_bound_scan\": ";
    print_single_u64_array(result.tile_bound_scan_us);
    std::cout << ",\n";
  }
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
  if (global_bound_scan_available) {
    print_single_timing_summary("global_bound_scan", result.global_bound_scan_us, true);
  }
  print_single_timing_summary("planning", result.plan_us, true);
  print_single_timing_summary("scheduling", result.schedule_query_us, true);
  if (result.tile_bound_scan_available) {
    print_single_timing_summary("tile_bound_scan", result.tile_bound_scan_us, true);
  }
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
  if (args.write_autotune_cache) {
    std::cerr << "write autotune cache: refused raw benchmark cache write; use "
                 "tools\\benchmark_sweep.py --review-mode release --write-autotune-cache\n";
    return 1;
  }

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
    case BenchSemantics::ExactWideSigned:
      result = run_exact_wide_signed(ctx, args, bound);
      break;
    case BenchSemantics::ExactWideUnsigned:
      result = run_exact_wide_unsigned(ctx, args, bound);
      break;
    case BenchSemantics::WrapU64Mod2_64:
      result = run_wrap_u64(ctx, args, bound);
      break;
    case BenchSemantics::FiniteRingU8:
    case BenchSemantics::FiniteFieldU8:
      result = run_finite_u8(ctx, args, bound);
      break;
  }
  rns8_destroy_context(ctx);
  const uint64_t effective_bound = result.effective_bound_available ? result.effective_bound : bound;
  print_json(args, info, result, effective_bound, cmdline);
  return 0;
}
