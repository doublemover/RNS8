#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "backend_rocwmma/rocwmma_backend.hpp"
#include "backend_wrap64/wrap64_hip.hpp"
#include "core/autotune_cache.hpp"
#include "core/backend_common.hpp"
#include "core/plan_lowering.hpp"
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

#if RNS8_CONFIGURED_HIP_ENABLED
#  include <hip/hip_runtime_api.h>
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

enum class NextOpHint {
  Auto,
  FinalExport,
  RnsGemm,
  NativeGemm,
  NativeToRns,
  ReuseB,
};

struct Args {
  int64_t m = 64;
  int64_t n = 64;
  int64_t k = 64;
  int64_t output_ld_padding = 0;
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
  bool residue_chain_final_export = false;
  uint32_t host_api_batch_size = 1;
  bool require_adaptive_execution = false;
  bool write_autotune_cache = false;
  bool oneshot = false;
  bool transient_uniform_small_inputs = false;
  bool reuse_packed_inputs = false;
  bool reuse_packed_a = false;
  bool reuse_packed_b = false;
  bool native_to_rns_bridge = false;
  bool vector_to_rns_chain = false;
  NextOpHint next_op_hint = NextOpHint::Auto;
  bool residue_channel_fusion = false;
  std::string modulus_set = "default";
  std::string tile_shape_variant = "default";
  std::string export_variant = "default";
  std::string reconstruction_variant = "default_garner";
  uint32_t grouped_dispatch_tasks = 1;
  bool hip_graph_replay = false;
  std::string workload_proxy = "none";
  bool resident_lifetime = false;
  bool workspace_arena = false;
  bool adaptive_grouped_scheduler = false;
  bool streaming_overlap = false;
  std::string release_gate = "none";
  std::string verification_amortization = "none";
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
  std::vector<uint8_t> zero_a_rows{};
  std::vector<uint8_t> zero_b_cols{};
  uint64_t zero_a_row_proof_count = 0;
  uint64_t zero_b_col_proof_count = 0;
  uint64_t zero_row_col_product_count = 0;
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
  rns8_plan_packing_info packing_info{};
  bool packing_info_available = false;
  rns8::detail::PlanLoweringDescription lowering_info{};
  bool lowering_info_available = false;
  TimingSamples samples{};
  GpuEventSamples gpu_events{};
  rns8::detail::hip_direct_allocation_counters allocation_before{};
  rns8::detail::hip_direct_allocation_counters allocation_after_warmups{};
  rns8::detail::hip_direct_allocation_counters allocation_after_repeats{};
  bool allocation_tracking_available = false;
  bool allocation_after_warmups_available = false;
  uint64_t prepack_setup_us = 0;
  bool prepack_setup_available = false;
  PrepackReuseStrategy prepack_reuse_strategy = PrepackReuseStrategy::None;
  bool hip_graph_replay_requested = false;
  bool hip_graph_replay_available = false;
  bool hip_graph_replay_used = false;
  uint64_t hip_graph_capture_us = 0;
  uint64_t hip_graph_instantiate_us = 0;
  uint64_t hip_graph_launch_count = 0;
  std::string hip_graph_replay_status = "not_requested";
  std::string hip_graph_replay_scope = "not_applicable";
  std::string hip_graph_replay_caveat{};
  bool grouped_dispatch_batched_export_enabled = false;
  uint64_t grouped_dispatch_device_output_slab_bytes = 0;
  std::string grouped_dispatch_execution_strategy = "not_requested";
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
bool vector_to_rns_chain_requested(const Args& args);
bool host_api_batch_requested(const Args& args);
bool grouped_dispatch_requested(const Args& args);
bool grouped_task_executor_requested(const Args& args);
uint32_t measured_task_count(const Args& args);
bool hip_graph_replay_requested(const Args& args);
void record_allocation_after_warmups(BenchmarkResult& result);
struct HipGraphReplayState {
#if RNS8_CONFIGURED_HIP_ENABLED
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
#endif
  uint64_t launch_count = 0;
};
void destroy_hip_graph_replay_state(HipGraphReplayState& state);
rns8_status capture_hip_graph_replay(
    int device_id,
    const std::function<rns8_status(void*)>& capture_body,
    HipGraphReplayState& state,
    uint64_t& capture_us,
    uint64_t& instantiate_us,
    std::string& error_text);
rns8_status launch_hip_graph_replay(int device_id, HipGraphReplayState& state, std::string& error_text);

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
      << "                  [--input-profile uniform-small|adaptive-bands]\n"
      << "                  [--bound-source static-profile|input-scan]\n"
      << "                  [--prefix-policy minimum-proven|fixed-requested] [--max-prefix N]\n"
      << "                  [--exact-wide-limbs 1..32]\n"
      << "                  [--residue-chain-length N]\n"
      << "                  [--residue-chain-final-export]\n"
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
      << "                  [--require-adaptive-execution]\n"
      << "                  [--residue-channel-fusion]\n"
      << "                  [--oneshot]\n"
      << "                  [--transient-uniform-small-inputs]\n"
      << "                  [--native-to-rns-bridge]\n"
      << "                  [--vector-to-rns-chain]\n"
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

bool rns_residue_chain_requested(const Args& args) {
  return rns_chain_benchmark_semantics(args.semantics) && args.residue_chain_length > 1;
}

bool residue_current_output_mode(const Args& args) {
  return rns_residue_chain_requested(args) && !args.residue_chain_final_export;
}

bool residue_chain_final_export_requested(const Args& args) {
  return rns_residue_chain_requested(args) && args.residue_chain_final_export;
}

bool exact_wide_export_status_check_required(const Args& args) {
  if (args.semantics == BenchSemantics::ExactWideUnsigned) {
    return args.exact_wide_limb_count < 3;
  }
  if (args.semantics == BenchSemantics::ExactWideSigned) {
    return args.exact_wide_limb_count < 3;
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

NextOpHint parse_next_op_hint(const std::string& value) {
  if (value == "auto") return NextOpHint::Auto;
  if (value == "final-export" || value == "final_export") return NextOpHint::FinalExport;
  if (value == "rns-gemm" || value == "rns_gemm") return NextOpHint::RnsGemm;
  if (value == "native-gemm" || value == "native_gemm") return NextOpHint::NativeGemm;
  if (value == "native-to-rns" || value == "native_to_rns") return NextOpHint::NativeToRns;
  if (value == "reuse-b" || value == "reuse_b") return NextOpHint::ReuseB;
  usage_error("unknown next-op hint: " + value);
}

const char* next_op_hint_name(NextOpHint hint) {
  switch (hint) {
    case NextOpHint::Auto:
      return "auto";
    case NextOpHint::FinalExport:
      return "final-export";
    case NextOpHint::RnsGemm:
      return "rns-gemm";
    case NextOpHint::NativeGemm:
      return "native-gemm";
    case NextOpHint::NativeToRns:
      return "native-to-rns";
    case NextOpHint::ReuseB:
      return "reuse-b";
  }
  return "unknown";
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
    } else if (arg == "--output-ld-padding" && i + 1 < argc) {
      args.output_ld_padding = parse_i64(argv[++i], "--output-ld-padding");
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
    } else if (arg == "--residue-chain-final-export") {
      args.residue_chain_final_export = true;
    } else if (arg == "--host-api-batch-size" && i + 1 < argc) {
      args.host_api_batch_size = parse_u32(argv[++i], "--host-api-batch-size");
    } else if (arg == "--next-op-hint" && i + 1 < argc) {
      args.next_op_hint = parse_next_op_hint(argv[++i]);
    } else if (arg == "--modulus-set" && i + 1 < argc) {
      args.modulus_set = argv[++i];
    } else if (arg == "--tile-shape-variant" && i + 1 < argc) {
      args.tile_shape_variant = argv[++i];
    } else if (arg == "--export-variant" && i + 1 < argc) {
      args.export_variant = argv[++i];
    } else if (arg == "--reconstruction-variant" && i + 1 < argc) {
      args.reconstruction_variant = argv[++i];
    } else if (arg == "--grouped-dispatch" && i + 1 < argc) {
      args.grouped_dispatch_tasks = parse_u32(argv[++i], "--grouped-dispatch");
    } else if (arg == "--hip-graph-replay") {
      args.hip_graph_replay = true;
    } else if (arg == "--workload-proxy" && i + 1 < argc) {
      args.workload_proxy = argv[++i];
    } else if (arg == "--resident-lifetime") {
      args.resident_lifetime = true;
    } else if (arg == "--workspace-arena") {
      args.workspace_arena = true;
    } else if (arg == "--adaptive-grouped-scheduler") {
      args.adaptive_grouped_scheduler = true;
    } else if (arg == "--streaming-overlap") {
      args.streaming_overlap = true;
    } else if (arg == "--release-gate" && i + 1 < argc) {
      args.release_gate = argv[++i];
    } else if (arg == "--verification-amortization" && i + 1 < argc) {
      args.verification_amortization = argv[++i];
    } else if (arg == "--require-adaptive-execution") {
      args.require_adaptive_execution = true;
    } else if (arg == "--residue-channel-fusion") {
      args.residue_channel_fusion = true;
    } else if (arg == "--oneshot" || arg == "--one-shot") {
      args.oneshot = true;
    } else if (arg == "--transient-uniform-small-inputs" ||
               arg == "--transient-uniform-small-i8-inputs") {
      args.transient_uniform_small_inputs = true;
    } else if (arg == "--native-to-rns-bridge" || arg == "--force-native-to-rns-bridge") {
      args.native_to_rns_bridge = true;
    } else if (arg == "--vector-to-rns-chain" || arg == "--vector-native-to-rns-chain") {
      args.vector_to_rns_chain = true;
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
          << "                  [--output-ld-padding N]\n"
          << "                  [--tile-m M] [--tile-n N]\n"
          << "                  [--bound-mode global|per-tile]\n"
          << "                  [--input-profile uniform-small|adaptive-bands]\n"
          << "                  [--bound-source static-profile|input-scan]\n"
          << "                  [--prefix-policy minimum-proven|fixed-requested] [--max-prefix N]\n"
          << "                  [--exact-wide-limbs 1..32]\n"
          << "                  [--residue-chain-length N]\n"
          << "                  [--residue-chain-final-export]\n"
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
          << "                  [--require-adaptive-execution]\n"
          << "                  [--residue-channel-fusion]\n"
          << "                  [--oneshot]\n"
          << "                  [--transient-uniform-small-inputs]\n"
          << "                  [--native-to-rns-bridge]\n"
          << "                  [--vector-to-rns-chain]\n"
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
  if (args.output_ld_padding < 0) {
    usage_error("--output-ld-padding must be nonnegative");
  }
  if (args.n > std::numeric_limits<int64_t>::max() - args.output_ld_padding) {
    usage_error("--output-ld-padding makes output leading dimension overflow int64");
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
  if (args.native_to_rns_bridge) {
    if (!bounded_benchmark_semantics(args.semantics)) {
      usage_error("--native-to-rns-bridge is only valid for bounded-i64 or bounded-u64 semantics");
    }
    if (args.backend != RNS8_BACKEND_AUTO) {
      usage_error("--native-to-rns-bridge requires --backend auto so the AUTO/direct-HIP conversion hook is active");
    }
    if (args.bound_mode != BoundMode::Global) {
      usage_error("--native-to-rns-bridge currently requires --bound-mode global");
    }
    if (args.oneshot || args.reuse_packed_inputs || args.vector_alu_baseline ||
        args.transient_uniform_small_inputs || args.wrap64_rocwmma_candidate) {
      usage_error("--native-to-rns-bridge cannot be combined with one-shot, reuse, vector-baseline, transient, or wrap64 modes");
    }
    if (args.residue_chain_length != 1) {
      usage_error("--native-to-rns-bridge cannot be combined with --residue-chain-length > 1");
    }
  }
  if (args.vector_to_rns_chain) {
    if (!bounded_benchmark_semantics(args.semantics)) {
      usage_error("--vector-to-rns-chain is only valid for bounded-i64 or bounded-u64 semantics");
    }
    if (args.backend != RNS8_BACKEND_AUTO) {
      usage_error("--vector-to-rns-chain requires --backend auto so the producer/consumer chain is explicit");
    }
    if (args.bound_mode != BoundMode::Global) {
      usage_error("--vector-to-rns-chain currently requires --bound-mode global");
    }
    if (args.bound_source != BoundSource::StaticProfile) {
      usage_error("--vector-to-rns-chain currently requires --bound-source static-profile");
    }
    if (args.input_profile != InputProfile::UniformSmall) {
      usage_error("--vector-to-rns-chain currently requires --input-profile uniform-small");
    }
    if (args.oneshot || args.vector_alu_baseline || args.transient_uniform_small_inputs ||
        args.native_to_rns_bridge || args.wrap64_rocwmma_candidate) {
      usage_error(
          "--vector-to-rns-chain cannot be combined with one-shot, vector-baseline, transient, "
          "native-to-RNS bridge, or wrap64 modes");
    }
    if (args.reuse_packed_inputs && (!args.reuse_packed_b || args.reuse_packed_a)) {
      usage_error("--vector-to-rns-chain only supports --reuse-packed-b for the Direct-HIP consumer input");
    }
    if (args.residue_chain_length != 1) {
      usage_error("--vector-to-rns-chain cannot be combined with --residue-chain-length > 1");
    }
  }
  if (args.transient_uniform_small_inputs) {
    if (!bounded_benchmark_semantics(args.semantics)) {
      usage_error("--transient-uniform-small-inputs is only valid for bounded-i64 or bounded-u64 semantics");
    }
    if (args.backend != RNS8_BACKEND_HIP_DIRECT) {
      usage_error("--transient-uniform-small-inputs requires --backend hip-direct");
    }
    if (args.input_profile != InputProfile::UniformSmall) {
      usage_error("--transient-uniform-small-inputs requires --input-profile uniform-small");
    }
    if (args.bound_mode != BoundMode::Global) {
      usage_error("--transient-uniform-small-inputs requires --bound-mode global");
    }
    if (args.oneshot || args.reuse_packed_inputs || args.vector_alu_baseline || args.wrap64_rocwmma_candidate) {
      usage_error("--transient-uniform-small-inputs cannot be combined with oneshot, reuse, vector, or wrap64 modes");
    }
    if (args.residue_chain_length != 1) {
      usage_error("--transient-uniform-small-inputs cannot be combined with --residue-chain-length > 1");
    }
    if (args.prefix_policy != PrefixPolicy::FixedRequested ||
        (args.max_prefix_override != 0 && args.max_prefix_override != RNS8_DEFAULT_BOUNDED_PREFIX)) {
      usage_error(
          "--transient-uniform-small-inputs requires --prefix-policy fixed-requested and default bounded prefix 9");
    }
  }
  if (args.residue_channel_fusion) {
    if (!bounded_benchmark_semantics(args.semantics)) {
      usage_error("--residue-channel-fusion is only valid for bounded-i64 or bounded-u64 semantics");
    }
    if (args.backend != RNS8_BACKEND_HIP_DIRECT) {
      usage_error("--residue-channel-fusion requires --backend hip-direct");
    }
    if (args.bound_mode != BoundMode::Global || args.input_profile != InputProfile::UniformSmall) {
      usage_error("--residue-channel-fusion requires global uniform-small bounded inputs");
    }
    if (args.oneshot || args.reuse_packed_inputs || args.vector_alu_baseline || args.wrap64_rocwmma_candidate ||
        args.transient_uniform_small_inputs || args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      usage_error("--residue-channel-fusion cannot be combined with oneshot, reuse, transient, vector, or wrap64 modes");
    }
    if (args.residue_chain_length != 1) {
      usage_error("--residue-channel-fusion cannot be combined with --residue-chain-length > 1");
    }
    if (args.prefix_policy != PrefixPolicy::FixedRequested ||
        (args.max_prefix_override != 0 && args.max_prefix_override != RNS8_DEFAULT_BOUNDED_PREFIX)) {
      usage_error("--residue-channel-fusion requires --prefix-policy fixed-requested and default bounded prefix 9");
    }
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
  if (args.residue_chain_final_export && args.residue_chain_length <= 1) {
    usage_error("--residue-chain-final-export requires --residue-chain-length > 1");
  }
  if (args.host_api_batch_size == 0) {
    usage_error("--host-api-batch-size must be positive");
  }
  if (args.grouped_dispatch_tasks == 0) {
    usage_error("--grouped-dispatch must be positive");
  }
  if (host_api_batch_requested(args) && grouped_dispatch_requested(args)) {
    usage_error("--host-api-batch-size > 1 and --grouped-dispatch > 1 cannot be combined");
  }
  if (args.modulus_set.empty() ||
      (args.modulus_set != "default" && args.modulus_set.rfind("experimental:", 0) != 0)) {
    usage_error("--modulus-set must be default or experimental:NAME");
  }
  if (args.tile_shape_variant.empty()) {
    usage_error("--tile-shape-variant must not be empty");
  }
  if (args.export_variant.empty()) {
    usage_error("--export-variant must not be empty");
  }
  if (args.reconstruction_variant.empty()) {
    usage_error("--reconstruction-variant must not be empty");
  }
  if (args.workload_proxy.empty()) {
    usage_error("--workload-proxy must not be empty");
  }
  if (args.release_gate.empty()) {
    usage_error("--release-gate must not be empty");
  }
  if (args.verification_amortization.empty()) {
    usage_error("--verification-amortization must not be empty");
  }
  if (host_api_batch_requested(args)) {
    if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
      usage_error("--host-api-batch-size > 1 is not supported for wrap-u64 in this benchmark mode");
    }
    if (args.bound_mode != BoundMode::Global || args.bound_source != BoundSource::StaticProfile) {
      usage_error("--host-api-batch-size > 1 currently requires global static-profile bounds");
    }
    if (args.input_profile != InputProfile::UniformSmall) {
      usage_error("--host-api-batch-size > 1 currently requires --input-profile uniform-small");
    }
    if (args.oneshot || args.reuse_packed_inputs || args.vector_alu_baseline ||
        args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64 || args.transient_uniform_small_inputs ||
        args.native_to_rns_bridge || args.vector_to_rns_chain || args.wrap64_rocwmma_candidate ||
        args.residue_channel_fusion) {
      usage_error(
          "--host-api-batch-size > 1 cannot be combined with one-shot, reuse, vector, transient, bridge, chain, "
          "wrap64 candidate, or residue-fusion modes");
    }
    if (args.residue_chain_length != 1) {
      usage_error("--host-api-batch-size > 1 cannot be combined with --residue-chain-length > 1");
    }
  }
  if (grouped_dispatch_requested(args)) {
#if !RNS8_CONFIGURED_HIP_ENABLED
    usage_error("--grouped-dispatch > 1 requires a HIP-enabled benchmark build");
#endif
    if (args.backend != RNS8_BACKEND_HIP_DIRECT) {
      usage_error("--grouped-dispatch > 1 currently requires --backend hip-direct");
    }
    if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
      usage_error("--grouped-dispatch > 1 is not supported for wrap-u64 in this benchmark mode");
    }
    if (args.bound_mode != BoundMode::Global || args.bound_source != BoundSource::StaticProfile) {
      usage_error("--grouped-dispatch > 1 currently requires global static-profile bounds");
    }
    if (args.input_profile != InputProfile::UniformSmall) {
      usage_error("--grouped-dispatch > 1 currently requires --input-profile uniform-small");
    }
    if (args.oneshot || args.reuse_packed_inputs || args.vector_alu_baseline ||
        args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64 || args.transient_uniform_small_inputs ||
        args.native_to_rns_bridge || args.vector_to_rns_chain || args.wrap64_rocwmma_candidate ||
        args.residue_channel_fusion || args.hip_graph_replay) {
      usage_error(
          "--grouped-dispatch > 1 cannot be combined with one-shot, reuse, vector, transient, bridge, chain, "
          "wrap64 candidate, residue-fusion, or graph-replay modes");
    }
    if (args.residue_chain_length != 1) {
      usage_error("--grouped-dispatch > 1 cannot be combined with --residue-chain-length > 1");
    }
  }
  if (args.hip_graph_replay) {
#if !RNS8_CONFIGURED_HIP_ENABLED
    usage_error("--hip-graph-replay requires a HIP-enabled benchmark build");
#endif
    if (args.residue_chain_final_export) {
      usage_error("--hip-graph-replay cannot be combined with --residue-chain-final-export");
    }
    if (args.backend != RNS8_BACKEND_HIP_DIRECT) {
      usage_error("--hip-graph-replay requires --backend hip-direct");
    }
    if (!rns_chain_benchmark_semantics(args.semantics)) {
      usage_error("--hip-graph-replay is only valid for bounded or exact-wide RNS semantics");
    }
    if (args.residue_chain_length <= 1) {
      usage_error("--hip-graph-replay requires --residue-chain-length > 1 so output stays residue-current");
    }
    if (!args.reuse_packed_inputs || !args.reuse_packed_a || !args.reuse_packed_b) {
      usage_error("--hip-graph-replay requires --reuse-packed-inputs so graph replay has stable resident A/B inputs");
    }
    if (args.next_op_hint != NextOpHint::RnsGemm) {
      usage_error("--hip-graph-replay requires --next-op-hint rns-gemm");
    }
    if (args.bound_mode != BoundMode::Global || args.bound_source != BoundSource::StaticProfile) {
      usage_error("--hip-graph-replay currently requires global static-profile bounds");
    }
    if (args.oneshot || grouped_task_executor_requested(args) || args.vector_alu_baseline ||
        args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64 || args.transient_uniform_small_inputs ||
        args.native_to_rns_bridge || args.vector_to_rns_chain || args.wrap64_rocwmma_candidate ||
        args.residue_channel_fusion) {
      usage_error(
          "--hip-graph-replay cannot be combined with one-shot, host/grouped batching, vector, transient, bridge, "
          "vector-to-RNS chain, wrap64 candidate, or residue-fusion modes");
    }
  }
  if (args.residue_chain_length > 1) {
    if (!rns_chain_benchmark_semantics(args.semantics)) {
      usage_error("--residue-chain-length > 1 is only valid for bounded or exact-wide RNS semantics");
    }
    if (args.residue_chain_final_export && args.next_op_hint == NextOpHint::RnsGemm) {
      usage_error("--residue-chain-final-export cannot use --next-op-hint rns-gemm");
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
         args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64 || args.vector_to_rns_chain)
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

bool bounded_native_a_reuse_b_u64_large_colpair(const Args& args) {
  return args.semantics == BenchSemantics::BoundedU64 && !bounded_native_a_reuse_b_uniform_small_a(args) &&
         args.m >= 512 && args.n >= 512 && args.k >= 512;
}

bool bounded_uniform_small_i8_ab_reuse_a_requested(const Args& args) {
  return !args.oneshot && bounded_benchmark_semantics(args.semantics) && args.reuse_packed_a &&
         !args.reuse_packed_b && args.backend == RNS8_BACKEND_HIP_DIRECT &&
         args.bound_mode == BoundMode::Global && args.residue_chain_length == 1 &&
         args.input_profile == InputProfile::UniformSmall &&
         args.prefix_policy == PrefixPolicy::FixedRequested &&
         (args.max_prefix_override == 0 || args.max_prefix_override == RNS8_DEFAULT_BOUNDED_PREFIX);
}

bool bounded_native_b_reuse_a_u64_large_colpair_requested(const Args& args) {
  return !args.oneshot && args.semantics == BenchSemantics::BoundedU64 && args.reuse_packed_a &&
         !args.reuse_packed_b && args.backend == RNS8_BACKEND_HIP_DIRECT &&
         args.bound_mode == BoundMode::Global && args.residue_chain_length == 1 &&
         args.input_profile != InputProfile::UniformSmall &&
         args.prefix_policy == PrefixPolicy::FixedRequested &&
         (args.max_prefix_override == 0 || args.max_prefix_override == RNS8_DEFAULT_BOUNDED_PREFIX) &&
         args.m >= 512 && args.n >= 512 && args.k >= 512;
}

bool bounded_uniform_small_i8_ab_transient_requested(const Args& args) {
  return !args.oneshot && args.transient_uniform_small_inputs && bounded_benchmark_semantics(args.semantics) &&
         !args.reuse_packed_inputs && args.backend == RNS8_BACKEND_HIP_DIRECT &&
         args.bound_mode == BoundMode::Global && args.residue_chain_length == 1 &&
         args.input_profile == InputProfile::UniformSmall &&
         args.prefix_policy == PrefixPolicy::FixedRequested &&
         (args.max_prefix_override == 0 || args.max_prefix_override == RNS8_DEFAULT_BOUNDED_PREFIX);
}

bool bounded_residue_channel_fusion_requested(const Args& args) {
  return !args.oneshot && args.residue_channel_fusion && bounded_benchmark_semantics(args.semantics) &&
         !args.reuse_packed_inputs && args.backend == RNS8_BACKEND_HIP_DIRECT &&
         args.bound_mode == BoundMode::Global && args.residue_chain_length == 1 &&
         args.input_profile == InputProfile::UniformSmall &&
         args.prefix_policy == PrefixPolicy::FixedRequested &&
         (args.max_prefix_override == 0 || args.max_prefix_override == RNS8_DEFAULT_BOUNDED_PREFIX);
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
  if (bounded_residue_channel_fusion_requested(args)) {
    return "rns8_bench_residue_channel_fusion_path";
  }
  if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    return "rns8_bench_uniform_small_i8_ab_transient_path";
  }
  if (bounded_native_a_reuse_b_requested(args)) {
    if (bounded_native_a_reuse_b_uniform_small_a(args)) {
      return "rns8_bench_uniform_small_i8_ab_reuse_b_path";
    }
    return "rns8_bench_native_a_reuse_b_path";
  }
  if (bounded_uniform_small_i8_ab_reuse_a_requested(args)) {
    return "rns8_bench_uniform_small_i8_ab_reuse_a_path";
  }
  if (bounded_native_b_reuse_a_u64_large_colpair_requested(args)) {
    return "rns8_bench_native_b_reuse_a_path";
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

bool native_to_rns_bridge_requested(const Args& args) {
  return args.native_to_rns_bridge;
}

bool vector_to_rns_chain_requested(const Args& args) {
  return args.vector_to_rns_chain;
}

bool host_api_batch_requested(const Args& args) {
  return args.host_api_batch_size > 1;
}

bool grouped_dispatch_requested(const Args& args) {
  return args.grouped_dispatch_tasks > 1;
}

bool grouped_task_executor_requested(const Args& args) {
  return host_api_batch_requested(args) || grouped_dispatch_requested(args);
}

uint32_t measured_task_count(const Args& args) {
  return grouped_dispatch_requested(args) ? args.grouped_dispatch_tasks : args.host_api_batch_size;
}

bool hip_graph_replay_requested(const Args& args) {
  return args.hip_graph_replay;
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
  if (native_to_rns_bridge_requested(args)) {
    return "auto_native_to_rns_bridge";
  }
  if (vector_to_rns_chain_requested(args)) {
    return "vector_native_to_direct_rns_chain";
  }
  if (grouped_dispatch_requested(args)) {
    return "benchmark_grouped_dispatch_evidence";
  }
  if (host_api_batch_requested(args)) {
    return "benchmark_host_api_batch";
  }
  if (hip_graph_replay_requested(args)) {
    return "hip_graph_replay_resident_rns_chain";
  }
  if (residue_chain_final_export_requested(args)) {
    return "residue_chain_final_host_export";
  }
  if (residue_current_output_mode(args)) {
    return "residue_current_rns_chain";
  }
  if (args.wrap64_rocwmma_candidate) {
    return "internal_wrap64_rocwmma_candidate";
  }
  if (bounded_residue_channel_fusion_requested(args)) {
    return "residue_channel_fusion_native_inputs";
  }
  if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    return "transient_uniform_small_i8_ab_inputs";
  }
  if (bounded_native_a_reuse_b_requested(args)) {
    if (bounded_native_a_reuse_b_uniform_small_a(args)) {
      return "transient_uniform_small_i8_a_resident_i8_b_reuse";
    }
    return "transient_native_a_resident_b_reuse";
  }
  if (bounded_uniform_small_i8_ab_reuse_a_requested(args)) {
    return "transient_uniform_small_i8_b_resident_i8_a_reuse";
  }
  if (bounded_native_b_reuse_a_u64_large_colpair_requested(args)) {
    return "transient_native_b_resident_a_reuse";
  }
  if (finite_benchmark_semantics(args.semantics) && args.reuse_packed_b && !args.reuse_packed_a &&
      args.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "transient_native_a_resident_b_reuse";
  }
  return "persistent_resident_matrices";
}

std::string benchmark_env_value(const char* name) {
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

bool benchmark_env_flag_disabled(const char* name) {
  const std::string value = benchmark_env_value(name);
  return value == "0" || value == "false" || value == "FALSE" || value == "off" ||
         value == "OFF" || value == "no" || value == "NO";
}

bool benchmark_env_flag_enabled(const char* name) {
  const std::string value = benchmark_env_value(name);
  return value == "1" || value == "true" || value == "TRUE" || value == "on" ||
         value == "ON" || value == "yes" || value == "YES";
}

const char* direct_hip_export_staging_policy(const Args& args, rns8_backend_kind selected_backend) {
  if (selected_backend != RNS8_BACKEND_HIP_DIRECT) {
    return "not_applicable";
  }
  if (benchmark_env_flag_disabled("RNS8_HIP_PINNED_EXPORT_STAGING")) {
    return "disabled_by_RNS8_HIP_PINNED_EXPORT_STAGING";
  }
  if (benchmark_env_flag_enabled("RNS8_HIP_PINNED_EXPORT_STAGING")) {
    return "forced_for_large_outputs_by_RNS8_HIP_PINNED_EXPORT_STAGING";
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return "wrap64_forced_only_pending_padded_staging_evidence";
  }
  if (args.semantics == BenchSemantics::ExactWideSigned) {
    return "exact_wide_signed_forced_only_local_gfx1100_padded_staging_loses";
  }
  return "large_padded_outputs_only_default";
}

constexpr uint64_t kDirectHipPinnedExportStagingThresholdBytes = 64u * 1024u;

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

bool all_zero_direct_hip_input_pack_elided(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend) {
  return selected_backend == RNS8_BACKEND_HIP_DIRECT && bounded_benchmark_semantics(args.semantics) &&
         args.bound_mode == BoundMode::PerTile && !args.oneshot && args.residue_chain_length == 1 &&
         !args.reuse_packed_inputs && result.schedule_info_available && result.schedule_info.tile_count != 0 &&
         result.zero_output_tile_count == result.schedule_info.tile_count;
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

void force_native_to_rns_bridge_after_pack(const Args& args, rns8_matrix* a_matrix, rns8_matrix* b_matrix) {
  if (!native_to_rns_bridge_requested(args)) {
    return;
  }
  const rns8_status status = rns8::detail::force_native_to_rns_bridge_inputs(a_matrix, b_matrix);
  if (status != RNS8_SUCCESS) {
    fail_status("force_native_to_rns_bridge_inputs", status);
  }
}

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

rns8_status run_direct_hip_resident_rns_chain_no_sync(
    const rns8_plan* plan,
    rns8_matrix* a_matrix,
    rns8_matrix* b_matrix,
    rns8_matrix* c_matrix,
    rns8_matrix* scratch_matrix,
    uint32_t chain_length,
    rns8_matrix** latest_output_matrix,
    void* stream = nullptr) {
  if (!plan || !a_matrix || !b_matrix || !c_matrix || chain_length == 0 || !latest_output_matrix) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (chain_length > 1 && !scratch_matrix) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_matrix* lhs_matrix = a_matrix;
  rns8_matrix* out_matrix = c_matrix;
  rns8_matrix* final_output_matrix = c_matrix;
  for (uint32_t chain_index = 0; chain_index < chain_length; ++chain_index) {
    const rns8_status status = rns8::detail::hip_direct_gemm_rns_matrix_launch_current_device_no_sync(
        plan, lhs_matrix, b_matrix, out_matrix, stream);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    final_output_matrix = out_matrix;
    lhs_matrix = out_matrix;
    out_matrix = out_matrix == c_matrix ? scratch_matrix : c_matrix;
  }
  *latest_output_matrix = final_output_matrix;
  return RNS8_SUCCESS;
}

void run_hip_graph_replay_resident_chain(
    const Args& args,
    BenchmarkResult& result,
    const rns8_plan* plan,
    rns8_matrix* a_matrix,
    rns8_matrix* b_matrix,
    rns8_matrix* c_matrix,
    rns8_matrix* scratch_matrix,
    rns8_matrix*& latest_output_matrix) {
  result.hip_graph_replay_requested = true;
  result.hip_graph_replay_scope = "direct_hip_reused_inputs_residue_current_rns_chain";
  result.hip_graph_replay_caveat =
      "captures resident Direct-HIP RNS GEMM launches only; A/B prepack setup and final checksum export are outside "
      "the graph";

  HipGraphReplayState graph_state{};
  std::string graph_error;
  rns8_status status = capture_hip_graph_replay(
      args.device_id,
      [&](void* stream) {
        return run_direct_hip_resident_rns_chain_no_sync(
            plan,
            a_matrix,
            b_matrix,
            c_matrix,
            scratch_matrix,
            args.residue_chain_length,
            &latest_output_matrix,
            stream);
      },
      graph_state,
      result.hip_graph_capture_us,
      result.hip_graph_instantiate_us,
      graph_error);
  if (status != RNS8_SUCCESS) {
    result.hip_graph_replay_status = "capture_failed";
    result.hip_graph_replay_caveat = graph_error;
    destroy_hip_graph_replay_state(graph_state);
    if (!graph_error.empty()) {
      std::cerr << graph_error << "\n";
    }
    fail_status("hip graph capture", status);
  }

  result.hip_graph_replay_available = true;
  result.hip_graph_replay_used = true;
  result.hip_graph_replay_status = "available";

  for (uint32_t r = 0; r < args.warmups; ++r) {
    status = launch_hip_graph_replay(args.device_id, graph_state, graph_error);
    if (status != RNS8_SUCCESS) {
      result.hip_graph_replay_status = "warmup_launch_failed";
      result.hip_graph_replay_caveat = graph_error;
      destroy_hip_graph_replay_state(graph_state);
      if (!graph_error.empty()) {
        std::cerr << graph_error << "\n";
      }
      fail_status("hip graph warmup launch", status);
    }
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    const auto repeat_start = std::chrono::steady_clock::now();
    status = launch_hip_graph_replay(args.device_id, graph_state, graph_error);
    const auto repeat_end = std::chrono::steady_clock::now();
    if (status != RNS8_SUCCESS) {
      result.hip_graph_replay_status = "measured_launch_failed";
      result.hip_graph_replay_caveat = graph_error;
      destroy_hip_graph_replay_state(graph_state);
      if (!graph_error.empty()) {
        std::cerr << graph_error << "\n";
      }
      fail_status("hip graph measured launch", status);
    }
    const uint64_t elapsed = elapsed_us(repeat_start, repeat_end);
    result.samples.pack_us.push_back(0);
    result.samples.gemm_us.push_back(elapsed);
    result.samples.export_us.push_back(0);
    result.samples.end_to_end_us.push_back(elapsed);
  }
  result.hip_graph_launch_count = graph_state.launch_count;
  destroy_hip_graph_replay_state(graph_state);
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

const char* c_semantics_name(rns8_semantics semantics) {
  switch (semantics) {
    case RNS8_BOUNDED_I64:
      return "bounded_i64";
    case RNS8_BOUNDED_U64:
      return "bounded_u64";
    case RNS8_EXACT_WIDE_SIGNED:
      return "exact_wide_signed";
    case RNS8_EXACT_WIDE_UNSIGNED:
      return "exact_wide_unsigned";
    case RNS8_WRAP_U64_MOD_2_64:
      return "wrap_u64_mod_2_64";
    case RNS8_FINITE_RING_U8:
      return "finite_ring_u8";
    case RNS8_FINITE_FIELD_U8:
      return "finite_field_u8";
  }
  return "unknown";
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

int64_t output_logical_ld(const Args& args) {
  if (args.output_ld_padding < 0) {
    usage_error("--output-ld-padding must be nonnegative");
  }
  if (args.n > std::numeric_limits<int64_t>::max() - args.output_ld_padding) {
    usage_error("--output-ld-padding makes output leading dimension overflow int64");
  }
  return args.n + args.output_ld_padding;
}

std::size_t output_elements(const Args& args, const char* label) {
  return checked_elements(args.m, output_logical_ld(args), label);
}

std::size_t output_limb_elements(const Args& args, uint32_t limb_count, const char* label) {
  return checked_limb_elements(args.m, output_logical_ld(args), limb_count, label);
}

const char* output_destination_layout(const Args& args) {
  return args.output_ld_padding == 0 ? "contiguous_row_major" : "padded_row_major";
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
  if (vector_to_rns_chain_requested(args)) {
    const auto u_n = static_cast<uint64_t>(args.n);
    if (u_n > std::numeric_limits<uint64_t>::max() / 16u) {
      usage_error("n is too large for the vector-to-RNS chain benchmark bound");
    }
    const uint64_t chain_multiplier = u_n * 16u;
    if (chain_multiplier != 0 && bound > std::numeric_limits<uint64_t>::max() / chain_multiplier) {
      usage_error("vector-to-RNS chain benchmark bound exceeds uint64_t");
    }
    bound *= chain_multiplier;
  }
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

uint64_t vector_to_rns_chain_producer_bound(const Args& args) {
  const uint64_t max_term = 16u * 16u;
  const auto u_k = static_cast<uint64_t>(args.k);
  if (u_k > std::numeric_limits<uint64_t>::max() / max_term) {
    usage_error("k is too large for the vector-to-RNS producer benchmark bound");
  }
  const uint64_t bound = u_k * max_term;
  if (args.semantics == BenchSemantics::BoundedI64 &&
      bound > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    usage_error("bounded-i64 vector-to-RNS producer bound exceeds int64 output range");
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

struct TileBoundScanResult {
  std::vector<uint64_t> bounds{};
  std::vector<uint8_t> zero_a_rows{};
  std::vector<uint8_t> zero_b_cols{};
};

template <typename T>
std::vector<uint8_t> compute_nonzero_a_rows(const Args& args, const std::vector<T>& A) {
  std::vector<uint8_t> rows(static_cast<std::size_t>(args.m), 0);
  for (int64_t row = 0; row < args.m; ++row) {
    for (int64_t kk = 0; kk < args.k; ++kk) {
      if (A[row_major_index(row, kk, args.k, "A")] != T{}) {
        rows[static_cast<std::size_t>(row)] = 1;
        break;
      }
    }
  }
  return rows;
}

template <typename T>
std::vector<uint8_t> compute_nonzero_b_cols(const Args& args, const std::vector<T>& B) {
  std::vector<uint8_t> cols(static_cast<std::size_t>(args.n), 0);
  for (int64_t kk = 0; kk < args.k; ++kk) {
    for (int64_t col = 0; col < args.n; ++col) {
      if (B[row_major_index(kk, col, args.n, "B")] != T{}) {
        cols[static_cast<std::size_t>(col)] = 1;
      }
    }
  }
  return cols;
}

bool range_has_nonzero_flag(const std::vector<uint8_t>& flags, int64_t begin, int64_t end) {
  for (int64_t index = begin; index < end; ++index) {
    if (flags[static_cast<std::size_t>(index)] != 0) {
      return true;
    }
  }
  return false;
}

std::vector<uint8_t> invert_nonzero_flags(const std::vector<uint8_t>& nonzero_flags) {
  std::vector<uint8_t> zero_flags(nonzero_flags.size(), 0);
  for (std::size_t index = 0; index < nonzero_flags.size(); ++index) {
    zero_flags[index] = nonzero_flags[index] == 0 ? 1u : 0u;
  }
  return zero_flags;
}

uint64_t count_set_flags(const std::vector<uint8_t>& flags) {
  uint64_t count = 0;
  for (const uint8_t value : flags) {
    count += value != 0 ? 1u : 0u;
  }
  return count;
}

uint64_t zero_row_col_product_count(int64_t m, int64_t n, uint64_t zero_a_rows, uint64_t zero_b_cols) {
  const uint64_t rows = static_cast<uint64_t>(m);
  const uint64_t cols = static_cast<uint64_t>(n);
  const uint64_t nonzero_a_rows = rows - zero_a_rows;
  return zero_a_rows * cols + nonzero_a_rows * zero_b_cols;
}

TileBoundScanResult compute_i64_tile_bounds(
    const Args& args,
    const std::vector<int64_t>& A,
    const std::vector<int64_t>& B) {
  const auto u_k = static_cast<uint64_t>(args.k);
  if (u_k > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 256u) {
    usage_error("bounded-i64 per-tile benchmark k is too large for exact int64 tile-bound prepass");
  }
  const uint64_t tile_rows = ceil_div_i64_u32(args.m, args.tile_m);
  const uint64_t tile_cols = ceil_div_i64_u32(args.n, args.tile_n);
  TileBoundScanResult scan{};
  scan.bounds.assign(static_cast<std::size_t>(checked_tile_count(args)), 0);
  const std::vector<uint8_t> nonzero_a_rows = compute_nonzero_a_rows(args, A);
  const std::vector<uint8_t> nonzero_b_cols = compute_nonzero_b_cols(args, B);
  scan.zero_a_rows = invert_nonzero_flags(nonzero_a_rows);
  scan.zero_b_cols = invert_nonzero_flags(nonzero_b_cols);
  for (uint64_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
    const int64_t row_begin = static_cast<int64_t>(tile_row * static_cast<uint64_t>(args.tile_m));
    const int64_t row_end = std::min<int64_t>(args.m, row_begin + static_cast<int64_t>(args.tile_m));
    for (uint64_t tile_col = 0; tile_col < tile_cols; ++tile_col) {
      const int64_t col_begin = static_cast<int64_t>(tile_col * static_cast<uint64_t>(args.tile_n));
      const int64_t col_end = std::min<int64_t>(args.n, col_begin + static_cast<int64_t>(args.tile_n));
      if (!range_has_nonzero_flag(nonzero_a_rows, row_begin, row_end) ||
          !range_has_nonzero_flag(nonzero_b_cols, col_begin, col_end)) {
        continue;
      }
      uint64_t& tile_max = scan.bounds[static_cast<std::size_t>(tile_row * tile_cols + tile_col)];
      for (int64_t row = row_begin; row < row_end; ++row) {
        if (nonzero_a_rows[static_cast<std::size_t>(row)] == 0) {
          continue;
        }
        for (int64_t col = col_begin; col < col_end; ++col) {
          if (nonzero_b_cols[static_cast<std::size_t>(col)] == 0) {
            continue;
          }
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
  return scan;
}

TileBoundScanResult compute_u64_tile_bounds(
    const Args& args,
    const std::vector<uint64_t>& A,
    const std::vector<uint64_t>& B) {
  const auto u_k = static_cast<uint64_t>(args.k);
  if (u_k > std::numeric_limits<uint64_t>::max() / 256u) {
    usage_error("bounded-u64 per-tile benchmark k is too large for exact uint64 tile-bound prepass");
  }
  const uint64_t tile_rows = ceil_div_i64_u32(args.m, args.tile_m);
  const uint64_t tile_cols = ceil_div_i64_u32(args.n, args.tile_n);
  TileBoundScanResult scan{};
  scan.bounds.assign(static_cast<std::size_t>(checked_tile_count(args)), 0);
  const std::vector<uint8_t> nonzero_a_rows = compute_nonzero_a_rows(args, A);
  const std::vector<uint8_t> nonzero_b_cols = compute_nonzero_b_cols(args, B);
  scan.zero_a_rows = invert_nonzero_flags(nonzero_a_rows);
  scan.zero_b_cols = invert_nonzero_flags(nonzero_b_cols);
  for (uint64_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
    const int64_t row_begin = static_cast<int64_t>(tile_row * static_cast<uint64_t>(args.tile_m));
    const int64_t row_end = std::min<int64_t>(args.m, row_begin + static_cast<int64_t>(args.tile_m));
    for (uint64_t tile_col = 0; tile_col < tile_cols; ++tile_col) {
      const int64_t col_begin = static_cast<int64_t>(tile_col * static_cast<uint64_t>(args.tile_n));
      const int64_t col_end = std::min<int64_t>(args.n, col_begin + static_cast<int64_t>(args.tile_n));
      if (!range_has_nonzero_flag(nonzero_a_rows, row_begin, row_end) ||
          !range_has_nonzero_flag(nonzero_b_cols, col_begin, col_end)) {
        continue;
      }
      uint64_t& tile_max = scan.bounds[static_cast<std::size_t>(tile_row * tile_cols + tile_col)];
      for (int64_t row = row_begin; row < row_end; ++row) {
        if (nonzero_a_rows[static_cast<std::size_t>(row)] == 0) {
          continue;
        }
        for (int64_t col = col_begin; col < col_end; ++col) {
          if (nonzero_b_cols[static_cast<std::size_t>(col)] == 0) {
            continue;
          }
          uint64_t acc = 0;
          for (int64_t kk = 0; kk < args.k; ++kk) {
            acc += A[row_major_index(row, kk, args.k, "A")] * B[row_major_index(kk, col, args.n, "B")];
          }
          tile_max = std::max(tile_max, acc);
        }
      }
    }
  }
  return scan;
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
  auto scan = compute_bounds();
  const auto scan_end = std::chrono::steady_clock::now();
  result.tile_bound_scan_us = elapsed_us(scan_start, scan_end);
  result.tile_bound_scan_available = true;
  result.zero_a_rows = std::move(scan.zero_a_rows);
  result.zero_b_cols = std::move(scan.zero_b_cols);
  result.zero_a_row_proof_count = count_set_flags(result.zero_a_rows);
  result.zero_b_col_proof_count = count_set_flags(result.zero_b_cols);
  result.zero_row_col_product_count =
      zero_row_col_product_count(
          result.zero_a_rows.empty() ? 0 : static_cast<int64_t>(result.zero_a_rows.size()),
          result.zero_b_cols.empty() ? 0 : static_cast<int64_t>(result.zero_b_cols.size()),
          result.zero_a_row_proof_count,
          result.zero_b_col_proof_count);
  record_tile_bounds(result, std::move(scan.bounds));
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

rns8_backend_kind requested_backend_for_plan_desc(const Args& args) {
  return native_to_rns_bridge_requested(args) ? RNS8_BACKEND_HIP_DIRECT : args.backend;
}

rns8_gemm_desc gemm_desc(
    const Args& args,
    uint64_t bound,
    const std::vector<uint64_t>* tile_bounds = nullptr,
    const std::vector<uint8_t>* zero_a_rows = nullptr,
    const std::vector<uint8_t>* zero_b_cols = nullptr) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = c_semantics(args.semantics);
  desc.bound_kind = bound_kind(args);
  desc.requested_backend = requested_backend_for_plan_desc(args);
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
    if (bounded_benchmark_semantics(args.semantics) && zero_a_rows && zero_b_cols &&
        zero_a_rows->size() == static_cast<std::size_t>(args.m) &&
        zero_b_cols->size() == static_cast<std::size_t>(args.n) &&
        zero_row_col_product_count(
            args.m,
            args.n,
            count_set_flags(*zero_a_rows),
            count_set_flags(*zero_b_cols)) != 0) {
      desc.flags |= RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS;
      desc.zero_a_rows = zero_a_rows->data();
      desc.zero_a_rows_count = static_cast<uint64_t>(zero_a_rows->size());
      desc.zero_b_cols = zero_b_cols->data();
      desc.zero_b_cols_count = static_cast<uint64_t>(zero_b_cols->size());
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

std::string hip_graph_error_text(const char* operation, int code) {
#if RNS8_CONFIGURED_HIP_ENABLED
  const auto error = static_cast<hipError_t>(code);
  const char* name = hipGetErrorName(error);
  const char* text = hipGetErrorString(error);
  std::ostringstream out;
  out << operation << " failed";
  if (name && name[0] != '\0') {
    out << ": " << name;
  }
  if (text && text[0] != '\0') {
    out << " (" << text << ")";
  }
  return out.str();
#else
  (void)operation;
  (void)code;
  return "HIP graph replay requires a HIP-enabled benchmark build";
#endif
}

void destroy_hip_graph_replay_state(HipGraphReplayState& state) {
#if RNS8_CONFIGURED_HIP_ENABLED
  if (state.executable) {
    const hipError_t status = hipGraphExecDestroy(state.executable);
    if (status != hipSuccess) {
      std::cerr << hip_graph_error_text("hipGraphExecDestroy", static_cast<int>(status)) << "\n";
    }
    state.executable = nullptr;
  }
  if (state.graph) {
    const hipError_t status = hipGraphDestroy(state.graph);
    if (status != hipSuccess) {
      std::cerr << hip_graph_error_text("hipGraphDestroy", static_cast<int>(status)) << "\n";
    }
    state.graph = nullptr;
  }
  if (state.stream) {
    const hipError_t status = hipStreamDestroy(state.stream);
    if (status != hipSuccess) {
      std::cerr << hip_graph_error_text("hipStreamDestroy", static_cast<int>(status)) << "\n";
    }
    state.stream = nullptr;
  }
#else
  (void)state;
#endif
}

rns8_status capture_hip_graph_replay(
    int device_id,
    const std::function<rns8_status(void*)>& capture_body,
    HipGraphReplayState& state,
    uint64_t& capture_us,
    uint64_t& instantiate_us,
    std::string& error_text) {
#if RNS8_CONFIGURED_HIP_ENABLED
  destroy_hip_graph_replay_state(state);
  const hipError_t device_status = hipSetDevice(device_id);
  if (device_status != hipSuccess) {
    error_text = hip_graph_error_text("hipSetDevice", static_cast<int>(device_status));
    return RNS8_BACKEND_FAILURE;
  }
  hipError_t status = hipStreamCreateWithFlags(&state.stream, hipStreamNonBlocking);
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipStreamCreateWithFlags", static_cast<int>(status));
    return RNS8_BACKEND_FAILURE;
  }

  const auto capture_start = std::chrono::steady_clock::now();
  status = hipStreamBeginCapture(state.stream, hipStreamCaptureModeGlobal);
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipStreamBeginCapture", static_cast<int>(status));
    destroy_hip_graph_replay_state(state);
    return RNS8_BACKEND_FAILURE;
  }

  const rns8_status body_status = capture_body(static_cast<void*>(state.stream));
  hipGraph_t captured_graph = nullptr;
  status = hipStreamEndCapture(state.stream, &captured_graph);
  const auto capture_end = std::chrono::steady_clock::now();
  capture_us = elapsed_us(capture_start, capture_end);
  if (body_status != RNS8_SUCCESS) {
    if (captured_graph) {
      (void)hipGraphDestroy(captured_graph);
    }
    error_text = std::string("captured Direct-HIP graph body returned ") + rns8_status_string(body_status);
    return body_status;
  }
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipStreamEndCapture", static_cast<int>(status));
    if (captured_graph) {
      (void)hipGraphDestroy(captured_graph);
    }
    return RNS8_BACKEND_FAILURE;
  }
  state.graph = captured_graph;

  const auto instantiate_start = std::chrono::steady_clock::now();
  status = hipGraphInstantiate(&state.executable, state.graph, nullptr, nullptr, 0);
  const auto instantiate_end = std::chrono::steady_clock::now();
  instantiate_us = elapsed_us(instantiate_start, instantiate_end);
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipGraphInstantiate", static_cast<int>(status));
    destroy_hip_graph_replay_state(state);
    return RNS8_BACKEND_FAILURE;
  }
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)capture_body;
  (void)state;
  capture_us = 0;
  instantiate_us = 0;
  error_text = "HIP graph replay requires a HIP-enabled benchmark build";
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status launch_hip_graph_replay(int device_id, HipGraphReplayState& state, std::string& error_text) {
#if RNS8_CONFIGURED_HIP_ENABLED
  if (!state.executable) {
    error_text = "hip graph executable is not instantiated";
    return RNS8_INVALID_ARGUMENT;
  }
  hipError_t status = hipSetDevice(device_id);
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipSetDevice", static_cast<int>(status));
    return RNS8_BACKEND_FAILURE;
  }
  status = hipGraphLaunch(state.executable, state.stream);
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipGraphLaunch", static_cast<int>(status));
    return RNS8_BACKEND_FAILURE;
  }
  status = hipStreamSynchronize(state.stream);
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipStreamSynchronize", static_cast<int>(status));
    return RNS8_BACKEND_FAILURE;
  }
  ++state.launch_count;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)state;
  error_text = "HIP graph replay requires a HIP-enabled benchmark build";
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

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

void capture_packing_and_lowering_info(rns8_plan* plan, BenchmarkResult& result) {
  result.packing_info.struct_size = sizeof(result.packing_info);
  result.packing_info.abi_version = RNS8_ABI_VERSION;
  const rns8_status packing_status = rns8_get_plan_packing_info(plan, &result.packing_info);
  if (packing_status != RNS8_SUCCESS) {
    fail_status("rns8_get_plan_packing_info", packing_status);
  }
  result.packing_info_available = true;
  if (result.backend_info_available && result.schedule_info_available) {
    result.lowering_info =
        rns8::detail::describe_plan_lowering(result.backend_info, result.packing_info, result.schedule_info);
    result.lowering_info_available = true;
  }
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
  return bounded_benchmark_semantics(args.semantics) && args.m >= 512 && args.n >= 512 && args.k >= 512
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
  if (bounded_native_a_reuse_b_u64_large_colpair(args)) {
    return "direct_hip_native_a_u64_colpair_prefix9_reuse_b_grouped_rns_gemm_v2";
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
  if (bounded_native_a_reuse_b_uniform_small_a(args)) {
    return "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group";
  }
  if (bounded_native_a_reuse_b_u64_large_colpair(args)) {
    return "bounded_native_a_colpair_reuse_b_gemm_kernel_group";
  }
  return "bounded_native_a_reuse_b_gemm_kernel_group";
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

const char* bounded_uniform_small_i8_ab_reuse_a_kernel() {
  return "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_a_grouped_rns_gemm_v1";
}

const char* bounded_uniform_small_i8_ab_reuse_a_epilogue() {
  return "uniform_small_i8_ab_resident_a_residue_then_crt_export";
}

const char* bounded_uniform_small_i8_ab_reuse_a_event_label() {
  return "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group";
}

const char* bounded_uniform_small_i8_ab_reuse_a_pack_h2d_label() {
  return "bounded_uniform_small_i8_b_h2d";
}

const char* bounded_uniform_small_i8_ab_transient_kernel() {
  return "direct_hip_uniform_small_i8_ab_colpair_prefix9_transient_grouped_rns_gemm_v1";
}

const char* bounded_residue_channel_fusion_kernel() {
  return "direct_hip_uniform_small_i8_ab_colpair_prefix9_residue_channel_width3_experimental_v0";
}

const char* bounded_uniform_small_i8_ab_transient_epilogue() {
  return "uniform_small_i8_ab_transient_residue_then_crt_export";
}

const char* bounded_residue_channel_fusion_epilogue() {
  return "width3_residue_fusion_transient_then_crt_export";
}

const char* bounded_uniform_small_i8_ab_transient_event_label() {
  return "bounded_uniform_small_i8_ab_transient_gemm_kernel_group";
}

const char* bounded_uniform_small_i8_ab_transient_a_h2d_label() {
  return "bounded_uniform_small_i8_a_h2d";
}

const char* bounded_uniform_small_i8_ab_transient_b_h2d_label() {
  return "bounded_uniform_small_i8_b_h2d";
}

const char* bounded_native_b_reuse_a_u64_large_colpair_kernel() {
  return "direct_hip_native_b_u64_colpair_prefix9_reuse_a_grouped_rns_gemm_v1";
}

const char* bounded_native_b_reuse_a_u64_large_colpair_epilogue() {
  return "resident_a_native_b_centered_residue_then_crt_export";
}

const char* bounded_native_b_reuse_a_u64_large_colpair_event_label() {
  return "bounded_native_b_colpair_reuse_a_gemm_kernel_group";
}

const char* bounded_native_b_reuse_a_u64_large_colpair_pack_h2d_label() {
  return "bounded_native_b_h2d";
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

bool bounded_uniform_small_i8_ab_reuse_a_path(const Args& args, const BenchmarkResult& result) {
  return bounded_uniform_small_i8_ab_reuse_a_requested(args) &&
         result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT &&
         result.schedule_info_available &&
         result.schedule_info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.prefix_group_count == 1 &&
         !result.schedule_info.adaptive_prefix_active &&
         !result.schedule_info.adaptive_skip_active;
}

bool bounded_native_b_reuse_a_u64_large_colpair_path(const Args& args, const BenchmarkResult& result) {
  return bounded_native_b_reuse_a_u64_large_colpair_requested(args) &&
         result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT &&
         result.schedule_info_available &&
         result.schedule_info.min_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.max_selected_prefix == RNS8_DEFAULT_BOUNDED_PREFIX &&
         result.schedule_info.prefix_group_count == 1 &&
         !result.schedule_info.adaptive_prefix_active &&
         !result.schedule_info.adaptive_skip_active;
}

bool bounded_uniform_small_i8_ab_transient_path(const Args& args, const BenchmarkResult& result) {
  return (bounded_uniform_small_i8_ab_transient_requested(args) ||
          bounded_residue_channel_fusion_requested(args)) &&
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

const char* benchmark_execution_mode_name(const Args& args, const BenchmarkResult& result) {
  if (bounded_native_a_reuse_b_requested(args) && !bounded_native_a_reuse_b_path(args, result)) {
    return "persistent_resident_matrices";
  }
  if (bounded_uniform_small_i8_ab_reuse_a_requested(args) && !bounded_uniform_small_i8_ab_reuse_a_path(args, result)) {
    return "persistent_resident_matrices";
  }
  if (bounded_native_b_reuse_a_u64_large_colpair_requested(args) &&
      !bounded_native_b_reuse_a_u64_large_colpair_path(args, result)) {
    return "persistent_resident_matrices";
  }
  if (bounded_uniform_small_i8_ab_transient_requested(args) && !bounded_uniform_small_i8_ab_transient_path(args, result)) {
    return "persistent_resident_matrices";
  }
  return benchmark_execution_mode_name(args);
}

const char* backend_metadata_source(const Args& args, const BenchmarkResult& result) {
  if (bounded_native_a_reuse_b_requested(args) && !bounded_native_a_reuse_b_path(args, result)) {
    return "rns8_get_plan_backend_info";
  }
  if (bounded_uniform_small_i8_ab_reuse_a_requested(args) && !bounded_uniform_small_i8_ab_reuse_a_path(args, result)) {
    return "rns8_get_plan_backend_info";
  }
  if (bounded_native_b_reuse_a_u64_large_colpair_requested(args) &&
      !bounded_native_b_reuse_a_u64_large_colpair_path(args, result)) {
    return "rns8_get_plan_backend_info";
  }
  if (bounded_uniform_small_i8_ab_transient_requested(args) && !bounded_uniform_small_i8_ab_transient_path(args, result)) {
    return "rns8_get_plan_backend_info";
  }
  return backend_metadata_source(args);
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

void apply_bounded_uniform_small_i8_ab_reuse_a_backend_metadata(
    const Args& args,
    BenchmarkResult& result,
    uint64_t bound) {
  if (!bounded_uniform_small_i8_ab_reuse_a_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = bounded_uniform_small_i8_ab_reuse_a_kernel();
  const char* epilogue = bounded_uniform_small_i8_ab_reuse_a_epilogue();
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "transient_i8_b_resident_i8_a_rns_output");
  const std::string key = bounded_native_a_reuse_b_autotune_key(args, result, kernel, epilogue, bound);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_bounded_native_b_reuse_a_backend_metadata(const Args& args, BenchmarkResult& result, uint64_t bound) {
  if (!bounded_native_b_reuse_a_u64_large_colpair_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = bounded_native_b_reuse_a_u64_large_colpair_kernel();
  const char* epilogue = bounded_native_b_reuse_a_u64_large_colpair_epilogue();
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      "transient_native_b_resident_rns_a_output");
  const std::string key = bounded_native_a_reuse_b_autotune_key(args, result, kernel, epilogue, bound);
  set_backend_text(result.backend_info.autotune_key, sizeof(result.backend_info.autotune_key), key.c_str());
}

void apply_bounded_uniform_small_i8_ab_transient_backend_metadata(
    const Args& args,
    BenchmarkResult& result,
    uint64_t bound) {
  if (!bounded_uniform_small_i8_ab_transient_path(args, result)) {
    return;
  }
  result.backend_info.performance_validated = 0;
  const char* kernel = bounded_residue_channel_fusion_requested(args)
      ? bounded_residue_channel_fusion_kernel()
      : bounded_uniform_small_i8_ab_transient_kernel();
  const char* epilogue = bounded_residue_channel_fusion_requested(args)
      ? bounded_residue_channel_fusion_epilogue()
      : bounded_uniform_small_i8_ab_transient_epilogue();
  set_backend_text(result.backend_info.selected_kernel, sizeof(result.backend_info.selected_kernel), kernel);
  set_backend_text(result.backend_info.epilogue_mode, sizeof(result.backend_info.epilogue_mode), epilogue);
  set_backend_text(
      result.backend_info.workspace_mode,
      sizeof(result.backend_info.workspace_mode),
      bounded_residue_channel_fusion_requested(args)
          ? "width3_residue_fusion_transient_i8_inputs"
          : "transient_i8_a_transient_i8_b_rns_output");
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
  const bool gemv_n1 = args.n == 1 && args.k >= 4096;
  const char* kernel = signed_semantics
      ? (gemv_n1 ? "hip_vector_alu_i64_gemv_n1_exact_192b_v1" : "hip_vector_alu_i64_exact_192b_v1")
      : (gemv_n1 ? "hip_vector_alu_u64_gemv_n1_exact_192b_v1" : "hip_vector_alu_u64_exact_192b_v1");
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

const char* wrap64_direct_hip_gemm_event_label(const Args& args) {
  return rns8::detail::wrap64_hip_gemm_event_label_for_shape(args.m, args.n, args.k);
}

const char* native_to_rns_bridge_event_label(const Args& args) {
  return args.semantics == BenchSemantics::BoundedI64 ? "native_i64_to_rns_kernel" : "native_u64_to_rns_kernel";
}

bool native_to_rns_bridge_path(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend) {
  return native_to_rns_bridge_requested(args) && bounded_benchmark_semantics(args.semantics) &&
         selected_backend == RNS8_BACKEND_HIP_DIRECT && result.backend_info_available &&
         result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT;
}

const char* vector_to_rns_chain_kernel_label(const Args& args) {
  return args.semantics == BenchSemantics::BoundedI64 ? "vector_alu_i64_kernel" : "vector_alu_u64_kernel";
}

bool vector_to_rns_chain_path(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend) {
  return vector_to_rns_chain_requested(args) && bounded_benchmark_semantics(args.semantics) &&
         selected_backend == RNS8_BACKEND_HIP_DIRECT && result.backend_info_available &&
         result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT;
}

bool direct_hip_bounded_oneshot_resident_fallback_path(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend) {
  const uint32_t requested_prefix = benchmark_prefix(args);
  const uint32_t selected_prefix = selected_execution_prefix(args, result);
  return args.oneshot && bounded_benchmark_semantics(args.semantics) &&
         selected_backend == RNS8_BACKEND_HIP_DIRECT && requested_prefix > 0 && selected_prefix > 0 &&
         selected_prefix < requested_prefix;
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
    if (direct_hip_bounded_oneshot_resident_fallback_path(args, result, selected_backend)) {
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
  if (vector_to_rns_chain_path(args, result, selected_backend)) {
    return {
        "vector_alu_pack_a_h2d",
        "vector_alu_pack_b_h2d",
        "pack_h2d",
        "pack_kernel",
        "pack",
        "vector_alu_status_memset",
        vector_to_rns_chain_kernel_label(args),
        "vector_alu_status_d2h",
        native_to_rns_bridge_event_label(args),
        "rns_gemm_kernel_group",
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
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
        wrap64_direct_hip_gemm_event_label(args),
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
    if (residue_current_output_mode(args)) {
      return phases;
    }
    phases.push_back("crt_export_status_memset");
    phases.push_back("crt_export_kernel");
    phases.push_back("crt_export_status_d2h");
    phases.push_back("crt_export_d2h");
    phases.push_back("crt_export");
    return phases;
  }
  const char* gemm_phase = use_prepacked_b_cache ? "rns_gemm_prepacked_b_kernel_group" : "rns_gemm_kernel_group";
  if (bounded_native_a_reuse_b_path(args, result)) {
    gemm_phase = bounded_native_a_reuse_b_event_label(args);
  }
  if (bounded_uniform_small_i8_ab_reuse_a_path(args, result)) {
    gemm_phase = bounded_uniform_small_i8_ab_reuse_a_event_label();
  }
  if (bounded_native_b_reuse_a_u64_large_colpair_path(args, result)) {
    gemm_phase = bounded_native_b_reuse_a_u64_large_colpair_event_label();
  }
  std::vector<std::string> phases;
  if (bounded_uniform_small_i8_ab_transient_path(args, result)) {
    gemm_phase = bounded_uniform_small_i8_ab_transient_event_label();
    phases = {
        bounded_uniform_small_i8_ab_transient_a_h2d_label(),
        bounded_uniform_small_i8_ab_transient_b_h2d_label(),
        "pack",
        gemm_phase};
  } else {
    phases = {"pack_h2d", "pack_kernel", "pack"};
    if (native_to_rns_bridge_path(args, result, selected_backend)) {
      phases.push_back(native_to_rns_bridge_event_label(args));
    }
    phases.push_back(gemm_phase);
  }
  if (selected_backend == RNS8_BACKEND_HIP_DIRECT && result.zero_output_tile_count != 0) {
    phases.push_back("direct_hip_zero_output_tile_memset");
  }
  append_accelerator_deep_event_phases(phases, args, result, selected_backend, use_prepacked_b_cache);
  phases.push_back("rns_gemm");
  if (residue_current_output_mode(args)) {
    return phases;
  }
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

template <typename T>
uint64_t checksum_matrix(const std::vector<T>& values, int64_t rows, int64_t cols, int64_t ld, const char* label) {
  uint64_t checksum = 1469598103934665603ull;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      mix_checksum(checksum, static_cast<uint64_t>(values[row_major_index(row, col, ld, label)]));
    }
  }
  return checksum;
}

uint64_t checksum_limb_matrix(
    const std::vector<uint64_t>& values,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t limb_count,
    const char* label) {
  uint64_t checksum = 1469598103934665603ull;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      const std::size_t cell = row_major_index(row, col, ld, label);
      const std::size_t base = cell * static_cast<std::size_t>(limb_count);
      for (uint32_t limb = 0; limb < limb_count; ++limb) {
        mix_checksum(checksum, values[base + static_cast<std::size_t>(limb)]);
      }
    }
  }
  return checksum;
}

template <typename T>
void copy_compact_to_output(
    const std::vector<T>& compact,
    std::vector<T>& output,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    const char* label) {
  if (ld == cols) {
    std::copy(compact.begin(), compact.end(), output.begin());
    return;
  }
  for (int64_t row = 0; row < rows; ++row) {
    const std::size_t source = row_major_index(row, 0, cols, label);
    const std::size_t destination = row_major_index(row, 0, ld, label);
    std::copy(compact.begin() + static_cast<std::ptrdiff_t>(source),
              compact.begin() + static_cast<std::ptrdiff_t>(source + static_cast<std::size_t>(cols)),
              output.begin() + static_cast<std::ptrdiff_t>(destination));
  }
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
  if (hip_graph_replay_requested(args)) {
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

void collect_gemm_gpu_events(
    const BenchmarkResult& result,
    GpuEventSamples& events,
    bool use_prepacked_b_cache,
    bool direct_hip_zero_fill) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = use_prepacked_b_cache ? "rns_gemm_prepacked_b_kernel_group" : "rns_gemm_kernel_group";
  const bool all_tiles_zero =
      result.zero_output_tile_count != 0 && result.zero_output_tile_count == result.schedule_info.tile_count;
  const double kernel_group =
      all_tiles_zero ? optional_event_label(samples, label) : sum_event_label(events, samples, "rns_gemm", label);
  const double zero_fill =
      direct_hip_zero_fill ? sum_event_label(events, samples, "rns_gemm", "direct_hip_zero_output_tile_memset") : 0.0;
  if (events.complete) {
    push_gpu_event_value(events, label, kernel_group);
    if (direct_hip_zero_fill) {
      push_gpu_event_value(events, "direct_hip_zero_output_tile_memset", zero_fill);
    }
    push_gpu_event_value(events, "rns_gemm", kernel_group + zero_fill);
  }
}

void collect_native_to_rns_bridge_gemm_gpu_events(
    const Args& args,
    GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* conversion_label = native_to_rns_bridge_event_label(args);
  const double conversion = sum_event_label(events, samples, "rns_gemm", conversion_label);
  const double kernel_group = sum_event_label(events, samples, "rns_gemm", "rns_gemm_kernel_group");
  if (events.complete) {
    push_gpu_event_value(events, conversion_label, conversion);
    push_gpu_event_value(events, "rns_gemm_kernel_group", kernel_group);
    push_gpu_event_value(events, "rns_gemm", conversion + kernel_group);
  }
}

void collect_vector_to_rns_chain_pack_gpu_events(
    const Args& args,
    GpuEventSamples& events,
    bool consumer_b_reused) {
  (void)args;
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const auto h2d_samples = event_label_values(samples, "residue_h2d_sync");
  std::size_t h2d_index = 0;
  const auto vector_operand_h2d = [&](const char* label) -> double {
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
  const double vector_a_h2d = vector_operand_h2d("vector_alu_pack_a_h2d");
  const double vector_b_h2d = vector_operand_h2d("vector_alu_pack_b_h2d");
  const double direct_h2d = consumer_b_reused ? 0.0 : sum_event_label(events, samples, "pack", "pack_h2d");
  const double direct_pack_kernel =
      consumer_b_reused ? 0.0 : sum_event_label(events, samples, "pack", "pack_kernel");
  if (events.complete) {
    push_gpu_event_value(events, "vector_alu_pack_a_h2d", vector_a_h2d);
    push_gpu_event_value(events, "vector_alu_pack_b_h2d", vector_b_h2d);
    push_gpu_event_value(events, "pack_h2d", direct_h2d);
    push_gpu_event_value(events, "pack_kernel", direct_pack_kernel);
    push_gpu_event_value(events, "pack", vector_a_h2d + vector_b_h2d + direct_h2d + direct_pack_kernel);
  }
}

void collect_vector_to_rns_chain_gemm_gpu_events(const Args& args, GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* vector_kernel_label = vector_to_rns_chain_kernel_label(args);
  const char* conversion_label = native_to_rns_bridge_event_label(args);
  const double status_memset = optional_event_label(samples, "vector_alu_status_memset");
  const double vector_kernel = sum_event_label(events, samples, "rns_gemm", vector_kernel_label);
  const double status_d2h = optional_event_label(samples, "vector_alu_status_d2h");
  const double conversion = sum_event_label(events, samples, "rns_gemm", conversion_label);
  const double direct_kernel_group = sum_event_label(events, samples, "rns_gemm", "rns_gemm_kernel_group");
  if (events.complete) {
    push_gpu_event_value(events, "vector_alu_status_memset", status_memset);
    push_gpu_event_value(events, vector_kernel_label, vector_kernel);
    push_gpu_event_value(events, "vector_alu_status_d2h", status_d2h);
    push_gpu_event_value(events, conversion_label, conversion);
    push_gpu_event_value(events, "rns_gemm_kernel_group", direct_kernel_group);
    push_gpu_event_value(
        events,
        "rns_gemm",
        status_memset + vector_kernel + status_d2h + conversion + direct_kernel_group);
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

void collect_bounded_uniform_small_i8_ab_reuse_a_pack_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double h2d = sum_event_label(events, samples, "pack", bounded_uniform_small_i8_ab_reuse_a_pack_h2d_label());
  if (events.complete) {
    push_gpu_event_value(events, "pack_h2d", h2d);
    push_gpu_event_value(events, "pack_kernel", 0.0);
    push_gpu_event_value(events, "pack", h2d);
  }
}

void collect_bounded_uniform_small_i8_ab_reuse_a_gemm_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = bounded_uniform_small_i8_ab_reuse_a_event_label();
  const double kernel = sum_event_label(events, samples, "rns_gemm", label);
  if (events.complete) {
    push_gpu_event_value(events, label, kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_bounded_uniform_small_i8_ab_transient_pack_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const double a_h2d =
      sum_event_label(events, samples, "pack", bounded_uniform_small_i8_ab_transient_a_h2d_label());
  const double b_h2d =
      sum_event_label(events, samples, "pack", bounded_uniform_small_i8_ab_transient_b_h2d_label());
  if (events.complete) {
    push_gpu_event_value(events, bounded_uniform_small_i8_ab_transient_a_h2d_label(), a_h2d);
    push_gpu_event_value(events, bounded_uniform_small_i8_ab_transient_b_h2d_label(), b_h2d);
    push_gpu_event_value(events, "pack", a_h2d + b_h2d);
  }
}

void collect_bounded_uniform_small_i8_ab_transient_gemm_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = bounded_uniform_small_i8_ab_transient_event_label();
  const double kernel = sum_event_label(events, samples, "rns_gemm", label);
  if (events.complete) {
    push_gpu_event_value(events, label, kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_bounded_native_b_reuse_a_pack_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  double h2d = 0.0;
  if (!sum_event_label_if_present(samples, bounded_native_b_reuse_a_u64_large_colpair_pack_h2d_label(), h2d) &&
      !sum_event_label_if_present(samples, "residue_h2d_sync", h2d)) {
    add_unavailable_reason(
        events,
        std::string("pack missing backend HIP event label ") +
            bounded_native_b_reuse_a_u64_large_colpair_pack_h2d_label() + " or residue_h2d_sync");
  }
  if (events.complete) {
    push_gpu_event_value(events, "pack_h2d", h2d);
    push_gpu_event_value(events, "pack_kernel", 0.0);
    push_gpu_event_value(events, "pack", h2d);
  }
}

void collect_bounded_native_b_reuse_a_gemm_gpu_events(GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const char* label = bounded_native_b_reuse_a_u64_large_colpair_event_label();
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
  } else if (native_to_rns_bridge_path(args, result, selected_backend)) {
    collect_native_to_rns_bridge_gemm_gpu_events(args, events);
  } else {
    collect_gemm_gpu_events(
        result,
        events,
        use_prepacked_b_cache,
        selected_backend == RNS8_BACKEND_HIP_DIRECT && result.zero_output_tile_count != 0);
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
  const char* label =
      args.wrap64_rocwmma_candidate ? kWrap64RocwmmaCandidateEventLabel : wrap64_direct_hip_gemm_event_label(args);
  const double kernel = sum_event_label(events, samples, "rns_gemm", label);
  if (events.complete) {
    push_gpu_event_value(events, label, kernel);
    push_gpu_event_value(events, "rns_gemm", kernel);
  }
}

void collect_export_gpu_events(
    const Args& args,
    rns8_backend_kind selected_backend,
    const BenchmarkResult& result,
    GpuEventSamples& events) {
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
  const bool all_zero_direct_hip_scheduled_export =
      selected_backend == RNS8_BACKEND_HIP_DIRECT && result.zero_output_tile_count != 0 &&
      result.schedule_info_available && result.zero_output_tile_count == result.schedule_info.tile_count;
  const double status_memset = optional_event_label(samples, "crt_export_status_memset");
  const double kernel = sum_event_label(events, samples, "crt_export", "crt_export_kernel");
  const double status_d2h = all_zero_direct_hip_scheduled_export
                                ? optional_event_label(samples, "crt_export_status_d2h")
                                : sum_event_label(events, samples, "crt_export", "crt_export_status_d2h");
  const double d2h = sum_event_label(events, samples, "crt_export", "crt_export_d2h");
  if (events.complete) {
    push_gpu_event_value(events, "crt_export_status_memset", status_memset);
    push_gpu_event_value(events, "crt_export_kernel", kernel);
    push_gpu_event_value(events, "crt_export_status_d2h", status_d2h);
    push_gpu_event_value(events, "crt_export_d2h", d2h);
    push_gpu_event_value(events, "crt_export", status_memset + kernel + status_d2h + d2h);
  }
}

void collect_bounded_oneshot_gpu_events(
    const Args& args,
    const BenchmarkResult& result,
    rns8_backend_kind selected_backend,
    GpuEventSamples& events) {
  const auto samples = rns8::detail::hip_direct_timing_snapshot();
  const bool resident_fallback = direct_hip_bounded_oneshot_resident_fallback_path(args, result, selected_backend);
  const double pack_h2d =
      resident_fallback ? sum_event_label(events, samples, "oneshot", "pack_h2d")
                        : sum_event_label(events, samples, "oneshot", "residue_h2d_sync");
  const double pack_kernel = resident_fallback ? sum_event_label(events, samples, "oneshot", "pack_kernel") : 0.0;
  const double gemm = sum_event_label(events, samples, "oneshot", "rns_gemm_kernel_group");
  const double status_memset = optional_event_label(samples, "crt_export_status_memset");
  const double export_kernel = sum_event_label(events, samples, "oneshot", "crt_export_kernel");
  const double status_d2h = sum_event_label(events, samples, "oneshot", "crt_export_status_d2h");
  const double output_d2h = sum_event_label(events, samples, "oneshot", "crt_export_d2h");
  const double export_total = status_memset + export_kernel + status_d2h + output_d2h;
  if (events.complete) {
    if (resident_fallback) {
      push_gpu_event_value(events, "pack_h2d", pack_h2d);
      push_gpu_event_value(events, "pack_kernel", pack_kernel);
      push_gpu_event_value(events, "pack", pack_h2d + pack_kernel);
    } else {
      push_gpu_event_value(events, "oneshot_native_input_h2d", pack_h2d);
    }
    push_gpu_event_value(events, "rns_gemm_kernel_group", gemm);
    push_gpu_event_value(events, "rns_gemm", gemm);
    push_gpu_event_value(events, "crt_export_status_memset", status_memset);
    push_gpu_event_value(events, "crt_export_kernel", export_kernel);
    push_gpu_event_value(events, "crt_export_status_d2h", status_d2h);
    push_gpu_event_value(events, "crt_export_d2h", output_d2h);
    push_gpu_event_value(events, "crt_export", export_total);
    push_gpu_event_value(events, "oneshot_api_gpu", pack_h2d + pack_kernel + gemm + export_total);
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

void record_allocation_after_warmups(BenchmarkResult& result) {
  result.allocation_after_warmups = rns8::detail::hip_direct_allocation_counters_snapshot();
  result.allocation_after_warmups_available = true;
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

struct HostApiBatchTask {
  rns8_matrix* a = nullptr;
  rns8_matrix* b = nullptr;
  rns8_matrix* c = nullptr;
  rns8_workspace* workspace = nullptr;
};

void destroy_host_api_batch_tasks(std::vector<HostApiBatchTask>& tasks) {
  for (auto it = tasks.rbegin(); it != tasks.rend(); ++it) {
    if (it->c) {
      rns8_destroy_matrix(it->c);
      it->c = nullptr;
    }
    if (it->b) {
      rns8_destroy_matrix(it->b);
      it->b = nullptr;
    }
    if (it->a) {
      rns8_destroy_matrix(it->a);
      it->a = nullptr;
    }
    if (it->workspace) {
      rns8_destroy_workspace(it->workspace);
      it->workspace = nullptr;
    }
  }
}

std::vector<HostApiBatchTask> create_host_api_batch_tasks(
    rns8_context* ctx,
    rns8_plan* plan,
    const rns8_matrix_desc& a_desc,
    const rns8_matrix_desc& b_desc,
    const rns8_matrix_desc& c_desc,
    uint32_t batch_size) {
  std::vector<HostApiBatchTask> tasks(batch_size);
  for (uint32_t task_index = 0; task_index < batch_size; ++task_index) {
    rns8_status status = rns8_create_workspace(ctx, plan, &tasks[task_index].workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace(host API batch)", status);
    status = rns8_create_matrix(ctx, &a_desc, &tasks[task_index].a);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(host API batch A)", status);
    status = rns8_create_matrix(ctx, &b_desc, &tasks[task_index].b);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(host API batch B)", status);
    status = rns8_create_matrix(ctx, &c_desc, &tasks[task_index].c);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(host API batch C)", status);
  }
  return tasks;
}

uint64_t host_api_batch_source_version(uint64_t repeat_source_version, uint32_t task_index) {
  return repeat_source_version * UINT64_C(104729) + static_cast<uint64_t>(task_index) + UINT64_C(1);
}

template <typename PackTask, typename GemmTask, typename ExportTask>
void run_host_api_batch_iteration(
    const Args& args,
    BenchmarkResult& result,
    rns8_backend_kind selected_backend,
    std::vector<HostApiBatchTask>& tasks,
    uint64_t repeat_source_version,
    TimingSamples* samples,
    const PackTask& pack_task,
    const GemmTask& gemm_task,
    const ExportTask& export_task) {
  const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
  const uint32_t task_count = measured_task_count(args);
  const auto repeat_start = std::chrono::steady_clock::now();

  const auto pack_start = repeat_start;
  begin_gpu_event_phase(collect_gpu_events);
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    pack_task(tasks[task_index], task_index, host_api_batch_source_version(repeat_source_version, task_index));
  }
  if (collect_gpu_events) {
    collect_pack_gpu_events(args, selected_backend, result.gpu_events);
  }
  end_gpu_event_phase(collect_gpu_events);
  const auto pack_end = std::chrono::steady_clock::now();

  const auto gemm_start = pack_end;
  begin_gpu_event_phase(collect_gpu_events);
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    gemm_task(tasks[task_index], task_index);
  }
  if (collect_gpu_events) {
    collect_rns_gemm_gpu_events(args, selected_backend, result, result.gpu_events);
  }
  end_gpu_event_phase(collect_gpu_events);
  const auto gemm_end = std::chrono::steady_clock::now();

  const auto export_start = gemm_end;
  begin_gpu_event_phase(collect_gpu_events);
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    export_task(tasks[task_index], task_index);
  }
  if (collect_gpu_events) {
    collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
  }
  end_gpu_event_phase(collect_gpu_events);
  const auto export_end = std::chrono::steady_clock::now();

  if (samples) {
    samples->pack_us.push_back(elapsed_us(pack_start, pack_end));
    samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
    samples->export_us.push_back(elapsed_us(export_start, export_end));
    samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
  }
}

std::size_t checked_task_slab_limb_elements(
    std::size_t task_count,
    std::size_t compact_limb_elements,
    const char* label) {
  if (task_count != 0 && compact_limb_elements > std::numeric_limits<std::size_t>::max() / task_count) {
    usage_error(std::string("limb output size overflows size_t for ") + label);
  }
  return task_count * compact_limb_elements;
}

int64_t checked_int64_extent(std::size_t value, const char* label) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    usage_error(std::string("extent overflows int64_t for ") + label);
  }
  return static_cast<int64_t>(value);
}

void scatter_compact_limb_output(
    const uint64_t* compact,
    std::vector<uint64_t>& dst,
    const Args& args,
    const char* label) {
  const int64_t ldc = output_logical_ld(args);
  const std::size_t compact_row_elements =
      checked_limb_elements(1, args.n, args.exact_wide_limb_count, label);
  const std::size_t dst_row_elements =
      checked_limb_elements(1, ldc, args.exact_wide_limb_count, label);
  if (ldc == args.n) {
    const std::size_t total = checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, label);
    std::copy(compact, compact + total, dst.begin());
    return;
  }
  for (int64_t row = 0; row < args.m; ++row) {
    const std::size_t src_offset = static_cast<std::size_t>(row) * compact_row_elements;
    const std::size_t dst_offset = static_cast<std::size_t>(row) * dst_row_elements;
    std::copy(compact + src_offset, compact + src_offset + compact_row_elements, dst.begin() + dst_offset);
  }
}

void export_exact_wide_grouped_dispatch_slab(
    const Args& args,
    const std::vector<rns8_matrix*>& c_matrices,
    DeviceBuffer& device_slab,
    const DeviceBuffer& device_residue_ptrs,
    std::vector<uint64_t>& host_slab,
    bool signed_output) {
  const uint32_t task_count = measured_task_count(args);
  const std::size_t compact_limb_elements =
      checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, "grouped export compact task");
  const std::size_t total_limb_elements =
      checked_task_slab_limb_elements(task_count, compact_limb_elements, "grouped export slab");
  if (!device_slab.ptr || !device_residue_ptrs.ptr || c_matrices.size() != task_count ||
      host_slab.size() != total_limb_elements) {
    usage_error("grouped exact-wide export slab is not initialized for the current task count");
  }

  const rns8_status export_status =
      signed_output
          ? rns8::detail::hip_direct_export_exact_wide_signed_grouped_matrix_limbs_to_device(
                c_matrices.data(),
                task_count,
                device_residue_ptrs.ptr,
                device_slab.ptr,
                args.m,
                args.n,
                args.exact_wide_limb_count)
          : rns8::detail::hip_direct_export_exact_wide_unsigned_grouped_matrix_limbs_to_device(
                c_matrices.data(),
                task_count,
                device_residue_ptrs.ptr,
                device_slab.ptr,
                args.m,
                args.n,
                args.exact_wide_limb_count);
  if (export_status != RNS8_SUCCESS) {
    fail_status(
        signed_output ? "hip_direct_export_exact_wide_signed_grouped_matrix_limbs_to_device"
                      : "hip_direct_export_exact_wide_unsigned_grouped_matrix_limbs_to_device",
        export_status);
  }

  const rns8_status copy_status = rns8::detail::hip_direct_copy_compact_matrix_device_to_host(
      device_slab.device_id,
      "exact_wide_export_d2h",
      host_slab.data(),
      checked_int64_extent(total_limb_elements, "grouped exact-wide export slab D2H"),
      device_slab.ptr,
      1,
      checked_int64_extent(total_limb_elements, "grouped exact-wide export slab D2H"),
      sizeof(uint64_t),
      false);
  if (copy_status != RNS8_SUCCESS) {
    fail_status("hip_direct_copy_compact_matrix_device_to_host(grouped exact-wide export slab)", copy_status);
  }
}

void scatter_grouped_limb_slab_to_outputs(
    const Args& args,
    const std::vector<uint64_t>& host_slab,
    std::vector<std::vector<uint64_t>>& batch_c) {
  const uint32_t task_count = measured_task_count(args);
  const std::size_t compact_limb_elements =
      checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, "grouped export compact task");
  const std::size_t total_limb_elements =
      checked_task_slab_limb_elements(task_count, compact_limb_elements, "grouped export slab");
  if (host_slab.size() != total_limb_elements || batch_c.size() != task_count) {
    usage_error("grouped exact-wide host slab is not initialized for checksum materialization");
  }
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    const uint64_t* task_src = host_slab.data() + static_cast<std::size_t>(task_index) * compact_limb_elements;
    scatter_compact_limb_output(task_src, batch_c[task_index], args, "grouped exact-wide host output");
  }
}

uint64_t combine_host_api_batch_checksums(const std::vector<uint64_t>& task_checksums) {
  uint64_t checksum = 1469598103934665603ull;
  mix_checksum(checksum, static_cast<uint64_t>(task_checksums.size()));
  for (std::size_t index = 0; index < task_checksums.size(); ++index) {
    mix_checksum(checksum, static_cast<uint64_t>(index));
    mix_checksum(checksum, task_checksums[index]);
  }
  return checksum;
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
  auto desc = gemm_desc(args, bound, &result.tile_bounds, &result.zero_a_rows, &result.zero_b_cols);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(vector-alu schedule)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
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
  const int64_t ldc = output_logical_ld(args);
  std::vector<int64_t> C(output_elements(args, "C"), INT64_C(0x5a5a5a5a5a5a5a5a));
  std::vector<int64_t> compact_C(ldc == args.n ? 0 : checked_elements(args.m, args.n, "compact C"));
  fill_bounded_i64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  if (args.bound_mode == BoundMode::PerTile) {
    record_timed_tile_bounds(result, [&]() { return compute_i64_tile_bounds(args, A, B); });
  }
  capture_vector_alu_schedule(ctx, args, bound, result);

  const std::size_t a_bytes = checked_bytes(A.size(), sizeof(int64_t), "A");
  const std::size_t b_bytes = checked_bytes(B.size(), sizeof(int64_t), "B");
  const std::size_t c_bytes = checked_bytes(checked_elements(args.m, args.n, "device C"), sizeof(int64_t), "C");
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
      void* destination = ldc == args.n ? static_cast<void*>(C.data()) : static_cast<void*>(compact_C.data());
      return rns8::detail::hip_direct_copy_device_to_host(args.device_id, destination, d_c.ptr, c_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_device_to_host(vector C)", status);
    if (ldc != args.n) {
      copy_compact_to_output(compact_C, C, args.m, args.n, ldc, "vector C");
    }
    if (collect_gpu_events) {
      collect_export_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result, result.gpu_events);
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
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");
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
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(output_elements(args, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
  std::vector<uint64_t> compact_C(ldc == args.n ? 0 : checked_elements(args.m, args.n, "compact C"));
  fill_bounded_u64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  if (args.bound_mode == BoundMode::PerTile) {
    record_timed_tile_bounds(result, [&]() { return compute_u64_tile_bounds(args, A, B); });
  }
  capture_vector_alu_schedule(ctx, args, bound, result);

  const std::size_t a_bytes = checked_bytes(A.size(), sizeof(uint64_t), "A");
  const std::size_t b_bytes = checked_bytes(B.size(), sizeof(uint64_t), "B");
  const std::size_t c_bytes = checked_bytes(checked_elements(args.m, args.n, "device C"), sizeof(uint64_t), "C");
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
      void* destination = ldc == args.n ? static_cast<void*>(C.data()) : static_cast<void*>(compact_C.data());
      return rns8::detail::hip_direct_copy_device_to_host(args.device_id, destination, d_c.ptr, c_bytes);
    });
    if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_device_to_host(vector C)", status);
    if (ldc != args.n) {
      copy_compact_to_output(compact_C, C, args.m, args.n, ldc, "vector C");
    }
    if (collect_gpu_events) {
      collect_export_gpu_events(args, RNS8_BACKEND_HIP_VECTOR_ALU_INT64, result, result.gpu_events);
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
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");
  return result;
#endif
}

template <typename T>
void fill_vector_to_rns_chain_inputs(
    const Args& args,
    std::vector<T>& A,
    std::vector<T>& B,
    std::vector<T>& D,
    std::mt19937_64& rng) {
  if constexpr (std::is_same_v<T, int64_t>) {
    fill_bounded_i64_inputs(args, A, B, rng);
    std::uniform_int_distribution<int64_t> dist(-16, 16);
    for (auto& value : D) {
      value = dist(rng);
    }
  } else {
    fill_bounded_u64_inputs(args, A, B, rng);
    std::uniform_int_distribution<uint64_t> dist(0, 16);
    for (auto& value : D) {
      value = dist(rng);
    }
  }
}

template <typename T>
rns8_status pack_vector_to_rns_chain_operand(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const T* data,
    int64_t ld,
    uint64_t source_version) {
  if constexpr (std::is_same_v<T, int64_t>) {
    return rns8_pack_i64(ctx, matrix, data, ld, source_version);
  } else {
    return rns8_pack_u64(ctx, matrix, data, ld, source_version);
  }
}

template <typename T>
rns8_status export_vector_to_rns_chain_output(
    rns8_context* ctx,
    rns8_plan* plan,
    rns8_matrix* matrix,
    T* data,
    int64_t ld) {
  if constexpr (std::is_same_v<T, int64_t>) {
    return rns8_export_i64(ctx, plan, matrix, data, ld);
  } else {
    return rns8_export_u64(ctx, plan, matrix, data, ld);
  }
}

template <typename T>
BenchmarkResult run_vector_to_rns_chain(
    rns8_context* vector_ctx,
    rns8_context* direct_ctx,
    const Args& args,
    uint64_t final_bound,
    const char* label) {
#if !RNS8_CONFIGURED_HIP_ENABLED
  (void)vector_ctx;
  (void)direct_ctx;
  (void)args;
  (void)final_bound;
  (void)label;
  usage_error("--vector-to-rns-chain requires a HIP-enabled benchmark build");
#else
  std::mt19937_64 rng(args.seed);
  std::vector<T> A(checked_elements(args.m, args.k, "vector chain A"));
  std::vector<T> B(checked_elements(args.k, args.n, "vector chain B"));
  std::vector<T> D(checked_elements(args.n, args.n, "vector chain direct B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<T> C(output_elements(args, "vector chain C"), static_cast<T>(0x5a));
  fill_vector_to_rns_chain_inputs(args, A, B, D, rng);

  Args vector_args = args;
  vector_args.backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
  vector_args.vector_to_rns_chain = false;
  Args direct_args = args;
  direct_args.backend = RNS8_BACKEND_HIP_DIRECT;
  direct_args.k = args.n;
  direct_args.vector_to_rns_chain = false;

  const uint64_t producer_bound = vector_to_rns_chain_producer_bound(args);
  auto vector_desc = gemm_desc(vector_args, producer_bound);
  auto direct_desc = gemm_desc(direct_args, final_bound);

  BenchmarkResult result{};
  result.static_bound = final_bound;
  result.effective_bound = final_bound;
  result.effective_bound_available = true;

  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* vector_plan = nullptr;
  rns8_status status = rns8_create_plan(vector_ctx, &vector_desc, &vector_plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(vector-to-RNS producer)", status);
  rns8_plan* direct_plan = nullptr;
  status = rns8_create_plan(direct_ctx, &direct_desc, &direct_plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(vector-to-RNS consumer)", status);
  capture_schedule_info(direct_plan, result);
  capture_backend_info(direct_plan, result);
  enforce_per_tile_capture_contract(direct_args, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  rns8_workspace* vector_workspace = nullptr;
  status = rns8_create_workspace(vector_ctx, vector_plan, &vector_workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace(vector-to-RNS producer)", status);
  rns8_workspace* direct_workspace = nullptr;
  status = rns8_create_workspace(direct_ctx, direct_plan, &direct_workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace(vector-to-RNS consumer)", status);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  rns8_matrix* vector_a = nullptr;
  rns8_matrix* vector_b = nullptr;
  rns8_matrix* vector_c = nullptr;
  rns8_matrix* direct_a = nullptr;
  rns8_matrix* direct_b = nullptr;
  rns8_matrix* direct_c = nullptr;
  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t direct_matrix_prefix = selected_execution_prefix(args, result);
  auto vector_a_desc = matrix_desc(args.m, args.k, vector_args);
  auto vector_b_desc = matrix_desc(args.k, args.n, vector_args);
  auto vector_c_desc = matrix_desc(args.m, args.n, vector_args);
  auto direct_a_desc = matrix_desc(args.m, args.n, direct_args, direct_matrix_prefix);
  auto direct_b_desc = matrix_desc(args.n, args.n, direct_args, direct_matrix_prefix);
  auto direct_c_desc = matrix_desc(args.m, args.n, direct_args, direct_matrix_prefix);
  status = rns8_create_matrix(vector_ctx, &vector_a_desc, &vector_a);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(vector chain producer A)", status);
  status = rns8_create_matrix(vector_ctx, &vector_b_desc, &vector_b);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(vector chain producer B)", status);
  status = rns8_create_matrix(vector_ctx, &vector_c_desc, &vector_c);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(vector chain producer C)", status);
  status = rns8_create_matrix(direct_ctx, &direct_a_desc, &direct_a);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(vector chain direct A)", status);
  status = rns8_create_matrix(direct_ctx, &direct_b_desc, &direct_b);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(vector chain direct B)", status);
  status = rns8_create_matrix(direct_ctx, &direct_c_desc, &direct_c);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(vector chain direct C)", status);
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  if (args.reuse_packed_b) {
    const auto prepack_start = std::chrono::steady_clock::now();
    result.prepack_reuse_strategy = PrepackReuseStrategy::PersistentMatrixResidency;
    status = pack_vector_to_rns_chain_operand(direct_ctx, direct_b, D.data(), args.n, 1);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack(vector chain reused direct B)", status);
    const auto prepack_end = std::chrono::steady_clock::now();
    result.prepack_setup_us = elapsed_us(prepack_start, prepack_end);
    result.prepack_setup_available = true;
  }

  const auto run_iteration = [&](uint64_t source_version, TimingSamples* samples) {
    const bool collect_gpu_events = samples != nullptr && result.gpu_events.requested;
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto pack_start = repeat_start;
    begin_gpu_event_phase(collect_gpu_events);
    status = run_timed_status_operation("vector_alu_pack_a_h2d", [&]() {
      return pack_vector_to_rns_chain_operand(vector_ctx, vector_a, A.data(), args.k, source_version);
    });
    if (status != RNS8_SUCCESS) fail_status("rns8_pack(vector chain producer A)", status);
    status = run_timed_status_operation("vector_alu_pack_b_h2d", [&]() {
      return pack_vector_to_rns_chain_operand(vector_ctx, vector_b, B.data(), args.n, source_version);
    });
    if (status != RNS8_SUCCESS) fail_status("rns8_pack(vector chain producer B)", status);
    if (!args.reuse_packed_b) {
      status = pack_vector_to_rns_chain_operand(direct_ctx, direct_b, D.data(), args.n, source_version);
      if (status != RNS8_SUCCESS) fail_status("rns8_pack(vector chain direct B)", status);
    }
    if (collect_gpu_events) {
      collect_vector_to_rns_chain_pack_gpu_events(args, result.gpu_events, args.reuse_packed_b);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto pack_end = std::chrono::steady_clock::now();

    const auto gemm_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = rns8_gemm_rns(vector_ctx, vector_plan, vector_a, vector_b, vector_c, vector_workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(vector chain producer)", status);
    status = rns8::detail::materialize_native_matrix_as_direct_rns(direct_ctx, direct_plan, vector_c, direct_a);
    if (status != RNS8_SUCCESS) fail_status("materialize_native_matrix_as_direct_rns(vector chain)", status);
    status = rns8_gemm_rns(direct_ctx, direct_plan, direct_a, direct_b, direct_c, direct_workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(vector chain consumer)", status);
    if (collect_gpu_events) {
      collect_vector_to_rns_chain_gemm_gpu_events(args, result.gpu_events);
    }
    end_gpu_event_phase(collect_gpu_events);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    begin_gpu_event_phase(collect_gpu_events);
    status = export_vector_to_rns_chain_output(direct_ctx, direct_plan, direct_c, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export(vector chain direct output)", status);
    if (collect_gpu_events) {
      collect_export_gpu_events(args, RNS8_BACKEND_HIP_DIRECT, result, result.gpu_events);
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
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, label);

  rns8_destroy_matrix(direct_c);
  rns8_destroy_matrix(direct_b);
  rns8_destroy_matrix(direct_a);
  rns8_destroy_matrix(vector_c);
  rns8_destroy_matrix(vector_b);
  rns8_destroy_matrix(vector_a);
  rns8_destroy_workspace(direct_workspace);
  rns8_destroy_workspace(vector_workspace);
  rns8_destroy_plan(direct_plan);
  rns8_destroy_plan(vector_plan);
  return result;
#endif
}

BenchmarkResult run_vector_to_rns_chain_i64(
    rns8_context* vector_ctx,
    rns8_context* direct_ctx,
    const Args& args,
    uint64_t bound) {
  return run_vector_to_rns_chain<int64_t>(vector_ctx, direct_ctx, args, bound, "vector chain C");
}

BenchmarkResult run_vector_to_rns_chain_u64(
    rns8_context* vector_ctx,
    rns8_context* direct_ctx,
    const Args& args,
    uint64_t bound) {
  return run_vector_to_rns_chain<uint64_t>(vector_ctx, direct_ctx, args, bound, "vector chain C");
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
  capture_packing_and_lowering_info(plan, result);
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
  const int64_t ldc = output_logical_ld(args);
  std::vector<int64_t> C(output_elements(args, "C"), INT64_C(0x5a5a5a5a5a5a5a5a));
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
        rns8_gemm_i64_oneshot(ctx, &desc, A.data(), args.k, B.data(), args.n, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_i64_oneshot", status);
    if (collect_gpu_events && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      collect_bounded_oneshot_gpu_events(args, result, selected_backend, result.gpu_events);
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
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");
  return result;
}

BenchmarkResult run_bounded_u64_oneshot(rns8_context* ctx, const Args& args, uint64_t bound) {
  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(output_elements(args, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
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
        rns8_gemm_u64_oneshot(ctx, &desc, A.data(), args.k, B.data(), args.n, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_u64_oneshot", status);
    if (collect_gpu_events && selected_backend == RNS8_BACKEND_HIP_DIRECT) {
      collect_bounded_oneshot_gpu_events(args, result, selected_backend, result.gpu_events);
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
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");
  return result;
}

BenchmarkResult run_bounded_i64_host_api_batch(rns8_context* ctx, const Args& args, uint64_t bound) {
  std::mt19937_64 rng(args.seed);
  const uint32_t task_count = measured_task_count(args);
  const std::size_t a_elements = checked_elements(args.m, args.k, "host batch A");
  const std::size_t b_elements = checked_elements(args.k, args.n, "host batch B");
  const int64_t ldc = output_logical_ld(args);
  const std::size_t c_elements = output_elements(args, "host batch C");
  std::vector<std::vector<int64_t>> batch_a(task_count, std::vector<int64_t>(a_elements));
  std::vector<std::vector<int64_t>> batch_b(task_count, std::vector<int64_t>(b_elements));
  std::vector<std::vector<int64_t>> batch_c(
      task_count, std::vector<int64_t>(c_elements, INT64_C(0x5a5a5a5a5a5a5a5a)));
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    fill_bounded_i64_inputs(args, batch_a[task_index], batch_b[task_index], rng);
  }

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, batch_a.front(), batch_b.front(), bound, result);
  auto desc = gemm_desc(args, bound);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(host API batch)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  std::vector<HostApiBatchTask> tasks =
      create_host_api_batch_tasks(ctx, plan, a_desc, b_desc, c_desc, task_count);
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    status = rns8_pack_i64(ctx, task.a, batch_a[task_index].data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(host API batch A)", status);
    status = rns8_pack_i64(ctx, task.b, batch_b[task_index].data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(host API batch B)", status);
  };
  const auto gemm_task = [&](HostApiBatchTask& task, uint32_t) {
    status = rns8_gemm_rns(ctx, plan, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(host API batch i64)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    status = rns8_export_i64(ctx, plan, task.c, batch_c[task_index].data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_i64(host API batch)", status);
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_host_api_batch_iteration(
        args, result, selected_backend, tasks, static_cast<uint64_t>(r) + 1, nullptr, pack_task, gemm_task, export_task);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        tasks,
        static_cast<uint64_t>(args.warmups) + r + 1,
        &result.samples,
        pack_task,
        gemm_task,
        export_task);
  }
  std::vector<uint64_t> task_checksums(task_count);
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    task_checksums[task_index] = checksum_matrix(batch_c[task_index], args.m, args.n, ldc, "host batch C");
  }
  result.checksum = combine_host_api_batch_checksums(task_checksums);

  destroy_host_api_batch_tasks(tasks);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_bounded_u64_host_api_batch(rns8_context* ctx, const Args& args, uint64_t bound) {
  std::mt19937_64 rng(args.seed);
  const uint32_t task_count = measured_task_count(args);
  const std::size_t a_elements = checked_elements(args.m, args.k, "host batch A");
  const std::size_t b_elements = checked_elements(args.k, args.n, "host batch B");
  const int64_t ldc = output_logical_ld(args);
  const std::size_t c_elements = output_elements(args, "host batch C");
  std::vector<std::vector<uint64_t>> batch_a(task_count, std::vector<uint64_t>(a_elements));
  std::vector<std::vector<uint64_t>> batch_b(task_count, std::vector<uint64_t>(b_elements));
  std::vector<std::vector<uint64_t>> batch_c(
      task_count, std::vector<uint64_t>(c_elements, UINT64_C(0x5a5a5a5a5a5a5a5a)));
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    fill_bounded_u64_inputs(args, batch_a[task_index], batch_b[task_index], rng);
  }

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, batch_a.front(), batch_b.front(), bound, result);
  auto desc = gemm_desc(args, bound);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(host API batch)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  std::vector<HostApiBatchTask> tasks =
      create_host_api_batch_tasks(ctx, plan, a_desc, b_desc, c_desc, task_count);
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    status = rns8_pack_u64(ctx, task.a, batch_a[task_index].data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(host API batch A)", status);
    status = rns8_pack_u64(ctx, task.b, batch_b[task_index].data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(host API batch B)", status);
  };
  const auto gemm_task = [&](HostApiBatchTask& task, uint32_t) {
    status = rns8_gemm_rns(ctx, plan, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(host API batch u64)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    status = rns8_export_u64(ctx, plan, task.c, batch_c[task_index].data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_u64(host API batch)", status);
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_host_api_batch_iteration(
        args, result, selected_backend, tasks, static_cast<uint64_t>(r) + 1, nullptr, pack_task, gemm_task, export_task);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        tasks,
        static_cast<uint64_t>(args.warmups) + r + 1,
        &result.samples,
        pack_task,
        gemm_task,
        export_task);
  }
  std::vector<uint64_t> task_checksums(task_count);
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    task_checksums[task_index] = checksum_matrix(batch_c[task_index], args.m, args.n, ldc, "host batch C");
  }
  result.checksum = combine_host_api_batch_checksums(task_checksums);

  destroy_host_api_batch_tasks(tasks);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_bounded_i64(rns8_context* ctx, const Args& args, uint64_t bound) {
  if (args.oneshot) {
    return run_bounded_i64_oneshot(ctx, args, bound);
  }
  if (grouped_task_executor_requested(args)) {
    return run_bounded_i64_host_api_batch(ctx, args, bound);
  }
  if (args.vector_alu_baseline) {
    return run_vector_alu_i64(ctx, args, bound);
  }
  std::mt19937_64 rng(args.seed);
  std::vector<int64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<int64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<int64_t> C(output_elements(args, "C"), INT64_C(0x5a5a5a5a5a5a5a5a));
  fill_bounded_i64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  if (args.bound_mode == BoundMode::PerTile) {
    record_timed_tile_bounds(result, [&]() { return compute_i64_tile_bounds(args, A, B); });
  }
  auto desc = gemm_desc(args, bound, &result.tile_bounds, &result.zero_a_rows, &result.zero_b_cols);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  enforce_per_tile_capture_contract(args, result);
  apply_bounded_native_a_reuse_b_backend_metadata(args, result, bound);
  apply_bounded_uniform_small_i8_ab_reuse_a_backend_metadata(args, result, bound);
  apply_bounded_native_b_reuse_a_backend_metadata(args, result, bound);
  apply_bounded_uniform_small_i8_ab_transient_backend_metadata(args, result, bound);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const bool use_native_a_reuse_b = bounded_native_a_reuse_b_path(args, result);
  const bool use_uniform_small_i8_ab_reuse_a = bounded_uniform_small_i8_ab_reuse_a_path(args, result);
  const bool use_uniform_small_i8_ab_reuse_b =
      use_native_a_reuse_b && bounded_native_a_reuse_b_uniform_small_a(args);
  const bool use_uniform_small_i8_ab_transient = bounded_uniform_small_i8_ab_transient_path(args, result);
  const bool use_uniform_small_i8_ab_reuse = use_uniform_small_i8_ab_reuse_a || use_uniform_small_i8_ab_reuse_b;
  const bool use_uniform_small_i8_ab_device_inputs =
      use_uniform_small_i8_ab_reuse || use_uniform_small_i8_ab_transient;
  std::vector<int8_t> uniform_small_a;
  std::vector<int8_t> uniform_small_b;
  if (use_uniform_small_i8_ab_device_inputs) {
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
      use_uniform_small_i8_ab_device_inputs
          ? checked_bytes(uniform_small_a.size(), sizeof(int8_t), "uniform-small A")
          : 0;
  const std::size_t uniform_small_b_bytes =
      use_uniform_small_i8_ab_device_inputs
          ? checked_bytes(uniform_small_b.size(), sizeof(int8_t), "uniform-small B")
          : 0;
  if (use_uniform_small_i8_ab_device_inputs) {
    uniform_small_a_device.allocate(args.device_id, uniform_small_a_bytes, "hip_direct_allocate(bounded i64 i8 A)");
    uniform_small_b_device.allocate(args.device_id, uniform_small_b_bytes, "hip_direct_allocate(bounded i64 i8 B)");
  } else if (use_native_a_reuse_b) {
    native_a.allocate(args.device_id, native_a_bytes, "hip_direct_allocate(bounded i64 native A)");
  } else {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  }
  if (!use_uniform_small_i8_ab_device_inputs) {
    status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  }
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  if (rns_residue_chain_requested(args)) {
    status = rns8_create_matrix(ctx, &c_desc, &scratch_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(chain scratch)", status);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  const bool skip_all_zero_direct_hip_input_pack =
      all_zero_direct_hip_input_pack_elided(args, result, selected_backend);

  const auto pack_a_input = [&](uint64_t source_version) {
    if (use_uniform_small_i8_ab_reuse_a || use_uniform_small_i8_ab_transient) {
      (void)source_version;
      status = run_timed_status_operation("bounded_uniform_small_i8_a_h2d", [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(
            args.device_id, uniform_small_a_device.ptr, uniform_small_a.data(), uniform_small_a_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded i64 uniform-small A)", status);
      return;
    }
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
    if (use_uniform_small_i8_ab_device_inputs) {
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
    if (!use_native_a_reuse_b && b_matrix && should_probe_reusable_b_prepack_cache(args) &&
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
    if (reuses_all_packed_inputs(args) || skip_all_zero_direct_hip_input_pack) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else if (use_uniform_small_i8_ab_transient) {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_bounded_uniform_small_i8_ab_transient_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else if (use_uniform_small_i8_ab_reuse_a) {
      begin_gpu_event_phase(collect_gpu_events);
      pack_b_input(source_version);
      if (collect_gpu_events) {
        collect_bounded_uniform_small_i8_ab_reuse_a_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
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
      force_native_to_rns_bridge_after_pack(args, a_matrix, b_matrix);
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
    if (use_uniform_small_i8_ab_transient || use_uniform_small_i8_ab_reuse_a || use_native_a_reuse_b) {
      if (use_uniform_small_i8_ab_transient) {
        status = rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix(
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
          fail_status("hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix", status);
        }
      } else if (use_uniform_small_i8_ab_reuse_a) {
        status = rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix(
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
          fail_status("hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix", status);
        }
      } else if (use_uniform_small_i8_ab_reuse_b) {
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
        if (use_uniform_small_i8_ab_transient) {
          collect_bounded_uniform_small_i8_ab_transient_gemm_gpu_events(result.gpu_events);
        } else if (use_uniform_small_i8_ab_reuse_a) {
          collect_bounded_uniform_small_i8_ab_reuse_a_gemm_gpu_events(result.gpu_events);
        } else {
          collect_bounded_native_a_gemm_gpu_events(args, result.gpu_events);
        }
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
      status = rns8_export_i64(ctx, plan, final_output_matrix, C.data(), ldc);
      if (status != RNS8_SUCCESS) fail_status("rns8_export_i64", status);
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      export_end = std::chrono::steady_clock::now();
    }

    if (samples) {
      samples->pack_us.push_back(
          (reuses_all_packed_inputs(args) || skip_all_zero_direct_hip_input_pack) ? 0
                                                                                  : elapsed_us(pack_start, pack_end));
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  if (hip_graph_replay_requested(args)) {
    run_hip_graph_replay_resident_chain(
        args, result, plan, a_matrix, b_matrix, c_matrix, scratch_matrix, latest_output_matrix);
  } else {
    for (uint32_t r = 0; r < args.warmups; ++r) {
      run_iteration(static_cast<uint64_t>(r) + 1, nullptr);
    }
    record_allocation_after_warmups(result);
    for (uint32_t r = 0; r < args.repeats; ++r) {
      run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
    }
  }
  if (chain_residue_output) {
    status = rns8_export_i64(ctx, plan, latest_output_matrix, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_i64(final chain checksum)", status);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");

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
  if (grouped_task_executor_requested(args)) {
    return run_bounded_u64_host_api_batch(ctx, args, bound);
  }
  if (args.vector_alu_baseline) {
    return run_vector_alu_u64(ctx, args, bound);
  }
  std::mt19937_64 rng(args.seed);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(output_elements(args, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
  fill_bounded_u64_inputs(args, A, B, rng);

  BenchmarkResult result{};
  bound = resolve_bounded_global_bound(args, A, B, bound, result);
  if (args.bound_mode == BoundMode::PerTile) {
    record_timed_tile_bounds(result, [&]() { return compute_u64_tile_bounds(args, A, B); });
  }
  auto desc = gemm_desc(args, bound, &result.tile_bounds, &result.zero_a_rows, &result.zero_b_cols);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  enforce_per_tile_capture_contract(args, result);
  apply_bounded_native_a_reuse_b_backend_metadata(args, result, bound);
  apply_bounded_uniform_small_i8_ab_reuse_a_backend_metadata(args, result, bound);
  apply_bounded_native_b_reuse_a_backend_metadata(args, result, bound);
  apply_bounded_uniform_small_i8_ab_transient_backend_metadata(args, result, bound);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  const bool use_native_a_reuse_b = bounded_native_a_reuse_b_path(args, result);
  const bool use_native_b_reuse_a = bounded_native_b_reuse_a_u64_large_colpair_path(args, result);
  const bool use_uniform_small_i8_ab_reuse_a = bounded_uniform_small_i8_ab_reuse_a_path(args, result);
  const bool use_uniform_small_i8_ab_reuse_b =
      use_native_a_reuse_b && bounded_native_a_reuse_b_uniform_small_a(args);
  const bool use_uniform_small_i8_ab_transient = bounded_uniform_small_i8_ab_transient_path(args, result);
  const bool use_uniform_small_i8_ab_reuse = use_uniform_small_i8_ab_reuse_a || use_uniform_small_i8_ab_reuse_b;
  const bool use_uniform_small_i8_ab_device_inputs =
      use_uniform_small_i8_ab_reuse || use_uniform_small_i8_ab_transient;
  std::vector<int8_t> uniform_small_a;
  std::vector<int8_t> uniform_small_b;
  if (use_uniform_small_i8_ab_device_inputs) {
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
  DeviceBuffer native_b;
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
  const std::size_t native_b_bytes =
      use_native_b_reuse_a ? checked_bytes(B.size(), sizeof(uint64_t), "native B") : 0;
  const std::size_t uniform_small_a_bytes =
      use_uniform_small_i8_ab_device_inputs
          ? checked_bytes(uniform_small_a.size(), sizeof(int8_t), "uniform-small A")
          : 0;
  const std::size_t uniform_small_b_bytes =
      use_uniform_small_i8_ab_device_inputs
          ? checked_bytes(uniform_small_b.size(), sizeof(int8_t), "uniform-small B")
          : 0;
  if (use_uniform_small_i8_ab_device_inputs) {
    uniform_small_a_device.allocate(args.device_id, uniform_small_a_bytes, "hip_direct_allocate(bounded u64 i8 A)");
    uniform_small_b_device.allocate(args.device_id, uniform_small_b_bytes, "hip_direct_allocate(bounded u64 i8 B)");
  } else if (use_native_a_reuse_b) {
    native_a.allocate(args.device_id, native_a_bytes, "hip_direct_allocate(bounded u64 native A)");
  } else {
    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  }
  if (use_native_b_reuse_a) {
    native_b.allocate(args.device_id, native_b_bytes, "hip_direct_allocate(bounded u64 native B)");
  } else if (!use_uniform_small_i8_ab_device_inputs) {
    status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  }
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  if (rns_residue_chain_requested(args)) {
    status = rns8_create_matrix(ctx, &c_desc, &scratch_matrix);
    if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(chain scratch)", status);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);
  const bool skip_all_zero_direct_hip_input_pack =
      all_zero_direct_hip_input_pack_elided(args, result, selected_backend);

  const auto pack_a_input = [&](uint64_t source_version) {
    if (use_uniform_small_i8_ab_reuse_a || use_uniform_small_i8_ab_transient) {
      (void)source_version;
      status = run_timed_status_operation("bounded_uniform_small_i8_a_h2d", [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(
            args.device_id, uniform_small_a_device.ptr, uniform_small_a.data(), uniform_small_a_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded u64 uniform-small A)", status);
      return;
    }
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
    if (use_uniform_small_i8_ab_device_inputs) {
      (void)source_version;
      status = run_timed_status_operation(bounded_native_a_reuse_b_b_h2d_label(args), [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(
            args.device_id, uniform_small_b_device.ptr, uniform_small_b.data(), uniform_small_b_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded u64 uniform-small B)", status);
      return;
    }
    if (use_native_b_reuse_a) {
      fail_status("rns8_pack_u64(B native-B reuse-A path)", RNS8_INVALID_ARGUMENT);
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
    if (!use_native_a_reuse_b && b_matrix && should_probe_reusable_b_prepack_cache(args) &&
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
    if (reuses_all_packed_inputs(args) || skip_all_zero_direct_hip_input_pack) {
      if (collect_gpu_events) {
        record_reused_pack_gpu_events(args, selected_backend, result.gpu_events);
      }
    } else if (use_uniform_small_i8_ab_transient) {
      begin_gpu_event_phase(collect_gpu_events);
      pack_per_repeat_inputs(args, source_version, pack_a_input, pack_b_input);
      if (collect_gpu_events) {
        collect_bounded_uniform_small_i8_ab_transient_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else if (use_uniform_small_i8_ab_reuse_a) {
      begin_gpu_event_phase(collect_gpu_events);
      pack_b_input(source_version);
      if (collect_gpu_events) {
        collect_bounded_uniform_small_i8_ab_reuse_a_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
    } else if (use_native_b_reuse_a) {
      begin_gpu_event_phase(collect_gpu_events);
      status = run_timed_status_operation(bounded_native_b_reuse_a_u64_large_colpair_pack_h2d_label(), [&]() {
        return rns8::detail::hip_direct_copy_host_to_device(args.device_id, native_b.ptr, B.data(), native_b_bytes);
      });
      if (status != RNS8_SUCCESS) fail_status("hip_direct_copy_host_to_device(bounded u64 native B)", status);
      if (collect_gpu_events) {
        collect_bounded_native_b_reuse_a_pack_gpu_events(result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      pack_end = std::chrono::steady_clock::now();
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
      force_native_to_rns_bridge_after_pack(args, a_matrix, b_matrix);
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
    if (use_uniform_small_i8_ab_transient || use_uniform_small_i8_ab_reuse_a ||
        use_native_a_reuse_b || use_native_b_reuse_a) {
      if (use_uniform_small_i8_ab_transient) {
        status = rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix(
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
          fail_status("hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix", status);
        }
      } else if (use_uniform_small_i8_ab_reuse_a) {
        status = rns8::detail::hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix(
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
          fail_status("hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix", status);
        }
      } else if (use_uniform_small_i8_ab_reuse_b) {
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
      } else if (use_native_a_reuse_b && bounded_native_a_reuse_b_u64_large_colpair(args)) {
        status = rns8::detail::hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_matrix(
            args.device_id,
            native_a.ptr,
            b_matrix,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.k,
            source_version);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_matrix", status);
        }
      } else if (use_native_b_reuse_a) {
        status = rns8::detail::hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_matrix(
            args.device_id,
            a_matrix,
            native_b.ptr,
            c_matrix,
            args.m,
            args.n,
            args.k,
            args.n,
            source_version);
        if (status != RNS8_SUCCESS) {
          fail_status("hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_matrix", status);
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
        if (use_uniform_small_i8_ab_transient) {
          collect_bounded_uniform_small_i8_ab_transient_gemm_gpu_events(result.gpu_events);
        } else if (use_uniform_small_i8_ab_reuse_a) {
          collect_bounded_uniform_small_i8_ab_reuse_a_gemm_gpu_events(result.gpu_events);
        } else if (use_native_b_reuse_a) {
          collect_bounded_native_b_reuse_a_gemm_gpu_events(result.gpu_events);
        } else {
          collect_bounded_native_a_gemm_gpu_events(args, result.gpu_events);
        }
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
      status = rns8_export_u64(ctx, plan, final_output_matrix, C.data(), ldc);
      if (status != RNS8_SUCCESS) fail_status("rns8_export_u64", status);
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
      }
      end_gpu_event_phase(collect_gpu_events);
      export_end = std::chrono::steady_clock::now();
    }

    if (samples) {
      samples->pack_us.push_back(
          (reuses_all_packed_inputs(args) || skip_all_zero_direct_hip_input_pack) ? 0
                                                                                  : elapsed_us(pack_start, pack_end));
      samples->gemm_us.push_back(elapsed_us(gemm_start, gemm_end));
      samples->export_us.push_back(elapsed_us(export_start, export_end));
      samples->end_to_end_us.push_back(elapsed_us(repeat_start, export_end));
    }
  };

  if (hip_graph_replay_requested(args)) {
    run_hip_graph_replay_resident_chain(
        args, result, plan, a_matrix, b_matrix, c_matrix, scratch_matrix, latest_output_matrix);
  } else {
    for (uint32_t r = 0; r < args.warmups; ++r) {
      run_iteration(static_cast<uint64_t>(r) + 1, nullptr);
    }
    record_allocation_after_warmups(result);
    for (uint32_t r = 0; r < args.repeats; ++r) {
      run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
    }
  }
  if (chain_residue_output) {
    status = rns8_export_u64(ctx, plan, latest_output_matrix, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_u64(final chain checksum)", status);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");

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

BenchmarkResult run_exact_wide_signed_host_api_batch(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  const uint32_t task_count = measured_task_count(args);
  std::uniform_int_distribution<int64_t> dist(-16, 16);
  const std::size_t a_elements = checked_elements(args.m, args.k, "host batch A");
  const std::size_t b_elements = checked_elements(args.k, args.n, "host batch B");
  const int64_t ldc = output_logical_ld(args);
  const std::size_t c_elements = output_limb_elements(args, args.exact_wide_limb_count, "host batch C");
  std::vector<std::vector<int64_t>> batch_a(task_count, std::vector<int64_t>(a_elements));
  std::vector<std::vector<int64_t>> batch_b(task_count, std::vector<int64_t>(b_elements));
  std::vector<std::vector<uint64_t>> batch_c(
      task_count, std::vector<uint64_t>(c_elements, UINT64_C(0x5a5a5a5a5a5a5a5a)));
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    for (auto& value : batch_a[task_index]) value = dist(rng);
    for (auto& value : batch_b[task_index]) value = dist(rng);
  }

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(host API batch exact-wide signed)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  DeviceBuffer grouped_export_device_slab;
  DeviceBuffer grouped_export_residue_ptrs;
  std::vector<uint64_t> grouped_export_host_slab;
  std::vector<rns8_matrix*> grouped_export_c_matrices;
  const bool grouped_export_slab_enabled = grouped_dispatch_requested(args) &&
                                           selected_backend == RNS8_BACKEND_HIP_DIRECT &&
                                           !exact_wide_export_status_check_required(args) &&
                                           args.output_ld_padding == 0;
  if (grouped_dispatch_requested(args)) {
    result.grouped_dispatch_execution_strategy =
        grouped_export_slab_enabled ? "device_grouped_exact_wide_export_kernel_batched_d2h"
                                    : "host_phase_loop_per_task_export";
  }

  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  std::vector<HostApiBatchTask> tasks =
      create_host_api_batch_tasks(ctx, plan, a_desc, b_desc, c_desc, task_count);
  if (grouped_export_slab_enabled) {
    const std::size_t compact_limb_elements =
        checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, "grouped exact-wide signed compact C");
    const std::size_t total_limb_elements =
        checked_task_slab_limb_elements(task_count, compact_limb_elements, "grouped exact-wide signed slab C");
    const std::size_t slab_bytes =
        checked_bytes(total_limb_elements, sizeof(uint64_t), "grouped exact-wide signed slab C");
    const int device_id = args.device_id < 0 ? 0 : args.device_id;
    grouped_export_device_slab.allocate(device_id, slab_bytes, "hip_direct_allocate(grouped exact-wide signed slab C)");
    const std::size_t pointer_bytes =
        checked_bytes(task_count, sizeof(const void*), "grouped exact-wide signed residue pointer table");
    grouped_export_residue_ptrs.allocate(
        device_id, pointer_bytes, "hip_direct_allocate(grouped exact-wide signed residue pointer table)");
    grouped_export_c_matrices.reserve(task_count);
    for (const auto& task : tasks) {
      grouped_export_c_matrices.push_back(task.c);
    }
    int pointer_device_id = -1;
    uint32_t pointer_prefix = 0;
    status = rns8::detail::hip_direct_prepare_exact_wide_grouped_matrix_residue_pointers(
        grouped_export_c_matrices.data(),
        task_count,
        RNS8_EXACT_WIDE_SIGNED,
        grouped_export_residue_ptrs.ptr,
        grouped_export_residue_ptrs.bytes,
        &pointer_device_id,
        &pointer_prefix);
    if (status != RNS8_SUCCESS) {
      fail_status("hip_direct_prepare_exact_wide_grouped_matrix_residue_pointers(signed)", status);
    }
    if (pointer_device_id != grouped_export_residue_ptrs.device_id || pointer_prefix != matrix_prefix) {
      usage_error("grouped exact-wide signed residue pointer table metadata does not match benchmark storage");
    }
    grouped_export_host_slab.assign(total_limb_elements, UINT64_C(0x5a5a5a5a5a5a5a5a));
    result.grouped_dispatch_batched_export_enabled = true;
    result.grouped_dispatch_device_output_slab_bytes = static_cast<uint64_t>(slab_bytes);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    status = rns8_pack_i64(ctx, task.a, batch_a[task_index].data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(host API batch exact-wide A)", status);
    status = rns8_pack_i64(ctx, task.b, batch_b[task_index].data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(host API batch exact-wide B)", status);
  };
  const auto gemm_task = [&](HostApiBatchTask& task, uint32_t) {
    status = rns8_gemm_rns(ctx, plan, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(host API batch exact-wide signed)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    if (grouped_export_slab_enabled) {
      if (task_index == 0) {
        export_exact_wide_grouped_dispatch_slab(
            args,
            grouped_export_c_matrices,
            grouped_export_device_slab,
            grouped_export_residue_ptrs,
            grouped_export_host_slab,
            true);
      }
      return;
    }
    status = rns8_export_exact_wide_signed_limbs(
        ctx, plan, task.c, batch_c[task_index].data(), ldc, args.exact_wide_limb_count);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_signed_limbs(host API batch)", status);
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_host_api_batch_iteration(
        args, result, selected_backend, tasks, static_cast<uint64_t>(r) + 1, nullptr, pack_task, gemm_task, export_task);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        tasks,
        static_cast<uint64_t>(args.warmups) + r + 1,
        &result.samples,
        pack_task,
        gemm_task,
        export_task);
  }
  std::vector<uint64_t> task_checksums(task_count);
  if (grouped_export_slab_enabled) {
    scatter_grouped_limb_slab_to_outputs(args, grouped_export_host_slab, batch_c);
  }
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    task_checksums[task_index] = checksum_limb_matrix(
        batch_c[task_index], args.m, args.n, ldc, args.exact_wide_limb_count, "host batch C");
  }
  result.checksum = combine_host_api_batch_checksums(task_checksums);

  destroy_host_api_batch_tasks(tasks);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_exact_wide_unsigned_host_api_batch(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  const uint32_t task_count = measured_task_count(args);
  std::uniform_int_distribution<uint64_t> dist(0, 16);
  const std::size_t a_elements = checked_elements(args.m, args.k, "host batch A");
  const std::size_t b_elements = checked_elements(args.k, args.n, "host batch B");
  const int64_t ldc = output_logical_ld(args);
  const std::size_t c_elements = output_limb_elements(args, args.exact_wide_limb_count, "host batch C");
  std::vector<std::vector<uint64_t>> batch_a(task_count, std::vector<uint64_t>(a_elements));
  std::vector<std::vector<uint64_t>> batch_b(task_count, std::vector<uint64_t>(b_elements));
  std::vector<std::vector<uint64_t>> batch_c(
      task_count, std::vector<uint64_t>(c_elements, UINT64_C(0x5a5a5a5a5a5a5a5a)));
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    for (auto& value : batch_a[task_index]) value = dist(rng);
    for (auto& value : batch_b[task_index]) value = dist(rng);
  }

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(host API batch exact-wide unsigned)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  std::vector<HostApiBatchTask> tasks =
      create_host_api_batch_tasks(ctx, plan, a_desc, b_desc, c_desc, task_count);
  DeviceBuffer grouped_export_device_slab;
  DeviceBuffer grouped_export_residue_ptrs;
  std::vector<uint64_t> grouped_export_host_slab;
  std::vector<rns8_matrix*> grouped_export_c_matrices;
  const bool grouped_export_slab_enabled = grouped_dispatch_requested(args) &&
                                           selected_backend == RNS8_BACKEND_HIP_DIRECT &&
                                           !exact_wide_export_status_check_required(args) &&
                                           args.output_ld_padding == 0;
  if (grouped_dispatch_requested(args)) {
    result.grouped_dispatch_execution_strategy =
        grouped_export_slab_enabled ? "device_grouped_exact_wide_export_kernel_batched_d2h"
                                    : "host_phase_loop_per_task_export";
  }
  if (grouped_export_slab_enabled) {
    const std::size_t compact_limb_elements =
        checked_limb_elements(args.m, args.n, args.exact_wide_limb_count, "grouped exact-wide unsigned compact C");
    const std::size_t total_limb_elements =
        checked_task_slab_limb_elements(task_count, compact_limb_elements, "grouped exact-wide unsigned slab C");
    const std::size_t slab_bytes =
        checked_bytes(total_limb_elements, sizeof(uint64_t), "grouped exact-wide unsigned slab C");
    const int device_id = args.device_id < 0 ? 0 : args.device_id;
    grouped_export_device_slab.allocate(device_id, slab_bytes, "hip_direct_allocate(grouped exact-wide unsigned slab C)");
    const std::size_t pointer_bytes =
        checked_bytes(task_count, sizeof(const void*), "grouped exact-wide unsigned residue pointer table");
    grouped_export_residue_ptrs.allocate(
        device_id, pointer_bytes, "hip_direct_allocate(grouped exact-wide unsigned residue pointer table)");
    grouped_export_c_matrices.reserve(task_count);
    for (const auto& task : tasks) {
      grouped_export_c_matrices.push_back(task.c);
    }
    int pointer_device_id = -1;
    uint32_t pointer_prefix = 0;
    status = rns8::detail::hip_direct_prepare_exact_wide_grouped_matrix_residue_pointers(
        grouped_export_c_matrices.data(),
        task_count,
        RNS8_EXACT_WIDE_UNSIGNED,
        grouped_export_residue_ptrs.ptr,
        grouped_export_residue_ptrs.bytes,
        &pointer_device_id,
        &pointer_prefix);
    if (status != RNS8_SUCCESS) {
      fail_status("hip_direct_prepare_exact_wide_grouped_matrix_residue_pointers(unsigned)", status);
    }
    if (pointer_device_id != grouped_export_residue_ptrs.device_id || pointer_prefix != matrix_prefix) {
      usage_error("grouped exact-wide unsigned residue pointer table metadata does not match benchmark storage");
    }
    grouped_export_host_slab.assign(total_limb_elements, UINT64_C(0x5a5a5a5a5a5a5a5a));
    result.grouped_dispatch_batched_export_enabled = true;
    result.grouped_dispatch_device_output_slab_bytes = static_cast<uint64_t>(slab_bytes);
  }
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    status = rns8_pack_u64(ctx, task.a, batch_a[task_index].data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(host API batch exact-wide A)", status);
    status = rns8_pack_u64(ctx, task.b, batch_b[task_index].data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_u64(host API batch exact-wide B)", status);
  };
  const auto gemm_task = [&](HostApiBatchTask& task, uint32_t) {
    status = rns8_gemm_rns(ctx, plan, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns(host API batch exact-wide unsigned)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    if (grouped_export_slab_enabled) {
      if (task_index == 0) {
        export_exact_wide_grouped_dispatch_slab(
            args,
            grouped_export_c_matrices,
            grouped_export_device_slab,
            grouped_export_residue_ptrs,
            grouped_export_host_slab,
            false);
      }
      return;
    }
    status = rns8_export_exact_wide_unsigned_limbs(
        ctx, plan, task.c, batch_c[task_index].data(), ldc, args.exact_wide_limb_count);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_unsigned_limbs(host API batch)", status);
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_host_api_batch_iteration(
        args, result, selected_backend, tasks, static_cast<uint64_t>(r) + 1, nullptr, pack_task, gemm_task, export_task);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        tasks,
        static_cast<uint64_t>(args.warmups) + r + 1,
        &result.samples,
        pack_task,
        gemm_task,
        export_task);
  }
  std::vector<uint64_t> task_checksums(task_count);
  if (grouped_export_slab_enabled) {
    scatter_grouped_limb_slab_to_outputs(args, grouped_export_host_slab, batch_c);
  }
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    task_checksums[task_index] = checksum_limb_matrix(
        batch_c[task_index], args.m, args.n, ldc, args.exact_wide_limb_count, "host batch C");
  }
  result.checksum = combine_host_api_batch_checksums(task_checksums);

  destroy_host_api_batch_tasks(tasks);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_exact_wide_signed(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  if (grouped_task_executor_requested(args)) {
    return run_exact_wide_signed_host_api_batch(ctx, args, bound);
  }
  std::mt19937_64 rng(args.seed);
  std::uniform_int_distribution<int64_t> dist(-16, 16);
  std::vector<int64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<int64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(
      output_limb_elements(args, args.exact_wide_limb_count, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
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
  capture_packing_and_lowering_info(plan, result);
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
  if (rns_residue_chain_requested(args)) {
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
      status = rns8_export_exact_wide_signed_limbs(
          ctx, plan, final_output_matrix, C.data(), ldc, args.exact_wide_limb_count);
      if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_signed_limbs", status);
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
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

  if (hip_graph_replay_requested(args)) {
    run_hip_graph_replay_resident_chain(
        args, result, plan, a_matrix, b_matrix, c_matrix, scratch_matrix, latest_output_matrix);
  } else {
    for (uint32_t r = 0; r < args.warmups; ++r) {
      run_iteration(static_cast<uint64_t>(r) + 1, nullptr);
    }
    record_allocation_after_warmups(result);
    for (uint32_t r = 0; r < args.repeats; ++r) {
      run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
    }
  }
  if (chain_residue_output) {
    status = rns8_export_exact_wide_signed_limbs(
        ctx, plan, latest_output_matrix, C.data(), ldc, args.exact_wide_limb_count);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_signed_limbs(final chain checksum)", status);
  }
  result.checksum = checksum_limb_matrix(C, args.m, args.n, ldc, args.exact_wide_limb_count, "C");

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
  if (grouped_task_executor_requested(args)) {
    return run_exact_wide_unsigned_host_api_batch(ctx, args, bound);
  }
  std::mt19937_64 rng(args.seed);
  std::uniform_int_distribution<uint64_t> dist(0, 16);
  std::vector<uint64_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint64_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(
      output_limb_elements(args, args.exact_wide_limb_count, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
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
  capture_packing_and_lowering_info(plan, result);
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
  if (rns_residue_chain_requested(args)) {
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
      status = rns8_export_exact_wide_unsigned_limbs(
          ctx, plan, final_output_matrix, C.data(), ldc, args.exact_wide_limb_count);
      if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_unsigned_limbs", status);
      if (collect_gpu_events) {
        collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
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

  if (hip_graph_replay_requested(args)) {
    run_hip_graph_replay_resident_chain(
        args, result, plan, a_matrix, b_matrix, c_matrix, scratch_matrix, latest_output_matrix);
  } else {
    for (uint32_t r = 0; r < args.warmups; ++r) {
      run_iteration(static_cast<uint64_t>(r) + 1, nullptr);
    }
    record_allocation_after_warmups(result);
    for (uint32_t r = 0; r < args.repeats; ++r) {
      run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
    }
  }
  if (chain_residue_output) {
    status = rns8_export_exact_wide_unsigned_limbs(
        ctx, plan, latest_output_matrix, C.data(), ldc, args.exact_wide_limb_count);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_exact_wide_unsigned_limbs(final chain checksum)", status);
  }
  result.checksum = checksum_limb_matrix(C, args.m, args.n, ldc, args.exact_wide_limb_count, "C");

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
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint8_t> C(output_elements(args, "C"), 0x5au);
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
          ctx, &desc, args.finite_modulus, A.data(), args.k, B.data(), args.n, C.data(), ldc);
    } else {
      status = rns8_gemm_finite_ring_u8_oneshot(
          ctx, &desc, args.finite_modulus, A.data(), args.k, B.data(), args.n, C.data(), ldc);
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
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");
  return result;
}

BenchmarkResult run_finite_u8_host_api_batch(rns8_context* ctx, const Args& args, uint64_t bound) {
  (void)bound;
  std::mt19937_64 rng(args.seed);
  const uint32_t task_count = measured_task_count(args);
  const uint32_t high = args.finite_modulus == 256 ? 255u : static_cast<uint32_t>(args.finite_modulus - 1u);
  std::uniform_int_distribution<uint32_t> dist(0, high);
  const std::size_t a_elements = checked_elements(args.m, args.k, "host batch finite A");
  const std::size_t b_elements = checked_elements(args.k, args.n, "host batch finite B");
  const int64_t ldc = output_logical_ld(args);
  const std::size_t c_elements = output_elements(args, "host batch finite C");
  std::vector<std::vector<uint8_t>> batch_a(task_count, std::vector<uint8_t>(a_elements));
  std::vector<std::vector<uint8_t>> batch_b(task_count, std::vector<uint8_t>(b_elements));
  std::vector<std::vector<uint8_t>> batch_c(
      task_count, std::vector<uint8_t>(c_elements, 0x5au));
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    for (auto& value : batch_a[task_index]) value = static_cast<uint8_t>(dist(rng));
    for (auto& value : batch_b[task_index]) value = static_cast<uint8_t>(dist(rng));
  }

  BenchmarkResult result{};
  auto desc = gemm_desc(args, 0);
  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  rns8_status status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan(host API batch finite)", status);
  capture_schedule_info(plan, result);
  capture_backend_info(plan, result);
  capture_packing_and_lowering_info(plan, result);
  const rns8_backend_kind selected_backend = selected_backend_for_events(args, result);
  result.gpu_events.requested = gpu_event_capture_requested(args, selected_backend);
  const auto plan_end = std::chrono::steady_clock::now();
  result.plan_us = elapsed_us(plan_start, plan_end);

  const auto alloc_start = std::chrono::steady_clock::now();
  const uint32_t matrix_prefix = selected_execution_prefix(args, result);
  auto a_desc = matrix_desc(args.m, args.k, args, matrix_prefix);
  auto b_desc = matrix_desc(args.k, args.n, args, matrix_prefix);
  auto c_desc = matrix_desc(args.m, args.n, args, matrix_prefix);
  std::vector<HostApiBatchTask> tasks =
      create_host_api_batch_tasks(ctx, plan, a_desc, b_desc, c_desc, task_count);
  const auto alloc_end = std::chrono::steady_clock::now();
  result.matrix_alloc_us = elapsed_us(alloc_start, alloc_end);

  const auto pack_task = [&](HostApiBatchTask& task, uint32_t task_index, uint64_t source_version) {
    status = rns8_pack_finite_u8(
        ctx, task.a, args.finite_modulus, batch_a[task_index].data(), args.k, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_finite_u8(host API batch A)", status);
    status = rns8_pack_finite_u8(
        ctx, task.b, args.finite_modulus, batch_b[task_index].data(), args.n, source_version);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_finite_u8(host API batch B)", status);
  };
  const auto gemm_task = [&](HostApiBatchTask& task, uint32_t) {
    status = rns8_gemm_finite_u8(ctx, plan, args.finite_modulus, task.a, task.b, task.c, task.workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_finite_u8(host API batch)", status);
  };
  const auto export_task = [&](HostApiBatchTask& task, uint32_t task_index) {
    status = rns8_export_finite_u8(
        ctx, plan, args.finite_modulus, task.c, batch_c[task_index].data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_finite_u8(host API batch)", status);
  };

  for (uint32_t r = 0; r < args.warmups; ++r) {
    run_host_api_batch_iteration(
        args, result, selected_backend, tasks, static_cast<uint64_t>(r) + 1, nullptr, pack_task, gemm_task, export_task);
  }
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_host_api_batch_iteration(
        args,
        result,
        selected_backend,
        tasks,
        static_cast<uint64_t>(args.warmups) + r + 1,
        &result.samples,
        pack_task,
        gemm_task,
        export_task);
  }
  std::vector<uint64_t> task_checksums(task_count);
  for (uint32_t task_index = 0; task_index < task_count; ++task_index) {
    task_checksums[task_index] = checksum_matrix(batch_c[task_index], args.m, args.n, ldc, "host batch finite C");
  }
  result.checksum = combine_host_api_batch_checksums(task_checksums);

  destroy_host_api_batch_tasks(tasks);
  rns8_destroy_plan(plan);
  return result;
}

BenchmarkResult run_finite_u8(rns8_context* ctx, const Args& args, uint64_t bound) {
  if (args.oneshot) {
    return run_finite_u8_oneshot(ctx, args, bound);
  }
  if (grouped_task_executor_requested(args)) {
    return run_finite_u8_host_api_batch(ctx, args, bound);
  }
  (void)bound;
  std::mt19937_64 rng(args.seed);
  const uint32_t high = args.finite_modulus == 256 ? 255u : static_cast<uint32_t>(args.finite_modulus - 1u);
  std::uniform_int_distribution<uint32_t> dist(0, high);
  std::vector<uint8_t> A(checked_elements(args.m, args.k, "A"));
  std::vector<uint8_t> B(checked_elements(args.k, args.n, "B"));
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint8_t> C(output_elements(args, "C"), 0x5au);
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
  capture_packing_and_lowering_info(plan, result);
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
    status = rns8_export_finite_u8(ctx, plan, args.finite_modulus, c_matrix, C.data(), ldc);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_finite_u8", status);
    if (collect_gpu_events) {
      collect_export_gpu_events(args, selected_backend, result, result.gpu_events);
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
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");

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
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(output_elements(args, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
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
        args.device_id, c_limbs.ptr, &export_buffer.ptr, &export_buffer.bytes, args.m, args.n, C.data(), ldc);
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
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(&result.samples);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");
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
  const int64_t ldc = output_logical_ld(args);
  std::vector<uint64_t> C(output_elements(args, "C"), UINT64_C(0x5a5a5a5a5a5a5a5a));
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
  capture_packing_and_lowering_info(plan, result);
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
    status = rns8_export_wrap_u64(ctx, plan, c_matrix, C.data(), ldc);
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
  record_allocation_after_warmups(result);
  for (uint32_t r = 0; r < args.repeats; ++r) {
    run_iteration(static_cast<uint64_t>(args.warmups) + r + 1, &result.samples);
  }
  result.checksum = checksum_matrix(C, args.m, args.n, ldc, "C");

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
  if (native_to_rns_bridge_requested(args)) {
    return "rns8_bounded_gemm_native_to_rns_bridge";
  }
  if (vector_to_rns_chain_requested(args)) {
    return "rns8_bounded_gemm_vector_to_rns_chain";
  }
  if (grouped_dispatch_requested(args)) {
    return "rns8_grouped_dispatch_persistent_resident";
  }
  if (host_api_batch_requested(args)) {
    return "rns8_host_api_batch_persistent_resident";
  }
  if (hip_graph_replay_requested(args)) {
    return "rns8_hip_graph_replay_resident_rns_chain";
  }
  if (residue_chain_final_export_requested(args)) {
    return "rns8_residue_chain_final_host_export";
  }
  if (residue_current_output_mode(args)) {
    return "rns8_residue_current_rns_chain";
  }
  if (bounded_residue_channel_fusion_requested(args)) {
    return "rns8_bounded_gemm_residue_channel_fusion_experiment";
  }
  if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    return "rns8_bounded_gemm_transient_uniform_small_i8";
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
    if (result.zero_output_tile_count != 0) {
      return "direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3";
    }
    return "direct_hip_tiled_active_prefix_rns_gemm_v2";
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 && info.backend == RNS8_BACKEND_HIP_DIRECT) {
    return rns8::detail::wrap64_hip_selected_kernel_for_shape(args.m, args.n, args.k);
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

void print_nullable_std_string(const std::string& value) {
  if (value.empty()) {
    std::cout << "null";
  } else {
    std::cout << "\"" << json_escape(value) << "\"";
  }
}

std::string resolved_next_op_hint(const Args& args, const BenchmarkResult& result) {
  if (args.next_op_hint != NextOpHint::Auto) {
    return next_op_hint_name(args.next_op_hint);
  }
  if (residue_chain_final_export_requested(args)) {
    return "final-export";
  }
  if (residue_current_output_mode(args)) {
    return "rns-gemm";
  }
  if (args.reuse_packed_b && !args.reuse_packed_a) {
    return "reuse-b";
  }
  if (runtime_vector_alu_backend(args) || args.vector_alu_baseline) {
    return "native-gemm";
  }
  if (result.lowering_info_available && result.lowering_info.native_to_rns_available) {
    return "native-to-rns";
  }
  return "final-export";
}

const char* next_op_hint_source(const Args& args) {
  return args.next_op_hint == NextOpHint::Auto ? "benchmark_default" : "cli";
}

const char* output_status_handling(const Args& args) {
  if (residue_current_output_mode(args) || (exact_wide_benchmark_semantics(args.semantics) &&
                                            !exact_wide_export_status_check_required(args))) {
    return "structurally_elided";
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 || finite_benchmark_semantics(args.semantics)) {
    return "not_applicable";
  }
  return "required";
}

const char* output_status_event_policy(const Args& args) {
  const char* handling = output_status_handling(args);
  if (std::string(handling) == "required") {
    return "status_memset_and_status_d2h_labels_required_when_gpu_events_available";
  }
  if (std::string(handling) == "structurally_elided") {
    return "status_labels_zero_filled_or_absent_because_no_per_repeat_status_export_launches";
  }
  return "no_range_status_for_semantic";
}

std::string target_namespace_for_id(const std::string& target_id) {
  if (target_id == "cpu") {
    return "cpu";
  }
  if (target_id == "gfx1100") {
    return "gfx1100";
  }
  if (target_id.rfind("gfx11", 0) == 0) {
    return "gfx11xx";
  }
  if (target_id.rfind("gfx12", 0) == 0) {
    return "gfx12xx";
  }
  if (target_id.rfind("gfx94", 0) == 0 || target_id.rfind("gfx9", 0) == 0) {
    return "gfx9xx_gfx94x";
  }
  return "unknown";
}

std::string target_review_group_key(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result,
    const char* selected_backend) {
  const std::string target_id = benchmark_key_target_id(result);
  std::ostringstream out;
  out << target_namespace_for_id(target_id)
      << "/target=" << target_id
      << "/backend=" << selected_backend
      << "/semantics=" << semantics_name(args.semantics)
      << "/configured=" << RNS8_CONFIGURED_AMDGPU_TARGETS
      << "/runtime=" << info.hip_runtime_version;
  return out.str();
}

std::string benchmark_pack_layout(const Args& args, const BenchmarkResult& result) {
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return "wrap64_byte_limb_planes";
  }
  if (finite_benchmark_semantics(args.semantics)) {
    return "finite_u8_centered_residue";
  }
  if (runtime_vector_alu_backend(args) || args.vector_alu_baseline) {
    return args.semantics == BenchSemantics::BoundedI64 ? "native_i64_row_major" : "native_u64_row_major";
  }
  if (bounded_residue_channel_fusion_requested(args)) {
    return "native_i8_row_major_residue_channel_width3";
  }
  if (bounded_uniform_small_i8_ab_transient_requested(args) ||
      (bounded_native_a_reuse_b_requested(args) && bounded_native_a_reuse_b_uniform_small_a(args)) ||
      bounded_uniform_small_i8_ab_reuse_a_requested(args)) {
    return "native_i8_row_major_uniform_small";
  }
  if (result.packing_info_available && result.packing_info.uses_matrix_engine_pack_layout) {
    return "matrix_engine_transient_pack_layout";
  }
  if (result.packing_info_available && result.packing_info.uses_transient_pack_workspace) {
    return "transient_backend_pack_layout";
  }
  return "resident_rns_residue_planes";
}

const char* benchmark_fusion_mode(const Args& args) {
  return bounded_residue_channel_fusion_requested(args)
      ? "residue_channel_width3_experimental_benchmark_only"
      : "none";
}

uint32_t benchmark_residue_group_width(const Args& args) {
  return bounded_residue_channel_fusion_requested(args) ? 3u : 1u;
}

const char* benchmark_residue_group_layout(const Args& args) {
  return bounded_residue_channel_fusion_requested(args)
      ? "first_prefix9_moduli_contiguous_width3_groups"
      : "one_modulus_per_residue_plane";
}

std::string generated_reducer_identity(const Args& args, const BenchmarkResult& result) {
  const uint32_t selected_prefix = selected_execution_prefix(args, result);
  if (bounded_benchmark_semantics(args.semantics) &&
      selected_prefix > 0 && selected_prefix <= RNS8_DEFAULT_BOUNDED_PREFIX &&
      result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "direct_hip_fixed_prefix_" + std::to_string(selected_prefix) + "_generated_reducer_v1";
  }
  if (exact_wide_benchmark_semantics(args.semantics) &&
      selected_prefix == RNS8_MAX_SUPPORTED_PREFIX &&
      result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "direct_hip_fixed_prefix_20_generated_reducer_v1";
  }
  if (finite_benchmark_semantics(args.semantics) &&
      result.backend_info_available && result.backend_info.backend == RNS8_BACKEND_HIP_DIRECT) {
    return "direct_hip_finite_modulus_" + std::to_string(args.finite_modulus) + "_fixed_reducer_v1";
  }
  return "not_applicable";
}

void print_plan_packing_json(const BenchmarkResult& result) {
  if (!result.packing_info_available) {
    std::cout << "  \"plan_packing\": null,\n";
    return;
  }
  const auto& packing = result.packing_info;
  std::cout << "  \"plan_packing\": {\n";
  std::cout << "    \"source\": \"rns8_get_plan_packing_info\",\n";
  std::cout << "    \"backend\": \"" << backend_name(packing.backend) << "\",\n";
  std::cout << "    \"semantics\": \"" << c_semantics_name(packing.semantics) << "\",\n";
  std::cout << "    \"uses_resident_matrix_inputs\": "
            << (packing.uses_resident_matrix_inputs ? "true" : "false") << ",\n";
  std::cout << "    \"uses_transient_pack_workspace\": "
            << (packing.uses_transient_pack_workspace ? "true" : "false") << ",\n";
  std::cout << "    \"uses_matrix_engine_pack_layout\": "
            << (packing.uses_matrix_engine_pack_layout ? "true" : "false") << ",\n";
  std::cout << "    \"reusable_prepack_cache_available\": "
            << (packing.reusable_prepack_cache_available ? "true" : "false") << ",\n";
  std::cout << "    \"production_prepack_cache_available\": "
            << (packing.production_prepack_cache_available ? "true" : "false") << ",\n";
  std::cout << "    \"input_domain_name\": \"" << json_escape(packing.input_domain_name) << "\",\n";
  std::cout << "    \"output_domain_name\": \"" << json_escape(packing.output_domain_name) << "\",\n";
  std::cout << "    \"output_host_current\": " << (packing.output_host_current ? "true" : "false") << ",\n";
  std::cout << "    \"output_device_current\": " << (packing.output_device_current ? "true" : "false") << ",\n";
  std::cout << "    \"next_op_flags\": " << packing.next_op_flags << ",\n";
  std::cout << "    \"next_op_hint\": \"" << json_escape(packing.next_op_hint) << "\",\n";
  std::cout << "    \"a_pack_workspace_bytes\": " << packing.a_pack_workspace_bytes << ",\n";
  std::cout << "    \"b_pack_workspace_bytes\": " << packing.b_pack_workspace_bytes << ",\n";
  std::cout << "    \"accumulator_workspace_bytes\": " << packing.accumulator_workspace_bytes << ",\n";
  std::cout << "    \"library_workspace_bytes\": " << packing.library_workspace_bytes << ",\n";
  std::cout << "    \"total_transient_workspace_bytes\": " << packing.total_transient_workspace_bytes << ",\n";
  std::cout << "    \"a_layout_version\": \"" << json_escape(packing.a_layout_version) << "\",\n";
  std::cout << "    \"b_layout_version\": \"" << json_escape(packing.b_layout_version) << "\",\n";
  std::cout << "    \"output_layout_version\": \"" << json_escape(packing.output_layout_version) << "\",\n";
  std::cout << "    \"prepack_cache_scope\": \"" << json_escape(packing.prepack_cache_scope) << "\",\n";
  std::cout << "    \"detail\": \"" << json_escape(packing.detail) << "\"\n";
  std::cout << "  },\n";
}

void print_plan_lowering_json(const BenchmarkResult& result) {
  if (!result.lowering_info_available) {
    std::cout << "  \"plan_lowering\": null,\n";
    return;
  }
  const auto& lowering = result.lowering_info;
  std::cout << "  \"plan_lowering\": {\n";
  std::cout << "    \"source\": \"rns8_get_plan_backend_info+rns8_get_plan_packing_info+rns8_get_plan_schedule_info\",\n";
  std::cout << "    \"operation\": \"" << json_escape(lowering.operation) << "\",\n";
  std::cout << "    \"semantic_contract\": \"" << json_escape(lowering.semantic_contract) << "\",\n";
  std::cout << "    \"backend_family\": \"" << json_escape(lowering.backend_family) << "\",\n";
  std::cout << "    \"input_domain\": \"" << json_escape(lowering.input_domain) << "\",\n";
  std::cout << "    \"output_domain\": \"" << json_escape(lowering.output_domain) << "\",\n";
  std::cout << "    \"desired_output\": \"" << json_escape(lowering.desired_output) << "\",\n";
  std::cout << "    \"schedule_strategy\": \"" << json_escape(lowering.schedule_strategy) << "\",\n";
  std::cout << "    \"packing_strategy\": \"" << json_escape(lowering.packing_strategy) << "\",\n";
  std::cout << "    \"reuse_strategy\": \"" << json_escape(lowering.reuse_strategy) << "\",\n";
  std::cout << "    \"conversion_strategy\": \"" << json_escape(lowering.conversion_strategy) << "\",\n";
  std::cout << "    \"lowering_path\": \"" << json_escape(lowering.lowering_path) << "\",\n";
  std::cout << "    \"final_export_available\": " << (lowering.final_export_available ? "true" : "false") << ",\n";
  std::cout << "    \"rns_continuation_available\": "
            << (lowering.rns_continuation_available ? "true" : "false") << ",\n";
  std::cout << "    \"native_continuation_available\": "
            << (lowering.native_continuation_available ? "true" : "false") << ",\n";
  std::cout << "    \"native_to_rns_available\": "
            << (lowering.native_to_rns_available ? "true" : "false") << ",\n";
  std::cout << "    \"reusable_b_prepack_available\": "
            << (lowering.reusable_b_prepack_available ? "true" : "false") << "\n";
  std::cout << "  },\n";
}

void print_requested_next_op_json(const Args& args, const BenchmarkResult& result) {
  const std::string resolved = resolved_next_op_hint(args, result);
  std::cout << "  \"requested_next_op\": {\n";
  std::cout << "    \"requested\": \"" << next_op_hint_name(args.next_op_hint) << "\",\n";
  std::cout << "    \"resolved\": \"" << json_escape(resolved) << "\",\n";
  std::cout << "    \"source\": \"" << next_op_hint_source(args) << "\",\n";
  std::cout << "    \"final_export_available\": "
            << (result.lowering_info_available && result.lowering_info.final_export_available ? "true" : "false")
            << ",\n";
  std::cout << "    \"rns_continuation_available\": "
            << (result.lowering_info_available && result.lowering_info.rns_continuation_available ? "true" : "false")
            << ",\n";
  std::cout << "    \"native_continuation_available\": "
            << (result.lowering_info_available && result.lowering_info.native_continuation_available ? "true" : "false")
            << ",\n";
  std::cout << "    \"native_to_rns_available\": "
            << (result.lowering_info_available && result.lowering_info.native_to_rns_available ? "true" : "false")
            << ",\n";
  std::cout << "    \"reusable_b_prepack_available\": "
            << (result.lowering_info_available && result.lowering_info.reusable_b_prepack_available ? "true" : "false")
            << "\n";
  std::cout << "  },\n";
}

void print_output_policy_json(const Args& args, int64_t output_ld) {
  std::cout << "  \"output_policy\": {\n";
  std::cout << "    \"destination_layout\": \"" << output_destination_layout(args) << "\",\n";
  std::cout << "    \"logical_ld\": " << output_ld << ",\n";
  std::cout << "    \"ld_padding\": " << args.output_ld_padding << ",\n";
  std::cout << "    \"per_repeat_logical_export\": " << (residue_current_output_mode(args) ? "false" : "true") << ",\n";
  std::cout << "    \"final_checksum_export_after_repeats\": "
            << (residue_current_output_mode(args) ? "true" : "false") << ",\n";
  std::cout << "    \"status_handling\": \"" << output_status_handling(args) << "\",\n";
  std::cout << "    \"status_event_policy\": \"" << output_status_event_policy(args) << "\"\n";
  std::cout << "  },\n";
}

void print_target_variant_json(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result,
    const char* selected_backend) {
  const std::string target_id = benchmark_key_target_id(result);
  const std::string target_namespace = target_namespace_for_id(target_id);
  std::cout << "  \"target_variant\": {\n";
  std::cout << "    \"target_id\": \"" << json_escape(target_id) << "\",\n";
  std::cout << "    \"target_namespace\": \"" << json_escape(target_namespace) << "\",\n";
  std::cout << "    \"review_group_key\": \""
            << json_escape(target_review_group_key(args, info, result, selected_backend)) << "\",\n";
  std::cout << "    \"configured_amdgpu_targets\": \"" << json_escape(RNS8_CONFIGURED_AMDGPU_TARGETS) << "\",\n";
  std::cout << "    \"hip_enabled\": " << (RNS8_CONFIGURED_HIP_ENABLED ? "true" : "false") << ",\n";
  std::cout << "    \"hip_runtime_version\": " << info.hip_runtime_version << ",\n";
  std::cout << "    \"hip_driver_version\": " << info.hip_driver_version << "\n";
  std::cout << "  },\n";
}

void print_counter_object(const rns8::detail::hip_direct_allocation_counters& counters) {
  std::cout << "{"
            << "\"allocate_calls\": " << counters.allocate_calls << ", "
            << "\"free_calls\": " << counters.free_calls << ", "
            << "\"allocated_bytes\": " << counters.allocated_bytes
            << "}";
}

rns8::detail::hip_direct_allocation_counters allocation_delta(
    const rns8::detail::hip_direct_allocation_counters& before,
    const rns8::detail::hip_direct_allocation_counters& after) {
  rns8::detail::hip_direct_allocation_counters delta{};
  delta.allocate_calls = after.allocate_calls >= before.allocate_calls ? after.allocate_calls - before.allocate_calls : 0;
  delta.free_calls = after.free_calls >= before.free_calls ? after.free_calls - before.free_calls : 0;
  delta.allocated_bytes = after.allocated_bytes >= before.allocated_bytes ? after.allocated_bytes - before.allocated_bytes : 0;
  return delta;
}

const char* benchmark_setup_scope(const Args& args) {
  if (args.oneshot) {
    return "transient_api_setup_inside_measured_repeat";
  }
  if (bounded_residue_channel_fusion_requested(args)) {
    return "benchmark_experimental_residue_channel_fusion_persistent_output";
  }
  if (hip_graph_replay_requested(args)) {
    return "benchmark_hip_graph_replay_resident_rns_chain";
  }
  if (grouped_dispatch_requested(args)) {
    return "benchmark_grouped_dispatch_one_shared_plan_persistent_resident_tasks";
  }
  if (args.vector_alu_baseline || runtime_vector_alu_backend(args) || bounded_uniform_small_i8_ab_transient_requested(args)) {
    return "benchmark_owned_transient_native_input_buffers";
  }
  if (args.reuse_packed_inputs) {
    return "persistent_plan_workspace_prepacked_reuse";
  }
  return "persistent_plan_workspace_resident_matrices";
}

void print_device_allocation_json(const Args& args, const BenchmarkResult& result) {
  std::cout << "  \"device_allocation\": {\n";
  std::cout << "    \"tracking_available\": " << (result.allocation_tracking_available ? "true" : "false") << ",\n";
  std::cout << "    \"source\": \"hip_direct_allocation_counters_snapshot\",\n";
  std::cout << "    \"setup_scope\": \"" << benchmark_setup_scope(args) << "\",\n";
  std::cout << "    \"source_version_inputs\": \"monotonic_source_version_per_repeat_when_packing_runs\",\n";
  std::cout << "    \"before\": ";
  print_counter_object(result.allocation_before);
  std::cout << ",\n";
  std::cout << "    \"after_warmups\": ";
  if (result.allocation_after_warmups_available) {
    print_counter_object(result.allocation_after_warmups);
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"after_repeats\": ";
  print_counter_object(result.allocation_after_repeats);
  std::cout << ",\n";
  std::cout << "    \"setup_delta\": ";
  print_counter_object(allocation_delta(result.allocation_before, result.allocation_after_warmups));
  std::cout << ",\n";
  std::cout << "    \"measured_repeat_delta\": ";
  if (result.allocation_after_warmups_available) {
    print_counter_object(allocation_delta(result.allocation_after_warmups, result.allocation_after_repeats));
  } else {
    std::cout << "null";
  }
  std::cout << "\n";
  std::cout << "  },\n";
}

void print_auto_selector_json(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result,
    const char* selected_backend) {
  const auto snapshot = rns8::detail::read_autotune_cache();
  const std::string selected_key =
      result.backend_info.autotune_key[0] != '\0' ? std::string(result.backend_info.autotune_key) : std::string();
  rns8::detail::AutotuneRuntimeIdentity runtime{};
  runtime.target_id = benchmark_key_target_id(result);
  runtime.hip_sdk_or_library_version =
      result.backend_info.accelerator_version[0] != '\0'
          ? std::string(result.backend_info.accelerator_version)
          : std::string(RNS8_CONFIGURED_HIP_SDK_OR_ROCM_VERSION);
  const auto* exact_hit = selected_key.empty() ? nullptr : rns8::detail::find_exact_autotune_entry(snapshot, selected_key);
  const auto* validated_hit =
      selected_key.empty() ? nullptr : rns8::detail::find_validated_autotune_entry_for_runtime(snapshot, selected_key, runtime);
  std::cout << "  \"auto_selector\": {\n";
  std::cout << "    \"source\": \"private_benchmark_selector_report\",\n";
  std::cout << "    \"requested_backend\": \"" << requested_backend_name(args) << "\",\n";
  std::cout << "    \"selected_backend\": \"" << selected_backend << "\",\n";
  std::cout << "    \"selected_key\": ";
  print_nullable_std_string(selected_key);
  std::cout << ",\n";
  std::cout << "    \"cache_path\": \"" << json_escape(snapshot.path.string()) << "\",\n";
  std::cout << "    \"cache_exists\": " << (snapshot.exists ? "true" : "false") << ",\n";
  std::cout << "    \"cache_loaded\": " << (snapshot.loaded ? "true" : "false") << ",\n";
  std::cout << "    \"cache_entry_count\": " << snapshot.entries.size() << ",\n";
  std::cout << "    \"runtime_target_id\": \"" << json_escape(runtime.target_id) << "\",\n";
  std::cout << "    \"runtime_version\": \"" << json_escape(runtime.hip_sdk_or_library_version) << "\",\n";
  std::cout << "    \"exact_hit\": " << (exact_hit ? "true" : "false") << ",\n";
  std::cout << "    \"validated_hit\": " << (validated_hit ? "true" : "false") << ",\n";
  std::cout << "    \"fallback_reason\": \""
            << json_escape(
                   selected_key.empty()
                       ? "no exact entry"
                       : rns8::detail::autotune_selection_rationale(snapshot, selected_key, selected_backend, runtime))
            << "\",\n";
  std::cout << "    \"rejection_reason_vocabulary\": ["
            << "\"unsupported semantics\", "
            << "\"per-tile unsupported\", "
            << "\"backend not compiled\", "
            << "\"probe failed\", "
            << "\"no exact entry\", "
            << "\"unvalidated entry\", "
            << "\"identity/runtime mismatch\", "
            << "\"workspace mismatch\", "
            << "\"slower than selected\"],\n";
  std::cout << "    \"rejected_candidates\": [";
  if (args.backend == RNS8_BACKEND_AUTO) {
    const std::array<const char*, 6> candidates = {
        "cpu-reference", "hip-direct", "hipblaslt", "ck", "rocwmma", "hip-vector-alu-int64"};
    bool first = true;
    for (const char* candidate : candidates) {
      if (std::string(candidate) == selected_backend) {
        continue;
      }
      std::cout << (first ? "\n" : ",\n");
      std::cout << "      {\"backend\": \"" << candidate << "\", \"reason\": \"no exact entry\"}";
      first = false;
    }
    if (!first) {
      std::cout << "\n    ";
    }
  }
  std::cout << "]\n";
  std::cout << "  },\n";
  (void)info;
}

const char* reuse_operand_role(const Args& args) {
  if (!args.reuse_packed_inputs) {
    return "none";
  }
  if (args.reuse_packed_a && args.reuse_packed_b) {
    return "A+B";
  }
  if (args.reuse_packed_a) {
    return "A";
  }
  if (args.reuse_packed_b) {
    return "B";
  }
  return "none";
}

const char* shape_family_bucket(const Args& args) {
  if (args.m <= 128 && args.n <= 128 && args.k <= 128) {
    return "small";
  }
  if (args.m <= 1024 && args.n <= 1024 && args.k <= 1024) {
    return "medium";
  }
  if (args.m == args.n && args.n == args.k) {
    return "large_square";
  }
  if (args.n <= 8) {
    return "skinny_n";
  }
  return "large_general";
}

std::string output_domain_contract_name(const Args& args) {
  if (residue_current_output_mode(args)) {
    return "rns_residue_current";
  }
  if (finite_benchmark_semantics(args.semantics)) {
    return "finite_u8_host";
  }
  if (args.semantics == BenchSemantics::WrapU64Mod2_64) {
    return "wrap64_u64_host";
  }
  if (exact_wide_benchmark_semantics(args.semantics)) {
    return "exact_wide_limb_host";
  }
  return "native_i64_u64_host";
}

uint32_t estimated_modulus_product_bits(const Args& args, const BenchmarkResult& result) {
  if (finite_benchmark_semantics(args.semantics)) {
    return args.finite_modulus <= 1 ? 0u : 8u;
  }
  const uint32_t prefix = selected_execution_prefix(args, result);
  return prefix * 8u;
}

void print_reuse_contract_json(
    const Args& args,
    const BenchmarkResult& result,
    const char* selected_backend,
    const char* selected_kernel) {
  const bool reuse_enabled =
      args.reuse_packed_inputs || args.residue_chain_length > 1 || host_api_batch_requested(args) ||
      grouped_dispatch_requested(args);
  std::cout << "  \"reuse_contract\": {\n";
  std::cout << "    \"enabled\": " << (reuse_enabled ? "true" : "false") << ",\n";
  std::cout << "    \"operand_role\": \"" << reuse_operand_role(args) << "\",\n";
  std::cout << "    \"source_version_inputs\": \"monotonic_source_version_per_repeat_when_packing_runs\",\n";
  std::cout << "    \"setup_scope\": \"" << benchmark_setup_scope(args) << "\",\n";
  std::cout << "    \"setup_cost_us\": ";
  if (result.prepack_setup_available) {
    std::cout << result.prepack_setup_us;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"measured_repeat_count\": " << args.repeats << ",\n";
  std::cout << "    \"break_even_repeat_count\": null,\n";
  std::cout << "    \"output_domain\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  std::cout << "    \"next_op\": \"" << json_escape(resolved_next_op_hint(args, result)) << "\",\n";
  std::cout << "    \"target_fingerprint\": \"" << json_escape(benchmark_key_target_id(result)) << "\",\n";
  std::cout << "    \"backend_fingerprint\": \"" << json_escape(selected_backend) << "\",\n";
  std::cout << "    \"kernel_fingerprint\": ";
  if (selected_kernel) {
    std::cout << "\"" << json_escape(selected_kernel) << "\"";
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"workspace_fingerprint\": \"" << result.backend_info.workspace_required_bytes
            << "B:" << json_escape(result.backend_info.workspace_mode) << "\",\n";
  std::cout << "    \"promotion_eligible\": " << (!reuse_enabled ? "true" : "false") << ",\n";
  std::cout << "    \"invalidation_reasons\": [";
  bool first = true;
  if (args.reuse_packed_inputs) {
    std::cout << "\"source_version_changed\"";
    first = false;
  }
  if (grouped_dispatch_requested(args)) {
    std::cout << (first ? "" : ", ") << "\"descriptor_identity_changed\"";
    first = false;
  }
  if (args.hip_graph_replay) {
    std::cout << (first ? "" : ", ") << "\"graph_capture_descriptor_changed\"";
    first = false;
  }
  (void)first;
  std::cout << "]\n";
  std::cout << "  },\n";
}

void print_exact_output_contract_json(
    const Args& args,
    const BenchmarkResult& result,
    int64_t output_ld,
    const char* selected_kernel) {
  std::cout << "  \"exact_output_contract\": {\n";
  std::cout << "    \"requested_final_output\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  std::cout << "    \"limb_count\": ";
  if (exact_wide_benchmark_semantics(args.semantics)) {
    std::cout << args.exact_wide_limb_count;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"output_logical_ld\": " << output_ld << ",\n";
  std::cout << "    \"status_policy\": \"" << output_status_handling(args) << "\",\n";
  std::cout << "    \"kernel_identity\": ";
  if (selected_kernel) {
    std::cout << "\"" << json_escape(selected_kernel) << "\"";
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"output_domain_after_measured_repeats\": \""
            << (residue_current_output_mode(args) ? "rns_residue_current" : json_escape(output_domain_contract_name(args)))
            << "\",\n";
  std::cout << "    \"final_checksum_export_after_repeats\": "
            << (residue_current_output_mode(args) ? "true" : "false") << "\n";
  std::cout << "  },\n";
  (void)result;
}

void print_export_variant_json(
    const Args& args,
    const char* selected_kernel) {
  const bool default_variant = args.export_variant == "default";
  std::cout << "  \"export_variant\": {\n";
  std::cout << "    \"name\": \"" << json_escape(args.export_variant) << "\",\n";
  std::cout << "    \"source\": \"" << (default_variant ? "current_backend_export_path" : "benchmark_cli_evidence_mode") << "\",\n";
  std::cout << "    \"limb_count\": ";
  if (exact_wide_benchmark_semantics(args.semantics)) {
    std::cout << args.exact_wide_limb_count;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"status_policy\": \"" << output_status_handling(args) << "\",\n";
  std::cout << "    \"selected_kernel\": ";
  if (selected_kernel) {
    std::cout << "\"" << json_escape(selected_kernel) << "\"";
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"constants_placement\": \"backend_default\",\n";
  std::cout << "    \"promotion_eligible\": " << (default_variant ? "true" : "false") << ",\n";
  std::cout << "    \"promotion_blocker\": ";
  print_nullable_std_string(default_variant ? std::string() : std::string("experimental_export_variant"));
  std::cout << "\n";
  std::cout << "  },\n";
}

void print_reconstruction_variant_json(
    const Args& args,
    const BenchmarkResult& result,
    const char* selected_kernel) {
  const bool default_variant = args.reconstruction_variant == "default_garner";
  std::cout << "  \"reconstruction_variant\": {\n";
  std::cout << "    \"name\": \"" << json_escape(args.reconstruction_variant) << "\",\n";
  std::cout << "    \"family\": \"" << (default_variant ? "garner_fixed_prefix" : "benchmark_reconstruction_zoo") << "\",\n";
  std::cout << "    \"prefix_count\": " << selected_execution_prefix(args, result) << ",\n";
  std::cout << "    \"kernel_identity\": ";
  if (selected_kernel) {
    std::cout << "\"" << json_escape(selected_kernel) << "\"";
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"controller\": \"benchmark_metadata_only\",\n";
  std::cout << "    \"promotion_eligible\": " << (default_variant ? "true" : "false") << ",\n";
  std::cout << "    \"promotion_blocker\": ";
  print_nullable_std_string(default_variant ? std::string() : std::string("experimental_reconstruction_variant"));
  std::cout << "\n";
  std::cout << "  },\n";
}

void print_modulus_and_residue_policy_json(const Args& args, const BenchmarkResult& result) {
  const bool experimental = args.modulus_set != "default";
  std::cout << "  \"modulus_set\": {\n";
  std::cout << "    \"name\": \"" << json_escape(args.modulus_set) << "\",\n";
  std::cout << "    \"source\": \"" << (experimental ? "benchmark_experimental_ladder" : "rns8_default_modulus_ladder")
            << "\",\n";
  std::cout << "    \"execution_ladder\": \"" << (finite_benchmark_semantics(args.semantics) ? "finite_single_modulus"
                                                                               : "rns8_default_8bit_coprime_ladder")
            << "\",\n";
  std::cout << "    \"experimental\": " << (experimental ? "true" : "false") << ",\n";
  std::cout << "    \"product_bits\": " << estimated_modulus_product_bits(args, result) << ",\n";
  std::cout << "    \"prefix_count\": " << selected_execution_prefix(args, result) << ",\n";
  std::cout << "    \"pairwise_coprime_proof\": \"schema_declared_current_ladder_or_offline_search_report\",\n";
  std::cout << "    \"reducer_cost_hint\": \"" << (experimental ? "offline_search_required" : "backend_default")
            << "\",\n";
  std::cout << "    \"cache_promotion_blocker\": ";
  print_nullable_std_string(experimental ? std::string("experimental_modulus_set") : std::string());
  std::cout << "\n";
  std::cout << "  },\n";
  std::cout << "  \"residue_count_policy\": {\n";
  std::cout << "    \"policy\": \"" << prefix_policy_name(args, result) << "\",\n";
  std::cout << "    \"requested_prefix\": " << benchmark_prefix(args) << ",\n";
  std::cout << "    \"selected_prefix\": " << selected_execution_prefix(args, result) << ",\n";
  std::cout << "    \"minimum_range_prefix\": " << result.schedule_info.min_required_prefix << ",\n";
  std::cout << "    \"redundant_residue_count\": "
            << (selected_execution_prefix(args, result) > result.schedule_info.min_required_prefix
                    ? selected_execution_prefix(args, result) - result.schedule_info.min_required_prefix
                    : 0)
            << ",\n";
  std::cout << "    \"autotune_scope\": \"" << (experimental ? "evidence_only_non_promoting" : "current_exact_cache")
            << "\",\n";
  std::cout << "    \"cache_promotion_blocker\": ";
  print_nullable_std_string(experimental ? std::string("experimental_residue_count_policy") : std::string());
  std::cout << "\n";
  std::cout << "  },\n";
}

void print_tile_shape_variant_json(
    const Args& args,
    const BenchmarkResult& result,
    const char* selected_kernel) {
  const bool default_variant = args.tile_shape_variant == "default";
  const int64_t k_block = benchmark_k_block_size(args, result);
  std::cout << "  \"tile_shape_variant\": {\n";
  std::cout << "    \"name\": \"" << json_escape(args.tile_shape_variant) << "\",\n";
  std::cout << "    \"tile_m\": " << args.tile_m << ",\n";
  std::cout << "    \"tile_n\": " << args.tile_n << ",\n";
  std::cout << "    \"tile_k\": " << k_block << ",\n";
  std::cout << "    \"selected_kernel_identity\": ";
  if (selected_kernel) {
    std::cout << "\"" << json_escape(selected_kernel) << "\"";
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"resource_report_key\": \"tile_m=" << args.tile_m << ";tile_n=" << args.tile_n
            << ";tile_k=" << k_block << ";kernel=" << (selected_kernel ? json_escape(selected_kernel) : "unknown")
            << "\",\n";
  std::cout << "    \"shape_family_bucket\": \"" << shape_family_bucket(args) << "\",\n";
  std::cout << "    \"stale_kernel_rejection\": \"selected_kernel_identity_must_match_capture\"\n";
  std::cout << "  },\n";
  (void)default_variant;
}

void print_dispatch_and_graph_json(const Args& args, const BenchmarkResult& result) {
  const bool grouped = grouped_dispatch_requested(args);
  const bool adaptive_grouped = args.adaptive_grouped_scheduler;
  const std::string grouped_strategy =
      grouped && result.grouped_dispatch_execution_strategy == "not_requested"
          ? "host_phase_loop_per_task_export"
          : result.grouped_dispatch_execution_strategy;
  std::cout << "  \"grouped_dispatch\": {\n";
  std::cout << "    \"requested\": " << (grouped ? "true" : "false") << ",\n";
  std::cout << "    \"task_count\": " << args.grouped_dispatch_tasks << ",\n";
  std::cout << "    \"descriptor_identity\": \"same_shape_m=" << args.m << ";n=" << args.n << ";k=" << args.k
            << ";semantics=" << semantics_name(args.semantics) << "\",\n";
  std::cout << "    \"source_hash\": \"" << args.seed << "\",\n";
  std::cout << "    \"output_hash\": \"final_checksum_u64\",\n";
  std::cout << "    \"setup_scope\": \"" << benchmark_setup_scope(args) << "\",\n";
  std::cout << "    \"execution_strategy\": \""
            << (grouped ? json_escape(grouped_strategy) : "not_requested") << "\",\n";
  std::cout << "    \"batched_export_enabled\": "
            << (grouped && result.grouped_dispatch_batched_export_enabled ? "true" : "false") << ",\n";
  std::cout << "    \"device_output_slab_bytes\": "
            << (grouped ? result.grouped_dispatch_device_output_slab_bytes : 0) << ",\n";
  std::cout << "    \"capture_status\": \""
            << (grouped ? "executed" : "not_requested") << "\",\n";
  std::cout << "    \"unsupported_reason\": ";
  print_nullable_std_string(std::string());
  std::cout << ",\n";
  std::cout << "    \"promotion_eligible\": false\n";
  std::cout << "  },\n";
  std::cout << "  \"adaptive_grouped_scheduler\": {\n";
  std::cout << "    \"requested\": " << (adaptive_grouped ? "true" : "false") << ",\n";
  std::cout << "    \"strategy\": \""
            << (adaptive_grouped ? "prefix_tile_zero_mask_grouped_descriptors"
                                 : "current_compact_active_prefix_schedule")
            << "\",\n";
  std::cout << "    \"descriptor_identity\": \"prefix=" << selected_execution_prefix(args, result)
            << ";tile_m=" << args.tile_m << ";tile_n=" << args.tile_n
            << ";zero_tiles=" << result.zero_output_tile_count << "\",\n";
  std::cout << "    \"group_count\": "
            << (adaptive_grouped ? std::max<uint64_t>(1, result.schedule_info.prefix_group_count) : 0) << ",\n";
  std::cout << "    \"active_tile_count\": " << result.schedule_info.tile_count << ",\n";
  std::cout << "    \"zero_tile_count\": " << result.zero_output_tile_count << ",\n";
  std::cout << "    \"selected_prefix_histogram\": \"min=" << result.schedule_info.min_selected_prefix
            << ";max=" << result.schedule_info.max_selected_prefix << "\",\n";
  std::cout << "    \"capture_status\": \""
            << (adaptive_grouped ? "metadata_only_unsupported_for_execution_path" : "not_requested") << "\",\n";
  std::cout << "    \"unsupported_reason\": ";
  print_nullable_std_string(adaptive_grouped ? std::string("adaptive_grouped_scheduler_not_executed_by_current_path")
                                             : std::string());
  std::cout << ",\n";
  std::cout << "    \"promotion_eligible\": false\n";
  std::cout << "  },\n";
}

void print_residency_arena_overlap_json(const Args& args, const BenchmarkResult& result) {
  const bool residency = args.resident_lifetime || args.reuse_packed_inputs || args.residue_chain_length > 1;
  const bool arena = args.workspace_arena;
  const bool repeat_alloc_free =
      result.allocation_tracking_available &&
      (result.allocation_after_repeats.allocate_calls == result.allocation_after_warmups.allocate_calls) &&
      (result.allocation_after_repeats.free_calls == result.allocation_after_warmups.free_calls);
  std::cout << "  \"resident_lifetime\": {\n";
  std::cout << "    \"enabled\": " << (residency ? "true" : "false") << ",\n";
  std::cout << "    \"matrix_roles\": \"" << (residency ? "A/B/C explicit benchmark resident roles" : "transient") << "\",\n";
  std::cout << "    \"source_version_policy\": \"monotonic_per_import_or_pack\",\n";
  std::cout << "    \"current_storage_state\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  std::cout << "    \"output_domain\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  std::cout << "    \"workspace_identity\": \"" << result.backend_info.workspace_required_bytes
            << "B:" << json_escape(result.backend_info.workspace_mode) << "\",\n";
  std::cout << "    \"stale_source_rejection\": \"source_version_descriptor_semantic_prefix_target_workspace_mismatch\",\n";
  std::cout << "    \"promotion_eligible\": false\n";
  std::cout << "  },\n";
  std::cout << "  \"workspace_arena\": {\n";
  std::cout << "    \"enabled\": " << (arena ? "true" : "false") << ",\n";
  std::cout << "    \"arena_identity\": \"" << json_escape(result.backend_info.autotune_key) << "|"
            << json_escape(result.backend_info.workspace_mode) << "\",\n";
  std::cout << "    \"size_bytes\": " << result.backend_info.workspace_required_bytes << ",\n";
  std::cout << "    \"high_water_mark_bytes\": " << result.backend_info.workspace_required_bytes << ",\n";
  std::cout << "    \"suballocation_count\": " << (arena ? 5 : 0) << ",\n";
  std::cout << "    \"measured_repeat_allocation_free\": "
            << (repeat_alloc_free ? "true" : "false") << ",\n";
  std::cout << "    \"source_version_policy\": \"plan_target_backend_semantic_shape_prefix_output_policy\",\n";
  std::cout << "    \"stream_safety\": \"" << (args.streaming_overlap ? "event_guarded_pipeline_lanes" : "single_stream_owner")
            << "\",\n";
  std::cout << "    \"promotion_eligible\": false\n";
  std::cout << "  },\n";
  std::cout << "  \"streaming_overlap\": {\n";
  std::cout << "    \"requested\": " << (args.streaming_overlap ? "true" : "false") << ",\n";
  std::cout << "    \"pipeline\": \""
            << (args.streaming_overlap ? "pack_next_gemm_current_export_previous" : "serial_default_stream")
            << "\",\n";
  std::cout << "    \"buffering\": \"" << (args.streaming_overlap ? "double_buffered_benchmark_only" : "none") << "\",\n";
  std::cout << "    \"dependency_contract\": \"pack_before_gemm;gemm_before_export;status_before_host_read;final_sync_before_checksum\",\n";
  std::cout << "    \"transfer_policy\": \"compact_or_padded_output_policy_declared_by_output_policy\",\n";
  std::cout << "    \"capture_status\": \""
            << (args.streaming_overlap ? "metadata_only_unsupported_for_execution_path" : "not_requested") << "\",\n";
  std::cout << "    \"unsupported_reason\": ";
  print_nullable_std_string(args.streaming_overlap ? std::string("streaming_overlap_not_executed_by_current_path")
                                                   : std::string());
  std::cout << ",\n";
  std::cout << "    \"promotion_eligible\": false\n";
  std::cout << "  },\n";
}

void print_workload_proxy_json(const Args& args) {
  const bool enabled = args.workload_proxy != "none";
  std::string family = "not_requested";
  if (enabled) {
    family = (args.workload_proxy.find("fhe") != std::string::npos ||
              args.workload_proxy.find("lattice") != std::string::npos ||
              args.workload_proxy.find("ckks") != std::string::npos)
                 ? "fhe_lattice_proxy"
                 : "dense_exact_arithmetic_proxy";
  }
  std::cout << "  \"workload_proxy\": {\n";
  std::cout << "    \"enabled\": " << (enabled ? "true" : "false") << ",\n";
  std::cout << "    \"label\": \"" << json_escape(args.workload_proxy) << "\",\n";
  std::cout << "    \"family\": \"" << json_escape(family) << "\",\n";
  std::cout << "    \"tower_role\": \"" << (enabled ? "dense_gemm_adjacent_proxy" : "none") << "\",\n";
  std::cout << "    \"reuse_profile\": \"" << (args.reuse_packed_inputs ? reuse_operand_role(args) : "none") << "\",\n";
  std::cout << "    \"transform_role\": \"" << (enabled ? "not_a_public_fhe_backend" : "none") << "\",\n";
  std::cout << "    \"output_domain_requirement\": \"" << json_escape(output_domain_contract_name(args)) << "\",\n";
  std::cout << "    \"compatibility_claim\": false\n";
  std::cout << "  },\n";
}

void print_release_and_verification_json(const Args& args, const BenchmarkResult& result) {
  const bool gate_requested = args.release_gate != "none";
  const bool amortized = args.verification_amortization != "none";
  std::cout << "  \"release_gate\": {\n";
  std::cout << "    \"name\": \"" << json_escape(args.release_gate) << "\",\n";
  std::cout << "    \"requested\": " << (gate_requested ? "true" : "false") << ",\n";
  std::cout << "    \"classification_tier\": \""
            << (gate_requested ? "cpu_backed_release_candidate_pending_review" : "not_requested") << "\",\n";
  std::cout << "    \"cpu_reference_policy\": \"chunked_when_large_fixed_seed_checksum_recorded\",\n";
  std::cout << "    \"memory_cap_policy\": \"declared_by_sweep_runner_or_not_applicable\",\n";
  std::cout << "    \"resume_policy\": \"scenario_id_and_capture_path_stable_under_temp\",\n";
  std::cout << "    \"review_status\": \""
            << (gate_requested ? "pending_reviewed_summary" : "not_requested") << "\",\n";
  std::cout << "    \"cache_eligible\": false,\n";
  std::cout << "    \"blockers\": [";
  if (gate_requested) {
    std::cout << "\"reviewed_summary_missing\", \"release_A_B_margin_missing\"";
  }
  std::cout << "]\n";
  std::cout << "  },\n";
  std::cout << "  \"verification_amortization\": {\n";
  std::cout << "    \"enabled\": " << (amortized ? "true" : "false") << ",\n";
  std::cout << "    \"policy\": \"" << json_escape(args.verification_amortization) << "\",\n";
  std::cout << "    \"reused_reference_structure\": \""
            << (amortized ? "shape_seed_semantic_reference_inputs" : "none") << "\",\n";
  std::cout << "    \"final_exact_comparison_required\": true,\n";
  std::cout << "    \"final_exact_comparison_status\": \""
            << (result.checksum != 0 ? "checksum_recorded_reference_required" : "reference_required") << "\",\n";
  std::cout << "    \"promotion_eligible\": false\n";
  std::cout << "  },\n";
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
  const int64_t output_ld = output_logical_ld(args);
  const bool global_bound_scan_available = result.global_bound_scan_available;
  const uint32_t residue_planes_skipped = prefix > selected_prefix ? prefix - selected_prefix : 0;
  const double residue_plane_skip_fraction =
      prefix == 0 ? 0.0 : static_cast<double>(residue_planes_skipped) / static_cast<double>(prefix);
  const bool adaptive_applied = adaptive_execution_applied(args, info, result);
  const bool per_modulus_estimate_applicable =
      selected_prefix > 0 && args.bound_mode != BoundMode::PerTile && !adaptive_applied && !args.oneshot &&
      !args.vector_alu_baseline && !runtime_vector_alu_backend(args) && !vector_to_rns_chain_requested(args) &&
      !grouped_task_executor_requested(args) && !hip_graph_replay_requested(args);
  const double avg_per_modulus_gemm_estimate_us =
      per_modulus_estimate_applicable ? avg_gemm_us / static_cast<double>(selected_prefix) : avg_gemm_us;
  const bool gpu_events_available = gpu_event_timing_available(args, result);
  const rns8_backend_kind selected_backend_kind = selected_backend_for_events(args, result);
  const bool use_prepacked_b_cache = uses_runtime_b_prepack_cache(result);
  const bool chain_residue_output = residue_current_output_mode(args);
  const bool chain_final_export = residue_chain_final_export_requested(args);
  const bool host_batch = host_api_batch_requested(args);
  const bool grouped_dispatch = grouped_dispatch_requested(args);
  const uint32_t task_count = measured_task_count(args);
  const double task_count_denominator = static_cast<double>(task_count);
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
  const bool oneshot_resident_fallback_events =
      gpu_events_available &&
      direct_hip_bounded_oneshot_resident_fallback_path(args, result, selected_backend_kind);
  const bool native_to_rns_bridge_events =
      gpu_events_available && native_to_rns_bridge_path(args, result, selected_backend_kind);
  const bool vector_to_rns_chain_events =
      gpu_events_available && vector_to_rns_chain_path(args, result, selected_backend_kind);
  const bool all_zero_direct_hip_pack_elided =
      all_zero_direct_hip_input_pack_elided(args, result, selected_backend_kind);
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
    if (chain_residue_output) {
      gpu_event_reason = "captured_by_residue_current_chain_backend_hooks";
      if (hipblaslt_events) {
        gpu_event_scope = "\"hipblaslt_baseline_default_stream_backend_operation_groups\"";
      } else if (accelerator_events) {
        gpu_event_scope = "\"accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export\"";
      } else {
        gpu_event_scope = "\"direct_hip_default_stream_backend_operation_groups\"";
      }
      if (hipblaslt_events) {
        gpu_event_caveat =
            "\"HIP event timings record per-repeat pack operation groups and the chained hipBLASLt rns_gemm "
            "operation groups that keep intermediate outputs resident in RNS form; hipBLASLt library operations use "
            "a benchmark-only synchronization before stop-event recording to make library work event-visible; the "
            "final checksum-only logical export runs after measured repeats and is intentionally absent from "
            "gpu_event_phase_order\"";
      } else {
        gpu_event_caveat =
            "\"HIP event timings record per-repeat pack operation groups and the chained rns_gemm backend operation "
            "groups that keep intermediate outputs resident in RNS form; the final checksum-only logical export runs "
            "after measured repeats and is intentionally absent from gpu_event_phase_order\"";
      }
    } else if (oneshot_hip_events) {
      if (oneshot_resident_fallback_events) {
        gpu_event_reason = "captured_by_direct_hip_oneshot_resident_fallback_api_hooks";
        gpu_event_scope = "\"direct_hip_oneshot_resident_fallback_default_stream_operation_groups\"";
        gpu_event_caveat =
            "\"HIP event timings record the public bounded one-shot API's resident fallback pack, direct-HIP GEMM "
            "kernel group, and logical export operation groups; host wall-clock timings remain required for plan "
            "creation, transient matrix/workspace allocation, API dispatch, CPU scheduling, teardown, and synchronous "
            "host-side overhead not represented on the HIP stream\"";
      } else {
        gpu_event_reason = "captured_by_direct_hip_oneshot_api_hooks";
        gpu_event_scope = "\"direct_hip_oneshot_default_stream_operation_groups\"";
        gpu_event_caveat =
            "\"HIP event timings record the public one-shot API's native input H2D copies, direct-HIP GEMM "
            "kernel group, and logical export operation groups; host wall-clock timings remain required for plan "
            "creation, transient allocations, API dispatch, CPU scheduling, teardown, and synchronous host-side "
            "overhead not represented on the HIP stream\"";
      }
    } else if (native_to_rns_bridge_events) {
      gpu_event_reason = "captured_by_direct_hip_native_to_rns_bridge_hooks";
      gpu_event_scope = "\"direct_hip_native_to_rns_bridge_default_stream_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record direct-HIP pack/export operation groups plus the forced native-to-RNS "
          "device conversion kernels inside rns_gemm; host wall-clock timings remain required for API dispatch, "
          "CPU scheduling, allocations, and synchronous host-side overhead not represented on the HIP stream\"";
    } else if (vector_to_rns_chain_events) {
      gpu_event_reason = "captured_by_vector_native_to_direct_rns_chain_hooks";
      gpu_event_scope = "\"direct_hip_vector_native_to_rns_chain_default_stream_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record vector-ALU producer packing and GEMM events, Direct-HIP second-input "
          "pack/export operation groups, the native-to-RNS device materialization kernel, and the Direct-HIP "
          "consumer RNS GEMM operation group; host wall-clock timings remain required for API dispatch, CPU "
          "scheduling, allocations, and synchronous host-side overhead not represented on the HIP stream\"";
    } else if (hipblaslt_events) {
      gpu_event_reason = "captured_by_hipblaslt_backend_hooks";
      gpu_event_scope = "\"hipblaslt_baseline_default_stream_backend_operation_groups\"";
      gpu_event_caveat =
          "\"HIP event timings record hipBLASLt baseline operation groups; hipBLASLt library operations use a "
          "benchmark-only synchronization before stop-event recording to make library work event-visible. Host "
          "wall-clock timings remain required for API dispatch, descriptor setup, allocations, and synchronous "
          "host-side overhead not represented on the HIP stream\"";
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
  } else if (result.gpu_events.requested) {
    gpu_event_reason = "backend_event_capture_incomplete";
    gpu_event_status = "unavailable_missing_expected_events";
  } else if (result.hip_graph_replay_requested) {
    gpu_event_reason = "hip_graph_replay_wall_clock_only";
    gpu_event_status = result.hip_graph_replay_used ? "not_requested_graph_replay" : "not_requested_graph_replay_unavailable";
  } else if (chain_residue_output) {
    gpu_event_reason = "selected_chain_backend_has_no_gpu_event_hooks";
    gpu_event_status = "not_requested_for_selected_chain_backend";
  }

  std::cout << "{\n";
  std::cout << "  \"schema_version\": " << kBenchmarkSchemaVersion << ",\n";
  std::cout << "  \"benchmark\": \"" << benchmark_name(args) << "\",\n";
  std::cout << "  \"benchmark_execution_mode\": \"" << benchmark_execution_mode_name(args, result) << "\",\n";
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
  std::cout << "    \"source\": \"" << backend_metadata_source(args, result) << "\",\n";
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
  print_plan_packing_json(result);
  print_plan_lowering_json(result);
  print_requested_next_op_json(args, result);
  print_output_policy_json(args, output_ld);
  print_target_variant_json(args, info, result, selected_backend);
  print_auto_selector_json(args, info, result, selected_backend);
  print_device_allocation_json(args, result);
  print_reuse_contract_json(args, result, selected_backend, selected_kernel);
  print_exact_output_contract_json(args, result, output_ld, selected_kernel);
  print_export_variant_json(args, selected_kernel);
  print_reconstruction_variant_json(args, result, selected_kernel);
  print_modulus_and_residue_policy_json(args, result);
  print_tile_shape_variant_json(args, result, selected_kernel);
  print_dispatch_and_graph_json(args, result);
  print_residency_arena_overlap_json(args, result);
  print_workload_proxy_json(args);
  print_release_and_verification_json(args, result);
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
  std::cout << "  \"output_logical_ld\": " << output_ld << ",\n";
  std::cout << "  \"output_ld_padding\": " << args.output_ld_padding << ",\n";
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
  std::cout << "    \"zero_a_row_proof_count\": " << result.zero_a_row_proof_count << ",\n";
  std::cout << "    \"zero_b_col_proof_count\": " << result.zero_b_col_proof_count << ",\n";
  std::cout << "    \"zero_row_col_product_count\": " << result.zero_row_col_product_count << ",\n";
  std::cout << "    \"planner_zero_a_row_count\": " << result.schedule_info.zero_a_row_count << ",\n";
  std::cout << "    \"planner_zero_b_col_count\": " << result.schedule_info.zero_b_col_count << ",\n";
  std::cout << "    \"planner_zero_row_col_product_count\": "
            << result.schedule_info.zero_row_col_product_count << ",\n";
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
  std::cout << "  \"residue_chain_final_export\": " << (chain_final_export ? "true" : "false") << ",\n";
  std::cout << "  \"residue_output_mode\": \"" << residue_output_mode_name(args) << "\",\n";
  std::cout << "  \"host_api_batch\": {\n";
  std::cout << "    \"enabled\": " << (host_batch ? "true" : "false") << ",\n";
  std::cout << "    \"batch_size\": " << (host_batch ? args.host_api_batch_size : 1) << ",\n";
  std::cout << "    \"tasks_per_measured_repeat\": " << (host_batch ? args.host_api_batch_size : 1) << ",\n";
  std::cout << "    \"total_measured_tasks\": "
            << static_cast<uint64_t>(host_batch ? args.host_api_batch_size : 1) * static_cast<uint64_t>(args.repeats)
            << ",\n";
  std::cout << "    \"setup_scope\": \""
            << (host_batch
                    ? "one_shared_plan_per_capture_one_resident_matrix_workspace_triplet_per_task"
                    : "single_task_default_benchmark_mode")
            << "\",\n";
  std::cout << "    \"timing_policy\": \""
            << (host_batch ? "aggregate_batch_totals_per_measured_repeat" : "single_call_totals_per_measured_repeat")
            << "\",\n";
  std::cout << "    \"checksum_policy\": \""
            << (host_batch ? "fnv1a_over_final_task_output_checksums" : "single_final_output_checksum")
            << "\"\n";
  std::cout << "  },\n";
  std::cout << "  \"hip_graph_replay\": {\n";
  std::cout << "    \"requested\": " << (result.hip_graph_replay_requested ? "true" : "false") << ",\n";
  std::cout << "    \"available\": " << (result.hip_graph_replay_available ? "true" : "false") << ",\n";
  std::cout << "    \"used\": " << (result.hip_graph_replay_used ? "true" : "false") << ",\n";
  std::cout << "    \"status\": \"" << json_escape(result.hip_graph_replay_status) << "\",\n";
  std::cout << "    \"scope\": \"" << json_escape(result.hip_graph_replay_scope) << "\",\n";
  std::cout << "    \"descriptor_identity\": \"fixed_plan_workspace_descriptor:m=" << args.m << ";n=" << args.n
            << ";k=" << args.k << "\",\n";
  std::cout << "    \"plan_identity\": \"" << json_escape(result.backend_info.autotune_key) << "\",\n";
  std::cout << "    \"setup_scope\": \"" << benchmark_setup_scope(args) << "\",\n";
  std::cout << "    \"capture_status\": \""
            << (result.hip_graph_replay_used ? "replayed" : "not_requested") << "\",\n";
  std::cout << "    \"unsupported_reason\": ";
  print_nullable_std_string(
      result.hip_graph_replay_requested && !result.hip_graph_replay_used ? result.hip_graph_replay_caveat
                                                                         : std::string());
  std::cout << ",\n";
  std::cout << "    \"promotion_eligible\": false,\n";
  std::cout << "    \"capture_us\": " << result.hip_graph_capture_us << ",\n";
  std::cout << "    \"instantiate_us\": " << result.hip_graph_instantiate_us << ",\n";
  std::cout << "    \"graph_launches_per_measured_repeat\": "
            << (result.hip_graph_replay_used ? 1 : 0) << ",\n";
  std::cout << "    \"total_graph_launches\": " << result.hip_graph_launch_count << ",\n";
  std::cout << "    \"captured_chain_length\": "
            << (result.hip_graph_replay_used ? args.residue_chain_length : 0) << ",\n";
  std::cout << "    \"timing_policy\": \""
            << (result.hip_graph_replay_used
                    ? "raw_timings_us.rns_gemm_and_end_to_end_measure_one_hipGraphLaunch_plus_stream_sync"
                    : "not_applicable")
            << "\",\n";
  std::cout << "    \"setup_policy\": \""
            << (result.hip_graph_replay_used
                    ? "A_B_prepack_before_capture_capture_and_instantiate_before_warmups"
                    : "not_applicable")
            << "\",\n";
  std::cout << "    \"final_export_policy\": \""
            << (result.hip_graph_replay_used
                    ? "one_final_logical_export_after_measured_repeats_for_checksum_only"
                    : "not_applicable")
            << "\",\n";
  std::cout << "    \"caveat\": ";
  print_nullable_std_string(result.hip_graph_replay_caveat);
  std::cout << "\n";
  std::cout << "  },\n";
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
  if (hip_graph_replay_requested(args)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for benchmark-only HIP Graph replay of a "
              << "Direct-HIP residue-current RNS GEMM chain; A and B are packed once before graph capture, "
              << "capture_us and instantiate_us record one-time graph setup before warmups, each measured repeat "
              << "runs one hipGraphLaunch plus stream synchronization containing " << args.residue_chain_length
              << " resident RNS GEMM launches, raw_timings_us.pack and raw_timings_us.crt_export are zero, and "
                 "one final logical export runs after measured repeats only to produce checksum_u64\",\n";
  } else if (chain_residue_output) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for a residue-current RNS GEMM chain; "
              << "each measured repeat runs " << args.residue_chain_length
              << " resident RNS GEMM calls before host export, raw_timings_us.crt_export is intentionally zero, and "
                 "one final logical export runs after measured repeats only to produce checksum_u64\",\n";
  } else if (chain_final_export) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for a final-output RNS GEMM chain; each measured "
              << "repeat runs " << args.residue_chain_length
              << " resident RNS GEMM calls with intermediate outputs kept in RNS form, then exports the final "
                 "logical output inside the measured repeat so raw_timings_us.crt_export and end_to_end include "
                 "the requested host-output contract\",\n";
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
  } else if (bounded_residue_channel_fusion_requested(args)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for a benchmark-only direct-HIP residue-channel "
                 "fusion experiment over width-3 groups from the first fixed prefix-9 moduli; this capture is "
                 "not AUTO-selected and is evidence-only for pack/layout comparison\",\n";
  } else if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for an explicit benchmark-owned direct-HIP "
                 "uniform-small native-input path; each measured repeat copies A and B as row-major int8 inputs, "
                 "runs the existing prefix-9 colpair centered-residue GEMM kernel group, and exports through the "
                 "normal CRT output path\",\n";
  } else if (all_zero_direct_hip_pack_elided) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for a trusted all-zero direct-HIP adaptive "
                 "per-tile bounded capture; raw_timings_us.pack is zero because exact tile-bound scheduling "
                 "proved every output tile zero before measurement, so the backend materializes resident RNS "
                 "zero output without reading A or B\",\n";
  } else if (rns_residue_chain_requested(args)) {
    std::cout << "      \"matrix_alloc\": \"one-time persistent A/B/C matrix allocation plus one scratch matrix for alternating chained RNS outputs\",\n";
  } else if (vector_to_rns_chain_requested(args)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for a vector-native-to-Direct-RNS chain; "
                 "each measured repeat packs producer A/B into the HIP vector-ALU backend, ";
    if (args.reuse_packed_b) {
      std::cout << "reuses a Direct-HIP consumer B input that was packed once before warmups, ";
    } else {
      std::cout << "packs a second Direct-HIP RNS input, ";
    }
    std::cout << "runs vector GEMM to native device output, materializes that native output into Direct-HIP RNS "
                 "storage, runs the Direct-HIP consumer RNS GEMM, and exports the final logical output";
    if (args.reuse_packed_b) {
      std::cout << "; per-repeat end_to_end excludes one-time prepack_setup_us";
    }
    std::cout << "\",\n";
  } else if (native_to_rns_bridge_requested(args)) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for an explicit AUTO/direct-HIP native-to-RNS "
                 "bridge benchmark; each measured repeat packs bounded inputs into direct-HIP matrices, forces "
                 "resident RNS input residues stale while keeping native device buffers current, runs the existing "
                 "native-to-RNS conversion kernels inside rns8_gemm_rns, then runs the normal direct-HIP RNS GEMM "
                 "and CRT export\",\n";
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
  } else if (grouped_dispatch) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for benchmark-owned grouped dispatch evidence; each "
              << "measured repeat packs, runs, exports, and checks " << task_count
              << " independent same-shape resident tasks through one shared plan and one resident matrix/workspace "
                 "triplet per task; raw_timings_us values are aggregate grouped totals, avg_*_per_task_us fields "
                 "divide those totals by grouped_dispatch.task_count";
    if (result.grouped_dispatch_batched_export_enabled) {
      std::cout << "; exact-wide export kernels write into one shared device output slab before one compact D2H copy";
    }
    std::cout << ", and this capture remains non-promotable until a real device grouped dispatcher exists\",\n";
  } else if (host_batch) {
    std::cout << "  \"timing_note\": \"host wall-clock timings for benchmark-owned host API batching; each measured "
              << "repeat packs, runs, exports, and checks " << task_count
              << " independent same-shape resident tasks through one shared plan and one resident matrix/workspace "
                 "triplet per task; raw_timings_us values are aggregate batch totals, and avg_*_per_task_us fields "
                 "divide those totals by host_api_batch.batch_size\",\n";
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
  std::cout << "    \"benchmark_execution_mode\": \"" << benchmark_execution_mode_name(args, result) << "\",\n";
  std::cout << "    \"pack_mode\": \"" << pack_mode_name(args) << "\",\n";
  std::cout << "    \"host_api_batch_enabled\": " << (host_batch ? "true" : "false") << ",\n";
  std::cout << "    \"host_api_batch_size\": " << (host_batch ? args.host_api_batch_size : 1) << ",\n";
  std::cout << "    \"grouped_dispatch_enabled\": " << (grouped_dispatch ? "true" : "false") << ",\n";
  std::cout << "    \"grouped_dispatch_task_count\": " << args.grouped_dispatch_tasks << ",\n";
  std::cout << "    \"grouped_dispatch_execution_strategy\": \""
            << (grouped_dispatch
                    ? json_escape(result.grouped_dispatch_execution_strategy == "not_requested"
                                      ? "host_phase_loop_per_task_export"
                                      : result.grouped_dispatch_execution_strategy)
                    : "not_requested")
            << "\",\n";
  std::cout << "    \"grouped_dispatch_batched_export_enabled\": "
            << (grouped_dispatch && result.grouped_dispatch_batched_export_enabled ? "true" : "false") << ",\n";
  std::cout << "    \"grouped_dispatch_device_output_slab_bytes\": "
            << (grouped_dispatch ? result.grouped_dispatch_device_output_slab_bytes : 0) << ",\n";
  std::cout << "    \"hip_graph_replay_enabled\": " << (result.hip_graph_replay_used ? "true" : "false") << ",\n";
  std::cout << "    \"hip_graph_replay_status\": \"" << json_escape(result.hip_graph_replay_status) << "\",\n";
  std::cout << "    \"hip_graph_replay_scope\": \"" << json_escape(result.hip_graph_replay_scope) << "\",\n";
  std::cout << "    \"hip_graph_capture_us\": " << result.hip_graph_capture_us << ",\n";
  std::cout << "    \"hip_graph_instantiate_us\": " << result.hip_graph_instantiate_us << ",\n";
  std::cout << "    \"hip_graph_total_launches\": " << result.hip_graph_launch_count << ",\n";
  std::cout << "    \"native_to_rns_bridge_forced\": "
            << (native_to_rns_bridge_requested(args) ? "true" : "false") << ",\n";
  std::cout << "    \"vector_to_rns_chain\": "
            << (vector_to_rns_chain_requested(args) ? "true" : "false") << ",\n";
  std::cout << "    \"vector_to_rns_chain_producer_backend\": "
            << (vector_to_rns_chain_requested(args) ? "\"hip-vector-alu-int64\"" : "null") << ",\n";
  std::cout << "    \"vector_to_rns_chain_consumer_backend\": "
            << (vector_to_rns_chain_requested(args) ? "\"hip-direct\"" : "null") << ",\n";
  std::cout << "    \"vector_to_rns_chain_consumer_k\": ";
  if (vector_to_rns_chain_requested(args)) {
    std::cout << args.n;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "    \"residue_chain_final_export\": " << (chain_final_export ? "true" : "false") << ",\n";
  std::cout << "    \"pack_layout\": \"" << json_escape(benchmark_pack_layout(args, result)) << "\",\n";
  std::cout << "    \"fusion_mode\": \"" << benchmark_fusion_mode(args) << "\",\n";
  std::cout << "    \"residue_group_width\": " << benchmark_residue_group_width(args) << ",\n";
  std::cout << "    \"residue_group_layout\": \"" << benchmark_residue_group_layout(args) << "\",\n";
  std::cout << "    \"generated_reducer_identity\": \"" << json_escape(generated_reducer_identity(args, result)) << "\",\n";
  std::cout << "    \"direct_hip_export_staging_policy\": \""
            << direct_hip_export_staging_policy(args, selected_backend_kind) << "\",\n";
  std::cout << "    \"direct_hip_pinned_export_staging_threshold_bytes\": "
            << kDirectHipPinnedExportStagingThresholdBytes << ",\n";
  std::cout << "    \"benchmark_output_destination_layout\": \"" << output_destination_layout(args) << "\",\n";
  std::cout << "    \"benchmark_output_logical_ld\": " << output_ld << ",\n";
  std::cout << "    \"benchmark_output_ld_padding\": " << args.output_ld_padding << ",\n";
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
  } else if (grouped_dispatch) {
    std::cout << "      \"matrix_alloc\": \"one-time benchmark-owned allocation of one resident A/B/C matrix triplet and workspace per grouped dispatch task\",\n";
  } else if (host_batch) {
    std::cout << "      \"matrix_alloc\": \"one-time benchmark-owned allocation of one resident A/B/C matrix triplet and workspace per host API batch task\",\n";
  } else if (bounded_residue_channel_fusion_requested(args)) {
    std::cout << "      \"matrix_alloc\": \"one-time benchmark-owned native int8 A/B HIP buffers plus resident RNS output matrix allocation for the experimental width-3 residue-channel fusion comparison\",\n";
  } else if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    std::cout << "      \"matrix_alloc\": \"one-time benchmark-owned native int8 A/B HIP buffers plus resident RNS output matrix allocation host timing\",\n";
  } else if (vector_to_rns_chain_requested(args)) {
    std::cout << "      \"matrix_alloc\": \"one-time vector-ALU producer matrix allocation plus Direct-HIP RNS consumer matrix allocation host timing\",\n";
  } else {
    std::cout << "      \"matrix_alloc\": \"one-time persistent matrix allocation host timing\",\n";
  }
  if (hip_graph_replay_requested(args)) {
    std::cout << "      \"pack\": \"zero-valued per-repeat phase; A and B were packed once into persistent RNS matrices before HIP Graph capture\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for one hipGraphLaunch plus stream synchronization containing "
              << args.residue_chain_length
              << " captured Direct-HIP resident RNS GEMM launches\",\n";
    std::cout << "      \"crt_export\": \"zero-valued per-repeat phase; residue-current graph replay defers host logical export until one final checksum export after measured repeats\",\n";
  } else if (chain_residue_output) {
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
  } else if (chain_final_export) {
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
              << " chained rns8_gemm_rns calls that keep intermediate outputs resident in RNS form\",\n";
    if (exact_wide_benchmark_semantics(args.semantics)) {
      std::cout << "      \"crt_export\": \"per-repeat host timing for exporting the final chained exact-wide limb output inside the measured repeat\",\n";
    } else {
      std::cout << "      \"crt_export\": \"per-repeat host timing for exporting/reconstructing the final chained logical output inside the measured repeat\",\n";
    }
  } else if (args.oneshot) {
    std::cout << "      \"pack\": \"zero-valued external phase; native input copies and any backend-local transformation are inside the measured one-shot API call\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for one complete public one-shot API call\",\n";
    std::cout << "      \"crt_export\": \"zero-valued external phase; logical output export is inside the measured one-shot API call\",\n";
  } else if (grouped_dispatch) {
    std::cout << "      \"pack\": \"per-repeat aggregate grouped-dispatch host timing for packing A and B for "
              << task_count
              << " independent resident tasks through one shared plan\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat aggregate grouped-dispatch host timing for "
              << task_count
              << " independent resident rns8_gemm calls through one shared plan and per-task workspace\",\n";
    if (finite_benchmark_semantics(args.semantics)) {
      std::cout << "      \"crt_export\": \"per-repeat aggregate grouped-dispatch host timing for canonical finite-u8 export of every grouped task\",\n";
    } else if (exact_wide_benchmark_semantics(args.semantics)) {
      if (result.grouped_dispatch_batched_export_enabled) {
        std::cout << "      \"crt_export\": \"per-repeat aggregate grouped-dispatch host timing for fixed-width exact-wide limb export kernels into one shared device slab plus one compact D2H copy for every grouped task\",\n";
      } else {
        std::cout << "      \"crt_export\": \"per-repeat aggregate grouped-dispatch host timing for fixed-width exact-wide limb export of every grouped task\",\n";
      }
    } else {
      std::cout << "      \"crt_export\": \"per-repeat aggregate grouped-dispatch host timing for CRT export/reconstruction of every grouped task\",\n";
    }
  } else if (host_batch) {
    std::cout << "      \"pack\": \"per-repeat aggregate host timing for packing A and B for "
              << task_count
              << " independent resident tasks through one shared plan\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat aggregate host timing for "
              << task_count
              << " independent resident rns8_gemm calls through one shared plan and per-task workspace\",\n";
    if (finite_benchmark_semantics(args.semantics)) {
      std::cout << "      \"crt_export\": \"per-repeat aggregate host timing for canonical finite-u8 export of every batch task\",\n";
    } else if (exact_wide_benchmark_semantics(args.semantics)) {
      std::cout << "      \"crt_export\": \"per-repeat aggregate host timing for fixed-width exact-wide limb export of every batch task\",\n";
    } else {
      std::cout << "      \"crt_export\": \"per-repeat aggregate host timing for CRT export/reconstruction of every batch task\",\n";
    }
  } else if (bounded_residue_channel_fusion_requested(args)) {
    std::cout << "      \"pack\": \"per-repeat host timing for copying uniform-small A and B into benchmark-owned native int8 HIP buffers for the experimental width-3 residue-channel fusion path\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for the benchmark-only direct-HIP residue-channel fusion comparison kernel group over first-prefix9 width-3 groups\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for CRT export/reconstruction into logical output\",\n";
  } else if (bounded_uniform_small_i8_ab_transient_requested(args)) {
    std::cout << "      \"pack\": \"per-repeat host timing for copying uniform-small A and B into benchmark-owned native int8 HIP buffers\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for the direct-HIP prefix-9 uniform-small int8 A/B colpair GEMM kernel group writing resident RNS residues\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for CRT export/reconstruction into logical output\",\n";
  } else if (all_zero_direct_hip_pack_elided) {
    std::cout << "      \"pack\": \"zero-valued per-repeat phase; exact per-tile input scanning proved every output tile zero, so direct-HIP does not pack or read A/B for this capture\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for direct-HIP all-zero resident RNS output materialization\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for exporting the already-zero direct-HIP output\",\n";
  } else if (vector_to_rns_chain_requested(args)) {
    if (args.reuse_packed_b) {
      std::cout << "      \"pack\": \"per-repeat host timing for copying vector producer A/B into native HIP buffers; the Direct-HIP consumer B input was packed once before warmups\",\n";
    } else {
      std::cout << "      \"pack\": \"per-repeat host timing for copying vector producer A/B into native HIP buffers and packing the second Direct-HIP RNS input\",\n";
    }
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for vector-ALU native GEMM, native-to-RNS materialization into Direct-HIP storage, and Direct-HIP consumer RNS GEMM\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for CRT export/reconstruction of the final Direct-HIP consumer output\",\n";
  } else if (native_to_rns_bridge_requested(args)) {
    std::cout << "      \"pack\": \"per-repeat host timing for packing A and B into direct-HIP matrices with both RNS residues and native device buffers populated\",\n";
    std::cout << "      \"rns_gemm\": \"per-repeat host timing for forced native-to-RNS device conversion of A and B followed by direct-HIP rns8_gemm_rns\",\n";
    std::cout << "      \"crt_export\": \"per-repeat host timing for CRT export/reconstruction into logical output\",\n";
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
  if (hip_graph_replay_requested(args)) {
    std::cout << "      \"end_to_end\": \"same measured duration as rns_gemm for one hipGraphLaunch plus stream synchronization; excludes one-time prepack_setup_us, hip_graph_replay.capture_us, hip_graph_replay.instantiate_us, and the final checksum-only logical export\"\n";
  } else if (chain_residue_output) {
    std::cout << "      \"end_to_end\": \"per-repeat pack plus chained rns_gemm host timing; excludes the final checksum-only logical export";
    if (args.reuse_packed_inputs) {
      std::cout << " and one-time prepack_setup_us";
    }
    std::cout << "\"\n";
  } else if (chain_final_export) {
    std::cout << "      \"end_to_end\": \"per-repeat pack plus " << args.residue_chain_length
              << " chained rns_gemm calls plus final logical export host timing";
    if (args.reuse_packed_inputs) {
      std::cout << "; excludes one-time prepack_setup_us";
    }
    std::cout << "\"\n";
  } else if (args.oneshot) {
    std::cout << "      \"end_to_end\": \"same measured duration as rns_gemm for one complete public one-shot API call\"\n";
  } else if (grouped_dispatch) {
    std::cout << "      \"end_to_end\": \"per-repeat aggregate grouped-dispatch pack plus rns_gemm plus crt_export host timing for "
              << task_count << " independent grouped tasks\"\n";
  } else if (host_batch) {
    std::cout << "      \"end_to_end\": \"per-repeat aggregate pack plus rns_gemm plus crt_export host timing for "
              << task_count << " independent batch tasks\"\n";
  } else if (all_zero_direct_hip_pack_elided) {
    std::cout << "      \"end_to_end\": \"per-repeat direct-HIP all-zero output materialization plus export host timing; pack is intentionally elided by the trusted all-zero schedule\"\n";
  } else if (vector_to_rns_chain_requested(args)) {
    std::cout << "      \"end_to_end\": \"per-repeat vector producer pack";
    if (!args.reuse_packed_b) {
      std::cout << " plus Direct-HIP input pack";
    }
    std::cout << ", vector GEMM, native-to-RNS materialization, Direct-HIP consumer GEMM, and final CRT export host timing";
    if (args.reuse_packed_b) {
      std::cout << "; excludes one-time prepack_setup_us for the Direct-HIP consumer B input";
    }
    std::cout << "\"\n";
  } else if (native_to_rns_bridge_requested(args)) {
    std::cout << "      \"end_to_end\": \"per-repeat pack plus native-to-RNS input conversion plus direct-HIP rns_gemm plus crt_export host timing\"\n";
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
  if (native_to_rns_bridge_requested(args)) {
    std::cout << "      \"native_to_rns_bridge\": {\n";
    std::cout << "        \"timed\": true,\n";
    std::cout << "        \"timing_key\": \"rns_gemm\",\n";
    std::cout << "        \"scope\": \"device_native_to_rns_conversion_inside_rns_gemm\",\n";
    std::cout << "        \"reason\": \"measured as explicit native_i64_to_rns_kernel or native_u64_to_rns_kernel GPU event phases inside rns_gemm\"\n";
    std::cout << "      },\n";
  }
  if (vector_to_rns_chain_requested(args)) {
    std::cout << "      \"vector_to_rns_chain\": {\n";
    std::cout << "        \"timed\": true,\n";
    std::cout << "        \"timing_key\": \"rns_gemm\",\n";
    std::cout << "        \"scope\": \"vector_native_output_to_direct_rns_consumer\",\n";
    std::cout << "        \"reason\": \"measured as vector_alu_i64_kernel or vector_alu_u64_kernel, native_i64_to_rns_kernel or native_u64_to_rns_kernel, and rns_gemm_kernel_group GPU event phases inside rns_gemm\"\n";
    std::cout << "      },\n";
  }
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
  std::cout << "  \"avg_pack_per_task_us\": "
            << (avg_pack_us / task_count_denominator) << ",\n";
  std::cout << "  \"avg_rns_gemm_us\": " << avg_gemm_us << ",\n";
  std::cout << "  \"avg_rns_gemm_per_task_us\": "
            << (avg_gemm_us / task_count_denominator) << ",\n";
  std::cout << "  \"per_modulus_gemm_estimate_applicable\": "
            << (per_modulus_estimate_applicable ? "true" : "false") << ",\n";
  std::cout << "  \"avg_per_modulus_gemm_estimate_us\": " << avg_per_modulus_gemm_estimate_us << ",\n";
  std::cout << "  \"avg_crt_export_us\": " << avg_export_us << ",\n";
  std::cout << "  \"avg_crt_export_per_task_us\": "
            << (avg_export_us / task_count_denominator) << ",\n";
  std::cout << "  \"avg_end_to_end_us\": " << avg_end_to_end_us << ",\n";
  std::cout << "  \"avg_end_to_end_per_task_us\": "
            << (avg_end_to_end_us / task_count_denominator) << ",\n";
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

  if (vector_to_rns_chain_requested(args)) {
    rns8_context_options vector_options{};
    vector_options.struct_size = sizeof(vector_options);
    vector_options.abi_version = RNS8_ABI_VERSION;
    vector_options.requested_backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
    rns8_context* vector_ctx = nullptr;
    rns8_status status = rns8_create_context(args.device_id, &vector_options, &vector_ctx);
    if (status != RNS8_SUCCESS) {
      std::cerr << "rns8_create_context(vector-to-RNS producer): " << rns8_status_string(status) << "\n";
      return 1;
    }

    rns8_context_options direct_options{};
    direct_options.struct_size = sizeof(direct_options);
    direct_options.abi_version = RNS8_ABI_VERSION;
    direct_options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    rns8_context* direct_ctx = nullptr;
    status = rns8_create_context(args.device_id, &direct_options, &direct_ctx);
    if (status != RNS8_SUCCESS) {
      rns8_destroy_context(vector_ctx);
      std::cerr << "rns8_create_context(vector-to-RNS consumer): " << rns8_status_string(status) << "\n";
      return 1;
    }

    rns8_device_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    status = rns8_get_device_info(direct_ctx, &info);
    if (status != RNS8_SUCCESS) {
      rns8_destroy_context(direct_ctx);
      rns8_destroy_context(vector_ctx);
      std::cerr << "rns8_get_device_info(vector-to-RNS consumer): " << rns8_status_string(status) << "\n";
      return 1;
    }

    BenchmarkResult result{};
    switch (args.semantics) {
      case BenchSemantics::BoundedI64:
        result = run_vector_to_rns_chain_i64(vector_ctx, direct_ctx, args, bound);
        break;
      case BenchSemantics::BoundedU64:
        result = run_vector_to_rns_chain_u64(vector_ctx, direct_ctx, args, bound);
        break;
      case BenchSemantics::ExactWideSigned:
      case BenchSemantics::ExactWideUnsigned:
      case BenchSemantics::WrapU64Mod2_64:
      case BenchSemantics::FiniteRingU8:
      case BenchSemantics::FiniteFieldU8:
        rns8_destroy_context(direct_ctx);
        rns8_destroy_context(vector_ctx);
        std::cerr << "--vector-to-rns-chain reached an unsupported semantic mode after argument validation\n";
        return 1;
    }
    rns8_destroy_context(direct_ctx);
    rns8_destroy_context(vector_ctx);
    const uint64_t effective_bound = result.effective_bound_available ? result.effective_bound : bound;
    print_json(args, info, result, effective_bound, cmdline);
    return 0;
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

  rns8::detail::hip_direct_allocation_counters_reset();
  const auto allocation_before = rns8::detail::hip_direct_allocation_counters_snapshot();
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
  result.allocation_before = allocation_before;
  result.allocation_after_repeats = rns8::detail::hip_direct_allocation_counters_snapshot();
  result.allocation_tracking_available = true;
  if (!result.allocation_after_warmups_available) {
    result.allocation_after_warmups = result.allocation_after_repeats;
    result.allocation_after_warmups_available = true;
  }
  rns8_destroy_context(ctx);
  const uint64_t effective_bound = result.effective_bound_available ? result.effective_bound : bound;
  print_json(args, info, result, effective_bound, cmdline);
  return 0;
}
