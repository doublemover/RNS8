#include "rns8_bench_args.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "rns8_bench_modes.hpp"
#include "rns8_bench_support.hpp"

#ifndef RNS8_CONFIGURED_HIP_ENABLED
#  define RNS8_CONFIGURED_HIP_ENABLED 0
#endif

namespace rns8::bench {

namespace {

constexpr uint32_t kWrap64RocwmmaCandidateTile = 16;
constexpr const char* kWrap64RocwmmaCandidateRequestedBackend = "rocwmma-wrap64-candidate";

bool host_api_batch_requested_for_args(const Args& args) {
  return args.host_api_batch_size > 1;
}

bool grouped_dispatch_requested_for_args(const Args& args) {
  return args.grouped_dispatch_tasks > 1;
}

bool grouped_task_executor_requested_for_args(const Args& args) {
  return host_api_batch_requested_for_args(args) || grouped_dispatch_requested_for_args(args);
}

bool finite_input_profile(InputProfile profile) {
  return profile == InputProfile::FiniteBinary || profile == InputProfile::FiniteSparse ||
         profile == InputProfile::FiniteLowHamming || profile == InputProfile::FiniteSmallCentered ||
         profile == InputProfile::FiniteFullUniform;
}

}  // namespace

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

uint64_t parse_u64(const char* text, const char* label) {
  const int64_t value = parse_i64(text, label);
  if (value < 0) {
    usage_error(std::string(label) + " must be non-negative");
  }
  return static_cast<uint64_t>(value);
}

uint64_t parse_u64_seed(const char* text) {
  return parse_u64(text, "--seed");
}

bool valid_tile_size(uint32_t value) {
  return value >= 64 && value <= 512 && (value & (value - 1u)) == 0;
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
  if (value == "amdgpu-builtins" || value == "amdgpu-builtin") {
    args.backend = RNS8_BACKEND_AMDGPU_BUILTINS;
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
  if (value == "finite-binary" || value == "finite_binary" || value == "binary") {
    return InputProfile::FiniteBinary;
  }
  if (value == "finite-sparse" || value == "finite_sparse" || value == "sparse") {
    return InputProfile::FiniteSparse;
  }
  if (value == "finite-low-hamming" || value == "finite_low_hamming" || value == "low-hamming" ||
      value == "low_hamming") {
    return InputProfile::FiniteLowHamming;
  }
  if (value == "finite-small-centered" || value == "finite_small_centered" || value == "small-centered" ||
      value == "small_centered") {
    return InputProfile::FiniteSmallCentered;
  }
  if (value == "finite-full-uniform" || value == "finite_full_uniform" || value == "full-uniform" ||
      value == "full_uniform") {
    return InputProfile::FiniteFullUniform;
  }
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

std::string parse_cpu_reference_mode(const std::string& value) {
  if (value == "timed-baseline" || value == "timed_baseline") {
    return "timed-baseline";
  }
  if (value == "correctness-anchor" || value == "correctness_anchor" || value == "anchor") {
    return "correctness-anchor";
  }
  usage_error("unknown CPU reference mode: " + value);
}

std::vector<std::string> parse_string_list(const std::string& value, const char* label) {
  std::vector<std::string> items;
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t comma = value.find(',', start);
    const std::size_t end = comma == std::string::npos ? value.size() : comma;
    std::string item = trim_ascii_whitespace(value.substr(start, end - start));
    if (!item.empty()) {
      items.push_back(item);
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  if (items.empty()) {
    usage_error(std::string(label) + " must contain at least one nonempty item");
  }
  return items;
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
    } else if (arg == "--cpu-threads" && i + 1 < argc) {
      args.cpu_threads = parse_u32(argv[++i], "--cpu-threads");
    } else if (arg == "--cpu-parallel-threshold" && i + 1 < argc) {
      args.cpu_parallel_threshold = parse_u64(argv[++i], "--cpu-parallel-threshold");
    } else if (arg == "--cpu-reference-mode" && i + 1 < argc) {
      args.cpu_reference_mode = parse_cpu_reference_mode(argv[++i]);
    } else if (arg == "--progress") {
      args.progress = true;
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
    } else if (arg == "--residue-chain-independent-final-export") {
      args.residue_chain_independent_final_export = true;
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
    } else if (arg == "--k-block-policy" && i + 1 < argc) {
      args.k_block_policy = argv[++i];
    } else if (arg == "--resident-redesign-candidate" && i + 1 < argc) {
      args.resident_redesign_candidate = argv[++i];
    } else if (arg == "--resident-redesign-dimensions" && i + 1 < argc) {
      args.resident_redesign_dimensions = parse_string_list(argv[++i], "--resident-redesign-dimensions");
    } else if (arg == "--release-gate" && i + 1 < argc) {
      args.release_gate = argv[++i];
    } else if (arg == "--verification-amortization" && i + 1 < argc) {
      args.verification_amortization = argv[++i];
    } else if (arg == "--error-detection-policy" && i + 1 < argc) {
      args.error_detection_policy = argv[++i];
    } else if (arg == "--cpu-small-shape-selector" && i + 1 < argc) {
      args.cpu_small_shape_selector = argv[++i];
    } else if (arg == "--incremental-result-cache" && i + 1 < argc) {
      args.incremental_result_cache = argv[++i];
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
    } else if (arg == "--vector-to-rns-chain-host-repack-control" ||
               arg == "--vector-native-to-rns-chain-host-repack-control") {
      args.vector_to_rns_chain = true;
      args.vector_to_rns_chain_host_repack_control = true;
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
          << "usage: rns8-bench [--backend auto|cpu|hip-direct|hipblaslt|ck|rocwmma|amdgpu-builtins|wrap64-byte-limb|rocwmma-wrap64-candidate|hip-vector-alu-int64|hip-vector-alu-int64-baseline]\n"
          << "                  [--semantics bounded-i64|bounded-u64|exact-wide-signed|exact-wide-unsigned|wrap-u64|finite-u8-ring|finite-u8-field]\n"
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
          << "                  [--k-block-policy NAME]\n"
          << "                  [--resident-redesign-candidate NAME]\n"
          << "                  [--resident-redesign-dimensions a,b,c]\n"
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
          << "                  [--write-autotune-cache]\n"
          << "                  [--cpu-threads N]\n"
          << "                  [--cpu-parallel-threshold OPS]\n"
          << "                  [--cpu-reference-mode timed-baseline|correctness-anchor]\n"
          << "                  [--progress]\n"
          << "                  [--warmups W] [--repeats R] [--seed S]\n";
      std::exit(0);
    } else {
      usage_error("unknown or incomplete argument: " + arg);
    }
  }

  if (args.m <= 0 || args.n <= 0 || args.k <= 0 || args.repeats == 0) {
    usage_error("matrix dimensions must be positive and repeats must be nonzero");
  }
  if (args.cpu_threads > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    usage_error("--cpu-threads must fit in int for the OpenMP runtime");
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
  if (args.input_profile == InputProfile::AdaptiveBands && !bounded_benchmark_semantics(args.semantics)) {
    usage_error("--input-profile adaptive-bands is only valid for bounded-i64 or bounded-u64 semantics");
  }
  if (finite_input_profile(args.input_profile) && !finite_benchmark_semantics(args.semantics)) {
    usage_error("--input-profile finite-* values are only valid for finite-u8 semantics");
  }
  if (args.input_profile != InputProfile::UniformSmall && args.input_profile != InputProfile::AdaptiveBands &&
      !finite_input_profile(args.input_profile)) {
    usage_error("unsupported input profile");
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
  if (args.residue_chain_final_export && !args.residue_chain_independent_final_export &&
      args.residue_chain_length <= 1) {
    usage_error("--residue-chain-final-export requires --residue-chain-length > 1");
  }
  if (args.residue_chain_independent_final_export && args.residue_chain_length <= 1) {
    usage_error("--residue-chain-independent-final-export requires --residue-chain-length > 1");
  }
  if (args.host_api_batch_size == 0) {
    usage_error("--host-api-batch-size must be positive");
  }
  if (args.grouped_dispatch_tasks == 0) {
    usage_error("--grouped-dispatch must be positive");
  }
  if (host_api_batch_requested_for_args(args) && grouped_dispatch_requested_for_args(args)) {
    usage_error("--host-api-batch-size > 1 and --grouped-dispatch > 1 cannot be combined");
  }
  if (args.modulus_set.empty() ||
      (args.modulus_set != "default" && args.modulus_set.rfind("experimental:", 0) != 0)) {
    usage_error("--modulus-set must be default or experimental:NAME");
  }
  if (args.tile_shape_variant.empty()) {
    usage_error("--tile-shape-variant must not be empty");
  }
  if (args.resident_redesign_candidate.empty() && !args.resident_redesign_dimensions.empty()) {
    usage_error("--resident-redesign-dimensions requires --resident-redesign-candidate");
  }
  if (!args.resident_redesign_candidate.empty() && args.resident_redesign_dimensions.empty()) {
    usage_error("--resident-redesign-candidate requires --resident-redesign-dimensions");
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
  if (args.error_detection_policy.empty()) {
    usage_error("--error-detection-policy must not be empty");
  }
  if (args.cpu_small_shape_selector.empty()) {
    usage_error("--cpu-small-shape-selector must not be empty");
  }
  if (args.incremental_result_cache.empty()) {
    usage_error("--incremental-result-cache must not be empty");
  }
  if (args.k_block_policy.empty()) {
    usage_error("--k-block-policy must not be empty");
  }
  if (host_api_batch_requested_for_args(args)) {
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
  if (grouped_dispatch_requested_for_args(args)) {
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
    const bool full_bounded_pack_export_graph =
        bounded_benchmark_semantics(args.semantics) && args.residue_chain_length == 1 &&
        !args.reuse_packed_inputs && args.next_op_hint != NextOpHint::RnsGemm;
    const bool full_finite_pack_export_graph =
        finite_benchmark_semantics(args.semantics) && args.residue_chain_length == 1 &&
        !args.reuse_packed_inputs && args.next_op_hint != NextOpHint::RnsGemm;
    const bool full_wrap64_pack_export_graph =
        args.semantics == BenchSemantics::WrapU64Mod2_64 && args.residue_chain_length == 1 &&
        !args.reuse_packed_inputs && args.next_op_hint != NextOpHint::RnsGemm;
    const bool resident_chain_graph =
        rns_chain_benchmark_semantics(args.semantics) && args.residue_chain_length > 1 &&
        args.reuse_packed_inputs && args.reuse_packed_a && args.reuse_packed_b &&
        args.next_op_hint == NextOpHint::RnsGemm;
    if (args.residue_chain_final_export || args.residue_chain_independent_final_export) {
      usage_error("--hip-graph-replay cannot be combined with residue-chain final-export modes");
    }
    if (args.backend != RNS8_BACKEND_HIP_DIRECT) {
      usage_error("--hip-graph-replay requires --backend hip-direct");
    }
    if (!resident_chain_graph && !full_bounded_pack_export_graph && !full_finite_pack_export_graph &&
        !full_wrap64_pack_export_graph) {
      usage_error(
          "--hip-graph-replay requires either bounded/finite/wrap64 single-GEMM host-output no-reuse mode or "
          "--reuse-packed-inputs --residue-chain-length > 1 --next-op-hint rns-gemm");
    }
    if (full_wrap64_pack_export_graph && args.output_ld_padding != 0) {
      usage_error("--hip-graph-replay wrap64 full pack/GEMM/export mode currently requires contiguous output");
    }
    if (args.bound_mode != BoundMode::Global || args.bound_source != BoundSource::StaticProfile) {
      usage_error("--hip-graph-replay currently requires global static-profile bounds");
    }
    if (args.oneshot || grouped_task_executor_requested_for_args(args) || args.vector_alu_baseline ||
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
    if (residue_chain_final_export_requested(args) && args.next_op_hint == NextOpHint::RnsGemm) {
      usage_error("residue-chain final-export modes cannot use --next-op-hint rns-gemm");
    }
    if (args.residue_chain_independent_final_export) {
      if (args.reuse_packed_inputs || args.oneshot || grouped_task_executor_requested_for_args(args) ||
          args.transient_uniform_small_inputs || args.native_to_rns_bridge || args.vector_to_rns_chain ||
          args.hip_graph_replay || args.residue_channel_fusion) {
        usage_error(
            "--residue-chain-independent-final-export cannot be combined with reuse, one-shot, grouped/host batch, "
            "transient, bridge, graph, or residue-fusion modes");
      }
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
        args.backend != RNS8_BACKEND_ROCWMMA && args.backend != RNS8_BACKEND_AMDGPU_BUILTINS &&
        args.backend != RNS8_BACKEND_AUTO && args.backend != RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      usage_error(
          "--bound-mode per-tile currently captures CPU, direct HIP, CK, rocWMMA, AMDGPU builtins, or hip-vector-alu-int64 paths");
    }
  }
  if (args.require_adaptive_execution && args.bound_mode != BoundMode::PerTile) {
    usage_error("--require-adaptive-execution requires --bound-mode per-tile");
  }
  if (args.device_id == std::numeric_limits<int>::min()) {
    args.device_id =
        (args.vector_alu_baseline || args.backend == RNS8_BACKEND_HIP_DIRECT || args.backend == RNS8_BACKEND_HIPBLASLT ||
         args.backend == RNS8_BACKEND_CK || args.backend == RNS8_BACKEND_ROCWMMA || args.backend == RNS8_BACKEND_AUTO ||
         args.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64 || args.backend == RNS8_BACKEND_AMDGPU_BUILTINS ||
         args.vector_to_rns_chain)
            ? 0
            : -1;
  }
  return args;
}

}  // namespace rns8::bench
