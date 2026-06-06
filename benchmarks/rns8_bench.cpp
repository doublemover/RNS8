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
#include "core/api_internal.hpp"
#include "core/backend_common.hpp"
#include "core/hip_resources.hpp"
#include "core/internal.hpp"
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

#include "rns8_bench_args.hpp"
#include "rns8_bench_support.hpp"
#include "rns8_bench_modes.hpp"
#include "rns8_bench_types.hpp"

namespace {

using rns8::bench::checked_add_bytes;
using rns8::bench::checked_bytes;
using rns8::bench::checked_elements;
using rns8::bench::checked_limb_elements;
using rns8::bench::Args;
using rns8::bench::BenchmarkResult;
using rns8::bench::BenchSemantics;
using rns8::bench::BoundMode;
using rns8::bench::bound_kind;
using rns8::bench::bound_kind_name;
using rns8::bench::bound_mode_name;
using rns8::bench::bounded_benchmark_semantics;
using rns8::bench::BoundSource;
using rns8::bench::command_line;
using rns8::bench::compiler_id;
using rns8::bench::compiler_version;
using rns8::bench::c_semantics;
using rns8::bench::c_semantics_name;
using rns8::bench::elapsed_us;
using rns8::bench::exact_wide_benchmark_semantics;
using rns8::bench::exact_wide_export_status_check_required;
using rns8::bench::fail_hip_runtime;
using rns8::bench::fail_status;
using rns8::bench::finite_benchmark_semantics;
using rns8::bench::global_bound_kind;
using rns8::bench::GpuEventSamples;
using rns8::bench::InputProfile;
using rns8::bench::json_escape;
using rns8::bench::kDefaultExactWideBenchmarkLimbCount;
using rns8::bench::mix_checksum;
using rns8::bench::NextOpHint;
using rns8::bench::next_op_hint_name;
using rns8::bench::PrepackReuseStrategy;
using rns8::bench::parse_args;
using rns8::bench::PrefixPolicy;
using rns8::bench::print_json_string_or_null;
using rns8::bench::print_nullable_string;
using rns8::bench::residue_chain_final_export_requested;
using rns8::bench::residue_chain_independent_final_export_requested;
using rns8::bench::residue_current_output_mode;
using rns8::bench::rns_chain_benchmark_semantics;
using rns8::bench::rns_residue_chain_requested;
using rns8::bench::runtime_git_commit;
using rns8::bench::semantics_name;
using rns8::bench::TimingSamples;
using rns8::bench::usage_error;
using rns8::bench::valid_finite_modulus;

constexpr uint32_t kBenchmarkSchemaVersion = 4;
constexpr uint32_t kWrap64RocwmmaCandidateTile = 16;
constexpr int64_t kWrap64RocwmmaCandidateMaxK = 32768;
constexpr const char* kWrap64RocwmmaCandidateRequestedBackend = "rocwmma-wrap64-candidate";
constexpr const char* kWrap64RocwmmaCandidateSelectedKernel = "rocwmma_wrap64_byte_gemm36_candidate_v0";
constexpr const char* kWrap64RocwmmaCandidateScheduleSource = "rns8_bench_wrap64_rocwmma_candidate_static_schedule";
constexpr const char* kWrap64RocwmmaCandidateBackendSource = "rns8_bench_wrap64_rocwmma_candidate";
constexpr const char* kWrap64RocwmmaCandidateEventLabel = "wrap64_rocwmma_candidate_gemm36_kernel_group";

template <typename Fn>
rns8_status run_timed_status_operation(const char* label, Fn&& fn);

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
  rns8::detail::hip_unique_stream stream;
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





boost::multiprecision::cpp_int exact_wide_value_from_limbs(
    const uint64_t* limbs,
    uint32_t limb_count,
    bool signed_values) {
  boost::multiprecision::cpp_int value = 0;
  for (uint32_t limb = 0; limb < limb_count; ++limb) {
    value += boost::multiprecision::cpp_int(limbs[limb]) << (64u * limb);
  }
  if (signed_values && limb_count != 0 && (limbs[limb_count - 1] & (UINT64_C(1) << 63u)) != 0) {
    value -= boost::multiprecision::cpp_int(1) << (64u * limb_count);
  }
  return value;
}

std::size_t exact_wide_limb_index(int64_t row, int64_t col, int64_t ld, uint32_t limb_count) {
  return static_cast<std::size_t>(row * ld + col) * static_cast<std::size_t>(limb_count);
}

#include "rns8_bench_core_helpers.inc"
#include "rns8_bench_input_bounds.inc"
#include "rns8_bench_hip_graph_buffers.inc"
#include "rns8_bench_result_metadata.inc"
#include "rns8_bench_backend_path_metadata.inc"
#include "rns8_bench_event_phase_helpers.inc"
#include "rns8_bench_event_collection.inc"
#include "rns8_bench_grouped_host_helpers.inc"
#include "rns8_bench_vector_lanes.inc"
#include "rns8_bench_vector_chain_helpers.inc"
#include "rns8_bench_bounded_lanes.inc"
#include "rns8_bench_exact_wide_lanes.inc"
#include "rns8_bench_finite_lanes.inc"
#include "rns8_bench_wrap64_lanes.inc"
#include "rns8_bench_metadata_names.inc"
#include "rns8_bench_json_sections.inc"
#include "rns8_bench_dispatch_graph_json.inc"
#include "rns8_bench_print_json.inc"
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
