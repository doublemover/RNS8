#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "backend_hip_direct/hip_backend.hpp"
#include "core/plan_lowering.hpp"
#include "rns8/rns8.h"

namespace rns8::bench {

inline constexpr uint32_t kDefaultExactWideBenchmarkLimbCount = 4;

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
  FiniteBinary,
  FiniteSparse,
  FiniteLowHamming,
  FiniteSmallCentered,
  FiniteFullUniform,
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
  uint32_t cpu_threads = 0;
  uint64_t cpu_parallel_threshold = UINT64_C(1) << 20;
  bool progress = false;
  std::string cpu_reference_mode = "timed-baseline";
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
  bool residue_chain_independent_final_export = false;
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
  bool vector_to_rns_chain_host_repack_control = false;
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
  std::string k_block_policy = "auto";
  std::string resident_redesign_candidate;
  std::vector<std::string> resident_redesign_dimensions;
  std::string release_gate = "none";
  std::string verification_amortization = "none";
  std::string error_detection_policy = "none";
  std::string cpu_small_shape_selector = "none";
  std::string incremental_result_cache = "none";
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
  uint64_t adaptive_grouped_active_entry_count = 0;
  uint32_t adaptive_grouped_active_prefix_count = 0;
  uint64_t adaptive_grouped_independent_launch_count = 0;
  uint64_t adaptive_grouped_aggregate_launch_count = 0;
  bool streaming_overlap_executed = false;
  uint32_t streaming_overlap_stream_count = 0;
  uint32_t streaming_overlap_buffer_count = 0;
  uint32_t streaming_overlap_measured_repeat_count = 0;
  uint64_t streaming_overlap_batch_wall_us = 0;
  uint64_t streaming_overlap_per_repeat_pipeline_us = 0;
  std::string schedule_source = "rns8_get_plan_schedule_info";
  std::string target_id = "cpu";
  rns8_plan_backend_info backend_info{};
  bool backend_info_available = false;
  rns8_plan_packing_info packing_info{};
  bool packing_info_available = false;
  rns8::detail::PlanLoweringDescription lowering_info{};
  bool lowering_info_available = false;
  bool export_plan_info_available = false;
  std::string export_plan_source = "rns8_internal_export_plan";
  std::string export_output_layout = "unknown";
  std::string export_selector_status_policy = "unknown";
  std::string export_d2h_policy = "unknown";
  std::string export_selected_kernel{};
  std::string export_selector_key{};
  std::string export_selector_policy{};
  std::string export_semantic_contract{};
  std::string export_backend{};
  std::string export_target_id{};
  std::string export_prefix_contract{};
  std::string export_signedness{};
  std::string export_final_output_mode{};
  std::string export_cache_visibility{};
  std::string export_stale_entry_reason{};
  std::string export_status_elision_reason{};
  uint32_t export_limb_count = 0;
  bool export_requires_tile_metadata = false;
  bool export_all_zero_tiled_output = false;
  TimingSamples samples{};
  GpuEventSamples gpu_events{};
  rns8::detail::hip_direct_allocation_counters allocation_before{};
  rns8::detail::hip_direct_allocation_counters allocation_after_warmups{};
  rns8::detail::hip_direct_allocation_counters allocation_after_repeats{};
  bool allocation_tracking_available = false;
  bool allocation_after_warmups_available = false;
  bool allocation_after_repeats_available = false;
  bool workspace_arena_info_available = false;
  uint64_t workspace_arena_size_bytes = 0;
  uint64_t workspace_arena_high_water_mark_bytes = 0;
  uint32_t workspace_arena_suballocation_count = 0;
  std::string workspace_arena_target_id{};
  std::string workspace_arena_policy{};
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
  bool incremental_result_cache_info_available = false;
  bool incremental_result_cache_public_contract_available = false;
  bool incremental_result_cache_stale_rejection_covered = false;
  std::string incremental_result_cache_candidate_role = "not_requested";
  std::string incremental_result_cache_status = "not_requested";
  std::string incremental_result_cache_stale_reason{};
  std::string incremental_result_cache_fallback_reason{};
  std::string incremental_result_cache_detail{};
  uint64_t incremental_result_cache_key_fingerprint = 0;
  uint64_t incremental_result_cache_a_matrix_instance_id = 0;
  uint64_t incremental_result_cache_b_matrix_instance_id = 0;
  uint64_t incremental_result_cache_c_matrix_instance_id = 0;
  uint64_t incremental_result_cache_a_source_version = 0;
  uint64_t incremental_result_cache_b_source_version = 0;
  uint64_t incremental_result_cache_c_source_version = 0;
  uint64_t incremental_result_cache_copied_from_cache_bytes = 0;
  uint64_t incremental_result_cache_recomputed_cell_count = 0;
  uint64_t incremental_result_cache_allocation_bytes = 0;
  uint64_t incremental_result_cache_snapshot_device_bytes = 0;
  uint32_t incremental_result_cache_dirty_region_count = 0;
  uint32_t incremental_result_cache_recomputed_region_count = 0;
  uint32_t incremental_result_cache_full_fallback = 0;
  uint32_t incremental_result_cache_last_cache_hit = 0;
  uint32_t incremental_result_cache_last_cache_miss = 0;
  uint32_t incremental_result_cache_last_stale_rejection = 0;
  std::vector<rns8_dirty_region> incremental_result_cache_dirty_regions{};
  uint64_t checksum = 0;
};

}  // namespace rns8::bench
