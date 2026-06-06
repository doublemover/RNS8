missing_event_phase_order = copy.deepcopy(bounded)
del missing_event_phase_order["timing_metadata"]["gpu_event_phase_order"]
expect_invalid(missing_event_phase_order, "gpu_event_phase_order must be an array of strings when events are available")

undeclared_event_phase = copy.deepcopy(bounded)
undeclared_event_phase["gpu_event_timings_us"]["old_event_scope_phase"] = [1.0, 1.0, 1.0]
expect_invalid(undeclared_event_phase, "undeclared phase old_event_scope_phase")

duplicate_event_phase_order = copy.deepcopy(v4_vector_i64)
duplicate_event_phase_order["timing_metadata"]["gpu_event_phase_order"].append("crt_export")
expect_invalid(duplicate_event_phase_order, "gpu_event_phase_order must not contain duplicates")

incomplete_vector_events = copy.deepcopy(v4_vector_i64)
incomplete_vector_events["timing_metadata"]["gpu_event_phase_order"].remove("vector_alu_status_d2h")
del incomplete_vector_events["gpu_event_timings_us"]["vector_alu_status_d2h"]
del incomplete_vector_events["gpu_event_timing_summary_us"]["vector_alu_status_d2h"]
expect_invalid(incomplete_vector_events, "vector-ALU GPU event phase set is incomplete")

stale_deep_scope = copy.deepcopy(v4_ck_adaptive_u64)
stale_deep_scope["timing_metadata"]["gpu_event_timing_source_scope"] = (
    "accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export"
)
expect_invalid(stale_deep_scope, "deep accelerator GPU event labels require")

undeclared_deep_phase = copy.deepcopy(v4_ck_adaptive_u64)
undeclared_deep_phase["timing_metadata"]["gpu_event_phase_order"].insert(6, "ck_prefix_99_fake_kernel")
undeclared_deep_phase["gpu_event_timings_us"]["ck_prefix_99_fake_kernel"] = [1.0, 1.0]
undeclared_deep_phase["gpu_event_timing_summary_us"]["ck_prefix_99_fake_kernel"] = zero_summary()
expect_invalid(undeclared_deep_phase, "deep accelerator GPU event phase set contains undeclared phases")

zero_deep_phase = copy.deepcopy(v4_ck_adaptive_u64)
zero_deep_count = zero_deep_phase["schedule_metadata"]["tile_count"]
zero_deep_phase["schedule_metadata"].update(
    {
        "flags": 1,
        "zero_output_tile_count": zero_deep_count,
        "zero_output_tile_fraction": 1.0,
        "zero_output_selected_residue_planes": zero_deep_count
        * zero_deep_phase["schedule_metadata"]["max_selected_prefix"],
        "zero_output_skip_active": True,
    }
)
insert_at = zero_deep_phase["timing_metadata"]["gpu_event_phase_order"].index("ck_add_centered_kernel") + 1
zero_deep_phase["timing_metadata"]["gpu_event_phase_order"].insert(insert_at, "ck_zero_output_tile_memset")
repeats = zero_deep_phase["repeats"]
zero_deep_phase["gpu_event_timings_us"]["ck_zero_output_tile_memset"] = [0.25] * repeats
zero_deep_phase["gpu_event_timing_summary_us"]["ck_zero_output_tile_memset"] = summary([0.25] * repeats)
validate_capture(zero_deep_phase)

bad_schedule_tile = copy.deepcopy(bounded)
bad_schedule_tile["tile_m"] = 96
bad_schedule_tile["schedule_metadata"]["tile_m"] = 96
expect_invalid(bad_schedule_tile, "tile_m must be a power of two")

bad_schedule_prefix = copy.deepcopy(bounded)
bad_schedule_prefix["schedule_metadata"]["max_selected_prefix"] = bad_schedule_prefix["prefix"]
expect_invalid(bad_schedule_prefix, "adaptive_skip_active must match")

bad_wrap_prefix = copy.deepcopy(wrap64)
bad_wrap_prefix["prefix"] = 9
expect_invalid(bad_wrap_prefix, "wrap64 captures must use prefix=0")

bad_wrap_backend = copy.deepcopy(wrap64)
bad_wrap_backend["backend_selected"] = "cpu-reference"
expect_invalid(bad_wrap_backend, "rocWMMA candidate")

bad_wrap64_hip_phase = copy.deepcopy(v4_wrap64_hip)
bad_wrap64_hip_phase["gpu_event_timing_summary_us"]["wrap64_export_d2h"]["avg"] = 999.0
expect_invalid(bad_wrap64_hip_phase, "gpu_event_timing_summary_us.wrap64_export_d2h.avg")

bad_candidate_schedule_source = copy.deepcopy(v4_wrap64_rocwmma_candidate)
bad_candidate_schedule_source["schedule_metadata"]["source"] = "rns8_get_plan_schedule_info"
expect_invalid(bad_candidate_schedule_source, "rns8_bench_wrap64_rocwmma_candidate_static_schedule")

bad_candidate_scope = copy.deepcopy(v4_wrap64_rocwmma_candidate)
bad_candidate_scope["timing_metadata"]["gpu_event_timing_source_scope"] = (
    "accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export"
)
expect_invalid(bad_candidate_scope, "rocwmma_wrap64_byte_gemm36_candidate_default_stream_operation_groups")

bad_candidate_correctness_flag = copy.deepcopy(v4_wrap64_rocwmma_candidate)
bad_candidate_correctness_flag["backend_metadata"]["correctness_backend"] = True
expect_invalid(bad_candidate_correctness_flag, "correctness_backend=False")

bad_baseline_prereq = copy.deepcopy(v4_adaptive_u64)
bad_baseline_prereq["comparison_baseline"]["required_before_speedup_claim"] = ["same_contract_cpu_reference"]
expect_invalid(bad_baseline_prereq, "same_contract_direct_hip_vector_alu_int64")

bad_speedup_claim = copy.deepcopy(v4_ck_i64)
bad_speedup_claim["comparison_baseline"]["speedup_claimed"] = True
expect_invalid(bad_speedup_claim, "speedup claims require a reviewed same-contract comparison baseline")

legacy_reviewed_speedup = copy.deepcopy(v4_ck_i64)
legacy_reviewed_speedup["comparison_baseline"]["status"] = "reviewed_same_contract_baseline"
legacy_reviewed_speedup["comparison_baseline"]["speedup_claimed"] = True
legacy_reviewed_speedup["comparison_baseline"]["selected_reference"] = "hip-direct"
validate_capture(legacy_reviewed_speedup)

bad_performance_promotion = copy.deepcopy(v4_rocwmma_i64)
bad_performance_promotion["backend_metadata"]["performance_validated"] = True
expect_invalid(
    bad_performance_promotion,
    "performance_validated captures require comparison_baseline.status=reviewed_release_same_contract_baseline",
)

bad_legacy_performance_promotion = copy.deepcopy(v4_rocwmma_i64)
bad_legacy_performance_promotion["comparison_baseline"]["status"] = "reviewed_same_contract_baseline"
bad_legacy_performance_promotion["comparison_baseline"]["selected_reference"] = "hip-direct"
bad_legacy_performance_promotion["backend_metadata"]["performance_validated"] = True
expect_invalid(
    bad_legacy_performance_promotion,
    "performance_validated captures require comparison_baseline.status=reviewed_release_same_contract_baseline",
)

bad_legacy_derived_tops = copy.deepcopy(v4_ck_i64)
bad_legacy_derived_tops["comparison_baseline"]["status"] = "reviewed_same_contract_baseline"
bad_legacy_derived_tops["comparison_baseline"]["selected_reference"] = "hip-direct"
bad_legacy_derived_tops["derived_tops_equivalent"] = 123.0
expect_invalid(
    bad_legacy_derived_tops,
    "derived_tops_equivalent requires a reviewed release same-contract comparison baseline",
)

release_performance_promotion = copy.deepcopy(v4_rocwmma_i64)
release_performance_promotion["comparison_baseline"]["status"] = "reviewed_release_same_contract_baseline"
release_performance_promotion["comparison_baseline"]["speedup_claimed"] = True
release_performance_promotion["comparison_baseline"]["selected_reference"] = "hip-direct"
release_performance_promotion["backend_metadata"]["performance_validated"] = True
release_performance_promotion["derived_tops_equivalent"] = 123.0
validate_capture(release_performance_promotion)

release_performance_capture_without_speedup_claim = copy.deepcopy(v4_rocwmma_i64)
release_performance_capture_without_speedup_claim["backend_requested"] = "auto"
release_performance_capture_without_speedup_claim["comparison_baseline"]["status"] = (
    "reviewed_release_same_contract_baseline"
)
release_performance_capture_without_speedup_claim["comparison_baseline"]["speedup_claimed"] = False
release_performance_capture_without_speedup_claim["comparison_baseline"]["selected_reference"] = None
release_performance_capture_without_speedup_claim["backend_metadata"]["performance_validated"] = True
validate_capture(release_performance_capture_without_speedup_claim)

bad_current_version = copy.deepcopy(v4_adaptive_u64)
bad_current_version["schema_version"] = 3
expect_invalid(bad_current_version, "expected 4")

missing_current_version = copy.deepcopy(v4_adaptive_u64)
del missing_current_version["schema_version"]
expect_invalid(missing_current_version, "missing required field schema_version")

bad_schedule_summary = copy.deepcopy(v4_adaptive_u64)
bad_schedule_summary["raw_timings_us"]["scheduling"] = [6]
expect_invalid(bad_schedule_summary, "timing_summary_us.scheduling.avg")

missing_tile_bound_scan = copy.deepcopy(v4_adaptive_u64)
del missing_tile_bound_scan["raw_timings_us"]["tile_bound_scan"]
expect_invalid(missing_tile_bound_scan, "raw_timings_us.tile_bound_scan must be an array")

bad_tile_bound_scan_summary = copy.deepcopy(v4_adaptive_u64)
bad_tile_bound_scan_summary["raw_timings_us"]["tile_bound_scan"] = [11]
expect_invalid(bad_tile_bound_scan_summary, "timing_summary_us.tile_bound_scan.avg")

bad_tile_bound_scan_availability = copy.deepcopy(v4_adaptive_u64)
del bad_tile_bound_scan_availability["timing_metadata"]["phase_availability"]["tile_bound_scan"]
expect_invalid(
    bad_tile_bound_scan_availability,
    "phase_availability.tile_bound_scan must be an object for per-tile captures",
)

bad_reduction_scope = copy.deepcopy(v4_wrap64_hip)
bad_reduction_scope["timing_metadata"]["phase_availability"]["reduction"]["scope"] = "fused_into_rns_gemm"
expect_invalid(bad_reduction_scope, "phase_availability.reduction.scope")

bad_v4_bound = copy.deepcopy(v4_adaptive_u64)
bad_v4_bound["bound"] = 1
expect_invalid(bad_v4_bound, "per-tile adaptive captures must use bound=0")

bad_v4_tile_count = copy.deepcopy(v4_adaptive_u64)
bad_v4_tile_count["tile_bounds_u64"]["count"] = 3
expect_invalid(bad_v4_tile_count, "tile_bounds_u64.count must match")

zero_skip_schedule = copy.deepcopy(v4_adaptive_u64)
zero_tile_count = zero_skip_schedule["schedule_metadata"]["tile_count"]
zero_skip_schedule["schedule_metadata"].update(
    {
        "flags": 1,
        "zero_output_tile_count": 1,
        "zero_output_tile_fraction": 1.0 / zero_tile_count,
        "zero_output_selected_residue_planes": zero_skip_schedule["schedule_metadata"]["min_selected_prefix"],
        "zero_output_skip_active": True,
    }
)
zero_skip_kernel = "direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3"
stale_zero_skip_kernel = "direct_hip_tiled_active_prefix_rns_gemm_v2"
zero_skip_schedule["selected_kernel"] = zero_skip_kernel
zero_skip_schedule["backend_metadata"]["selected_kernel"] = zero_skip_kernel
zero_skip_schedule["backend_metadata"]["workspace_required_bytes"] = 1040
zero_skip_schedule["backend_metadata"]["autotune_key"] = zero_skip_schedule["backend_metadata"][
    "autotune_key"
].replace(stale_zero_skip_kernel, zero_skip_kernel)
zero_insert_at = zero_skip_schedule["timing_metadata"]["gpu_event_phase_order"].index("rns_gemm_kernel_group") + 1
zero_skip_schedule["timing_metadata"]["gpu_event_phase_order"].insert(
    zero_insert_at, "direct_hip_zero_output_tile_memset"
)
repeats = zero_skip_schedule["repeats"]
zero_skip_schedule["gpu_event_timings_us"]["direct_hip_zero_output_tile_memset"] = [0.25] * repeats
zero_skip_schedule["gpu_event_timing_summary_us"]["direct_hip_zero_output_tile_memset"] = summary([0.25] * repeats)
zero_skip_schedule["gpu_event_timings_us"]["rns_gemm"] = [
    value + 0.25 for value in zero_skip_schedule["gpu_event_timings_us"]["rns_gemm_kernel_group"]
]
zero_skip_schedule["gpu_event_timing_summary_us"]["rns_gemm"] = summary(
    zero_skip_schedule["gpu_event_timings_us"]["rns_gemm"]
)
validate_capture(zero_skip_schedule)

def add_row_col_autotune_fields(capture: dict) -> None:
    schedule = capture["schedule_metadata"]
    capture["backend_metadata"]["autotune_key"] = (
        capture["backend_metadata"]["autotune_key"]
        + f";schedule_flags={schedule['flags']}"
        + f";zero_a_rows={schedule['zero_a_row_proof_count']}"
        + f";zero_b_cols={schedule['zero_b_col_proof_count']}"
        + f";zero_row_col_products={schedule['zero_row_col_product_count']}"
    )

zero_row_col_schedule = copy.deepcopy(v4_adaptive_u64)
zero_row_col_kernel = "direct_hip_tiled_active_prefix_zero_row_col_skip_rns_gemm_v1"
zero_row_col_schedule["schedule_metadata"].update(
    {
        "flags": 2,
        "zero_a_row_proof_count": 1,
        "zero_b_col_proof_count": 1,
        "zero_row_col_product_count": 129,
        "planner_zero_a_row_count": 1,
        "planner_zero_b_col_count": 1,
        "planner_zero_row_col_product_count": 129,
    }
)
zero_row_col_schedule["selected_kernel"] = zero_row_col_kernel
zero_row_col_schedule["backend_metadata"]["selected_kernel"] = zero_row_col_kernel
zero_row_col_schedule["backend_metadata"]["autotune_key"] = zero_row_col_schedule["backend_metadata"][
    "autotune_key"
].replace(stale_zero_skip_kernel, zero_row_col_kernel)
add_row_col_autotune_fields(zero_row_col_schedule)
validate_capture(zero_row_col_schedule)

zero_tile_row_col_schedule = copy.deepcopy(zero_skip_schedule)
zero_tile_row_col_kernel = "direct_hip_tiled_active_prefix_zero_tile_row_col_skip_rns_gemm_v1"
zero_tile_row_col_schedule["schedule_metadata"].update(
    {
        "flags": 3,
        "zero_a_row_proof_count": 1,
        "zero_b_col_proof_count": 1,
        "zero_row_col_product_count": 129,
        "planner_zero_a_row_count": 1,
        "planner_zero_b_col_count": 1,
        "planner_zero_row_col_product_count": 129,
    }
)
zero_tile_row_col_schedule["selected_kernel"] = zero_tile_row_col_kernel
zero_tile_row_col_schedule["backend_metadata"]["selected_kernel"] = zero_tile_row_col_kernel
zero_tile_row_col_schedule["backend_metadata"]["autotune_key"] = zero_tile_row_col_schedule[
    "backend_metadata"
]["autotune_key"].replace(zero_skip_kernel, zero_tile_row_col_kernel)
add_row_col_autotune_fields(zero_tile_row_col_schedule)
validate_capture(zero_tile_row_col_schedule)

all_zero_skip_schedule = copy.deepcopy(zero_skip_schedule)
all_zero_planes = (
    all_zero_skip_schedule["schedule_metadata"]["tile_count"]
    * all_zero_skip_schedule["schedule_metadata"]["min_selected_prefix"]
)
all_zero_skip_schedule["schedule_metadata"].update(
    {
        "zero_output_tile_count": zero_tile_count,
        "zero_output_tile_fraction": 1.0,
        "zero_output_selected_residue_planes": all_zero_planes,
    }
)
all_zero_skip_schedule["backend_metadata"]["workspace_required_bytes"] = 0
all_zero_skip_schedule["raw_timings_us"]["pack"] = [0] * repeats
all_zero_skip_schedule["timing_summary_us"]["pack"] = zero_summary()
all_zero_skip_schedule["avg_pack_us"] = 0.0
for phase in ["pack_h2d", "pack_kernel", "pack"]:
    all_zero_skip_schedule["gpu_event_timings_us"][phase] = [0.0] * repeats
    all_zero_skip_schedule["gpu_event_timing_summary_us"][phase] = zero_summary()
all_zero_skip_schedule["gpu_event_timings_us"]["rns_gemm_kernel_group"] = [0.0] * repeats
all_zero_skip_schedule["gpu_event_timing_summary_us"]["rns_gemm_kernel_group"] = summary([0.0] * repeats)
all_zero_skip_schedule["gpu_event_timings_us"]["rns_gemm"] = [0.25] * repeats
all_zero_skip_schedule["gpu_event_timing_summary_us"]["rns_gemm"] = summary([0.25] * repeats)
all_zero_skip_schedule["gpu_event_timings_us"]["crt_export_status_memset"] = [0.0] * repeats
all_zero_skip_schedule["gpu_event_timing_summary_us"]["crt_export_status_memset"] = summary([0.0] * repeats)
all_zero_skip_schedule["gpu_event_timings_us"]["crt_export_status_d2h"] = [0.0] * repeats
all_zero_skip_schedule["gpu_event_timing_summary_us"]["crt_export_status_d2h"] = summary([0.0] * repeats)
validate_capture(all_zero_skip_schedule)

bad_all_zero_pack_timing = copy.deepcopy(all_zero_skip_schedule)
bad_all_zero_pack_timing["raw_timings_us"]["pack"][0] = 1
bad_all_zero_pack_timing["timing_summary_us"]["pack"] = summary(bad_all_zero_pack_timing["raw_timings_us"]["pack"])
bad_all_zero_pack_timing["avg_pack_us"] = 1.0 / repeats
expect_invalid(
    bad_all_zero_pack_timing,
    "all-zero direct-HIP adaptive captures must report raw_timings_us.pack",
)

bad_all_zero_pack_event = copy.deepcopy(all_zero_skip_schedule)
bad_all_zero_pack_event["gpu_event_timings_us"]["pack_h2d"][0] = 1.0
bad_all_zero_pack_event["gpu_event_timing_summary_us"]["pack_h2d"] = summary(
    bad_all_zero_pack_event["gpu_event_timings_us"]["pack_h2d"]
)
expect_invalid(
    bad_all_zero_pack_event,
    "all-zero direct-HIP adaptive captures must report gpu_event_timings_us.pack_h2d",
)

bad_zero_skip_stale_kernel = copy.deepcopy(zero_skip_schedule)
bad_zero_skip_stale_kernel["selected_kernel"] = stale_zero_skip_kernel
bad_zero_skip_stale_kernel["backend_metadata"]["selected_kernel"] = stale_zero_skip_kernel
bad_zero_skip_stale_kernel["backend_metadata"]["autotune_key"] = bad_zero_skip_stale_kernel["backend_metadata"][
    "autotune_key"
].replace(zero_skip_kernel, stale_zero_skip_kernel)
expect_invalid(bad_zero_skip_stale_kernel, "direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3")

bad_zero_row_col_stale_kernel = copy.deepcopy(zero_row_col_schedule)
bad_zero_row_col_stale_kernel["selected_kernel"] = stale_zero_skip_kernel
bad_zero_row_col_stale_kernel["backend_metadata"]["selected_kernel"] = stale_zero_skip_kernel
bad_zero_row_col_stale_kernel["backend_metadata"]["autotune_key"] = bad_zero_row_col_stale_kernel[
    "backend_metadata"
]["autotune_key"].replace(zero_row_col_kernel, stale_zero_skip_kernel)
expect_invalid(
    bad_zero_row_col_stale_kernel,
    "direct_hip_tiled_active_prefix_zero_row_col_skip_rns_gemm_v1",
)

bad_zero_row_col_product_count = copy.deepcopy(zero_row_col_schedule)
bad_zero_row_col_product_count["schedule_metadata"]["zero_row_col_product_count"] = 128
bad_zero_row_col_product_count["schedule_metadata"]["planner_zero_row_col_product_count"] = 128
bad_zero_row_col_product_count["backend_metadata"]["autotune_key"] = bad_zero_row_col_product_count[
    "backend_metadata"
]["autotune_key"].replace("zero_row_col_products=129", "zero_row_col_products=128")
expect_invalid(bad_zero_row_col_product_count, "zero_row_col_product_count must match")

bad_zero_row_col_planner_mismatch = copy.deepcopy(zero_row_col_schedule)
bad_zero_row_col_planner_mismatch["schedule_metadata"]["planner_zero_a_row_count"] = 2
expect_invalid(bad_zero_row_col_planner_mismatch, "planner_zero_a_row_count must match")

bad_zero_row_col_missing_key = copy.deepcopy(zero_row_col_schedule)
bad_zero_row_col_missing_key["backend_metadata"]["autotune_key"] = bad_zero_row_col_missing_key[
    "backend_metadata"
]["autotune_key"].replace(";zero_row_col_products=129", "")
expect_invalid(bad_zero_row_col_missing_key, "autotune_key must include zero_row_col_products=129")

bad_zero_skip_unknown_flag = copy.deepcopy(zero_skip_schedule)
bad_zero_skip_unknown_flag["schedule_metadata"]["flags"] = 4
expect_invalid(bad_zero_skip_unknown_flag, "unknown tile schedule flags")

bad_zero_skip_missing_flag = copy.deepcopy(zero_skip_schedule)
del bad_zero_skip_missing_flag["schedule_metadata"]["flags"]
expect_invalid(bad_zero_skip_missing_flag, "requires ZERO_OUTPUT schedule flag")

bad_zero_skip_count = copy.deepcopy(zero_skip_schedule)
bad_zero_skip_count["schedule_metadata"]["zero_output_tile_count"] = zero_tile_count + 1
expect_invalid(bad_zero_skip_count, "zero_output_tile_count must be <= tile_count")

bad_zero_skip_fraction = copy.deepcopy(zero_skip_schedule)
bad_zero_skip_fraction["schedule_metadata"]["zero_output_tile_fraction"] = 1.0
expect_invalid(bad_zero_skip_fraction, "zero_output_tile_fraction must match")

bad_v4_skip_flag = copy.deepcopy(v4_adaptive_u64)
bad_v4_skip_flag["schedule_metadata"]["adaptive_skip_active"] = False
expect_invalid(bad_v4_skip_flag, "adaptive_skip_active must match")

bad_v4_per_modulus = copy.deepcopy(v4_adaptive_u64)
bad_v4_per_modulus["per_modulus_gemm_estimate_applicable"] = True
expect_invalid(bad_v4_per_modulus, "fixed-prefix contract")

bad_v4_toolchain = copy.deepcopy(v4_adaptive_u64)
del bad_v4_toolchain["hip_toolchain"]
expect_invalid(bad_v4_toolchain, "missing required field hip_toolchain")

bad_v4_toolchain_enabled = copy.deepcopy(v4_adaptive_u64)
bad_v4_toolchain_enabled["hip_toolchain"]["enabled"] = False
expect_invalid(bad_v4_toolchain_enabled, "HIP backend captures must set hip_toolchain.enabled=true")

bad_v4_scope = copy.deepcopy(v4_adaptive_u64)
bad_v4_scope["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
expect_invalid(bad_v4_scope, "bounded_adaptive")

bad_v4_wrap64_kernel = copy.deepcopy(v4_wrap64_hip)
bad_v4_wrap64_kernel["selected_kernel"] = "direct_hip_wrap64_comba_correctness_v1"
expect_invalid(bad_v4_wrap64_kernel, "byte_gemm36")

stale_v3_wrap64_kernel = copy.deepcopy(v4_wrap64_hip)
stale_v3_wrap64_kernel["selected_kernel"] = "direct_hip_wrap64_byte_gemm36_tiled_2d_v3"
stale_v3_wrap64_kernel["backend_metadata"]["selected_kernel"] = "direct_hip_wrap64_byte_gemm36_tiled_2d_v3"
stale_v3_wrap64_kernel["backend_metadata"]["autotune_key"] = stale_v3_wrap64_kernel["backend_metadata"][
    "autotune_key"
].replace(
    "kernel=direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4",
    "kernel=direct_hip_wrap64_byte_gemm36_tiled_2d_v3",
)
expect_invalid(stale_v3_wrap64_kernel, "direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4")

bad_event_nullability = copy.deepcopy(wrap64)
bad_event_nullability["timing_metadata"]["gpu_event_timing"] = False
bad_event_nullability["timing_metadata"]["gpu_event_timing_source"] = None
bad_event_nullability["timing_metadata"]["gpu_event_timing_source_scope"] = None
bad_event_nullability["gpu_event_timings_us"] = {"pack": [1.0, 2.0]}
bad_event_nullability["gpu_event_timing_summary_us"] = None
expect_invalid(bad_event_nullability, "gpu_event_timings_us must be null")

bad_event_phase_order_nullability = copy.deepcopy(wrap64)
bad_event_phase_order_nullability["timing_metadata"]["gpu_event_timing"] = False
bad_event_phase_order_nullability["timing_metadata"]["gpu_event_timing_source"] = None
bad_event_phase_order_nullability["timing_metadata"]["gpu_event_timing_source_scope"] = None
bad_event_phase_order_nullability["gpu_event_timings_us"] = None
bad_event_phase_order_nullability["gpu_event_timing_summary_us"] = None
expect_invalid(bad_event_phase_order_nullability, "gpu_event_phase_order must be null")
