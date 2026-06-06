v4_wrap64_hip = expect_valid("v4_wrap64_hip.json")
v4_adaptive_u64 = expect_valid("v4_bounded_u64_adaptive_hip.json")
v4_adaptive_i64 = expect_valid("v4_bounded_i64_adaptive_hip.json")
v4_hipblaslt_i64 = expect_valid("v4_bounded_i64_hipblaslt.json")
v4_ck_i64 = expect_valid("v4_bounded_i64_ck.json")
v4_ck_adaptive_u64 = expect_valid("v4_bounded_u64_adaptive_ck.json")
v4_rocwmma_i64 = expect_valid("v4_bounded_i64_rocwmma.json")
v4_rocwmma_adaptive_u64 = expect_valid("v4_bounded_u64_adaptive_rocwmma.json")
v4_vector_i64 = expect_valid("v4_bounded_i64_vector_alu.json")
v4_vector_u64 = expect_valid("v4_bounded_u64_vector_alu.json")
v4_finite_ring_ck = expect_valid("v4_finite_ring_u8_ck.json")
v4_finite_field_rocwmma = expect_valid("v4_finite_field_u8_rocwmma.json")
bounded = v4_adaptive_i64

vector_gemv = copy.deepcopy(v4_vector_u64)
vector_gemv["m"] = 128
vector_gemv["n"] = 1
vector_gemv["k"] = 4096
vector_gemv["selected_kernel"] = "hip_vector_alu_u64_gemv_n1_exact_192b_v1"
vector_gemv["backend_metadata"]["selected_kernel"] = "hip_vector_alu_u64_gemv_n1_exact_192b_v1"
vector_gemv["backend_metadata"]["autotune_key"] = (
    vector_gemv["backend_metadata"]["autotune_key"]
    .replace(";m=16;", ";m=128;")
    .replace(";n=16;", ";n=1;")
    .replace(";k=16;", ";k=4096;")
    .replace("k_block_size=16;", "k_block_size=4096;")
    .replace("kernel=hip_vector_alu_u64_exact_192b_v1", "kernel=hip_vector_alu_u64_gemv_n1_exact_192b_v1")
)
vector_gemv["backend_metadata"]["accumulator_safety"]["k_block_size"] = 4096
vector_gemv["k_block_size"] = 4096
validate_capture(vector_gemv)

adaptive_vector_runtime = copy.deepcopy(v4_adaptive_i64)
adaptive_vector_runtime["benchmark"] = "rns8_bounded_gemm_hip_vector_alu_int64_runtime"
adaptive_vector_runtime["benchmark_execution_mode"] = "public_runtime_vector_alu_native_buffers"
adaptive_vector_runtime["backend_requested"] = "hip-vector-alu-int64"
adaptive_vector_runtime["backend_selected"] = "hip-vector-alu-int64"
adaptive_vector_runtime["selected_kernel"] = "hip_vector_alu_i64_exact_192b_v1"
adaptive_vector_runtime["epilogue_type"] = "direct_int64_export"
adaptive_vector_runtime["packed_layout_version"] = "native_i64_rowmajor_v1"
adaptive_vector_runtime["k_block_size"] = adaptive_vector_runtime["k"]
adaptive_vector_runtime["schedule_metadata"]["adaptive_execution_applied"] = False
adaptive_vector_runtime["comparison_baseline"]["required_before_speedup_claim"] = [
    "same_contract_cpu_reference",
    "same_contract_direct_hip_correctness",
]
adaptive_vector_runtime["backend_metadata"] = copy.deepcopy(v4_vector_i64["backend_metadata"])
adaptive_vector_runtime["backend_metadata"]["source"] = "rns8_get_plan_backend_info"
adaptive_vector_runtime["backend_metadata"]["capability_status"] = "implemented_native_bounded_vector_backend"
adaptive_vector_runtime["backend_metadata"]["workspace_mode"] = "native_device_i64_u64_buffers"
adaptive_vector_runtime["backend_metadata"]["workspace_required_bytes"] = 0
adaptive_vector_runtime["backend_metadata"]["accumulator_safety"]["k_block_size"] = adaptive_vector_runtime["k"]
adaptive_vector_runtime["backend_metadata"]["autotune_key"] = (
    "backend=hip-vector-alu-int64;target_id=gfx1100;semantics=bounded_i64;"
    f"m={adaptive_vector_runtime['m']};n={adaptive_vector_runtime['n']};k={adaptive_vector_runtime['k']};"
    "bound_kind=per_tile_max_abs;prefix=9;requested_max_prefix=9;prefix_policy=per_tile_minimum;"
    "tile_m=64;tile_n=64;groups=4;adaptive_prefix=1;adaptive_skip=1;"
    "accumulator_type=software_192bit_limb;accumulator_signedness=signed_i64x_signed_i64;"
    "accumulator_modulus_policy=native_exact_integer_output;"
    f"k_block_size={adaptive_vector_runtime['k']};k_block_cap=0;"
    "kernel=hip_vector_alu_i64_exact_192b_v1;epilogue=direct_int64_export"
)
adaptive_vector_runtime["timing_metadata"]["benchmark_execution_mode"] = (
    "public_runtime_vector_alu_native_buffers"
)
adaptive_vector_runtime["timing_metadata"]["pack_layout"] = "native_i64_row_major"
adaptive_vector_runtime["timing_metadata"]["gpu_event_timing_reason"] = (
    "captured_by_vector_alu_native_backend_hooks"
)
adaptive_vector_runtime["timing_metadata"]["gpu_event_timing_source_scope"] = (
    "vector_alu_default_stream_native_int64_operation_groups"
)
adaptive_vector_runtime["timing_metadata"]["gpu_event_timing_caveat"] = (
    "HIP event timings record benchmark/API vector-ALU native-buffer operation groups; host wall-clock timings "
    "remain required for CPU staging, range checks, API dispatch, allocations, and synchronous host-side "
    "overhead not represented on the HIP stream"
)
vector_event_order = copy.deepcopy(v4_vector_i64["timing_metadata"]["gpu_event_phase_order"])
adaptive_vector_runtime["timing_metadata"]["gpu_event_phase_order"] = vector_event_order
adaptive_vector_runtime["timing_metadata"]["phase_availability"]["reduction"]["scope"] = (
    "not_applicable_native_vector_output"
)
adaptive_vector_runtime["timing_metadata"]["phase_availability"]["reduction"]["reason"] = (
    "runtime vector-ALU computes exact logical outputs directly and does not use centered RNS residue reduction"
)
adaptive_vector_runtime["gpu_event_timings_us"] = {
    phase: [1.0 for _ in range(adaptive_vector_runtime["repeats"])] for phase in vector_event_order
}
adaptive_vector_runtime["gpu_event_timing_summary_us"] = {
    phase: summary(values) for phase, values in adaptive_vector_runtime["gpu_event_timings_us"].items()
}
validate_capture(adaptive_vector_runtime)

stale_adaptive_vector_flag = copy.deepcopy(adaptive_vector_runtime)
stale_adaptive_vector_flag["schedule_metadata"]["adaptive_execution_applied"] = True
expect_invalid(stale_adaptive_vector_flag, "per-tile adaptive vector runtime captures")

stale_vector_gemv_kernel = copy.deepcopy(vector_gemv)
stale_vector_gemv_kernel["selected_kernel"] = "hip_vector_alu_u64_exact_192b_v1"
stale_vector_gemv_kernel["backend_metadata"]["selected_kernel"] = "hip_vector_alu_u64_exact_192b_v1"
stale_vector_gemv_kernel["backend_metadata"]["autotune_key"] = stale_vector_gemv_kernel["backend_metadata"][
    "autotune_key"
].replace("kernel=hip_vector_alu_u64_gemv_n1_exact_192b_v1", "kernel=hip_vector_alu_u64_exact_192b_v1")
expect_invalid(stale_vector_gemv_kernel, "selected_kernel=hip_vector_alu_u64_gemv_n1_exact_192b_v1")

native_to_rns_bridge = as_native_to_rns_bridge_capture(v4_adaptive_i64, "native_i64_to_rns_kernel")
validate_capture(native_to_rns_bridge)

missing_native_to_rns_event = copy.deepcopy(native_to_rns_bridge)
missing_native_to_rns_event["timing_metadata"]["gpu_event_phase_order"].remove("native_i64_to_rns_kernel")
del missing_native_to_rns_event["gpu_event_timings_us"]["native_i64_to_rns_kernel"]
del missing_native_to_rns_event["gpu_event_timing_summary_us"]["native_i64_to_rns_kernel"]
expect_invalid(
    missing_native_to_rns_event,
    "direct-HIP native-to-RNS bridge GPU event phase set is incomplete",
)

stale_native_to_rns_label = copy.deepcopy(native_to_rns_bridge)
phases = stale_native_to_rns_label["timing_metadata"]["gpu_event_phase_order"]
phases[phases.index("native_i64_to_rns_kernel")] = "native_u64_to_rns_kernel"
stale_native_to_rns_label["gpu_event_timings_us"]["native_u64_to_rns_kernel"] = (
    stale_native_to_rns_label["gpu_event_timings_us"].pop("native_i64_to_rns_kernel")
)
stale_native_to_rns_label["gpu_event_timing_summary_us"]["native_u64_to_rns_kernel"] = (
    stale_native_to_rns_label["gpu_event_timing_summary_us"].pop("native_i64_to_rns_kernel")
)
expect_invalid(
    stale_native_to_rns_label,
    "direct-HIP native-to-RNS bridge GPU event phase set is incomplete",
)

stale_native_to_rns_request = copy.deepcopy(native_to_rns_bridge)
stale_native_to_rns_request["backend_requested"] = "hip-direct"
expect_invalid(stale_native_to_rns_request, "native-to-RNS bridge captures must use backend_requested=auto")

stale_native_to_rns_metadata = copy.deepcopy(native_to_rns_bridge)
stale_native_to_rns_metadata["timing_metadata"]["native_to_rns_bridge_forced"] = False
expect_invalid(
    stale_native_to_rns_metadata,
    "native-to-RNS bridge captures must set timing_metadata.native_to_rns_bridge_forced=true",
)

vector_to_rns_chain = as_vector_to_rns_chain_capture(
    v4_adaptive_i64,
    "native_i64_to_rns_kernel",
    "vector_alu_i64_kernel",
)
validate_capture(vector_to_rns_chain)

missing_chain_conversion = copy.deepcopy(vector_to_rns_chain)
missing_chain_conversion["timing_metadata"]["gpu_event_phase_order"].remove("native_i64_to_rns_kernel")
del missing_chain_conversion["gpu_event_timings_us"]["native_i64_to_rns_kernel"]
del missing_chain_conversion["gpu_event_timing_summary_us"]["native_i64_to_rns_kernel"]
expect_invalid(
    missing_chain_conversion,
    "direct-HIP vector-to-RNS chain GPU event phase set is incomplete",
)

missing_chain_vector_kernel = copy.deepcopy(vector_to_rns_chain)
missing_chain_vector_kernel["timing_metadata"]["gpu_event_phase_order"].remove("vector_alu_i64_kernel")
del missing_chain_vector_kernel["gpu_event_timings_us"]["vector_alu_i64_kernel"]
del missing_chain_vector_kernel["gpu_event_timing_summary_us"]["vector_alu_i64_kernel"]
expect_invalid(
    missing_chain_vector_kernel,
    "direct-HIP vector-to-RNS chain GPU event phase set is incomplete",
)

stale_chain_metadata = copy.deepcopy(vector_to_rns_chain)
stale_chain_metadata["timing_metadata"]["vector_to_rns_chain"] = False
expect_invalid(
    stale_chain_metadata,
    "vector-to-RNS chain captures must set timing_metadata.vector_to_rns_chain=true",
)

vector_to_rns_chain_reuse_b = as_vector_to_rns_chain_capture(
    v4_adaptive_i64,
    "native_i64_to_rns_kernel",
    "vector_alu_i64_kernel",
    reuse_consumer_b=True,
)
validate_capture(vector_to_rns_chain_reuse_b)

stale_chain_reuse_strategy = copy.deepcopy(vector_to_rns_chain_reuse_b)
stale_chain_reuse_strategy["prepack_reuse_strategy"] = "none"
expect_invalid(
    stale_chain_reuse_strategy,
    "vector-to-RNS chain captures must use prepack_reuse_strategy=persistent_matrix_residency",
)

stale_chain_reuse_metadata = copy.deepcopy(vector_to_rns_chain_reuse_b)
stale_chain_reuse_metadata["timing_metadata"]["pack_mode"] = "per_repeat_repack"
expect_invalid(
    stale_chain_reuse_metadata,
    "vector-to-RNS chain captures must keep timing_metadata.pack_mode in sync",
)

padded_output = add_output_padding_fields(copy.deepcopy(v4_ck_i64), 7)
validate_capture(padded_output)

helper_lane_ck = add_helper_lane_fields(copy.deepcopy(v4_ck_i64))
validate_capture(helper_lane_ck)

host_api_batch = as_host_api_batch_capture(v4_ck_i64)
validate_capture(host_api_batch)

missing_host_batch = copy.deepcopy(host_api_batch)
del missing_host_batch["host_api_batch"]
expect_invalid(missing_host_batch, "benchmark_host_api_batch captures must include host_api_batch metadata")

bad_host_batch_size = copy.deepcopy(host_api_batch)
bad_host_batch_size["host_api_batch"]["batch_size"] = 1
bad_host_batch_size["timing_metadata"]["host_api_batch_size"] = 1
expect_invalid(bad_host_batch_size, "benchmark_host_api_batch captures must use host_api_batch.batch_size > 1")

stale_host_batch_note = copy.deepcopy(host_api_batch)
stale_host_batch_note["timing_metadata"]["phase_notes"]["end_to_end"] = (
    "per-repeat pack plus rns_gemm plus crt_export host timing"
)
expect_invalid(stale_host_batch_note, "benchmark_host_api_batch phase note end_to_end")

grouped_dispatch = as_grouped_dispatch_capture(v4_ck_i64)
validate_capture(grouped_dispatch)

grouped_device_pack_gemm = copy.deepcopy(grouped_dispatch)
grouped_device_pack_gemm["grouped_dispatch"][
    "execution_strategy"
] = GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS
grouped_device_pack_gemm["timing_metadata"][
    "grouped_dispatch_execution_strategy"
] = GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS
grouped_device_pack_gemm["grouped_dispatch"]["task_descriptor_contract"][
    "device_descriptor_policy"
] = "device_pointer_tables_and_compact_slabs"
validate_capture(grouped_device_pack_gemm)

stale_registry_grouped = as_grouped_dispatch_capture(v4_ck_i64)
stale_registry_grouped["grouped_dispatch"]["execution_strategy"] = "device_grouped_unregistered_strategy"
stale_registry_grouped["timing_metadata"]["grouped_dispatch_execution_strategy"] = "device_grouped_unregistered_strategy"
expect_invalid(stale_registry_grouped, "grouped_dispatch.execution_strategy must be a known grouped strategy")
assert GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS in GROUPED_DISPATCH_EXECUTION_STRATEGIES
assert "device_pointer_tables_and_compact_slabs" in GROUPED_TASK_DEVICE_DESCRIPTOR_POLICIES

stale_grouped_status = copy.deepcopy(grouped_dispatch)
stale_grouped_status["grouped_dispatch"]["capture_status"] = "metadata_only_unsupported_for_execution_path"
stale_grouped_status["grouped_dispatch"][
    "unsupported_reason"
] = "grouped_dispatch_not_executed_by_current_benchmark_path"
expect_invalid(
    stale_grouped_status,
    "benchmark_grouped_dispatch_evidence captures must set grouped_dispatch.capture_status=executed",
)

grouped_host_batch_leak = copy.deepcopy(grouped_dispatch)
grouped_host_batch_leak["host_api_batch"]["enabled"] = True
expect_invalid(
    grouped_host_batch_leak,
    "host_api_batch.enabled must be false for this benchmark_execution_mode",
)

grouped_bad_per_task_average = copy.deepcopy(grouped_dispatch)
grouped_bad_per_task_average["avg_end_to_end_per_task_us"] = grouped_bad_per_task_average["avg_end_to_end_us"]
expect_invalid(
    grouped_bad_per_task_average,
    "avg_end_to_end_per_task_us must equal avg_end_to_end_us / grouped_dispatch.task_count",
)

grouped_missing_strategy = copy.deepcopy(grouped_dispatch)
grouped_missing_strategy["grouped_dispatch"]["execution_strategy"] = "not_requested"
grouped_missing_strategy["timing_metadata"]["grouped_dispatch_execution_strategy"] = "not_requested"
expect_invalid(
    grouped_missing_strategy,
    "benchmark_grouped_dispatch_evidence captures must declare an executed grouped strategy",
)

grouped_missing_descriptor = copy.deepcopy(grouped_dispatch)
del grouped_missing_descriptor["grouped_dispatch"]["task_descriptor_contract"]
expect_invalid(
    grouped_missing_descriptor,
    "benchmark_grouped_dispatch_evidence captures must include task_descriptor_contract",
)

grouped_bad_descriptor_task_count = copy.deepcopy(grouped_dispatch)
grouped_bad_descriptor_task_count["grouped_dispatch"]["task_descriptor_contract"]["task_count"] += 1
expect_invalid(
    grouped_bad_descriptor_task_count,
    "grouped task descriptor task_count must match grouped_dispatch.task_count",
)

grouped_bad_descriptor_policy = copy.deepcopy(grouped_dispatch)
grouped_bad_descriptor_policy["grouped_dispatch"]["task_descriptor_contract"][
    "device_descriptor_policy"
] = "device_pointer_tables_and_compact_slabs"
expect_invalid(
    grouped_bad_descriptor_policy,
    "grouped task descriptor device_descriptor_policy must match execution strategy",
)

grouped_missing_ownership_policy = copy.deepcopy(grouped_dispatch)
del grouped_missing_ownership_policy["grouped_dispatch"]["task_descriptor_contract"]["matrix_ownership_policy"]
expect_invalid(
    grouped_missing_ownership_policy,
    "grouped_dispatch.task_descriptor_contract.matrix_ownership_policy must be known",
)

grouped_bad_reuse_policy = copy.deepcopy(grouped_dispatch)
grouped_bad_reuse_policy["grouped_dispatch"]["task_descriptor_contract"][
    "descriptor_reuse_policy"
] = "not_requested"
expect_invalid(
    grouped_bad_reuse_policy,
    "grouped task descriptor descriptor_reuse_policy must require validated reuse",
)

grouped_bad_stride_policy = copy.deepcopy(grouped_dispatch)
grouped_bad_stride_policy["grouped_dispatch"]["task_descriptor_contract"]["stride_policy"] = "not_requested"
expect_invalid(
    grouped_bad_stride_policy,
    "grouped task descriptor stride_policy must declare explicit matrix/output strides",
)

grouped_bad_currentness_policy = copy.deepcopy(grouped_dispatch)
grouped_bad_currentness_policy["grouped_dispatch"]["task_descriptor_contract"][
    "output_currentness_policy"
] = "not_requested"
expect_invalid(
    grouped_bad_currentness_policy,
    "grouped task descriptor output_currentness_policy must bind device-current outputs",
)

grouped_bad_lifetime_policy = copy.deepcopy(grouped_dispatch)
grouped_bad_lifetime_policy["grouped_dispatch"]["task_descriptor_contract"]["lifetime_policy"] = "not_requested"
expect_invalid(
    grouped_bad_lifetime_policy,
    "grouped task descriptor lifetime_policy must describe capture lifetime",
)

grouped_bad_batched_strategy = copy.deepcopy(grouped_dispatch)
grouped_bad_batched_strategy["grouped_dispatch"]["batched_export_enabled"] = True
grouped_bad_batched_strategy["timing_metadata"]["grouped_dispatch_batched_export_enabled"] = True
expect_invalid(
    grouped_bad_batched_strategy,
    "grouped_dispatch batched export requires a registered batched export strategy",
)

helper_lane_direct = add_helper_lane_fields(copy.deepcopy(v4_adaptive_i64))
add_timing_helper_fields(
    helper_lane_direct,
    reducer="direct_hip_fixed_prefix_9_generated_reducer_v1",
)
validate_capture(helper_lane_direct)

missing_helper_target = copy.deepcopy(helper_lane_ck)
del missing_helper_target["target_variant"]
expect_invalid(missing_helper_target, "HIP helper-lane captures must include target_variant")

bad_helper_target_namespace = copy.deepcopy(helper_lane_ck)
bad_helper_target_namespace["target_variant"]["target_namespace"] = "unknown"
expect_invalid(bad_helper_target_namespace, "concrete target_namespace")

stale_generated_reducer = copy.deepcopy(helper_lane_direct)
stale_generated_reducer["timing_metadata"]["generated_reducer_identity"] = "generic"
expect_invalid(stale_generated_reducer, "declared reducer identity")

bad_selector_reason = copy.deepcopy(helper_lane_ck)
bad_selector_reason["auto_selector"]["rejected_candidates"][0]["reason"] = "unsupported"
expect_invalid(bad_selector_reason, "fixed rejection reason")

bad_allocation_bytes = copy.deepcopy(helper_lane_ck)
bad_allocation_bytes["device_allocation"]["measured_repeat_delta"]["allocated_bytes"] = -1
expect_invalid(bad_allocation_bytes, "device_allocation.measured_repeat_delta.allocated_bytes")

bad_output_policy = copy.deepcopy(helper_lane_ck)
bad_output_policy["output_policy"]["destination_layout"] = "padded_row_major"
expect_invalid(bad_output_policy, "output_policy.destination_layout must match output_ld_padding")

stale_output_ld = copy.deepcopy(padded_output)
stale_output_ld["output_logical_ld"] += 1
expect_invalid(stale_output_ld, "output_logical_ld must equal n + output_ld_padding")

stale_output_layout = copy.deepcopy(padded_output)
stale_output_layout["timing_metadata"]["benchmark_output_destination_layout"] = "contiguous_row_major"
expect_invalid(stale_output_layout, "benchmark_output_destination_layout must be padded_row_major")

stale_output_metadata = copy.deepcopy(padded_output)
stale_output_metadata["timing_metadata"]["benchmark_output_logical_ld"] += 1
expect_invalid(stale_output_metadata, "benchmark_output_logical_ld must match output_logical_ld")

stale_staging_policy = copy.deepcopy(padded_output)
stale_staging_policy["timing_metadata"]["direct_hip_export_staging_policy"] = "large_padded_outputs_always"
expect_invalid(stale_staging_policy, "direct_hip_export_staging_policy")

missing_accumulator_safety = copy.deepcopy(v4_ck_i64)
del missing_accumulator_safety["backend_metadata"]["accumulator_safety"]
expect_invalid(missing_accumulator_safety, "backend_metadata.accumulator_safety must be an object")

stale_ck_rns_kernel = copy.deepcopy(v4_ck_i64)
stale_ck_rns_kernel["selected_kernel"] = "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1"
stale_ck_rns_kernel["backend_metadata"]["selected_kernel"] = "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1"
stale_ck_rns_kernel["backend_metadata"]["autotune_key"] = stale_ck_rns_kernel["backend_metadata"][
    "autotune_key"
].replace(
    "kernel=ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2",
    "kernel=ck_wmma_cshuffle_i8_i32_centered_epilogue_v1",
)
expect_invalid(stale_ck_rns_kernel, "CK captures must report a known CK selected_kernel")

stale_ck_tiled_rns_kernel = copy.deepcopy(v4_ck_adaptive_u64)
stale_ck_tiled_rns_kernel["selected_kernel"] = "ck_wmma_cshuffle_tiled_i8_i32_centered_epilogue_v1"
stale_ck_tiled_rns_kernel["backend_metadata"][
    "selected_kernel"
] = "ck_wmma_cshuffle_tiled_i8_i32_centered_epilogue_v1"
stale_ck_tiled_rns_kernel["backend_metadata"]["autotune_key"] = stale_ck_tiled_rns_kernel["backend_metadata"][
    "autotune_key"
].replace(
    "kernel=ck_wmma_cshuffle_tiled_i8_i32_mod251_255_256_centered_epilogue_v2",
    "kernel=ck_wmma_cshuffle_tiled_i8_i32_centered_epilogue_v1",
)
expect_invalid(stale_ck_tiled_rns_kernel, "CK captures must report a known CK selected_kernel")

stale_ck_accumulator_cap = copy.deepcopy(v4_ck_i64)
stale_ck_accumulator_cap["backend_metadata"]["accumulator_safety"]["k_block_cap"] = 65536
stale_ck_accumulator_cap["backend_metadata"]["autotune_key"] = stale_ck_accumulator_cap["backend_metadata"][
    "autotune_key"
].replace("k_block_cap=32768", "k_block_cap=65536")
expect_invalid(stale_ck_accumulator_cap, "backend_metadata.accumulator_safety.k_block_cap must be 32768")

missing_accumulator_key_field = copy.deepcopy(v4_ck_i64)
missing_accumulator_key_field["backend_metadata"]["autotune_key"] = missing_accumulator_key_field[
    "backend_metadata"
]["autotune_key"].replace("accumulator_type=int32;", "")
expect_invalid(missing_accumulator_key_field, "backend_metadata.autotune_key must include accumulator_type=int32")

unsafe_accumulator_flag = copy.deepcopy(v4_ck_i64)
unsafe_accumulator_flag["backend_metadata"]["accumulator_safety"]["safe_for_k_block"] = False
expect_invalid(unsafe_accumulator_flag, "int32 accumulator captures must set safe_for_k_block=true")

prefixed_adaptive = add_prefix_policy_fields(copy.deepcopy(v4_adaptive_i64), "per_tile_minimum")
validate_capture(prefixed_adaptive)

incomplete_prefix_policy = copy.deepcopy(prefixed_adaptive)
del incomplete_prefix_policy["residue_planes_skipped"]
expect_invalid(incomplete_prefix_policy, "prefix policy metadata fields must be complete")

stale_selected_prefix = copy.deepcopy(prefixed_adaptive)
stale_selected_prefix["selected_prefix"] = stale_selected_prefix["prefix"]
expect_invalid(stale_selected_prefix, "selected_prefix must match")

bad_prefix_skip_fraction = copy.deepcopy(prefixed_adaptive)
bad_prefix_skip_fraction["residue_plane_skip_fraction"] = 0.0
expect_invalid(bad_prefix_skip_fraction, "residue_plane_skip_fraction must match")

bad_prefix_policy_scope = copy.deepcopy(prefixed_adaptive)
bad_prefix_policy_scope["contract_prefix_policy"] = "minimum_proven"
expect_invalid(bad_prefix_policy_scope, "contract_prefix_policy=minimum_proven requires bound_mode=global")

scanned_bound = add_global_bound_scan_fields(copy.deepcopy(v4_hipblaslt_i64))
validate_capture(scanned_bound)

stale_scanned_bound = copy.deepcopy(scanned_bound)
stale_scanned_bound["bound_discovery"]["discovered_global_bound"] += 1
expect_invalid(stale_scanned_bound, "bound_discovery.discovered_global_bound must equal")

incomplete_scanned_timing = copy.deepcopy(scanned_bound)
del incomplete_scanned_timing["timing_metadata"]["phase_availability"]["global_bound_scan"]
expect_invalid(incomplete_scanned_timing, "phase_availability.global_bound_scan must be an object")

per_tile_input_scan = add_per_tile_input_scan_fields(copy.deepcopy(v4_adaptive_i64))
validate_capture(per_tile_input_scan)

per_tile_scan_without_tile_bounds = copy.deepcopy(per_tile_input_scan)
per_tile_scan_without_tile_bounds["tile_bounds_u64"] = None
expect_invalid(per_tile_scan_without_tile_bounds, "input_exact_tile_bounds captures must include tile_bounds_u64")

stale_per_tile_scan_global_field = copy.deepcopy(per_tile_input_scan)
stale_per_tile_scan_global_field["bound_discovery"]["discovered_global_bound"] = 1
expect_invalid(
    stale_per_tile_scan_global_field,
    "input_exact_tile_bounds captures must use bound_discovery.discovered_global_bound=null",
)

stale_per_tile_scan_global_availability = copy.deepcopy(per_tile_input_scan)
stale_per_tile_scan_global_availability["timing_metadata"]["phase_availability"]["global_bound_scan"] = {
    "timed": True,
    "timing_key": "global_bound_scan",
    "scope": "input_row_column_abs_summary",
    "reason": "stale global scan metadata from a non-per-tile input-scan capture",
}
expect_invalid(stale_per_tile_scan_global_availability, "phase_availability.global_bound_scan.timed must be false")
wrap64 = v4_wrap64_hip
large_wrap64_colpair = as_large_wrap64_colpair_capture(v4_wrap64_hip)
validate_capture(large_wrap64_colpair)

default_large_wrap64_kernel = copy.deepcopy(large_wrap64_colpair)
default_large_wrap64_kernel["selected_kernel"] = "direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4"
default_large_wrap64_kernel["backend_metadata"]["selected_kernel"] = (
    "direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4"
)
default_large_wrap64_kernel["backend_metadata"]["autotune_key"] = default_large_wrap64_kernel[
    "backend_metadata"
]["autotune_key"].replace(
    "kernel=direct_hip_wrap64_byte_gemm36_u32acc_colpair_2d_v5",
    "kernel=direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4",
)
default_large_wrap64_kernel["timing_metadata"]["gpu_event_phase_order"] = [
    "wrap64_byte_gemm36_tiled_2d_kernel"
    if item == "wrap64_byte_gemm36_colpair_2d_kernel"
    else item
    for item in default_large_wrap64_kernel["timing_metadata"]["gpu_event_phase_order"]
]
default_large_wrap64_kernel["gpu_event_timings_us"]["wrap64_byte_gemm36_tiled_2d_kernel"] = (
    default_large_wrap64_kernel["gpu_event_timings_us"].pop("wrap64_byte_gemm36_colpair_2d_kernel")
)
default_large_wrap64_kernel["gpu_event_timing_summary_us"]["wrap64_byte_gemm36_tiled_2d_kernel"] = (
    default_large_wrap64_kernel["gpu_event_timing_summary_us"].pop("wrap64_byte_gemm36_colpair_2d_kernel")
)
validate_capture(default_large_wrap64_kernel)

too_small_wrap64_colpair_kernel = copy.deepcopy(large_wrap64_colpair)
too_small_wrap64_colpair_kernel["m"] = 128
too_small_wrap64_colpair_kernel["n"] = 128
too_small_wrap64_colpair_kernel["backend_metadata"]["autotune_key"] = too_small_wrap64_colpair_kernel[
    "backend_metadata"
]["autotune_key"].replace(";m=256;", ";m=128;").replace(";n=256;", ";n=128;")
expect_invalid(too_small_wrap64_colpair_kernel, "direct-HIP wrap64 captures must use selected_kernel")

stale_large_wrap64_colpair_event = copy.deepcopy(large_wrap64_colpair)
stale_large_wrap64_colpair_event["timing_metadata"]["gpu_event_phase_order"] = [
    "wrap64_byte_gemm36_tiled_2d_kernel"
    if item == "wrap64_byte_gemm36_colpair_2d_kernel"
    else item
    for item in stale_large_wrap64_colpair_event["timing_metadata"]["gpu_event_phase_order"]
]
stale_large_wrap64_colpair_event["gpu_event_timings_us"]["wrap64_byte_gemm36_tiled_2d_kernel"] = (
    stale_large_wrap64_colpair_event["gpu_event_timings_us"].pop("wrap64_byte_gemm36_colpair_2d_kernel")
)
stale_large_wrap64_colpair_event["gpu_event_timing_summary_us"]["wrap64_byte_gemm36_tiled_2d_kernel"] = (
    stale_large_wrap64_colpair_event["gpu_event_timing_summary_us"].pop("wrap64_byte_gemm36_colpair_2d_kernel")
)
expect_invalid(stale_large_wrap64_colpair_event, "wrap64_byte_gemm36_colpair_2d_kernel")

v4_wrap64_rocwmma_candidate = as_wrap64_rocwmma_candidate_capture(v4_wrap64_hip)
validate_capture(v4_wrap64_rocwmma_candidate)

stale_rocwmma_backend_spelling = copy.deepcopy(v4_rocwmma_i64)
stale_rocwmma_backend_spelling["backend_requested"] = "wmma"
stale_rocwmma_backend_spelling["backend_selected"] = "wmma"
stale_rocwmma_backend_spelling["backend_metadata"]["autotune_key"] = stale_rocwmma_backend_spelling[
    "backend_metadata"
]["autotune_key"].replace("backend=rocwmma;", "backend=wmma;")
expect_invalid(stale_rocwmma_backend_spelling, "backend_selected must be one of")

stale_candidate_request_spelling = copy.deepcopy(v4_wrap64_rocwmma_candidate)
stale_candidate_request_spelling["backend_requested"] = "wrap64-wmma-candidate"
stale_candidate_request_spelling["backend_selected"] = "wmma"
stale_candidate_request_spelling["backend_metadata"]["autotune_key"] = stale_candidate_request_spelling[
    "backend_metadata"
]["autotune_key"].replace("backend=rocwmma-wrap64-candidate;", "backend=wrap64-wmma-candidate;")
expect_invalid(stale_candidate_request_spelling, "backend_selected must be one of")

v4_cpu_adaptive_i64 = copy.deepcopy(v4_adaptive_i64)
v4_cpu_adaptive_i64["backend_requested"] = "cpu-reference"
v4_cpu_adaptive_i64["backend_selected"] = "cpu-reference"
v4_cpu_adaptive_i64["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
v4_cpu_adaptive_i64["backend_metadata"]["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
v4_cpu_adaptive_i64["backend_metadata"]["accelerator_library"] = None
v4_cpu_adaptive_i64["backend_metadata"]["workspace_mode"] = "host_reference_workspace"
v4_cpu_adaptive_i64["backend_metadata"]["workspace_required_bytes"] = 0
v4_cpu_adaptive_i64["backend_metadata"]["isa_evidence"] = "not_applicable_cpu"
v4_cpu_adaptive_i64["backend_metadata"]["autotune_key"] = (
    "backend=cpu-reference;target_id=cpu;semantics=bounded_i64;m=65;n=65;k=64;prefix=9;tile_m=64;tile_n=64;"
    "groups=4;adaptive_prefix=1;adaptive_skip=1;accumulator_type=int32;"
    "accumulator_signedness=signed_i8x_signed_i8;"
    "accumulator_modulus_policy=selected_rns_modulus_ladder;k_block_size=64;k_block_cap=65536;"
    "kernel=cpu_reference_scalar_rns_gemm_v1;"
    "epilogue=fused_centered_residue_then_crt_export"
)
v4_cpu_adaptive_i64["comparison_baseline"]["required_before_speedup_claim"] = [
    "same_contract_cpu_reference",
    "same_contract_direct_hip_vector_alu_int64",
    "same_contract_direct_hip_correctness",
]
v4_cpu_adaptive_i64["device"] = {
    "device_id": -1,
    "name": "CPU reference",
    "gcn_arch": "none",
    "hip_available": 0,
    "hip_runtime_version": 0,
    "hip_driver_version": 0,
    "global_mem_bytes": 0,
}
v4_cpu_adaptive_i64["timing_note"] = (
    "host wall-clock timings for the CPU adaptive per-tile bounded reference path; no GPU event timing is "
    "requested for this backend"
)
v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing"] = False
v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_reason"] = "not_supported_for_selected_backend"
v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_status"] = "not_requested_for_selected_backend"
v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_source"] = None
v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_source_scope"] = None
v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_caveat"] = None
v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_phase_order"] = None
v4_cpu_adaptive_i64["gpu_event_timings_us"] = None
v4_cpu_adaptive_i64["gpu_event_timing_summary_us"] = None
validate_capture(v4_cpu_adaptive_i64)

