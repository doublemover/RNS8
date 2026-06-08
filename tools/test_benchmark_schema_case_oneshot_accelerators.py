bad_reused_pack = copy.deepcopy(reused_ck_i64)
bad_reused_pack["raw_timings_us"]["pack"][0] = 1
expect_invalid(bad_reused_pack, "zero-valued repeats")

bad_reused_prepack = copy.deepcopy(reused_ck_i64)
bad_reused_prepack["prepack_setup_us"] = None
expect_invalid(bad_reused_prepack, "prepack_setup_us")

bad_reused_mode = copy.deepcopy(reused_ck_i64)
bad_reused_mode["timing_metadata"]["pack_mode"] = "per_repeat_repack"
expect_invalid(bad_reused_mode, "timing_metadata.pack_mode")

bad_reused_operands = copy.deepcopy(reused_a_ck_i64)
bad_reused_operands["prepack_reuse_operands"] = ["B"]
expect_invalid(bad_reused_operands, "prepack_reuse_operands")

bad_reused_metadata_operands = copy.deepcopy(reused_a_ck_i64)
bad_reused_metadata_operands["timing_metadata"]["prepack_reuse_operands"] = ["A", "B"]
expect_invalid(bad_reused_metadata_operands, "timing_metadata.prepack_reuse_operands")

bad_reused_strategy = copy.deepcopy(reused_ck_i64)
bad_reused_strategy["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
bad_reused_strategy["timing_metadata"]["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
expect_invalid(bad_reused_strategy, "pack_mode=prepacked_reuse_b")

bad_reused_strategy_backend = copy.deepcopy(reused_ck_i64)
bad_reused_strategy_backend["pack_mode"] = "prepacked_reuse_b"
bad_reused_strategy_backend["prepack_reuse_operands"] = ["B"]
bad_reused_strategy_backend["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
bad_reused_strategy_backend["reuse_contract"] = {
    "enabled": True,
    "operand_role": "B",
    "source_version_inputs": "monotonic_source_version_per_repeat_when_packing_runs",
    "setup_scope": "persistent_plan_workspace_prepacked_reuse",
    "setup_cost_us": 123,
    "measured_repeat_count": bad_reused_strategy_backend["repeats"],
    "break_even_repeat_count": None,
    "output_domain": "native_i64_u64_host",
    "next_op": "host_export",
    "target_fingerprint": "gfx1100",
    "backend_fingerprint": "ck",
    "kernel_fingerprint": "ck_kernel",
    "workspace_fingerprint": "1024B:transient",
    "runtime_prepack_cache": {
        "source": "rns8_get_prepack_cache_info",
        "backend": "rocwmma",
        "semantics": "bounded_i64",
        "operand_role": "B",
        "cache_key_valid": True,
        "reusable_prepack_cache_available": True,
        "production_prepack_cache_available": True,
        "hip_device_id": 0,
        "matrix_rows": 64,
        "matrix_cols": 128,
        "k": 64,
        "max_prefix": 9,
        "finite_modulus": 0,
        "source_version": 1,
        "plan_fingerprint": 123,
        "cache_key_hash": 456,
        "device_bytes": 1024,
        "operand_pack_bytes": 1024,
        "matrix_layout_version": "rns_centered_residue_planes_v1",
        "operand_layout_version": "rns_i8_tile_swizzled_b_v1",
        "cache_scope": "runtime_production_b_prepack_cache",
        "cache_key": "prepack-v2;backend=rocwmma;operand=B;source_version=1;hash=456",
        "detail": "synthetic schema fixture",
    },
    "promotion_eligible": False,
    "invalidation_reasons": ["source_version_changed"],
}
bad_reused_strategy_backend["timing_metadata"]["pack_mode"] = "prepacked_reuse_b"
bad_reused_strategy_backend["timing_metadata"]["prepack_reuse_operands"] = ["B"]
bad_reused_strategy_backend["timing_metadata"]["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
expect_invalid(bad_reused_strategy_backend, "backend_selected=rocwmma")

bad_runtime_cache_zero_source = copy.deepcopy(bad_reused_strategy_backend)
bad_runtime_cache_zero_source["reuse_contract"]["runtime_prepack_cache"]["source_version"] = 0
bad_runtime_cache_zero_source["reuse_contract"]["runtime_prepack_cache"]["cache_key"] = (
    "prepack-v2;backend=rocwmma;operand=B;source_version=0;hash=456"
)
expect_invalid(bad_runtime_cache_zero_source, "runtime prepack cache captures must report nonzero source_version")

rocwmma_runtime_b_cache = add_helper_lane_fields(copy.deepcopy(v4_rocwmma_i64))
rocwmma_runtime_b_cache["pack_mode"] = "prepacked_reuse_b"
rocwmma_runtime_b_cache["reuse_packed_inputs"] = True
rocwmma_runtime_b_cache["prepack_reuse_operands"] = ["B"]
rocwmma_runtime_b_cache["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
rocwmma_runtime_b_cache["prepack_setup_us"] = 123
rocwmma_runtime_b_cache["avg_prepack_setup_us"] = 123.0
rocwmma_runtime_b_cache["timing_metadata"]["pack_mode"] = "prepacked_reuse_b"
rocwmma_runtime_b_cache["timing_metadata"]["prepack_reuse_operands"] = ["B"]
rocwmma_runtime_b_cache["timing_metadata"]["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
rocwmma_runtime_b_cache["timing_metadata"]["phase_availability"]["prepack_setup"] = {
    "timed": True,
    "timing_key": "prepack_setup_us",
    "scope": "one_time_before_warmups",
    "reason": "B prepack cache created once before warmups and reused for each measured repeat",
}
rocwmma_runtime_b_cache["reuse_contract"] = {
    "enabled": True,
    "operand_role": "B",
    "source_version_inputs": "runtime_prepack_cache.source_version_and_cache_key",
    "setup_scope": "runtime_prepack_cache",
    "setup_cost_us": 123.0,
    "setup_amortized_us": 61.5,
    "repeat_median_end_to_end_us": rocwmma_runtime_b_cache["timing_summary_us"]["end_to_end"]["median"],
    "setup_inclusive_median_end_to_end_us": rocwmma_runtime_b_cache["timing_summary_us"]["end_to_end"]["median"]
    + 61.5,
    "setup_inclusive_policy": "one_time_setup_amortized_over_measured_repeats",
    "measured_repeat_count": rocwmma_runtime_b_cache["repeats"],
    "break_even_repeat_count": None,
    "output_domain": "native_i64_u64_host",
    "next_op": "host_export",
    "target_fingerprint": "gfx1100",
    "backend_fingerprint": "rocwmma",
    "kernel_fingerprint": rocwmma_runtime_b_cache["selected_kernel"],
    "workspace_fingerprint": "runtime_b_cache_fixture_workspace",
    "production_runtime_prepack_cache_available": True,
    "setup_inclusive_cache_promotion_candidate": True,
    "promotion_eligible": False,
    "invalidation_reasons": ["source_version_changed"],
    "runtime_prepack_cache": {
        "source": "rns8_get_prepack_cache_info",
        "backend": "rocwmma",
        "semantics": "bounded_i64",
        "operand_role": "B",
        "cache_key_valid": True,
        "reusable_prepack_cache_available": True,
        "production_prepack_cache_available": True,
        "hip_device_id": 0,
        "matrix_rows": rocwmma_runtime_b_cache["k"],
        "matrix_cols": rocwmma_runtime_b_cache["n"],
        "k": rocwmma_runtime_b_cache["k"],
        "max_prefix": rocwmma_runtime_b_cache["prefix"],
        "finite_modulus": 0,
        "source_version": 5,
        "plan_fingerprint": 123,
        "cache_key_hash": 456,
        "device_bytes": 2048,
        "operand_pack_bytes": 1024,
        "matrix_layout_version": "rns_centered_residue_planes_v1",
        "operand_layout_version": "rns_i8_tile_swizzled_b_v1",
        "cache_scope": "runtime_production_b_prepack_cache",
        "cache_key": "prepack-v2;backend=rocwmma;operand=B;source_version=5;hash=456",
        "detail": "synthetic schema fixture",
    },
}
validate_capture(rocwmma_runtime_b_cache)

bad_runtime_cache_backend = copy.deepcopy(rocwmma_runtime_b_cache)
bad_runtime_cache_backend["reuse_contract"]["runtime_prepack_cache"]["backend"] = "ck"
expect_invalid(bad_runtime_cache_backend, "runtime prepack cache backend must match backend_selected")

bad_runtime_cache_semantics = copy.deepcopy(rocwmma_runtime_b_cache)
bad_runtime_cache_semantics["reuse_contract"]["runtime_prepack_cache"]["semantics"] = "bounded_u64"
expect_invalid(bad_runtime_cache_semantics, "runtime prepack cache semantics must match capture semantics")

bad_runtime_cache_rows = copy.deepcopy(rocwmma_runtime_b_cache)
bad_runtime_cache_rows["reuse_contract"]["runtime_prepack_cache"]["matrix_rows"] = (
    rocwmma_runtime_b_cache["k"] + 1
)
expect_invalid(bad_runtime_cache_rows, "runtime prepack cache matrix_rows must match capture k for B operand")

bad_runtime_cache_cols = copy.deepcopy(rocwmma_runtime_b_cache)
bad_runtime_cache_cols["reuse_contract"]["runtime_prepack_cache"]["matrix_cols"] = (
    rocwmma_runtime_b_cache["n"] + 1
)
expect_invalid(bad_runtime_cache_cols, "runtime prepack cache matrix_cols must match capture n for B operand")

bad_runtime_cache_prefix = copy.deepcopy(rocwmma_runtime_b_cache)
bad_runtime_cache_prefix["reuse_contract"]["runtime_prepack_cache"]["max_prefix"] = (
    rocwmma_runtime_b_cache["prefix"] - 1
)
expect_invalid(bad_runtime_cache_prefix, "runtime prepack cache max_prefix must match capture prefix")

bad_runtime_cache_key = copy.deepcopy(rocwmma_runtime_b_cache)
bad_runtime_cache_key["reuse_contract"]["runtime_prepack_cache"]["cache_key"] = (
    "prepack-v2;backend=rocwmma;operand=B;hash=456"
)
expect_invalid(bad_runtime_cache_key, "runtime prepack cache cache_key must include source_version")

bad_runtime_cache_hash = copy.deepcopy(rocwmma_runtime_b_cache)
bad_runtime_cache_hash["reuse_contract"]["runtime_prepack_cache"]["cache_key_hash"] = 0
expect_invalid(bad_runtime_cache_hash, "runtime prepack cache cache_key_hash must be nonzero")

bad_runtime_cache_bytes = copy.deepcopy(rocwmma_runtime_b_cache)
bad_runtime_cache_bytes["reuse_contract"]["runtime_prepack_cache"]["operand_pack_bytes"] = (
    rocwmma_runtime_b_cache["reuse_contract"]["runtime_prepack_cache"]["device_bytes"] + 1
)
expect_invalid(bad_runtime_cache_bytes, "runtime prepack cache operand_pack_bytes must not exceed device_bytes")

bad_reused_metadata_strategy = copy.deepcopy(reused_ck_i64)
bad_reused_metadata_strategy["timing_metadata"]["prepack_reuse_strategy"] = "none"
expect_invalid(bad_reused_metadata_strategy, "timing_metadata.prepack_reuse_strategy")

bad_hipblaslt_full_reuse_stale_event = copy.deepcopy(reused_hipblaslt_i64)
bad_hipblaslt_full_reuse_stale_event["gpu_event_timings_us"]["hipblaslt_pack_transpose_centered"] = [
    0.0
] * bad_hipblaslt_full_reuse_stale_event["repeats"]
bad_hipblaslt_full_reuse_stale_event["gpu_event_timing_summary_us"][
    "hipblaslt_pack_transpose_centered"
] = zero_summary()
expect_invalid(
    bad_hipblaslt_full_reuse_stale_event,
    "undeclared phase hipblaslt_pack_transpose_centered",
)

bad_oneshot_pack_timing = copy.deepcopy(direct_hip_oneshot_i64)
bad_oneshot_pack_timing["raw_timings_us"]["pack"][0] = 1
expect_invalid(bad_oneshot_pack_timing, "public one-shot captures must report raw_timings_us.pack")

bad_oneshot_scope = copy.deepcopy(direct_hip_oneshot_i64)
bad_oneshot_scope["timing_metadata"]["gpu_event_timing_source_scope"] = (
    "direct_hip_default_stream_backend_operation_groups"
)
expect_invalid(bad_oneshot_scope, "direct_hip_oneshot_default_stream_operation_groups")

bad_oneshot_event_phase = copy.deepcopy(direct_hip_oneshot_i64)
bad_oneshot_event_phase["timing_metadata"]["gpu_event_phase_order"].remove("oneshot_native_input_h2d")
del bad_oneshot_event_phase["gpu_event_timings_us"]["oneshot_native_input_h2d"]
del bad_oneshot_event_phase["gpu_event_timing_summary_us"]["oneshot_native_input_h2d"]
expect_invalid(bad_oneshot_event_phase, "direct-HIP one-shot GPU event phase set is incomplete")

resident_fallback_oneshot = as_direct_hip_resident_fallback_oneshot_capture(v4_ck_i64)
validate_capture(resident_fallback_oneshot)

bad_resident_fallback_native_metadata = copy.deepcopy(resident_fallback_oneshot)
bad_resident_fallback_native_metadata["backend_metadata"][
    "epilogue_mode"
] = "native_input_centered_residue_then_crt_export"
expect_invalid(
    bad_resident_fallback_native_metadata,
    "backend_metadata.epilogue_mode=fused_centered_residue_then_crt_export",
)

bad_resident_fallback_stale_scope = copy.deepcopy(resident_fallback_oneshot)
bad_resident_fallback_stale_scope["timing_metadata"]["gpu_event_timing"] = True
bad_resident_fallback_stale_scope["timing_metadata"][
    "gpu_event_timing_reason"
] = "captured_by_direct_hip_oneshot_api_hooks"
bad_resident_fallback_stale_scope["timing_metadata"]["gpu_event_timing_status"] = "available"
bad_resident_fallback_stale_scope["timing_metadata"]["gpu_event_timing_source"] = "hipEventElapsedTime"
bad_resident_fallback_stale_scope["timing_metadata"][
    "gpu_event_timing_source_scope"
] = "direct_hip_oneshot_default_stream_operation_groups"
bad_resident_fallback_stale_scope["timing_metadata"]["gpu_event_phase_order"] = [
    "oneshot_native_input_h2d",
    "rns_gemm_kernel_group",
    "rns_gemm",
    "crt_export_status_memset",
    "crt_export_kernel",
    "crt_export_status_d2h",
    "crt_export_d2h",
    "crt_export",
    "oneshot_api_gpu",
]
bad_resident_fallback_stale_scope["gpu_event_timings_us"] = {
    phase: [1.0 for _ in range(bad_resident_fallback_stale_scope["repeats"])]
    for phase in bad_resident_fallback_stale_scope["timing_metadata"]["gpu_event_phase_order"]
}
bad_resident_fallback_stale_scope["gpu_event_timing_summary_us"] = {
    phase: summary(values)
    for phase, values in bad_resident_fallback_stale_scope["gpu_event_timings_us"].items()
}
expect_invalid(
    bad_resident_fallback_stale_scope,
    "direct_hip_oneshot_resident_fallback_default_stream_operation_groups",
)

bad_resident_fallback_missing_pack = copy.deepcopy(resident_fallback_oneshot)
bad_resident_fallback_missing_pack["timing_metadata"]["gpu_event_phase_order"].remove("pack_kernel")
del bad_resident_fallback_missing_pack["gpu_event_timings_us"]["pack_kernel"]
del bad_resident_fallback_missing_pack["gpu_event_timing_summary_us"]["pack_kernel"]
expect_invalid(
    bad_resident_fallback_missing_pack,
    "direct-HIP one-shot resident fallback GPU event phase set is incomplete",
)

small_u64_oneshot = copy.deepcopy(direct_hip_oneshot_i64)
small_u64_oneshot["semantics"] = "bounded_u64"
small_u64_oneshot["bound_kind"] = "global_max_unsigned"
small_u64_oneshot["backend_metadata"]["autotune_key"] = small_u64_oneshot[
    "backend_metadata"
]["autotune_key"].replace("semantics=bounded_i64", "semantics=bounded_u64")
validate_capture(small_u64_oneshot)

large_i64_oneshot = copy.deepcopy(direct_hip_oneshot_i64)
large_i64_oneshot["m"] = 512
large_i64_oneshot["n"] = 512
large_i64_oneshot["k"] = 512
large_oneshot_kernel = "direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2"
large_i64_oneshot["selected_kernel"] = large_oneshot_kernel
large_i64_oneshot["backend_metadata"]["selected_kernel"] = large_oneshot_kernel
apply_int32_accumulator_contract(large_i64_oneshot)
large_i64_oneshot["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
    large_i64_oneshot["backend_metadata"]["autotune_key"].replace(
        "m=64;n=128;k=64",
        "m=512;n=512;k=512",
    ).replace(
        "direct_hip_prefix9_native_input_grouped_rns_gemm_v1",
        large_oneshot_kernel,
    ),
    large_i64_oneshot,
)
validate_capture(large_i64_oneshot)

bad_i64_oneshot_stale_kernel = copy.deepcopy(large_i64_oneshot)
old_oneshot_kernel = "direct_hip_prefix9_native_input_grouped_rns_gemm_v1"
bad_i64_oneshot_stale_kernel["selected_kernel"] = old_oneshot_kernel
bad_i64_oneshot_stale_kernel["backend_metadata"]["selected_kernel"] = old_oneshot_kernel
bad_i64_oneshot_stale_kernel["backend_metadata"]["autotune_key"] = bad_i64_oneshot_stale_kernel[
    "backend_metadata"
]["autotune_key"].replace(
    large_oneshot_kernel,
    old_oneshot_kernel,
)
expect_invalid(bad_i64_oneshot_stale_kernel, "direct-HIP one-shot bounded captures must use selected_kernel")

large_u64_oneshot = copy.deepcopy(small_u64_oneshot)
large_u64_oneshot["m"] = 512
large_u64_oneshot["n"] = 512
large_u64_oneshot["k"] = 512
large_u64_oneshot["selected_kernel"] = large_oneshot_kernel
large_u64_oneshot["backend_metadata"]["selected_kernel"] = large_oneshot_kernel
apply_int32_accumulator_contract(large_u64_oneshot)
large_u64_key = large_u64_oneshot["backend_metadata"]["autotune_key"]
large_u64_oneshot["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
    large_u64_key.replace(
        "m=64;n=128;k=64",
        "m=512;n=512;k=512",
    ).replace(
        "direct_hip_prefix9_native_input_grouped_rns_gemm_v1",
        large_oneshot_kernel,
    ),
    large_u64_oneshot,
)
validate_capture(large_u64_oneshot)

bad_oneshot_stale_kernel = copy.deepcopy(large_u64_oneshot)
bad_oneshot_stale_kernel["selected_kernel"] = old_oneshot_kernel
bad_oneshot_stale_kernel["backend_metadata"]["selected_kernel"] = old_oneshot_kernel
bad_oneshot_stale_kernel["backend_metadata"]["autotune_key"] = bad_oneshot_stale_kernel[
    "backend_metadata"
]["autotune_key"].replace(
    large_oneshot_kernel,
    old_oneshot_kernel,
)
expect_invalid(bad_oneshot_stale_kernel, "direct-HIP one-shot bounded captures must use selected_kernel")

bad_finite_oneshot_pack_timing = copy.deepcopy(direct_hip_finite_oneshot)
bad_finite_oneshot_pack_timing["raw_timings_us"]["pack"][0] = 1
expect_invalid(bad_finite_oneshot_pack_timing, "public one-shot captures must report raw_timings_us.pack")

bad_finite_oneshot_stale_pack_event = copy.deepcopy(direct_hip_finite_oneshot)
bad_finite_oneshot_stale_pack_event["timing_metadata"]["gpu_event_phase_order"].insert(1, "finite_pack_kernel")
bad_finite_oneshot_stale_pack_event["gpu_event_timings_us"]["finite_pack_kernel"] = [1.0, 1.0]
bad_finite_oneshot_stale_pack_event["gpu_event_timing_summary_us"]["finite_pack_kernel"] = {
    "avg": 1.0,
    "median": 1.0,
    "p95": 1.0,
}
expect_invalid(bad_finite_oneshot_stale_pack_event, "direct-HIP finite one-shot GPU event phase set contains undeclared phases")

bad_repack_prepack = copy.deepcopy(v4_ck_i64)
bad_repack_prepack["reuse_packed_inputs"] = False
bad_repack_prepack["pack_mode"] = "per_repeat_repack"
bad_repack_prepack["prepack_setup_us"] = 1
bad_repack_prepack["avg_prepack_setup_us"] = 1.0
bad_repack_prepack["timing_metadata"]["pack_mode"] = "per_repeat_repack"
expect_invalid(bad_repack_prepack, "prepack_setup_us=null")

bad_repack_strategy = copy.deepcopy(v4_ck_i64)
bad_repack_strategy["prepack_reuse_strategy"] = "persistent_matrix_residency"
bad_repack_strategy["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
expect_invalid(bad_repack_strategy, "prepack_reuse_strategy=none")

bad_length = copy.deepcopy(bounded)
bad_length["raw_timings_us"]["pack"].pop()
expect_invalid(bad_length, "raw_timings_us.pack length")

bad_summary = copy.deepcopy(bounded)
bad_summary["gpu_event_timing_summary_us"]["crt_export"]["avg"] = 999.0
expect_invalid(bad_summary, "gpu_event_timing_summary_us.crt_export.avg")

bad_event_source = copy.deepcopy(bounded)
bad_event_source["timing_metadata"]["gpu_event_timing_source"] = "std::chrono::steady_clock"
expect_invalid(bad_event_source, "gpu_event_timing_source must be hipEventElapsedTime")

bad_event_scope = copy.deepcopy(bounded)
bad_event_scope["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_unknown_scope"
expect_invalid(bad_event_scope, "known direct-HIP scope")

bad_hipblaslt_scope = copy.deepcopy(v4_hipblaslt_i64)
bad_hipblaslt_scope["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
expect_invalid(bad_hipblaslt_scope, "known hipBLASLt scope")

bad_ck_library = copy.deepcopy(v4_ck_i64)
bad_ck_library["backend_metadata"]["accelerator_library"] = "HIP runtime"
expect_invalid(bad_ck_library, "Composable Kernel")

bad_ck_kernel = copy.deepcopy(v4_ck_adaptive_u64)
bad_ck_kernel["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
bad_ck_kernel["backend_metadata"]["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
expect_invalid(bad_ck_kernel, "per-tile adaptive ck captures")

bad_ck_events = copy.deepcopy(v4_ck_adaptive_u64)
bad_ck_events["timing_metadata"]["gpu_event_timing_source_scope"] = "ck_default_stream"
expect_invalid(bad_ck_events, "accelerator_backend_default_stream_deep_kernel_events")

bad_rocwmma_library = copy.deepcopy(v4_rocwmma_i64)
bad_rocwmma_library["backend_metadata"]["accelerator_library"] = "HIP runtime"
expect_invalid(bad_rocwmma_library, "rocWMMA")

stale_rocwmma_rns_kernel = copy.deepcopy(v4_rocwmma_i64)
stale_rocwmma_rns_kernel["selected_kernel"] = "rocwmma_i8_i32_signed_hot_residue_v1"
stale_rocwmma_rns_kernel["backend_metadata"]["selected_kernel"] = "rocwmma_i8_i32_signed_hot_residue_v1"
stale_rocwmma_rns_kernel["backend_metadata"]["autotune_key"] = stale_rocwmma_rns_kernel["backend_metadata"][
    "autotune_key"
].replace(
    "kernel=rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2",
    "kernel=rocwmma_i8_i32_signed_hot_residue_v1",
)
expect_invalid(stale_rocwmma_rns_kernel, "rocWMMA captures must report a known rocWMMA selected_kernel")

stale_rocwmma_tiled_rns_kernel = copy.deepcopy(v4_rocwmma_adaptive_u64)
stale_rocwmma_tiled_rns_kernel["selected_kernel"] = "rocwmma_i8_i32_signed_tiled_hot_residue_v1"
stale_rocwmma_tiled_rns_kernel["backend_metadata"][
    "selected_kernel"
] = "rocwmma_i8_i32_signed_tiled_hot_residue_v1"
stale_rocwmma_tiled_rns_kernel["backend_metadata"][
    "autotune_key"
] = stale_rocwmma_tiled_rns_kernel["backend_metadata"]["autotune_key"].replace(
    "kernel=rocwmma_i8_i32_signed_tiled_mod251_255_256_hot_residue_v2",
    "kernel=rocwmma_i8_i32_signed_tiled_hot_residue_v1",
)
expect_invalid(stale_rocwmma_tiled_rns_kernel, "rocWMMA captures must report a known rocWMMA selected_kernel")

bad_rocwmma_kernel = copy.deepcopy(v4_rocwmma_adaptive_u64)
bad_rocwmma_kernel["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
bad_rocwmma_kernel["backend_metadata"]["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
expect_invalid(bad_rocwmma_kernel, "per-tile adaptive rocwmma captures")

bad_rocwmma_events = copy.deepcopy(v4_rocwmma_adaptive_u64)
bad_rocwmma_events["timing_metadata"]["gpu_event_timing_source_scope"] = "rocwmma_default_stream"
expect_invalid(bad_rocwmma_events, "accelerator_backend_default_stream_deep_kernel_events")

amdgpu_builtins = copy.deepcopy(v4_ck_i64)
amdgpu_kernel = "amdgpu_builtin_rdna3_wmma_i32_16x16x16_iu8_centered_epilogue_v1"
amdgpu_builtins["backend_requested"] = "amdgpu-builtins"
amdgpu_builtins["backend_selected"] = "amdgpu-builtins"
amdgpu_builtins["selected_kernel"] = amdgpu_kernel
amdgpu_metadata = amdgpu_builtins["backend_metadata"]
amdgpu_metadata["source"] = "rns8_get_plan_backend_info"
amdgpu_metadata["selected_kernel"] = amdgpu_kernel
amdgpu_metadata["accelerator_backend"] = True
amdgpu_metadata["correctness_backend"] = True
amdgpu_metadata["matrix_engine_backend"] = True
amdgpu_metadata["compiled_kernel_available"] = True
amdgpu_metadata["exact_differential_validated"] = True
amdgpu_metadata["performance_validated"] = False
amdgpu_metadata["accelerator_library"] = "AMDGPU builtins"
amdgpu_metadata["accelerator_version"] = "compiled_target_specific"
amdgpu_metadata["capability_status"] = "implemented_opt_in_amdgpu_builtin_backend"
amdgpu_metadata["epilogue_mode"] = "amdgpu_builtin_fused_i32_to_centered_residue_then_chained_crt_export"
amdgpu_metadata[
    "workspace_mode"
] = "resident_device_buffers_direct_amdgpu_builtin_matrix_core_no_dense_pack_workspace"
amdgpu_metadata["workspace_required_bytes"] = 0
amdgpu_metadata["isa_evidence"] = "amdgpu_builtin_matrix_isa_gate_no_divide"
amdgpu_metadata["matrix_instruction_family"] = "wmma"
amdgpu_metadata["matrix_instruction_shape"] = "16x16x16"
amdgpu_metadata["matrix_instruction_dtype"] = "iu8"
amdgpu_metadata["matrix_instruction_sparsity"] = "dense"
apply_int32_accumulator_contract(amdgpu_builtins)
amdgpu_metadata["autotune_key"] = with_accumulator_key_fields(
    amdgpu_metadata["autotune_key"]
    .replace("backend=ck", "backend=amdgpu-builtins")
    .replace("kernel=ck_wmma_cshuffle_i8_i32_default_moduli_static_centered_epilogue_v3", f"kernel={amdgpu_kernel}")
    .replace(
        "epilogue=ck_fused_centered_residue",
        "epilogue=amdgpu_builtin_fused_i32_to_centered_residue_then_chained_crt_export",
    ),
    amdgpu_builtins,
)
validate_capture(amdgpu_builtins)

bad_amdgpu_family = copy.deepcopy(amdgpu_builtins)
bad_amdgpu_family["backend_metadata"]["matrix_instruction_family"] = "mfma"
expect_invalid(bad_amdgpu_family, "backend_metadata.matrix_instruction_family=wmma")

bad_amdgpu_sparsity = copy.deepcopy(amdgpu_builtins)
bad_amdgpu_sparsity["backend_metadata"]["matrix_instruction_sparsity"] = "structured_4_2"
expect_invalid(bad_amdgpu_sparsity, "backend_metadata.matrix_instruction_sparsity=dense")

research_amdgpu_kernel = copy.deepcopy(amdgpu_builtins)
research_kernel = "amdgpu_builtin_rdna3_wmma_i32_16x16x16_iu4_research_v1"
research_amdgpu_kernel["selected_kernel"] = research_kernel
research_amdgpu_kernel["backend_metadata"]["selected_kernel"] = research_kernel
research_amdgpu_kernel["backend_metadata"]["matrix_instruction_dtype"] = "iu4"
research_amdgpu_kernel["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
    research_amdgpu_kernel["backend_metadata"]["autotune_key"].replace(
        f"kernel={amdgpu_kernel}",
        f"kernel={research_kernel}",
    ),
    research_amdgpu_kernel,
)
expect_invalid(research_amdgpu_kernel, "research-only amdgpu-builtins kernels cannot be reported")

bad_hip_target = copy.deepcopy(v4_ck_i64)
bad_hip_target["device"]["gcn_arch"] = "unknown"
expect_invalid(bad_hip_target, "HIP backend captures must include non-placeholder device.gcn_arch")

missing_target_key = copy.deepcopy(v4_ck_i64)
missing_target_key["backend_metadata"]["autotune_key"] = missing_target_key["backend_metadata"][
    "autotune_key"
].replace(";target_id=gfx1100", "")
expect_invalid(missing_target_key, "backend_metadata.autotune_key must include target_id=gfx1100")

wrong_target_key = copy.deepcopy(v4_ck_i64)
wrong_target_key["backend_metadata"]["autotune_key"] = wrong_target_key["backend_metadata"][
    "autotune_key"
].replace(";target_id=gfx1100;", ";target_id=gfx1101;")
expect_invalid(wrong_target_key, "backend_metadata.autotune_key must include target_id=gfx1100")

bad_hip_available = copy.deepcopy(v4_rocwmma_i64)
bad_hip_available["device"]["hip_available"] = 0
expect_invalid(bad_hip_available, "HIP backend captures must use device.hip_available=1")

multi_gpu_metadata = copy.deepcopy(v4_ck_i64)
multi_gpu_metadata["runtime_environment"] = {
    "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "ROCR_VISIBLE_DEVICES": None,
    "GPU_DEVICE_ORDINAL": None,
    "ROCM_PATH": "/opt/rocm",
    "HIP_PATH": "/opt/rocm",
    "LD_LIBRARY_PATH": "/opt/rocm/lib",
}
multi_gpu_metadata["device"]["device_index"] = multi_gpu_metadata["device"]["device_id"]
multi_gpu_metadata["device"]["device_name"] = multi_gpu_metadata["device"]["name"]
multi_gpu_metadata["device"]["target_arch"] = multi_gpu_metadata["device"]["gcn_arch"]
multi_gpu_metadata["device"]["visible_device_count"] = 8
multi_gpu_metadata["device"]["node_gpu_count"] = 8
add_target_variant_fields(multi_gpu_metadata)
validate_capture(multi_gpu_metadata)

bad_device_alias = copy.deepcopy(multi_gpu_metadata)
bad_device_alias["device"]["target_arch"] = "gfx942"
expect_invalid(bad_device_alias, "device.target_arch must match device.gcn_arch")

bad_runtime_environment = copy.deepcopy(multi_gpu_metadata)
bad_runtime_environment["runtime_environment"]["HIP_VISIBLE_DEVICES"] = 0
expect_invalid(bad_runtime_environment, "runtime_environment.HIP_VISIBLE_DEVICES must be a string or null")

bad_vector_source = copy.deepcopy(v4_vector_i64)
bad_vector_source["backend_metadata"]["source"] = "rns8_get_plan_backend_info"
expect_invalid(bad_vector_source, "rns8_bench_vector_alu_baseline")

bad_vector_accelerator = copy.deepcopy(v4_vector_u64)
bad_vector_accelerator["backend_metadata"]["accelerator_backend"] = True
expect_invalid(bad_vector_accelerator, "accelerator_backend")

bad_vector_epilogue = copy.deepcopy(v4_vector_i64)
bad_vector_epilogue["epilogue_type"] = "crt_export"
expect_invalid(bad_vector_epilogue, "direct_int64_export")

bad_vector_prereq = copy.deepcopy(v4_vector_u64)
bad_vector_prereq["comparison_baseline"]["required_before_speedup_claim"] = ["same_contract_cpu_reference"]
expect_invalid(bad_vector_prereq, "same_contract_direct_hip_correctness")

bad_finite_ring_modulus = copy.deepcopy(v4_finite_ring_ck)
bad_finite_ring_modulus["finite_modulus"] = 1
expect_invalid(bad_finite_ring_modulus, "finite_ring_u8 finite_modulus")

bad_finite_field_modulus = copy.deepcopy(v4_finite_field_rocwmma)
bad_finite_field_modulus["finite_modulus"] = 255
expect_invalid(bad_finite_field_modulus, "finite_field_u8 finite_modulus")

bad_finite_prefix = copy.deepcopy(v4_finite_ring_ck)
bad_finite_prefix["prefix"] = 9
expect_invalid(bad_finite_prefix, "finite-u8 captures must use prefix=0")

bad_finite_key = copy.deepcopy(v4_finite_ring_ck)
bad_finite_key["backend_metadata"]["autotune_key"] = bad_finite_key["backend_metadata"]["autotune_key"].replace(
    ";finite_modulus=255",
    "",
)
expect_invalid(bad_finite_key, "finite-u8 backend_metadata.autotune_key must include finite_modulus")

bad_ck_finite_kernel = copy.deepcopy(v4_finite_ring_ck)
bad_ck_finite_kernel["finite_modulus"] = 256
bad_ck_finite_kernel["backend_metadata"]["autotune_key"] = bad_ck_finite_kernel["backend_metadata"][
    "autotune_key"
].replace(
    "finite_modulus=255",
    "finite_modulus=256",
)
expect_invalid(bad_ck_finite_kernel, "CK finite-u8 modulus 256 captures")

unsupported_ck_finite_modulus = copy.deepcopy(v4_finite_ring_ck)
unsupported_ck_finite_modulus["finite_modulus"] = 2
unsupported_ck_finite_modulus["selected_kernel"] = "ck_wmma_cshuffle_finite_u8_static_modulus_centered_epilogue_v2"
unsupported_ck_finite_modulus["backend_metadata"]["selected_kernel"] = (
    "ck_wmma_cshuffle_finite_u8_static_modulus_centered_epilogue_v2"
)
unsupported_ck_finite_modulus["backend_metadata"]["autotune_key"] = unsupported_ck_finite_modulus[
    "backend_metadata"
]["autotune_key"].replace(
    "finite_modulus=255",
    "finite_modulus=2",
).replace(
    "kernel=ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2",
    "kernel=ck_wmma_cshuffle_finite_u8_static_modulus_centered_epilogue_v2",
)
expect_invalid(unsupported_ck_finite_modulus, "unsupported by the static CK reducer set")

bad_rocwmma_finite_kernel = copy.deepcopy(v4_finite_field_rocwmma)
bad_rocwmma_finite_kernel["semantics"] = "finite_ring_u8"
bad_rocwmma_finite_kernel["finite_modulus"] = 256
bad_rocwmma_finite_kernel["backend_metadata"]["autotune_key"] = bad_rocwmma_finite_kernel["backend_metadata"][
    "autotune_key"
].replace(
    "semantics=finite_field_u8",
    "semantics=finite_ring_u8",
).replace(
    "finite_modulus=251",
    "finite_modulus=256",
)
expect_invalid(bad_rocwmma_finite_kernel, "rocWMMA finite-u8 modulus 256 captures")

unsupported_rocwmma_finite_modulus = copy.deepcopy(v4_finite_field_rocwmma)
unsupported_rocwmma_finite_modulus["semantics"] = "finite_ring_u8"
unsupported_rocwmma_finite_modulus["finite_modulus"] = 2
unsupported_rocwmma_finite_modulus["selected_kernel"] = "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1"
unsupported_rocwmma_finite_modulus["backend_metadata"]["semantic_contract"] = "finite_ring_u8"
unsupported_rocwmma_finite_modulus["backend_metadata"]["selected_kernel"] = (
    "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1"
)
unsupported_rocwmma_finite_modulus["backend_metadata"]["autotune_key"] = unsupported_rocwmma_finite_modulus[
    "backend_metadata"
]["autotune_key"].replace(
    "semantics=finite_field_u8",
    "semantics=finite_ring_u8",
).replace(
    "finite_modulus=251",
    "finite_modulus=2",
).replace(
    "kernel=rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2",
    "kernel=rocwmma_i8_i32_signed_finite_u8_hot_residue_v1",
)
expect_invalid(unsupported_rocwmma_finite_modulus, "unsupported by the static rocWMMA reducer set")

bad_finite_epilogue = copy.deepcopy(v4_finite_field_rocwmma)
bad_finite_epilogue["epilogue_type"] = "crt_export"
expect_invalid(bad_finite_epilogue, "canonical_u8_export")

bad_finite_distribution = copy.deepcopy(v4_finite_ring_ck)
bad_finite_distribution["input_distribution"] = "u8_modulus_inferred_from_hot_path"
expect_invalid(bad_finite_distribution, "registered finite input_distribution")

direct_finite_specialized = as_direct_hip_finite_capture(
    v4_finite_ring_ck,
    255,
    "direct_hip_tiled_finite_u8_gemm_mod255_v1",
    "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide",
)
validate_capture(direct_finite_specialized)

bad_direct_finite_specialized_isa = copy.deepcopy(direct_finite_specialized)
bad_direct_finite_specialized_isa["backend_metadata"]["isa_evidence"] = (
    "rns8_hip_direct_reciprocal_isa_gate"
)
expect_invalid(
    bad_direct_finite_specialized_isa,
    "direct-HIP finite-u8 specialized captures",
)

bad_direct_finite_specialized_kernel = copy.deepcopy(direct_finite_specialized)
bad_direct_finite_specialized_kernel["selected_kernel"] = (
    "direct_hip_tiled_finite_u8_gemm_v1"
)
bad_direct_finite_specialized_kernel["backend_metadata"]["selected_kernel"] = (
    "direct_hip_tiled_finite_u8_gemm_v1"
)
expect_invalid(
    bad_direct_finite_specialized_kernel,
    "direct-HIP finite-u8 modulus 255 captures",
)

direct_finite_generic = as_direct_hip_finite_capture(
    v4_finite_ring_ck,
    127,
    "direct_hip_tiled_finite_u8_gemm_v1",
    "rns8_hip_direct_reciprocal_isa_gate",
)
validate_capture(direct_finite_generic)

bad_direct_finite_generic_isa = copy.deepcopy(direct_finite_generic)
bad_direct_finite_generic_isa["backend_metadata"]["isa_evidence"] = (
    "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide"
)
expect_invalid(
    bad_direct_finite_generic_isa,
    "direct-HIP generic finite-u8 captures",
)

