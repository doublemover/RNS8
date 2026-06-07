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
bad_reused_strategy_backend["timing_metadata"]["pack_mode"] = "prepacked_reuse_b"
bad_reused_strategy_backend["timing_metadata"]["prepack_reuse_operands"] = ["B"]
bad_reused_strategy_backend["timing_metadata"]["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
expect_invalid(bad_reused_strategy_backend, "backend_selected=rocwmma")

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

