def add_output_padding_fields(capture: dict, padding: int) -> dict:
    output_ld = capture["n"] + padding
    capture["output_logical_ld"] = output_ld
    capture["output_ld_padding"] = padding
    capture["timing_metadata"]["benchmark_output_destination_layout"] = (
        "contiguous_row_major" if padding == 0 else "padded_row_major"
    )
    capture["timing_metadata"]["benchmark_output_logical_ld"] = output_ld
    capture["timing_metadata"]["benchmark_output_ld_padding"] = padding
    capture["timing_metadata"].setdefault("direct_hip_export_staging_policy", "not_applicable")
    return capture


def add_target_variant_fields(capture: dict, target_id: str = "gfx1100") -> dict:
    namespace = "gfx1100" if target_id == "gfx1100" else "unknown"
    device = capture.get("device", {})
    device_name = device.get("name", "")
    hip_toolchain = capture.get("hip_toolchain", {})
    target_cache_key = (
        f"arch={target_id};device_name={device_name};"
        f"hip_sdk_or_rocm={hip_toolchain.get('hip_sdk_or_rocm_version', '')};"
        f"hip_runtime={device.get('hip_runtime_version', 0)}"
    )
    capture["target_variant"] = {
        "target_id": target_id,
        "target_arch": device.get("gcn_arch", target_id),
        "target_cache_key": target_cache_key,
        "target_instance_id": (
            f"{target_cache_key};device_index={device.get('device_id', -1)};"
            "visibility=runtime-default-visible-devices"
        ),
        "target_namespace": namespace,
        "review_group_key": (
            f"{namespace}/target={target_id}/backend={capture['backend_selected']}/"
            f"device_index={device.get('device_id', -1)}/device_name={device.get('name', '')}/"
            f"semantics={capture['semantics']}/configured={capture.get('configured_amdgpu_targets', '')}/runtime="
            f"{device.get('hip_runtime_version', 0)}"
        ),
        "configured_amdgpu_targets": capture.get("configured_amdgpu_targets", ""),
        "device_index": device.get("device_id", -1),
        "device_name": device.get("name", ""),
        "visible_device_count": device.get("visible_device_count", 1 if device.get("hip_available") else 0),
        "node_gpu_count": device.get("node_gpu_count", 1 if device.get("hip_available") else 0),
        "hip_enabled": capture.get("hip_toolchain", {}).get("enabled", False),
        "hip_runtime_version": device.get("hip_runtime_version", 0),
        "hip_driver_version": device.get("hip_driver_version", 0),
    }
    return capture


def add_requested_next_op_fields(
    capture: dict,
    resolved: str = "final-export",
    requested: str = "auto",
    source: str = "benchmark_default",
) -> dict:
    capture["requested_next_op"] = {
        "requested": requested,
        "resolved": resolved,
        "source": source,
        "final_export_available": resolved == "final-export",
        "rns_continuation_available": resolved == "rns-gemm",
        "native_continuation_available": resolved == "native-gemm",
        "native_to_rns_available": resolved == "native-to-rns",
        "reusable_b_prepack_available": resolved == "reuse-b",
    }
    return capture


def add_output_policy_fields(
    capture: dict,
    status_handling: str = "required",
    per_repeat_export: bool = True,
    final_checksum_export: bool = False,
) -> dict:
    padding = int(capture.get("output_ld_padding", 0) or 0)
    logical_ld = int(capture.get("output_logical_ld", capture["n"] + padding))
    capture["output_logical_ld"] = logical_ld
    capture["output_ld_padding"] = padding
    capture["timing_metadata"]["benchmark_output_destination_layout"] = (
        "contiguous_row_major" if padding == 0 else "padded_row_major"
    )
    capture["timing_metadata"]["benchmark_output_logical_ld"] = logical_ld
    capture["timing_metadata"]["benchmark_output_ld_padding"] = padding
    capture["timing_metadata"].setdefault("direct_hip_export_staging_policy", "not_applicable")
    capture["output_policy"] = {
        "destination_layout": "contiguous_row_major" if padding == 0 else "padded_row_major",
        "logical_ld": logical_ld,
        "ld_padding": padding,
        "per_repeat_logical_export": per_repeat_export,
        "final_checksum_export_after_repeats": final_checksum_export,
        "status_handling": status_handling,
        "status_event_policy": (
            "status_memset_and_status_d2h_labels_required_when_gpu_events_available"
            if status_handling == "required"
            else "status_labels_zero_filled_or_absent_because_no_per_repeat_status_export_launches"
            if status_handling == "structurally_elided"
            else "no_range_status_for_semantic"
        ),
    }
    return capture


def add_device_allocation_fields(capture: dict) -> dict:
    zero = {"allocate_calls": 0, "free_calls": 0, "allocated_bytes": 0}
    capture["device_allocation"] = {
        "tracking_available": True,
        "source": "hip_direct_allocation_counters_snapshot",
        "setup_scope": "persistent_plan_workspace_resident_matrices",
        "source_version_inputs": "monotonic_source_version_per_repeat_when_packing_runs",
        "before": dict(zero),
        "after_warmups": {"allocate_calls": 2, "free_calls": 0, "allocated_bytes": 4096},
        "after_repeats": {"allocate_calls": 2, "free_calls": 0, "allocated_bytes": 4096},
        "setup_delta": {"allocate_calls": 2, "free_calls": 0, "allocated_bytes": 4096},
        "measured_repeat_delta": dict(zero),
    }
    return capture


def add_auto_selector_fields(capture: dict) -> dict:
    capture["auto_selector"] = {
        "source": "rns8_bench_private_selector_report",
        "requested_backend": capture["backend_requested"],
        "selected_backend": capture["backend_selected"],
        "selected_key": capture["backend_metadata"].get("autotune_key"),
        "validated_hit": False,
        "cache_load_state": "missing",
        "runtime_target_id": capture.get("device", {}).get("gcn_arch", "gfx1100"),
        "runtime_version": str(capture.get("device", {}).get("hip_runtime_version", 0)),
        "fallback_reason": "no exact entry",
        "rejection_reason_vocabulary": [
            "unsupported semantics",
            "per-tile unsupported",
            "backend not compiled",
            "probe failed",
            "no exact entry",
            "unvalidated entry",
            "identity/runtime mismatch",
            "workspace mismatch",
            "slower than selected",
        ],
        "rejected_candidates": [{"backend": "ck", "reason": "backend not compiled"}],
    }
    return capture


def add_timing_helper_fields(
    capture: dict,
    pack_layout: str = "resident_rns_residue_planes",
    fusion_mode: str = "none",
    reducer: str = "not_applicable",
) -> dict:
    capture["timing_metadata"]["pack_layout"] = pack_layout
    capture["timing_metadata"]["fusion_mode"] = fusion_mode
    capture["timing_metadata"]["residue_group_width"] = 1 if fusion_mode == "none" else 3
    capture["timing_metadata"]["residue_group_layout"] = (
        "one_modulus_per_residue_plane"
        if fusion_mode == "none"
        else "first_prefix9_moduli_contiguous_width3_groups"
    )
    capture["timing_metadata"]["generated_reducer_identity"] = reducer
    return capture


def add_helper_lane_fields(capture: dict, resolved_next_op: str = "final-export") -> dict:
    add_target_variant_fields(capture)
    add_requested_next_op_fields(capture, resolved=resolved_next_op)
    add_output_policy_fields(capture)
    add_device_allocation_fields(capture)
    add_auto_selector_fields(capture)
    add_timing_helper_fields(capture)
    return capture


def int32_accumulator_safety(capture: dict, cap: int = 65536) -> dict:
    k_block = min(capture["k"], cap)
    finite = capture.get("semantics") in {"finite_ring_u8", "finite_field_u8"}
    return {
        "input_domain": "centered_i8_finite_u8_residues" if finite else "centered_i8_rns_residue_planes",
        "signedness": "signed_i8x_signed_i8",
        "accumulator_type": "int32",
        "modulus_policy": "finite_u8_modulus" if finite else "selected_rns_modulus_ladder",
        "modulus": capture.get("finite_modulus") if finite else 0,
        "uses_int32_inner_product": True,
        "k_block_size": k_block,
        "k_block_cap": cap,
        "max_lhs_abs": 128,
        "max_rhs_abs": 128,
        "max_product": 128 * 128,
        "safe_for_k_block": True,
        "status": "safe_int32_k_block_split",
    }


def apply_int32_accumulator_contract(capture: dict, cap: int = 65536) -> dict:
    safety = int32_accumulator_safety(capture, cap)
    capture["backend_metadata"]["accumulator_safety"] = safety
    capture["k_block_size"] = safety["k_block_size"]
    return capture


def with_accumulator_key_fields(key: str, capture: dict) -> str:
    safety = capture["backend_metadata"]["accumulator_safety"]
    parts = [
        part
        for part in key.split(";")
        if part.split("=", 1)[0]
        not in {
            "target_id",
            "accumulator_type",
            "accumulator_signedness",
            "accumulator_modulus_policy",
            "k_block_size",
            "k_block_cap",
        }
    ]
    target = capture.get("device", {}).get("gcn_arch", "cpu")
    if capture.get("backend_selected") not in {
        "hip-direct",
        "hipblaslt",
        "ck",
        "rocwmma",
        "amdgpu-builtins",
        "hip-vector-alu-int64",
    }:
        target = "cpu"
    if target in {"", "none", "unknown"}:
        target = "cpu"
    insert_target_at = 1 if parts and parts[0].startswith("backend=") else 0
    parts = parts[:insert_target_at] + [f"target_id={target}"] + parts[insert_target_at:]
    insert_at = next((index for index, part in enumerate(parts) if part.startswith("kernel=")), len(parts))
    additions = [
        f"accumulator_type={safety['accumulator_type']}",
        f"accumulator_signedness={safety['signedness']}",
        f"accumulator_modulus_policy={safety['modulus_policy']}",
        f"k_block_size={safety['k_block_size']}",
        f"k_block_cap={safety['k_block_cap']}",
    ]
    return ";".join(parts[:insert_at] + additions + parts[insert_at:])


def as_direct_hip_finite_capture(
    capture: dict, modulus: int, kernel: str, isa_evidence: str
) -> dict:
    direct = copy.deepcopy(capture)
    metadata = direct["backend_metadata"]
    direct["backend_requested"] = "hip-direct"
    direct["backend_selected"] = "hip-direct"
    direct["selected_kernel"] = kernel
    direct["finite_modulus"] = modulus
    direct["backend_metadata"]["selected_kernel"] = kernel
    metadata["accelerator_backend"] = False
    metadata["matrix_engine_backend"] = False
    metadata["accelerator_library"] = "HIP runtime"
    metadata["accelerator_version"] = None
    metadata["capability_status"] = "implemented_correctness_backend"
    metadata["epilogue_mode"] = "fused_centered_residue_then_canonical_u8_export"
    metadata["workspace_mode"] = "resident_device_buffers"
    metadata["workspace_required_bytes"] = 0
    metadata["isa_evidence"] = isa_evidence
    apply_int32_accumulator_contract(direct)
    metadata["autotune_key"] = with_accumulator_key_fields(
        (
        f"backend=hip-direct;semantics={direct['semantics']};m={direct['m']};n={direct['n']};k={direct['k']};"
        f"finite_modulus={modulus};prefix=0;tile_m={direct['tile_m']};tile_n={direct['tile_n']};"
        f"groups=0;adaptive_prefix=0;adaptive_skip=0;kernel={kernel};"
        "epilogue=fused_centered_residue_then_canonical_u8_export"
        ),
        direct,
    )
    direct["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "direct_hip_default_stream_backend_operation_groups"
    )
    return direct


def as_direct_hip_oneshot_capture(capture: dict) -> dict:
    oneshot = copy.deepcopy(capture)
    repeats = oneshot["repeats"]
    kernel = "direct_hip_prefix9_native_input_grouped_rns_gemm_v1"
    epilogue = "native_input_centered_residue_then_crt_export"
    oneshot["benchmark"] = "rns8_bounded_gemm_public_oneshot"
    oneshot["benchmark_execution_mode"] = "public_oneshot_transient_native_inputs"
    oneshot["backend_requested"] = "hip-direct"
    oneshot["backend_selected"] = "hip-direct"
    oneshot["selected_kernel"] = kernel
    oneshot["prefix"] = 9
    oneshot["requested_max_prefix"] = 9
    oneshot["selected_prefix"] = 9
    oneshot["contract_prefix_policy"] = "minimum_proven"
    oneshot["residue_planes_requested"] = 9
    oneshot["residue_planes_selected"] = 9
    oneshot["residue_planes_skipped"] = 0
    oneshot["residue_plane_skip_fraction"] = 0.0
    oneshot["schedule_metadata"]["min_required_prefix"] = 9
    oneshot["schedule_metadata"]["max_required_prefix"] = 9
    oneshot["schedule_metadata"]["min_selected_prefix"] = 9
    oneshot["schedule_metadata"]["max_selected_prefix"] = 9
    oneshot["schedule_metadata"]["prefix_group_count"] = 1
    oneshot["schedule_metadata"]["adaptive_prefix_active"] = False
    oneshot["schedule_metadata"]["adaptive_skip_active"] = False
    oneshot["schedule_metadata"]["adaptive_execution_applied"] = False
    oneshot["backend_metadata"]["source"] = "rns8_bench_public_oneshot_api"
    oneshot["backend_metadata"]["selected_kernel"] = kernel
    oneshot["backend_metadata"]["accelerator_backend"] = False
    oneshot["backend_metadata"]["matrix_engine_backend"] = False
    oneshot["backend_metadata"]["accelerator_library"] = "HIP runtime"
    oneshot["backend_metadata"]["accelerator_version"] = "7.1"
    oneshot["backend_metadata"]["capability_status"] = "implemented_correctness_backend"
    oneshot["backend_metadata"]["epilogue_mode"] = epilogue
    oneshot["backend_metadata"]["workspace_mode"] = "transient_native_inputs_to_resident_rns_output"
    oneshot["backend_metadata"]["workspace_required_bytes"] = 0
    oneshot["backend_metadata"]["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
    apply_int32_accumulator_contract(oneshot)
    oneshot["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;prefix=9;tile_m=128;tile_n=128;"
        "groups=1;adaptive_prefix=0;adaptive_skip=0;execution=public_oneshot_transient_native_inputs;"
        f"kernel={kernel};epilogue={epilogue}"
        ),
        oneshot,
    )
    oneshot["comparison_baseline"]["required_before_speedup_claim"] = [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_vector_alu_int64",
        "same_contract_direct_hip_persistent_rns",
    ]
    oneshot["timing_note"] = (
        "host wall-clock timings for the public bounded one-shot API; raw_timings_us.rns_gemm and "
        "raw_timings_us.end_to_end both measure one complete call"
    )
    oneshot["timing_metadata"]["benchmark_execution_mode"] = "public_oneshot_transient_native_inputs"
    oneshot["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_direct_hip_oneshot_api_hooks"
    oneshot["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_oneshot_default_stream_operation_groups"
    oneshot["timing_metadata"]["gpu_event_phase_order"] = [
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
    oneshot["timing_metadata"]["phase_notes"]["matrix_alloc"] = (
        "zero-valued external phase; transient API allocations are inside the measured one-shot call"
    )
    oneshot["timing_metadata"]["phase_notes"]["pack"] = (
        "zero-valued external phase; native input copies are inside the measured one-shot API call"
    )
    oneshot["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for one complete public bounded one-shot API call"
    )
    oneshot["timing_metadata"]["phase_notes"]["crt_export"] = (
        "zero-valued external phase; logical output export is inside the measured one-shot API call"
    )
    oneshot["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "same measured duration as rns_gemm for one complete public bounded one-shot API call"
    )
    oneshot["matrix_alloc_us"] = 0
    oneshot["avg_matrix_alloc_us"] = 0.0
    oneshot["avg_pack_us"] = 0.0
    oneshot["avg_crt_export_us"] = 0.0
    oneshot["avg_rns_gemm_us"] = 1000.0
    oneshot["avg_end_to_end_us"] = 1000.0
    oneshot["per_modulus_gemm_estimate_applicable"] = False
    oneshot["avg_per_modulus_gemm_estimate_us"] = 1000.0
    oneshot["raw_timings_us"]["matrix_alloc"] = [0]
    oneshot["raw_timings_us"]["pack"] = [0] * repeats
    oneshot["raw_timings_us"]["rns_gemm"] = [900, 1100]
    oneshot["raw_timings_us"]["crt_export"] = [0] * repeats
    oneshot["raw_timings_us"]["end_to_end"] = [900, 1100]
    oneshot["timing_summary_us"]["matrix_alloc"] = zero_summary()
    oneshot["timing_summary_us"]["pack"] = zero_summary()
    oneshot["timing_summary_us"]["rns_gemm"] = {"avg": 1000.0, "median": 1100.0, "p95": 1100.0}
    oneshot["timing_summary_us"]["crt_export"] = zero_summary()
    oneshot["timing_summary_us"]["end_to_end"] = {"avg": 1000.0, "median": 1100.0, "p95": 1100.0}
    event_values = {
        "oneshot_native_input_h2d": [10.0, 12.0],
        "rns_gemm_kernel_group": [100.0, 110.0],
        "rns_gemm": [100.0, 110.0],
        "crt_export_status_memset": [0.0, 0.0],
        "crt_export_kernel": [20.0, 22.0],
        "crt_export_status_d2h": [1.0, 1.0],
        "crt_export_d2h": [8.0, 9.0],
        "crt_export": [29.0, 32.0],
        "oneshot_api_gpu": [139.0, 154.0],
    }
    oneshot["gpu_event_timings_us"] = event_values
    oneshot["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    return oneshot


def as_direct_hip_resident_fallback_oneshot_capture(capture: dict) -> dict:
    fallback = copy.deepcopy(capture)
    repeats = fallback["repeats"]
    kernel = "direct_hip_grouped_active_prefix_schedule_rns_gemm_v3"
    epilogue = "fused_centered_residue_then_crt_export"
    fallback["benchmark"] = "rns8_bounded_gemm_public_oneshot"
    fallback["benchmark_execution_mode"] = "public_oneshot_transient_native_inputs"
    fallback["backend_requested"] = "hip-direct"
    fallback["backend_selected"] = "hip-direct"
    fallback["semantics"] = "bounded_i64"
    fallback["m"] = 32
    fallback["n"] = 32
    fallback["k"] = 32
    fallback["k_block_size"] = 32
    fallback["bound_kind"] = "global_max_abs"
    fallback["bound_mode"] = "global"
    fallback["prefix"] = 9
    fallback["requested_max_prefix"] = 9
    fallback["selected_prefix"] = 2
    fallback["contract_prefix_policy"] = "minimum_proven"
    fallback["residue_planes_requested"] = 9
    fallback["residue_planes_selected"] = 2
    fallback["residue_planes_skipped"] = 7
    fallback["residue_plane_skip_fraction"] = 7.0 / 9.0
    fallback["selected_kernel"] = kernel
    fallback["pack_mode"] = "per_repeat_repack"
    fallback["prepack_reuse_strategy"] = "none"
    fallback["reuse_packed_inputs"] = False
    fallback["residue_output_mode"] = "host_export"
    fallback["residue_chain_length"] = 1
    fallback["epilogue_type"] = "crt_export"
    fallback["tile_bounds_u64"] = None
    fallback["schedule_metadata"]["min_required_prefix"] = 2
    fallback["schedule_metadata"]["max_required_prefix"] = 2
    fallback["schedule_metadata"]["min_selected_prefix"] = 2
    fallback["schedule_metadata"]["max_selected_prefix"] = 2
    fallback["schedule_metadata"]["prefix_group_count"] = 1
    fallback["schedule_metadata"]["adaptive_prefix_active"] = False
    fallback["schedule_metadata"]["adaptive_skip_active"] = True
    fallback["schedule_metadata"]["adaptive_execution_applied"] = False
    fallback["schedule_metadata"]["zero_output_tile_count"] = 0
    fallback["backend_metadata"]["source"] = "rns8_bench_public_oneshot_api"
    fallback["backend_metadata"]["selected_kernel"] = kernel
    fallback["backend_metadata"]["accelerator_backend"] = False
    fallback["backend_metadata"]["matrix_engine_backend"] = False
    fallback["backend_metadata"]["accelerator_library"] = "HIP runtime"
    fallback["backend_metadata"]["accelerator_version"] = None
    fallback["backend_metadata"]["capability_status"] = "implemented_correctness_backend"
    fallback["backend_metadata"]["epilogue_mode"] = epilogue
    fallback["backend_metadata"]["workspace_mode"] = "resident_device_buffers"
    fallback["backend_metadata"]["workspace_required_bytes"] = 0
    fallback["backend_metadata"]["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
    apply_int32_accumulator_contract(fallback)
    fallback["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
            "backend=hip-direct;semantics=bounded_i64;m=32;n=32;k=32;bound_kind=global_max_abs;"
            "prefix=2;requested_max_prefix=9;prefix_policy=minimum_proven;tile_m=128;tile_n=128;"
            "groups=1;adaptive_prefix=0;adaptive_skip=1;schedule_flags=0;zero_output_tiles=0;"
            f"kernel={kernel};epilogue={epilogue}"
        ),
        fallback,
    )
    fallback["comparison_baseline"]["required_before_speedup_claim"] = [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_vector_alu_int64",
        "same_contract_direct_hip_persistent_rns",
    ]
    fallback["timing_note"] = (
        "host wall-clock timings for the public bounded one-shot API; raw_timings_us.rns_gemm and "
        "raw_timings_us.end_to_end both measure one complete call"
    )
    fallback["timing_metadata"]["benchmark_execution_mode"] = "public_oneshot_transient_native_inputs"
    fallback["timing_metadata"]["pack_mode"] = "per_repeat_repack"
    fallback["timing_metadata"]["gpu_event_timing"] = True
    fallback["timing_metadata"][
        "gpu_event_timing_reason"
    ] = "captured_by_direct_hip_oneshot_resident_fallback_api_hooks"
    fallback["timing_metadata"]["gpu_event_timing_status"] = "available"
    fallback["timing_metadata"]["gpu_event_timing_source"] = "hipEventElapsedTime"
    fallback["timing_metadata"][
        "gpu_event_timing_source_scope"
    ] = "direct_hip_oneshot_resident_fallback_default_stream_operation_groups"
    fallback["timing_metadata"]["gpu_event_timing_caveat"] = (
        "HIP event timings record the public bounded one-shot API's resident fallback pack, direct-HIP GEMM "
        "kernel group, and logical export operation groups; host wall-clock timings remain required"
    )
    fallback["timing_metadata"].pop("gpu_event_timing_unavailable_reasons", None)
    fallback["timing_metadata"]["gpu_event_phase_order"] = [
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
        "oneshot_api_gpu",
    ]
    fallback["timing_metadata"]["phase_notes"]["matrix_alloc"] = (
        "zero-valued external phase; transient API allocations are inside the measured one-shot call"
    )
    fallback["timing_metadata"]["phase_notes"]["pack"] = (
        "zero-valued external phase; native input copies are inside the measured one-shot API call"
    )
    fallback["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for one complete public bounded one-shot API call"
    )
    fallback["timing_metadata"]["phase_notes"]["crt_export"] = (
        "zero-valued external phase; logical output export is inside the measured one-shot API call"
    )
    fallback["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "same measured duration as rns_gemm for one complete public bounded one-shot API call"
    )
    fallback["matrix_alloc_us"] = 0
    fallback["avg_matrix_alloc_us"] = 0.0
    fallback["avg_pack_us"] = 0.0
    fallback["avg_crt_export_us"] = 0.0
    fallback["avg_rns_gemm_us"] = 1000.0
    fallback["avg_end_to_end_us"] = 1000.0
    fallback["per_modulus_gemm_estimate_applicable"] = False
    fallback["avg_per_modulus_gemm_estimate_us"] = 500.0
    fallback["raw_timings_us"]["matrix_alloc"] = [0]
    fallback["raw_timings_us"]["pack"] = [0] * repeats
    fallback["raw_timings_us"]["rns_gemm"] = [900, 1100]
    fallback["raw_timings_us"]["crt_export"] = [0] * repeats
    fallback["raw_timings_us"]["end_to_end"] = [900, 1100]
    fallback["timing_summary_us"]["matrix_alloc"] = zero_summary()
    fallback["timing_summary_us"]["pack"] = zero_summary()
    fallback["timing_summary_us"]["rns_gemm"] = {"avg": 1000.0, "median": 1100.0, "p95": 1100.0}
    fallback["timing_summary_us"]["crt_export"] = zero_summary()
    fallback["timing_summary_us"]["end_to_end"] = {"avg": 1000.0, "median": 1100.0, "p95": 1100.0}
    event_values = {
        "pack_h2d": [9.0, 10.0][:repeats],
        "pack_kernel": [11.0, 12.0][:repeats],
        "pack": [20.0, 22.0][:repeats],
        "rns_gemm_kernel_group": [90.0, 100.0][:repeats],
        "rns_gemm": [90.0, 100.0][:repeats],
        "crt_export_status_memset": [0.5, 0.5][:repeats],
        "crt_export_kernel": [30.0, 31.0][:repeats],
        "crt_export_status_d2h": [1.0, 1.0][:repeats],
        "crt_export_d2h": [10.0, 11.0][:repeats],
        "crt_export": [41.5, 43.5][:repeats],
        "oneshot_api_gpu": [151.5, 165.5][:repeats],
    }
    fallback["gpu_event_timings_us"] = event_values
    fallback["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    return fallback


