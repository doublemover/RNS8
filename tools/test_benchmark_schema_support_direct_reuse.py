def as_direct_hip_finite_oneshot_capture(capture: dict) -> dict:
    oneshot = as_direct_hip_finite_capture(
        capture,
        255,
        "direct_hip_native_finite_u8_gemm_mod255_v1",
        "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide",
    )
    repeats = oneshot["repeats"]
    epilogue = "native_u8_centered_residue_then_canonical_u8_export"
    oneshot["benchmark"] = "rns8_finite_u8_public_oneshot"
    oneshot["benchmark_execution_mode"] = "public_oneshot_transient_native_inputs"
    oneshot["backend_metadata"]["source"] = "rns8_bench_public_oneshot_api"
    oneshot["backend_metadata"]["epilogue_mode"] = epilogue
    oneshot["backend_metadata"]["workspace_mode"] = "transient_native_u8_inputs_to_resident_finite_output"
    oneshot["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=finite_ring_u8;m=64;n=128;k=64;finite_modulus=255;"
        "tile_m=128;tile_n=128;execution=public_oneshot_transient_native_inputs;"
        "kernel=direct_hip_native_finite_u8_gemm_mod255_v1;"
        f"epilogue={epilogue}"
        ),
        oneshot,
    )
    oneshot["comparison_baseline"]["required_before_speedup_claim"] = [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_persistent_finite_u8",
    ]
    oneshot["timing_note"] = (
        "host wall-clock timings for the public finite-u8 one-shot API; raw_timings_us.rns_gemm and "
        "raw_timings_us.end_to_end both measure one complete call"
    )
    oneshot["timing_metadata"]["benchmark_execution_mode"] = "public_oneshot_transient_native_inputs"
    oneshot["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_direct_hip_oneshot_api_hooks"
    oneshot["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_oneshot_default_stream_operation_groups"
    oneshot["timing_metadata"]["gpu_event_phase_order"] = [
        "oneshot_native_input_h2d",
        "finite_native_gemm_kernel",
        "rns_gemm",
        "finite_export_kernel",
        "finite_export_d2h",
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
        "per-repeat host timing for one complete public one-shot API call"
    )
    oneshot["timing_metadata"]["phase_notes"]["crt_export"] = (
        "zero-valued external phase; logical output export is inside the measured one-shot API call"
    )
    oneshot["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "same measured duration as rns_gemm for one complete public one-shot API call"
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
    oneshot["raw_timings_us"]["rns_gemm"] = [900, 1100][:repeats]
    oneshot["raw_timings_us"]["crt_export"] = [0] * repeats
    oneshot["raw_timings_us"]["end_to_end"] = [900, 1100][:repeats]
    oneshot["timing_summary_us"]["matrix_alloc"] = zero_summary()
    oneshot["timing_summary_us"]["pack"] = zero_summary()
    oneshot["timing_summary_us"]["rns_gemm"] = summary(oneshot["raw_timings_us"]["rns_gemm"])
    oneshot["timing_summary_us"]["crt_export"] = zero_summary()
    oneshot["timing_summary_us"]["end_to_end"] = summary(oneshot["raw_timings_us"]["end_to_end"])
    event_values = {
        "oneshot_native_input_h2d": [10.0, 12.0][:repeats],
        "finite_native_gemm_kernel": [100.0, 110.0][:repeats],
        "rns_gemm": [100.0, 110.0][:repeats],
        "finite_export_kernel": [20.0, 22.0][:repeats],
        "finite_export_d2h": [8.0, 9.0][:repeats],
        "crt_export": [28.0, 31.0][:repeats],
        "oneshot_api_gpu": [138.0, 153.0][:repeats],
    }
    oneshot["gpu_event_timings_us"] = event_values
    oneshot["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    return oneshot


def as_direct_hip_finite_native_a_reuse_b_capture(capture: dict) -> dict:
    reused = as_direct_hip_finite_capture(
        capture,
        255,
        "direct_hip_native_a_finite_u8_gemm_mod255_v1",
        "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide",
    )
    repeats = reused["repeats"]
    epilogue = "native_a_centered_resident_b_residue_then_canonical_u8_export"
    reused["benchmark_execution_mode"] = "transient_native_a_resident_b_reuse"
    reused["backend_metadata"]["source"] = "rns8_bench_native_a_reuse_b_path"
    reused["backend_metadata"]["epilogue_mode"] = epilogue
    reused["backend_metadata"]["workspace_mode"] = "transient_native_u8_a_resident_finite_b_output"
    reused["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=finite_ring_u8;m=64;n=128;k=64;finite_modulus=255;"
        "tile_m=128;tile_n=128;execution=transient_native_a_resident_b_reuse;"
        "kernel=direct_hip_native_a_finite_u8_gemm_mod255_v1;"
        f"epilogue={epilogue}"
        ),
        reused,
    )
    reused["pack_mode"] = "prepacked_reuse_b"
    reused["reuse_packed_inputs"] = True
    reused["prepack_reuse_operands"] = ["B"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 300
    reused["avg_prepack_setup_us"] = 300.0
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "B was packed once before warmups and reused for every measured repeat",
    }
    reused["timing_metadata"]["benchmark_execution_mode"] = "transient_native_a_resident_b_reuse"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["B"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["gpu_event_phase_order"] = [
        "finite_pack_h2d",
        "finite_pack_kernel",
        "pack",
        "finite_native_a_gemm_kernel",
        "rns_gemm",
        "finite_export_kernel",
        "finite_export_d2h",
        "crt_export",
    ]
    event_values = {
        "finite_pack_h2d": [12.0, 13.0][:repeats],
        "finite_pack_kernel": [0.0, 0.0][:repeats],
        "pack": [12.0, 13.0][:repeats],
        "finite_native_a_gemm_kernel": [80.0, 82.0][:repeats],
        "rns_gemm": [80.0, 82.0][:repeats],
        "finite_export_kernel": [20.0, 22.0][:repeats],
        "finite_export_d2h": [8.0, 9.0][:repeats],
        "crt_export": [28.0, 31.0][:repeats],
    }
    reused["gpu_event_timings_us"] = event_values
    reused["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    reused["raw_timings_us"]["pack"] = [100, 110][:repeats]
    reused["raw_timings_us"]["rns_gemm"] = [200, 210][:repeats]
    reused["raw_timings_us"]["crt_export"] = [90, 100][:repeats]
    reused["raw_timings_us"]["end_to_end"] = [390, 420][:repeats]
    reused["timing_summary_us"]["pack"] = summary(reused["raw_timings_us"]["pack"])
    reused["timing_summary_us"]["rns_gemm"] = summary(reused["raw_timings_us"]["rns_gemm"])
    reused["timing_summary_us"]["crt_export"] = summary(reused["raw_timings_us"]["crt_export"])
    reused["timing_summary_us"]["end_to_end"] = summary(reused["raw_timings_us"]["end_to_end"])
    reused["avg_pack_us"] = reused["timing_summary_us"]["pack"]["avg"]
    reused["avg_rns_gemm_us"] = reused["timing_summary_us"]["rns_gemm"]["avg"]
    reused["avg_crt_export_us"] = reused["timing_summary_us"]["crt_export"]["avg"]
    reused["avg_end_to_end_us"] = reused["timing_summary_us"]["end_to_end"]["avg"]
    return reused


def as_direct_hip_bounded_native_a_reuse_b_capture(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    repeats = reused["repeats"]
    kernel = "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_b_grouped_rns_gemm_v2"
    epilogue = "uniform_small_i8_ab_resident_b_residue_then_crt_export"
    gemm_event = "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    reused["benchmark_execution_mode"] = "transient_uniform_small_i8_a_resident_i8_b_reuse"
    reused["backend_requested"] = "hip-direct"
    reused["backend_selected"] = "hip-direct"
    reused["selected_kernel"] = kernel
    reused["backend_metadata"]["source"] = "rns8_bench_uniform_small_i8_ab_reuse_b_path"
    reused["backend_metadata"]["selected_kernel"] = kernel
    reused["backend_metadata"]["accelerator_backend"] = False
    reused["backend_metadata"]["matrix_engine_backend"] = False
    reused["backend_metadata"]["accelerator_library"] = "HIP runtime"
    reused["backend_metadata"]["accelerator_version"] = "7.1"
    reused["backend_metadata"]["capability_status"] = "implemented_correctness_backend"
    reused["backend_metadata"]["epilogue_mode"] = epilogue
    reused["backend_metadata"]["workspace_mode"] = "transient_native_a_resident_rns_b_output"
    reused["backend_metadata"]["workspace_required_bytes"] = 0
    reused["backend_metadata"]["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
    apply_int32_accumulator_contract(reused)
    reused["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
        "input_profile=uniform-small;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=transient_uniform_small_i8_a_resident_i8_b_reuse;"
        f"kernel={kernel};epilogue={epilogue}"
        ),
        reused,
    )
    reused["pack_mode"] = "prepacked_reuse_b"
    reused["reuse_packed_inputs"] = True
    reused["prepack_reuse_operands"] = ["B"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 400
    reused["avg_prepack_setup_us"] = 400.0
    reused["timing_metadata"]["benchmark_execution_mode"] = "transient_uniform_small_i8_a_resident_i8_b_reuse"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["B"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "direct_hip_default_stream_backend_operation_groups"
    )
    reused["timing_metadata"]["gpu_event_phase_order"] = [
        "pack_h2d",
        "pack_kernel",
        "pack",
        gemm_event,
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
    ]
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "B was packed once before warmups and reused for every measured repeat",
    }
    event_values = {
        "pack_h2d": [18.0, 19.0][:repeats],
        "pack_kernel": [0.0, 0.0][:repeats],
        "pack": [18.0, 19.0][:repeats],
        gemm_event: [150.0, 151.0][:repeats],
        "rns_gemm": [150.0, 151.0][:repeats],
        "crt_export_status_memset": [0.5, 0.5][:repeats],
        "crt_export_kernel": [30.0, 31.0][:repeats],
        "crt_export_status_d2h": [1.0, 1.0][:repeats],
        "crt_export_d2h": [10.0, 11.0][:repeats],
        "crt_export": [41.5, 43.5][:repeats],
    }
    reused["gpu_event_timings_us"] = event_values
    reused["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    reused["raw_timings_us"]["pack"] = [120, 125][:repeats]
    reused["raw_timings_us"]["rns_gemm"] = [260, 270][:repeats]
    reused["raw_timings_us"]["crt_export"] = [95, 100][:repeats]
    reused["raw_timings_us"]["end_to_end"] = [475, 495][:repeats]
    reused["timing_summary_us"]["pack"] = summary(reused["raw_timings_us"]["pack"])
    reused["timing_summary_us"]["rns_gemm"] = summary(reused["raw_timings_us"]["rns_gemm"])
    reused["timing_summary_us"]["crt_export"] = summary(reused["raw_timings_us"]["crt_export"])
    reused["timing_summary_us"]["end_to_end"] = summary(reused["raw_timings_us"]["end_to_end"])
    reused["avg_pack_us"] = reused["timing_summary_us"]["pack"]["avg"]
    reused["avg_rns_gemm_us"] = reused["timing_summary_us"]["rns_gemm"]["avg"]
    reused["avg_crt_export_us"] = reused["timing_summary_us"]["crt_export"]["avg"]
    reused["avg_end_to_end_us"] = reused["timing_summary_us"]["end_to_end"]["avg"]
    reused["avg_per_modulus_gemm_estimate_us"] = reused["avg_rns_gemm_us"] / reused["prefix"]
    return reused


def as_direct_hip_bounded_uniform_small_transient_capture(capture: dict) -> dict:
    transient = copy.deepcopy(capture)
    repeats = transient["repeats"]
    kernel = "direct_hip_uniform_small_i8_ab_colpair_prefix9_transient_grouped_rns_gemm_v1"
    epilogue = "uniform_small_i8_ab_transient_residue_then_crt_export"
    gemm_event = "bounded_uniform_small_i8_ab_transient_gemm_kernel_group"
    transient["benchmark"] = "rns8_bounded_gemm_transient_uniform_small_i8"
    transient["benchmark_execution_mode"] = "transient_uniform_small_i8_ab_inputs"
    transient["backend_requested"] = "hip-direct"
    transient["backend_selected"] = "hip-direct"
    transient["selected_kernel"] = kernel
    transient["backend_metadata"]["source"] = "rns8_bench_uniform_small_i8_ab_transient_path"
    transient["backend_metadata"]["selected_kernel"] = kernel
    transient["backend_metadata"]["accelerator_backend"] = False
    transient["backend_metadata"]["matrix_engine_backend"] = False
    transient["backend_metadata"]["accelerator_library"] = "HIP runtime"
    transient["backend_metadata"]["accelerator_version"] = "7.1"
    transient["backend_metadata"]["capability_status"] = "implemented_correctness_backend"
    transient["backend_metadata"]["epilogue_mode"] = epilogue
    transient["backend_metadata"]["workspace_mode"] = "transient_i8_a_transient_i8_b_rns_output"
    transient["backend_metadata"]["workspace_required_bytes"] = 0
    transient["backend_metadata"]["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
    apply_int32_accumulator_contract(transient)
    transient["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
        "input_profile=uniform-small;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=transient_uniform_small_i8_ab_inputs;"
        f"kernel={kernel};epilogue={epilogue}"
        ),
        transient,
    )
    transient["pack_mode"] = "per_repeat_repack"
    transient["reuse_packed_inputs"] = False
    transient["prepack_reuse_operands"] = []
    transient["prepack_reuse_strategy"] = "none"
    transient["prepack_setup_us"] = None
    transient["avg_prepack_setup_us"] = None
    transient["timing_note"] = (
        "host wall-clock timings for an explicit benchmark-owned direct-HIP uniform-small native-input path"
    )
    transient["timing_metadata"]["benchmark_execution_mode"] = "transient_uniform_small_i8_ab_inputs"
    transient["timing_metadata"]["pack_mode"] = "per_repeat_repack"
    transient["timing_metadata"]["prepack_reuse_operands"] = []
    transient["timing_metadata"]["prepack_reuse_strategy"] = "none"
    transient["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "direct_hip_default_stream_backend_operation_groups"
    )
    transient["timing_metadata"]["gpu_event_phase_order"] = [
        "bounded_uniform_small_i8_a_h2d",
        "bounded_uniform_small_i8_b_h2d",
        "pack",
        gemm_event,
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
    ]
    transient["timing_metadata"]["phase_notes"]["pack"] = (
        "per-repeat host timing for copying uniform-small A and B into benchmark-owned native int8 HIP buffers"
    )
    event_values = {
        "bounded_uniform_small_i8_a_h2d": [14.0, 15.0][:repeats],
        "bounded_uniform_small_i8_b_h2d": [16.0, 17.0][:repeats],
        "pack": [30.0, 32.0][:repeats],
        gemm_event: [142.0, 143.0][:repeats],
        "rns_gemm": [142.0, 143.0][:repeats],
        "crt_export_status_memset": [0.5, 0.5][:repeats],
        "crt_export_kernel": [30.0, 31.0][:repeats],
        "crt_export_status_d2h": [1.0, 1.0][:repeats],
        "crt_export_d2h": [10.0, 11.0][:repeats],
        "crt_export": [41.5, 43.5][:repeats],
    }
    transient["gpu_event_timings_us"] = event_values
    transient["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    transient["raw_timings_us"]["pack"] = [115, 120][:repeats]
    transient["raw_timings_us"]["rns_gemm"] = [250, 260][:repeats]
    transient["raw_timings_us"]["crt_export"] = [95, 100][:repeats]
    transient["raw_timings_us"]["end_to_end"] = [460, 480][:repeats]
    transient["timing_summary_us"]["pack"] = summary(transient["raw_timings_us"]["pack"])
    transient["timing_summary_us"]["rns_gemm"] = summary(transient["raw_timings_us"]["rns_gemm"])
    transient["timing_summary_us"]["crt_export"] = summary(transient["raw_timings_us"]["crt_export"])
    transient["timing_summary_us"]["end_to_end"] = summary(transient["raw_timings_us"]["end_to_end"])
    transient["avg_pack_us"] = transient["timing_summary_us"]["pack"]["avg"]
    transient["avg_rns_gemm_us"] = transient["timing_summary_us"]["rns_gemm"]["avg"]
    transient["avg_crt_export_us"] = transient["timing_summary_us"]["crt_export"]["avg"]
    transient["avg_end_to_end_us"] = transient["timing_summary_us"]["end_to_end"]["avg"]
    transient["avg_per_modulus_gemm_estimate_us"] = transient["avg_rns_gemm_us"] / transient["prefix"]
    return transient


def as_direct_hip_bounded_residue_channel_fusion_capture(capture: dict) -> dict:
    fusion = as_direct_hip_bounded_uniform_small_transient_capture(capture)
    kernel = "direct_hip_uniform_small_i8_ab_colpair_prefix9_residue_channel_width3_experimental_v0"
    epilogue = "width3_residue_fusion_transient_then_crt_export"
    fusion["benchmark"] = "rns8_bounded_gemm_residue_channel_fusion_experiment"
    fusion["benchmark_execution_mode"] = "residue_channel_fusion_native_inputs"
    fusion["selected_kernel"] = kernel
    fusion["backend_metadata"]["source"] = "rns8_bench_residue_channel_fusion_path"
    fusion["backend_metadata"]["selected_kernel"] = kernel
    fusion["backend_metadata"]["epilogue_mode"] = epilogue
    fusion["backend_metadata"]["workspace_mode"] = "width3_residue_fusion_transient_i8_inputs"
    fusion["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
        "input_profile=uniform-small;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=residue_channel_fusion_native_inputs;"
        f"kernel={kernel};epilogue={epilogue}"
        ),
        fusion,
    )
    fusion["timing_metadata"]["benchmark_execution_mode"] = "residue_channel_fusion_native_inputs"
    add_target_variant_fields(fusion)
    add_requested_next_op_fields(fusion, resolved="final-export")
    add_output_policy_fields(fusion)
    add_timing_helper_fields(
        fusion,
        pack_layout="native_i8_row_major_residue_channel_width3",
        fusion_mode="residue_channel_width3_experimental_benchmark_only",
        reducer="direct_hip_fixed_prefix_9_generated_reducer_v1",
    )
    return fusion


def as_direct_hip_bounded_uniform_small_reuse_a_capture(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    repeats = reused["repeats"]
    kernel = "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_a_grouped_rns_gemm_v1"
    epilogue = "uniform_small_i8_ab_resident_a_residue_then_crt_export"
    gemm_event = "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group"
    reused["benchmark_execution_mode"] = "transient_uniform_small_i8_b_resident_i8_a_reuse"
    reused["backend_requested"] = "hip-direct"
    reused["backend_selected"] = "hip-direct"
    reused["selected_kernel"] = kernel
    reused["backend_metadata"]["source"] = "rns8_bench_uniform_small_i8_ab_reuse_a_path"
    reused["backend_metadata"]["selected_kernel"] = kernel
    reused["backend_metadata"]["accelerator_backend"] = False
    reused["backend_metadata"]["matrix_engine_backend"] = False
    reused["backend_metadata"]["accelerator_library"] = "HIP runtime"
    reused["backend_metadata"]["accelerator_version"] = "7.1"
    reused["backend_metadata"]["capability_status"] = "implemented_correctness_backend"
    reused["backend_metadata"]["epilogue_mode"] = epilogue
    reused["backend_metadata"]["workspace_mode"] = "transient_i8_b_resident_i8_a_rns_output"
    reused["backend_metadata"]["workspace_required_bytes"] = 0
    reused["backend_metadata"]["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
    apply_int32_accumulator_contract(reused)
    reused["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
        "input_profile=uniform-small;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=transient_uniform_small_i8_b_resident_i8_a_reuse;"
        f"kernel={kernel};epilogue={epilogue}"
        ),
        reused,
    )
    reused["pack_mode"] = "prepacked_reuse_a"
    reused["reuse_packed_inputs"] = True
    reused["prepack_reuse_operands"] = ["A"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 320
    reused["avg_prepack_setup_us"] = 320.0
    reused["timing_metadata"]["benchmark_execution_mode"] = "transient_uniform_small_i8_b_resident_i8_a_reuse"
    reused["timing_metadata"]["pack_mode"] = "prepacked_reuse_a"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["A"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "direct_hip_default_stream_backend_operation_groups"
    )
    reused["timing_metadata"]["gpu_event_phase_order"] = [
        "pack_h2d",
        "pack_kernel",
        "pack",
        gemm_event,
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
    ]
    reused["timing_metadata"]["phase_notes"]["pack"] = (
        "per-repeat host timing for copying uniform-small B; A was copied once before warmups"
    )
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "A was copied once before warmups and reused for every measured repeat",
    }
    event_values = {
        "pack_h2d": [17.0, 18.0][:repeats],
        "pack_kernel": [0.0, 0.0][:repeats],
        "pack": [17.0, 18.0][:repeats],
        gemm_event: [145.0, 146.0][:repeats],
        "rns_gemm": [145.0, 146.0][:repeats],
        "crt_export_status_memset": [0.5, 0.5][:repeats],
        "crt_export_kernel": [30.0, 31.0][:repeats],
        "crt_export_status_d2h": [1.0, 1.0][:repeats],
        "crt_export_d2h": [10.0, 11.0][:repeats],
        "crt_export": [41.5, 43.5][:repeats],
    }
    reused["gpu_event_timings_us"] = event_values
    reused["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    reused["raw_timings_us"]["pack"] = [118, 123][:repeats]
    reused["raw_timings_us"]["rns_gemm"] = [255, 265][:repeats]
    reused["raw_timings_us"]["crt_export"] = [95, 100][:repeats]
    reused["raw_timings_us"]["end_to_end"] = [468, 488][:repeats]
    reused["timing_summary_us"]["pack"] = summary(reused["raw_timings_us"]["pack"])
    reused["timing_summary_us"]["rns_gemm"] = summary(reused["raw_timings_us"]["rns_gemm"])
    reused["timing_summary_us"]["crt_export"] = summary(reused["raw_timings_us"]["crt_export"])
    reused["timing_summary_us"]["end_to_end"] = summary(reused["raw_timings_us"]["end_to_end"])
    reused["avg_pack_us"] = reused["timing_summary_us"]["pack"]["avg"]
    reused["avg_rns_gemm_us"] = reused["timing_summary_us"]["rns_gemm"]["avg"]
    reused["avg_crt_export_us"] = reused["timing_summary_us"]["crt_export"]["avg"]
    reused["avg_end_to_end_us"] = reused["timing_summary_us"]["end_to_end"]["avg"]
    reused["avg_per_modulus_gemm_estimate_us"] = reused["avg_rns_gemm_us"] / reused["prefix"]
    return reused


def as_direct_hip_bounded_native_b_reuse_a_capture(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    repeats = reused["repeats"]
    kernel = "direct_hip_native_b_u64_colpair_prefix9_reuse_a_grouped_rns_gemm_v1"
    epilogue = "resident_a_native_b_centered_residue_then_crt_export"
    gemm_event = "bounded_native_b_colpair_reuse_a_gemm_kernel_group"
    reused["semantics"] = "bounded_u64"
    reused["bound_kind"] = "global_max_unsigned"
    reused["input_distribution"] = "unsigned_adaptive_bands_0_16"
    reused["m"] = 512
    reused["n"] = 512
    reused["k"] = 512
    reused["benchmark_execution_mode"] = "transient_native_b_resident_a_reuse"
    reused["backend_requested"] = "hip-direct"
    reused["backend_selected"] = "hip-direct"
    reused["selected_kernel"] = kernel
    reused["backend_metadata"]["source"] = "rns8_bench_native_b_reuse_a_path"
    reused["backend_metadata"]["selected_kernel"] = kernel
    reused["backend_metadata"]["accelerator_backend"] = False
    reused["backend_metadata"]["matrix_engine_backend"] = False
    reused["backend_metadata"]["accelerator_library"] = "HIP runtime"
    reused["backend_metadata"]["accelerator_version"] = "7.1"
    reused["backend_metadata"]["capability_status"] = "implemented_correctness_backend"
    reused["backend_metadata"]["epilogue_mode"] = epilogue
    reused["backend_metadata"]["workspace_mode"] = "transient_native_b_resident_rns_a_output"
    reused["backend_metadata"]["workspace_required_bytes"] = 0
    reused["backend_metadata"]["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
    apply_int32_accumulator_contract(reused)
    reused["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_u64;m=512;n=512;k=512;bound=16384;"
        "input_profile=adaptive-bands;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=transient_native_b_resident_a_reuse;"
        f"kernel={kernel};epilogue={epilogue}"
        ),
        reused,
    )
    reused["pack_mode"] = "prepacked_reuse_a"
    reused["reuse_packed_inputs"] = True
    reused["prepack_reuse_operands"] = ["A"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 360
    reused["avg_prepack_setup_us"] = 360.0
    reused["timing_metadata"]["benchmark_execution_mode"] = "transient_native_b_resident_a_reuse"
    reused["timing_metadata"]["pack_mode"] = "prepacked_reuse_a"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["A"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "direct_hip_default_stream_backend_operation_groups"
    )
    reused["timing_metadata"]["gpu_event_phase_order"] = [
        "pack_h2d",
        "pack_kernel",
        "pack",
        gemm_event,
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
    ]
    reused["timing_metadata"]["phase_notes"]["pack"] = (
        "per-repeat host timing for copying native B; A was packed once before warmups"
    )
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "A was packed once before warmups and reused for every measured repeat",
    }
    event_values = {
        "pack_h2d": [19.0, 20.0][:repeats],
        "pack_kernel": [0.0, 0.0][:repeats],
        "pack": [19.0, 20.0][:repeats],
        gemm_event: [148.0, 149.0][:repeats],
        "rns_gemm": [148.0, 149.0][:repeats],
        "crt_export_status_memset": [0.5, 0.5][:repeats],
        "crt_export_kernel": [30.0, 31.0][:repeats],
        "crt_export_status_d2h": [1.0, 1.0][:repeats],
        "crt_export_d2h": [10.0, 11.0][:repeats],
        "crt_export": [41.5, 43.5][:repeats],
    }
    reused["gpu_event_timings_us"] = event_values
    reused["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    reused["raw_timings_us"]["pack"] = [122, 127][:repeats]
    reused["raw_timings_us"]["rns_gemm"] = [258, 268][:repeats]
    reused["raw_timings_us"]["crt_export"] = [95, 100][:repeats]
    reused["raw_timings_us"]["end_to_end"] = [475, 495][:repeats]
    reused["timing_summary_us"]["pack"] = summary(reused["raw_timings_us"]["pack"])
    reused["timing_summary_us"]["rns_gemm"] = summary(reused["raw_timings_us"]["rns_gemm"])
    reused["timing_summary_us"]["crt_export"] = summary(reused["raw_timings_us"]["crt_export"])
    reused["timing_summary_us"]["end_to_end"] = summary(reused["raw_timings_us"]["end_to_end"])
    reused["avg_pack_us"] = reused["timing_summary_us"]["pack"]["avg"]
    reused["avg_rns_gemm_us"] = reused["timing_summary_us"]["rns_gemm"]["avg"]
    reused["avg_crt_export_us"] = reused["timing_summary_us"]["crt_export"]["avg"]
    reused["avg_end_to_end_us"] = reused["timing_summary_us"]["end_to_end"]["avg"]
    reused["avg_per_modulus_gemm_estimate_us"] = reused["avg_rns_gemm_us"] / reused["prefix"]
    return reused


def as_direct_hip_bounded_skinny_gemv_n1_capture(capture: dict) -> dict:
    skinny = copy.deepcopy(capture)
    repeats = skinny["repeats"]
    semantics = skinny["semantics"]
    signed = semantics == "bounded_i64"
    kernel = (
        "direct_hip_prefix9_rns_gemv_n1_i64_v1"
        if signed
        else "direct_hip_prefix9_rns_gemv_n1_u64_v1"
    )
    epilogue = "resident_rns_gemv_n1_centered_residue_then_crt_export"
    bound_kind = "global_max_abs" if signed else "global_max_unsigned"
    input_profile = "uniform-small" if signed else "adaptive-bands"
    distribution = "signed_uniform_-16_16" if signed else "unsigned_adaptive_bands_0_16"
    accumulator_signedness = "signed_i8x_signed_i8" if signed else "unsigned_i8x_unsigned_i8"
    skinny["benchmark"] = "rns8_bounded_gemm_direct_hip_skinny_gemv_n1"
    skinny["benchmark_execution_mode"] = "direct_hip_skinny_gemv_n1_resident_rns"
    skinny["backend_requested"] = "hip-direct"
    skinny["backend_selected"] = "hip-direct"
    skinny["m"] = 256
    skinny["n"] = 1
    skinny["k"] = 4096
    skinny["output_logical_ld"] = 1
    skinny["output_ld_padding"] = 0
    skinny["output_ld"] = 1
    skinny["bound_kind"] = bound_kind
    skinny["bound_mode"] = "global"
    skinny["bound_source"] = "static_profile"
    skinny["bound"] = 1048576
    skinny["input_profile"] = input_profile
    skinny["input_distribution"] = distribution
    skinny["selected_kernel"] = kernel
    skinny["epilogue_type"] = "crt_export"
    skinny["prefix"] = 9
    skinny["selected_prefix"] = 9
    skinny["max_prefix"] = 9
    skinny["requested_max_prefix"] = 9
    skinny["prefix_policy"] = "minimum_proven"
    skinny["prefix_group_count"] = 1
    skinny["tile_m"] = 128
    skinny["tile_n"] = 128
    skinny["k_block_size"] = 4096
    skinny["reuse_packed_inputs"] = False
    skinny["prepack_reuse_operands"] = []
    skinny["prepack_reuse_strategy"] = "none"
    skinny["prepack_setup_us"] = None
    skinny["avg_prepack_setup_us"] = None
    skinny["pack_mode"] = "per_repeat_repack"
    skinny["residue_chain_length"] = 1
    skinny["residue_output_mode"] = "host_export"
    skinny["next_op_contract"] = "host_export"
    skinny["residue_channel_fusion"] = False
    skinny["native_to_rns_bridge"] = False
    skinny["vector_to_rns_chain"] = False
    skinny["host_api_batch_size"] = 1
    skinny["grouped_dispatch"] = None
    skinny["oneshot"] = False
    skinny["hip_graph_replay"] = None
    skinny["transient_uniform_small_inputs"] = False
    skinny["release_gate"] = None
    skinny["verification_amortization"] = None
    skinny["error_detection_policy"] = None
    skinny["cpu_small_shape_selector"] = None
    skinny["incremental_result_cache"] = None
    skinny["schedule_metadata"]["tile_m"] = skinny["tile_m"]
    skinny["schedule_metadata"]["tile_n"] = skinny["tile_n"]
    skinny["schedule_metadata"]["tile_rows"] = 2
    skinny["schedule_metadata"]["tile_cols"] = 1
    skinny["schedule_metadata"]["tile_count"] = 2
    skinny["schedule_metadata"]["min_required_prefix"] = 3
    skinny["schedule_metadata"]["max_required_prefix"] = 3
    skinny["schedule_metadata"]["min_selected_prefix"] = 9
    skinny["schedule_metadata"]["max_selected_prefix"] = 9
    skinny["schedule_metadata"]["prefix_group_count"] = 1
    skinny["schedule_metadata"]["adaptive_prefix_active"] = False
    skinny["schedule_metadata"]["adaptive_skip_active"] = False
    skinny["schedule_metadata"]["adaptive_execution_applied"] = False
    skinny["schedule_metadata"]["range_bit_length"] = 21
    skinny["schedule_metadata"]["zero_a_row_proof_count"] = 0
    skinny["schedule_metadata"]["zero_b_col_proof_count"] = 0
    skinny["schedule_metadata"]["zero_row_col_product_count"] = 0
    skinny["schedule_metadata"]["planner_zero_a_row_count"] = 0
    skinny["schedule_metadata"]["planner_zero_b_col_count"] = 0
    skinny["schedule_metadata"]["planner_zero_row_col_product_count"] = 0
    add_prefix_policy_fields(skinny, "minimum_proven")
    add_output_policy_fields(skinny)
    add_target_variant_fields(skinny)
    skinny["backend_metadata"] = copy.deepcopy(skinny["backend_metadata"])
    skinny["backend_metadata"]["source"] = "rns8_bench_skinny_gemv_n1_path"
    skinny["backend_metadata"]["selected_kernel"] = kernel
    skinny["backend_metadata"]["accelerator_backend"] = False
    skinny["backend_metadata"]["matrix_engine_backend"] = False
    skinny["backend_metadata"]["accelerator_library"] = "HIP runtime"
    skinny["backend_metadata"]["accelerator_version"] = "7.1"
    skinny["backend_metadata"]["capability_status"] = "implemented_correctness_backend"
    skinny["backend_metadata"]["epilogue_mode"] = epilogue
    skinny["backend_metadata"]["workspace_mode"] = "resident_rns_inputs_skinny_n1_output"
    skinny["backend_metadata"]["workspace_required_bytes"] = 0
    skinny["backend_metadata"]["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
    apply_int32_accumulator_contract(skinny)
    skinny["backend_metadata"]["accumulator_safety"]["accumulator_signedness"] = accumulator_signedness
    skinny["backend_metadata"]["accumulator_safety"]["k_block_size"] = skinny["k"]
    skinny["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
            f"backend=hip-direct;semantics={semantics};m=256;n=1;k=4096;bound=1048576;"
            f"input_profile={input_profile};"
            "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
            "execution=direct_hip_skinny_gemv_n1_resident_rns;"
            f"kernel={kernel};epilogue={epilogue}"
        ),
        skinny,
    )
    skinny["timing_metadata"]["benchmark_execution_mode"] = "direct_hip_skinny_gemv_n1_resident_rns"
    skinny["timing_metadata"]["pack_mode"] = "per_repeat_repack"
    skinny["timing_metadata"]["prepack_reuse_operands"] = []
    skinny["timing_metadata"]["prepack_reuse_strategy"] = "none"
    skinny["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "direct_hip_default_stream_backend_operation_groups"
    )
    skinny["timing_metadata"]["gpu_event_phase_order"] = [
        "pack_h2d",
        "pack_kernel",
        "pack",
        "rns_gemv_n1_kernel_group",
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
    ]
    skinny["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "HIP events around the skinny-N=1 resident-RNS GEMV kernel group"
    )
    skinny["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": False,
        "timing_key": None,
        "scope": "not_requested_per_repeat_repack",
        "reason": "skinny GEMV path repacks inputs per repeat and does not use a persistent prepack setup",
    }
    event_values = {
        "pack_h2d": [20.0, 21.0][:repeats],
        "pack_kernel": [42.0, 43.0][:repeats],
        "pack": [62.0, 64.0][:repeats],
        "rns_gemv_n1_kernel_group": [88.0, 89.0][:repeats],
        "rns_gemm": [88.0, 89.0][:repeats],
        "crt_export_status_memset": [0.5, 0.5][:repeats],
        "crt_export_kernel": [24.0, 25.0][:repeats],
        "crt_export_status_d2h": [1.0, 1.0][:repeats],
        "crt_export_d2h": [4.0, 4.5][:repeats],
        "crt_export": [29.5, 31.0][:repeats],
    }
    skinny["gpu_event_timings_us"] = event_values
    skinny["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    skinny["raw_timings_us"]["pack"] = [105, 110][:repeats]
    skinny["raw_timings_us"]["rns_gemm"] = [120, 123][:repeats]
    skinny["raw_timings_us"]["crt_export"] = [58, 60][:repeats]
    skinny["raw_timings_us"]["end_to_end"] = [305, 315][:repeats]
    skinny["timing_summary_us"]["pack"] = summary(skinny["raw_timings_us"]["pack"])
    skinny["timing_summary_us"]["rns_gemm"] = summary(skinny["raw_timings_us"]["rns_gemm"])
    skinny["timing_summary_us"]["crt_export"] = summary(skinny["raw_timings_us"]["crt_export"])
    skinny["timing_summary_us"]["end_to_end"] = summary(skinny["raw_timings_us"]["end_to_end"])
    skinny["avg_pack_us"] = skinny["timing_summary_us"]["pack"]["avg"]
    skinny["avg_rns_gemm_us"] = skinny["timing_summary_us"]["rns_gemm"]["avg"]
    skinny["avg_crt_export_us"] = skinny["timing_summary_us"]["crt_export"]["avg"]
    skinny["avg_end_to_end_us"] = skinny["timing_summary_us"]["end_to_end"]["avg"]
    skinny["avg_per_modulus_gemm_estimate_us"] = skinny["avg_rns_gemm_us"] / skinny["prefix"]
    skinny["comparison_baseline"]["required_before_speedup_claim"] = [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_vector_alu_int64",
        "same_contract_direct_hip_tiled",
    ]
    skinny["comparison_baseline"]["notes"] = (
        "skinny N=1 direct-HIP GEMV evidence must beat the same-contract tiled direct-HIP route"
    )
    return skinny


def as_direct_hip_bounded_skinny_gemv_small_n_capture(capture: dict, *, n: int = 4) -> dict:
    if n <= 1 or n > 4:
        raise ValueError("small-N skinny GEMV test fixture requires 2 <= n <= 4")
    skinny = as_direct_hip_bounded_skinny_gemv_n1_capture(capture)
    semantics = skinny["semantics"]
    signed = semantics == "bounded_i64"
    kernel = (
        "direct_hip_prefix9_rns_gemv_small_n_i64_v1"
        if signed
        else "direct_hip_prefix9_rns_gemv_small_n_u64_v1"
    )
    epilogue = "resident_rns_gemv_small_n_centered_residue_then_crt_export"
    old_kernel = skinny["selected_kernel"]
    old_epilogue = skinny["backend_metadata"]["epilogue_mode"]
    old_event = "rns_gemv_n1_kernel_group"
    new_event = "rns_gemv_small_n_kernel_group"

    skinny["benchmark"] = "rns8_bounded_gemm_direct_hip_skinny_gemv_small_n"
    skinny["benchmark_execution_mode"] = "direct_hip_skinny_gemv_small_n_resident_rns"
    skinny["m"] = 512
    skinny["n"] = n
    skinny["k"] = 512
    skinny["output_logical_ld"] = n
    skinny["output_ld"] = n
    skinny["bound"] = 131072
    skinny["selected_kernel"] = kernel
    skinny["k_block_size"] = 512
    skinny["schedule_metadata"]["tile_rows"] = 4
    skinny["schedule_metadata"]["tile_cols"] = 1
    skinny["schedule_metadata"]["tile_count"] = 4
    skinny["schedule_metadata"]["range_bit_length"] = 18

    add_output_policy_fields(skinny)
    skinny["backend_metadata"]["source"] = "rns8_bench_skinny_gemv_small_n_path"
    skinny["backend_metadata"]["selected_kernel"] = kernel
    skinny["backend_metadata"]["epilogue_mode"] = epilogue
    skinny["backend_metadata"]["workspace_mode"] = "resident_rns_inputs_skinny_n_le4_output"
    skinny["backend_metadata"]["accumulator_safety"]["k_block_size"] = skinny["k"]
    skinny["backend_metadata"]["autotune_key"] = (
        skinny["backend_metadata"]["autotune_key"]
        .replace(";m=256;", ";m=512;")
        .replace(";n=1;", f";n={n};")
        .replace(";k=4096;", ";k=512;")
        .replace(";bound=1048576;", ";bound=131072;")
        .replace("execution=direct_hip_skinny_gemv_n1_resident_rns", "execution=direct_hip_skinny_gemv_small_n_resident_rns")
        .replace(f"kernel={old_kernel}", f"kernel={kernel}")
        .replace(f"epilogue={old_epilogue}", f"epilogue={epilogue}")
        .replace("k_block_size=4096;", "k_block_size=512;")
    )

    skinny["timing_metadata"]["benchmark_execution_mode"] = "direct_hip_skinny_gemv_small_n_resident_rns"
    skinny["timing_metadata"]["gpu_event_phase_order"] = [
        new_event if phase == old_event else phase
        for phase in skinny["timing_metadata"]["gpu_event_phase_order"]
    ]
    skinny["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "HIP events around the skinny small-N resident-RNS GEMV kernel group"
    )
    skinny["gpu_event_timings_us"][new_event] = skinny["gpu_event_timings_us"].pop(old_event)
    skinny["gpu_event_timing_summary_us"][new_event] = skinny["gpu_event_timing_summary_us"].pop(old_event)
    skinny["raw_timings_us"]["rns_gemm"] = [130, 133][: skinny["repeats"]]
    skinny["raw_timings_us"]["end_to_end"] = [325, 335][: skinny["repeats"]]
    skinny["timing_summary_us"]["rns_gemm"] = summary(skinny["raw_timings_us"]["rns_gemm"])
    skinny["timing_summary_us"]["end_to_end"] = summary(skinny["raw_timings_us"]["end_to_end"])
    skinny["avg_rns_gemm_us"] = skinny["timing_summary_us"]["rns_gemm"]["avg"]
    skinny["avg_end_to_end_us"] = skinny["timing_summary_us"]["end_to_end"]["avg"]
    skinny["avg_per_modulus_gemm_estimate_us"] = skinny["avg_rns_gemm_us"] / skinny["prefix"]
    skinny["comparison_baseline"]["notes"] = (
        "skinny small-N direct-HIP GEMV evidence must beat the same-contract tiled direct-HIP route"
    )
    return skinny


