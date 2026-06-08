reused_ck_i64 = as_reused_pack_capture(v4_ck_i64)
validate_capture(reused_ck_i64)
reused_a_ck_i64 = as_reused_a_capture(v4_ck_i64)
validate_capture(reused_a_ck_i64)
reused_hipblaslt_i64 = as_hipblaslt_reused_ab_capture(v4_hipblaslt_i64)
validate_capture(reused_hipblaslt_i64)
direct_hip_oneshot_i64 = as_direct_hip_oneshot_capture(v4_ck_i64)
validate_capture(direct_hip_oneshot_i64)
direct_hip_finite_oneshot = as_direct_hip_finite_oneshot_capture(v4_finite_ring_ck)
validate_capture(direct_hip_finite_oneshot)
direct_hip_finite_native_a_reuse_b = as_direct_hip_finite_native_a_reuse_b_capture(v4_finite_ring_ck)
validate_capture(direct_hip_finite_native_a_reuse_b)
stale_ck_finite_mod255 = copy.deepcopy(v4_finite_ring_ck)
stale_ck_finite_mod255["selected_kernel"] = "ck_wmma_cshuffle_finite_u8_static_modulus_centered_epilogue_v2"
stale_ck_finite_mod255["backend_metadata"]["selected_kernel"] = "ck_wmma_cshuffle_finite_u8_static_modulus_centered_epilogue_v2"
stale_ck_finite_mod255["backend_metadata"]["autotune_key"] = stale_ck_finite_mod255["backend_metadata"][
    "autotune_key"
].replace(
    "kernel=ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2",
    "kernel=ck_wmma_cshuffle_finite_u8_static_modulus_centered_epilogue_v2",
)
expect_invalid(
    stale_ck_finite_mod255,
    "CK finite-u8 modulus 255 captures must use selected_kernel",
)
stale_rocwmma_finite_mod251 = copy.deepcopy(v4_finite_field_rocwmma)
stale_rocwmma_finite_mod251["selected_kernel"] = "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1"
stale_rocwmma_finite_mod251["backend_metadata"][
    "selected_kernel"
] = "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1"
stale_rocwmma_finite_mod251["backend_metadata"]["autotune_key"] = stale_rocwmma_finite_mod251[
    "backend_metadata"
]["autotune_key"].replace(
    "kernel=rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2",
    "kernel=rocwmma_i8_i32_signed_finite_u8_hot_residue_v1",
)
expect_invalid(
    stale_rocwmma_finite_mod251,
    "rocWMMA finite-u8 modulus 251 captures must use selected_kernel",
)
direct_hip_bounded_native_a_reuse_b = as_direct_hip_bounded_native_a_reuse_b_capture(v4_ck_i64)
validate_capture(direct_hip_bounded_native_a_reuse_b)
direct_hip_bounded_uniform_small_transient = as_direct_hip_bounded_uniform_small_transient_capture(v4_ck_i64)
validate_capture(direct_hip_bounded_uniform_small_transient)
direct_hip_residue_channel_fusion = as_direct_hip_bounded_residue_channel_fusion_capture(v4_ck_i64)
validate_capture(direct_hip_residue_channel_fusion)

stale_fusion_pack_layout = copy.deepcopy(direct_hip_residue_channel_fusion)
stale_fusion_pack_layout["timing_metadata"]["pack_layout"] = "native_i8_row_major_uniform_small"
expect_invalid(stale_fusion_pack_layout, "pack_layout=native_i8_row_major_residue_channel_width3")

stale_fusion_execution = copy.deepcopy(direct_hip_residue_channel_fusion)
stale_fusion_execution["backend_metadata"]["autotune_key"] = stale_fusion_execution["backend_metadata"][
    "autotune_key"
].replace(
    "execution=residue_channel_fusion_native_inputs",
    "execution=transient_uniform_small_i8_ab_inputs",
)
expect_invalid(stale_fusion_execution, "execution=residue_channel_fusion_native_inputs")

stale_transient_kernel = copy.deepcopy(direct_hip_bounded_uniform_small_transient)
stale_transient_kernel_name = "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_b_grouped_rns_gemm_v2"
stale_transient_kernel["selected_kernel"] = stale_transient_kernel_name
stale_transient_kernel["backend_metadata"]["selected_kernel"] = stale_transient_kernel_name
stale_transient_kernel["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
    (
    "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
    "input_profile=uniform-small;"
    "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
    "execution=transient_uniform_small_i8_ab_inputs;"
    f"kernel={stale_transient_kernel_name};epilogue=uniform_small_i8_ab_transient_residue_then_crt_export"
    ),
    stale_transient_kernel,
)
expect_invalid(
    stale_transient_kernel,
    "direct-HIP bounded uniform-small transient captures must use selected_kernel",
)
stale_transient_phase = copy.deepcopy(direct_hip_bounded_uniform_small_transient)
stale_transient_phase["timing_metadata"]["gpu_event_phase_order"] = [
    "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    if phase == "bounded_uniform_small_i8_ab_transient_gemm_kernel_group"
    else phase
    for phase in stale_transient_phase["timing_metadata"]["gpu_event_phase_order"]
]
stale_transient_phase["gpu_event_timings_us"][
    "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
] = stale_transient_phase["gpu_event_timings_us"].pop(
    "bounded_uniform_small_i8_ab_transient_gemm_kernel_group"
)
stale_transient_phase["gpu_event_timing_summary_us"][
    "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
] = stale_transient_phase["gpu_event_timing_summary_us"].pop(
    "bounded_uniform_small_i8_ab_transient_gemm_kernel_group"
)
expect_invalid(
    stale_transient_phase,
    "direct-HIP bounded uniform-small transient GPU event phase set is incomplete",
)
direct_hip_bounded_uniform_small_reuse_a = as_direct_hip_bounded_uniform_small_reuse_a_capture(v4_ck_i64)
validate_capture(direct_hip_bounded_uniform_small_reuse_a)
direct_hip_bounded_native_b_reuse_a = as_direct_hip_bounded_native_b_reuse_a_capture(v4_ck_i64)
validate_capture(direct_hip_bounded_native_b_reuse_a)
stale_native_b_reuse_a_kernel = copy.deepcopy(direct_hip_bounded_native_b_reuse_a)
stale_native_b_kernel = "direct_hip_native_a_u64_colpair_prefix9_reuse_b_grouped_rns_gemm_v2"
stale_native_b_reuse_a_kernel["selected_kernel"] = stale_native_b_kernel
stale_native_b_reuse_a_kernel["backend_metadata"]["selected_kernel"] = stale_native_b_kernel
stale_native_b_reuse_a_kernel["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
    (
    "backend=hip-direct;semantics=bounded_u64;m=512;n=512;k=512;bound=16384;"
    "input_profile=adaptive-bands;"
    "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
    "execution=transient_native_b_resident_a_reuse;"
    f"kernel={stale_native_b_kernel};epilogue=resident_a_native_b_centered_residue_then_crt_export"
    ),
    stale_native_b_reuse_a_kernel,
)
expect_invalid(
    stale_native_b_reuse_a_kernel,
    "direct-HIP bounded native-B reuse-A captures must use selected_kernel",
)
stale_native_b_reuse_a_phase = copy.deepcopy(direct_hip_bounded_native_b_reuse_a)
stale_native_b_reuse_a_phase["timing_metadata"]["gpu_event_phase_order"] = [
    "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
    if phase == "bounded_native_b_colpair_reuse_a_gemm_kernel_group"
    else phase
    for phase in stale_native_b_reuse_a_phase["timing_metadata"]["gpu_event_phase_order"]
]
stale_native_b_reuse_a_phase["gpu_event_timings_us"]["bounded_native_a_colpair_reuse_b_gemm_kernel_group"] = (
    stale_native_b_reuse_a_phase["gpu_event_timings_us"].pop(
        "bounded_native_b_colpair_reuse_a_gemm_kernel_group"
    )
)
stale_native_b_reuse_a_phase["gpu_event_timing_summary_us"][
    "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
] = stale_native_b_reuse_a_phase["gpu_event_timing_summary_us"].pop(
    "bounded_native_b_colpair_reuse_a_gemm_kernel_group"
)
expect_invalid(
    stale_native_b_reuse_a_phase,
    "direct-HIP bounded native-B reuse-A GPU event phase set is incomplete",
)
adaptive_direct_hip_bounded_native_a = copy.deepcopy(direct_hip_bounded_native_a_reuse_b)
centered_kernel = "direct_hip_native_a_i64_prefix9_reuse_b_grouped_rns_gemm_v1"
centered_epilogue = "native_a_centered_resident_b_residue_then_crt_export"
adaptive_direct_hip_bounded_native_a["input_distribution"] = "signed_adaptive_bands_-16_16"
adaptive_direct_hip_bounded_native_a["benchmark_execution_mode"] = "transient_native_a_resident_b_reuse"
adaptive_direct_hip_bounded_native_a["selected_kernel"] = centered_kernel
adaptive_direct_hip_bounded_native_a["backend_metadata"]["source"] = "rns8_bench_native_a_reuse_b_path"
adaptive_direct_hip_bounded_native_a["backend_metadata"]["selected_kernel"] = centered_kernel
adaptive_direct_hip_bounded_native_a["backend_metadata"]["epilogue_mode"] = centered_epilogue
adaptive_direct_hip_bounded_native_a["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
    (
    "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
    "input_profile=adaptive-bands;"
    "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
    "execution=transient_native_a_resident_b_reuse;"
    f"kernel={centered_kernel};epilogue={centered_epilogue}"
    ),
    adaptive_direct_hip_bounded_native_a,
)
adaptive_direct_hip_bounded_native_a["timing_metadata"][
    "benchmark_execution_mode"
] = "transient_native_a_resident_b_reuse"
adaptive_direct_hip_bounded_native_a["timing_metadata"]["gpu_event_phase_order"] = [
    "bounded_native_a_reuse_b_gemm_kernel_group"
    if phase == "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    else phase
    for phase in adaptive_direct_hip_bounded_native_a["timing_metadata"]["gpu_event_phase_order"]
]
adaptive_direct_hip_bounded_native_a["gpu_event_timings_us"]["bounded_native_a_reuse_b_gemm_kernel_group"] = (
    adaptive_direct_hip_bounded_native_a["gpu_event_timings_us"].pop(
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    )
)
adaptive_direct_hip_bounded_native_a["gpu_event_timing_summary_us"][
    "bounded_native_a_reuse_b_gemm_kernel_group"
] = adaptive_direct_hip_bounded_native_a["gpu_event_timing_summary_us"].pop(
    "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
)
validate_capture(adaptive_direct_hip_bounded_native_a)

large_u64_colpair_native_a = copy.deepcopy(adaptive_direct_hip_bounded_native_a)
large_u64_colpair_kernel = "direct_hip_native_a_u64_colpair_prefix9_reuse_b_grouped_rns_gemm_v2"
large_u64_colpair_native_a["semantics"] = "bounded_u64"
large_u64_colpair_native_a["bound_kind"] = "global_max_unsigned"
large_u64_colpair_native_a["input_distribution"] = "unsigned_adaptive_bands_0_16"
large_u64_colpair_native_a["m"] = 512
large_u64_colpair_native_a["n"] = 512
large_u64_colpair_native_a["k"] = 512
large_u64_colpair_native_a["selected_kernel"] = large_u64_colpair_kernel
large_u64_colpair_native_a["backend_metadata"]["selected_kernel"] = large_u64_colpair_kernel
apply_int32_accumulator_contract(large_u64_colpair_native_a)
large_u64_colpair_native_a["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
    (
    "backend=hip-direct;semantics=bounded_u64;m=512;n=512;k=512;bound=16384;"
    "input_profile=adaptive-bands;"
    "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
    "execution=transient_native_a_resident_b_reuse;"
    f"kernel={large_u64_colpair_kernel};epilogue={centered_epilogue}"
    ),
    large_u64_colpair_native_a,
)
large_u64_colpair_native_a["timing_metadata"]["gpu_event_phase_order"] = [
    "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
    if phase == "bounded_native_a_reuse_b_gemm_kernel_group"
    else phase
    for phase in large_u64_colpair_native_a["timing_metadata"]["gpu_event_phase_order"]
]
large_u64_colpair_native_a["gpu_event_timings_us"][
    "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
] = large_u64_colpair_native_a["gpu_event_timings_us"].pop("bounded_native_a_reuse_b_gemm_kernel_group")
large_u64_colpair_native_a["gpu_event_timing_summary_us"][
    "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
] = large_u64_colpair_native_a["gpu_event_timing_summary_us"].pop("bounded_native_a_reuse_b_gemm_kernel_group")
validate_capture(large_u64_colpair_native_a)

generic_persistent_reuse_b = copy.deepcopy(direct_hip_bounded_native_a_reuse_b)
generic_kernel = "direct_hip_tiled_active_prefix_rns_gemm_v2"
generic_epilogue = "fused_centered_residue_then_crt_export"
generic_persistent_reuse_b["benchmark_execution_mode"] = "persistent_resident_matrices"
generic_persistent_reuse_b["selected_kernel"] = generic_kernel
generic_persistent_reuse_b["backend_metadata"]["source"] = "rns8_get_plan_backend_info"
generic_persistent_reuse_b["backend_metadata"]["selected_kernel"] = generic_kernel
generic_persistent_reuse_b["backend_metadata"]["epilogue_mode"] = generic_epilogue
generic_persistent_reuse_b["backend_metadata"]["workspace_mode"] = "resident_device_buffers"
generic_persistent_reuse_b["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
    (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=persistent_resident_matrices;"
        f"kernel={generic_kernel};epilogue={generic_epilogue}"
    ),
    generic_persistent_reuse_b,
)
generic_persistent_reuse_b["timing_metadata"]["benchmark_execution_mode"] = "persistent_resident_matrices"
generic_persistent_reuse_b["timing_metadata"]["gpu_event_phase_order"] = [
    "rns_gemm_kernel_group"
    if phase == "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    else phase
    for phase in generic_persistent_reuse_b["timing_metadata"]["gpu_event_phase_order"]
]
generic_persistent_reuse_b["gpu_event_timings_us"]["rns_gemm_kernel_group"] = (
    generic_persistent_reuse_b["gpu_event_timings_us"].pop(
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    )
)
generic_persistent_reuse_b["gpu_event_timing_summary_us"]["rns_gemm_kernel_group"] = (
    generic_persistent_reuse_b["gpu_event_timing_summary_us"].pop(
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    )
)
validate_capture(generic_persistent_reuse_b)

stale_generic_persistent_reuse_b_mode = copy.deepcopy(generic_persistent_reuse_b)
stale_generic_persistent_reuse_b_mode["benchmark_execution_mode"] = (
    "transient_uniform_small_i8_a_resident_i8_b_reuse"
)
stale_generic_persistent_reuse_b_mode["timing_metadata"]["benchmark_execution_mode"] = (
    "transient_uniform_small_i8_a_resident_i8_b_reuse"
)
expect_invalid(
    stale_generic_persistent_reuse_b_mode,
    "direct-HIP bounded native-A reuse-B captures must use selected_kernel",
)

stale_large_u64_colpair_kernel = copy.deepcopy(large_u64_colpair_native_a)
stale_large_u64_kernel = "direct_hip_native_a_u64_prefix9_reuse_b_grouped_rns_gemm_v1"
stale_large_u64_colpair_kernel["selected_kernel"] = stale_large_u64_kernel
stale_large_u64_colpair_kernel["backend_metadata"]["selected_kernel"] = stale_large_u64_kernel
stale_large_u64_colpair_kernel["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
    (
    "backend=hip-direct;semantics=bounded_u64;m=512;n=512;k=512;bound=16384;"
    "input_profile=adaptive-bands;"
    "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
    "execution=transient_native_a_resident_b_reuse;"
    f"kernel={stale_large_u64_kernel};epilogue={centered_epilogue}"
    ),
    stale_large_u64_colpair_kernel,
)
expect_invalid(
    stale_large_u64_colpair_kernel,
    "direct-HIP bounded native-A reuse-B captures must use selected_kernel",
)

stale_large_u64_colpair_phase = copy.deepcopy(large_u64_colpair_native_a)
stale_large_u64_colpair_phase["timing_metadata"]["gpu_event_phase_order"] = [
    "bounded_native_a_reuse_b_gemm_kernel_group"
    if phase == "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
    else phase
    for phase in stale_large_u64_colpair_phase["timing_metadata"]["gpu_event_phase_order"]
]
stale_large_u64_colpair_phase["gpu_event_timings_us"]["bounded_native_a_reuse_b_gemm_kernel_group"] = (
    stale_large_u64_colpair_phase["gpu_event_timings_us"].pop(
        "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
    )
)
stale_large_u64_colpair_phase["gpu_event_timing_summary_us"]["bounded_native_a_reuse_b_gemm_kernel_group"] = (
    stale_large_u64_colpair_phase["gpu_event_timing_summary_us"].pop(
        "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
    )
)
expect_invalid(
    stale_large_u64_colpair_phase,
    "direct-HIP bounded native-A reuse-B GPU event phase set is incomplete",
)

bad_bounded_native_a_phase = copy.deepcopy(direct_hip_bounded_native_a_reuse_b)
bad_bounded_native_a_phase["timing_metadata"]["gpu_event_phase_order"] = [
    "rns_gemm_kernel_group" if phase == "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group" else phase
    for phase in bad_bounded_native_a_phase["timing_metadata"]["gpu_event_phase_order"]
]
bad_bounded_native_a_phase["gpu_event_timings_us"]["rns_gemm_kernel_group"] = (
    bad_bounded_native_a_phase["gpu_event_timings_us"].pop(
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    )
)
bad_bounded_native_a_phase["gpu_event_timing_summary_us"]["rns_gemm_kernel_group"] = (
    bad_bounded_native_a_phase["gpu_event_timing_summary_us"].pop(
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    )
)
expect_invalid(
    bad_bounded_native_a_phase,
    "direct-HIP bounded native-A reuse-B GPU event phase set is incomplete",
)
bad_bounded_uniform_small_reuse_a_phase = copy.deepcopy(direct_hip_bounded_uniform_small_reuse_a)
bad_bounded_uniform_small_reuse_a_phase["timing_metadata"]["gpu_event_phase_order"] = [
    "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    if phase == "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group"
    else phase
    for phase in bad_bounded_uniform_small_reuse_a_phase["timing_metadata"]["gpu_event_phase_order"]
]
bad_bounded_uniform_small_reuse_a_phase["gpu_event_timings_us"][
    "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
] = bad_bounded_uniform_small_reuse_a_phase["gpu_event_timings_us"].pop(
    "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group"
)
bad_bounded_uniform_small_reuse_a_phase["gpu_event_timing_summary_us"][
    "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
] = bad_bounded_uniform_small_reuse_a_phase["gpu_event_timing_summary_us"].pop(
    "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group"
)
expect_invalid(
    bad_bounded_uniform_small_reuse_a_phase,
    "direct-HIP bounded uniform-small reuse-A GPU event phase set is incomplete",
)
stale_generic_bounded_native_a = copy.deepcopy(direct_hip_bounded_native_a_reuse_b)
stale_kernel = "direct_hip_native_a_i64_prefix9_reuse_b_grouped_rns_gemm_v1"
stale_epilogue = "native_a_centered_resident_b_residue_then_crt_export"
stale_generic_bounded_native_a["selected_kernel"] = stale_kernel
stale_generic_bounded_native_a["backend_metadata"]["selected_kernel"] = stale_kernel
stale_generic_bounded_native_a["backend_metadata"]["epilogue_mode"] = stale_epilogue
stale_generic_bounded_native_a["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
    (
    "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
    "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
    "execution=transient_native_a_resident_b_reuse;"
    f"kernel={stale_kernel};epilogue={stale_epilogue}"
    ),
    stale_generic_bounded_native_a,
)
expect_invalid(
    stale_generic_bounded_native_a,
    "direct-HIP bounded native-A reuse-B captures must use selected_kernel",
)
stale_v1_uniform_small_bounded_native_a = copy.deepcopy(direct_hip_bounded_native_a_reuse_b)
stale_v1_kernel = "direct_hip_uniform_small_i8_ab_prefix9_reuse_b_grouped_rns_gemm_v1"
stale_v1_uniform_small_bounded_native_a["selected_kernel"] = stale_v1_kernel
stale_v1_uniform_small_bounded_native_a["backend_metadata"]["selected_kernel"] = stale_v1_kernel
stale_v1_uniform_small_bounded_native_a["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
    (
    "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
    "input_profile=uniform-small;"
    "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
    "execution=transient_uniform_small_i8_a_resident_i8_b_reuse;"
    f"kernel={stale_v1_kernel};epilogue=uniform_small_i8_ab_resident_b_residue_then_crt_export"
    ),
    stale_v1_uniform_small_bounded_native_a,
)
expect_invalid(
    stale_v1_uniform_small_bounded_native_a,
    "direct-HIP bounded native-A reuse-B captures must use selected_kernel",
)

exact_wide_ck = as_exact_wide_capture(v4_ck_i64)
validate_capture(exact_wide_ck)
exact_wide_no_status = copy.deepcopy(exact_wide_ck)
exact_wide_no_status["exact_wide_export_status_check"] = "elided_full_width_device_reconstruction"
exact_wide_no_status["gpu_event_timings_us"]["exact_wide_export_status_memset"] = [
    0.0 for _ in range(exact_wide_no_status["repeats"])
]
exact_wide_no_status["gpu_event_timings_us"]["exact_wide_export_status_d2h"] = [
    0.0 for _ in range(exact_wide_no_status["repeats"])
]
exact_wide_no_status["gpu_event_timing_summary_us"]["exact_wide_export_status_memset"] = zero_summary()
exact_wide_no_status["gpu_event_timing_summary_us"]["exact_wide_export_status_d2h"] = zero_summary()
validate_capture(exact_wide_no_status)
grouped_exact_wide_device_export = as_grouped_dispatch_capture(exact_wide_no_status)
grouped_exact_wide_device_export["grouped_dispatch"][
    "execution_strategy"
] = "device_grouped_exact_wide_export_kernel_batched_d2h"
grouped_exact_wide_device_export["grouped_dispatch"]["task_descriptor_contract"][
    "device_descriptor_policy"
] = "device_pointer_tables_and_compact_slabs"
grouped_exact_wide_device_export["grouped_dispatch"]["batched_export_enabled"] = True
grouped_exact_wide_device_export["grouped_dispatch"]["device_output_slab_bytes"] = 786432
grouped_exact_wide_device_export["timing_metadata"][
    "grouped_dispatch_execution_strategy"
] = "device_grouped_exact_wide_export_kernel_batched_d2h"
grouped_exact_wide_device_export["timing_metadata"]["grouped_dispatch_batched_export_enabled"] = True
grouped_exact_wide_device_export["timing_metadata"]["grouped_dispatch_device_output_slab_bytes"] = 786432
validate_capture(grouped_exact_wide_device_export)
grouped_exact_wide_device_pack_export = copy.deepcopy(grouped_exact_wide_device_export)
grouped_exact_wide_device_pack_export["grouped_dispatch"][
    "execution_strategy"
] = "device_grouped_pack_and_exact_wide_export_kernels_batched_d2h"
grouped_exact_wide_device_pack_export["timing_metadata"][
    "grouped_dispatch_execution_strategy"
] = "device_grouped_pack_and_exact_wide_export_kernels_batched_d2h"
validate_capture(grouped_exact_wide_device_pack_export)
grouped_exact_wide_device_pack_gemm_export = copy.deepcopy(grouped_exact_wide_device_export)
grouped_exact_wide_device_pack_gemm_export["grouped_dispatch"][
    "execution_strategy"
] = "device_grouped_pack_gemm_and_exact_wide_export_kernels_batched_d2h"
grouped_exact_wide_device_pack_gemm_export["timing_metadata"][
    "grouped_dispatch_execution_strategy"
] = "device_grouped_pack_gemm_and_exact_wide_export_kernels_batched_d2h"
validate_capture(grouped_exact_wide_device_pack_gemm_export)
grouped_exact_wide_missing_batched_flag = copy.deepcopy(grouped_exact_wide_device_export)
grouped_exact_wide_missing_batched_flag["grouped_dispatch"]["batched_export_enabled"] = False
grouped_exact_wide_missing_batched_flag["timing_metadata"]["grouped_dispatch_batched_export_enabled"] = False
expect_invalid(
    grouped_exact_wide_missing_batched_flag,
    "grouped_dispatch batched export strategy requires batched_export_enabled=true",
)
exact_wide_signed_3_limb = copy.deepcopy(exact_wide_no_status)
exact_wide_signed_3_limb["exact_wide_limb_count"] = 3
validate_capture(exact_wide_signed_3_limb)
exact_wide_unsigned_3_limb = copy.deepcopy(exact_wide_no_status)
exact_wide_unsigned_3_limb["semantics"] = "exact_wide_unsigned"
exact_wide_unsigned_3_limb["epilogue_type"] = "exact_wide_unsigned_limb_export"
exact_wide_unsigned_3_limb["exact_wide_limb_count"] = 3
exact_wide_unsigned_3_limb["backend_metadata"]["semantic_contract"] = "exact_wide_unsigned"
validate_capture(exact_wide_unsigned_3_limb)
exact_chain_ck = as_residue_current_chain_capture(v4_ck_i64)
validate_capture(exact_chain_ck)
exact_chain_final_export = as_residue_chain_final_export_capture(v4_ck_i64)
validate_capture(exact_chain_final_export)
bounded_chain_ck = as_bounded_residue_current_chain_capture(v4_ck_i64)
validate_capture(bounded_chain_ck)
bounded_chain_independent_final_export = as_bounded_residue_chain_independent_final_export_capture(v4_ck_i64)
validate_capture(bounded_chain_independent_final_export)
exact_chain_independent_final_export = as_exact_wide_residue_chain_independent_final_export_capture(v4_ck_i64)
validate_capture(exact_chain_independent_final_export)

bad_chain_missing_next_op = copy.deepcopy(exact_chain_ck)
del bad_chain_missing_next_op["requested_next_op"]
expect_invalid(bad_chain_missing_next_op, "residue-current chain captures must declare requested_next_op")

bad_chain_next_op = copy.deepcopy(exact_chain_ck)
bad_chain_next_op["requested_next_op"]["resolved"] = "final-export"
expect_invalid(bad_chain_next_op, "requested_next_op.resolved=rns-gemm")

bad_chain_output_policy = copy.deepcopy(exact_chain_ck)
bad_chain_output_policy["output_policy"]["per_repeat_logical_export"] = True
expect_invalid(bad_chain_output_policy, "output_policy.per_repeat_logical_export=false")

bad_exact_bound = copy.deepcopy(exact_wide_ck)
bad_exact_bound["bound_kind"] = "global_max_abs"
expect_invalid(bad_exact_bound, "exact-wide captures must use bound_kind=none and bound=0")

bad_exact_epilogue = copy.deepcopy(exact_wide_ck)
bad_exact_epilogue["epilogue_type"] = "crt_export"
expect_invalid(bad_exact_epilogue, "exact_wide_signed_limb_export")

bad_exact_limb_count = copy.deepcopy(exact_wide_ck)
bad_exact_limb_count["exact_wide_limb_count"] = 33
expect_invalid(bad_exact_limb_count, "exact_wide_limb_count in [1, 32]")

missing_exact_limb_count = copy.deepcopy(exact_wide_ck)
del missing_exact_limb_count["exact_wide_limb_count"]
expect_invalid(missing_exact_limb_count, "exact_wide_limb_count in [1, 32]")

bad_exact_backend_epilogue = copy.deepcopy(exact_wide_ck)
bad_exact_backend_epilogue["backend_metadata"]["epilogue_mode"] = "ck_fused_i32_to_centered_residue_then_crt_export"
expect_invalid(bad_exact_backend_epilogue, "ck_fused_i32_to_centered_residue_rns_output")

bad_exact_status_check = copy.deepcopy(exact_wide_no_status)
bad_exact_status_check["exact_wide_export_status_check"] = "required_for_range_check"
expect_invalid(bad_exact_status_check, "exact_wide_export_status_check")

bad_exact_signed_2_limb_no_status = copy.deepcopy(exact_wide_no_status)
bad_exact_signed_2_limb_no_status["exact_wide_limb_count"] = 2
expect_invalid(bad_exact_signed_2_limb_no_status, "exact_wide_export_status_check")

bad_exact_status_elision_events = copy.deepcopy(exact_wide_no_status)
bad_exact_status_elision_events["gpu_event_timings_us"]["exact_wide_export_status_d2h"][0] = 1.0
expect_invalid(bad_exact_status_elision_events, "status-elided captures")

bad_chain_export = copy.deepcopy(exact_chain_ck)
bad_chain_export["raw_timings_us"]["crt_export"][0] = 1
bad_chain_export["timing_summary_us"]["crt_export"]["avg"] = 1
bad_chain_export["avg_crt_export_us"] = 1
expect_invalid(bad_chain_export, "residue-current chain captures must report raw_timings_us.crt_export")

bad_chain_export_event = copy.deepcopy(exact_chain_ck)
bad_chain_export_event["timing_metadata"]["gpu_event_phase_order"].append("crt_export")
bad_chain_repeats = bad_chain_export_event["repeats"]
bad_chain_export_event["gpu_event_timings_us"]["crt_export"] = [1.0 for _ in range(bad_chain_repeats)]
bad_chain_export_event["gpu_event_timing_summary_us"]["crt_export"] = {
    "avg": 1.0,
    "median": 1.0,
    "p95": 1.0,
}
expect_invalid(bad_chain_export_event, "deep accelerator GPU event phase set contains undeclared phases")

bad_chain_mode = copy.deepcopy(exact_chain_ck)
bad_chain_mode["residue_output_mode"] = "host_export"
expect_invalid(bad_chain_mode, "residue_chain_final_export must match residue_output_mode")

bad_final_chain_next_op = copy.deepcopy(exact_chain_final_export)
bad_final_chain_next_op["requested_next_op"]["resolved"] = "rns-gemm"
expect_invalid(bad_final_chain_next_op, "requested_next_op.resolved=final-export")

bad_final_chain_output_policy = copy.deepcopy(exact_chain_final_export)
bad_final_chain_output_policy["output_policy"]["per_repeat_logical_export"] = False
expect_invalid(bad_final_chain_output_policy, "output_policy.per_repeat_logical_export=true")

bad_final_chain_mode = copy.deepcopy(exact_chain_final_export)
bad_final_chain_mode["benchmark_execution_mode"] = "persistent_resident_matrices"
bad_final_chain_mode["timing_metadata"]["benchmark_execution_mode"] = "persistent_resident_matrices"
expect_invalid(bad_final_chain_mode, "benchmark_execution_mode=residue_chain_final_host_export")

bad_independent_chain_non_rns = copy.deepcopy(exact_chain_independent_final_export)
bad_independent_chain_non_rns["semantics"] = "finite_ring_u8"
bad_independent_chain_non_rns["backend_metadata"]["semantic_contract"] = "finite_ring_u8"
expect_invalid(bad_independent_chain_non_rns, "residue_chain_length > 1 captures must use bounded or exact-wide")

bad_independent_chain_stale_mode = copy.deepcopy(bounded_chain_independent_final_export)
bad_independent_chain_stale_mode["residue_chain_independent_final_export"] = False
bad_independent_chain_stale_mode["timing_metadata"]["residue_chain_independent_final_export"] = False
expect_invalid(
    bad_independent_chain_stale_mode,
    "benchmark_execution_mode=residue_chain_independent_final_host_export requires",
)

bad_independent_chain_reuse = copy.deepcopy(bounded_chain_independent_final_export)
bad_independent_chain_reuse["reuse_packed_inputs"] = True
bad_independent_chain_reuse["pack_mode"] = "prepacked_reuse_b"
bad_independent_chain_reuse["prepack_reuse_operands"] = ["B"]
bad_independent_chain_reuse["prepack_reuse_strategy"] = "persistent_matrix_residency"
expect_invalid(bad_independent_chain_reuse, "must use pack_mode=per_repeat_repack")

bad_chain_shape = copy.deepcopy(exact_chain_ck)
bad_chain_shape["n"] = 128
expect_invalid(bad_chain_shape, "square m=n=k shapes")

bad_bounded_chain_vector = copy.deepcopy(bounded_chain_ck)
bad_bounded_chain_vector["backend_selected"] = "hip-vector-alu-int64"
bad_bounded_chain_vector["selected_kernel"] = "hip_vector_alu_i64_exact_192b_v1"
bad_bounded_chain_vector["backend_metadata"]["selected_kernel"] = "hip_vector_alu_i64_exact_192b_v1"
expect_invalid(bad_bounded_chain_vector, "must not select hip-vector-alu-int64")

bad_bounded_chain_bound_mode = copy.deepcopy(bounded_chain_ck)
bad_bounded_chain_bound_mode["bound_mode"] = "per_tile"
expect_invalid(bad_bounded_chain_bound_mode, "bounded residue chains must use bound_mode=global")

