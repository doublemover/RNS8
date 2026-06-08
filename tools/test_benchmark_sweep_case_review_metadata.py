from benchmark_sweep_lib.review import build_review_summary


ck = finite_capture("ck", 190)
direct = finite_capture("hip-direct", 300)
cpu = finite_capture("cpu-reference", 500)
for capture in (ck, direct, cpu):
    capture["tile_shape_variant"] = {"name": "finite-ring-u8-default-128x128", "tile_m": 128, "tile_n": 128, "tile_k": 256}
smoke_report = benchmark_sweep.review_captures([ck, direct, cpu])
assert smoke_report["schema_version"] == 3
assert smoke_report["review_mode"] == "smoke"
assert smoke_report["promotable_autotune_entries"] == []
assert "not_release_review" in smoke_report["groups"][0]["candidates"][0]["promotion_blockers"]

report = benchmark_sweep.review_captures([ck, direct, cpu], review_mode="release")
benchmark_sweep.attach_cache_write_status(report, False, Path("unused.json"), 0)
assert report["schema_version"] == 3
assert report["review_mode"] == "release"
assert report["group_count"] == 1
assert len(report["promotable_autotune_entries"]) == 1
assert report["promotable_autotune_entries"][0]["selected_backend"] == "ck"
assert report["summary"]["group_count"] == 1
assert report["summary"]["promotable_autotune_entry_count"] == 1
assert report["summary"]["missing_required_baseline_group_count"] == 0
assert report["summary"]["checksum_mismatch_group_count"] == 0
assert report["summary"]["fastest_production_route_counts"] == {"ck": 1}
assert report["summary"]["fastest_accelerator_route_counts"] == {"ck": 1}
assert report["summary"]["direct_hip_production_wins"] == []
assert report["summary"]["loss_phase_counts"] == {}
assert report["summary"]["loss_phase_by_backend"] == {}
assert report["summary"]["loss_phase_by_semantics"] == {}
assert report["summary"]["loss_phase_by_shape_family"] == {}
assert report["summary"]["loss_phase_by_scenario_family"] == {}
assert report["summary"]["next_work"] == []
group = report["groups"][0]
ck_candidate = next(item for item in group["candidates"] if item["backend"] == "ck")
assert ck_candidate["tile_shape_variant"] == "finite-ring-u8-default-128x128"
assert group["shape_family"] == "rectangular"
assert group["scenario_families"] == []
assert group["scenario_names"] == []
assert len(group["phase_ratio_summary"]) == 3
ck_phase_summary = next(item for item in group["phase_ratio_summary"] if item["backend"] == "ck")
assert ck_phase_summary["slowest_phase_vs_direct_hip"] is None
assert ck_phase_summary["phase_speedups_vs_direct_hip"]["end_to_end"] > 1.0
assert group["missing_required_baselines"] == []
assert group["release_review_satisfied"] is True
assert group["source_metadata"]["target_ids"] == ["gfx1100"]
assert group["source_metadata"]["configured_amdgpu_targets"] == ["gfx1100"]
assert group["source_metadata"]["hip_runtime_versions"] == ["70260201"]
assert group["source_metadata"]["hip_driver_versions"] == ["70260201"]
assert group["source_metadata"]["compilers"] == ["msvc 1944.194435227"]
assert group["source_metadata"]["git_commits"] == ["fixture"]
assert group["source_metadata"]["seeds"] == [13]
assert group["source_metadata"]["warmups"] == [benchmark_sweep.RELEASE_MIN_WARMUPS]
assert group["source_metadata"]["repeats"] == [benchmark_sweep.RELEASE_MIN_REPEATS]
assert group["missing_gpu_targets"] == []
assert group["gpu_target_identity_complete"] is True
assert group["gpu_target_compatible"] is True
assert group["missing_hip_toolchain_versions"] == []
assert group["hip_toolchain_version_complete"] is True
assert group["hip_toolchain_version_compatible"] is True
assert group["missing_configured_gpu_targets"] == []
assert group["configured_target_identity_complete"] is True
assert group["configured_target_compatible"] is True
assert group["missing_hip_runtime_versions"] == []
assert group["hip_runtime_version_complete"] is True
assert group["hip_runtime_version_compatible"] is True
assert group["checksum_reference_backend"] == "cpu-reference"
assert group["checksum_reference"] == 987654321
assert group["checksum_consistent"] is True
assert group["checksum_mismatches"] == []

amdgpu = finite_capture("ck", 100)
amdgpu_kernel = "amdgpu_builtin_rdna3_wmma_i32_16x16x16_iu8_finite_u8_epilogue_v1"
amdgpu["backend_requested"] = "amdgpu-builtins"
amdgpu["backend_selected"] = "amdgpu-builtins"
amdgpu["selected_kernel"] = amdgpu_kernel
amdgpu_metadata = amdgpu["backend_metadata"]
amdgpu_metadata["selected_kernel"] = amdgpu_kernel
amdgpu_metadata["accelerator_backend"] = True
amdgpu_metadata["correctness_backend"] = True
amdgpu_metadata["matrix_engine_backend"] = True
amdgpu_metadata["accelerator_library"] = "AMDGPU builtins"
amdgpu_metadata["accelerator_version"] = "compiled_target_specific"
amdgpu_metadata["capability_status"] = "implemented_opt_in_amdgpu_builtin_backend"
amdgpu_metadata["epilogue_mode"] = "amdgpu_builtin_fused_i32_to_centered_residue_then_canonical_u8_export"
amdgpu_metadata["workspace_mode"] = "resident_device_buffers_direct_amdgpu_builtin_matrix_core_no_dense_pack_workspace"
amdgpu_metadata["isa_evidence"] = "amdgpu_builtin_matrix_isa_gate_no_divide"
amdgpu_metadata["matrix_instruction_family"] = "wmma"
amdgpu_metadata["matrix_instruction_shape"] = "16x16x16"
amdgpu_metadata["matrix_instruction_dtype"] = "iu8"
amdgpu_metadata["matrix_instruction_sparsity"] = "dense"
amdgpu_metadata["autotune_key"] = with_accumulator_key_fields(
    amdgpu_metadata["autotune_key"]
    .replace("backend=ck", "backend=amdgpu-builtins")
    .replace("kernel=ck_wmma_cshuffle_finite_u8_static_modulus_centered_epilogue_v2", f"kernel={amdgpu_kernel}")
    .replace(
        "epilogue=ck_fused_centered_residue_then_canonical_u8_export",
        "epilogue=amdgpu_builtin_fused_i32_to_centered_residue_then_canonical_u8_export",
    ),
    amdgpu,
)
amdgpu_missing_isa_report = benchmark_sweep.review_captures([amdgpu, direct, cpu], review_mode="release")
amdgpu_missing_isa_group = amdgpu_missing_isa_report["groups"][0]
amdgpu_missing_isa_candidate = next(
    item for item in amdgpu_missing_isa_group["candidates"] if item["backend"] == "amdgpu-builtins"
)
assert amdgpu_missing_isa_report["promotable_autotune_entries"] == []
assert "missing_amdgpu_builtin_matrix_isa_histogram" in amdgpu_missing_isa_candidate["promotion_blockers"]
amdgpu_missing_isa_next_work = {row["work"] for row in amdgpu_missing_isa_report["summary"]["next_work"]}
assert "attach_compiled_matrix_isa_reports_before_builtin_promotion" in amdgpu_missing_isa_next_work

amdgpu_wrong_isa_index = {
    "amdgpu-builtins|gfx1100": [
        {"isa_matrix_instruction_histogram": {"v_wmma_i32_16x16x16_iu4": 4}},
    ],
}
amdgpu_wrong_isa_report = benchmark_sweep.review_captures(
    [amdgpu, direct, cpu],
    review_mode="release",
    isa_index=amdgpu_wrong_isa_index,
)
amdgpu_wrong_isa_group = amdgpu_wrong_isa_report["groups"][0]
amdgpu_wrong_isa_candidate = next(
    item for item in amdgpu_wrong_isa_group["candidates"] if item["backend"] == "amdgpu-builtins"
)
assert "missing_amdgpu_builtin_matrix_isa_histogram" not in amdgpu_wrong_isa_candidate["promotion_blockers"]
assert "missing_selected_amdgpu_builtin_matrix_instruction" in amdgpu_wrong_isa_candidate["promotion_blockers"]
assert amdgpu_wrong_isa_candidate["expected_matrix_instruction_mnemonic"] == "v_wmma_i32_16x16x16_iu8"
amdgpu_wrong_isa_next_work = {row["work"] for row in amdgpu_wrong_isa_report["summary"]["next_work"]}
assert "compile_selected_amdgpu_builtin_kernel_with_expected_matrix_instruction" in amdgpu_wrong_isa_next_work

amdgpu_isa_index = {
    "amdgpu-builtins|gfx1100": [
        {"isa_matrix_instruction_histogram": {"v_wmma_i32_16x16x16_iu8": 4}},
    ],
}
amdgpu_with_isa_report = benchmark_sweep.review_captures(
    [amdgpu, direct, cpu],
    review_mode="release",
    isa_index=amdgpu_isa_index,
)
amdgpu_with_isa_group = amdgpu_with_isa_report["groups"][0]
amdgpu_with_isa_candidate = next(
    item for item in amdgpu_with_isa_group["candidates"] if item["backend"] == "amdgpu-builtins"
)
assert "missing_amdgpu_builtin_matrix_isa_histogram" not in amdgpu_with_isa_candidate["promotion_blockers"]
assert "missing_selected_amdgpu_builtin_matrix_instruction" not in amdgpu_with_isa_candidate["promotion_blockers"]
assert amdgpu_with_isa_candidate["matrix_instruction_histogram"] == {"v_wmma_i32_16x16x16_iu8": 4}
assert amdgpu_with_isa_candidate["expected_matrix_instruction_mnemonic"] == "v_wmma_i32_16x16x16_iu8"
assert amdgpu_with_isa_report["promotable_autotune_entries"][0]["selected_backend"] == "amdgpu-builtins"

summary_fixture_group = {
    "semantics": "bounded_i64",
    "shape": {"m": 64, "n": 64, "k": 64},
    "shape_family": "small_square",
    "scenario_families": ["repeated-b"],
    "missing_required_baselines": ["hip-vector-alu-int64"],
    "checksum_mismatches": ["ck"],
    "fastest_production_route": {
        "backend": "hip-direct",
        "selected_kernel": "direct_kernel",
        "median_end_to_end_us": 100.0,
        "bottleneck": {"class": "pack_bound", "phase": "pack"},
        "capture": "direct.json",
    },
    "fastest_accelerator_route": {
        "backend": "ck",
        "selected_kernel": "ck_kernel",
        "median_end_to_end_us": 140.0,
        "bottleneck": {"class": "pack_bound", "phase": "pack"},
        "capture": "ck.json",
    },
    "candidates": [
        {
            "backend": "ck",
            "accelerator_backend": True,
            "scenario_promotion_scope": "release_review_candidate",
            "selected_kernel": "ck_kernel",
            "median_end_to_end_us": 140.0,
            "speedup_vs_direct_hip": 0.71,
            "speedup_vs_vector_alu": 0.8,
            "primary_loss_phase_vs_direct_hip": "pack",
            "bottleneck": {"class": "pack_bound", "phase": "pack"},
            "capture": "ck.json",
            "promotion_blockers": [
                "reuse_not_faster_than_same_backend_setup_inclusive",
                "reuse_not_faster_than_best_nonreuse_setup_inclusive",
                "graph_not_faster_than_non_graph_setup_inclusive",
                "missing_graph_setup_inclusive_timing",
                "not_faster_than_direct_hip",
                "not_faster_than_vector_alu",
            ],
            "prepacked_reuse_review": {
                "setup_inclusive_median_end_to_end_us": 180.0,
                "same_backend_nonreuse_median_end_to_end_us": 150.0,
            },
            "hip_graph_replay_review": {
                "setup_inclusive_median_end_to_end_us": None,
                "baseline_setup_inclusive_median_end_to_end_us": 120.0,
            },
        },
        {
            "backend": "hip-direct",
            "accelerator_backend": False,
            "scenario_promotion_scope": "release_review_candidate",
            "promotion_blockers": ["not_accelerator_backend"],
        },
        {
            "backend": "ck",
            "accelerator_backend": True,
            "scenario_promotion_scope": "proxy_evidence_only",
            "promotion_blockers": ["scenario_scope_not_autotune_promotable"],
        },
    ],
}
summary_fixture = build_review_summary([summary_fixture_group], [])
assert summary_fixture["missing_required_baseline_group_count"] == 1
assert summary_fixture["checksum_mismatch_group_count"] == 1
assert summary_fixture["review_blocker_counts"]["not_accelerator_backend"] == 1
assert summary_fixture["review_blocker_counts"]["scenario_scope_not_autotune_promotable"] == 1
assert summary_fixture["actionable_blocker_counts"]["not_faster_than_direct_hip"] == 1
assert summary_fixture["actionable_blocker_counts"]["not_faster_than_vector_alu"] == 1
assert summary_fixture["actionable_blocker_counts"]["graph_not_faster_than_non_graph_setup_inclusive"] == 1
assert summary_fixture["actionable_blocker_counts"]["missing_graph_setup_inclusive_timing"] == 1
assert summary_fixture["loss_phase_counts"] == {"pack": 1}
assert summary_fixture["loss_phase_by_backend"] == {"ck": {"pack": 1}}
assert summary_fixture["loss_phase_by_semantics"] == {"bounded_i64": {"pack": 1}}
assert summary_fixture["loss_phase_by_shape_family"] == {"small_square": {"pack": 1}}
assert summary_fixture["loss_phase_by_scenario_family"] == {"repeated-b": {"pack": 1}}
assert summary_fixture["fastest_production_route_counts"] == {"hip-direct": 1}
assert summary_fixture["fastest_accelerator_route_counts"] == {"ck": 1}
assert len(summary_fixture["direct_hip_production_wins"]) == 1
assert len(summary_fixture["setup_sensitive_candidates"]) == 1
next_work_items = {row["work"] for row in summary_fixture["next_work"]}
assert "fix_checksum_mismatches_before_performance_promotion" in next_work_items
assert "fix_missing_required_baselines_or_reclassify_invalid_scenarios" in next_work_items
assert "reduce_prepack_setup_or_reuse_steady_state_cost" in next_work_items
assert "reduce_prepack_setup_or_raise_declared_reuse_count_with_contract_evidence" in next_work_items
assert "improve_graph_replay_break_even_or_keep_graph_benchmark_only" in next_work_items
assert "fix_graph_setup_inclusive_timing_metadata" in next_work_items
assert "optimize_accelerator_loss_phase_or_keep_direct_hip_production_winner" in next_work_items
assert "specialize_native_vector_or_small_shape_path_before_matrix_engine_promotion" in next_work_items
assert "optimize_pack_phase" in next_work_items
assert "address_pack_bound" in next_work_items
assert "treat_direct_hip_as_current_production_winner_and_target_accelerator_loss_phases" in next_work_items

checksum_bad_ck = copy.deepcopy(ck)
checksum_bad_ck["checksum_u64"] = checksum_bad_ck["checksum_u64"] + 1
checksum_mismatch_report = benchmark_sweep.review_captures(
    [checksum_bad_ck, direct, cpu],
    review_mode="release",
)
checksum_mismatch_group = checksum_mismatch_report["groups"][0]
assert checksum_mismatch_report["promotable_autotune_entries"] == []
assert checksum_mismatch_group["checksum_reference_backend"] == "cpu-reference"
assert checksum_mismatch_group["checksum_consistent"] is False
assert checksum_mismatch_group["checksum_mismatches"] == ["ck"]
checksum_bad_candidate = next(item for item in checksum_mismatch_group["candidates"] if item["backend"] == "ck")
assert checksum_bad_candidate["checksum_matches_reference"] is False
assert "checksum_mismatch_vs_reference" in checksum_bad_candidate["promotion_blockers"]

bounded_ck = bounded_capture("ck", 700)
bounded_direct = bounded_capture("hip-direct", 300)
bounded_cpu = bounded_capture("cpu-reference", 5000)
bounded_vector = bounded_capture("hip-vector-alu-int64", 900)
bounded_ck["timing_metadata"]["pack_layout"] = "matrix_engine_transient_pack_layout"
bounded_ck["target_variant"] = {
    "target_id": "gfx1100",
    "target_namespace": "gfx1100",
    "review_group_key": "gfx1100/target=gfx1100/backend=ck",
}
bounded_direct["timing_metadata"]["generated_reducer_identity"] = (
    "direct_hip_fixed_prefix_2_generated_reducer_v1"
)
bounded_direct["target_variant"] = {
    "target_id": "gfx1100",
    "target_namespace": "gfx1100",
    "review_group_key": "gfx1100/target=gfx1100/backend=hip-direct",
}
bounded_cpu["target_variant"] = {
    "target_id": "cpu",
    "target_namespace": "cpu",
    "review_group_key": "cpu/target=cpu/backend=cpu-reference",
}
bounded_vector["benchmark_execution_mode"] = "public_runtime_vector_alu_native_buffers"
bounded_vector["requested_next_op"] = {
    "requested": "native-gemm",
    "resolved": "native-gemm",
    "source": "benchmark_default",
}
bounded_vector["timing_metadata"]["pack_layout"] = "native_i64_row_major"
bounded_vector["target_variant"] = {
    "target_id": "gfx1100",
    "target_namespace": "gfx1100",
    "review_group_key": "gfx1100/target=gfx1100/backend=hip-vector-alu-int64",
}
implementation_split_report = benchmark_sweep.review_captures(
    [bounded_ck, bounded_direct, bounded_cpu, bounded_vector],
    review_mode="release",
)
assert implementation_split_report["group_count"] == 1
implementation_split_group = implementation_split_report["groups"][0]
assert implementation_split_group["missing_required_baselines"] == []
assert {
    candidate["backend"] for candidate in implementation_split_group["candidates"]
} == {"ck", "cpu-reference", "hip-direct", "hip-vector-alu-int64"}

missing_export_metadata_ck = copy.deepcopy(bounded_ck)
missing_export_metadata_ck.pop("exact_output_contract", None)
missing_export_metadata_ck.pop("export_variant", None)
missing_export_metadata_ck.pop("reconstruction_variant", None)
missing_export_metadata_report = benchmark_sweep.review_captures(
    [missing_export_metadata_ck, bounded_direct, bounded_cpu, bounded_vector],
    review_mode="release",
)
missing_export_metadata_group = missing_export_metadata_report["groups"][0]
missing_export_metadata_candidate = next(
    item for item in missing_export_metadata_group["candidates"] if item["backend"] == "ck"
)
assert missing_export_metadata_report["promotable_autotune_entries"] == []
assert "missing_final_output_contract_metadata" in missing_export_metadata_candidate["promotion_blockers"]
assert "missing_export_variant_metadata" in missing_export_metadata_candidate["promotion_blockers"]
assert "missing_reconstruction_variant_metadata" in missing_export_metadata_candidate["promotion_blockers"]

bounded_chain_ck = bounded_capture("ck", 200)
bounded_chain_direct = bounded_capture("hip-direct", 450)
bounded_chain_cpu = bounded_capture("cpu-reference", 5000)
chain_metadata = {
    "family": "rns-chain",
    "name": "bounded-i64-chain3",
    "promotion_eligibility": "release_review_candidate",
    "output_domain": "residue_current_rns",
    "metadata": {
        "workflow_name": "rns_gemm_chain",
        "output_domain_requirement": "lazy_export",
    },
}
for item in [bounded_chain_ck, bounded_chain_direct, bounded_chain_cpu]:
    item["scenario_metadata"] = copy.deepcopy(chain_metadata)
    item["residue_chain_length"] = 3
    item["residue_output_mode"] = "residue_current_rns"
    item["requested_next_op"] = {
        "requested": "rns-gemm",
        "resolved": "rns-gemm",
        "source": "scenario_contract",
    }
chain_report = benchmark_sweep.review_captures(
    [bounded_chain_ck, bounded_chain_direct, bounded_chain_cpu],
    review_mode="release",
)
chain_group = chain_report["groups"][0]
chain_ck_candidate = next(item for item in chain_group["candidates"] if item["backend"] == "ck")
assert chain_group["required_baselines"] == ["cpu-reference", "hip-direct"]
assert chain_group["missing_required_baselines"] == []
assert "missing_required_baselines" not in chain_ck_candidate["promotion_blockers"]
assert "hip-vector-alu-int64" not in chain_group["required_baselines"]

eventless_ck = finite_capture("ck", 190)
remove_gpu_events(eventless_ck)
eventless_report = benchmark_sweep.review_captures([eventless_ck, direct, cpu], review_mode="release")
eventless_group = eventless_report["groups"][0]
assert eventless_report["promotable_autotune_entries"] == []
assert eventless_group["fastest_promotable"] is None
eventless_candidate = next(item for item in eventless_group["candidates"] if item["backend"] == "ck")
assert "missing_required_gpu_events" in eventless_candidate["promotion_blockers"]

cpu_faster = finite_capture("cpu-reference", 100)
cpu_faster_report = benchmark_sweep.review_captures([ck, direct, cpu_faster], review_mode="release")
cpu_faster_group = cpu_faster_report["groups"][0]
assert cpu_faster_report["promotable_autotune_entries"] == []
assert cpu_faster_group["fastest_promotable"] is None
cpu_faster_candidate = next(item for item in cpu_faster_group["candidates"] if item["backend"] == "ck")
assert "not_faster_than_cpu_reference" in cpu_faster_candidate["promotion_blockers"]

cpu_anchor = finite_capture("cpu-reference", 100)
cpu_anchor["warmups"] = 0
cpu_anchor["repeats"] = 1
cpu_anchor["cpu_parallel"] = {
    "reference_mode": "correctness-anchor",
    "correctness_anchor": True,
    "timed_cpu_baseline": False,
}
anchor_report = benchmark_sweep.review_captures([ck, direct, cpu_anchor], review_mode="release")
anchor_group = anchor_report["groups"][0]
anchor_ck_candidate = next(item for item in anchor_group["candidates"] if item["backend"] == "ck")
assert anchor_report["promotable_autotune_entries"][0]["selected_backend"] == "ck"
assert anchor_group["release_review_satisfied"] is True
assert "cpu-reference" not in anchor_group["warmup_counts"]
assert "cpu-reference" not in anchor_group["repeat_counts"]
assert "not_release_review" not in anchor_ck_candidate["promotion_blockers"]
assert "missing_warmup_count" not in anchor_ck_candidate["promotion_blockers"]
assert "repeat_count_mismatch" not in anchor_ck_candidate["promotion_blockers"]
assert "not_faster_than_cpu_reference" not in anchor_ck_candidate["promotion_blockers"]

reuse_baseline_metadata = {
    "family": "repeated-b",
    "name": "bounded-i64-512-production-baselines",
    "promotion_eligibility": "release_review_candidate",
    "output_domain": "host_export",
    "metadata": {
        "workflow_name": "repeated_b",
        "reuse_contract_role": "same_contract_nonreuse_baseline",
    },
}
reuse_candidate_metadata = copy.deepcopy(reuse_baseline_metadata)
reuse_candidate_metadata["name"] = "bounded-i64-512-production-reuse-b"
reuse_candidate_metadata["metadata"]["reuse_contract_role"] = "stable_b_production_candidate"

reuse_cpu = bounded_capture("cpu-reference", 5000)
reuse_direct = bounded_capture("hip-direct", 800)
reuse_vector = bounded_capture("hip-vector-alu-int64", 900)
reuse_rocwmma_baseline = bounded_capture("rocwmma", 900)
for item in [reuse_cpu, reuse_direct, reuse_vector, reuse_rocwmma_baseline]:
    item["scenario_metadata"] = copy.deepcopy(reuse_baseline_metadata)
reuse_rocwmma_candidate = copy.deepcopy(reuse_rocwmma_baseline)
reuse_rocwmma_candidate["_path"] = "rocwmma-reuse-b-production.json"
reuse_rocwmma_candidate["scenario_metadata"] = copy.deepcopy(reuse_candidate_metadata)
reuse_rocwmma_candidate["reuse_packed_inputs"] = True
reuse_rocwmma_candidate["pack_mode"] = "prepacked_reuse_b"
reuse_rocwmma_candidate["prepack_reuse_operands"] = ["B"]
reuse_rocwmma_candidate["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
reuse_rocwmma_candidate["prepack_setup_us"] = 90
reuse_rocwmma_candidate["avg_prepack_setup_us"] = 9000.0
reuse_rocwmma_candidate["timing_metadata"]["pack_mode"] = "prepacked_reuse_b"
reuse_rocwmma_candidate["timing_metadata"]["prepack_reuse_operands"] = ["B"]
reuse_rocwmma_candidate["timing_metadata"]["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
set_phase(reuse_rocwmma_candidate, 600)
reuse_rocwmma_candidate["reuse_contract"] = {
    "setup_cost_us": 90.0,
    "setup_amortized_us": 10.0,
    "repeat_median_end_to_end_us": 600.0,
    "setup_inclusive_median_end_to_end_us": 610.0,
}
reuse_report = benchmark_sweep.review_captures(
    [reuse_cpu, reuse_direct, reuse_vector, reuse_rocwmma_baseline, reuse_rocwmma_candidate],
    review_mode="release",
)
reuse_candidate_group = next(
    group
    for group in reuse_report["groups"]
    if any(candidate["capture"] == "rocwmma-reuse-b-production.json" for candidate in group["candidates"])
)
reuse_candidate = reuse_candidate_group["candidates"][0]
assert reuse_candidate["promotable"] is True
assert reuse_candidate["selection_end_to_end_us"] == 610.0
assert reuse_candidate["prepacked_reuse_review"]["prepack_setup_us"] == 90.0
assert reuse_candidate["prepacked_reuse_review"]["setup_inclusive_median_end_to_end_us"] == 610.0
assert reuse_candidate["prepacked_reuse_review"]["same_backend_nonreuse_median_end_to_end_us"] == 900.0
assert reuse_candidate["prepacked_reuse_review"]["best_nonreuse_backend"] == "hip-direct"
assert reuse_candidate["prepacked_reuse_review"]["blockers"] == []
assert "prepacked_reuse_not_autotune_promotable" not in reuse_candidate["promotion_blockers"]
assert reuse_report["promotable_autotune_entries"][0]["selected_backend"] == "rocwmma"
assert reuse_report["promotable_autotune_entries"][0]["selection_end_to_end_us"] == 610.0

slow_reuse_candidate = copy.deepcopy(reuse_rocwmma_candidate)
slow_reuse_candidate["_path"] = "rocwmma-reuse-b-slow.json"
slow_reuse_candidate["prepack_setup_us"] = 3600
slow_reuse_candidate["avg_prepack_setup_us"] = 3600.0
slow_reuse_candidate["reuse_contract"]["setup_cost_us"] = 3600.0
slow_reuse_candidate["reuse_contract"]["setup_amortized_us"] = 400.0
slow_reuse_candidate["reuse_contract"]["setup_inclusive_median_end_to_end_us"] = 1000.0
slow_reuse_report = benchmark_sweep.review_captures(
    [reuse_cpu, reuse_direct, reuse_vector, reuse_rocwmma_baseline, slow_reuse_candidate],
    review_mode="release",
)
slow_group = next(
    group
    for group in slow_reuse_report["groups"]
    if any(candidate["capture"] == "rocwmma-reuse-b-slow.json" for candidate in group["candidates"])
)
slow_candidate = slow_group["candidates"][0]
assert slow_candidate["promotable"] is False
assert "reuse_not_faster_than_best_nonreuse_setup_inclusive" in slow_candidate["promotion_blockers"]

graph_baseline = mark_reused_pack(bounded_capture("hip-direct", 1000))
graph_baseline["scenario_metadata"] = {
    "family": "hip-graph-replay",
    "name": "bounded-i64-full-pack-export-512-baseline",
    "promotion_eligibility": "release_review_candidate",
    "output_domain": "host_export",
    "metadata": {"workflow_name": "hip_graph_replay", "graph_role": "same_contract_non_graph_baseline"},
}
graph_candidate = copy.deepcopy(graph_baseline)
graph_candidate["_path"] = "hip-direct-graph.json"
graph_candidate["benchmark_execution_mode"] = "hip_graph_replay_bounded_pack_gemm_export"
graph_candidate["timing_metadata"]["benchmark_execution_mode"] = "hip_graph_replay_bounded_pack_gemm_export"
graph_candidate["scenario_metadata"] = copy.deepcopy(graph_baseline["scenario_metadata"])
graph_candidate["scenario_metadata"]["name"] = "bounded-i64-full-pack-export-512-graph"
graph_candidate["scenario_metadata"]["metadata"]["graph_role"] = "graph_replay_candidate"
graph_candidate["hip_graph_replay"] = {
    "requested": True,
    "used": True,
    "status": "available",
    "capture_status": "replayed",
    "capture_us": 10.0,
    "instantiate_us": 8.0,
}
set_phase(graph_candidate, 900)
graph_report = benchmark_sweep.review_captures([graph_baseline, graph_candidate], review_mode="release")
graph_group = next(
    group
    for group in graph_report["groups"]
    if any(candidate["capture"] == "hip-direct-graph.json" for candidate in group["candidates"])
)
graph_review = graph_group["candidates"][0]["hip_graph_replay_review"]
assert graph_review["baseline_setup_inclusive_median_end_to_end_us"] == 1000.0 + 11.0 / 9.0
assert graph_review["setup_inclusive_median_end_to_end_us"] == 900.0 + 29.0 / 9.0
assert graph_review["baseline_total_setup_us"] == 11.0
assert graph_review["graph_total_setup_us"] == 29.0
assert graph_review["graph_setup_overhead_vs_baseline_us"] == 18.0
assert graph_review["steady_state_delta_us"] == 100.0
assert graph_review["break_even_repeat_count"] == 1
assert graph_review["declared_repeat_count"] == 9
assert graph_review["declared_repeats_meet_break_even"] is True
assert graph_review["blockers"] == []
assert group["missing_hip_driver_versions"] == []
assert group["hip_driver_version_complete"] is True
assert group["hip_driver_version_compatible"] is True
assert group["missing_compiler_identities"] == []
assert group["compiler_identity_complete"] is True
assert group["compiler_identity_compatible"] is True
assert group["missing_git_commits"] == []
assert group["git_commit_identity_complete"] is True
assert group["git_commit_identity_compatible"] is True
assert group["missing_warmup_counts"] == []
assert group["warmup_count_complete"] is True
assert group["warmup_count_compatible"] is True
assert group["missing_repeat_counts"] == []
assert group["repeat_count_complete"] is True
assert group["repeat_count_compatible"] is True
assert group["duplicate_backends"] == []
assert group["finite_modulus"] == 255
assert group["fastest_promotable"]["backend"] == "ck"
assert group["candidates"][0]["promotion_blockers"] == []
assert group["candidates"][0]["bottleneck"]["class"] in {
    "compute_bound",
    "export_bound",
    "launch_or_api_bound",
    "mixed_bound",
    "pack_bound",
    "unknown",
}

non_promoting_ck = copy.deepcopy(ck)
non_promoting_direct = copy.deepcopy(direct)
non_promoting_cpu = copy.deepcopy(cpu)
scenario_metadata = {
    "family": "finite-modulus-map",
    "name": "finite-ring-map-1024",
    "promotion_eligibility": "non_promoting_modulus_map",
    "metadata": {"promotion_scope": "non_promoting_modulus_map"},
}
for item in [non_promoting_ck, non_promoting_direct, non_promoting_cpu]:
    item["scenario_metadata"] = scenario_metadata
non_promoting_report = benchmark_sweep.review_captures(
    [non_promoting_ck, non_promoting_direct, non_promoting_cpu],
    review_mode="release",
)
non_promoting_group = non_promoting_report["groups"][0]
assert non_promoting_report["promotable_autotune_entries"] == []
assert non_promoting_group["scenario_promotion_scopes"] == ["non_promoting_modulus_map"]
non_promoting_ck_candidate = next(
    item for item in non_promoting_group["candidates"] if item["backend"] == "ck"
)
assert non_promoting_ck_candidate["scenario_promotion_scope"] == "non_promoting_modulus_map"
assert "scenario_scope_not_autotune_promotable" in non_promoting_ck_candidate["promotion_blockers"]
assert "missing_required_baselines" not in non_promoting_ck_candidate["promotion_blockers"]
assert "not_faster_than_direct_hip" not in non_promoting_ck_candidate["promotion_blockers"]

missing_target_ck = copy.deepcopy(ck)
missing_target_ck["device"]["gcn_arch"] = "unknown"
missing_target_report = benchmark_sweep.review_captures(
    [missing_target_ck, direct, cpu],
    review_mode="release",
)
missing_target_group = missing_target_report["groups"][0]
assert missing_target_report["promotable_autotune_entries"] == []
assert missing_target_group["missing_gpu_targets"] == ["ck"]
assert missing_target_group["gpu_target_identity_complete"] is False
assert missing_target_group["gpu_target_compatible"] is False
missing_target_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_target_group["candidates"]
}
assert "missing_gpu_target_id" in missing_target_blockers["ck"]

mismatched_target_ck = copy.deepcopy(ck)
mismatched_target_ck["device"]["gcn_arch"] = "gfx1101"
mismatched_target_report = benchmark_sweep.review_captures(
    [mismatched_target_ck, direct, cpu],
    review_mode="release",
)
mismatched_target_group = mismatched_target_report["groups"][0]
assert mismatched_target_report["promotable_autotune_entries"] == []
assert mismatched_target_group["missing_gpu_targets"] == []
assert mismatched_target_group["gpu_target_identity_complete"] is True
assert mismatched_target_group["gpu_target_compatible"] is False
mismatched_target_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_target_group["candidates"]
}
assert "gpu_target_mismatch" in mismatched_target_blockers["ck"]

missing_version_direct = copy.deepcopy(direct)
missing_version_direct["hip_toolchain"]["hip_sdk_or_rocm_version"] = None
missing_version_report = benchmark_sweep.review_captures(
    [ck, missing_version_direct, cpu],
    review_mode="release",
)
missing_version_group = missing_version_report["groups"][0]
assert missing_version_report["promotable_autotune_entries"] == []
assert missing_version_group["missing_hip_toolchain_versions"] == ["hip-direct"]
assert missing_version_group["hip_toolchain_version_complete"] is False
assert missing_version_group["hip_toolchain_version_compatible"] is False
missing_version_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_version_group["candidates"]
}
assert "missing_hip_toolchain_version" in missing_version_blockers["ck"]

mismatched_version_ck = copy.deepcopy(ck)
mismatched_version_ck["hip_toolchain"]["hip_sdk_or_rocm_version"] = "70260299"
mismatched_version_report = benchmark_sweep.review_captures(
    [mismatched_version_ck, direct, cpu],
    review_mode="release",
)
mismatched_version_group = mismatched_version_report["groups"][0]
assert mismatched_version_report["promotable_autotune_entries"] == []
assert mismatched_version_group["missing_hip_toolchain_versions"] == []
assert mismatched_version_group["hip_toolchain_version_complete"] is True
assert mismatched_version_group["hip_toolchain_version_compatible"] is False
mismatched_version_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_version_group["candidates"]
}
assert "hip_toolchain_version_mismatch" in mismatched_version_blockers["ck"]

missing_configured_direct = copy.deepcopy(direct)
missing_configured_direct["configured_amdgpu_targets"] = "unknown"
missing_configured_report = benchmark_sweep.review_captures(
    [ck, missing_configured_direct, cpu],
    review_mode="release",
)
missing_configured_group = missing_configured_report["groups"][0]
assert missing_configured_report["promotable_autotune_entries"] == []
assert missing_configured_group["missing_configured_gpu_targets"] == ["hip-direct"]
assert missing_configured_group["configured_target_identity_complete"] is False
assert missing_configured_group["configured_target_compatible"] is False
missing_configured_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_configured_group["candidates"]
}
assert "missing_configured_gpu_target" in missing_configured_blockers["ck"]

mismatched_configured_ck = copy.deepcopy(ck)
mismatched_configured_ck["configured_amdgpu_targets"] = "gfx1101"
mismatched_configured_report = benchmark_sweep.review_captures(
    [mismatched_configured_ck, direct, cpu],
    review_mode="release",
)
mismatched_configured_group = mismatched_configured_report["groups"][0]
assert mismatched_configured_report["promotable_autotune_entries"] == []
assert mismatched_configured_group["missing_configured_gpu_targets"] == []
assert mismatched_configured_group["configured_target_identity_complete"] is True
assert mismatched_configured_group["configured_target_compatible"] is False
mismatched_configured_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_configured_group["candidates"]
}
assert "configured_gpu_target_mismatch" in mismatched_configured_blockers["ck"]

missing_runtime_direct = copy.deepcopy(direct)
missing_runtime_direct["device"]["hip_runtime_version"] = 0
missing_runtime_report = benchmark_sweep.review_captures(
    [ck, missing_runtime_direct, cpu],
    review_mode="release",
)
missing_runtime_group = missing_runtime_report["groups"][0]
assert missing_runtime_report["promotable_autotune_entries"] == []
assert missing_runtime_group["missing_hip_runtime_versions"] == ["hip-direct"]
assert missing_runtime_group["hip_runtime_version_complete"] is False
assert missing_runtime_group["hip_runtime_version_compatible"] is False
missing_runtime_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_runtime_group["candidates"]
}
assert "missing_hip_runtime_version" in missing_runtime_blockers["ck"]

mismatched_runtime_ck = copy.deepcopy(ck)
mismatched_runtime_ck["device"]["hip_runtime_version"] = 70260299
mismatched_runtime_report = benchmark_sweep.review_captures(
    [mismatched_runtime_ck, direct, cpu],
    review_mode="release",
)
mismatched_runtime_group = mismatched_runtime_report["groups"][0]
assert mismatched_runtime_report["promotable_autotune_entries"] == []
assert mismatched_runtime_group["missing_hip_runtime_versions"] == []
assert mismatched_runtime_group["hip_runtime_version_complete"] is True
assert mismatched_runtime_group["hip_runtime_version_compatible"] is False
mismatched_runtime_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_runtime_group["candidates"]
}
assert "hip_runtime_version_mismatch" in mismatched_runtime_blockers["ck"]

missing_driver_direct = copy.deepcopy(direct)
missing_driver_direct["device"]["hip_driver_version"] = 0
missing_driver_report = benchmark_sweep.review_captures(
    [ck, missing_driver_direct, cpu],
    review_mode="release",
)
missing_driver_group = missing_driver_report["groups"][0]
assert missing_driver_report["promotable_autotune_entries"] == []
assert missing_driver_group["missing_hip_driver_versions"] == ["hip-direct"]
assert missing_driver_group["hip_driver_version_complete"] is False
assert missing_driver_group["hip_driver_version_compatible"] is False
missing_driver_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_driver_group["candidates"]
}
assert "missing_hip_driver_version" in missing_driver_blockers["ck"]

mismatched_driver_ck = copy.deepcopy(ck)
mismatched_driver_ck["device"]["hip_driver_version"] = 70260299
mismatched_driver_report = benchmark_sweep.review_captures(
    [mismatched_driver_ck, direct, cpu],
    review_mode="release",
)
mismatched_driver_group = mismatched_driver_report["groups"][0]
assert mismatched_driver_report["promotable_autotune_entries"] == []
assert mismatched_driver_group["missing_hip_driver_versions"] == []
assert mismatched_driver_group["hip_driver_version_complete"] is True
assert mismatched_driver_group["hip_driver_version_compatible"] is False
mismatched_driver_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_driver_group["candidates"]
}
assert "hip_driver_version_mismatch" in mismatched_driver_blockers["ck"]

missing_compiler_direct = copy.deepcopy(direct)
missing_compiler_direct["compiler"]["version"] = ""
missing_compiler_report = benchmark_sweep.review_captures(
    [ck, missing_compiler_direct, cpu],
    review_mode="release",
)
missing_compiler_group = missing_compiler_report["groups"][0]
assert missing_compiler_report["promotable_autotune_entries"] == []
assert missing_compiler_group["missing_compiler_identities"] == ["hip-direct"]
assert missing_compiler_group["compiler_identity_complete"] is False
assert missing_compiler_group["compiler_identity_compatible"] is False
missing_compiler_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_compiler_group["candidates"]
}
assert "missing_compiler_identity" in missing_compiler_blockers["ck"]

mismatched_compiler_ck = copy.deepcopy(ck)
mismatched_compiler_ck["compiler"]["version"] = "1944.999999"
mismatched_compiler_report = benchmark_sweep.review_captures(
    [mismatched_compiler_ck, direct, cpu],
    review_mode="release",
)
mismatched_compiler_group = mismatched_compiler_report["groups"][0]
assert mismatched_compiler_report["promotable_autotune_entries"] == []
assert mismatched_compiler_group["missing_compiler_identities"] == []
assert mismatched_compiler_group["compiler_identity_complete"] is True
assert mismatched_compiler_group["compiler_identity_compatible"] is False
mismatched_compiler_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_compiler_group["candidates"]
}
assert "compiler_identity_mismatch" in mismatched_compiler_blockers["ck"]

missing_git_direct = copy.deepcopy(direct)
missing_git_direct["git_commit"] = "unknown"
missing_git_report = benchmark_sweep.review_captures(
    [ck, missing_git_direct, cpu],
    review_mode="release",
)
missing_git_group = missing_git_report["groups"][0]
assert missing_git_report["promotable_autotune_entries"] == []
assert missing_git_group["missing_git_commits"] == ["hip-direct"]
assert missing_git_group["git_commit_identity_complete"] is False
assert missing_git_group["git_commit_identity_compatible"] is False
missing_git_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_git_group["candidates"]
}
assert "missing_git_commit" in missing_git_blockers["ck"]

mismatched_git_ck = copy.deepcopy(ck)
mismatched_git_ck["git_commit"] = "different-fixture"
mismatched_git_report = benchmark_sweep.review_captures(
    [mismatched_git_ck, direct, cpu],
    review_mode="release",
)
mismatched_git_group = mismatched_git_report["groups"][0]
assert mismatched_git_report["promotable_autotune_entries"] == []
assert mismatched_git_group["missing_git_commits"] == []
assert mismatched_git_group["git_commit_identity_complete"] is True
assert mismatched_git_group["git_commit_identity_compatible"] is False
mismatched_git_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_git_group["candidates"]
}
assert "git_commit_mismatch" in mismatched_git_blockers["ck"]

missing_warmups_direct = copy.deepcopy(direct)
missing_warmups_direct["warmups"] = 0
missing_warmups_report = benchmark_sweep.review_captures(
    [ck, missing_warmups_direct, cpu],
    review_mode="release",
)
missing_warmups_group = missing_warmups_report["groups"][0]
assert missing_warmups_report["promotable_autotune_entries"] == []
assert missing_warmups_group["missing_warmup_counts"] == ["hip-direct"]
assert missing_warmups_group["warmup_count_complete"] is False
assert missing_warmups_group["warmup_count_compatible"] is False
missing_warmups_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_warmups_group["candidates"]
}
assert "missing_warmup_count" in missing_warmups_blockers["ck"]

mismatched_warmups_ck = copy.deepcopy(ck)
mismatched_warmups_ck["warmups"] = benchmark_sweep.RELEASE_MIN_WARMUPS + 1
mismatched_warmups_report = benchmark_sweep.review_captures(
    [mismatched_warmups_ck, direct, cpu],
    review_mode="release",
)
mismatched_warmups_group = mismatched_warmups_report["groups"][0]
assert mismatched_warmups_report["promotable_autotune_entries"] == []
assert mismatched_warmups_group["missing_warmup_counts"] == []
assert mismatched_warmups_group["warmup_count_complete"] is True
assert mismatched_warmups_group["warmup_count_compatible"] is False
mismatched_warmups_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_warmups_group["candidates"]
}
assert "warmup_count_mismatch" in mismatched_warmups_blockers["ck"]

missing_repeats_direct = copy.deepcopy(direct)
missing_repeats_direct["repeats"] = 0
missing_repeats_report = benchmark_sweep.review_captures(
    [ck, missing_repeats_direct, cpu],
    review_mode="release",
)
missing_repeats_group = missing_repeats_report["groups"][0]
assert missing_repeats_report["promotable_autotune_entries"] == []
assert missing_repeats_group["missing_repeat_counts"] == ["hip-direct"]
assert missing_repeats_group["repeat_count_complete"] is False
assert missing_repeats_group["repeat_count_compatible"] is False
missing_repeats_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_repeats_group["candidates"]
}
assert "missing_repeat_count" in missing_repeats_blockers["ck"]

mismatched_repeats_ck = copy.deepcopy(ck)
mismatched_repeats_ck["repeats"] = benchmark_sweep.RELEASE_MIN_REPEATS + 1
mismatched_repeats_report = benchmark_sweep.review_captures(
    [mismatched_repeats_ck, direct, cpu],
    review_mode="release",
)
mismatched_repeats_group = mismatched_repeats_report["groups"][0]
assert mismatched_repeats_report["promotable_autotune_entries"] == []
assert mismatched_repeats_group["missing_repeat_counts"] == []
assert mismatched_repeats_group["repeat_count_complete"] is True
assert mismatched_repeats_group["repeat_count_compatible"] is False
mismatched_repeats_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_repeats_group["candidates"]
}
assert "repeat_count_mismatch" in mismatched_repeats_blockers["ck"]

duplicate_ck_report = benchmark_sweep.review_captures(
    [ck, copy.deepcopy(ck), direct, cpu],
    review_mode="release",
)
duplicate_ck_group = duplicate_ck_report["groups"][0]
assert duplicate_ck_report["promotable_autotune_entries"] == []
assert duplicate_ck_group["duplicate_backends"] == ["ck"]
duplicate_ck_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in duplicate_ck_group["candidates"]
}
assert "duplicate_backend_capture" in duplicate_ck_blockers["ck"]

host_batch_ck = as_host_api_batch_capture(ck)
host_batch_report = benchmark_sweep.review_captures(
    [ck, host_batch_ck, direct, cpu],
    review_mode="release",
)
assert host_batch_report["group_count"] == 2
host_batch_group = next(
    group
    for group in host_batch_report["groups"]
    if any(candidate["backend"] == "ck-hostbatch" for candidate in group["candidates"])
)
assert host_batch_group["duplicate_backends"] == []
host_batch_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in host_batch_group["candidates"]
}
assert "host_api_batch_not_autotune_promotable" in host_batch_blockers["ck-hostbatch"]

scenario_a_cpu = copy.deepcopy(cpu)
scenario_a_direct = copy.deepcopy(direct)
scenario_b_cpu = copy.deepcopy(cpu)
scenario_b_direct = copy.deepcopy(direct)
scenario_a_cpu["checksum_u64"] = 111
scenario_a_direct["checksum_u64"] = 111
scenario_b_cpu["checksum_u64"] = 222
scenario_b_direct["checksum_u64"] = 222
for item in [scenario_a_cpu, scenario_a_direct]:
    item["scenario_metadata"] = {
        "family": "many-small",
        "name": "bounded-i64-64-proxy",
        "promotion_eligibility": "release_review_candidate",
        "output_domain": "host_export",
        "metadata": {"workflow_name": "many_small"},
    }
for item in [scenario_b_cpu, scenario_b_direct]:
    item["scenario_metadata"] = {
        "family": "small-oneshot",
        "name": "bounded-i64-64-oneshot",
        "promotion_eligibility": "release_review_candidate",
        "output_domain": "host_export",
        "metadata": {"workflow_name": "small_oneshot"},
    }
scenario_split_report = benchmark_sweep.review_captures(
    [scenario_a_cpu, scenario_a_direct, scenario_b_cpu, scenario_b_direct],
    review_mode="release",
)
assert scenario_split_report["group_count"] == 2
assert all(group["checksum_mismatches"] == [] for group in scenario_split_report["groups"])

reuse_report = benchmark_sweep.review_captures(
    [mark_reused_pack(ck), mark_reused_pack(direct), mark_reused_pack(cpu)],
    review_mode="release",
)
reuse_group = reuse_report["groups"][0]
assert reuse_group["source_metadata"]["pack_modes"] == ["prepacked_reuse"]
assert reuse_group["source_metadata"]["prepack_reuse_strategies"] == ["persistent_matrix_residency"]
assert reuse_group["source_metadata"]["prepack_reuse_operands"] == ["A/B"]
assert reuse_report["promotable_autotune_entries"] == []
reuse_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in reuse_group["candidates"]
}
assert "missing_same_backend_nonreuse_baseline" in reuse_blockers["ck"]
assert "missing_best_nonreuse_contract_baseline" in reuse_blockers["ck"]
assert "prepacked_reuse_not_autotune_promotable" not in reuse_blockers["ck"]

reuse_a_report = benchmark_sweep.review_captures(
    [mark_reused_a_pack(ck), mark_reused_a_pack(direct), mark_reused_a_pack(cpu)],
    review_mode="release",
)
reuse_a_group = reuse_a_report["groups"][0]
assert reuse_a_group["source_metadata"]["pack_modes"] == ["prepacked_reuse_a"]
assert reuse_a_group["source_metadata"]["prepack_reuse_strategies"] == ["persistent_matrix_residency"]
assert reuse_a_group["source_metadata"]["prepack_reuse_operands"] == ["A"]
reuse_a_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in reuse_a_group["candidates"]
}
assert "missing_same_backend_nonreuse_baseline" in reuse_a_blockers["ck"]
assert "missing_best_nonreuse_contract_baseline" in reuse_a_blockers["ck"]
assert "prepacked_reuse_not_autotune_promotable" not in reuse_a_blockers["ck"]

reuse_evidence_direct = mark_reused_a_pack(direct)
reuse_evidence_direct["scenario_metadata"] = {
    "family": "direct-hip-reuse-expansion",
    "name": "bounded-u64-adaptive-512-reuse-a",
    "promotion_eligibility": "reuse_contract_evidence_only",
    "metadata": {
        "workflow_name": "direct_hip_reuse_expansion",
        "reuse_contract_role": "stable_a_candidate",
        "promotion_scope": "reuse_contract_evidence_only",
    },
}
reuse_evidence_report = benchmark_sweep.review_captures([reuse_evidence_direct], review_mode="release")
reuse_evidence_group = reuse_evidence_report["groups"][0]
assert reuse_evidence_group["required_baselines"] == []
assert reuse_evidence_group["missing_required_baselines"] == []
reuse_evidence_blockers = reuse_evidence_group["candidates"][0]["promotion_blockers"]
assert "missing_required_baselines" not in reuse_evidence_blockers
assert "prepacked_reuse_not_autotune_promotable" not in reuse_evidence_blockers
assert "scenario_scope_not_autotune_promotable" in reuse_evidence_blockers

graph_evidence_direct = mark_reused_a_pack(direct)
graph_evidence_direct["timing_metadata"]["benchmark_execution_mode"] = "hip_graph_replay_resident_rns_chain"
graph_evidence_direct["benchmark_execution_mode"] = "hip_graph_replay_resident_rns_chain"
graph_evidence_direct["scenario_metadata"] = {
    "family": "hip-graph-replay",
    "name": "bounded-i64-chain3-512-graph",
    "promotion_eligibility": "hip_graph_replay_evidence_only",
    "metadata": {
        "workflow_name": "hip_graph_replay",
        "graph_role": "graph_replay_candidate",
    },
}
graph_evidence_report = benchmark_sweep.review_captures([graph_evidence_direct], review_mode="release")
graph_evidence_group = graph_evidence_report["groups"][0]
assert graph_evidence_group["required_baselines"] == []
assert graph_evidence_group["missing_required_baselines"] == []
graph_evidence_blockers = graph_evidence_group["candidates"][0]["promotion_blockers"]
assert "missing_required_baselines" not in graph_evidence_blockers
assert "hip_graph_replay_not_autotune_promotable" not in graph_evidence_blockers
assert "scenario_scope_not_autotune_promotable" in graph_evidence_blockers

variant_direct_a = exact_wide_capture("hip-direct", 3000)
variant_direct_b = copy.deepcopy(variant_direct_a)
variant_direct_a["export_variant"] = {
    "name": "compact-d2h-export-candidate",
}
variant_direct_b["export_variant"] = {
    "name": "tree-crt-export-candidate",
}
variant_direct_b["reconstruction_variant"] = {
    "name": "tree_crt_candidate",
}
variant_report = benchmark_sweep.review_captures(
    [variant_direct_a, variant_direct_b],
    review_mode="release",
)
assert variant_report["group_count"] == 2
assert all(group["duplicate_backends"] == [] for group in variant_report["groups"])
assert sorted(group["contract_key"].split("export_variant=", 1)[1].split(";", 1)[0] for group in variant_report["groups"]) == [
    "compact-d2h-export-candidate",
    "tree-crt-export-candidate",
]

