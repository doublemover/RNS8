ck = finite_capture("ck", 190)
direct = finite_capture("hip-direct", 300)
cpu = finite_capture("cpu-reference", 500)
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
group = report["groups"][0]
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
assert "prepacked_reuse_not_autotune_promotable" in reuse_blockers["ck"]

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
assert "prepacked_reuse_not_autotune_promotable" in reuse_a_blockers["ck"]

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
assert "prepacked_reuse_not_autotune_promotable" in reuse_evidence_blockers
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
assert "hip_graph_replay_not_autotune_promotable" in graph_evidence_blockers
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

