with tempfile.TemporaryDirectory() as temp_dir:
    cache_path = Path(temp_dir) / "finite-autotune.json"
    promoted = benchmark_sweep.write_promoted_cache_entries(report, [ck, direct, cpu], cache_path)
    benchmark_sweep.attach_cache_write_status(report, True, cache_path, promoted)
    assert promoted == 1
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    entry = cache["entries"][0]
    assert entry["finite_modulus"] == 255
    assert ";finite_modulus=255;" in f";{entry['key']};"

blocked = benchmark_sweep.review_captures([ck], review_mode="release")
benchmark_sweep.attach_cache_write_status(blocked, False, Path("unused.json"), 0)
assert blocked["promotable_autotune_entries"] == []
assert blocked["cache_write"]["status"] == "not_requested"
assert blocked["groups"][0]["missing_required_baselines"] == ["cpu-reference", "hip-direct"]

wrap64_direct = wrap64_capture("hip-direct", 200)
wrap64_cpu = wrap64_capture("wrap64-byte-limb", 500)
wrap64_report = benchmark_sweep.review_captures([wrap64_direct, wrap64_cpu], review_mode="release")
assert wrap64_report["promotable_autotune_entries"] == []
wrap64_group = wrap64_report["groups"][0]
assert wrap64_group["semantics"] == "wrap_u64_mod_2_64"
assert wrap64_group["missing_required_baselines"] == []
assert wrap64_group["release_review_satisfied"] is True
assert wrap64_group["fastest_promotable"] is None
blockers_by_backend = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in wrap64_group["candidates"]
}
assert "not_accelerator_backend" in blockers_by_backend["hip-direct"]
assert "not_accelerator_backend" in blockers_by_backend["wrap64-byte-limb"]
assert "not_faster_than_direct_hip" in blockers_by_backend["wrap64-byte-limb"]

wrap64_candidate = wrap64_capture(benchmark_sweep.WRAP64_ROCWMMA_CANDIDATE_BACKEND, 150)
validate_capture(wrap64_candidate)
wrap64_candidate_report = benchmark_sweep.review_captures(
    [wrap64_direct, wrap64_cpu, wrap64_candidate], review_mode="release"
)
candidate_group = wrap64_candidate_report["groups"][0]
assert candidate_group["missing_required_baselines"] == []
assert wrap64_candidate_report["promotable_autotune_entries"] == []
candidate_blockers = {
    candidate["backend"]: candidate["promotion_blockers"] for candidate in candidate_group["candidates"]
}
assert "internal_candidate_not_public_backend" in candidate_blockers[
    benchmark_sweep.WRAP64_ROCWMMA_CANDIDATE_BACKEND
]

exact_ck = exact_wide_capture("ck", 170)
exact_direct = exact_wide_capture("hip-direct", 300)
exact_cpu = exact_wide_capture("cpu-reference", 520)
exact_report = benchmark_sweep.review_captures([exact_ck, exact_direct, exact_cpu], review_mode="release")
exact_group = exact_report["groups"][0]
assert exact_group["semantics"] == "exact_wide_signed"
assert exact_group["required_baselines"] == ["cpu-reference", "hip-direct"]
assert exact_group["missing_required_baselines"] == []
assert len(exact_report["promotable_autotune_entries"]) == 1
assert exact_report["promotable_autotune_entries"][0]["selected_backend"] == "ck"

exact_blocked = benchmark_sweep.review_captures([exact_ck], review_mode="release")
assert exact_blocked["groups"][0]["missing_required_baselines"] == ["cpu-reference", "hip-direct"]

with tempfile.TemporaryDirectory() as temp_dir:
    bounded_ck = bounded_capture("ck", 180)
    bounded_direct = bounded_capture("hip-direct", 300)
    bounded_vector = bounded_capture("hip-vector-alu-int64", 240)
    bounded_cpu = bounded_capture("cpu-reference", 500)
    bounded_report = benchmark_sweep.review_captures(
        [bounded_ck, bounded_direct, bounded_vector, bounded_cpu], review_mode="release"
    )
    assert len(bounded_report["promotable_autotune_entries"]) == 1
    assert bounded_report["promotable_autotune_entries"][0]["selected_backend"] == "ck"
    assert bounded_report["promotable_autotune_entries"][0]["target_id"] == "gfx1100"
    assert bounded_report["groups"][0]["fastest_promotable"]["backend"] == "ck"
    cache_path = Path(temp_dir) / "autotune.json"
    promoted = benchmark_sweep.write_promoted_cache_entries(
        bounded_report, [bounded_ck, bounded_direct, bounded_vector, bounded_cpu], cache_path
    )
    benchmark_sweep.attach_cache_write_status(bounded_report, True, cache_path, promoted)
    assert promoted == 1
    assert bounded_report["cache_write"]["status"] == "written"
    assert bounded_report["groups"][0]["fastest_promotable"]["cache_write_status"] == "written"
    assert bounded_report["groups"][0]["candidates"][1]["cache_write_status"] != "written"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    entry = cache["entries"][0]
    assert entry["performance_validated"] is True
    assert entry["validation_status"] == "reviewed_release_same_contract_fastest_windows_gfx1100"
    assert entry["epilogue"] == bounded_ck["backend_metadata"]["epilogue_mode"]
    assert f";epilogue={entry['epilogue']}" in entry["key"]

validate_capture(load_capture(FIXTURE_DIR / "v4_finite_ring_u8_ck.json"))
validate_capture(load_capture(FIXTURE_DIR / "v4_finite_field_u8_rocwmma.json"))
