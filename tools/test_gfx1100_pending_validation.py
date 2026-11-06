#!/usr/bin/env python3
"""Self-test gfx1100 pending validation command planning."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gfx1100_pending_validation
import pending_validation
from benchmark_schema import load_capture


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def _generic_config(tmp: Path) -> pending_validation.PendingValidationConfig:
    return pending_validation.PendingValidationConfig(
        target="cdna-smoke",
        target_ids=frozenset({"gfx942"}),
        policy="fixture_target_policy",
        plan_schema="fixture_plan_v1",
        summary_schema="fixture_summary_v1",
        command_plan_title="Fixture Target Command Plan",
        summary_title="Fixture Target Summary",
        summary_scope_lines=("Scope: fixture target only.",),
        default_out_dir=tmp,
        bench_for={"hip-direct": tmp / "rns8-bench"},
        build_presets=("linux-cdna-release",),
        scenarios=("tiny-scenario",),
        primary_bench_backend="hip-direct",
        build_command_prefix=("cmake", "--build", "--preset"),
        build_command_uses_python=False,
    )


def _write_capture(path: Path, backend: str = "ck") -> None:
    capture = copy.deepcopy(load_capture(FIXTURE_DIR / "v4_bounded_i64_ck.json"))
    capture["backend_requested"] = backend
    capture["backend_selected"] = backend
    capture["backend_metadata"]["performance_validated"] = False
    capture["comparison_baseline"]["status"] = "required_not_recorded"
    capture["comparison_baseline"]["speedup_claimed"] = False
    capture["comparison_baseline"]["selected_reference"] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_review_report(path: Path, capture_path: Path, *, promotable: bool) -> None:
    blockers = [] if promotable else ["missing_required_baselines"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_mode": "release",
                "groups": [
                    {
                        "review_mode": "release",
                        "release_review_satisfied": True,
                        "missing_required_baselines": [] if promotable else ["cpu-reference"],
                        "duplicate_backends": [],
                        "required_baselines": ["cpu-reference", "hip-direct"],
                        "contract_key": "test-contract",
                        "fastest_promotable": {
                            "backend": "ck",
                            "selected_kernel": "ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2",
                        }
                        if promotable
                        else None,
                        "candidates": [
                            {
                                "backend": "ck",
                                "capture": str(capture_path),
                                "cache_write_status": "eligible_after_review"
                                if promotable
                                else "not_eligible",
                                "median_end_to_end_us": 10.0,
                                "promotable": promotable,
                                "promotion_blockers": blockers,
                                "promotion_reason": "beats_required_same_contract_gpu_baselines"
                                if promotable
                                else "blocked",
                                "release_review_capture": True,
                                "selected_kernel": "ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2",
                                "speedup_vs_direct_hip": 1.2 if promotable else 0.9,
                            }
                        ],
                    }
                ],
                "promotable_autotune_entries": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def receipt_round_trip_controls():
    from benchmark_sweep_lib.review import review_captures
    from diagnostic_companion import build_binding
    from report_inputs import load_validated_capture
    from test_benchmark_sweep_support import finite_capture, set_phase
    from test_benchmark_schema_support_metadata import add_target_variant_fields
    from test_diagnostic_companion import authored_capture_pair
    from test_review_reference_support import add_synthetic_full_reference

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        captures, bindings = [], {}
        for backend, duration in (("cpu-reference", 5000), ("hip-direct", 1000), ("ck", 800)):
            for run in range(2):
                lane = root / f"{backend}-{run}"
                lane.mkdir()
                fixture = add_synthetic_full_reference(finite_capture(backend, duration))
                _, _, paths = authored_capture_pair(lane, fixture)
                captures.append(load_validated_capture(paths[0]))
                if backend != "cpu-reference":
                    bindings[str(paths[0].resolve())] = build_binding(*paths)
        report = review_captures(captures, review_mode="release", diagnostic_bindings=bindings)
        review_path = root / "review_report.json"
        review_path.write_text(json.dumps(report), encoding="utf-8")
        index = pending_validation.load_review_index([review_path])
        reviewed = next(
            item
            for group in report["groups"]
            for item in group["candidates"]
            if item["backend"] == "ck"
        )
        candidate = next(
            capture
            for capture in captures
            if pending_validation.capture_identity(capture) == reviewed["capture_sha256"]
        )
        path = Path(candidate["_path"])

        def decision(current_path=path, current_index=index):
            return pending_validation.decision_for_capture(
                gfx1100_pending_validation.CONFIG, current_path, review_index=current_index
            )

        # Replay the actual producer's backend/semantic disposition separately
        # from measurement qualification. Independent reruns are not duplicates.
        row = decision()
        assert row["review_receipt"]["bound"], row["blockers"]
        assert row["review_receipt"]["qualified"], row["blockers"]
        assert row["review_receipt"]["measurement_cohort"]["independent_runs"] == 2
        assert row["disposition"] == (
            "promote locally" if reviewed["promotable"] else "keep experimental"
        )
        original = path.read_text(encoding="utf-8")

        # Change a bound companion after admission, before the standalone
        # decision's publication snapshot closes. No pathname/mtime credit.
        companion = Path(bindings[str(path.resolve())]["diagnostic_capture"])
        companion_bytes = companion.read_text(encoding="utf-8")
        finish = pending_validation.ReviewReceiptReader.finish

        def change_companion_before_finish(reader):
            changed = json.loads(companion_bytes)
            changed["checksum_u64"] += 1
            companion.write_text(json.dumps(changed), encoding="utf-8")
            finish(reader)

        try:
            with patch.object(
                pending_validation.ReviewReceiptReader, "finish", change_companion_before_finish
            ):
                changed_decision = decision()
            assert not changed_decision["review_receipt"]["bound"]
            assert not changed_decision["review_receipt"]["qualified"]
            assert "review_input_changed_during_decision" in changed_decision["blockers"]
            assert changed_decision["disposition"] == "keep experimental"
        finally:
            companion.write_text(companion_bytes, encoding="utf-8")
        for mutation in ("checksum", "timing", "seed", "build", "device"):
            changed = json.loads(original)
            if mutation == "checksum":
                changed["checksum_u64"] += 1
            elif mutation == "timing":
                set_phase(changed, 850)
            elif mutation == "seed":
                changed["seed"] += 1
            elif mutation == "build":
                changed["build_manifest"]["source_digest"] = "d" * 64
            else:
                changed["device"]["name"] = "Different physical GPU"
                add_target_variant_fields(changed)
            path.write_text(json.dumps(changed), encoding="utf-8")
            load_validated_capture(path)
            row = decision()
            assert not row["review_receipt"]["bound"], mutation
            assert "review_capture_content_mismatch" in row["blockers"]
            assert not row["review_report_promotable"]
            assert row["review_speedup_vs_direct_hip"] is None
            path.write_text(original, encoding="utf-8")
        renamed = root / "unchanged-new-name.json"
        renamed.write_text(original, encoding="utf-8")
        row = decision(renamed)
        assert row["review_receipt"]["bound"]
        assert row["review_receipt"]["qualified"]
        assert row["review_receipt"]["measurement_cohort"]["independent_runs"] == 2

        def change_alias_before_finish(reader):
            changed = json.loads(original)
            changed["checksum_u64"] += 1
            renamed.write_text(json.dumps(changed), encoding="utf-8")
            finish(reader)

        try:
            with patch.object(
                pending_validation.ReviewReceiptReader, "finish", change_alias_before_finish
            ):
                changed_decision = decision(renamed)
            assert not changed_decision["review_receipt"]["bound"]
            assert "review_input_changed_during_decision" in changed_decision["blockers"]
        finally:
            renamed.write_text(original, encoding="utf-8")

        conflicting = copy.deepcopy(report)
        for group in conflicting["groups"]:
            for item in group["candidates"]:
                if item["capture_sha256"] == pending_validation.capture_identity(candidate):
                    item["promotable"] = not item["promotable"]
        sidecar_dir = root / "conflicting"
        sidecar_dir.mkdir()
        sidecar = sidecar_dir / "review_report.json"
        sidecar.write_text(json.dumps(conflicting), encoding="utf-8")
        for reports in ([review_path, sidecar], [sidecar, review_path]):
            conflicts = pending_validation.load_review_index(reports)
            row = decision(current_index=conflicts)
            assert "conflicting_review_receipts" in row["blockers"]
            assert not row["review_receipt"]["bound"]
        reviewed_bytes = review_path.read_text(encoding="utf-8")
        review_path.write_text(json.dumps(conflicting), encoding="utf-8")
        assert "review_report_content_changed" in decision()["blockers"]
        review_path.write_text(reviewed_bytes, encoding="utf-8")

        baseline_path = Path(captures[2]["_path"])
        baseline_original = baseline_path.read_text(encoding="utf-8")
        changed = json.loads(baseline_original)
        changed["checksum_u64"] += 1
        baseline_path.write_text(json.dumps(changed), encoding="utf-8")
        assert "review_input_content_missing_or_changed" in decision()["blockers"]
        baseline_path.write_text(baseline_original, encoding="utf-8")
        diagnostic_path = Path(bindings[str(path.resolve())]["diagnostic_capture"])
        diagnostic_original = diagnostic_path.read_text(encoding="utf-8")
        changed = json.loads(diagnostic_original)
        changed["checksum_u64"] += 1
        diagnostic_path.write_text(json.dumps(changed), encoding="utf-8")
        row = decision()
        assert not row["review_receipt"]["bound"]
        assert "review_diagnostic_binding_unavailable_or_changed" in row["blockers"]
        diagnostic_path.write_text(diagnostic_original, encoding="utf-8")

        singleton = review_captures(
            captures[::2], review_mode="release", diagnostic_bindings=bindings
        )
        singleton_path = root / "singleton-review.json"
        singleton_path.write_text(json.dumps(singleton), encoding="utf-8")
        singleton_index = pending_validation.load_review_index([singleton_path])
        row = decision(Path(captures[4]["_path"]), singleton_index)
        assert row["review_receipt"]["bound"]
        assert not row["review_receipt"]["qualified"]
        assert row["review_receipt"]["measurement_cohort"]["independent_runs"] == 1
        assert row["disposition"] == "keep experimental"
        # Self-reported historical review flags never replace a current receipt.
        history = json.loads(original)
        history["backend_metadata"]["performance_validated"] = True
        history["comparison_baseline"]["status"] = "reviewed_release_same_contract_baseline"
        historical_path = root / "historical-only.json"
        historical_path.write_text(json.dumps(history), encoding="utf-8")
        load_validated_capture(historical_path)
        row = decision(historical_path, {})
        assert "release_review_not_found" in row["blockers"]
        assert row["disposition"] == "keep experimental"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        config = _generic_config(tmp)
        args = SimpleNamespace(
            out_dir=tmp,
            skip_build=False,
            warmups=1,
            repeats=2,
            seed=42,
            max_new_captures=0,
            capture_timeout_seconds=60,
            refresh_scenario=["tiny-scenario"],
        )
        commands = pending_validation.command_plan(config, args)
        assert [item.name for item in commands] == [
            "build_linux-cdna-release",
            "sweep_tiny-scenario",
        ]
        assert commands[0].command == ["cmake", "--build", "--preset", "linux-cdna-release"]
        sweep_text = " ".join(commands[1].command)
        assert "--scenario tiny-scenario" in sweep_text
        assert "--skip-existing" not in sweep_text
        assert "--max-new-captures 0" in sweep_text
        outputs = pending_validation.write_command_plan(config, commands, tmp)
        payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
        assert payload["schema"] == "fixture_plan_v1"
        assert payload["target"] == "cdna-smoke"

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        args = SimpleNamespace(
            out_dir=tmp,
            skip_build=True,
            warmups=3,
            repeats=9,
            seed=20260606,
            max_new_captures=2,
            capture_timeout_seconds=300,
            refresh_scenario=["resident-lifetime-arena"],
        )
        commands = gfx1100_pending_validation.command_plan(args)
        assert len(commands) == 4
        names = [item.name for item in commands]
        assert names == [
            "sweep_exact-wide-export",
            "sweep_export-bound-limb-variants",
            "sweep_resident-lifetime-arena",
            "sweep_k-block-tile-variants",
        ]
        for item in commands:
            text = " ".join(item.command)
            assert "--review-mode release" in text
            assert "--warmups 3" in text
            assert "--repeats 9" in text
            assert "--bench-for hip-direct=" in text
            assert "--bench-for hipblaslt=" in text
            assert "--bench-for ck=" in text
            assert "--bench-for rocwmma=" in text
            assert "--max-new-captures 2" in text
            assert "--capture-timeout-seconds 300" in text
            if item.name == "sweep_resident-lifetime-arena":
                assert "--skip-existing" not in text
            else:
                assert "--skip-existing" in text
        outputs = gfx1100_pending_validation.write_command_plan(commands, tmp)
        payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
        assert payload["target"] == "gfx1100"
        assert len(payload["commands"]) == 4

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        valid_capture = (
            tmp
            / "captures"
            / "k-block-tile-variants"
            / "scenarios"
            / "k-block-tile-variants"
            / "valid.json"
        )
        failed_capture = (
            tmp
            / "captures"
            / "k-block-tile-variants"
            / "scenarios"
            / "k-block-tile-variants"
            / "timed-out.failed.json"
        )
        _write_capture(valid_capture)
        failed_capture.write_text(
            json.dumps({"timed_out": True, "returncode": None}, indent=2) + "\n",
            encoding="utf-8",
        )
        assert gfx1100_pending_validation.discover_captures(tmp) == [valid_capture]

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        config = _generic_config(tmp)
        capture_path = (
            tmp / "captures" / "tiny-scenario" / "scenarios" / "tiny-scenario" / "candidate.json"
        )
        review_path = tmp / "captures" / "tiny-scenario" / "review_report.json"
        _write_capture(capture_path)
        _write_review_report(review_path, capture_path, promotable=True)
        outputs = pending_validation.write_summary(config, tmp, [], [capture_path], [])
        payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
        row = payload["decision_rows"][0]
        assert payload["schema"] == "fixture_summary_v1"
        assert payload["target"] == "cdna-smoke"
        assert row["review_report_promotable"] is False
        assert row["review_report_declared_promotable"] is True
        assert row["disposition"] == "keep experimental"
        assert "not_cdna-smoke" in row["blockers"]

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        capture_path = (
            tmp
            / "captures"
            / "exact-wide-export"
            / "scenarios"
            / "exact-wide-export"
            / "candidate.json"
        )
        review_path = tmp / "captures" / "exact-wide-export" / "review_report.json"
        _write_capture(capture_path)
        _write_review_report(review_path, capture_path, promotable=True)
        outputs = gfx1100_pending_validation.write_summary(tmp, [], [capture_path], [])
        payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
        row = payload["decision_rows"][0]
        assert payload["review_report_count"] == 1
        assert row["review_report_promotable"] is False
        assert row["review_report_declared_promotable"] is True
        assert row["review_cache_write_status"] == "eligible_after_review"
        assert row["review_promotion_reason"] == "beats_required_same_contract_gpu_baselines"
        assert row["review_speedup_vs_direct_hip"] is None
        assert row["review_report_declared_speedup_vs_direct_hip"] == 1.2
        assert row["disposition"] == "keep experimental"
        assert "unsupported_review_schema" in row["blockers"]

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        capture_path = (
            tmp
            / "captures"
            / "resident-lifetime-arena"
            / "scenarios"
            / "resident-lifetime-arena"
            / "candidate.json"
        )
        review_path = tmp / "captures" / "resident-lifetime-arena" / "review_report.json"
        _write_capture(capture_path)
        _write_review_report(review_path, capture_path, promotable=False)
        outputs = gfx1100_pending_validation.write_summary(tmp, [], [capture_path], [])
        payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
        row = payload["decision_rows"][0]
        assert row["review_report_promotable"] is False
        assert row["disposition"] == "keep experimental"
        assert "missing_required_baseline:cpu-reference" in row["blockers"]
        assert "review_blocker:missing_required_baselines" in row["blockers"]

    assert pending_validation.validation_run_passed(
        [{"name": "sweep", "returncode": 0, "passed": True}],
        [
            {
                "name": "shape_family_shadow_report",
                "returncode": None,
                "passed": True,
                "skipped_reason": "not provided",
            }
        ],
    )
    assert not pending_validation.validation_run_passed(
        [{"name": "sweep", "returncode": 0, "passed": True}],
        [{"name": "benchmark_schema", "returncode": 1, "passed": False}],
    )
    assert not pending_validation.validation_run_passed(
        [{"name": "sweep", "returncode": 0, "passed": True}],
        [{"name": "unknown", "returncode": None, "passed": True}],
    )

    receipt_round_trip_controls()
    print("gfx1100 pending validation planner self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
