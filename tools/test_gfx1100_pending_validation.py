#!/usr/bin/env python3
"""Self-test gfx1100 pending validation command planning."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

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
                                "cache_write_status": "eligible_after_review" if promotable else "not_eligible",
                                "median_end_to_end_us": 10.0,
                                "promotable": promotable,
                                "promotion_blockers": blockers,
                                "promotion_reason": "beats_required_same_contract_gpu_baselines" if promotable else "blocked",
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
        assert [item.name for item in commands] == ["build_linux-cdna-release", "sweep_tiny-scenario"]
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
        valid_capture = tmp / "captures" / "k-block-tile-variants" / "scenarios" / "k-block-tile-variants" / "valid.json"
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
        capture_path = tmp / "captures" / "tiny-scenario" / "scenarios" / "tiny-scenario" / "candidate.json"
        review_path = tmp / "captures" / "tiny-scenario" / "review_report.json"
        _write_capture(capture_path)
        _write_review_report(review_path, capture_path, promotable=True)
        outputs = pending_validation.write_summary(config, tmp, [], [capture_path], [])
        payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
        row = payload["decision_rows"][0]
        assert payload["schema"] == "fixture_summary_v1"
        assert payload["target"] == "cdna-smoke"
        assert row["review_report_promotable"] is True
        assert row["disposition"] == "keep experimental"
        assert "not_cdna-smoke" in row["blockers"]

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        capture_path = tmp / "captures" / "exact-wide-export" / "scenarios" / "exact-wide-export" / "candidate.json"
        review_path = tmp / "captures" / "exact-wide-export" / "review_report.json"
        _write_capture(capture_path)
        _write_review_report(review_path, capture_path, promotable=True)
        outputs = gfx1100_pending_validation.write_summary(tmp, [], [capture_path], [])
        payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
        row = payload["decision_rows"][0]
        assert payload["review_report_count"] == 1
        assert row["review_report_promotable"] is True
        assert row["review_cache_write_status"] == "eligible_after_review"
        assert row["review_promotion_reason"] == "beats_required_same_contract_gpu_baselines"
        assert row["review_speedup_vs_direct_hip"] == 1.2
        assert row["disposition"] == "promote locally"
        assert row["blockers"] == []

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        capture_path = tmp / "captures" / "resident-lifetime-arena" / "scenarios" / "resident-lifetime-arena" / "candidate.json"
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

    print("gfx1100 pending validation planner self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
